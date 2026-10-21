"""MQTT bridge from two Class-A LilyGO updates to Class-C unicast globals."""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
import tomllib
from typing import Any

from controller import ExperimentController
from protocol import ProtocolError, dev_eui_from_device_id


class CsvEventLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=("timestamp_utc", "event", "device_id", "details_json"),
        )
        self._writer.writeheader()
        self._lock = threading.Lock()

    def __call__(self, event: str, device_id: str, details: dict) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._writer.writerow(
                {
                    "timestamp_utc": timestamp,
                    "event": event,
                    "device_id": device_id,
                    "details_json": json.dumps(details, sort_keys=True),
                }
            )
            self._file.flush()
        print(f"{timestamp} {event:26s} {device_id:18s} {details}", flush=True)

    def close(self) -> None:
        self._file.close()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    tts = config["tts"]
    experiment = config["experiment"]
    device_ids = experiment["device_ids"]
    if not device_ids:
        raise ValueError("experiment.device_ids must name at least one device")
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("experiment.device_ids contains duplicates")
    # Every node runs one identical binary and seeds its payload from its own
    # DevEUI, so the server can only reproduce that payload if the device_id
    # carries the DevEUI. Registering a device by its DevEUI in TTS gives
    # exactly this "eui-<16 hex digits>" form.
    for device_id in device_ids:
        dev_eui_from_device_id(device_id)

    # A multicast group never sends an uplink, so the network server has no
    # gateway to derive a downlink path from and rejects every frame with
    # "no_downlink_path". Naming the gateway is therefore not optional.
    if str(experiment.get("multicast_device_id", "")).strip():
        gateways = [
            str(gateway_id).strip()
            for gateway_id in experiment.get("multicast_gateway_ids", [])
            if str(gateway_id).strip()
        ]
        if not gateways:
            raise ValueError(
                "multicast_device_id is set but multicast_gateway_ids is empty; "
                "a multicast group has no uplinks, so the network server cannot "
                "work out which gateway to transmit through and every downlink "
                "fails with no_downlink_path"
            )
    if tts.get("tenant_id", ""):
        raise ValueError("this local Open Source deployment must leave tenant_id empty")
    return config


class TtsMqttExperiment:
    def __init__(self, config: dict[str, Any], output_dir: Path) -> None:
        try:
            import paho.mqtt.client as mqtt_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "paho-mqtt is not installed; run python -m pip install -r requirements.txt"
            ) from exc
        global mqtt
        mqtt = mqtt_module
        self.config = config
        self.tts = config["tts"]
        self.experiment = config["experiment"]
        self.app_id = self.tts["application_id"]
        self.topic_root = f"v3/{self.app_id}"
        self.output_dir = output_dir
        self.log = CsvEventLog(output_dir / "events.csv")
        self.connected = threading.Event()

        api_key = os.environ.get(self.tts.get("api_key_env", "TTS_MQTT_API_KEY"), "")
        if not api_key:
            raise RuntimeError(
                f"set ${self.tts.get('api_key_env', 'TTS_MQTT_API_KEY')} to the TTS MQTT API key"
            )

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.tts.get("client_id", "tinyfl-v91-hardware-server"),
            protocol=mqtt.MQTTv311,
        )
        self.client.username_pw_set(self.tts.get("username", self.app_id), api_key)
        if self.tts.get("tls", False):
            self.client.tls_set(ca_certs=self.tts.get("ca_certs") or None)
            if self.tts.get("tls_insecure", False):
                self.client.tls_insecure_set(True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self.multicast_device_id = str(
            self.experiment.get("multicast_device_id", "")
        ).strip() or None
        self.multicast_gateway_ids = [
            str(gateway_id).strip()
            for gateway_id in self.experiment.get("multicast_gateway_ids", [])
            if str(gateway_id).strip()
        ]
        self.controller = ExperimentController(
            device_ids=list(self.experiment["device_ids"]),
            round_id=int(self.experiment.get("round_id", 1)),
            publish=self._publish_downlink,
            log=self.log,
            class_c_guard_seconds=float(
                self.experiment.get("class_c_guard_seconds", 8.0)
            ),
            retry_guard_seconds=float(
                self.experiment.get("retry_guard_seconds", 8.0)
            ),
            max_downlink_attempts=int(
                self.experiment.get("max_downlink_attempts", 3)
            ),
            max_repair_rounds=int(
                self.experiment.get("max_repair_rounds", 4)
            ),
            multicast_report_window_seconds=float(
                self.experiment.get("multicast_report_window_seconds", 180.0)
            ),
            multicast_device_id=self.multicast_device_id,
        )

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        del userdata, flags, properties
        if reason_code != 0:
            self.log("mqtt_connect_failed", "server", {"reason": str(reason_code)})
            return
        client.subscribe(f"{self.topic_root}/devices/+/up", qos=0)
        client.subscribe(f"{self.topic_root}/devices/+/down/sent", qos=0)
        client.subscribe(f"{self.topic_root}/devices/+/down/failed", qos=0)
        self.connected.set()
        self.log("mqtt_connected", "server", {"topic_root": self.topic_root})

    def _on_disconnect(
        self, client, userdata, disconnect_flags, reason_code, properties
    ) -> None:
        del client, userdata, disconnect_flags, properties
        self.connected.clear()
        self.log("mqtt_disconnected", "server", {"reason": str(reason_code)})

    def _topic_device_and_suffix(self, topic: str) -> tuple[str, str]:
        prefix = f"{self.topic_root}/devices/"
        if not topic.startswith(prefix):
            raise ValueError("unexpected MQTT topic root")
        remainder = topic[len(prefix) :]
        device_id, suffix = remainder.split("/", 1)
        return device_id, suffix

    def _on_message(self, client, userdata, message) -> None:
        del client, userdata
        try:
            device_id, suffix = self._topic_device_and_suffix(message.topic)
            event = json.loads(message.payload.decode("utf-8"))
            if suffix == "up":
                uplink = event["uplink_message"]
                encoded = uplink.get("frm_payload")
                if encoded is None:
                    # TTS omits frm_payload entirely when an uplink carries no
                    # application payload. LMIC sends such frames routinely to
                    # deliver MAC answers (the LinkADRAns that follows a join),
                    # so they are normal traffic rather than a protocol error
                    # and must not abort the run.
                    self.log(
                        "mac_only_uplink",
                        device_id,
                        {"f_port": int(uplink.get("f_port", 0))},
                    )
                    return
                frame = base64.b64decode(encoded, validate=True)
                metadata = uplink.get("rx_metadata", [])
                settings = uplink.get("settings", {})
                radio = settings.get("data_rate", {}).get("lora", {})
                self.log(
                    "radio_uplink",
                    device_id,
                    {
                        "application_bytes": len(frame),
                        "spreading_factor": radio.get("spreading_factor"),
                        "bandwidth": radio.get("bandwidth"),
                        "frequency": settings.get("frequency"),
                        "rssi": metadata[0].get("rssi") if metadata else None,
                        "snr": metadata[0].get("snr") if metadata else None,
                    },
                )
                self.controller.handle_uplink(
                    device_id, int(uplink.get("f_port", 0)), frame
                )
            elif suffix == "down/sent":
                self.controller.handle_downlink_sent(device_id)
            elif suffix == "down/failed":
                reason = json.dumps(event.get("downlink_failed", event), sort_keys=True)
                self.controller.handle_downlink_failed(device_id, reason)
        except (KeyError, ValueError, ProtocolError, json.JSONDecodeError) as exc:
            self.controller.failure_reason = f"MQTT event rejected: {exc}"
            self.log(
                "event_rejected",
                "server",
                {"topic": message.topic, "error": str(exc)},
            )

    def _publish_downlink(self, device_id: str, frame: bytes, correlation: str) -> None:
        topic = f"{self.topic_root}/devices/{device_id}/down/push"
        downlink = {
            "f_port": 10,
            "frm_payload": base64.b64encode(frame).decode("ascii"),
            "priority": "NORMAL",
            "confirmed": False,
            "correlation_ids": [correlation],
        }
        # For a normal device the network server picks the gateway from that
        # device's most recent uplink token. A multicast group NEVER uplinks, so
        # there is nothing to derive a path from and the downlink is rejected
        # with "no_downlink_path". The gateway therefore has to be named
        # explicitly for Class B/C group downlinks.
        if device_id == self.multicast_device_id and self.multicast_gateway_ids:
            downlink["class_b_c"] = {
                "gateways": [
                    {"gateway_ids": {"gateway_id": gateway_id}, "group_index": 0}
                    for gateway_id in self.multicast_gateway_ids
                ]
            }
        command = {"downlinks": [downlink]}
        result = self.client.publish(
            topic, json.dumps(command, separators=(",", ":")), qos=0, retain=False
        )
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed immediately with code {result.rc}")

    def run(self) -> int:
        host = self.tts.get("host", "127.0.0.1")
        port = int(self.tts.get("port", 1883))
        keepalive = int(self.tts.get("keepalive_seconds", 60))
        timeout_seconds = float(self.experiment.get("run_timeout_minutes", 45)) * 60
        started = time.monotonic()

        self.client.connect(host, port, keepalive)
        self.client.loop_start()
        try:
            if not self.connected.wait(timeout=15):
                raise RuntimeError("MQTT connection did not become ready within 15 seconds")
            while not self.controller.finished and not self.controller.failed:
                self.controller.tick()
                if time.monotonic() - started > timeout_seconds:
                    self.controller.failure_reason = "overall experiment timeout"
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.controller.failure_reason = "stopped by operator"
        finally:
            summary = self.controller.summary()
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            self.log("run_summary", "server", summary)
            self.client.disconnect()
            self.client.loop_stop()
            self.log.close()
        return 0 if self.controller.finished else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--output", type=Path, default=Path("results") / "latest")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check_config:
        print("Configuration is valid")
        return 0
    return TtsMqttExperiment(config, args.output).run()


if __name__ == "__main__":
    raise SystemExit(main())
