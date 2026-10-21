"""MQTT bridge for the protocol-v3 federated-learning hardware experiment.

    python tts_v3_server.py --config config.toml --output results\\<stamp> \\
        --enroll-minutes 10

Start it BEFORE powering the boards: application uplinks are not retained for an
offline subscriber. With --enroll-minutes it opens an enrollment window, so any
board holding the shared AppKey is registered in The Things Stack automatically
the first time it tries to join (see enrollment.py). The sweep begins by itself
once the configured number of boards have sent BOARD_HELLO.

While the server runs, an interactive terminal accepts:

    enroll [minutes] [quota]     open (or re-open) the enrollment window
    enroll-status                window state, registered and refused DevEUIs
    enroll-close                 close the window now
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import threading
import time
import tomllib
import traceback
from typing import Any

from controller_v3 import HardwareExperimentController, Phase
from enrollment import build_enrollment
from protocol_v3 import HARDWARE_LEVELS, N_BOARDS
from run_outputs import RunOutputs
from study_plan import load_machine_order, validate_levels
from tts_class_c_server import CsvEventLog


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    tts, experiment = config["tts"], config["experiment"]
    v3 = config.get("v3", {})
    if tts.get("tenant_id", ""):
        raise ValueError("this local Open Source deployment must leave tenant_id empty")
    group = str(experiment.get("multicast_device_id", "")).strip()
    if not group:
        raise ValueError(
            "v3 delivers GLOBAL by multicast: set experiment.multicast_device_id "
            "(python configure_multicast.py creates the group's config)"
        )
    gateways = [str(g).strip() for g in experiment.get("multicast_gateway_ids", []) if str(g).strip()]
    if not gateways:
        raise ValueError(
            "experiment.multicast_gateway_ids is empty; a multicast group never uplinks, "
            "so every downlink would fail with no_downlink_path"
        )
    validate_levels(v3.get("levels", list(HARDWARE_LEVELS)))
    expected = int(v3.get("expected_boards", N_BOARDS))
    if not 1 <= expected <= N_BOARDS:
        raise ValueError(f"v3.expected_boards must be 1..{N_BOARDS}")
    return config


def decode_up_event(event: dict) -> dict | None:
    """Turn a TTS uplink event into {f_port, frame, f_cnt, radio}, or None for a MAC-only uplink.

    TTS omits ``frm_payload`` entirely when an uplink carries no application
    payload (LMIC sends such frames to deliver MAC answers), which is normal
    traffic rather than a protocol error.
    """
    uplink = event["uplink_message"]
    encoded = uplink.get("frm_payload")
    if encoded is None:
        return None
    metadata = uplink.get("rx_metadata", [])
    settings = uplink.get("settings", {})
    radio = settings.get("data_rate", {}).get("lora", {})
    f_cnt = uplink.get("f_cnt")
    return {
        "f_port": int(uplink.get("f_port", 0)),
        "frame": base64.b64decode(encoded, validate=True),
        "f_cnt": None if f_cnt is None else int(f_cnt),
        "radio": {
            "spreading_factor": radio.get("spreading_factor"),
            "bandwidth": radio.get("bandwidth"),
            "frequency": settings.get("frequency"),
            "rssi": metadata[0].get("rssi") if metadata else None,
            "snr": metadata[0].get("snr") if metadata else None,
        },
    }


class TtsMqttV3Experiment:
    def __init__(self, config: dict[str, Any], output_dir: Path,
                 levels_override=None, expected_boards_override=None) -> None:
        try:
            import paho.mqtt.client as mqtt_module
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "paho-mqtt is not installed; run python -m pip install -r requirements.txt"
            ) from exc
        self.mqtt = mqtt_module
        self.config = config
        self.tts = config["tts"]
        self.experiment = config["experiment"]
        self.v3 = config.get("v3", {})
        self.app_id = self.tts["application_id"]
        self.topic_root = f"v3/{self.app_id}"
        self.output_dir = output_dir
        self.log = CsvEventLog(output_dir / "events.csv")
        self.connected = threading.Event()
        self.enrollment = None
        self.listener = None

        key_env = self.tts.get("api_key_env", "TTS_MQTT_API_KEY")
        api_key = os.environ.get(key_env, "")
        if not api_key:
            raise RuntimeError(f"set ${key_env} to the TTS MQTT API key")

        self.client = mqtt_module.Client(
            callback_api_version=mqtt_module.CallbackAPIVersion.VERSION2,
            client_id=self.tts.get("client_id", "tinyfl-v3-hardware-server"),
            protocol=mqtt_module.MQTTv311,
        )
        self.client.username_pw_set(self.tts.get("username", self.app_id), api_key)
        if self.tts.get("tls", False):
            self.client.tls_set(ca_certs=self.tts.get("ca_certs") or None)
            if self.tts.get("tls_insecure", False):
                self.client.tls_insecure_set(True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self.multicast_device_id = str(self.experiment["multicast_device_id"]).strip()
        self.multicast_gateway_ids = [
            str(g).strip() for g in self.experiment.get("multicast_gateway_ids", []) if str(g).strip()
        ]

        levels = validate_levels(levels_override or self.v3.get("levels", list(HARDWARE_LEVELS)))
        research_root = self.v3.get("research_root") or None
        order, order_source = load_machine_order(research_root)
        self.log("machine_order", "server", {"source": order_source, "positions": len(order)})
        self.outputs = RunOutputs(output_dir)
        exp = self.experiment
        self.controller = HardwareExperimentController(
            levels=levels,
            seed=int(self.v3.get("seed", 17070)),
            expected_boards=int(expected_boards_override
                                or self.v3.get("expected_boards", N_BOARDS)),
            publish=self._publish_downlink,
            log=self.log,
            multicast_device_id=self.multicast_device_id,
            outputs=self.outputs,
            machine_order=order,
            research_root=research_root,
            class_c_guard_seconds=float(exp.get("class_c_guard_seconds", 15.0)),
            retry_guard_seconds=float(exp.get("retry_guard_seconds", 8.0)),
            max_downlink_attempts=int(exp.get("max_downlink_attempts", 5)),
            max_repair_rounds=int(exp.get("max_repair_rounds", 4)),
            max_global_repair_rounds=int(self.v3.get("max_global_repair_rounds", 4)),
            multicast_report_window_seconds=float(exp.get("multicast_report_window_seconds", 180.0)),
            ready_wait_seconds=float(self.v3.get("ready_wait_seconds", 240.0)),
            round_timeout_seconds=float(self.v3.get("round_timeout_seconds", 900.0)),
            post_run_collect_seconds=float(self.v3.get("post_run_collect_seconds", 1800.0)),
        )

    # -- enrollment ----------------------------------------------------------
    def enable_enrollment(self) -> None:
        if self.enrollment is None:
            self.enrollment, self.listener = build_enrollment(
                self.config, self.log, Path(__file__).resolve().parent
            )

    def open_enrollment(self, minutes: float | None, quota: int | None, actor: str) -> dict:
        self.enable_enrollment()
        seconds = None if minutes is None else int(minutes * 60)
        status = self.enrollment.open_window(actor, seconds=seconds, quota=quota)
        self.listener.start()
        return status

    def _command_loop(self) -> None:
        for line in sys.stdin:
            parts = line.split()
            if not parts:
                continue
            try:
                if parts[0] == "enroll":
                    minutes = float(parts[1]) if len(parts) > 1 else None
                    quota = int(parts[2]) if len(parts) > 2 else None
                    print(json.dumps(self.open_enrollment(minutes, quota, "console")), flush=True)
                elif parts[0] == "enroll-status" and self.enrollment:
                    print(json.dumps(self.enrollment.status()), flush=True)
                elif parts[0] == "enroll-close" and self.enrollment:
                    print(json.dumps(self.enrollment.close_window("console")), flush=True)
                else:
                    print("commands: enroll [minutes] [quota] | enroll-status | enroll-close",
                          flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"command failed: {exc}", flush=True)

    # -- MQTT ---------------------------------------------------------------------
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

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties) -> None:
        del client, userdata, disconnect_flags, properties
        self.connected.clear()
        self.log("mqtt_disconnected", "server", {"reason": str(reason_code)})

    def _on_message(self, client, userdata, message) -> None:
        del client, userdata
        try:
            prefix = f"{self.topic_root}/devices/"
            if not message.topic.startswith(prefix):
                raise ValueError("unexpected MQTT topic root")
            device_id, suffix = message.topic[len(prefix):].split("/", 1)
            event = json.loads(message.payload.decode("utf-8"))
            if suffix == "up":
                decoded = decode_up_event(event)
                if decoded is None:
                    self.log("mac_only_uplink", device_id, {})
                    return
                self.log("radio_uplink", device_id,
                         {"application_bytes": len(decoded["frame"]), "f_cnt": decoded["f_cnt"],
                          **decoded["radio"]})
                self.controller.handle_uplink(device_id, decoded["f_port"], decoded["frame"])
            elif suffix == "down/sent":
                self.controller.handle_downlink_sent(device_id)
            elif suffix == "down/failed":
                reason = json.dumps(event.get("downlink_failed", event), sort_keys=True)
                self.controller.handle_downlink_failed(device_id, reason)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self.log("event_rejected", "server", {"topic": message.topic, "error": str(exc)})
        except Exception:  # noqa: BLE001 - a bug must be visible, not swallowed by paho
            self.log("controller_exception", "server",
                     {"topic": message.topic, "traceback": traceback.format_exc()})

    def _publish_downlink(self, device_id: str, frame: bytes, correlation: str) -> None:
        topic = f"{self.topic_root}/devices/{device_id}/down/push"
        downlink = {
            "f_port": 10,
            "frm_payload": base64.b64encode(frame).decode("ascii"),
            "priority": "NORMAL",
            "confirmed": False,
            "correlation_ids": [correlation],
        }
        # A multicast group never uplinks, so the network server cannot derive a
        # downlink path; the gateway has to be named for group downlinks.
        if device_id == self.multicast_device_id and self.multicast_gateway_ids:
            downlink["class_b_c"] = {
                "gateways": [
                    {"gateway_ids": {"gateway_id": g}, "group_index": 0}
                    for g in self.multicast_gateway_ids
                ]
            }
        result = self.client.publish(
            topic, json.dumps({"downlinks": [downlink]}, separators=(",", ":")),
            qos=0, retain=False,
        )
        if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed immediately with code {result.rc}")

    def run(self, enroll_minutes: float | None, enroll_quota: int | None) -> int:
        host = self.tts.get("host", "127.0.0.1")
        port = int(self.tts.get("port", 1883))
        keepalive = int(self.tts.get("keepalive_seconds", 60))
        timeout_seconds = float(self.v3.get("run_timeout_minutes", 900)) * 60
        started = time.monotonic()

        self.client.connect(host, port, keepalive)
        self.client.loop_start()
        try:
            if not self.connected.wait(timeout=15):
                raise RuntimeError("MQTT connection did not become ready within 15 seconds")
            if enroll_minutes is not None:
                status = self.open_enrollment(enroll_minutes, enroll_quota, "cli")
                self.log("enrollment_status", "server", status)
            if sys.stdin is not None and sys.stdin.isatty():
                threading.Thread(target=self._command_loop, name="console", daemon=True).start()
            while not self.controller.finished and not self.controller.failed:
                self.controller.tick()
                if time.monotonic() - started > timeout_seconds:
                    self.controller.fail("overall experiment timeout")
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            # Ctrl-C while collecting alarms is the normal way to end the run.
            if self.controller.phase is not Phase.COLLECTING:
                self.controller.fail("stopped by operator")
        finally:
            if self.listener is not None:
                self.listener.stop()
            self.controller.finalize()
            summary = self.controller.summary()
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.log("run_summary", "server",
                     {k: summary[k] for k in ("phase", "finished", "failed", "failure_reason")})
            self.client.disconnect()
            self.client.loop_stop()
            self.outputs.close()
            self.log.close()
        return 0 if not self.controller.failed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--output", type=Path, default=Path("results") / "v3-latest")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--levels", type=int, nargs="+", default=None,
                        help="override [v3] levels, e.g. --levels 4 8")
    parser.add_argument("--expected-boards", type=int, default=None,
                        help="override [v3] expected_boards")
    parser.add_argument("--enroll-minutes", type=float, default=None,
                        help="open an enrollment window for this many minutes at start")
    parser.add_argument("--enroll-quota", type=int, default=None,
                        help="registrations allowed in the window (default enrollment.quota)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.levels:
        validate_levels(args.levels)
    if args.check_config:
        v3 = config.get("v3", {})
        from study_plan import load_fl_modules
        load_fl_modules(v3.get("research_root") or None)     # raises if the research folder is missing
        _, source = load_machine_order(v3.get("research_root") or None)
        print(f"Research modules import; machine order from: {source}")
        print("Configuration is valid")
        return 0
    return TtsMqttV3Experiment(
        config, args.output, args.levels, args.expected_boards
    ).run(args.enroll_minutes, args.enroll_quota)


if __name__ == "__main__":
    raise SystemExit(main())
