"""MQTT bridge for the protocol-v4 federated-learning server (PROTOCOL_V4.md).

    python tts_v4_server.py --config config.toml --output results\\<stamp> [--enroll-minutes 10]
    python tts_v4_server.py --config config.toml --resume results\\<stamp>

Start it BEFORE powering the boards: application uplinks are not retained for an offline
subscriber. Boards are provisioned first (provision_board.py) so they appear in
roster.json; with --enroll-minutes (or the ``enroll`` console command) the server then
registers each rostered board in The Things Stack when it tries to join.

While the server runs, the terminal accepts:

    status                       the full state: stage, phase/round, members, counters
    roster                       rostered DevEUIs and their membership state
    enroll [minutes]             open (or re-open) the enrollment window
    enroll-status | enroll-close
    checkpoint                   write a checkpoint now
    stop                         graceful stop: checkpoint, then exit (resume later)
    help

Ctrl-C does the same as ``stop``: the run state is checkpointed and the exit code is 130,
meaning "resumable with --resume <run_dir>".
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
from typing import Any

from config_v4 import ConfigError, V4Config, check_environment
from controller_v4 import ControllerV4
from logging_v4 import JsonEventLog, format_status
from protocol_v4 import F_PORT, device_id_of_dev_eui
from roster_v4 import Roster
from run_outputs_v4 import RunOutputs, load_checkpoint
from study_plan import load_fl_modules, load_machine_order
from tts_v3_server import decode_up_event   # TTS uplink event -> {f_port, frame, f_cnt, radio}

EXIT_DONE, EXIT_FAILED, EXIT_STOPPED, EXIT_CONFIG = 0, 1, 130, 2


def _correlation_ids(event: dict, key: str) -> tuple:
    """Every correlation id a down/sent or down/failed event carries (top level, the downlink itself,
    or, for down/failed, the downlink nested in the failure). The controller matches on these and never
    on the device alone; an event naming none matches nothing."""
    found = list(event.get("correlation_ids") or ())
    body = event.get(key)
    if isinstance(body, dict):
        found += list(body.get("correlation_ids") or ())
        inner = body.get("downlink")
        if isinstance(inner, dict):
            found += list(inner.get("correlation_ids") or ())
    return tuple(str(c) for c in found)


class TtsMqttV4Server:
    def __init__(self, cfg: V4Config, output_dir: Path, resume: bool = False,
                 verbose: bool = False, env: dict | None = None, mqtt_module=None) -> None:
        if mqtt_module is None:                      # tests inject a fake: no broker, no paho needed
            try:
                import paho.mqtt.client as mqtt_module
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "paho-mqtt is not installed; run python -m pip install -r requirements.txt") from exc
        self.mqtt = mqtt_module
        self.cfg = cfg
        env = os.environ if env is None else env
        self.tts = cfg.raw["tts"]
        self.app_id = self.tts["application_id"]
        self.topic_root = f"v3/{self.app_id}"
        self.output_dir = Path(output_dir)
        self.log = JsonEventLog(self.output_dir / "events.jsonl", verbose=verbose)
        self.connected = threading.Event()
        self.enrollment = None
        self.listener = None
        self.stop_requested = threading.Event()
        self.multicast_device_id = cfg.multicast_device_id
        self.multicast_gateway_ids = list(cfg.multicast_gateway_ids)

        key_env = self.tts.get("api_key_env", "TTS_MQTT_API_KEY")
        api_key = env.get(key_env, "")
        if not api_key:
            raise RuntimeError(f"set ${key_env} to the TTS MQTT API key")
        self.client = mqtt_module.Client(
            callback_api_version=mqtt_module.CallbackAPIVersion.VERSION2,
            client_id=self.tts.get("client_id", "tinyfl-v4-server"), protocol=mqtt_module.MQTTv311)
        self.client.username_pw_set(self.tts.get("username", self.app_id), api_key)
        if self.tts.get("tls", False):
            self.client.tls_set(ca_certs=self.tts.get("ca_certs") or None)
            if self.tts.get("tls_insecure", False):
                self.client.tls_insecure_set(True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        # Roster: the authorisation list (provision_board.py appends to it).
        if cfg.require_roster and not cfg.roster_path.exists():
            raise RuntimeError(
                f"{cfg.roster_path} does not exist: provision at least one board "
                "(provision_board.py) or set v4.require_roster = false for a test run")
        self.roster = Roster(cfg.roster_path)
        self.outputs = RunOutputs(self.output_dir)
        order = []
        if cfg.mode == "planned":
            order, source = load_machine_order(cfg.research_root)
            self.log("machine_order", "server", {"source": source, "positions": len(order)})
        fl_modules = load_fl_modules(cfg.research_root)
        is_rostered = self.roster.contains if cfg.require_roster else None
        common = dict(publish=self._publish_downlink, publish_replace=self._publish_replace, log=self.log,
                      multicast_device_id=self.multicast_device_id, params=cfg.params,
                      outputs=self.outputs, is_rostered=is_rostered, fl_modules=fl_modules)
        if resume:
            checkpoint = load_checkpoint(self.output_dir)
            if checkpoint is None:
                raise RuntimeError(f"{self.output_dir} has no checkpoint.json to resume from")
            self.controller = ControllerV4.resume(checkpoint, **common)
            planned = cfg.build_plan(order).to_dict()
            if planned["phases"] != checkpoint["plan"]["phases"] or planned["seed"] != checkpoint["plan"]["seed"]:
                self.log("resume_plan_differs", "server", {
                    "note": "the checkpoint's plan is used; config.toml's plan changed since the run began"})
        else:
            self.controller = ControllerV4(cfg.build_plan(order), **common)
        self._sync_roster()

    # -- roster ------------------------------------------------------------------
    def _sync_roster(self) -> None:
        self.controller.sync_roster(self.roster.dev_euis(), device_id_of_dev_eui)

    # -- enrollment ----------------------------------------------------------------
    def enable_enrollment(self) -> None:
        if self.enrollment is None:
            from enrollment_v4 import build_enrollment
            self.enrollment, self.listener = build_enrollment(
                self.cfg.raw, self.log, Path(__file__).resolve().parent, roster=self.roster)

    def open_enrollment(self, minutes: float | None, actor: str) -> dict:
        self.enable_enrollment()
        seconds = None if minutes is None else int(minutes * 60)
        status = self.enrollment.open_window(actor, seconds=seconds)
        self.listener.start()
        return status

    # -- console ----------------------------------------------------------------------
    def full_status(self) -> dict:
        status = self.controller.status()
        status["roster_devices"] = len(self.roster)
        status["counters"] = dict(self.controller.counters)
        status["enrollment"] = self.enrollment.status() if self.enrollment else None
        status["mqtt_connected"] = self.connected.is_set()
        return status

    def _command_loop(self) -> None:
        for line in sys.stdin:
            parts = line.split()
            if not parts:
                continue
            try:
                self._command(parts)
            except Exception as exc:  # noqa: BLE001
                print(f"command failed: {exc}", flush=True)

    def _command(self, parts: list[str]) -> None:
        cmd = parts[0]
        if cmd == "status":
            print(json.dumps(self.full_status(), indent=2, sort_keys=True), flush=True)
        elif cmd == "roster":
            rows = {m.dev_eui: {"state": m.state.value, "board_id": m.board_id,
                                "positions": list(m.positions)}
                    for m in self.controller.membership.members.values()}
            print(json.dumps(rows, indent=2, sort_keys=True), flush=True)
        elif cmd == "enroll":
            minutes = float(parts[1]) if len(parts) > 1 else None
            print(json.dumps(self.open_enrollment(minutes, "console")), flush=True)
        elif cmd == "enroll-status" and self.enrollment:
            print(json.dumps(self.enrollment.status()), flush=True)
        elif cmd == "enroll-close" and self.enrollment:
            print(json.dumps(self.enrollment.close_window("console")), flush=True)
        elif cmd == "checkpoint":
            self.controller.checkpoint_now()
            print("checkpoint written", flush=True)
        elif cmd in ("stop", "quit", "exit"):
            self.stop_requested.set()
        else:
            print("commands: status | roster | enroll [minutes] | enroll-status | enroll-close | "
                  "checkpoint | stop", flush=True)

    # -- MQTT ----------------------------------------------------------------------------
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
                    return                                   # MAC-only uplink
                self.controller.handle_uplink(device_id, decoded["f_port"], decoded["frame"])
            elif suffix == "down/sent":
                self.controller.handle_downlink_sent(
                    device_id, _correlation_ids(event, "downlink_sent"))
            elif suffix == "down/failed":
                reason = json.dumps(event.get("downlink_failed", event), sort_keys=True)
                self.controller.handle_downlink_failed(
                    device_id, reason, _correlation_ids(event, "downlink_failed"))
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self.log("event_rejected", "server", {"topic": message.topic, "error": str(exc)})
        except Exception:  # noqa: BLE001 - a bug must be visible, not swallowed by paho
            self.log("controller_exception", "server",
                     {"topic": message.topic, "traceback": traceback.format_exc()})

    def _publish_replace(self, device_id: str, frame: bytes | None, correlation: str) -> None:
        """Unicast downlinks (rev 4.3, section 4): ``.../down/replace`` swaps the device's whole
        downlink queue for the list given, so at most one UPDATE_REPAIR is pending per device;
        ``frame=None`` publishes an empty list, which clears the queue."""
        self._publish_downlink(device_id, frame, correlation, replace=True)

    def _publish_downlink(self, device_id: str, frame: bytes | None, correlation: str,
                          replace: bool = False) -> None:
        topic = f"{self.topic_root}/devices/{device_id}/down/{'replace' if replace else 'push'}"
        if frame is None:
            result = self.client.publish(topic, json.dumps({"downlinks": []}, separators=(",", ":")),
                                         qos=0, retain=False)
            if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
                raise RuntimeError(f"MQTT publish failed immediately with code {result.rc}")
            return
        downlink = {
            "f_port": F_PORT,
            "frm_payload": base64.b64encode(frame).decode("ascii"),
            "priority": "NORMAL", "confirmed": False, "correlation_ids": [correlation],
        }
        # A multicast group never uplinks, so the network server cannot derive a downlink
        # path; the gateway has to be named for group downlinks.
        if device_id == self.multicast_device_id and self.multicast_gateway_ids:
            downlink["class_b_c"] = {"gateways": [
                {"gateway_ids": {"gateway_id": g}, "group_index": 0}
                for g in self.multicast_gateway_ids]}
        result = self.client.publish(
            topic, json.dumps({"downlinks": [downlink]}, separators=(",", ":")),
            qos=0, retain=False)
        if result.rc != self.mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT publish failed immediately with code {result.rc}")

    # -- run -------------------------------------------------------------------------------
    def request_stop(self, *_args) -> None:
        self.stop_requested.set()

    def run(self, enroll_minutes: float | None, interactive: bool = True) -> int:
        tts = self.tts
        host, port = tts.get("host", "127.0.0.1"), int(tts.get("port", 1883))
        keepalive = int(tts.get("keepalive_seconds", 60))
        timeout_seconds = self.cfg.run_timeout_minutes * 60
        started = time.monotonic()
        last_status = last_roster = started

        previous_handlers = {}
        for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    previous_handlers[sig] = signal.signal(sig, self.request_stop)
                except ValueError:
                    pass                                    # not the main thread (tests)
        self.client.connect(host, port, keepalive)
        self.client.loop_start()
        exit_code = EXIT_DONE
        try:
            if not self.connected.wait(timeout=15):
                raise RuntimeError("MQTT connection did not become ready within 15 seconds")
            if enroll_minutes is not None:
                self.log("enrollment_status", "server", self.open_enrollment(enroll_minutes, "cli"))
            if interactive and sys.stdin is not None and sys.stdin.isatty():
                threading.Thread(target=self._command_loop, name="console", daemon=True).start()
            print("type 'status' for details, 'stop' (or Ctrl-C) to checkpoint and exit", flush=True)
            consecutive_errors = 0
            while not self.controller.finished and not self.controller.failed:
                if self.stop_requested.is_set():
                    exit_code = EXIT_STOPPED
                    break
                try:
                    self.controller.tick()
                    consecutive_errors = 0
                except Exception:  # noqa: BLE001 - e.g. MQTT briefly down: log, retry, give up if it persists
                    consecutive_errors += 1
                    self.log("controller_exception", "server", {
                        "traceback": traceback.format_exc(), "consecutive": consecutive_errors})
                    if consecutive_errors >= 30:
                        self.controller.fail("tick failed 30 times in a row; see controller_exception events")
                        break
                    time.sleep(1.0)
                now = time.monotonic()
                if now - started > timeout_seconds:
                    self.controller.fail("overall run timeout")
                    break
                if self.cfg.status_line_seconds and now - last_status >= self.cfg.status_line_seconds:
                    last_status = now
                    print(format_status(self.controller.status()), flush=True)
                if now - last_roster >= 5:
                    last_roster = now
                    try:
                        if self.roster.reload():
                            self._sync_roster()
                            self.log("roster_reloaded", "server", {"devices": len(self.roster)})
                    except ValueError as exc:
                        self.log("roster_reload_failed", "server", {"error": str(exc)})
                time.sleep(0.2)
        finally:
            if self.listener is not None:
                self.listener.stop()
            if exit_code == EXIT_STOPPED:
                self.controller.checkpoint_now()
                self.log("run_stopped_by_operator", "server", {
                    "resume_with": f"--resume {self.output_dir}"})
            self.controller.finalize()
            summary = self.controller.summary()
            summary["exit"] = ("stopped_resumable" if exit_code == EXIT_STOPPED else
                               "failed" if self.controller.failed else "done")
            (self.output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.log("run_summary", "server", {k: summary[k] for k in (
                "stage", "finished", "failed", "failure_reason", "exit")})
            self.client.disconnect()
            self.client.loop_stop()
            self.outputs.close()
            self.log.close()
            for sig, handler in previous_handlers.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, TypeError):
                    pass
        if exit_code == EXIT_STOPPED:
            return EXIT_STOPPED
        return EXIT_FAILED if self.controller.failed else EXIT_DONE


def apply_overrides(config: dict[str, Any], args) -> dict[str, Any]:
    if not (args.mode or args.min_boards or args.phases):
        return config                       # leave a missing [v4] missing: all defaults, and say so
    v4 = config.setdefault("v4", {})
    if args.mode:
        v4["mode"] = args.mode
    if args.min_boards:
        v4["min_boards"] = args.min_boards
    if args.phases:
        v4["phases"] = args.phases
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--output", type=Path, default=None,
                        help="run directory (default results\\v4-<timestamp>)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="continue the run in this directory from its checkpoint")
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--mode", choices=["planned", "elastic"], default=None,
                        help="override [v4] mode")
    parser.add_argument("--phases", type=int, nargs="+", default=None,
                        help="override [v4] phases (planned): e.g. --phases 4 8")
    parser.add_argument("--min-boards", type=int, default=None, help="override [v4] min_boards")
    parser.add_argument("--enroll-minutes", type=float, default=None,
                        help="open an enrollment window for this many minutes at start")
    parser.add_argument("--verbose", action="store_true", help="echo every event to the console")
    args = parser.parse_args(argv)

    try:
        import tomllib
        with args.config.open("rb") as handle:
            raw = tomllib.load(handle)
        from config_v4 import parse_config
        cfg = parse_config(apply_overrides(raw, args), args.config.resolve().parent)
    except FileNotFoundError:
        print(f"config error: {args.config} not found; copy config.toml.example to config.toml",
              file=sys.stderr)
        return EXIT_CONFIG
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except Exception as exc:  # noqa: BLE001 - TOML syntax errors
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    problems = check_environment(cfg, need_enrollment=args.enroll_minutes is not None)
    if args.check_config:
        order, source = ([], "n/a (elastic)")
        if cfg.mode == "planned":
            order, source = load_machine_order(cfg.research_root)
        load_fl_modules(cfg.research_root)                  # raises if the research folder is missing
        plan = cfg.build_plan(order)
        print(f"Research modules import; machine order from: {source}")
        print(f"mode {plan.mode}: {len(plan.phases)} phase(s), "
              + ", ".join(f"{p.name}x{p.rounds}" for p in plan.phases[:10])
              + (" ..." if len(plan.phases) > 10 else ""))
        if not cfg.has_v4_section:
            print("note: config.toml has no [v4] section; every default applies "
                  "(copy [v4] from config.toml.example to change them)")
        print(f"roster: {cfg.roster_path} ({'present' if cfg.roster_path.exists() else 'MISSING'})")
        for problem in problems:
            print(f"environment: {problem}")
        print("Configuration is valid" if not problems else "Configuration is valid, environment incomplete")
        return 0
    if problems:
        for problem in problems:
            print(f"environment error: {problem}", file=sys.stderr)
        return EXIT_CONFIG

    if args.resume is not None:
        output, resume = args.resume, True
    else:
        output = args.output or Path("results") / f"v4-{time.strftime('%Y%m%d-%H%M%S')}"
        resume = False
        if (output / "checkpoint.json").exists():
            print(f"error: {output} already holds a run; use --resume {output} to continue it "
                  "or choose another --output", file=sys.stderr)
            return EXIT_CONFIG
    try:
        server = TtsMqttV4Server(cfg, output, resume=resume, verbose=args.verbose)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"startup error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    return server.run(args.enroll_minutes)


if __name__ == "__main__":
    raise SystemExit(main())
