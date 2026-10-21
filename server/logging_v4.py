"""Structured JSON logging and the concise console status line of the v4 server.

``JsonEventLog`` is the controller's ``log(event, device_id, details)`` callable. Every
event becomes one flushed JSON line in ``events.jsonl``:

    {"timestamp_utc": "...", "level": "INFO", "event": "round_complete",
     "device_id": "server", "details": {...}}

The console stays quiet: only warnings, errors and a few milestones are echoed, plus the
periodic one-line status (``format_status``) printed by the runner. A last line of
defence redacts any field whose name looks like key material, so a future bug cannot
write a key into a log.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading

ERROR_EVENTS = frozenset({
    "experiment_failed", "controller_exception", "mqtt_connect_failed", "event_rejected",
})
WARN_EVENTS = frozenset({
    "frame_rejected", "hello_rejected", "update_rejected", "update_abandoned", "downlink_failed",
    "downlink_dropped", "boards_not_ready", "phase_paused", "phase_idle_timeout", "phase_aborted",
    "board_holds_other_run", "board_not_counted", "board_rebooted_mid_run", "alarm_rejected",
    "update_fragment_conflict", "update_position_rejected", "enrollment_bad_mic",
    "enrollment_not_rostered", "enrollment_replayed_dev_nonce", "enrollment_registration_failed",
    "enrollment_malformed_join", "enrollment_stream_refused", "mqtt_disconnected",
    "planned_positions_without_board", "multicast_repair", "global_late_repair_exhausted",
})
MILESTONE_EVENTS = frozenset({
    "experiment_started", "phase_started", "phase_complete", "round_complete", "sweep_complete",
    "run_done", "resume_loaded", "run_control_replay_queued", "run_control_end_queued",
    "enrollment_window_opened", "enrollment_window_closed", "enrollment_registered",
    "mqtt_connected", "membership", "run_summary",
})
SECRET_HINTS = ("appkey", "app_key", "nwkskey", "appskey", "master", "api_key", "password",
                "secret", "token", "authorization")


def _scrub(value, key: str = ""):
    lowered = key.lower()
    if any(hint in lowered for hint in SECRET_HINTS) and not lowered.endswith("_env"):
        return "<redacted>"
    if isinstance(value, dict):
        return {k: _scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v, key) for v in value]
    return value


def level_of(event: str) -> str:
    if event in ERROR_EVENTS:
        return "ERROR"
    if event in WARN_EVENTS:
        return "WARN"
    return "INFO"


class JsonEventLog:
    def __init__(self, path: Path | str | None, console: bool = True, verbose: bool = False,
                 stream=None) -> None:
        self._lock = threading.Lock()
        self._file = None
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")
        self.console = console
        self.verbose = verbose
        self.stream = stream if stream is not None else sys.stdout

    def __call__(self, event: str, device_id: str, details: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        level = level_of(event)
        record = {"timestamp_utc": now, "level": level, "event": event, "device_id": device_id,
                  "details": _scrub(details)}
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock:
            if self._file is not None:
                self._file.write(line + "\n")
                self._file.flush()
            if self.console and (self.verbose or level != "INFO" or event in MILESTONE_EVENTS):
                print(self._console_line(now, level, event, device_id, record["details"]),
                      file=self.stream, flush=True)

    @staticmethod
    def _console_line(now: str, level: str, event: str, device_id: str, details: dict) -> str:
        brief = ", ".join(f"{k}={v}" for k, v in list(details.items())[:6]
                          if not isinstance(v, (dict, list)) or len(str(v)) < 60)
        if len(brief) > 200:
            brief = brief[:197] + "..."
        return f"{now[11:19]} {level:5s} {event} {device_id if device_id != 'server' else ''} {brief}".rstrip()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


def format_status(status: dict) -> str:
    """The one-line status the runner prints every ``status_line_seconds``."""
    where = "-" if status["phase"] is None else (
        f"phase {status['phase'] + 1}/{status['phases']} round {status['round']}/{status['rounds']}")
    return (f"[{status['stage']}] {where} | boards active={status['active']} "
            f"suspect={status['suspect']} departed={status['departed']} bound={status['bound']} "
            f"rostered={status['rostered']} | arrived {status['arrived_positions']}/"
            f"{status['expected_positions']}")
