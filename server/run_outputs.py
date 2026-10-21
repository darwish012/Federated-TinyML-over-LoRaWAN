"""Result sinks for the v3 experiment.

Everything the analysis needs lands in one output directory:

    events.csv            every controller event (same format as v2)
    delivery_log.jsonl    one JSON record per level start, per round, per level end
    replay_masks.json     uplink/downlink delivery masks per level, [rounds][positions]
    global_models.jsonl   every GLOBAL that was multicast (int8 codes + scales)
    alarms.csv            ALARM records from the boards
    self_checks.csv       SELF_CHECK records from the boards
    replay_done.csv       REPLAY_DONE records (one per position)
    replay_summary.json   per position: alarm / no_alarm / never_finished / incomplete_levels
    level_summary.json    one summary per level, rewritten as results arrive
    boards.json           board registry (board number, DevEUI, firmware, flags)
    summary.json          run-level summary (written by the runner at exit)

The controller only ever talks to the small RunOutputs interface, so tests use
MemoryOutputs and never touch the disk.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import threading

ALARM_FIELDS = (
    "timestamp_utc", "device_id", "board", "position", "level_index", "level",
    "acquisition_index", "score", "threshold",
)
SELF_CHECK_FIELDS = (
    "timestamp_utc", "device_id", "board", "position", "rows_checked",
    "max_abs_diff", "median_abs_diff",
)
REPLAY_DONE_FIELDS = (
    "timestamp_utc", "device_id", "board", "position", "level_mask", "acquisitions_scored",
    "alarms_latched",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryOutputs:
    """In-memory sink used by the tests (and as the interface reference)."""

    def __init__(self) -> None:
        self.delivery_records: list[dict] = []
        self.global_models: list[dict] = []
        self.alarms: list[dict] = []
        self.self_checks: list[dict] = []
        self.replay_dones: list[dict] = []
        self.replay_summary: dict | None = None
        self.level_summaries: list[dict] = []
        self.replay_masks: dict | None = None
        self.boards: dict | None = None

    def delivery(self, record: dict) -> None:
        self.delivery_records.append(record)

    def global_model(self, record: dict) -> None:
        self.global_models.append(record)

    def alarm(self, record: dict) -> None:
        self.alarms.append(record)

    def self_check(self, record: dict) -> None:
        self.self_checks.append(record)

    def replay_done(self, record: dict) -> None:
        self.replay_dones.append(record)

    def write_replay_summary(self, summary: dict) -> None:
        self.replay_summary = json.loads(json.dumps(summary))

    def write_level_summaries(self, summaries: list[dict]) -> None:
        self.level_summaries = json.loads(json.dumps(summaries))

    def write_replay_masks(self, masks: dict) -> None:
        self.replay_masks = json.loads(json.dumps(masks))

    def write_boards(self, boards: dict) -> None:
        self.boards = json.loads(json.dumps(boards))

    def records(self, kind: str) -> list[dict]:
        return [r for r in self.delivery_records if r.get("record") == kind]


class RunOutputs(MemoryOutputs):
    """Writes the sinks above under ``directory``. Also keeps the in-memory lists."""

    def __init__(self, directory: Path) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._delivery = (self.directory / "delivery_log.jsonl").open("a", encoding="utf-8")
        self._models = (self.directory / "global_models.jsonl").open("a", encoding="utf-8")
        self._alarm_file = (self.directory / "alarms.csv").open("w", newline="", encoding="utf-8")
        self._alarm_writer = csv.DictWriter(self._alarm_file, fieldnames=ALARM_FIELDS)
        self._alarm_writer.writeheader()
        self._check_file = (self.directory / "self_checks.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._check_writer = csv.DictWriter(self._check_file, fieldnames=SELF_CHECK_FIELDS)
        self._check_writer.writeheader()
        self._replay_file = (self.directory / "replay_done.csv").open(
            "w", newline="", encoding="utf-8")
        self._replay_writer = csv.DictWriter(self._replay_file, fieldnames=REPLAY_DONE_FIELDS)
        self._replay_writer.writeheader()

    @staticmethod
    def _line(handle, record: dict) -> None:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()

    def _write_json(self, name: str, payload) -> None:
        target = self.directory / name
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp.replace(target)

    def delivery(self, record: dict) -> None:
        super().delivery(record)
        with self._lock:
            self._line(self._delivery, record)

    def global_model(self, record: dict) -> None:
        super().global_model(record)
        with self._lock:
            self._line(self._models, record)

    def alarm(self, record: dict) -> None:
        super().alarm(record)
        with self._lock:
            self._alarm_writer.writerow({k: record.get(k, "") for k in ALARM_FIELDS})
            self._alarm_file.flush()

    def self_check(self, record: dict) -> None:
        super().self_check(record)
        with self._lock:
            self._check_writer.writerow({k: record.get(k, "") for k in SELF_CHECK_FIELDS})
            self._check_file.flush()

    def replay_done(self, record: dict) -> None:
        super().replay_done(record)
        with self._lock:
            self._replay_writer.writerow({k: record.get(k, "") for k in REPLAY_DONE_FIELDS})
            self._replay_file.flush()

    def write_replay_summary(self, summary: dict) -> None:
        super().write_replay_summary(summary)
        with self._lock:
            self._write_json("replay_summary.json", summary)

    def write_level_summaries(self, summaries: list[dict]) -> None:
        super().write_level_summaries(summaries)
        with self._lock:
            self._write_json("level_summary.json", summaries)

    def write_replay_masks(self, masks: dict) -> None:
        super().write_replay_masks(masks)
        with self._lock:
            self._write_json("replay_masks.json", masks)

    def write_boards(self, boards: dict) -> None:
        super().write_boards(boards)
        with self._lock:
            self._write_json("boards.json", boards)

    def close(self) -> None:
        with self._lock:
            for handle in (self._delivery, self._models, self._alarm_file, self._check_file,
                           self._replay_file):
                handle.close()
