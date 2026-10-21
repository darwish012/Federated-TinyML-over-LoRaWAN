"""Result sinks for the v4 server. Everything lands in one run directory:

    events.jsonl             structured server log, one JSON object per line (logging_v4.py)
    delivery_log.jsonl       per phase start / round / phase end: participants, arrived positions,
                             base_slot per position, stale positions, joins/leaves, fragments,
                             repairs, timings
    global_models.jsonl      every GLOBAL multicast: int8 codes + scales + CRC (+ slot, phase, round)
    membership_events.jsonl  every membership state change
    replay_masks.json        "tinyfl-replay-masks-v4": per phase uplink / downlink / base_slot
                             [round][position] for analysis/compare_hw_twin_v4.py (PROTOCOL_V4 4.1)
    alarms.csv, self_checks.csv, replay_done.csv   (REPLAY_DONE is per (position, phase))
    replay_summary.json      per position: alarm / no_alarm / never_finished / incomplete_phases
    phase_summary.json       one summary per phase
    boards.json              board registry (DevEUI, board_id, positions, state)
    checkpoint.json          atomic run checkpoint after every round (+ checkpoint.prev.json)
    summary.json             run-level summary (written by the runner at exit)

The controller only talks to this small interface, so tests use MemoryOutputs and
never touch the disk. Files are opened for append, so ``--resume`` keeps writing the
same logs; the sinks remember which GLOBAL slots and round records are already on
disk so a replayed step after a crash does not duplicate them.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import threading

# PROTOCOL_V4 4.1: the first columns of alarms.csv and replay_done.csv are normative; the
# trailing ones (who / when) are extras the twin comparison ignores.
ALARM_FIELDS = (
    "run_id", "position", "phase_index", "acquisition_index", "score", "threshold", "source",
    "timestamp_utc", "device_id", "board_id",
)
SELF_CHECK_FIELDS = (
    "timestamp_utc", "device_id", "board_id", "position", "rows_checked", "max_abs_diff",
    "median_abs_diff",
)
REPLAY_DONE_FIELDS = (
    "run_id", "position", "phase_index", "acquisitions_scored", "alarm_latched",
    "first_alarm_index", "timestamp_utc", "device_id", "board_id",
)


class MemoryOutputs:
    """In-memory sink used by the tests (and as the interface reference)."""

    def __init__(self) -> None:
        self.delivery_records: list[dict] = []
        self.global_models: list[dict] = []
        self.membership_events: list[dict] = []
        self.alarms: list[dict] = []
        self.self_checks: list[dict] = []
        self.replay_dones: list[dict] = []
        self.replay_summary: dict | None = None
        self.phase_summaries: list[dict] = []
        self.replay_masks: dict | None = None
        self.boards: dict | None = None
        self.checkpoint: dict | None = None
        self._slots_logged: set = set()
        self._rounds_logged: set = set()
        self._alarms_logged: set = set()       # (position, phase_index)
        self._dones_logged: set = set()

    # -- append-only records -----------------------------------------------
    def delivery(self, record: dict) -> bool:
        """Append a delivery-log record. A repeated (phase, round) record is dropped."""
        key = self._delivery_key(record)
        if key is not None:
            if key in self._rounds_logged:
                return False
            self._rounds_logged.add(key)
        self.delivery_records.append(record)
        return True

    @staticmethod
    def _delivery_key(record: dict):
        kind = record.get("record")
        if kind == "round":
            return (kind, record["phase_index"], record["round"])
        if kind in ("phase_start", "phase_end"):
            return (kind, record["phase_index"])
        return None

    def global_model(self, record: dict) -> bool:
        if record["slot"] in self._slots_logged:
            return False
        self._slots_logged.add(record["slot"])
        self.global_models.append(record)
        return True

    def membership_event(self, record: dict) -> None:
        self.membership_events.append(record)

    def alarm(self, record: dict) -> bool:
        """Append an alarm. One row per (position, phase_index): a repeat is dropped."""
        key = (record["position"], record["phase_index"])
        if key in self._alarms_logged:
            return False
        self._alarms_logged.add(key)
        self.alarms.append(record)
        return True

    def self_check(self, record: dict) -> None:
        self.self_checks.append(record)

    def replay_done(self, record: dict) -> bool:
        """Append a REPLAY_DONE. One row per (position, phase_index); repeats are dropped."""
        key = (record["position"], record["phase_index"])
        if key in self._dones_logged:
            return False
        self._dones_logged.add(key)
        self.replay_dones.append(record)
        return True

    # -- rewritten documents -----------------------------------------------------
    def write_replay_summary(self, summary: dict) -> None:
        self.replay_summary = json.loads(json.dumps(summary))

    def write_phase_summaries(self, summaries: list) -> None:
        self.phase_summaries = json.loads(json.dumps(summaries))

    def write_replay_masks(self, masks: dict) -> None:
        self.replay_masks = json.loads(json.dumps(masks))

    def write_boards(self, boards: dict) -> None:
        self.boards = json.loads(json.dumps(boards))

    def write_checkpoint(self, checkpoint: dict) -> None:
        self.checkpoint = json.loads(json.dumps(checkpoint))

    def read_checkpoint(self) -> dict | None:
        return None if self.checkpoint is None else json.loads(json.dumps(self.checkpoint))

    def records(self, kind: str) -> list:
        return [r for r in self.delivery_records if r.get("record") == kind]

    def close(self) -> None:
        pass


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            break              # a torn last line from a crash; everything before it is good
    return rows


class RunOutputs(MemoryOutputs):
    """Writes the sinks above under ``directory`` (append mode, resumable)."""

    def __init__(self, directory: Path | str) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # What is already on disk (a resumed run): used only for de-duplication.
        for row in _read_jsonl(self.directory / "global_models.jsonl"):
            self._slots_logged.add(row["slot"])
        for row in _read_jsonl(self.directory / "delivery_log.jsonl"):
            key = self._delivery_key(row)
            if key is not None:
                self._rounds_logged.add(key)
        for name, keys in (("alarms.csv", self._alarms_logged), ("replay_done.csv", self._dones_logged)):
            path = self.directory / name
            if path.exists():
                with path.open(newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        try:
                            keys.add((int(row["position"]), int(row["phase_index"])))
                        except (KeyError, ValueError):
                            continue
        self._delivery = (self.directory / "delivery_log.jsonl").open("a", encoding="utf-8")
        self._models = (self.directory / "global_models.jsonl").open("a", encoding="utf-8")
        self._membership = (self.directory / "membership_events.jsonl").open("a", encoding="utf-8")
        self._csv: dict = {}
        for name, fields in (("alarms.csv", ALARM_FIELDS), ("self_checks.csv", SELF_CHECK_FIELDS),
                             ("replay_done.csv", REPLAY_DONE_FIELDS)):
            path = self.directory / name
            fresh = not path.exists() or path.stat().st_size == 0
            handle = path.open("a", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=fields)
            if fresh:
                writer.writeheader()
                handle.flush()
            self._csv[name] = (handle, writer, fields)

    @staticmethod
    def _line(handle, record: dict) -> None:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()

    def _row(self, name: str, record: dict) -> None:
        handle, writer, fields = self._csv[name]
        writer.writerow({k: record.get(k, "") for k in fields})
        handle.flush()

    def _write_json(self, name: str, payload, keep_previous: bool = False) -> None:
        target = self.directory / name
        temp = target.with_suffix(target.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if keep_previous and target.exists():
            shutil.copyfile(target, target.with_name(target.stem + ".prev" + target.suffix))
        os.replace(temp, target)

    def delivery(self, record: dict) -> bool:
        fresh = super().delivery(record)
        if fresh:
            with self._lock:
                self._line(self._delivery, record)
        return fresh

    def global_model(self, record: dict) -> bool:
        fresh = super().global_model(record)
        if fresh:
            with self._lock:
                self._line(self._models, record)
        return fresh

    def membership_event(self, record: dict) -> None:
        super().membership_event(record)
        with self._lock:
            self._line(self._membership, record)

    def alarm(self, record: dict) -> bool:
        fresh = super().alarm(record)
        if fresh:
            with self._lock:
                self._row("alarms.csv", record)
        return fresh

    def self_check(self, record: dict) -> None:
        super().self_check(record)
        with self._lock:
            self._row("self_checks.csv", record)

    def replay_done(self, record: dict) -> bool:
        fresh = super().replay_done(record)
        if fresh:
            with self._lock:
                self._row("replay_done.csv", record)
        return fresh

    def write_replay_summary(self, summary: dict) -> None:
        super().write_replay_summary(summary)
        with self._lock:
            self._write_json("replay_summary.json", summary)

    def write_phase_summaries(self, summaries: list) -> None:
        super().write_phase_summaries(summaries)
        with self._lock:
            self._write_json("phase_summary.json", summaries)

    def write_replay_masks(self, masks: dict) -> None:
        super().write_replay_masks(masks)
        with self._lock:
            self._write_json("replay_masks.json", masks)

    def write_boards(self, boards: dict) -> None:
        super().write_boards(boards)
        with self._lock:
            self._write_json("boards.json", boards)

    def write_checkpoint(self, checkpoint: dict) -> None:
        super().write_checkpoint(checkpoint)
        with self._lock:
            self._write_json("checkpoint.json", checkpoint, keep_previous=True)

    def read_checkpoint(self) -> dict | None:
        for name in ("checkpoint.json", "checkpoint.prev.json"):
            path = self.directory / name
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    continue
        return None

    def close(self) -> None:
        with self._lock:
            for handle in (self._delivery, self._models, self._membership):
                handle.close()
            for handle, _, _ in self._csv.values():
                handle.close()


def load_checkpoint(run_dir: Path | str) -> dict | None:
    """Read a run directory's checkpoint without opening any sink (used by --resume)."""
    directory = Path(run_dir)
    for name in ("checkpoint.json", "checkpoint.prev.json"):
        path = directory / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
    return None
