"""Protocol-v3 experiment controller: real federated learning over LoRaWAN.

Pure state machine, independent of MQTT (like controller.py for v2, which is
left untouched). The runner feeds it uplinks and downlink status events and
calls tick(); everything it wants to transmit goes through ``publish``.

Timeline of one level of N machines (positions 0..N-1, board = p mod 10 + 1):

    D0  multicast GLOBAL(level, 0) = quantised fl.init(seed); repair until every
        participating board holds it (a board that never does cannot join the
        level, which fails the run)
    U0  boards train and send one 3-fragment UPDATE per machine; the round
        closes when every active position has delivered (or been abandoned
        after the repair budget), or at the round timeout
        -> twin_server.Server.round() over the delivered deltas
    D1  multicast GLOBAL(level, 1) = quantised server model, repaired ...
    ...
    U11 -> D12  GLOBAL(level, 12) is the level's final model

Every round writes a delivery record (which positions' UPDATEs arrived, which
boards held the resulting GLOBAL, timings, fragment and repair counts) and the
run keeps [rounds][positions] uplink/downlink masks the twin can replay.

Where v2 aborted the whole run on a problem with one node, v3 records the
loss and carries on: losses are the experimental variable, not a fault. Only
deterministic misconfiguration (a downlink TTS rejects on every retry, a level
whose boards never received its first global) stops the run.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import secrets
import threading
import time
from typing import Callable

import numpy as np

from controller import ExperimentController, _synchronized
from protocol_v3 import (
    ALL_GLOBAL_FRAGMENTS,
    F_PORT,
    GLOBAL_BYTES,
    GLOBAL_FRAGMENTS,
    HARDWARE_LEVELS,
    LAYERS,
    MSG_ALARM,
    MSG_REPLAY_DONE,
    MSG_BOARD_HELLO,
    MSG_CLASS_C_READY,
    MSG_GLOBAL_FRAGMENT,
    MSG_GLOBAL_STATUS,
    MSG_SELF_CHECK,
    MSG_UPDATE_FRAGMENT,
    MAX_POSITION,
    N_BOARDS,
    ROUND_SLOTS,
    ROUNDS_PER_LEVEL,
    UPDATE_BYTES,
    UPDATE_CHUNK_BYTES,
    UPDATE_FRAGMENTS,
    UPLINK_TYPES,
    Frame,
    ConflictingFragment,
    PartialMessage,
    ProtocolError,
    QuantisedModel,
    board_of_position,
    decode_alarm,
    decode_board_hello,
    decode_frame,
    decode_global,
    decode_global_status,
    decode_replay_done,
    decode_self_check,
    decode_update,
    encode_global,
    encode_repair_request,
    fragment_payload,
    global_round_id,
    level_index_of,
    quantize_model,
    split_update_fragment,
    unflatten,
    update_round_id,
    validate_crc,
)
from run_outputs import MemoryOutputs
from study_plan import (
    active_positions,
    boards_for_level,
    load_fl_modules,
    positions_on_board,
    validate_levels,
)

EventLogger = Callable[[str, str, dict], None]
Publisher = Callable[[str, bytes, str], None]


class Phase(str, Enum):
    WAITING_BOARDS = "waiting_boards"
    DOWNLINK = "downlink"
    UPLOAD = "upload"
    COLLECTING = "collecting"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Board:
    number: int
    device_id: str
    n_machines: int
    fw_version: int
    flags: int
    counted: bool
    hello_count: int = 1
    flash_run_id: int = 0       # run_id the board reported holding in flash
    ready: bool = False
    ready_since: float = 0.0
    ready_slot: int = -1        # round_id slot of the GLOBAL the board said it waits for (3.6)
    counters: Counter = field(default_factory=Counter)


@dataclass
class AcceptedUpdate:
    position: int
    deq: tuple                  # dequantised delta, float64, tensor shapes
    n_train_rows: int
    train_loss: float
    scales: tuple
    arrived_t: float
    base_slot: int = -1         # slot of the GLOBAL the machine trained from
    stale: bool = False         # base_slot older than the round's seed global


@dataclass
class UploadState:
    level_index: int
    round: int                  # 0-based training round (the UPDATE payload's own field)
    header_id: int              # frame round_id slot of its UPDATEs: level_index*13 + round + 1
    seed_header_id: int         # slot of the GLOBAL that seeds it: level_index*13 + round
    started_t: float
    started_utc: str
    updates: dict = field(default_factory=dict)          # position -> AcceptedUpdate
    partials: dict = field(default_factory=dict)         # position -> PartialMessage
    repair_rounds: Counter = field(default_factory=Counter)
    abandoned: set = field(default_factory=set)
    fragments_rx: Counter = field(default_factory=Counter)   # per position, incl. duplicates
    fragments_total: int = 0                                  # every UPDATE fragment received
    repair_downlinks_sent: int = 0
    seed_held_at_close: tuple = ()
    closed: bool = False
    close_reason: str = ""
    ended_t: float = 0.0
    ended_utc: str = ""


@dataclass
class GlobalState:
    """The multicast currently being (or last) delivered."""

    level_index: int
    round: int
    header_id: int
    model: QuantisedModel
    payload: bytes
    crc: int
    frames: list
    participants: tuple                      # boards that must hold it
    pending: deque = field(default_factory=deque)
    attempts: Counter = field(default_factory=Counter)
    board_missing: dict = field(default_factory=dict)
    reported: set = field(default_factory=set)
    late_repairs: Counter = field(default_factory=Counter)
    armed: bool = False                      # multicast allowed to start
    ever_armed: bool = False
    next_send_at: float = 0.0
    burst_ended_at: float | None = None
    repair_rounds: int = 0
    ready_deadline: float = 0.0
    started_t: float = 0.0
    started_utc: str = ""
    ended_t: float = 0.0
    fragments_queued: int = 0
    fragments_sent: int = 0
    bursts: int = 1
    ended: bool = False

    def held(self) -> tuple:
        return tuple(sorted(b for b in self.participants if self.board_missing.get(b) == 0))


@dataclass
class Outbound:
    kind: str                 # "global" | "repair"
    device_id: str
    frame: bytes
    correlation: str
    index: int = 0            # fragment index for "global"
    position: int = -1        # for "repair"
    header_id: int = -1       # round_id of the global a "global" fragment belongs to
    attempts: int = 0
    deadline: float = 0.0


@dataclass
class LevelState:
    level: int
    level_index: int
    positions: tuple
    boards: tuple
    server: object
    started_t: float
    started_utc: str
    rounds: list = field(default_factory=list)          # round records
    up_mask: list = field(default_factory=list)         # [round][position] bool
    base_rows: list = field(default_factory=list)       # [round][position] base_slot or None
    globals_deq: dict = field(default_factory=dict)     # slot -> dequantised GLOBAL (float64, 562)
    down_mask: list = field(default_factory=list)
    initial_global_crc: str = ""
    initial_boards: tuple = ()
    final_crc: str = ""
    final_codes_sha256: str = ""
    status: str = "running"
    ended_t: float | None = None
    ended_utc: str = ""


class HardwareExperimentController:
    def __init__(
        self,
        levels,
        seed: int,
        expected_boards: int,
        publish: Publisher,
        log: EventLogger,
        multicast_device_id: str,
        outputs=None,
        fl_modules=None,
        machine_order: list | None = None,
        research_root=None,
        run_id: int | None = None,
        class_c_guard_seconds: float = 15.0,
        retry_guard_seconds: float = 8.0,
        max_downlink_attempts: int = 5,
        max_repair_rounds: int = 4,
        max_global_repair_rounds: int = 4,
        multicast_report_window_seconds: float = 180.0,
        inflight_downlink_timeout_seconds: float = 45.0,
        ready_wait_seconds: float = 240.0,
        round_timeout_seconds: float = 900.0,
        post_run_collect_seconds: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], str] | None = None,
    ) -> None:
        if not multicast_device_id:
            raise ValueError(
                "v3 delivers GLOBAL by multicast only; set experiment.multicast_device_id"
            )
        if not 1 <= expected_boards <= N_BOARDS:
            raise ValueError(f"expected_boards must be 1..{N_BOARDS}")
        self.levels = validate_levels(levels)
        self.seed = int(seed)
        self.expected_boards = int(expected_boards)
        self.publish = publish
        self.log = log
        self.multicast_device_id = multicast_device_id
        self.out = outputs if outputs is not None else MemoryOutputs()
        self.machine_order = machine_order
        # Random non-zero u16, constant for the whole run, in every GLOBAL (PROTOCOL_V3 3.2).
        self.run_id = int(run_id) if run_id is not None else secrets.randbelow(0xFFFF) + 1
        if not 1 <= self.run_id <= 0xFFFF:
            raise ValueError("run_id must be a non-zero u16")
        self.class_c_guard_seconds = class_c_guard_seconds
        self.retry_guard_seconds = retry_guard_seconds
        self.max_downlink_attempts = max_downlink_attempts
        self.max_repair_rounds = max_repair_rounds
        self.max_global_repair_rounds = max_global_repair_rounds
        self.multicast_report_window_seconds = multicast_report_window_seconds
        self.inflight_downlink_timeout_seconds = inflight_downlink_timeout_seconds
        self.ready_wait_seconds = ready_wait_seconds
        self.round_timeout_seconds = round_timeout_seconds
        self.post_run_collect_seconds = post_run_collect_seconds
        self.clock = clock
        self.wall_clock = wall_clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._lock = threading.RLock()

        # twin_server is imported UNCHANGED from the research folder.
        if fl_modules is None:
            fl_modules = load_fl_modules(research_root)
        self.twin_server, self.fl = fl_modules

        self.boards: dict[int, Board] = {}
        self._board_by_device: dict[str, int] = {}
        self.phase = Phase.WAITING_BOARDS
        self.failure_reason = ""
        self._t0 = clock()
        self._level_pos = -1
        self.level: LevelState | None = None
        self.level_history: list[LevelState] = []
        self._g: GlobalState | None = None
        self._u: UploadState | None = None
        self._pending_round: dict | None = None    # round record awaiting its downlink phase
        self._outbox: deque[Outbound] = deque()
        self._inflight: Outbound | None = None
        self._alarm_keys: set = set()              # (position, level_index)
        self._alarm_levels: dict = {}              # position -> {level_index}
        self.replay_done: dict = {}                # position -> record
        self.counters: Counter = Counter()
        self.alarm_count: Counter = Counter()      # level_index -> alarms
        self.self_check_count = 0
        self._collect_deadline = 0.0
        self._started_utc = ""

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    @property
    @_synchronized
    def finished(self) -> bool:
        return self.phase is Phase.DONE

    @property
    @_synchronized
    def failed(self) -> bool:
        return self.phase is Phase.FAILED

    def _rel(self, now: float) -> float:
        return round(now - self._t0, 3)

    def _fail(self, reason: str) -> None:
        if self.phase is Phase.FAILED:
            return
        self.failure_reason = reason
        self.phase = Phase.FAILED
        self.log("experiment_failed", "server", {"reason": reason})
        if self.level is not None and self.level.status == "running":
            self.level.status = "failed"
        self._outbox.clear()
        self._inflight = None
        self.finalize()

    def fail(self, reason: str) -> None:
        """External stop (operator, overall timeout)."""
        with self._lock:
            self._fail(reason)

    # ------------------------------------------------------------------
    # Uplink entry point
    # ------------------------------------------------------------------
    @_synchronized
    def handle_uplink(self, device_id: str, f_port: int, payload: bytes) -> None:
        if f_port != F_PORT:
            self.log("ignored_wrong_fport", device_id, {"f_port": f_port})
            return
        try:
            frame = decode_frame(payload)
            if frame.message_type not in UPLINK_TYPES:
                raise ProtocolError(
                    f"message type 0x{frame.message_type:02X} is a downlink; it cannot be an uplink"
                )
        except ProtocolError as exc:
            self._reject(device_id, exc)
            return

        try:
            if frame.message_type == MSG_BOARD_HELLO:
                self._handle_hello(device_id, frame)
                return
            board = self._board_for(device_id)
            if board is None:
                self.log(
                    "ignored_device_without_hello",
                    device_id,
                    {"message_type": f"0x{frame.message_type:02X}",
                     "hint": "a board is only known after its BOARD_HELLO"},
                )
                return
            board.counters["uplinks"] += 1
            board.counters["uplink_application_bytes"] += len(payload)
            if frame.message_type == MSG_UPDATE_FRAGMENT:
                self._handle_update_fragment(board, frame)
            elif frame.message_type == MSG_CLASS_C_READY:
                self._handle_ready(board, frame)
            elif frame.message_type == MSG_GLOBAL_STATUS:
                self._handle_status(board, frame)
            elif frame.message_type == MSG_ALARM:
                self._handle_alarm(board, frame)
            elif frame.message_type == MSG_SELF_CHECK:
                self._handle_self_check(board, frame)
            elif frame.message_type == MSG_REPLAY_DONE:
                self._handle_replay_done(board, frame)
        except ProtocolError as exc:
            self._reject(device_id, exc)

    def _reject(self, device_id: str, exc: Exception) -> None:
        self.counters["frames_rejected"] += 1
        self.log("frame_rejected", device_id, {"error": str(exc)})

    def _board_for(self, device_id: str) -> Board | None:
        number = self._board_by_device.get(device_id)
        return None if number is None else self.boards[number]

    # ------------------------------------------------------------------
    # BOARD_HELLO and run start
    # ------------------------------------------------------------------
    def _handle_hello(self, device_id: str, frame: Frame) -> None:
        hello = decode_board_hello(frame)
        if not 1 <= hello.board_number <= N_BOARDS:
            raise ProtocolError(f"BOARD_HELLO board_number {hello.board_number} is outside 1..10")
        # Identity (rev 3.2): board_number is bound to the DevEUI of its FIRST hello.
        known_number = self._board_by_device.get(device_id)
        holder = self.boards.get(hello.board_number)
        if known_number is not None and known_number != hello.board_number:
            self.counters["board_binding_conflicts"] += 1
            self.log(
                "board_number_conflict", device_id,
                {"bound_as": known_number, "now_claims": hello.board_number,
                 "hint": "board.txt on this card changed; ignored until the server restarts"},
            )
            return
        if holder is not None and holder.device_id != device_id:
            self.counters["board_binding_conflicts"] += 1
            self.log(
                "board_number_conflict", device_id,
                {"board": hello.board_number, "bound_to": holder.device_id,
                 "hint": "a second DevEUI claims a bound board number; rejected"},
            )
            return

        # Counts towards the start: SD mounted AND self-check passed, or IMU source
        # (bit 1 is waived for IMU boards).
        counted = hello.imu_source or (hello.sd_mounted and hello.self_check_passed)
        if holder is None:
            board = Board(
                number=hello.board_number, device_id=device_id,
                n_machines=hello.n_machines, fw_version=hello.fw_version,
                flags=hello.flags, counted=counted, flash_run_id=hello.run_id,
            )
            self.boards[board.number] = board
            self._board_by_device[device_id] = board.number
        else:
            board = holder
            board.hello_count += 1          # a reboot: same board, not counted twice
            board.n_machines, board.fw_version, board.flags = (
                hello.n_machines, hello.fw_version, hello.flags,
            )
            board.counted = counted
            board.flash_run_id = hello.run_id
            # A rebooted board has lost its Class C state.
            board.ready = False
        board.counters["uplinks"] += 1

        details = {
            "board": board.number, "n_machines_on_card": hello.n_machines,
            "fw_version": hello.fw_version, "flags": f"0x{hello.flags:02X}",
            "sd_mounted": hello.sd_mounted, "self_check_passed": hello.self_check_passed,
            "imu_source": hello.imu_source, "hello_count": board.hello_count,
            "run_id_in_flash": hello.run_id, "server_run_id": self.run_id,
        }
        self.log("board_hello", device_id, details)
        if hello.run_id not in (0, self.run_id):
            self.log("board_holds_other_run", device_id, {
                "board": board.number, "flash_run_id": hello.run_id, "server_run_id": self.run_id,
                "hint": "the board discards its persisted state when GLOBAL(level, 0) arrives"})
        if not counted:
            self.log("board_not_counted", device_id, {
                "board": board.number,
                "reason": "needs flags bit0 (SD mounted) and bit1 (self-check passed), "
                          "or bit2 (IMU source)", "flags": f"0x{hello.flags:02X}"})
        if board.hello_count > 1 and self.phase not in (Phase.WAITING_BOARDS, Phase.DONE):
            self.log("board_rebooted_mid_run", device_id,
                     {"board": board.number, "phase": self.phase.value,
                      "level": None if self.level is None else self.level.level})
        self._write_boards()
        self._maybe_start()

    def counted_boards(self) -> list[int]:
        return sorted(b.number for b in self.boards.values() if b.counted)

    def _maybe_start(self) -> None:
        if self.phase is not Phase.WAITING_BOARDS:
            return
        counted = self.counted_boards()
        if len(counted) < self.expected_boards:
            return
        needed = boards_for_level(max(self.levels))
        missing = [b for b in needed if b not in counted]
        if missing:
            self._fail(
                f"{len(counted)} boards said hello but level {max(self.levels)} needs "
                f"boards {list(needed)}; missing {missing}"
            )
            return
        self._started_utc = self.wall_clock()
        self.log("experiment_started", "server",
                 {"boards": counted, "levels": self.levels, "seed": self.seed,
                  "run_id": self.run_id})
        self._begin_level(0)

    def _write_boards(self) -> None:
        self.out.write_boards({
            str(n): {
                "device_id": b.device_id, "n_machines_on_card": b.n_machines,
                "fw_version": b.fw_version, "flags": b.flags, "counted": b.counted,
                "hello_count": b.hello_count, "flash_run_id": b.flash_run_id,
            }
            for n, b in sorted(self.boards.items())
        })

    # ------------------------------------------------------------------
    # Level / global model
    # ------------------------------------------------------------------
    def _begin_level(self, level_pos: int) -> None:
        now = self.clock()
        self._level_pos = level_pos
        level = self.levels[level_pos]
        level_index = level_index_of(level)
        server = self.twin_server.Server(self.fl.init(list(LAYERS), self.seed))
        self.level = LevelState(
            level=level, level_index=level_index,
            positions=active_positions(level), boards=boards_for_level(level),
            server=server, started_t=now, started_utc=self.wall_clock(),
        )
        self.level_history.append(self.level)
        self.log("level_started", "server", {
            "level": level, "level_index": level_index,
            "positions": len(self.level.positions), "boards": list(self.level.boards),
        })
        self._begin_downlink(self._make_global(0), now)

    def _make_global(self, round_index: int) -> GlobalState:
        level = self.level
        # Same path as twin.downlink_message: float32-cast server weights,
        # then the Q_DYN_TENSOR codec.
        params32 = tuple(np.asarray(p, dtype=np.float32) for p in level.server.p)
        model = quantize_model(params32)
        payload = encode_global(level.level_index, round_index, model, self.run_id)
        crc = validate_crc(payload, GLOBAL_BYTES)
        header = global_round_id(level.level_index, round_index)
        # Every dequantised GLOBAL of the level is kept: a machine may train from an
        # older one (base_slot) and the server must reconstruct its model.
        level.globals_deq[header] = decode_global(payload).model.dequantize()
        frames = fragment_payload(payload, MSG_GLOBAL_FRAGMENT, header)
        if len(frames) != GLOBAL_FRAGMENTS:
            raise AssertionError("GLOBAL did not produce 3 fragments")
        self.out.global_model({
            "level": level.level, "level_index": level.level_index, "round": round_index,
            "round_id": header, "run_id": self.run_id, "crc32": f"{crc:08X}", "scales_f32": list(model.wire_scales()),
            "codes": [int(c) for c in model.codes], "codes_sha256": model.codes_sha256(),
        })
        return GlobalState(
            level_index=level.level_index, round=round_index, header_id=header,
            model=model, payload=payload, crc=crc, frames=frames,
            participants=level.boards,
        )

    def _begin_downlink(self, g: GlobalState, now: float) -> None:
        self._g = g
        self.phase = Phase.DOWNLINK
        g.started_t = now
        g.started_utc = self.wall_clock()
        g.ready_deadline = now + self.ready_wait_seconds
        g.board_missing = {b: ALL_GLOBAL_FRAGMENTS for b in g.participants}
        g.pending = deque(range(GLOBAL_FRAGMENTS))
        # The round this global seeds. It is created NOW, not when the delivery
        # phase ends: a board that hears the global first trains and uploads at
        # once, long before a slower board has been repaired.
        if g.round < ROUNDS_PER_LEVEL:
            self._u = UploadState(
                level_index=g.level_index, round=g.round,
                header_id=update_round_id(g.level_index, g.round),
                seed_header_id=g.header_id,
                started_t=now, started_utc=self.wall_clock(),
            )
        else:
            self._u = None
        self.log("global_prepared", "server", {
            "level": self.level.level, "round": g.round, "round_id": g.header_id,
            "crc32": f"{g.crc:08X}", "participants": list(g.participants),
        })

    def _tick_downlink(self, now: float) -> None:
        g = self._g
        if g is None or g.ended:
            return
        if not g.armed and not g.ever_armed:
            waiting = [b for b in g.participants if not self._ready_for(self.boards[b], g)]
            if not waiting:
                latest = max(self.boards[b].ready_since for b in g.participants)
                if now >= latest + self.class_c_guard_seconds:
                    self._arm(g, now, "all_boards_ready")
            elif now >= g.ready_deadline:
                self.log("boards_not_ready", "server", {"boards": waiting,
                         "waited_seconds": self.ready_wait_seconds})
                self._arm(g, now, "ready_wait_expired")
            return
        # A burst has drained and the report window has lapsed: decide what to repair.
        if (
            g.armed is False and g.burst_ended_at is not None and not g.pending
            and self._inflight_kind("global") is None
            and now - g.burst_ended_at >= self.multicast_report_window_seconds
            and not g.ended
        ):
            self._refresh_global(g, now)

    @staticmethod
    def _ready_for(board: Board, g: GlobalState) -> bool:
        """Is the board in Class C waiting for THIS global (PROTOCOL_V3 3.6)?

        READY carries the slot of the GLOBAL the board waits for. A level's first
        global is also accepted on any level-start slot (a multiple of 13): a
        board reads its next level from levels.csv, which lists all seven
        levels, whereas a sweep may run only some of them.
        """
        if not board.ready:
            return False
        if board.ready_slot == g.header_id:
            return True
        return g.round == 0 and board.ready_slot >= 0 and board.ready_slot % ROUND_SLOTS == 0

    def _arm(self, g: GlobalState, now: float, why: str) -> None:
        g.armed = True
        g.ever_armed = True
        g.next_send_at = now
        self.log("global_multicast_starting", "server",
                 {"round": g.round, "round_id": g.header_id, "why": why,
                  "fragments": len(g.frames)})

    def _inflight_kind(self, kind: str):
        if self._inflight is not None and self._inflight.kind == kind:
            return self._inflight
        return None

    def _refresh_global(self, g: GlobalState, now: float) -> None:
        """v2's union repair: one transmission serves every board still missing anything."""
        union = 0
        for board in g.participants:
            union |= g.board_missing.get(board, ALL_GLOBAL_FRAGMENTS)
        if union == 0:
            self._end_downlink(now, gave_up=False)
            return
        if g.pending or self._inflight_kind("global") is not None:
            return      # a burst is still on the air; decide once it has drained
        outstanding = set(g.participants) - g.reported
        window_open = (
            g.burst_ended_at is not None
            and now - g.burst_ended_at < self.multicast_report_window_seconds
        )
        if outstanding and window_open:
            self.log("multicast_awaiting_reports", "server", {"outstanding": sorted(outstanding)})
            return
        g.repair_rounds += 1
        if g.repair_rounds > self.max_global_repair_rounds:
            self._end_downlink(now, gave_up=True)
            return
        wanted = [i for i in range(GLOBAL_FRAGMENTS) if union & (1 << i)]
        g.attempts.clear()
        g.reported.clear()
        g.burst_ended_at = None
        g.pending = deque(wanted)
        g.armed = True
        g.bursts += 1
        g.next_send_at = now + self.retry_guard_seconds
        self.log("multicast_repair", "server", {
            "round": g.round, "repair_round": g.repair_rounds, "union_mask": f"{union:04X}",
            "fragments": wanted,
            "still_missing": {str(b): f"{m:04X}" for b, m in g.board_missing.items() if m},
        })

    def _end_downlink(self, now: float, gave_up: bool) -> None:
        g = self._g
        g.ended = True
        g.armed = False
        g.pending.clear()
        if self._inflight_kind("global") is not None:
            self._inflight = None
        g.ended_t = now
        held = g.held()
        missing = sorted(set(g.participants) - set(held))
        self.log("global_delivery_end", "server", {
            "level": self.level.level, "round": g.round, "held": list(held),
            "missing": missing, "gave_up": gave_up,
            "fragments_sent": g.fragments_sent, "bursts": g.bursts,
        })
        if g.round == 0:
            self.level.initial_global_crc = f"{g.crc:08X}"
            self.level.initial_boards = held
            self.out.delivery({
                "record": "level_start", "level": self.level.level,
                "level_index": self.level.level_index, "seed": self.seed,
                "active_positions": list(self.level.positions),
                "participating_boards": list(self.level.boards),
                "boards_holding_initial_global": list(held),
                "initial_global_crc32": self.level.initial_global_crc,
                "global_fragments_sent": g.fragments_sent, "global_bursts": g.bursts,
                "started_utc": self.level.started_utc,
                "delivery_duration_s": round(now - g.started_t, 3),
            })
            if missing:
                self._fail(
                    f"boards {missing} never received GLOBAL(level {self.level.level}, 0) "
                    f"and cannot take part in the level"
                )
                return
        else:
            self._finish_round_record(g)
        if g.round == ROUNDS_PER_LEVEL:
            self._end_level(now)
        else:
            self._begin_upload(now, g.round)

    # ------------------------------------------------------------------
    # Upload phase
    # ------------------------------------------------------------------
    def _begin_upload(self, now: float, round_index: int) -> None:
        level = self.level
        u = self._u
        assert u is not None and u.round == round_index
        self.phase = Phase.UPLOAD
        u.started_t = now              # the round timeout runs from the end of delivery
        u.started_utc = self.wall_clock()
        self.log("upload_started", "server", {
            "level": level.level, "round": round_index,
            "expected_positions": len(level.positions),
            "already_arrived": sorted(u.updates),
        })
        # Boards that finished uploading during delivery announced readiness
        # then; nothing has looked at what they are missing yet.
        for number in level.boards:
            board = self.boards[number]
            if board.ready and board.ready_slot == u.header_id:
                self._repair_board(board)

    def _tick_upload(self, now: float) -> None:
        u = self._u
        if u is None or u.closed:
            return
        level = self.level
        resolved = all(p in u.updates or p in u.abandoned for p in level.positions)
        if resolved:
            reason = "all_delivered" if not (set(level.positions) - set(u.updates)) else "all_resolved"
            self._close_round(now, reason)
        elif now - u.started_t >= self.round_timeout_seconds:
            self._close_round(now, "timeout")

    def _close_round(self, now: float, reason: str) -> None:
        level, u, seed_global = self.level, self._u, self._g
        u.closed = True
        u.close_reason = reason
        u.ended_t = now
        u.ended_utc = self.wall_clock()
        u.seed_held_at_close = seed_global.held() if seed_global is not None else ()
        arrived = sorted(u.updates)
        missing = [p for p in level.positions if p not in u.updates]
        pairs = [(u.updates[p].deq, 1.0) for p in arrived]
        self.log("round_closing", "server", {
            "level": level.level, "round": u.round, "reason": reason,
            "arrived": arrived, "missing": missing, "abandoned": sorted(u.abandoned),
        })
        # Empty pairs leave the server unchanged and do not advance FedAdam's t.
        level.server.round(pairs)
        # Any UPDATE_REPAIR still queued belongs to a round that no longer exists.
        self._outbox = deque(o for o in self._outbox if o.kind != "repair")
        self._pending_round = {
            "record": "round",
            "level": level.level, "level_index": level.level_index,
            "round": u.round, "round_id": u.header_id, "seed_round_id": u.seed_header_id,
            "active_positions": list(level.positions),
            "participating_boards": list(level.boards),
            "update_arrived_positions": arrived,
            "update_missing_positions": missing,
            "update_abandoned_positions": sorted(u.abandoned),
            "boards_holding_seed_global": list(u.seed_held_at_close),
            "close_reason": reason,
            "update_fragments_received": u.fragments_total,
            "update_fragments_by_position": {str(p): int(n) for p, n in sorted(u.fragments_rx.items())},
            "update_repair_requests": int(sum(u.repair_rounds.values())),
            "update_repairs_by_position": {str(p): int(n) for p, n in sorted(u.repair_rounds.items())},
            "update_repair_downlinks_sent": u.repair_downlinks_sent,
            "base_slot_by_position": {str(p): u.updates[p].base_slot for p in arrived},
            "stale_base_positions": [p for p in arrived if u.updates[p].stale],
            "train_loss": {str(p): u.updates[p].train_loss for p in arrived},
            "n_train_rows": {str(p): u.updates[p].n_train_rows for p in arrived},
            "timings": {
                "upload_start_s": self._rel(u.started_t),
                "upload_end_s": self._rel(u.ended_t),
                "upload_duration_s": round(u.ended_t - u.started_t, 3),
                "upload_start_utc": u.started_utc,
                "upload_end_utc": u.ended_utc,
            },
        }
        self._begin_downlink(self._make_global(u.round + 1), now)

    def _finish_round_record(self, g: GlobalState) -> None:
        record = self._pending_round
        level = self.level
        held = g.held()
        record["boards_holding_result_global"] = list(held)
        record["result_global_round"] = g.round
        record["result_global_crc32"] = f"{g.crc:08X}"
        record["global_fragments_sent"] = g.fragments_sent
        record["global_fragments_queued"] = g.fragments_queued
        record["global_bursts"] = g.bursts
        record["global_repair_rounds"] = g.repair_rounds
        record["timings"].update({
            "downlink_start_s": self._rel(g.started_t),
            "downlink_end_s": self._rel(g.ended_t),
            "downlink_duration_s": round(g.ended_t - g.started_t, 3),
            "downlink_start_utc": g.started_utc,
            "downlink_end_utc": self.wall_clock(),
        })
        arrived = set(record["update_arrived_positions"])
        level.up_mask.append([p in arrived for p in level.positions])
        bases = record["base_slot_by_position"]
        level.base_rows.append([bases.get(str(p)) for p in level.positions])
        level.down_mask.append([board_of_position(p) in held for p in level.positions])
        level.rounds.append(record)
        self._pending_round = None
        self.out.delivery(record)
        self.log("round_complete", "server", {
            "level": level.level, "round": record["round"],
            "arrived": len(arrived), "expected": len(level.positions),
            "boards_holding_result_global": list(held),
        })
        self._write_masks()

    def _end_level(self, now: float) -> None:
        level = self.level
        level.status = "complete"
        level.ended_t = now
        level.ended_utc = self.wall_clock()
        level.final_crc = f"{self._g.crc:08X}"
        level.final_codes_sha256 = self._g.model.codes_sha256()
        self.log("level_complete", "server", {
            "level": level.level, "rounds": len(level.rounds),
            "final_crc32": level.final_crc,
        })
        summary = self._level_summary(level)
        self.out.delivery({"record": "level_end", **summary})
        self.finalize()
        if self._level_pos + 1 < len(self.levels):
            self._begin_level(self._level_pos + 1)
        else:
            self.phase = Phase.COLLECTING
            self._collect_deadline = now + self.post_run_collect_seconds
            self.log("sweep_complete", "server", {
                "collecting_alarms_for_seconds": self.post_run_collect_seconds})

    # ------------------------------------------------------------------
    # UPDATE fragments
    # ------------------------------------------------------------------
    def _handle_update_fragment(self, board: Board, frame: Frame) -> None:
        """One UPDATE fragment. Rev 3.1: its body starts with the position byte, so
        it is attributable on its own, whatever else the board sent around it."""
        u = self._u
        board.counters["update_fragments_received"] += 1
        if self.phase not in (Phase.DOWNLINK, Phase.UPLOAD) or u is None or u.closed:
            self.counters["late_update_fragments"] += 1
            self.log("late_update_fragment", board.device_id, {
                "board": board.number, "phase": self.phase.value,
                "round_id": frame.round_id, "fragment": frame.fragment_index,
            })
            return
        if frame.round_id != u.header_id:
            self.counters["stale_update_fragments"] += 1
            self.log("stale_update_fragment", board.device_id, {
                "board": board.number, "round_id": frame.round_id, "expected": u.header_id,
            })
            return
        if frame.fragment_count != UPDATE_FRAGMENTS:
            raise ProtocolError(
                f"UPDATE fragment count is {frame.fragment_count}; expected {UPDATE_FRAGMENTS}"
            )
        position, chunk = split_update_fragment(frame)
        u.fragments_total += 1
        if not self._position_valid(board, position, "update_position_rejected"):
            return
        u.fragments_rx[position] += 1
        if position in u.updates:
            self.log("duplicate_update_fragment", board.device_id,
                     {"position": position, "fragment": frame.fragment_index})
            return
        partial = u.partials.setdefault(
            position, PartialMessage(UPDATE_BYTES, UPDATE_FRAGMENTS, UPDATE_CHUNK_BYTES))
        try:
            payload = partial.add(chunk)
        except ConflictingFragment as exc:
            # Rev 3.2: a duplicate with different bytes replaces nothing.
            self.counters["update_fragment_conflicts"] += 1
            self.log("update_fragment_conflict", board.device_id, {
                "position": position, "fragment": frame.fragment_index, "error": str(exc)})
            return
        except ProtocolError as exc:
            # Wrong length or a duplicate that changed its bytes: start the position over.
            u.partials.pop(position, None)
            self.log("update_partial_reset", board.device_id, {"position": position, "error": str(exc)})
            return
        self.log("update_fragment", board.device_id, {
            "position": position, "fragment": frame.fragment_index,
            "have": sorted(partial.parts)})
        if payload is not None:
            self._complete_update(board, position, payload)

    def _position_valid(self, board: Board, position: int, event: str, frame=None) -> bool:
        level = self.level
        if board_of_position(position) != board.number:
            self.log(event, board.device_id, {
                "board": board.number, "position": position,
                "reason": f"position {position} belongs to board {board_of_position(position)}"})
            return False
        if level is None or position not in level.positions:
            self.log(event, board.device_id, {
                "board": board.number, "position": position,
                "reason": "position is not active at this level"})
            return False
        return True

    def _reject_update(self, u: UploadState, board: Board, position: int, counter: str,
                       reason: str) -> None:
        """A CRC-valid UPDATE that must not be aggregated. Resending would not help."""
        u.partials.pop(position, None)
        u.abandoned.add(position)
        self.counters[counter] += 1
        self.log("update_rejected", board.device_id, {
            "position": position, "board": board.number, "error": reason})

    def _complete_update(self, board: Board, hint: int, payload: bytes) -> None:
        u, level = self._u, self.level
        try:
            update = decode_update(payload)
        except ProtocolError as exc:
            u.partials.pop(hint, None)
            self.counters["update_crc_failures"] += 1
            self.log("update_rejected", board.device_id,
                     {"position": hint, "error": str(exc),
                      "hint": "fragments discarded; the repair flow asks for the position again"})
            return
        if update.level_index != level.level_index or update.round != u.round:
            u.partials.pop(hint, None)
            self.log("update_rejected", board.device_id, {
                "position": update.position,
                "error": f"UPDATE is for level_index {update.level_index} round {update.round}; "
                         f"expected {level.level_index} round {u.round}"})
            return
        position = update.position
        if position != hint:
            # The fragments' position byte and the position inside the UPDATE disagree.
            u.partials.pop(hint, None)
            self.counters["update_position_mismatches"] += 1
            self.log("update_rejected", board.device_id, {
                "position": hint,
                "error": f"fragments carry position {hint} but the reassembled UPDATE says {position}"})
            return
        if position in u.updates:
            u.partials.pop(position, None)
            return
        if update.run_id != self.run_id:
            self._reject_update(u, board, position, "update_run_id_mismatches",
                                f"UPDATE run_id {update.run_id} != this run's {self.run_id}")
            return
        first_slot = global_round_id(level.level_index, 0)
        if not (first_slot <= update.base_slot <= u.seed_header_id
                and update.base_slot in level.globals_deq):
            self._reject_update(
                u, board, position, "update_base_slot_rejections",
                f"base_slot {update.base_slot} is unknown or outside level {level.level} "
                f"(valid {first_slot}..{u.seed_header_id})")
            return
        delta_flat = update.model.dequantize()
        stale = update.base_slot != u.seed_header_id
        if stale:
            # The machine trained from an older GLOBAL: rebuild its model and take
            # the difference to the CURRENT global, as dropout_fl's stale copy does.
            client_model = level.globals_deq[update.base_slot] + delta_flat
            deq = unflatten(client_model - level.globals_deq[u.seed_header_id])
            self.counters["stale_base_updates"] += 1
        else:
            deq = unflatten(delta_flat)
        u.updates[position] = AcceptedUpdate(
            position=position, deq=deq, n_train_rows=update.n_train_rows,
            train_loss=update.train_loss, scales=update.model.scales,
            arrived_t=self.clock(), base_slot=update.base_slot, stale=stale,
        )
        u.partials.pop(position, None)
        self.log("update_complete", board.device_id, {
            "position": position, "board": board.number, "crc32": f"{update.crc32:08X}",
            "train_loss": update.train_loss, "n_train_rows": update.n_train_rows,
            "base_slot": update.base_slot, "stale_base": stale})
        g = self._g
        # A stale-base UPDATE says the board did NOT hold the seed global, so it proves nothing.
        if g is not None and g.header_id == u.seed_header_id and not stale:
            self._infer_seed_held(board, g, "update_received")
            self._finish_delivery_if_complete(g, self.clock())

    # ------------------------------------------------------------------
    # CLASS_C_READY and repair
    # ------------------------------------------------------------------
    def _set_ready(self, board: Board, slot: int) -> None:
        board.ready = True
        board.ready_slot = slot
        board.ready_since = self.clock()

    def _handle_ready(self, board: Board, frame: Frame) -> None:
        if frame.fragment_index != 0 or frame.fragment_count != GLOBAL_FRAGMENTS:
            raise ProtocolError("CLASS_C_READY declared the wrong GLOBAL fragment count")
        if len(frame.data) != 2:
            raise ProtocolError("CLASS_C_READY must report uplink DR and RX2 DR")
        uplink_dr, rx2_dr = frame.data
        if rx2_dr not in (4, 5):
            self.log("class_c_ready_bad_rx2", board.device_id, {
                "board": board.number, "rx2_dr": rx2_dr,
                "hint": "221-byte Class-C frames need DR4/DR5; check the device's "
                        "desired_rx2_data_rate_index and rejoin"})
            return
        slot = frame.round_id       # the GLOBAL the board waits for (PROTOCOL_V3 3.6)
        self._set_ready(board, slot)
        self.log("class_c_ready", board.device_id, {
            "board": board.number, "waiting_for_slot": slot,
            "uplink_dr": uplink_dr, "rx2_dr": rx2_dr})
        u = self._u
        # READY for the slot this round's UPDATEs carry = "my upload (or repair) is done".
        if (self.phase in (Phase.DOWNLINK, Phase.UPLOAD) and u is not None
                and not u.closed and slot == u.header_id):
            self._repair_board(board)

    def _repair_board(self, board: Board) -> None:
        """The board says its upload is done: ask for whatever is still missing.

        Only for a board known to hold the global that seeds this round: before
        that it has nothing to upload, and its readiness merely means it is
        listening for the multicast.
        """
        u, level, g = self._u, self.level, self._g
        if u is None or u.closed or g is None or g.header_id != u.seed_header_id:
            return
        if board.number not in level.boards or g.board_missing.get(board.number) != 0:
            return
        asked = False
        for position in positions_on_board(board.number, level.level):
            if position in u.updates or position in u.abandoned:
                continue
            partial = u.partials.get(position)
            missing = partial.missing_mask() if partial is not None else (1 << UPDATE_FRAGMENTS) - 1
            if not missing:
                continue
            asked |= self._request_update_repair(board, position, missing)
        if asked:
            # Repair makes the board leave Class C to resend, so its readiness is stale.
            board.ready = False

    def _request_update_repair(self, board: Board, position: int, missing: int) -> bool:
        u = self._u
        u.repair_rounds[position] += 1
        rounds = u.repair_rounds[position]
        if rounds > self.max_repair_rounds:
            u.repair_rounds[position] -= 1
            u.abandoned.add(position)
            self.log("update_abandoned", board.device_id, {
                "position": position, "board": board.number,
                "repair_rounds": self.max_repair_rounds,
                "missing_mask": f"{missing:04X}"})
            return False
        frame = encode_repair_request(u.header_id, position, missing)
        correlation = f"v3-{u.header_id}-p{position}-repair{rounds}"
        self._outbox.append(Outbound(
            kind="repair", device_id=board.device_id, frame=frame,
            correlation=correlation, position=position))
        board.counters["update_repairs_requested"] += 1
        self.log("update_repair_requested", board.device_id, {
            "position": position, "board": board.number, "round": rounds,
            "missing_mask": f"{missing:04X}",
            "missing_fragments": [i for i in range(UPDATE_FRAGMENTS) if missing >> i & 1],
            "application_bytes": len(frame), "correlation_id": correlation})
        return True

    # ------------------------------------------------------------------
    # GLOBAL_STATUS
    # ------------------------------------------------------------------
    def _handle_status(self, board: Board, frame: Frame) -> None:
        if frame.fragment_count != GLOBAL_FRAGMENTS:
            raise ProtocolError("GLOBAL_STATUS declares the wrong GLOBAL fragment count")
        missing = decode_global_status(frame)          # exactly 2 bytes, bits 0..2 (rev 3.2)
        g = self._g
        now = self.clock()
        if g is None or board.number not in g.participants:
            # A board with no machines at this level, or before the sweep began:
            # it is in Class C, which is all that matters here.
            self._set_ready(board, frame.round_id)
            self.log("global_status_ignored", board.device_id,
                     {"board": board.number, "missing": f"{missing:03X}",
                      "reason": "board is not a participant"})
            return

        if self.phase not in (Phase.DOWNLINK, Phase.UPLOAD):
            self._set_ready(board, frame.round_id)
            return
        if frame.round_id == g.header_id:
            if self.phase is Phase.UPLOAD:
                self._record_late_seed_report(board, g, missing, now)
            elif not g.ever_armed:
                # Nothing has been sent yet: the board is simply waiting in Class C.
                self._set_ready(board, frame.round_id)
                self.log("global_status_while_waiting", board.device_id,
                         {"board": board.number, "implied_class_c_ready": True})
            else:
                self._record_global_report(board, g, missing, now)
                self._refresh_global_if_due(g, now)
        elif frame.round_id == g.header_id + 1 and g.round < ROUNDS_PER_LEVEL:
            # The board is waiting for the NEXT global, so it has finished this
            # round; it heard nothing because nothing has been sent. That is both
            # an implied CLASS_C_READY and proof it holds the seed global.
            self._infer_seed_held(board, g, "status_for_next_global")
            self._set_ready(board, frame.round_id)
            self.log("global_status_while_waiting", board.device_id, {
                "board": board.number, "implied_class_c_ready": True})
            self._repair_board(board)
            self._finish_delivery_if_complete(g, now)
        else:
            self.log("global_status_wrong_round", board.device_id, {
                "board": board.number, "round_id": frame.round_id,
                "expected": g.header_id})

    def _finish_delivery_if_complete(self, g: GlobalState, now: float) -> None:
        """Inferred acks may be the last thing a delivery phase was waiting for."""
        if (
            self.phase is Phase.DOWNLINK and not g.ended and g.ever_armed
            and all(g.board_missing.get(b) == 0 for b in g.participants)
        ):
            self._refresh_global(g, now)

    def _infer_seed_held(self, board: Board, g: GlobalState, why: str) -> None:
        """Evidence that a board holds the seed global even though its ack never arrived.

        Acks are unconfirmed uplinks and can be lost; a valid UPDATE for round r
        (or a status asking for global r+1) can only come from a board that
        received global r.
        """
        if board.number in g.participants and g.board_missing.get(board.number) != 0:
            g.board_missing[board.number] = 0
            g.reported.add(board.number)
            self.log("global_held_inferred", board.device_id,
                     {"board": board.number, "round": g.round, "evidence": why})

    def _record_global_report(self, board: Board, g: GlobalState, missing: int,
                              now: float) -> None:
        g.board_missing[board.number] = missing
        g.reported.add(board.number)
        if missing == 0:
            board.ready = False       # it has left Class C to train
            board.counters["global_acknowledged"] += 1
            self.log("global_acknowledged", board.device_id,
                     {"board": board.number, "round": g.round})
        else:
            board.ready = True        # still listening, still missing something
            board.ready_slot = g.header_id
            board.ready_since = now
            self.log("global_missing", board.device_id, {
                "board": board.number, "missing_bitmap": f"0x{missing:03X}",
                "fragments": [i for i in range(GLOBAL_FRAGMENTS) if missing >> i & 1]})

    def _refresh_global_if_due(self, g: GlobalState, now: float) -> None:
        if g.ended or not g.ever_armed:
            return
        self._refresh_global(g, now)

    def _record_late_seed_report(self, board: Board, g: GlobalState, missing: int,
                                 now: float) -> None:
        """A board reports about the global that seeded THIS round (during upload)."""
        if missing == 0:
            g.board_missing[board.number] = 0
            self.log("global_late_acknowledged", board.device_id,
                     {"board": board.number, "round": g.round})
            return
        g.board_missing[board.number] = missing
        if g.late_repairs[board.number] >= self.max_global_repair_rounds:
            self.log("global_late_repair_exhausted", board.device_id, {"board": board.number})
            return
        g.late_repairs[board.number] += 1
        wanted = [i for i in range(GLOBAL_FRAGMENTS) if missing >> i & 1]
        for index in wanted:
            if index not in g.pending:
                g.pending.append(index)
        if self._inflight_kind("global") is None:
            g.attempts.clear()
        g.armed = True
        g.bursts += 1
        g.next_send_at = now + self.retry_guard_seconds
        self.log("global_late_repair", board.device_id, {
            "board": board.number, "fragments": wanted, "round": g.round})

    # ------------------------------------------------------------------
    # ALARM / SELF_CHECK
    # ------------------------------------------------------------------
    def _own_position(self, board: Board, position: int, event: str) -> bool:
        """position mod 10 + 1 must equal the board number bound to this DevEUI."""
        if position > MAX_POSITION or board_of_position(position) != board.number:
            self.counters["results_rejected"] += 1
            self.log(event, board.device_id, {
                "board": board.number, "position": position,
                "reason": f"position {position} does not belong to bound board {board.number}"})
            return False
        return True

    def _handle_alarm(self, board: Board, frame: Frame) -> None:
        alarm = decode_alarm(frame)
        if not self._own_position(board, alarm.position, "alarm_rejected"):
            return
        if alarm.level_index >= len(HARDWARE_LEVELS):
            raise ProtocolError(f"ALARM level_index {alarm.level_index} is outside 0..6")
        key = (alarm.position, alarm.level_index)        # once per (position, level_index)
        if key in self._alarm_keys:
            self.log("duplicate_alarm", board.device_id,
                     {"position": alarm.position, "level_index": alarm.level_index})
            return
        self._alarm_keys.add(key)
        self._alarm_levels.setdefault(alarm.position, set()).add(alarm.level_index)
        record = {
            "timestamp_utc": self.wall_clock(), "device_id": board.device_id,
            "board": board.number, "position": alarm.position,
            "level_index": alarm.level_index, "level": HARDWARE_LEVELS[alarm.level_index],
            "acquisition_index": alarm.acquisition_index,
            "score": alarm.score, "threshold": alarm.threshold,
        }
        self.out.alarm(record)
        self.alarm_count[alarm.level_index] += 1
        self.log("alarm", board.device_id, {k: record[k] for k in (
            "position", "level_index", "acquisition_index", "score", "threshold")})

    def _handle_self_check(self, board: Board, frame: Frame) -> None:
        check = decode_self_check(frame)
        if not self._own_position(board, check.position, "self_check_rejected"):
            return
        record = {
            "timestamp_utc": self.wall_clock(), "device_id": board.device_id,
            "board": board.number, "position": check.position,
            "rows_checked": check.rows_checked, "max_abs_diff": check.max_abs_diff,
            "median_abs_diff": check.median_abs_diff,
        }
        self.out.self_check(record)
        self.self_check_count += 1
        self.log("self_check", board.device_id, {k: record[k] for k in (
            "position", "rows_checked", "max_abs_diff", "median_abs_diff")})

    def _handle_replay_done(self, board: Board, frame: Frame) -> None:
        done = decode_replay_done(frame)
        if not self._own_position(board, done.position, "replay_done_rejected"):
            return
        record = {
            "timestamp_utc": self.wall_clock(), "device_id": board.device_id,
            "board": board.number, "position": done.position,
            "level_mask": done.level_mask, "levels_scored": done.levels(),
            "acquisitions_scored": done.acquisitions_scored,
            "alarms_latched": done.alarms_latched,
        }
        if done.position in self.replay_done:
            self.log("duplicate_replay_done", board.device_id, {"position": done.position})
        self.replay_done[done.position] = record
        self.out.replay_done(record)
        self.log("replay_done", board.device_id, {k: record[k] for k in (
            "position", "levels_scored", "acquisitions_scored", "alarms_latched")})

    def replay_summary(self) -> dict:
        """Per position: alarm, no alarm, or never finished (rev 3.2, REPLAY_DONE).

        A machine with no alarm is only "no_alarm" if its board said REPLAY_DONE for
        every level it took part in; otherwise silence means the replay did not finish.
        """
        completed = [lv for lv in self.level_history if lv.status == "complete"]
        expected_positions = sorted({p for lv in completed for p in lv.positions})
        positions = {}
        counts = Counter()
        for position in expected_positions:
            expected_levels = sorted(lv.level_index for lv in completed if position in lv.positions)
            alarm_levels = sorted(self._alarm_levels.get(position, ()))
            done = self.replay_done.get(position)
            entry = {
                "board": board_of_position(position),
                "levels_expected": expected_levels,
                "replay_done": done is not None,
                "levels_scored": [] if done is None else done["levels_scored"],
                "levels_missing": expected_levels if done is None else [
                    i for i in expected_levels if i not in done["levels_scored"]],
                "acquisitions_scored": 0 if done is None else done["acquisitions_scored"],
                "alarms_latched": None if done is None else done["alarms_latched"],
                "alarm_messages": len(alarm_levels),
                "alarm_level_indices": alarm_levels,
            }
            if done is None:
                status = "never_finished"
            elif entry["levels_missing"]:
                status = "incomplete_levels"
            elif done["alarms_latched"] > 0 or alarm_levels:
                status = "alarm"
            else:
                status = "no_alarm"
            entry["status"] = status
            entry["alarm_messages_lost"] = (
                done is not None and done["alarms_latched"] > len(alarm_levels))
            if machine_id := (self.machine_order[position] if self.machine_order else None):
                entry["machine"] = machine_id
            positions[str(position)] = entry
            counts[status] += 1
        return {
            "run_id": self.run_id,
            "levels_completed": [lv.level for lv in completed],
            "positions_expected": len(expected_positions),
            "counts": dict(sorted(counts.items())),
            "alarm_messages_lost_positions": [
                int(p) for p, e in positions.items() if e["alarm_messages_lost"]],
            "positions": positions,
        }

    # ------------------------------------------------------------------
    # Downlink pump
    # ------------------------------------------------------------------
    @_synchronized
    def tick(self) -> None:
        now = self.clock()
        if self.phase in (Phase.WAITING_BOARDS, Phase.DONE, Phase.FAILED):
            return
        self._expire_inflight(now)
        if self.phase is Phase.DOWNLINK:
            self._tick_downlink(now)
        if self.phase is Phase.UPLOAD:
            self._tick_upload(now)
        if self.phase is Phase.COLLECTING and now >= self._collect_deadline:
            self.phase = Phase.DONE
            self.log("run_done", "server", {})
            self.finalize()
            return
        if self.phase in (Phase.DOWNLINK, Phase.UPLOAD, Phase.COLLECTING):
            self._pump(now)

    def _expire_inflight(self, now: float) -> None:
        out = self._inflight
        if out is None or now < out.deadline:
            return
        self._inflight = None
        self.log("downlink_status_timeout", out.device_id, {
            "kind": out.kind, "fragment": out.index, "position": out.position,
            "attempt": out.attempts, "timeout_seconds": self.inflight_downlink_timeout_seconds})
        self._requeue(out, now)

    def _requeue(self, out: Outbound, now: float) -> None:
        g = self._g
        if out.kind == "global":
            # Only if it still belongs to the global being delivered.
            if g is not None and not g.ended and g.header_id == out.header_id:
                g.pending.appendleft(out.index)
                g.armed = True
                g.next_send_at = now + self.retry_guard_seconds
            return
        if out.attempts >= self.max_downlink_attempts:
            self.log("update_repair_dropped", out.device_id, {
                "position": out.position, "attempts": out.attempts})
            return
        self._outbox.appendleft(out)

    def _pump(self, now: float) -> None:
        # One downlink in flight across the whole gateway: it is half duplex.
        if self._inflight is not None:
            return
        g = self._g
        if self._outbox:
            out = self._outbox.popleft()
            out.attempts += 1
            out.deadline = now + self.inflight_downlink_timeout_seconds
            self._inflight = out
            try:
                self.publish(out.device_id, out.frame, out.correlation)
            except Exception:
                self._inflight = None
                out.attempts -= 1
                self._outbox.appendleft(out)
                raise
            self.log("update_repair_queued", out.device_id, {
                "position": out.position, "application_bytes": len(out.frame),
                "attempt": out.attempts})
            return
        if g is None or not g.armed or not g.pending or now < g.next_send_at:
            return
        index = g.pending.popleft()
        g.attempts[index] += 1
        if g.attempts[index] > self.max_downlink_attempts:
            self._fail(f"{self.multicast_device_id} fragment {index} exceeded the downlink retry limit")
            return
        correlation = f"v3-L{g.level_index}-g{g.round}-f{index}-a{g.attempts[index]}"
        out = Outbound(
            kind="global", device_id=self.multicast_device_id, frame=g.frames[index],
            correlation=correlation, index=index, attempts=g.attempts[index],
            header_id=g.header_id,
            deadline=now + self.inflight_downlink_timeout_seconds)
        # Claim the slot BEFORE publishing: down/sent can be delivered on the
        # network thread while publish() is still on the stack.
        self._inflight = out
        try:
            self.publish(out.device_id, out.frame, correlation)
        except Exception:
            self._inflight = None
            g.pending.appendleft(index)
            g.attempts[index] -= 1
            g.next_send_at = now + self.retry_guard_seconds
            raise
        g.fragments_queued += 1
        self.log("global_fragment_queued", out.device_id, {
            "level": self.level.level, "round": g.round, "fragment": index,
            "count": GLOBAL_FRAGMENTS, "application_bytes": len(out.frame),
            "attempt": g.attempts[index], "correlation_id": correlation})

    @_synchronized
    def handle_downlink_sent(self, device_id: str) -> None:
        out = self._inflight
        if out is None or out.device_id != device_id:
            self.log("downlink_sent_unmatched", device_id, {})
            return
        self._inflight = None
        now = self.clock()
        if out.kind == "repair":
            board = self._board_for(device_id)
            if board is not None:
                board.counters["update_repair_downlinks_sent"] += 1
                board.counters["downlink_application_bytes_sent"] += len(out.frame)
            if self._u is not None:
                self._u.repair_downlinks_sent += 1
            self.log("update_repair_sent", device_id,
                     {"position": out.position, "application_bytes": len(out.frame)})
            return
        g = self._g
        self.counters["global_fragments_sent"] += 1
        self.counters["global_application_bytes_sent"] += len(out.frame)
        if g is None or g.ended or g.header_id != out.header_id:
            self.log("global_fragment_sent_for_finished_global", device_id,
                     {"fragment": out.index, "round_id": out.header_id})
            return
        g.fragments_sent += 1
        g.next_send_at = now
        self.log("global_fragment_sent", device_id, {"fragment": out.index, "round": g.round})
        if not g.pending:
            g.burst_ended_at = now
            if self.phase is Phase.DOWNLINK:
                g.armed = False    # wait for reports before deciding on repairs

    @_synchronized
    def handle_downlink_failed(self, device_id: str, reason: str) -> None:
        out = self._inflight
        if out is None or out.device_id != device_id:
            self.log("downlink_failed_unmatched", device_id, {"reason": reason})
            return
        self._inflight = None
        now = self.clock()
        name, explanation = ExperimentController._explain_downlink_failure(reason)
        self.log("downlink_failed", device_id, {
            "kind": out.kind, "fragment": out.index, "position": out.position,
            "error": name or None, "explanation": explanation, "reason": reason})
        if explanation is not None:
            self._fail(name + ": " + explanation)
            return
        self._requeue(out, now)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def _write_masks(self) -> None:
        levels = {}
        for level in self.level_history:
            rounds_done = len(level.up_mask)
            levels[str(level.level)] = {
                "level_index": level.level_index,
                "n_machines": len(level.positions),
                "positions": list(level.positions),
                "machines": (
                    [self.machine_order[p] for p in level.positions]
                    if self.machine_order else None
                ),
                "rounds_completed": rounds_done,
                "complete": level.status == "complete",
                "uplink": [list(row) for row in level.up_mask],
                "downlink": [list(row) for row in level.down_mask],
                # Slot / round (g) of the GLOBAL each delivered UPDATE was trained from;
                # null where the UPDATE did not arrive. base_round[r][i] < r marks a
                # stale downlink the twin must replay (dropout.json stale-copy semantics).
                "base_slot": [list(row) for row in level.base_rows],
                "base_round": [
                    [None if s is None else s - level.level_index * ROUND_SLOTS for s in row]
                    for row in level.base_rows
                ],
            }
        self.out.write_replay_masks({
            "schema": "tinyfl-replay-masks-v2",
            "run_id": self.run_id,
            "semantics": (
                "uplink[r][i]: the UPDATE of level position i for training round r reached the "
                "server before the round closed. downlink[r][i]: the board holding position i "
                "had the complete GLOBAL produced by round r (GLOBAL round r+1) when its "
                "delivery phase ended. Same convention as fl_algorithm_comparison_20260929/"
                "dropout.json; the initial GLOBAL(level, 0) is required to be delivered to "
                "every participating board (the level fails otherwise)."
            ),
            "seed": self.seed,
            "rounds_per_level": ROUNDS_PER_LEVEL,
            "started_utc": self._started_utc,
            "levels": levels,
        })

    def _level_summary(self, level: LevelState) -> dict:
        rounds = level.rounds
        return {
            "level": level.level,
            "level_index": level.level_index,
            "status": level.status,
            "n_machines": len(level.positions),
            "machines": ([self.machine_order[p] for p in level.positions]
                         if self.machine_order else None),
            "boards": list(level.boards),
            "rounds_completed": len(rounds),
            "updates_expected_per_round": len(level.positions),
            "updates_delivered_per_round": [len(r["update_arrived_positions"]) for r in rounds],
            "updates_missing_total": sum(len(r["update_missing_positions"]) for r in rounds),
            "rounds_closed_by_timeout": sum(1 for r in rounds if r["close_reason"] == "timeout"),
            "update_fragments_received": sum(r["update_fragments_received"] for r in rounds),
            "update_repair_requests": sum(r["update_repair_requests"] for r in rounds),
            "update_repair_downlinks_sent": sum(r["update_repair_downlinks_sent"] for r in rounds),
            "global_fragments_sent": sum(r["global_fragments_sent"] for r in rounds),
            "global_bursts": sum(r["global_bursts"] for r in rounds),
            "downlink_holes": sum(
                len(r["participating_boards"]) - len(r["boards_holding_result_global"])
                for r in rounds),
            "initial_global_crc32": level.initial_global_crc,
            "final_global_crc32": level.final_crc,
            "final_global_codes_sha256": level.final_codes_sha256,
            "duration_s": (None if level.ended_t is None
                           else round(level.ended_t - level.started_t, 3)),
            "started_utc": level.started_utc,
            "ended_utc": level.ended_utc,
            "alarms": int(self.alarm_count.get(level.level_index, 0)),
        }

    @_synchronized
    def finalize(self) -> None:
        """Rewrite the summary files (called per level, on failure, at the end)."""
        self._write_masks()
        summaries = [self._level_summary(level) for level in self.level_history]
        self.out.write_level_summaries(summaries)
        self.out.write_replay_summary(self.replay_summary())
        self._write_boards()

    @_synchronized
    def summary(self) -> dict:
        return {
            "protocol": "v3",
            "seed": self.seed,
            "levels": self.levels,
            "phase": self.phase.value,
            "finished": self.finished,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "boards": {
                str(n): {
                    "device_id": b.device_id, "counted": b.counted,
                    "hello_count": b.hello_count, **dict(sorted(b.counters.items())),
                }
                for n, b in sorted(self.boards.items())
            },
            "levels_summary": [self._level_summary(lv) for lv in self.level_history],
            "counters": dict(sorted(self.counters.items())),
            "alarms": int(sum(self.alarm_count.values())),
            "self_checks": self.self_check_count,
            "run_id": self.run_id,
            "replay": self.replay_summary()["counts"],
            "multicast_group": self.multicast_device_id,
        }
