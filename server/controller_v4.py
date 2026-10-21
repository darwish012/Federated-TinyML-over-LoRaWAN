"""Protocol-v4 controller: federated learning over LoRaWAN for any number of boards.

Pure state machine, independent of MQTT (like controller_v3.py, which stays as is).
The runner feeds it uplinks and downlink status events and calls tick(); everything
it wants to transmit goes through ``publish``.

One phase (rounds 1..R, one model fl.init(seed)):

    GLOBAL(0)   multicast, repaired by union until every gating board holds it
    round 1     boards train and send one 3-fragment UPDATE per hosted position;
                the round closes when every participant position delivered, was
                abandoned after max_repair_rounds, or at round_timeout_seconds
                -> twin_server.Server.round() over the delivered deltas
    GLOBAL(1)   ...
    GLOBAL(R)   final model of the phase; the next phase starts from fl.init(seed)

Membership (membership_v4.py): boards join at round boundaries (a BOUND board that
acknowledges a GLOBAL becomes ACTIVE), and leave by silence (SUSPECT, DEPARTED).
Losses and absences are recorded, not fatal. After the last phase the server
repeats RUN_CONTROL REPLAY until every expected board reported REPLAY_DONE (or the
collect window ends) and then sends RUN_CONTROL END.

Crash recovery: after every round (and at every phase boundary) the complete run
state is written atomically (``write_checkpoint``); ``resume`` rebuilds a
controller that continues bit-identically.
"""

from __future__ import annotations

import base64
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
from membership_v4 import Member, Membership, State
from plan_v4 import PhaseSpec, RunPlan, positions_without_board
from protocol_v4 import (
    ALL_GLOBAL_FRAGMENTS,
    CMD_ADOPT,
    CMD_END,
    CMD_REPLAY,
    F_PORT,
    GLOBAL_BYTES,
    GLOBAL_FRAGMENTS,
    HELLO_RUN_COMPLETE,
    LAYERS,
    MAX_PHASES_PER_RUN_CONTROL,
    MAX_SLOT,
    MSG_ALARM,
    MSG_BOARD_HELLO,
    MSG_CLASS_C_READY,
    MSG_GLOBAL_STATUS,
    MSG_REPLAY_DONE,
    MSG_SELF_CHECK,
    MSG_UPDATE_FRAGMENT,
    NO_ALARM_INDEX,
    NO_SLOT,
    SHAPES,
    UPDATE_FRAGMENTS,
    UPLINK_TYPES,
    ConflictingFragment,
    Frame,
    ProtocolError,
    QuantisedModel,
    decode_alarm,
    decode_board_hello,
    decode_class_c_ready,
    decode_frame,
    decode_global,
    decode_global_status,
    decode_replay_done,
    decode_self_check,
    decode_update,
    dev_eui_of_device_id,
    encode_global,
    encode_run_control,
    encode_update_repair,
    fragment_global,
    new_update_partial,
    quantize_model,
    split_update_fragment,
    unflatten,
    validate_crc,
)
from run_outputs_v4 import MemoryOutputs
from study_plan import load_fl_modules

EventLogger = Callable[[str, str, dict], None]
Publisher = Callable[[str, bytes, str], None]

CHECKPOINT_VERSION = 5          # 5: rev 4.3 (624 B GLOBALs with the participation mask, ENDING stage)


class Stage(str, Enum):
    WAITING_START = "waiting_start"
    DOWNLINK = "downlink"
    UPLOAD = "upload"
    IDLE = "idle"
    COLLECTING = "collecting"
    ENDING = "ending"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ControllerParams:
    """Every tunable of the controller (config_v4.py builds this from [v4] / [experiment])."""

    min_boards: int = 1
    start_after_seconds: float = 0.0
    require_all_positions: bool = False
    max_machines_per_board: int = 8
    class_c_guard_seconds: float = 15.0
    retry_guard_seconds: float = 8.0
    max_downlink_attempts: int = 5
    max_repair_rounds: int = 4
    max_global_repair_rounds: int = 4
    multicast_report_window_seconds: float = 180.0
    inflight_downlink_timeout_seconds: float = 45.0
    ready_wait_seconds: float = 240.0
    round_timeout_seconds: float = 900.0
    idle_phase_timeout_seconds: float = 3600.0
    idle_retry_seconds: float = 120.0
    suspect_after_rounds: int = 2
    depart_after_rounds: int = 4
    post_run_collect_seconds: float = 1800.0
    replay_repeat_seconds: float = 300.0
    end_repeats: int = 2                    # END copies per send (a send is one burst of this many frames)
    downlink_guard_seconds: float = 4.0     # rev 4.3: no multicast within this long after an uplink
    end_repeat_seconds: float = 30.0        # at most one END re-send per this long
    end_quiet_seconds: float = 420.0        # DONE after this long with no uplink since the last END
    end_linger_seconds: float = 1800.0      # ... or at the latest this long after ENDING began
    adopt_period_seconds: float = 120.0     # ADOPT every this long in WAITING_START; min spacing for unbound-board ADOPTs
    prompt_repair: bool = True              # not a config key: false only to measure the rev 4.2 behaviour
    prompt_repair_min_gap_seconds: float = 30.0   # at most one prompt repair per board per this long
    resume_repair_delay_seconds: float = 20.0
    liveness_seconds: float = 600.0


@dataclass
class AcceptedUpdate:
    position: int
    deq: tuple                  # dequantised delta (current-global relative), float64, tensor shapes
    n_train_rows: int
    train_loss: float
    scales: tuple
    arrived_t: float
    base_slot: int = -1
    stale: bool = False


@dataclass
class UploadState:
    round: int                  # 1-based: the training round this state collects
    seed_slot: int              # slot of the GLOBAL that opened the round
    started_t: float
    started_utc: str
    participants: tuple = ()    # DevEUIs waited for (fixed when delivery ends)
    watch: tuple = ()           # DevEUIs whose misses are counted at close
    expected: set = field(default_factory=set)        # positions waited for
    updates: dict = field(default_factory=dict)
    partials: dict = field(default_factory=dict)
    repair_rounds: Counter = field(default_factory=Counter)
    abandoned: set = field(default_factory=set)
    fragments_rx: Counter = field(default_factory=Counter)
    fragments_total: int = 0
    repair_downlinks_sent: int = 0
    began: bool = False          # delivery ended and the round timeout is running
    proactive_repair_at: float | None = None
    closed: bool = False


@dataclass
class GlobalState:
    phase_index: int
    round: int
    slot: int
    model: QuantisedModel
    payload: bytes
    crc: int
    frames: list
    required: set = field(default_factory=set)          # DevEUIs that gate and must hold it
    pending: deque = field(default_factory=deque)
    attempts: Counter = field(default_factory=Counter)
    board_missing: dict = field(default_factory=dict)   # DevEUI -> missing mask
    reported: set = field(default_factory=set)
    late_repairs: Counter = field(default_factory=Counter)
    armed: bool = False
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
    resumed: bool = False
    repair_after: float | None = None               # a prompt repair burst is due at this time
    prompt_repairs: int = 0
    prompt_burst: bool = False                      # the burst on the air is a prompt repair (it never moves burst_ended_at)
    last_prompt: dict = field(default_factory=dict)  # DevEUI -> t of the last prompt repair it caused
    dropped: set = field(default_factory=set)       # required boards dropped for silence (may re-enter)
    not_gated: set = field(default_factory=set)     # DevEUIs not waited for because they were silent


@dataclass
class Outbound:
    kind: str                   # "global" | "repair" | "control"
    device_id: str
    frame: bytes
    correlation: str
    index: int = 0
    position: int = -1
    slot: int = -1
    attempts: int = 0
    deadline: float = 0.0


@dataclass
class PhaseState:
    spec: PhaseSpec
    server: object | None
    started_t: float
    started_utc: str
    first_slot: int = 0
    globals_deq: dict = field(default_factory=dict)        # slot -> dequantised GLOBAL (562 f64)
    globals_payload: dict = field(default_factory=dict)    # slot -> GLOBAL bytes (checkpoint)
    slot_round: dict = field(default_factory=dict)         # slot -> GLOBAL round
    rounds: list = field(default_factory=list)             # round records
    masks: list = field(default_factory=list)              # per round {"uplink", "downlink", "base"}
    positions_seen: set = field(default_factory=set)
    held_by_slot: dict = field(default_factory=dict)       # slot -> {DevEUI that holds that GLOBAL}
    initial_global_crc: str = ""
    final_crc: str = ""
    final_codes_sha256: str = ""
    boards_holding_final: list = field(default_factory=list)
    status: str = "running"        # running | complete | aborted_idle
    ended_t: float | None = None
    ended_utc: str = ""

    @property
    def final_slot(self) -> int | None:
        for slot, r in self.slot_round.items():
            if r == self.spec.rounds:
                return slot
        return None

    # -- checkpoint (a finished phase needs no model state) -------------------
    def to_dict(self, with_model: bool = False) -> dict:
        d = {
            "spec": self.spec.to_dict(), "started_utc": self.started_utc,
            "first_slot": self.first_slot, "rounds": self.rounds, "masks": self.masks,
            "slot_round": {str(k): v for k, v in self.slot_round.items()},
            "held_by_slot": {str(k): sorted(v) for k, v in self.held_by_slot.items()},
            "positions_seen": sorted(self.positions_seen),
            "initial_global_crc": self.initial_global_crc, "final_crc": self.final_crc,
            "final_codes_sha256": self.final_codes_sha256,
            "boards_holding_final": list(self.boards_holding_final),
            "status": self.status, "ended_utc": self.ended_utc,
            "duration_s": None if self.ended_t is None else round(self.ended_t - self.started_t, 3),
        }
        if with_model:
            d["globals"] = {str(s): base64.b64encode(p).decode("ascii")
                            for s, p in self.globals_payload.items()}
        return d


def _enc_arrays(arrays) -> list:
    return [base64.b64encode(np.ascontiguousarray(a, dtype=np.float64).tobytes()).decode("ascii")
            for a in arrays]


def _dec_arrays(items) -> tuple:
    return tuple(
        np.frombuffer(base64.b64decode(item), dtype=np.float64).reshape(SHAPES[i]).copy()
        for i, item in enumerate(items))


class ControllerV4:
    def __init__(
        self,
        plan: RunPlan,
        publish: Publisher,
        log: EventLogger,
        multicast_device_id: str,
        params: ControllerParams | None = None,
        outputs=None,
        is_rostered: Callable[[str], bool] | None = None,
        fl_modules=None,
        research_root=None,
        run_id: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], str] | None = None,
        publish_replace: Callable[[str, bytes | None, str], None] | None = None,
    ) -> None:
        if not multicast_device_id:
            raise ValueError("v4 delivers GLOBAL by multicast only; set experiment.multicast_device_id")
        self.plan = plan
        self.params = params or ControllerParams()
        self.publish = publish
        # Rev 4.3 (section 4, unicast queue lifecycle): UPDATE_REPAIR replaces the device's pending
        # downlink; ``publish_replace(device_id, None, "")`` clears it. Without one, repairs fall back
        # to ``publish`` and nothing is cleared.
        self.publish_replace = publish_replace
        self._repair_pending: dict = {}            # device_id -> correlation of a repair the NS may still hold
        self.log = log
        self.multicast_device_id = multicast_device_id
        self.out = outputs if outputs is not None else MemoryOutputs()
        self.clock = clock
        self.wall_clock = wall_clock or (lambda: datetime.now(timezone.utc).isoformat())
        self.run_id = int(run_id) if run_id is not None else secrets.randbelow(0xFFFFFFFF) + 1
        if not 1 <= self.run_id <= 0xFFFFFFFF:
            raise ValueError("run_id must be a non-zero u32")
        self._lock = threading.RLock()

        # twin_server is imported UNCHANGED from the research folder.
        if fl_modules is None:
            fl_modules = load_fl_modules(research_root)
        self.twin_server, self.fl = fl_modules

        self.membership = Membership(is_rostered, self._on_membership, clock)
        self.stage = Stage.WAITING_START
        self.failure_reason = ""
        self._t0 = clock()
        self.slot_counter = 0
        self.phases: list[PhaseState] = []         # every phase begun (finished + current)
        self.phase: PhaseState | None = None
        self._g: GlobalState | None = None
        self._u: UploadState | None = None
        self._pending_round: dict | None = None
        self._round_changes: list[dict] = []       # membership transitions since the last round close
        self._outbox: deque[Outbound] = deque()
        self._inflight: Outbound | None = None
        self._alarms: dict = {}                    # (position, phase_index) -> {"index", "source"}
        self._alarm_phases: dict = {}              # position -> {phase_index}
        self.replay_done: dict = {}                # (position, phase_index) -> record
        self.counters: Counter = Counter()
        self.alarm_count: Counter = Counter()      # phase_index -> alarms
        self.self_check_count = 0
        self.started_utc = ""
        self._idle_since: float | None = None
        self._idle_retry_at = 0.0
        self._ignored_notified: set = set()
        self._adopt_waiting: dict = {}             # DevEUI -> t of a HELLO that reported another run
        self._adopt_last_t: float | None = None    # when an ADOPT was last queued (any reason)
        self._adopt_unbound: dict = {}             # DevEUI -> t of an uplink from a rostered, unbound board
        self._adopt_unbound_sent_t: float | None = None   # when the last unbound-board ADOPT was queued
        self._not_gated_notified: set = set()      # (DevEUI, slot) already logged as not gated
        # COLLECTING
        self._collect_deadline = 0.0
        self._replay_phases: list[int] = []
        self._next_replay_at = 0.0
        self._last_uplink_t = float("-inf")        # last valid uplink of any board of the run
        self._ending_why = ""
        self._ending_started = 0.0
        self._last_end_sent_t: float | None = None
        self._last_end_queued_t: float | None = None
        self.end_unconfirmed: list = []
        self._collect_dirty = False
        self._last_collect_checkpoint = 0.0
        self._boards_dirty = False
        self._last_boards_write = float("-inf")
        self._last_repair_state: dict = {}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    @property
    @_synchronized
    def finished(self) -> bool:
        return self.stage is Stage.DONE

    @property
    @_synchronized
    def failed(self) -> bool:
        return self.stage is Stage.FAILED

    def _rel(self, now: float) -> float:
        return round(now - self._t0, 3)

    def _fail(self, reason: str) -> None:
        if self.stage is Stage.FAILED:
            return
        self.failure_reason = reason
        self.stage = Stage.FAILED
        self.log("experiment_failed", "server", {"reason": reason})
        if self.phase is not None and self.phase.status == "running":
            self.phase.status = "failed"
        self._outbox.clear()
        self._inflight = None
        self.finalize()
        self._checkpoint()

    def fail(self, reason: str) -> None:
        """External stop (operator, overall timeout)."""
        with self._lock:
            self._fail(reason)

    @_synchronized
    def status(self) -> dict:
        counts = self.membership.counts()
        phase = self.phase
        u = self._u
        expected = len(u.expected) if u is not None else 0
        arrived = len(u.updates) if u is not None else 0
        return {
            "stage": self.stage.value, "run_id": self.run_id, "mode": self.plan.mode,
            "phase": None if phase is None else phase.spec.index,
            "phases": len(self.plan.phases),
            "phase_name": None if phase is None else phase.spec.name,
            "round": None if (self._g is None) else self._g.round if u is None else u.round,
            "rounds": None if phase is None else phase.spec.rounds,
            "slot": self.slot_counter,
            "rostered": counts["ROSTERED"], "bound": counts["BOUND"],
            "active": counts["ACTIVE"], "suspect": counts["SUSPECT"],
            "departed": counts["DEPARTED"],
            "arrived_positions": arrived, "expected_positions": expected,
            "failure_reason": self.failure_reason,
        }

    def status_line(self) -> str:
        s = self.status()
        where = "-" if s["phase"] is None else (
            f"phase {s['phase'] + 1}/{s['phases']} round {s['round']}/{s['rounds']}")
        return (f"[{s['stage']}] {where} boards active={s['active']} suspect={s['suspect']} "
                f"departed={s['departed']} bound={s['bound']} "
                f"arrived={s['arrived_positions']}/{s['expected_positions']}")

    # ------------------------------------------------------------------
    # Membership plumbing
    # ------------------------------------------------------------------
    def _on_membership(self, event: dict) -> None:
        phase = None if self.phase is None else self.phase.spec.index
        record = {
            "timestamp_utc": self.wall_clock(), "phase_index": phase,
            "stage": self.stage.value, "slot": self.slot_counter, **event,
        }
        if self._u is not None:
            record["round"] = self._u.round
        self._round_changes.append(record)
        self.out.membership_event(record)
        self.log("membership", event["dev_eui"], {k: v for k, v in record.items()
                                                  if k not in ("timestamp_utc", "dev_eui")})

    @_synchronized
    def sync_roster(self, dev_euis, device_id_of: Callable[[str], str]) -> None:
        """Create ROSTERED members for every rostered DevEUI not seen yet."""
        for dev_eui in dev_euis:
            if dev_eui not in self.membership.members:
                self.membership.ensure_rostered(dev_eui, device_id_of(dev_eui))

    def _hosting(self, member: Member, spec: PhaseSpec) -> tuple:
        return tuple(p for p in member.positions if spec.covers(p))

    def _phase_hosting(self, member: Member) -> tuple:
        return () if self.phase is None else self._hosting(member, self.phase.spec)

    def _admit(self, member: Member, why: str) -> None:
        """BOUND / SUSPECT -> ACTIVE (the board acknowledged, or delivered for, a GLOBAL)."""
        if member.state is State.BOUND:
            member.misses = 0
            self.membership.set_state(member, State.ACTIVE, why)
        elif member.state is State.SUSPECT:
            self.membership.set_state(member, State.ACTIVE, why)

    def _previous_participants(self) -> set:
        """Positions that took part in the round before the one being collected."""
        if self._pending_round is not None:
            return set(self._pending_round["record"]["participants"])
        if self.phase is not None and self.phase.rounds:
            return set(self.phase.rounds[-1]["participants"])
        return set()

    def _phase_of_slot(self, slot: int) -> PhaseState | None:
        for phase in reversed(self.phases):
            if slot in phase.slot_round:
                return phase
        return None

    def _note_held(self, member: Member, slot: int) -> None:
        """This board holds the GLOBAL of ``slot``: it acknowledged it, an UPDATE or READY proves
        it, it trained from it (base_slot), or it replayed with it. The downlink mask the twin
        reads is built from this, so evidence that arrives late still counts."""
        phase = self._phase_of_slot(slot)
        if phase is None:
            return
        phase.held_by_slot.setdefault(slot, set()).add(member.dev_eui)
        if slot == phase.final_slot:
            member.held_final_phases.add(phase.spec.index)

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
                    f"message type 0x{frame.message_type:02X} is a downlink; it cannot be an uplink")
            dev_eui = dev_eui_of_device_id(device_id)
        except ProtocolError as exc:
            self._reject(device_id, exc)
            return
        try:
            if frame.message_type == MSG_BOARD_HELLO:
                self._handle_hello(device_id, dev_eui, frame, len(payload))
                return
            member = self.membership.get(dev_eui)
            if member is None or member.board_id is None:
                self.counters["ignored_device_without_hello"] += 1
                self._note_unbound_uplink(dev_eui)
                if dev_eui not in self._ignored_notified:
                    self._ignored_notified.add(dev_eui)
                    self.log("ignored_device_without_hello", device_id, {
                        "message_type": f"0x{frame.message_type:02X}",
                        "hint": "a board is only known after a valid BOARD_HELLO from a rostered DevEUI"})
                return
            self._heard(member, len(payload))
            if frame.message_type == MSG_UPDATE_FRAGMENT:
                self._handle_update_fragment(member, frame)
            elif frame.message_type == MSG_CLASS_C_READY:
                self._handle_ready(member, frame)
            elif frame.message_type == MSG_GLOBAL_STATUS:
                self._handle_status(member, frame)
            elif frame.message_type == MSG_ALARM:
                self._handle_alarm(member, frame)
            elif frame.message_type == MSG_SELF_CHECK:
                self._handle_self_check(member, frame)
            elif frame.message_type == MSG_REPLAY_DONE:
                self._handle_replay_done(member, frame)
        except ProtocolError as exc:
            self._reject(device_id, exc)

    def _reject(self, device_id: str, exc: Exception) -> None:
        self.counters["frames_rejected"] += 1
        self.log("frame_rejected", device_id, {"error": str(exc)})

    def _heard(self, member: Member, nbytes: int) -> None:
        member.last_heard_t = self.clock()
        self._touch_uplink()
        member.counters["uplinks"] += 1
        member.counters["uplink_application_bytes"] += nbytes
        if member.state is State.DEPARTED:
            member.misses = 0
            member.ready = False
            self.membership.set_state(member, State.BOUND, "spoke_again")
        self._regate(member)

    # ------------------------------------------------------------------
    # BOARD_HELLO and run start
    # ------------------------------------------------------------------
    def _is_repeated_hello(self, dev_eui: str, hello) -> bool:
        """Rev 4.3 (3.3): HELLO is unacknowledged, so a board repeats it until it hears its run.
        A HELLO identical to the bound one (same DevEUI, board_id, positions, stored run, flags and
        firmware) from a board that is not DEPARTED changes nothing but liveness."""
        m = self.membership.members.get(dev_eui)
        return (m is not None and m.board_id is not None and m.state is not State.DEPARTED
                and m.board_id == hello.board_id and m.positions == tuple(hello.positions)
                and m.flash_run_id == hello.stored_run_id and m.flags == hello.flags
                and m.fw_version == hello.fw_version)

    def _is_adopt_ack(self, dev_eui: str, hello) -> bool:
        """Rev 4.3: a board answers every ADOPT with a HELLO. A bound board that adopted this run
        (its only change is stored_run_id, now the server's run, and the run-complete flag) is
        acknowledging the ADOPT, not rebooting."""
        m = self.membership.members.get(dev_eui)
        keep = ~HELLO_RUN_COMPLETE
        recent = (self._adopt_last_t is not None
                  and self.clock() - self._adopt_last_t <= self.params.adopt_period_seconds)
        return (recent and m is not None and m.board_id is not None and m.state is not State.DEPARTED
                and m.board_id == hello.board_id and m.positions == tuple(hello.positions)
                and m.fw_version == hello.fw_version and (m.flags & keep) == (hello.flags & keep)
                and hello.stored_run_id == self.run_id and m.flash_run_id != self.run_id)

    def _note_repeated_hello(self, member: Member, nbytes: int, hello=None) -> None:
        member.last_heard_t = self.clock()                 # liveness only: never resets state
        self._touch_uplink()
        member.counters["uplinks"] += 1
        member.counters["uplink_application_bytes"] += nbytes
        member.counters["hello_repeats"] += 1
        self.counters["hello_repeats"] += 1
        kind = "repeat"
        if hello is not None and member.flash_run_id != hello.stored_run_id:
            member.flash_run_id = hello.stored_run_id      # it adopted our run (ADOPT acknowledged)
            member.flags = hello.flags
            self._boards_dirty = True
            self.counters["hello_adopt_acks"] += 1
            kind = "adopt_ack"
        self.log("hello_noop", member.device_id, {
            "level": "debug", "board_id": member.board_id, "kind": kind,
            "state": member.state.value, "stored_run_id": member.flash_run_id})
        self._regate(member)

    def _handle_hello(self, device_id: str, dev_eui: str, frame: Frame, nbytes: int) -> None:
        hello = decode_board_hello(frame)
        if len(hello.positions) > self.params.max_machines_per_board:
            raise ProtocolError(
                f"BOARD_HELLO lists {len(hello.positions)} positions; "
                f"max_machines_per_board is {self.params.max_machines_per_board}")
        self._adopt_unbound.pop(dev_eui, None)
        if self._is_repeated_hello(dev_eui, hello) or self._is_adopt_ack(dev_eui, hello):
            return self._note_repeated_hello(self.membership.members[dev_eui], nbytes, hello)
        member, rejection = self.membership.bind(dev_eui, device_id, hello)
        if rejection is not None:
            self.counters[f"hello_rejected_{rejection['code']}"] += 1
            self.log("hello_rejected", device_id, {"board_id": hello.board_id, **rejection})
            return
        member.ready = False                 # a (re)booted board has lost its Class C state
        self._touch_uplink()
        self._regate(member)
        member.counters["uplinks"] += 1
        member.counters["uplink_application_bytes"] += nbytes
        self.log("board_hello", device_id, {
            "board_id": hello.board_id, "positions": list(hello.positions),
            "fw_version": hello.fw_version, "flags": f"0x{hello.flags:02X}",
            "sd_mounted": hello.sd_mounted, "self_check_passed": hello.self_check_passed,
            "imu_source": hello.imu_source, "run_complete": hello.run_complete,
            "hello_count": member.hello_count, "stored_run_id": hello.stored_run_id,
            "server_run_id": self.run_id, "state": member.state.value})
        if hello.stored_run_id not in (0, self.run_id):
            self.log("board_holds_other_run", device_id, {
                "board_id": hello.board_id, "stored_run_id": hello.stored_run_id,
                "server_run_id": self.run_id,
                "hint": "RUN_CONTROL ADOPT is multicast so the board archives its old run and adopts this one"})
            if self.stage is not Stage.WAITING_START:
                self._adopt_waiting[member.dev_eui] = self.clock()
        if not member.counted:
            self.log("board_not_counted", device_id, {
                "board_id": hello.board_id, "flags": f"0x{hello.flags:02X}",
                "reason": "needs flags bit0 (SD mounted) and bit1 (self-check passed), or bit2 (IMU source)"})
        if member.hello_count > 1 and self.stage not in (Stage.WAITING_START, Stage.DONE):
            self.log("board_rebooted_mid_run", device_id, {
                "board_id": hello.board_id, "stage": self.stage.value, "state": member.state.value})
        self._boards_dirty = True                     # boards.json is rewritten at most every 2 s
        if self.stage is Stage.WAITING_START:
            self._maybe_start(self.clock())

    def counted_boards(self) -> list[Member]:
        return [m for m in self.membership.members.values()
                if m.state is State.BOUND and m.counted and m.positions]

    def _maybe_start(self, now: float) -> None:
        if self.stage is not Stage.WAITING_START:
            return
        counted = self.counted_boards()
        deadline = self.params.start_after_seconds
        timed_out = deadline >= 1 and now - self._t0 >= deadline
        if len(counted) < self.params.min_boards and not (timed_out and counted):
            return
        hosted = {p for m in counted for p in m.positions}
        missing = positions_without_board(self.plan, hosted)
        if self.params.require_all_positions and missing:
            if timed_out:
                self._fail("start_after_seconds elapsed but planned positions have no board: "
                           + ", ".join(f"phase {i}: {v[:8]}{'...' if len(v) > 8 else ''}"
                                       for i, v in sorted(missing.items())))
            return
        self.started_utc = self.wall_clock()
        self.log("experiment_started", "server", {
            "boards": sorted(m.board_id for m in counted), "mode": self.plan.mode,
            "phases": [p.to_dict() for p in self.plan.phases] if self.plan.mode == "planned" else len(self.plan.phases),
            "seed": self.plan.seed, "run_id": self.run_id,
            "started_by": "start_after_seconds" if len(counted) < self.params.min_boards else "min_boards"})
        for index, positions in sorted(missing.items()):
            self.log("planned_positions_without_board", "server", {
                "phase_index": index, "n_missing": len(positions), "positions": positions,
                "note": "logged as never participating; the run does not wait for them"})
        self._begin_phase(0, now)

    def _write_boards(self) -> None:
        self._boards_dirty = False
        self._last_boards_write = self.clock()
        self.out.write_boards({
            m.dev_eui: {
                "device_id": m.device_id, "board_id": m.board_id, "state": m.state.value,
                "positions": list(m.positions), "fw_version": m.fw_version,
                "flags": m.flags, "counted": m.counted, "hello_count": m.hello_count,
                "flash_run_id": m.flash_run_id, "misses": m.misses,
                "held_final_phases": sorted(m.held_final_phases),
            }
            for m in sorted(self.membership.members.values(), key=lambda x: x.dev_eui)
        })

    # ------------------------------------------------------------------
    # Phases and GLOBAL models
    # ------------------------------------------------------------------
    def _next_slot(self) -> int:
        self.slot_counter += 1
        if self.slot_counter > MAX_SLOT:
            self._fail(f"the run used all {MAX_SLOT} slots (65 535 per run)")
        return self.slot_counter

    def _begin_phase(self, index: int, now: float) -> None:
        spec = self.plan.phase(index)
        server = self.twin_server.Server(self.fl.init(list(LAYERS), self.plan.seed))
        self.phase = PhaseState(spec=spec, server=server, started_t=now,
                                started_utc=self.wall_clock())
        self.phases.append(self.phase)
        self._idle_since = None
        self._pending_round = None
        self.log("phase_started", "server", {
            "phase_index": index, "name": spec.name, "rounds": spec.rounds,
            "positions": None if spec.positions is None else len(spec.positions)})
        g = self._make_global(0)
        if g is None:
            return
        self.phase.first_slot = g.slot
        self._install_global(g, now)

    def _make_global(self, round_index: int) -> GlobalState | None:
        phase = self.phase
        spec = phase.spec
        slot = self._next_slot()
        if self.stage is Stage.FAILED:
            return None
        # Same path as twin.downlink_message: float32-cast server weights, then Q_DYN_TENSOR.
        params32 = tuple(np.asarray(p, dtype=np.float32) for p in phase.server.p)
        model = quantize_model(params32)
        payload = encode_global(self.run_id, spec.index, round_index, spec.rounds, model,
                                spec.participation())
        crc = validate_crc(payload, GLOBAL_BYTES)
        # Every dequantised GLOBAL of the phase is kept: a board may train from an older
        # one (base_slot) and the server must reconstruct its model.
        phase.globals_deq[slot] = decode_global(payload).model.dequantize()
        phase.globals_payload[slot] = payload
        phase.slot_round[slot] = round_index
        frames = fragment_global(payload, slot)
        if len(frames) != GLOBAL_FRAGMENTS:
            raise AssertionError("GLOBAL did not produce 3 fragments")
        return GlobalState(phase_index=spec.index, round=round_index, slot=slot, model=model,
                           payload=payload, crc=crc, frames=frames)

    def _install_global(self, g: GlobalState, now: float, resumed: bool = False) -> None:
        """Checkpoint first (the state now includes g), then log it and start delivering."""
        self._g = g
        self._checkpoint()
        self._log_global(g)
        self._begin_downlink(g, now, resumed=resumed)

    def _log_global(self, g: GlobalState) -> None:
        spec = self.plan.phase(g.phase_index)
        self.out.global_model({
            "phase_index": g.phase_index, "round": g.round, "slot": g.slot,
            "rounds_in_phase": spec.rounds, "final": g.round == spec.rounds,
            "run_id": self.run_id, "crc32": f"{g.crc:08X}",
            "scales_f32": list(g.model.wire_scales()),
            "codes": [int(c) for c in g.model.codes], "codes_sha256": g.model.codes_sha256(),
            # rev 4.3: the participation multicast with this GLOBAL (identical in a phase)
            "participation": spec.participation().to_dict(),
            "global_bytes": len(g.payload),
        })

    # ---- gating -------------------------------------------------------
    def _is_live(self, member: Member, now: float | None = None) -> bool:
        """LIVE = any valid uplink within ``liveness_seconds``. A silent board is never waited for."""
        now = self.clock() if now is None else now
        return now - member.last_heard_t <= self.params.liveness_seconds

    def _would_gate(self, m: Member, g: GlobalState) -> bool:
        """Would this board gate g if it were alive: ACTIVE and hosting phase positions, and at a
        phase start (round 0) every counted BOUND board as well."""
        if not self._hosting(m, self.phase.spec):
            return False
        return m.state is State.ACTIVE or (g.round == 0 and m.state is State.BOUND and m.counted)

    def _note_not_gated(self, m: Member, g: GlobalState) -> None:
        g.not_gated.add(m.dev_eui)
        key = (m.dev_eui, g.slot)
        if key not in self._not_gated_notified:
            self._not_gated_notified.add(key)
            self.log("board_not_gated_silent", m.device_id, {
                "board_id": m.board_id, "slot": g.slot, "state": m.state.value,
                "silent_seconds": round(self.clock() - m.last_heard_t, 1),
                "liveness_seconds": self.params.liveness_seconds})

    def _required_for(self, g: GlobalState) -> set:
        """DevEUIs that gate delivery of g: LIVE boards that would gate it (see _would_gate). A
        silent one is logged once per (board, slot); it can still receive the multicast, deliver
        UPDATEs and be repaired, and any uplink makes it LIVE again for the next GLOBAL."""
        required = set()
        now = self.clock()
        for m in self.membership.members.values():
            if not self._would_gate(m, g):
                continue
            if self._is_live(m, now):
                required.add(m.dev_eui)
            else:
                self._note_not_gated(m, g)
        return required

    def _ready_candidates(self) -> set:
        spec = self.phase.spec
        now = self.clock()
        return {m.dev_eui for m in self.membership.members.values()
                if m.state in (State.BOUND, State.SUSPECT, State.ACTIVE)
                and m.ready and self._hosting(m, spec) and self._is_live(m, now)}

    def _drop_silent_from_gate(self, g: GlobalState, now: float) -> None:
        """A required board that went silent during this downlink stops being waited for."""
        dropped = [d for d in g.required if not self._is_live(self.membership.members[d], now)]
        for dev in dropped:
            member = self.membership.members[dev]
            g.required.discard(dev)
            g.board_missing.pop(dev, None)
            g.reported.discard(dev)
            g.dropped.add(dev)
            g.not_gated.add(dev)
            self.log("board_dropped_from_gate", member.device_id, {
                "board_id": member.board_id, "slot": g.slot,
                "silent_seconds": round(now - member.last_heard_t, 1)})
        if dropped and g.ever_armed and not g.ended and self._union_missing(g) == 0:
            self._refresh_global(g, now)

    def _regate(self, member: Member) -> None:
        """A board dropped from this downlink's gate spoke again: wait for it once more."""
        g = self._g
        if (g is None or g.ended or self.stage is not Stage.DOWNLINK
                or member.dev_eui not in g.dropped or not self._would_gate(member, g)):
            return
        g.dropped.discard(member.dev_eui)
        g.not_gated.discard(member.dev_eui)
        g.required.add(member.dev_eui)
        g.board_missing.setdefault(member.dev_eui, ALL_GLOBAL_FRAGMENTS)
        self.log("board_regated", member.device_id, {"board_id": member.board_id, "slot": g.slot})

    def _tick_silence(self, now: float) -> None:
        """A board silent for more than 2 x liveness_seconds is SUSPECT even if fewer rounds
        have passed (the missed-round rules still apply as well)."""
        limit = 2 * self.params.liveness_seconds
        for m in self.membership.members.values():
            if (m.state is State.ACTIVE and now - m.last_heard_t > limit
                    and self._phase_hosting(m)):
                m.ready = False
                self.membership.set_state(m, State.SUSPECT, "silence",
                                          silent_seconds=round(now - m.last_heard_t, 1))

    def _silent_owners(self, positions, now: float) -> set:
        owners = set()
        for p in positions:
            if self._silent_position(p, now):
                owners.add(self.membership.position_owner[p])
        return owners

    def _silent_position(self, position: int, now: float) -> bool:
        owner = self.membership.position_owner.get(position)
        member = None if owner is None else self.membership.members.get(owner)
        return member is not None and not self._is_live(member, now)

    def _begin_downlink(self, g: GlobalState, now: float, resumed: bool = False) -> None:
        self._g = g
        phase = self.phase
        spec = phase.spec
        g.dropped, g.not_gated = set(), set()
        gate = self._required_for(g) or self._ready_candidates()
        if not gate and not resumed:
            self._enter_idle(f"no board hosting positions of phase {spec.index} is available", now)
            return
        self.stage = Stage.DOWNLINK
        g.required = set(gate)
        g.started_t = now
        g.started_utc = self.wall_clock()
        g.ready_deadline = now + self.params.ready_wait_seconds
        g.board_missing = {dev: ALL_GLOBAL_FRAGMENTS for dev in g.required}
        g.reported = set()
        g.pending = deque(range(GLOBAL_FRAGMENTS))
        g.attempts = Counter()
        g.armed = g.ever_armed = False
        g.burst_ended_at = None
        g.repair_after = None
        g.prompt_burst = False
        g.last_prompt = {}
        g.ended = False
        g.resumed = resumed
        # The round this GLOBAL opens is created NOW, not when delivery ends: a board that
        # hears the GLOBAL first trains and uploads at once, long before a slower board is repaired.
        if g.round < spec.rounds and (self._u is None or self._u.seed_slot != g.slot):
            self._u = UploadState(round=g.round + 1, seed_slot=g.slot, started_t=now,
                                  started_utc=self.wall_clock())
        elif g.round >= spec.rounds:
            self._u = None
        self.log("global_prepared", "server", {
            "phase_index": spec.index, "round": g.round, "slot": g.slot,
            "crc32": f"{g.crc:08X}", "required_boards": len(g.required), "resumed": resumed})
        if resumed:
            self._arm(g, now, "resume")
            if self._u is not None:
                self._u.proactive_repair_at = now + self.params.resume_repair_delay_seconds

    def _ready_for(self, member: Member, g: GlobalState) -> bool:
        if not member.ready:
            return False
        return member.ready_slot == NO_SLOT or member.ready_slot <= g.slot

    def _tick_downlink(self, now: float) -> None:
        g = self._g
        if g is None or g.ended:
            return
        self._drop_silent_from_gate(g, now)
        if g.ended:
            return
        if not g.armed and not g.ever_armed:
            waiting = [d for d in g.required
                       if not self._ready_for(self.membership.members[d], g)]
            # Rev 4.3: the first multicast of a round never goes out within downlink_guard_seconds of
            # any required board's last uplink: that board may still be inside its RX windows.
            quiet = now >= max((self.membership.members[d].last_heard_t for d in g.required),
                               default=float("-inf")) + self.params.downlink_guard_seconds
            if not waiting:
                latest = max((self.membership.members[d].ready_since for d in g.required),
                             default=now)
                if quiet and now >= latest + self.params.class_c_guard_seconds:
                    self._arm(g, now, "all_boards_ready")
            elif now >= g.ready_deadline and quiet:
                self.log("boards_not_ready", "server", {
                    "boards": sorted(self.membership.members[d].board_id for d in waiting),
                    "waited_seconds": self.params.ready_wait_seconds})
                self._arm(g, now, "ready_wait_expired")
            return
        self._tick_prompt_repair(g, now)
        if (not g.armed and g.burst_ended_at is not None and not g.pending
                and self._inflight_kind("global") is None
                and now - g.burst_ended_at >= self.params.multicast_report_window_seconds):
            self._refresh_global(g, now)

    def _request_prompt_repair(self, member: Member, g: GlobalState, missing: int, now: float,
                               why: str) -> bool:
        """Rev 4.3 (section 4): a required board that does not hold g said so (CLASS_C_READY awaiting
        it, or GLOBAL_STATUS with missing bits). Re-multicast the missing fragments
        ``downlink_guard_seconds`` after that uplink instead of waiting for the periodic repair timer.

        At most one prompt repair per board per ``prompt_repair_min_gap_seconds``, and prompt repairs
        count toward ``max_global_repair_rounds`` (they stop once it is used up; the periodic timer,
        which prompts never postpone, then gives up). Returns True if a prompt repair was scheduled;
        the board's report is recorded either way."""
        if (not self.params.prompt_repair or g.ended or self.stage is not Stage.DOWNLINK
                or not g.ever_armed or member.dev_eui not in g.required):
            return False
        g.board_missing[member.dev_eui] = missing
        g.reported.add(member.dev_eui)
        last = g.last_prompt.get(member.dev_eui)
        if last is not None and now - last < self.params.prompt_repair_min_gap_seconds:
            self.counters["global_prompt_repairs_throttled"] += 1
            return False
        if g.repair_rounds >= self.params.max_global_repair_rounds:
            self.counters["global_prompt_repairs_exhausted"] += 1
            return False
        g.last_prompt[member.dev_eui] = now
        g.repair_after = max(g.repair_after or 0.0, now + self.params.downlink_guard_seconds)
        self.log("global_prompt_repair_requested", member.device_id, {
            "board_id": member.board_id, "slot": g.slot, "why": why, "missing_mask": f"{missing:02X}",
            "due_in_seconds": self.params.downlink_guard_seconds})
        return True

    def _tick_prompt_repair(self, g: GlobalState, now: float) -> None:
        if g.repair_after is None or now < g.repair_after:
            return
        if g.pending or self._inflight_kind("global") is not None:
            return                                  # a burst is on the air; look again when it ends
        union = 0
        for dev in g.reported:
            if dev in g.required:
                union |= g.board_missing.get(dev, 0)
        g.repair_after = None
        if union == 0:
            return
        wanted = [i for i in range(GLOBAL_FRAGMENTS) if union & (1 << i)]
        for dev in list(g.reported):
            if g.board_missing.get(dev):
                g.reported.discard(dev)             # it must report again after this burst
        g.attempts.clear()
        g.pending = deque(wanted)
        g.armed = True
        g.bursts += 1
        g.prompt_repairs += 1
        g.repair_rounds += 1                        # counts toward max_global_repair_rounds
        g.prompt_burst = True                       # ... and never moves the periodic timer
        g.next_send_at = now
        self.counters["global_prompt_repairs"] += 1
        self.log("multicast_prompt_repair", "server", {
            "round": g.round, "slot": g.slot, "union_mask": f"{union:02X}", "fragments": wanted,
            "prompt_repair": g.prompt_repairs})

    def _arm(self, g: GlobalState, now: float, why: str) -> None:
        g.armed = True
        g.ever_armed = True
        g.next_send_at = now
        self.log("global_multicast_starting", "server", {
            "round": g.round, "slot": g.slot, "why": why, "fragments": len(g.frames)})
        if g.slot == 1 and why != "resume":
            # Run start: the periodic WAITING_START ADOPTs have normally made every board adopt this
            # run; repeat it only for a bound board that still reports another stored run (every ADOPT
            # makes all boards answer with a HELLO, which would collide with the first GLOBAL).
            stale = [m for m in self.membership.members.values()
                     if m.board_id is not None and m.state is not State.DEPARTED
                     and m.flash_run_id not in (0, self.run_id)]
            if stale:
                self._queue_adopt("run_start")

    # ------------------------------------------------------------------
    # RUN_CONTROL ADOPT (PROTOCOL_V4 3.4 / 5.1)
    # ------------------------------------------------------------------
    def _queue_adopt(self, why: str) -> None:
        """Multicast ADOPT once; requests that arrive while one is still queued coalesce."""
        if any(o.kind == "control" and o.correlation.startswith("v4-adopt") for o in self._outbox):
            return
        self.counters["adopt_commands"] += 1
        self._adopt_last_t = self.clock()
        self._outbox.append(Outbound(
            kind="control", device_id=self.multicast_device_id,
            frame=encode_run_control(self.run_id, CMD_ADOPT),
            correlation=f"v4-adopt{self.counters['adopt_commands']}"))
        self.log("run_control_adopt_queued", "server", {"why": why})

    def _note_unbound_uplink(self, dev_eui: str) -> None:
        """Rev 4.3 (3.4): a rostered DevEUI that is not BOUND is speaking, so its HELLO was lost.
        Queue one ADOPT (every received ADOPT makes a board send a HELLO, also for the run it
        already holds), no earlier than the guard after this uplink and no more than one per
        adopt_period_seconds."""
        if self.stage in (Stage.DONE, Stage.FAILED, Stage.ENDING) or not self.membership.authorised(dev_eui):
            return
        self._adopt_unbound.setdefault(dev_eui, self.clock())

    def _tick_adopt_unbound(self, now: float) -> None:
        if not self._adopt_unbound:
            return
        p = self.params
        # one per adopt_period_seconds among the ADOPTs this path sends (the WAITING_START ADOPTs
        # have their own clock: a HELLO lost just after one of them must not wait a whole period)
        if (self._adopt_unbound_sent_t is not None
                and now < self._adopt_unbound_sent_t + p.adopt_period_seconds):
            return
        newest = max(self._adopt_unbound.values())
        if now < newest + p.downlink_guard_seconds:
            return                                  # its RX windows are still open
        devs = sorted(self._adopt_unbound)
        self._adopt_unbound.clear()
        self._adopt_unbound_sent_t = now
        self.counters["adopt_for_unbound"] += 1
        self._queue_adopt("unbound_uplink:" + ",".join(devs[:8]))

    def _tick_waiting_adopt(self, now: float) -> None:
        """WAITING_START: boards that completed an earlier run are silent in Class C and only answer
        an ADOPT. Multicast one as soon as the run exists and then every adopt_period_seconds."""
        if self._adopt_last_t is None or now >= self._adopt_last_t + self.params.adopt_period_seconds:
            self._queue_adopt("waiting_start")

    def _tick_adopt(self, now: float) -> None:
        """A board that said HELLO with a different non-zero stored run gets ADOPT once it is
        listening (READY + the Class C guard), so the multicast actually reaches it."""
        if not self._adopt_waiting:
            return
        due = []
        for dev in list(self._adopt_waiting):
            member = self.membership.members.get(dev)
            if member is None or member.state is State.DEPARTED:
                del self._adopt_waiting[dev]
            elif member.ready and now >= member.ready_since + self.params.class_c_guard_seconds:
                due.append(dev)
        if due:
            for dev in due:
                del self._adopt_waiting[dev]
            self._queue_adopt("hello_with_other_run:" + ",".join(
                str(self._board_label(d)) for d in due[:8]))

    def _inflight_kind(self, kind: str):
        if self._inflight is not None and self._inflight.kind == kind:
            return self._inflight
        return None

    def _union_missing(self, g: GlobalState) -> int:
        union = 0
        for dev in g.required | g.reported:
            union |= g.board_missing.get(dev, ALL_GLOBAL_FRAGMENTS)
        return union

    def _refresh_global(self, g: GlobalState, now: float) -> None:
        """v2's union repair: one transmission serves every board still missing anything."""
        union = self._union_missing(g)
        if union == 0:
            self._end_downlink(now, gave_up=False)
            return
        if g.pending or self._inflight_kind("global") is not None:
            return
        outstanding = g.required - g.reported
        window_open = (g.burst_ended_at is not None
                       and now - g.burst_ended_at < self.params.multicast_report_window_seconds)
        if outstanding and window_open:
            self.log("multicast_awaiting_reports", "server", {
                "outstanding": sorted(self.membership.members[d].board_id for d in outstanding)})
            return
        heard = max((self.membership.members[d].last_heard_t for d in g.required | g.reported),
                    default=float("-inf"))
        if now < heard + self.params.downlink_guard_seconds:
            return                          # a board that must hear this may still be in its RX windows
        g.repair_rounds += 1
        if g.repair_rounds > self.params.max_global_repair_rounds:
            self._end_downlink(now, gave_up=True)
            return
        wanted = [i for i in range(GLOBAL_FRAGMENTS) if union & (1 << i)]
        g.attempts.clear()
        g.reported.clear()
        g.burst_ended_at = None
        g.prompt_burst = False
        g.pending = deque(wanted)
        g.armed = True
        g.bursts += 1
        g.next_send_at = now + self.params.retry_guard_seconds
        self.log("multicast_repair", "server", {
            "round": g.round, "repair_round": g.repair_rounds, "union_mask": f"{union:02X}",
            "fragments": wanted,
            "still_missing": {self._board_label(d): f"{m:02X}"
                              for d, m in g.board_missing.items() if m}})

    def _board_label(self, dev_eui: str):
        """board_id (an int) for logs and records; the DevEUI for a board never bound."""
        member = self.membership.members.get(dev_eui)
        return dev_eui if member is None or member.board_id is None else member.board_id

    def _end_downlink(self, now: float, gave_up: bool) -> None:
        g, phase = self._g, self.phase
        spec = phase.spec
        g.ended = True
        g.armed = False
        g.pending.clear()
        if self._inflight_kind("global") is not None:
            self._inflight = None
        g.ended_t = now
        held = sorted(d for d, m in g.board_missing.items() if m == 0)
        for dev in held:
            self._note_held(self.membership.members[dev], g.slot)
        missing = sorted(set(g.required) - set(held))
        self.log("global_delivery_end", "server", {
            "phase_index": spec.index, "round": g.round, "slot": g.slot,
            "held_boards": [self._board_label(d) for d in held],
            "missing_boards": [self._board_label(d) for d in missing], "gave_up": gave_up,
            "fragments_sent": g.fragments_sent, "bursts": g.bursts})
        if g.round == 0:
            phase.initial_global_crc = f"{g.crc:08X}"
            self.out.delivery({
                "record": "phase_start", "phase_index": spec.index, "name": spec.name,
                "mode": self.plan.mode, "seed": self.plan.seed, "rounds": spec.rounds,
                "planned_positions": None if spec.positions is None else list(spec.positions),
                "boards_holding_initial_global": [self._board_label(d) for d in held],
                "boards_missing_initial_global": [self._board_label(d) for d in missing],
                "initial_global_crc32": phase.initial_global_crc, "slot": g.slot,
                "global_fragments_sent": g.fragments_sent, "global_bursts": g.bursts,
                "started_utc": phase.started_utc,
                "delivery_duration_s": round(now - g.started_t, 3)})
        else:
            self._finish_round_record(g, held)
        if g.round == spec.rounds:
            phase.boards_holding_final = [self._board_label(d) for d in held]
            self._end_phase("complete", now)
        else:
            self._begin_upload(now)

    # ------------------------------------------------------------------
    # Upload phase
    # ------------------------------------------------------------------
    def _begin_upload(self, now: float) -> None:
        phase, u = self.phase, self._u
        spec = phase.spec
        assert u is not None
        participants = [m for m in self.membership.members.values()
                        if m.state is State.ACTIVE and self._hosting(m, spec)]
        u.participants = tuple(sorted(m.dev_eui for m in participants))
        u.expected = {p for m in participants for p in self._hosting(m, spec)}
        u.watch = tuple(sorted(
            m.dev_eui for m in self.membership.members.values()
            if m.state in (State.ACTIVE, State.SUSPECT) and self._hosting(m, spec)))
        if not u.expected and not u.updates:
            self._enter_idle("delivery ended with no participating board", now)
            return
        phase.positions_seen |= u.expected
        self._idle_since = None
        u.began = True
        u.started_t = now              # the round timeout runs from the end of delivery
        u.started_utc = self.wall_clock()
        self.stage = Stage.UPLOAD
        self.log("upload_started", "server", {
            "phase_index": spec.index, "round": u.round,
            "participants": [self._board_label(d) for d in u.participants],
            "expected_positions": len(u.expected), "already_arrived": sorted(u.updates)})
        # Boards that finished uploading during delivery announced readiness then;
        # nothing has looked at what they are missing yet.
        for dev in u.participants:
            member = self.membership.members[dev]
            if member.ready and member.ready_slot == u.seed_slot + 1:
                self._repair_member(member)

    def _tick_upload(self, now: float) -> None:
        u = self._u
        if u is None or u.closed:
            return
        # A position whose board has gone silent (not LIVE) is not waited for; if the board speaks
        # again before the round closes it is waited for once more.
        resolved = all(p in u.updates or p in u.abandoned or self._silent_position(p, now)
                       for p in u.expected)
        if resolved:
            reason = "all_delivered" if not (u.expected - set(u.updates)) else "all_resolved"
            self._close_round(now, reason)
        elif now - u.started_t >= self.params.round_timeout_seconds:
            self._close_round(now, "timeout")

    def _maybe_proactive_repair(self, now: float) -> None:
        u = self._u
        if u is None or u.closed or u.proactive_repair_at is None or now < u.proactive_repair_at:
            return
        u.proactive_repair_at = None
        self._proactive_repair()

    def _proactive_repair(self) -> None:
        """After a restart: whatever arrived before the crash is lost, but boards keep their
        outbox, so ask every ACTIVE / SUSPECT board for every position still missing, once."""
        u = self._u
        asked = 0
        for member in list(self.membership.members.values()):
            if member.state not in (State.ACTIVE, State.SUSPECT):
                continue
            for position in self._hosting(member, self.phase.spec):
                if position in u.updates or position in u.abandoned:
                    continue
                asked += int(self._request_update_repair(
                    member, position, (1 << UPDATE_FRAGMENTS) - 1))
        self.log("resume_repair_requested", "server", {"requests": asked})

    def _close_round(self, now: float, reason: str) -> None:
        phase, u, seed_g = self.phase, self._u, self._g
        spec = phase.spec
        u.closed = True
        arrived = sorted(u.updates)
        missing = sorted(u.expected - set(u.updates))
        pairs = [(u.updates[p].deq, 1.0) for p in arrived]
        self.log("round_closing", "server", {
            "phase_index": spec.index, "round": u.round, "reason": reason, "arrived": arrived,
            "missing": missing, "abandoned": sorted(u.abandoned)})
        # Empty pairs leave the server unchanged and do not advance FedAdam's t.
        phase.server.round(pairs)
        self._clear_pending_repairs("round_closed")
        # Membership: who delivered, who missed the round.
        self._update_misses(u, spec)
        changes = [c for c in self._round_changes]
        self._round_changes = []
        ended_utc = self.wall_clock()
        base_by_pos = {str(p): u.updates[p].base_slot for p in arrived}
        # participants (PROTOCOL_V4 4.1): the positions of the boards that took part in this round:
        # those waited for plus any board admitted during the round by acknowledging the seed or
        # delivering an UPDATE trained from it (mid-phase join).
        participants = sorted(u.expected | set(arrived))
        previous = set(phase.rounds[-1]["participants"]) if phase.rounds else None
        joined = [] if previous is None else sorted(set(participants) - previous)
        left = [] if previous is None else sorted(previous - set(participants))
        record = {
            "record": "round", "run_id": self.run_id, "phase_index": spec.index,
            "phase_name": spec.name, "round": u.round, "seed_slot": u.seed_slot,
            "result_slot": u.seed_slot + 1,
            "participants": participants, "arrived_positions": arrived,
            "stale_positions": [p for p in arrived if u.updates[p].stale],
            "joined_positions": joined, "left_positions": left,
            "expected_positions": sorted(u.expected),
            "missing_positions": missing, "abandoned_positions": sorted(u.abandoned),
            "not_gated_boards": sorted(
                self._board_label(d) for d in seed_g.not_gated | self._silent_owners(missing, now)),
            "participating_boards": sorted({
                self._board_label(self.membership.position_owner[p]) for p in participants
                if p in self.membership.position_owner}),
            "boards_holding_seed_global": sorted(
                self._board_label(d) for d, m in seed_g.board_missing.items() if m == 0),
            "close_reason": reason,
            "update_fragments_received": u.fragments_total,
            "update_fragments_by_position": {str(p): int(n) for p, n in sorted(u.fragments_rx.items())},
            "update_repair_requests": int(sum(u.repair_rounds.values())),
            "update_repairs_by_position": {str(p): int(n) for p, n in sorted(u.repair_rounds.items())},
            "update_repair_downlinks_sent": u.repair_downlinks_sent,
            "base_slot_by_position": base_by_pos,
            "train_loss": {str(p): u.updates[p].train_loss for p in arrived},
            "n_train_rows": {str(p): u.updates[p].n_train_rows for p in arrived},
            "membership_changes": [
                {k: c[k] for k in ("dev_eui", "board_id", "from", "to", "reason") if k in c}
                for c in changes],
            "joined_boards": [c["board_id"] for c in changes if c["to"] == "ACTIVE"],
            "left_boards": [c["board_id"] for c in changes if c["to"] in ("SUSPECT", "DEPARTED")],
            "timings": {
                "upload_start_s": self._rel(u.started_t), "upload_end_s": self._rel(now),
                "upload_duration_s": round(now - u.started_t, 3),
                "upload_start_utc": u.started_utc, "upload_end_utc": ended_utc},
        }
        self._last_repair_state = {
            "phase_index": spec.index, "round": u.round,
            "repair_rounds": {str(p): int(n) for p, n in u.repair_rounds.items()},
            "abandoned": sorted(u.abandoned)}
        self._pending_round = {
            "record": record,
            # downlink is derived from phase.held_by_slot[result_slot] when the masks are written
            "mask": {"slot": u.seed_slot + 1,
                     "uplink": {str(p): True for p in arrived},
                     "base": {str(p): u.updates[p].base_slot for p in arrived}},
        }
        g = self._make_global(seed_g.round + 1)
        if g is None:
            return
        self._install_global(g, now)

    def _update_misses(self, u: UploadState, spec: PhaseSpec) -> None:
        for dev in u.watch:
            member = self.membership.members[dev]
            if member.state not in (State.ACTIVE, State.SUSPECT):
                continue
            delivered = any(p in u.updates for p in self._hosting(member, spec))
            if delivered:
                member.misses = 0
                self._admit(member, "delivered")
                continue
            member.misses += 1
            if member.misses >= self.params.depart_after_rounds:
                self.membership.set_state(member, State.DEPARTED, "missed_rounds",
                                          misses=member.misses)
                member.ready = False
            elif member.misses >= self.params.suspect_after_rounds and member.state is State.ACTIVE:
                self.membership.set_state(member, State.SUSPECT, "missed_rounds",
                                          misses=member.misses)

    def _finish_round_record(self, g: GlobalState, held: list) -> None:
        """The GLOBAL produced by the pending round was delivered: complete its record."""
        pending = self._pending_round
        if pending is None:
            return
        phase = self.phase
        record = pending["record"]
        held_set = set(held)
        record["boards_holding_result_global"] = sorted(self._board_label(d) for d in held)
        record["result_global_round"] = g.round
        record["result_global_crc32"] = f"{g.crc:08X}"
        record["global_fragments_sent"] = g.fragments_sent
        record["global_fragments_queued"] = g.fragments_queued
        record["global_bursts"] = g.bursts
        record["global_repair_rounds"] = g.repair_rounds
        record["global_prompt_repairs"] = g.prompt_repairs
        record["timings"].update({
            "downlink_start_s": self._rel(g.started_t), "downlink_end_s": self._rel(g.ended_t),
            "downlink_duration_s": round(g.ended_t - g.started_t, 3),
            "downlink_start_utc": g.started_utc, "downlink_end_utc": self.wall_clock()})
        for dev in held_set:
            phase.held_by_slot.setdefault(g.slot, set()).add(dev)
        phase.masks.append(pending["mask"])
        phase.rounds.append(record)
        self._pending_round = None
        if self.out.delivery(record):
            self.log("round_complete", "server", {
                "phase_index": record["phase_index"], "round": record["round"],
                "arrived": len(record["arrived_positions"]),
                "expected": len(record["expected_positions"]),
                "boards_holding_result_global": record["boards_holding_result_global"]})
        self._write_masks()

    def _end_phase(self, status: str, now: float) -> None:
        phase = self.phase
        spec = phase.spec
        if self._pending_round is not None:
            # The last round closed but its result GLOBAL was never delivered: keep its record.
            self._finish_round_record(self._g, [])
        phase.status = status
        phase.ended_t = now
        phase.ended_utc = self.wall_clock()
        if status == "complete":
            g = self._g
            phase.final_crc = f"{g.crc:08X}"
            phase.final_codes_sha256 = g.model.codes_sha256()
        self._idle_since = None
        self.log("phase_complete" if status == "complete" else "phase_aborted", "server", {
            "phase_index": spec.index, "status": status, "rounds": len(phase.rounds),
            "final_crc32": phase.final_crc})
        self.out.delivery({"record": "phase_end", **self._phase_summary(phase)})
        self._write_summaries()
        self._u = None
        if spec.index + 1 < len(self.plan.phases):
            self._begin_phase(spec.index + 1, now)
        else:
            self._begin_collecting(now)

    # ------------------------------------------------------------------
    # IDLE: a phase with zero participants pauses
    # ------------------------------------------------------------------
    def _enter_idle(self, reason: str, now: float) -> None:
        if self._idle_since is None:
            self._idle_since = now
            self.log("phase_paused", "server", {
                "phase_index": self.phase.spec.index, "reason": reason,
                "timeout_seconds": self.params.idle_phase_timeout_seconds})
        self.stage = Stage.IDLE
        self._idle_retry_at = now + self.params.idle_retry_seconds

    def _tick_idle(self, now: float) -> None:
        g = self._g
        if g is None:
            return
        candidates = self._required_for(g) or self._ready_candidates()
        if candidates and now >= self._idle_retry_at:
            self.log("phase_resuming", "server", {
                "phase_index": self.phase.spec.index, "candidates": len(candidates)})
            self._u = None if g.round >= self.phase.spec.rounds else self._u
            self._begin_downlink(g, now)

    def _check_idle_timeout(self, now: float) -> None:
        if (self._idle_since is not None and self.stage in (Stage.IDLE, Stage.DOWNLINK)
                and now - self._idle_since >= self.params.idle_phase_timeout_seconds):
            self.log("phase_idle_timeout", "server", {
                "phase_index": self.phase.spec.index,
                "idle_seconds": round(now - self._idle_since, 1)})
            self._clear_pending_repairs("phase_aborted")
            if self._inflight_kind("global") is not None:
                self._inflight = None
            self._end_phase("aborted_idle", now)

    # ------------------------------------------------------------------
    # UPDATE fragments
    # ------------------------------------------------------------------
    def _handle_update_fragment(self, member: Member, frame: Frame) -> None:
        """One UPDATE fragment: its body starts with the u16 position, so it is attributable
        on its own whatever else the board sent around it."""
        u = self._u
        member.counters["update_fragments_received"] += 1
        if self.stage not in (Stage.DOWNLINK, Stage.UPLOAD) or u is None or u.closed:
            self.counters["late_update_fragments"] += 1
            self.log("late_update_fragment", member.device_id, {
                "board_id": member.board_id, "stage": self.stage.value,
                "round_id": frame.round_id, "fragment": frame.fragment_index})
            return
        if frame.round_id != u.seed_slot:
            self.counters["stale_update_fragments"] += 1
            self.log("stale_update_fragment", member.device_id, {
                "board_id": member.board_id, "round_id": frame.round_id,
                "expected_seed_slot": u.seed_slot})
            return
        position, chunk = split_update_fragment(frame)
        u.fragments_total += 1
        if not self._position_valid(member, position):
            return
        u.fragments_rx[position] += 1
        if position in u.updates:
            self.log("duplicate_update_fragment", member.device_id,
                     {"position": position, "fragment": frame.fragment_index})
            return
        partial = u.partials.setdefault(position, new_update_partial())
        try:
            payload = partial.add(chunk)
        except ConflictingFragment as exc:
            self.counters["update_fragment_conflicts"] += 1
            self.log("update_fragment_conflict", member.device_id, {
                "position": position, "fragment": frame.fragment_index, "error": str(exc)})
            return
        except ProtocolError as exc:
            u.partials.pop(position, None)
            self.log("update_partial_reset", member.device_id, {"position": position, "error": str(exc)})
            return
        self.log("update_fragment", member.device_id, {
            "position": position, "fragment": frame.fragment_index, "have": sorted(partial.parts)})
        if payload is not None:
            self._complete_update(member, position, payload)

    def _position_valid(self, member: Member, position: int) -> bool:
        spec = self.phase.spec
        if position not in member.positions:
            self.counters["update_positions_rejected"] += 1
            self.log("update_position_rejected", member.device_id, {
                "board_id": member.board_id, "position": position,
                "reason": "position is not hosted by this board (its HELLO listed "
                          f"{list(member.positions)})"})
            return False
        if not spec.covers(position):
            # The board uploads every hosted position; the plan simply does not use this one.
            self.counters["updates_ignored_not_in_phase"] += 1
            key = (spec.index, position)
            if key not in self._ignored_notified:
                self._ignored_notified.add(key)
                self.log("update_position_ignored", member.device_id, {
                    "board_id": member.board_id, "position": position,
                    "reason": f"position is not part of phase {spec.index}"})
            return False
        return True

    def _reject_update(self, u: UploadState, member: Member, position: int, counter: str,
                       reason: str) -> None:
        """A CRC-valid UPDATE that must not be aggregated. Resending would not help."""
        u.partials.pop(position, None)
        u.abandoned.add(position)
        self.counters[counter] += 1
        self.log("update_rejected", member.device_id, {
            "position": position, "board_id": member.board_id, "error": reason})

    def _complete_update(self, member: Member, hint: int, payload: bytes) -> None:
        u, phase = self._u, self.phase
        spec = phase.spec
        try:
            update = decode_update(payload)
        except ProtocolError as exc:
            u.partials.pop(hint, None)
            self.counters["update_crc_failures"] += 1
            self.log("update_rejected", member.device_id, {
                "position": hint, "error": str(exc),
                "hint": "fragments discarded; the repair flow asks for the position again"})
            return
        if update.position != hint:
            u.partials.pop(hint, None)
            self.counters["update_position_mismatches"] += 1
            self.log("update_rejected", member.device_id, {
                "position": hint,
                "error": f"fragments carry position {hint} but the reassembled UPDATE says {update.position}"})
            return
        position = update.position
        if position in u.updates:
            u.partials.pop(position, None)
            return
        if update.run_id != self.run_id:
            self._reject_update(u, member, position, "update_run_id_mismatches",
                                f"UPDATE run_id {update.run_id} != this run's {self.run_id}")
            return
        if update.phase_index != spec.index or update.round != u.round:
            u.partials.pop(position, None)
            self.log("update_rejected", member.device_id, {
                "position": position,
                "error": f"UPDATE is for phase {update.phase_index} round {update.round}; "
                         f"expected phase {spec.index} round {u.round}"})
            return
        if not (phase.first_slot <= update.base_slot <= u.seed_slot
                and update.base_slot in phase.globals_deq):
            self._reject_update(
                u, member, position, "update_base_slot_rejections",
                f"base_slot {update.base_slot} is unknown or outside phase {spec.index} "
                f"(valid {phase.first_slot}..{u.seed_slot})")
            return
        delta_flat = update.model.dequantize()
        stale = update.base_slot != u.seed_slot
        if stale and position not in self._previous_participants():
            # Mid-phase join (PROTOCOL_V4 4): a board is ACTIVE from the first round whose seed
            # GLOBAL it holds, and the twin replays a joiner from that seed. An UPDATE trained from
            # an older GLOBAL by a position that did not take part in the previous round does not
            # show that, so the position stays absent this round.
            u.partials.pop(position, None)
            self.counters["updates_ignored_unadmitted_stale_base"] += 1
            self.log("update_ignored_unadmitted", member.device_id, {
                "position": position, "board_id": member.board_id, "base_slot": update.base_slot,
                "seed_slot": u.seed_slot,
                "reason": "the board has not shown it holds this round's seed GLOBAL"})
            return
        if stale:
            # The board trained from an older GLOBAL: rebuild its model and take the
            # difference to the CURRENT global, as dropout_fl's stale copy does.
            client_model = phase.globals_deq[update.base_slot] + delta_flat
            deq = unflatten(client_model - phase.globals_deq[u.seed_slot])
            self.counters["stale_base_updates"] += 1
        else:
            deq = unflatten(delta_flat)
        u.updates[position] = AcceptedUpdate(
            position=position, deq=deq, n_train_rows=update.n_train_rows,
            train_loss=update.train_loss, scales=update.model.scales,
            arrived_t=self.clock(), base_slot=update.base_slot, stale=stale)
        u.partials.pop(position, None)
        phase.positions_seen.add(position)
        self._note_held(member, update.base_slot)       # it trained from that GLOBAL, so it held it
        member.counters["updates_accepted"] += 1
        self.log("update_complete", member.device_id, {
            "position": position, "board_id": member.board_id, "crc32": f"{update.crc32:08X}",
            "train_loss": update.train_loss, "n_train_rows": update.n_train_rows,
            "base_slot": update.base_slot, "stale_base": stale})
        g = self._g
        # A stale-base UPDATE says the board did NOT hold the seed GLOBAL, so it proves nothing
        # about it; a fresh one proves it did (and admits a BOUND board).
        if g is not None and g.slot == u.seed_slot and not stale:
            self._mark_held(member, g, "update_received")
            self._finish_delivery_if_complete(g, self.clock())

    # ------------------------------------------------------------------
    # CLASS_C_READY and repair
    # ------------------------------------------------------------------
    def _set_ready(self, member: Member, slot: int) -> None:
        member.ready = True
        member.ready_slot = slot
        member.ready_since = self.clock()

    def _handle_ready(self, member: Member, frame: Frame) -> None:
        ready = decode_class_c_ready(frame)
        self._set_ready(member, ready.awaited_slot)
        self.log("class_c_ready", member.device_id, {
            "board_id": member.board_id, "awaited_slot": ready.awaited_slot,
            "run_id": ready.run_id, "state": member.state.value})
        u, g = self._u, self._g
        # READY for the slot after this round's seed = "my upload (or repair) is done".
        if (self.stage in (Stage.DOWNLINK, Stage.UPLOAD) and u is not None and not u.closed
                and g is not None and g.slot == u.seed_slot
                and ready.awaited_slot == u.seed_slot + 1 and self._phase_hosting(member)):
            self._mark_held(member, g, "ready_for_next_slot")
            self._repair_member(member)
            self._finish_delivery_if_complete(g, self.clock())
        elif (self.stage is Stage.DOWNLINK and g is not None and not g.ended and g.ever_armed
                and member.dev_eui in g.required and g.board_missing.get(member.dev_eui) != 0
                and (ready.awaited_slot == NO_SLOT or ready.awaited_slot <= g.slot)):
            # waiting in Class C for a GLOBAL it does not hold, after the multicast went out: it
            # missed it (it was transmitting or inside its RX windows). Repair promptly.
            self._request_prompt_repair(member, g, ALL_GLOBAL_FRAGMENTS, self.clock(), "class_c_ready")

    def _repair_member(self, member: Member, force: bool = False) -> None:
        """The board says its upload is done: ask for whatever is still missing.

        Only for a board known to hold the GLOBAL that seeds the round: before that it has
        nothing to upload, and its readiness merely means it is listening for the multicast.
        """
        u, g = self._u, self._g
        if u is None or u.closed or g is None or g.slot != u.seed_slot:
            return
        if not force and g.board_missing.get(member.dev_eui) != 0:
            return
        asked = False
        for position in self._hosting(member, self.phase.spec):
            if position in u.updates or position in u.abandoned:
                continue
            partial = u.partials.get(position)
            missing = partial.missing_mask() if partial is not None else (1 << UPDATE_FRAGMENTS) - 1
            if not missing:
                continue
            asked |= self._request_update_repair(member, position, missing)
        if asked:
            member.ready = False        # repair makes the board leave Class C to resend

    def _request_update_repair(self, member: Member, position: int, missing: int) -> bool:
        u = self._u
        u.repair_rounds[position] += 1
        rounds = u.repair_rounds[position]
        if rounds > self.params.max_repair_rounds:
            u.repair_rounds[position] -= 1
            u.abandoned.add(position)
            self.log("update_abandoned", member.device_id, {
                "position": position, "board_id": member.board_id,
                "repair_rounds": self.params.max_repair_rounds, "missing_mask": f"{missing:02X}"})
            return False
        frame = encode_update_repair(u.seed_slot, self.run_id, position, missing)
        correlation = f"v4-s{u.seed_slot}-p{position}-repair{rounds}"
        self._outbox.append(Outbound(kind="repair", device_id=member.device_id, frame=frame,
                                     correlation=correlation, position=position, slot=u.seed_slot))
        member.counters["update_repairs_requested"] += 1
        self.log("update_repair_requested", member.device_id, {
            "position": position, "board_id": member.board_id, "round": rounds,
            "missing_mask": f"{missing:02X}",
            "missing_fragments": [i for i in range(UPDATE_FRAGMENTS) if missing >> i & 1],
            "application_bytes": len(frame), "correlation_id": correlation})
        return True

    # ------------------------------------------------------------------
    # GLOBAL_STATUS
    # ------------------------------------------------------------------
    def _handle_status(self, member: Member, frame: Frame) -> None:
        status = decode_global_status(frame)
        g, now = self._g, self.clock()
        if status.complete and status.run_id != self.run_id:
            self.log("global_status_other_run", member.device_id, {
                "board_id": member.board_id, "status_run_id": status.run_id,
                "server_run_id": self.run_id, "slot": status.slot})
            return
        if (g is None or self.stage not in (Stage.DOWNLINK, Stage.UPLOAD)
                or not self._phase_hosting(member)):
            # No GLOBAL in flight, or a board with no machines in this phase: it is in
            # Class C, which is all that matters here.
            self._set_ready(member, status.slot)
            self.log("global_status_ignored", member.device_id, {
                "board_id": member.board_id, "slot": status.slot,
                "missing": f"{status.missing_mask:02X}",
                "reason": "no GLOBAL in flight" if g is None or self.stage not in (
                    Stage.DOWNLINK, Stage.UPLOAD) else "board hosts no position of this phase"})
            return
        spec = self.phase.spec
        if status.slot == g.slot:
            if self.stage is Stage.UPLOAD:
                self._record_late_seed_report(member, g, status.missing_mask, now)
            elif not g.ever_armed:
                # Nothing has been sent yet: the board is simply waiting in Class C.
                self._set_ready(member, status.slot)
                self.log("global_status_while_waiting", member.device_id, {
                    "board_id": member.board_id, "implied_class_c_ready": True})
            else:
                self._record_global_report(member, g, status.missing_mask, now)
                # repaired promptly, downlink_guard_seconds after this uplink (never while the board is
                # still inside its RX windows), at most once per prompt_repair_min_gap_seconds; a
                # throttled report falls through to the periodic timer
                if not (status.missing_mask and self.params.prompt_repair
                        and self._request_prompt_repair(member, g, status.missing_mask, now,
                                                        "global_status")):
                    self._refresh_global_if_due(g, now)
        elif status.slot == g.slot + 1 and g.round < spec.rounds:
            # The board is waiting for the NEXT GLOBAL, so it has finished this round; it
            # heard nothing because nothing has been sent. That is both an implied
            # CLASS_C_READY and proof it holds the seed GLOBAL.
            self._mark_held(member, g, "status_for_next_slot")
            self._set_ready(member, status.slot)
            self.log("global_status_while_waiting", member.device_id, {
                "board_id": member.board_id, "implied_class_c_ready": True})
            self._repair_member(member)
            self._finish_delivery_if_complete(g, now)
        elif status.slot == g.slot + 1:
            # After the phase's final GLOBAL: the board is listening for the next phase.
            self._set_ready(member, status.slot)
        else:
            self.log("global_status_wrong_slot", member.device_id, {
                "board_id": member.board_id, "slot": status.slot, "expected": g.slot})

    def _finish_delivery_if_complete(self, g: GlobalState, now: float) -> None:
        """Inferred acknowledgements may be the last thing a delivery phase was waiting for."""
        if (self.stage is Stage.DOWNLINK and not g.ended and g.ever_armed and g.required
                and all(g.board_missing.get(d) == 0 for d in g.required)):
            self._refresh_global(g, now)

    def _mark_held(self, member: Member, g: GlobalState, why: str) -> None:
        """The board holds GLOBAL g, inferred from an UPDATE / READY / STATUS for the next slot.

        Acks are unconfirmed uplinks and can be lost; a valid UPDATE for round r (or a READY
        for the slot after) can only come from a board that received the GLOBAL that opened it.
        """
        if g.board_missing.get(member.dev_eui) != 0:
            g.board_missing[member.dev_eui] = 0
            g.reported.add(member.dev_eui)
            self.log("global_held_inferred", member.device_id, {
                "board_id": member.board_id, "round": g.round, "slot": g.slot, "evidence": why})
        self._note_held(member, g.slot)
        self._admit(member, f"{why}_slot_{g.slot}")

    def _record_global_report(self, member: Member, g: GlobalState, missing: int, now: float) -> None:
        g.board_missing[member.dev_eui] = missing
        g.reported.add(member.dev_eui)
        if missing == 0:
            member.ready = False       # it has left Class C to train
            member.counters["global_acknowledged"] += 1
            self.log("global_acknowledged", member.device_id, {
                "board_id": member.board_id, "round": g.round, "slot": g.slot})
            self._note_held(member, g.slot)
            self._admit(member, f"acknowledged_slot_{g.slot}")
        else:
            member.ready = True        # still listening, still missing something
            member.ready_slot = g.slot
            member.ready_since = now
            self.log("global_missing", member.device_id, {
                "board_id": member.board_id, "missing_mask": f"0x{missing:02X}",
                "fragments": [i for i in range(GLOBAL_FRAGMENTS) if missing >> i & 1]})

    def _refresh_global_if_due(self, g: GlobalState, now: float) -> None:
        if g.ended or not g.ever_armed:
            return
        self._refresh_global(g, now)

    def _record_late_seed_report(self, member: Member, g: GlobalState, missing: int,
                                 now: float) -> None:
        """A board reports about the GLOBAL that seeded THIS round (during upload)."""
        if missing == 0:
            g.board_missing[member.dev_eui] = 0
            self.log("global_late_acknowledged", member.device_id, {
                "board_id": member.board_id, "round": g.round})
            self._note_held(member, g.slot)
            self._admit(member, f"acknowledged_slot_{g.slot}")
            return
        g.board_missing[member.dev_eui] = missing
        if g.late_repairs[member.dev_eui] >= self.params.max_global_repair_rounds:
            self.log("global_late_repair_exhausted", member.device_id, {"board_id": member.board_id})
            return
        g.late_repairs[member.dev_eui] += 1
        wanted = [i for i in range(GLOBAL_FRAGMENTS) if missing >> i & 1]
        for index in wanted:
            if index not in g.pending:
                g.pending.append(index)
        if self._inflight_kind("global") is None:
            g.attempts.clear()
        g.armed = True
        g.prompt_burst = False
        g.bursts += 1
        g.next_send_at = now + self.params.retry_guard_seconds
        self.log("global_late_repair", member.device_id, {
            "board_id": member.board_id, "fragments": wanted, "round": g.round})

    # ------------------------------------------------------------------
    # ALARM / SELF_CHECK / REPLAY_DONE
    # ------------------------------------------------------------------
    def _own_position(self, member: Member, position: int, event: str) -> bool:
        if position not in member.positions:
            self.counters["results_rejected"] += 1
            self.log(event, member.device_id, {
                "board_id": member.board_id, "position": position,
                "reason": f"position {position} is not hosted by bound board {member.board_id}"})
            return False
        return True

    def _same_run(self, member: Member, run_id: int, event: str) -> bool:
        if run_id != self.run_id:
            self.counters["results_other_run"] += 1
            self.log(event, member.device_id, {
                "board_id": member.board_id, "run_id": run_id, "server_run_id": self.run_id,
                "reason": "message belongs to another run"})
            return False
        return True

    def _handle_alarm(self, member: Member, frame: Frame) -> None:
        alarm = decode_alarm(frame)
        if not self._same_run(member, alarm.run_id, "alarm_rejected"):
            return
        if not self._own_position(member, alarm.position, "alarm_rejected"):
            return
        if alarm.phase_index >= len(self.plan.phases):
            raise ProtocolError(f"ALARM phase_index {alarm.phase_index} is not a phase of this plan")
        key = (alarm.position, alarm.phase_index)        # once per (position, phase)
        known = self._alarms.get(key)
        if known is not None:
            if known["index"] != alarm.acquisition_index:
                self._violation(member, "alarm_index_differs", position=alarm.position,
                                phase_index=alarm.phase_index, alarm_index=alarm.acquisition_index,
                                recorded_index=known["index"], recorded_source=known["source"])
            self.log("duplicate_alarm", member.device_id, {
                "position": alarm.position, "phase_index": alarm.phase_index,
                "recorded_source": known["source"]})
            if known["source"] == "replay_done":
                # the ALARM arrived after the REPLAY_DONE that already carried its index: the
                # row stays as written (source=replay_done); only the diagnostics are lost
                self.counters["alarm_after_replay_done"] += 1
            return
        self._alarms[key] = {"index": alarm.acquisition_index, "source": "alarm"}
        self._alarm_phases.setdefault(alarm.position, set()).add(alarm.phase_index)
        record = {
            "run_id": self.run_id, "position": alarm.position, "phase_index": alarm.phase_index,
            "acquisition_index": alarm.acquisition_index, "score": alarm.score,
            "threshold": alarm.threshold, "source": "alarm",
            "timestamp_utc": self.wall_clock(), "device_id": member.device_id,
            "board_id": member.board_id,
        }
        self.out.alarm(record)
        self._collect_dirty = True
        self.alarm_count[alarm.phase_index] += 1
        self.log("alarm", member.device_id, {k: record[k] for k in (
            "position", "phase_index", "acquisition_index", "score", "threshold")})

    def _handle_self_check(self, member: Member, frame: Frame) -> None:
        check = decode_self_check(frame)
        if not self._own_position(member, check.position, "self_check_rejected"):
            return
        record = {
            "timestamp_utc": self.wall_clock(), "device_id": member.device_id,
            "board_id": member.board_id, "position": check.position,
            "rows_checked": check.rows_checked, "max_abs_diff": check.max_abs_diff,
            "median_abs_diff": check.median_abs_diff,
        }
        self.out.self_check(record)
        self.self_check_count += 1
        self.log("self_check", member.device_id, {k: record[k] for k in (
            "position", "rows_checked", "max_abs_diff", "median_abs_diff")})

    def _handle_replay_done(self, member: Member, frame: Frame) -> None:
        done = decode_replay_done(frame)
        if not self._same_run(member, done.run_id, "replay_done_rejected"):
            return
        if not self._own_position(member, done.position, "replay_done_rejected"):
            return
        if done.phase_index >= len(self.plan.phases):
            raise ProtocolError(f"REPLAY_DONE phase_index {done.phase_index} is not a phase of this plan")
        key = (done.position, done.phase_index)
        first_index = None if done.first_alarm_index == NO_ALARM_INDEX else done.first_alarm_index
        consistent = done.consistent
        if not consistent:
            self._violation(member, "replay_done_latched_vs_index", position=done.position,
                            phase_index=done.phase_index, alarm_latched=done.alarm_latched,
                            first_alarm_index=done.first_alarm_index)
        record = {
            "run_id": self.run_id, "position": done.position, "phase_index": done.phase_index,
            "acquisitions_scored": done.acquisitions_scored, "alarm_latched": done.alarm_latched,
            "first_alarm_index": done.first_alarm_index,
            "timestamp_utc": self.wall_clock(), "device_id": member.device_id,
            "board_id": member.board_id,
        }
        previous = self.replay_done.get(key)
        if previous is not None:
            if previous["first_alarm_index"] != done.first_alarm_index:
                self._violation(member, "replay_done_changed", position=done.position,
                                phase_index=done.phase_index,
                                first_alarm_index=done.first_alarm_index,
                                previous=previous["first_alarm_index"])
            self.log("duplicate_replay_done", member.device_id,
                     {"position": done.position, "phase_index": done.phase_index})
        # REPLAY_DONE is authoritative for the alarm (3.3): reconcile it with the ALARM message.
        if consistent and first_index is not None:
            known = self._alarms.get(key)
            if known is None:
                self._alarms[key] = {"index": first_index, "source": "replay_done"}
                self.out.alarm({
                    "run_id": self.run_id, "position": done.position, "phase_index": done.phase_index,
                    "acquisition_index": first_index, "score": "", "threshold": "",
                    "source": "replay_done", "timestamp_utc": self.wall_clock(),
                    "device_id": member.device_id, "board_id": member.board_id})
                self.alarm_count[done.phase_index] += 1
                self.counters["alarms_from_replay_done"] += 1
                self.log("alarm_from_replay_done", member.device_id, {
                    "position": done.position, "phase_index": done.phase_index,
                    "acquisition_index": first_index,
                    "note": "no ALARM message arrived; only score and threshold are lost"})
            elif known["index"] != first_index:
                self._violation(member, "alarm_index_vs_replay_done", position=done.position,
                                phase_index=done.phase_index, alarm_index=known["index"],
                                first_alarm_index=first_index)
        self.replay_done[key] = record
        self.out.replay_done(record)
        self._collect_dirty = True
        # A board that replays with a phase's final model holds it (the downlink mask says so).
        phase = next((p for p in self.phases if p.spec.index == done.phase_index), None)
        if phase is not None and phase.final_slot is not None:
            self._note_held(member, phase.final_slot)
        self.log("replay_done", member.device_id, {k: record[k] for k in (
            "position", "phase_index", "acquisitions_scored", "alarm_latched", "first_alarm_index")})

    def _violation(self, member: Member, kind: str, **details) -> None:
        """A board broke a normative rule of the contract (logged; the message is still processed)."""
        self.counters["protocol_violations"] += 1
        self.log("protocol_violation", member.device_id, {
            "kind": kind, "board_id": member.board_id, **details})

    # ------------------------------------------------------------------
    # COLLECTING: RUN_CONTROL REPLAY until REPLAY_DONE, then END
    # ------------------------------------------------------------------
    def _completed_phases(self) -> list[int]:
        return [ph.spec.index for ph in self.phases if ph.status == "complete"]

    def _begin_collecting(self, now: float) -> None:
        self.stage = Stage.COLLECTING
        self._u = None
        self._collect_deadline = now + self.params.post_run_collect_seconds
        self._replay_phases = self._completed_phases()
        self._next_replay_at = now
        self.log("sweep_complete", "server", {
            "completed_phases": self._replay_phases,
            "collect_seconds": self.params.post_run_collect_seconds})
        self.finalize()
        self._checkpoint()

    def _expected_replay(self) -> set:
        """{(position, phase)}: every position of a non-departed board that holds the final model
        of a completed phase must report REPLAY_DONE for that phase."""
        completed = set(self._completed_phases())
        expected = set()
        for m in self.membership.members.values():
            if m.state is State.DEPARTED:
                continue
            for phase in m.held_final_phases & completed:
                spec = self.plan.phase(phase)
                for p in m.positions:
                    if spec.covers(p):            # rev 4.3: a board replays only its active positions
                        expected.add((p, phase))
        return expected

    def _replay_complete(self) -> bool:
        expected = self._expected_replay()
        return bool(expected) and expected <= set(self.replay_done)

    def _tick_collecting(self, now: float) -> None:
        if self._collect_dirty and now - self._last_collect_checkpoint >= 5.0:
            self._collect_dirty = False
            self._last_collect_checkpoint = now
            self._checkpoint()                    # results since the last checkpoint survive a crash
        if not self._replay_phases:
            self._begin_ending("nothing_to_replay", now)
        elif self.counters["replay_commands"] == 0:
            self._queue_replay()                       # always at least one REPLAY
            self._next_replay_at = now + self.params.replay_repeat_seconds
        elif self._replay_complete():
            self._begin_ending("all_replay_done", now)
        elif not self._expected_replay():
            self._begin_ending("no_board_holds_a_final_model", now)
        elif now >= self._collect_deadline:
            self._begin_ending("collect_window_elapsed", now)
        elif (now >= self._next_replay_at and not any(o.kind == "control" for o in self._outbox)
              and not (self._inflight is not None and self._inflight.kind == "control")):
            self._queue_replay()
            self._next_replay_at = now + self.params.replay_repeat_seconds

    def _control_frames(self, command: int, phases: list[int]) -> list[bytes]:
        if command == CMD_END or not phases:
            return [encode_run_control(self.run_id, command, ())]
        return [encode_run_control(self.run_id, command, phases[i:i + MAX_PHASES_PER_RUN_CONTROL])
                for i in range(0, len(phases), MAX_PHASES_PER_RUN_CONTROL)]

    def _queue_replay(self) -> None:
        self.counters["replay_commands"] += 1
        for n, frame in enumerate(self._control_frames(CMD_REPLAY, self._replay_phases)):
            self._outbox.append(Outbound(
                kind="control", device_id=self.multicast_device_id, frame=frame,
                correlation=f"v4-replay{self.counters['replay_commands']}-{n}"))
        self.log("run_control_replay_queued", "server", {
            "phases": self._replay_phases, "attempt": self.counters["replay_commands"]})

    # ------------------------------------------------------------------
    # ENDING (rev 4.3, 3.4): END is repeated until the boards go quiet
    # ------------------------------------------------------------------
    def _touch_uplink(self) -> None:
        self._last_uplink_t = self.clock()

    def _begin_ending(self, why: str, now: float) -> None:
        """A board misses any multicast while transmitting or inside its RX windows, and END has no
        acknowledgement: so END is sent only once the gateway has been quiet for
        ``downlink_guard_seconds``, again after any later uplink, and silence is the acknowledgement."""
        self.stage = Stage.ENDING
        self._ending_why = why
        self._ending_started = now
        self._last_end_sent_t = None
        self._last_end_queued_t = None
        self.end_unconfirmed = []
        self.log("run_control_ending", "server", {
            "why": why, "downlink_guard_seconds": self.params.downlink_guard_seconds,
            "end_quiet_seconds": self.params.end_quiet_seconds,
            "end_linger_seconds": self.params.end_linger_seconds})
        self.finalize()
        self._checkpoint()

    def _end_in_flight(self) -> bool:
        return (any(o.kind == "control" and o.correlation.startswith("v4-end") for o in self._outbox)
                or (self._inflight is not None and self._inflight.correlation.startswith("v4-end")))

    def _queue_end_burst(self, now: float, why: str) -> None:
        self.counters["end_sends"] += 1
        self._last_end_queued_t = now
        copies = max(1, self.params.end_repeats)
        for n in range(copies):
            self._outbox.append(Outbound(
                kind="control", device_id=self.multicast_device_id,
                frame=self._control_frames(CMD_END, [])[0],
                correlation=f"v4-end{self.counters['end_sends']}-{n}"))
        self.log("run_control_end_queued", "server", {
            "why": why, "copies": copies, "send": self.counters["end_sends"]})

    def _tick_ending(self, now: float) -> None:
        if self._collect_dirty and now - self._last_collect_checkpoint >= 5.0:
            self._collect_dirty = False
            self._last_collect_checkpoint = now
            self._checkpoint()
        p = self.params
        sent = self._last_end_sent_t
        if now - self._ending_started >= p.end_linger_seconds:
            # still talking: spoke within the last quiet period although END has been (re)sent
            talking = sorted(
                self._board_label(m.dev_eui) for m in self.membership.members.values()
                if m.board_id is not None and m.last_heard_t >= now - p.end_quiet_seconds
                and m.last_heard_t > self._ending_started)
            self._finish_ending(now, talking)
            return
        if sent is not None and self._last_uplink_t <= sent and now - sent >= p.end_quiet_seconds:
            self._finish_ending(now, [])
            return
        if self._end_in_flight():
            return
        quiet = now >= self._last_uplink_t + p.downlink_guard_seconds
        if self._last_end_queued_t is None:
            if quiet:
                self._queue_end_burst(now, self._ending_why)
        elif (sent is not None and self._last_uplink_t > sent and quiet
              and now - self._last_end_queued_t >= p.end_repeat_seconds):
            self._queue_end_burst(now, "uplink_after_end")

    def _finish_ending(self, now: float, unconfirmed: list) -> None:
        self.end_unconfirmed = list(unconfirmed)
        self.stage = Stage.DONE
        self.log("run_done", "server", {
            "end_unconfirmed": self.end_unconfirmed, "end_sends": self.counters["end_sends"],
            "ending_seconds": round(now - self._ending_started, 1)})
        self.finalize()
        self._checkpoint()

    # ------------------------------------------------------------------
    # Downlink pump
    # ------------------------------------------------------------------
    @_synchronized
    def tick(self) -> None:
        now = self.clock()
        if self._boards_dirty and now - self._last_boards_write >= 2.0:
            self._write_boards()
        if self.stage in (Stage.DONE, Stage.FAILED):
            return
        if self.stage is Stage.WAITING_START:
            self._expire_inflight(now)
            self._maybe_start(now)
            if self.stage is Stage.WAITING_START:
                self._tick_waiting_adopt(now)
                self._tick_adopt_unbound(now)
                self._pump(now)
            return
        self._expire_inflight(now)
        self._check_idle_timeout(now)
        if self.stage in (Stage.DOWNLINK, Stage.UPLOAD, Stage.IDLE, Stage.COLLECTING):
            self._tick_adopt_unbound(now)
        if self.stage in (Stage.DOWNLINK, Stage.UPLOAD, Stage.IDLE):
            self._tick_adopt(now)
            self._tick_silence(now)
        if self.stage in (Stage.DOWNLINK, Stage.UPLOAD):
            self._maybe_proactive_repair(now)
        if self.stage is Stage.DOWNLINK:
            self._tick_downlink(now)
        if self.stage is Stage.UPLOAD:
            self._tick_upload(now)
        if self.stage is Stage.IDLE:
            self._tick_idle(now)
        if self.stage is Stage.COLLECTING:
            self._tick_collecting(now)
        if self.stage is Stage.ENDING:
            self._tick_ending(now)
        if self.stage in (Stage.DOWNLINK, Stage.UPLOAD, Stage.IDLE, Stage.COLLECTING, Stage.ENDING):
            self._pump(now)

    def _expire_inflight(self, now: float) -> None:
        out = self._inflight
        if out is None or now < out.deadline:
            return
        self._inflight = None
        self.log("downlink_status_timeout", out.device_id, {
            "kind": out.kind, "fragment": out.index, "position": out.position,
            "attempt": out.attempts, "timeout_seconds": self.params.inflight_downlink_timeout_seconds})
        self._requeue(out, now)

    def _requeue(self, out: Outbound, now: float) -> None:
        g = self._g
        if out.kind == "global":
            # Only if it still belongs to the GLOBAL being delivered.
            if g is not None and not g.ended and g.slot == out.slot:
                g.pending.appendleft(out.index)
                g.armed = True
                g.next_send_at = now + self.params.retry_guard_seconds
            return
        if out.kind == "repair":
            u = self._u
            if (u is None or u.closed or u.seed_slot != out.slot or out.position in u.updates
                    or out.position in u.abandoned):
                self.log("update_repair_obsolete", out.device_id, {"position": out.position})
                return                                  # its round is over: do not resend it
        if out.attempts >= self.params.max_downlink_attempts:
            self.log("downlink_dropped", out.device_id, {
                "kind": out.kind, "position": out.position, "attempts": out.attempts})
            return
        self._outbox.appendleft(out)

    def _clear_pending_repairs(self, why: str) -> None:
        """Rev 4.3 (section 4): the network server keeps a unicast downlink until the device's next
        uplink, so when a round closes every device that may still hold an UPDATE_REPAIR gets a
        down/replace with an empty list. A repair still in flight is cancelled with it."""
        self._outbox = deque(o for o in self._outbox if o.kind != "repair")
        if self._inflight is not None and self._inflight.kind == "repair":
            self._inflight = None
        pending, self._repair_pending = self._repair_pending, {}
        if self.publish_replace is None:
            return
        for device_id in sorted(pending):
            self.publish_replace(device_id, None, "")
            self.counters["update_repair_queue_cleared"] += 1
            self.log("update_repair_queue_cleared", device_id, {"why": why, "correlation_id": pending[device_id]})

    def _pump(self, now: float) -> None:
        # One downlink in flight across the whole gateway: it is half duplex.
        if self._inflight is not None:
            return
        g = self._g
        if self._outbox:
            if self.stage in (Stage.IDLE, Stage.WAITING_START):
                # no phase is being delivered: only RUN_CONTROL goes out (UPDATE repairs wait)
                out = next((o for o in self._outbox if o.kind == "control"), None)
                if out is None:
                    return
                self._outbox.remove(out)
            else:
                out = self._outbox.popleft()
            out.attempts += 1
            out.deadline = now + self.params.inflight_downlink_timeout_seconds
            self._inflight = out
            try:
                if out.kind == "repair" and self.publish_replace is not None:
                    # rev 4.3: down/replace, so at most one repair is pending per device
                    self._repair_pending[out.device_id] = out.correlation
                    self.publish_replace(out.device_id, out.frame, out.correlation)
                else:
                    self.publish(out.device_id, out.frame, out.correlation)
            except Exception:
                self._inflight = None
                self._repair_pending.pop(out.device_id, None)
                out.attempts -= 1
                self._outbox.appendleft(out)
                raise
            self.log("update_repair_queued" if out.kind == "repair" else "run_control_queued",
                     out.device_id, {"position": out.position, "application_bytes": len(out.frame),
                                     "attempt": out.attempts})
            return
        if (self.stage not in (Stage.DOWNLINK, Stage.UPLOAD) or g is None or not g.armed
                or not g.pending or now < g.next_send_at):
            return
        index = g.pending.popleft()
        g.attempts[index] += 1
        if g.attempts[index] > self.params.max_downlink_attempts:
            self._fail(f"{self.multicast_device_id} fragment {index} exceeded the downlink retry limit")
            return
        correlation = f"v4-s{g.slot}-f{index}-a{g.attempts[index]}"
        out = Outbound(kind="global", device_id=self.multicast_device_id, frame=g.frames[index],
                       correlation=correlation, index=index, attempts=g.attempts[index],
                       slot=g.slot,
                       deadline=now + self.params.inflight_downlink_timeout_seconds)
        # Claim the slot BEFORE publishing: down/sent can be delivered on the network thread
        # while publish() is still on the stack.
        self._inflight = out
        try:
            self.publish(out.device_id, out.frame, correlation)
        except Exception:
            self._inflight = None
            g.pending.appendleft(index)
            g.attempts[index] -= 1
            g.next_send_at = now + self.params.retry_guard_seconds
            raise
        g.fragments_queued += 1
        self.log("global_fragment_queued", out.device_id, {
            "phase_index": g.phase_index, "round": g.round, "slot": g.slot, "fragment": index,
            "count": GLOBAL_FRAGMENTS, "application_bytes": len(out.frame),
            "attempt": g.attempts[index], "correlation_id": correlation})

    def _matches_inflight(self, device_id: str, correlation_ids) -> bool:
        """A down/sent or down/failed event belongs to the in-flight downlink only if it names its
        correlation id (rev 4.3: a delayed event for an older, replaced repair must not be credited to
        the newer one). An event naming no correlation id at all (``None`` or empty, e.g. a network
        server that does not echo them) falls back to matching by device, as before rev 4.3."""
        out = self._inflight
        if out is None or out.device_id != device_id:
            return False
        return not correlation_ids or out.correlation in set(correlation_ids)

    @_synchronized
    def handle_downlink_sent(self, device_id: str, correlation_ids=None) -> None:
        if not self._matches_inflight(device_id, correlation_ids):
            self.counters["downlink_sent_unmatched"] += 1
            self.log("downlink_sent_unmatched", device_id, {
                "correlation_ids": list(correlation_ids or ()),
                "inflight": None if self._inflight is None else self._inflight.correlation})
            return
        out = self._inflight
        self._inflight = None
        if out.kind == "repair":
            self._repair_pending.pop(device_id, None)
        now = self.clock()
        if out.kind == "repair":
            member = self.membership.get(self._dev_eui_or_none(device_id))
            if member is not None:
                member.counters["update_repair_downlinks_sent"] += 1
                member.counters["downlink_application_bytes_sent"] += len(out.frame)
            if self._u is not None:
                self._u.repair_downlinks_sent += 1
            self.log("update_repair_sent", device_id,
                     {"position": out.position, "application_bytes": len(out.frame)})
            return
        if out.kind == "control":
            self.counters["run_control_sent"] += 1
            if out.correlation.startswith("v4-end"):
                self._last_end_sent_t = now
            self.log("run_control_sent", device_id, {"application_bytes": len(out.frame)})
            return
        g = self._g
        self.counters["global_fragments_sent"] += 1
        self.counters["global_application_bytes_sent"] += len(out.frame)
        if g is None or g.ended or g.slot != out.slot:
            self.log("global_fragment_sent_for_finished_global", device_id,
                     {"fragment": out.index, "slot": out.slot})
            return
        g.fragments_sent += 1
        g.next_send_at = now
        self.log("global_fragment_sent", device_id, {"fragment": out.index, "round": g.round})
        if not g.pending:
            if not g.prompt_burst or g.burst_ended_at is None:
                g.burst_ended_at = now          # a prompt burst leaves the periodic repair timer alone
            if self.stage is Stage.DOWNLINK:
                g.armed = False    # wait for reports before deciding on repairs

    @staticmethod
    def _dev_eui_or_none(device_id: str):
        try:
            return dev_eui_of_device_id(device_id)
        except ProtocolError:
            return None

    @_synchronized
    def handle_downlink_failed(self, device_id: str, reason: str, correlation_ids=None) -> None:
        if not self._matches_inflight(device_id, correlation_ids):
            self.counters["downlink_failed_unmatched"] += 1
            self.log("downlink_failed_unmatched", device_id, {
                "reason": reason, "correlation_ids": list(correlation_ids or ())})
            return
        out = self._inflight
        self._inflight = None
        if out.kind == "repair":
            self._repair_pending.pop(device_id, None)
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
    # Replay summary
    # ------------------------------------------------------------------
    def replay_summary(self) -> dict:
        """Per position: alarm, no alarm, or never finished (REPLAY_DONE).

        A position with no alarm is only "no_alarm" if its board said REPLAY_DONE for every
        phase whose final model it held; otherwise silence means the replay did not finish.
        """
        completed = set(self._completed_phases())
        positions = {}
        counts = Counter()
        owner = self.membership.position_owner
        done_by_position: dict = {}
        for (pos, ph), row in self.replay_done.items():
            done_by_position.setdefault(pos, {})[ph] = row
        for position in sorted(owner):
            member = self.membership.members[owner[position]]
            expected = sorted(i for i in member.held_final_phases if i in completed)
            if not expected:
                continue
            alarm_phases = sorted(self._alarm_phases.get(position, ()))
            by_phase = done_by_position.get(position, {})
            scored = sorted(by_phase)
            rows = [by_phase[ph] for ph in scored]
            latched = sorted(r["phase_index"] for r in rows if r["alarm_latched"])
            entry = {
                "board_id": member.board_id, "dev_eui": member.dev_eui,
                "phases_expected": expected, "replay_done": bool(rows),
                "phases_scored": scored,
                "phases_missing": [i for i in expected if i not in scored],
                "acquisitions_scored": sum(r["acquisitions_scored"] for r in rows),
                "phases_alarm_latched": latched,
                "alarm_messages": len(alarm_phases), "alarm_phase_indices": alarm_phases,
            }
            if not rows:
                status = "never_finished"
            elif entry["phases_missing"]:
                status = "incomplete_phases"
            elif latched or alarm_phases:
                status = "alarm"
            else:
                status = "no_alarm"
            entry["status"] = status
            entry["alarm_messages_lost"] = bool(set(latched) - set(alarm_phases))
            machine = self.plan.machine_of(position)
            if machine:
                entry["machine"] = machine
            positions[str(position)] = entry
            counts[status] += 1
        return {
            "run_id": self.run_id,
            "phases_completed": sorted(completed),
            "positions_expected": len(positions),
            "counts": dict(sorted(counts.items())),
            "alarm_messages_lost_positions": [
                int(p) for p, e in positions.items() if e["alarm_messages_lost"]],
            "positions": positions,
        }

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def _phase_masks(self, phase: PhaseState) -> dict:
        spec = phase.spec
        planned = list(spec.positions or ())
        positions = sorted(set(planned) | set(phase.positions_seen))
        owner = {}                                        # position -> DevEUI (current binding)
        for m in self.membership.members.values():
            for p in m.positions:
                owner[p] = m.dev_eui
        uplink, downlink, base_slot, base_round = [], [], [], []
        for mask in phase.masks:
            up, base = mask["uplink"], mask["base"]
            holders = phase.held_by_slot.get(mask["slot"], ())
            uplink.append([bool(up.get(str(p), False)) for p in positions])
            downlink.append([owner.get(p) in holders for p in positions])
            row = [base.get(str(p)) for p in positions]
            base_slot.append(row)
            base_round.append([None if s is None else phase.slot_round.get(int(s)) for s in row])
        return {
            "phase_index": spec.index, "name": spec.name,
            "rounds_in_phase": spec.rounds, "rounds_completed": len(phase.masks),
            "complete": phase.status == "complete", "status": phase.status,
            "participation": spec.participation().to_dict(),
            "positions": positions, "planned_positions": planned,
            "machines": ([self.plan.machine_of(p) for p in positions]
                         if self.plan.machine_order else None),
            "uplink": uplink, "downlink": downlink, "base_slot": base_slot,
            "first_slot": phase.first_slot, "base_round": base_round,
            "participants": [r["participants"] for r in phase.rounds],
            "slot_of_global_round": {str(r): s for s, r in sorted(phase.slot_round.items(),
                                                                  key=lambda kv: kv[1])},
        }

    def _write_masks(self) -> None:
        phases = []
        for ph in self.phases:
            masks = self._phase_masks(ph)
            if self.plan.mode != "planned":
                del masks["planned_positions"]
            if masks["machines"] is None:
                del masks["machines"]
            phases.append(masks)
        self.out.write_replay_masks({
            "schema": "tinyfl-replay-masks-v4",
            "run_id": self.run_id,
            "semantics": (
                "Row k is training round k+1. uplink[k][i]: the UPDATE of position positions[i] for "
                "round k+1 reached the server before the round closed. downlink[k][i]: the board "
                "hosting that position held the complete GLOBAL produced by round k+1 (acknowledged "
                "at the end of its delivery, or proved later by an UPDATE trained from it or a "
                "REPLAY_DONE). base_slot[k][i]: slot of the GLOBAL the UPDATE was trained from, null "
                "exactly where uplink is false; base_round[k][i] < k is a stale base the twin must "
                "replay (dropout.json stale-copy semantics). positions = planned positions plus every "
                "position seen in the phase; a position is absent (false/false) before it joins and "
                "after it departs."),
            "seed": self.plan.seed, "mode": self.plan.mode, "started_utc": self.started_utc,
            "phases": phases,
        })

    def _phase_summary(self, phase: PhaseState) -> dict:
        rounds = phase.rounds
        return {
            "phase_index": phase.spec.index, "name": phase.spec.name, "status": phase.status,
            "rounds_planned": phase.spec.rounds, "rounds_completed": len(rounds),
            "planned_positions": None if phase.spec.positions is None else len(phase.spec.positions),
            "positions_seen": sorted(phase.positions_seen),
            "updates_arrived_per_round": [len(r["arrived_positions"]) for r in rounds],
            "updates_expected_per_round": [len(r["expected_positions"]) for r in rounds],
            "updates_missing_total": sum(len(r["missing_positions"]) for r in rounds),
            "rounds_closed_by_timeout": sum(1 for r in rounds if r["close_reason"] == "timeout"),
            "update_fragments_received": sum(r["update_fragments_received"] for r in rounds),
            "update_repair_requests": sum(r["update_repair_requests"] for r in rounds),
            "stale_base_updates": sum(len(r["stale_positions"]) for r in rounds),
            "global_fragments_sent": sum(r.get("global_fragments_sent", 0) for r in rounds),
            "initial_global_crc32": phase.initial_global_crc,
            "final_global_crc32": phase.final_crc,
            "final_global_codes_sha256": phase.final_codes_sha256,
            "boards_holding_final": phase.boards_holding_final,
            "duration_s": None if phase.ended_t is None else round(phase.ended_t - phase.started_t, 3),
            "started_utc": phase.started_utc, "ended_utc": phase.ended_utc,
            "alarms": int(self.alarm_count.get(phase.spec.index, 0)),
        }

    def _write_summaries(self) -> None:
        self.out.write_phase_summaries([self._phase_summary(p) for p in self.phases])

    @_synchronized
    def finalize(self) -> None:
        """Rewrite the summary files (per phase, on failure, at the end)."""
        self._write_masks()
        self._write_summaries()
        self.out.write_replay_summary(self.replay_summary())
        self._write_boards()

    @_synchronized
    def summary(self) -> dict:
        return {
            "protocol": "v4", "mode": self.plan.mode, "seed": self.plan.seed,
            "stage": self.stage.value, "finished": self.finished, "failed": self.failed,
            "failure_reason": self.failure_reason, "run_id": self.run_id,
            "end_unconfirmed": list(self.end_unconfirmed), "end_sends": self.counters["end_sends"],
            "membership": self.membership.counts(),
            "boards": {
                m.dev_eui: {"board_id": m.board_id, "state": m.state.value,
                            "positions": list(m.positions), **dict(sorted(m.counters.items()))}
                for m in sorted(self.membership.members.values(), key=lambda x: x.dev_eui)
                if m.board_id is not None},
            "phases_summary": [self._phase_summary(p) for p in self.phases],
            "counters": dict(sorted(self.counters.items())),
            "alarms": int(sum(self.alarm_count.values())),
            "self_checks": self.self_check_count,
            "replay": self.replay_summary()["counts"],
            "multicast_group": self.multicast_device_id,
        }

    # ------------------------------------------------------------------
    # Checkpoint / resume
    # ------------------------------------------------------------------
    def _checkpoint(self) -> None:
        phase = self.phase
        data = {
            "version": CHECKPOINT_VERSION, "saved_utc": self.wall_clock(),
            "run_id": self.run_id, "plan": self.plan.to_dict(), "stage": self.stage.value,
            "failure_reason": self.failure_reason,
            "slot_counter": self.slot_counter, "started_utc": self.started_utc,
            "current_slot": None if self._g is None else self._g.slot,
            "phase_index": None if phase is None else phase.spec.index,
            "round": None if self._g is None else self._g.round,
            "fedadam": None, "phase": None,
            "finished_phases": [p.to_dict() for p in self.phases if p is not phase],
            "pending_round": self._pending_round,
            "membership": self._membership_with_liveness(),
            "repair_state": self._last_repair_state,
            "counters": dict(self.counters),
            "alarm_keys": sorted([k[0], k[1], v["index"], v["source"]] for k, v in self._alarms.items()),
            "alarm_count": {str(k): v for k, v in self.alarm_count.items()},
            "self_check_count": self.self_check_count,
            "replay_done": {f"{k[0]}:{k[1]}": v for k, v in self.replay_done.items()},
            "round_changes": self._round_changes,
            "replay_phases": self._replay_phases,
        }
        if phase is not None:
            server = phase.server
            data["fedadam"] = {"p": _enc_arrays(server.p), "m": _enc_arrays(server.m),
                               "v": _enc_arrays(server.v), "t": int(server.t),
                               "dtype": "float64"}
            data["phase"] = phase.to_dict(with_model=True)
        self.out.write_checkpoint(data)

    def _membership_with_liveness(self) -> dict:
        """Membership plus how long each board had been silent. The monotonic clock does not
        survive a restart, so on --resume every board counts as heard at resume time; the age is
        recorded for the operator only."""
        data = self.membership.to_dict()
        now = self.clock()
        for entry in data["members"]:
            member = self.membership.members[entry["dev_eui"]]
            entry["silent_seconds_at_checkpoint"] = (
                round(max(0.0, now - member.last_heard_t), 1) if member.hello_count else None)
        return data

    @_synchronized
    def checkpoint_now(self) -> None:
        """Operator stop (Ctrl-C): persist whatever stage the run is in."""
        self._checkpoint()

    @classmethod
    def resume(cls, checkpoint: dict, publish: Publisher, log: EventLogger,
               multicast_device_id: str, params: ControllerParams | None = None,
               outputs=None, is_rostered=None, fl_modules=None, research_root=None,
               clock=time.monotonic, wall_clock=None, publish_replace=None) -> "ControllerV4":
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                f"checkpoint version {checkpoint.get('version')!r} is not {CHECKPOINT_VERSION}")
        plan = RunPlan.from_dict(checkpoint["plan"])
        controller = cls(plan, publish, log, multicast_device_id, params, outputs, is_rostered,
                         fl_modules, research_root, run_id=checkpoint["run_id"], clock=clock,
                         wall_clock=wall_clock, publish_replace=publish_replace)
        controller._restore(checkpoint)
        return controller

    def _restore(self, ck: dict) -> None:
        now = self.clock()
        self._last_uplink_t = now          # the gateway's recent past is unknown: guard from the resume
        self.slot_counter = int(ck["slot_counter"])
        self.started_utc = ck.get("started_utc", "")
        self.failure_reason = ck.get("failure_reason", "")
        self.membership.restore(ck["membership"])
        for member in self.membership.members.values():
            member.ready = False
            member.last_heard_t = now
        self.counters = Counter(ck.get("counters", {}))
        self._alarms = {(k[0], k[1]): {"index": k[2], "source": k[3]} for k in ck.get("alarm_keys", [])}
        self._alarm_phases = {}
        for (position, phase_index), v in self._alarms.items():
            if v["source"] == "alarm":
                self._alarm_phases.setdefault(position, set()).add(phase_index)
        self.alarm_count = Counter({int(k): v for k, v in ck.get("alarm_count", {}).items()})
        self.self_check_count = int(ck.get("self_check_count", 0))
        self.replay_done = {tuple(int(x) for x in k.split(":")): v
                            for k, v in ck.get("replay_done", {}).items()}
        self._round_changes = list(ck.get("round_changes", []))
        self._last_repair_state = ck.get("repair_state", {})
        self._replay_phases = list(ck.get("replay_phases", []))
        self.phases = []
        for entry in ck.get("finished_phases", []):
            self.phases.append(self._phase_from_dict(entry, now))
        stage = Stage(ck["stage"])
        self.log("resume_loaded", "server", {
            "run_id": self.run_id, "stage": stage.value, "phase_index": ck.get("phase_index"),
            "round": ck.get("round"), "slot_counter": self.slot_counter})
        if ck.get("phase") is None:
            # Never started: wait for boards again.
            self.stage = Stage.WAITING_START
            return
        phase = self._phase_from_dict(ck["phase"], now, with_model=True)
        fed = ck["fedadam"]
        server = self.twin_server.Server(self.fl.init(list(LAYERS), self.plan.seed))
        server.p, server.m, server.v = (_dec_arrays(fed["p"]), _dec_arrays(fed["m"]),
                                        _dec_arrays(fed["v"]))
        server.t = int(fed["t"])
        phase.server = server
        self.phases.append(phase)
        self.phase = phase
        if stage in (Stage.DONE, Stage.FAILED, Stage.COLLECTING, Stage.ENDING) or phase.status != "running":
            # Past the last phase: re-enter collection (REPLAY is repeated until REPLAY_DONE).
            if stage is Stage.DONE:
                self.stage = Stage.DONE
                return
            if stage is Stage.FAILED:
                self.stage = Stage.FAILED
                return
            self._begin_collecting(now)
            if stage is Stage.ENDING:
                # END is repeated until the boards go quiet: start that again (the guard counts
                # from the resume, the linger and quiet timers restart)
                self._begin_ending("resume", now)
            return
        self._pending_round = ck.get("pending_round")
        slot = int(ck["current_slot"])
        payload = phase.globals_payload[slot]
        decoded = decode_global(payload)
        model = QuantisedModel(decoded.model.codes, decoded.model.scales)
        g = GlobalState(phase_index=phase.spec.index, round=decoded.round, slot=slot,
                        model=model, payload=payload, crc=decoded.crc32,
                        frames=fragment_global(payload, slot))
        self._g = g
        self._log_global(g)                  # de-duplicated by slot if it is already on disk
        self._begin_downlink(g, now, resumed=True)

    def _phase_from_dict(self, d: dict, now: float, with_model: bool = False) -> PhaseState:
        spec = PhaseSpec.from_dict(d["spec"])
        phase = PhaseState(spec=spec, server=None, started_t=now, started_utc=d["started_utc"],
                           first_slot=d["first_slot"], rounds=list(d["rounds"]),
                           masks=list(d["masks"]),
                           slot_round={int(k): v for k, v in d["slot_round"].items()},
                           held_by_slot={int(k): set(v) for k, v in d.get("held_by_slot", {}).items()},
                           positions_seen=set(d["positions_seen"]),
                           initial_global_crc=d["initial_global_crc"], final_crc=d["final_crc"],
                           final_codes_sha256=d["final_codes_sha256"],
                           boards_holding_final=list(d["boards_holding_final"]),
                           status=d["status"], ended_utc=d.get("ended_utc", ""))
        if d.get("duration_s") is not None:
            phase.ended_t = now
            phase.started_t = now - d["duration_s"]
        if with_model:
            for slot, b64 in d["globals"].items():
                payload = base64.b64decode(b64)
                phase.globals_payload[int(slot)] = payload
                phase.globals_deq[int(slot)] = decode_global(payload).model.dequantize()
        return phase
