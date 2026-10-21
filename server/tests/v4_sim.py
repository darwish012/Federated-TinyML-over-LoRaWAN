"""A tiny discrete-time network for exercising ControllerV4 with fake boards.

FakeBoards behave the way PROTOCOL_V4 section 5 says a board does: say hello,
announce Class C readiness (awaited slot 0xFFFF = "any phase start"), receive the
multicast GLOBAL, acknowledge it, train (a deterministic pseudo-delta per position),
send one 3-fragment UPDATE per hosted position back to back in ascending position
order, then announce readiness for the next slot. They obey UPDATE_REPAIR by resending
exactly the named fragments from their outbox, and RUN_CONTROL REPLAY / END.

Run adoption is modelled literally (PROTOCOL_V4 3.2 / 5.1, rev 4.1): a board adopts a run (archive,
clear run state, store the new run_id) on RUN_CONTROL ADOPT, on any GLOBAL of round 0, or on
any CRC-valid GLOBAL when it holds no run (stored_run_id == 0) or its stored run is complete.
A GLOBAL of another run with round > 0 is otherwise ignored.

There is no radio, so a "downlink sent" event is confirmed immediately.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sys

import numpy as np

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from controller_v4 import ControllerParams, ControllerV4, Stage  # noqa: E402
from plan_v4 import RunPlan  # noqa: E402
from protocol import validate_crc  # noqa: E402
import protocol_v4 as p4  # noqa: E402
from protocol_v4 import (  # noqa: E402
    CMD_ADOPT,
    CMD_END,
    GLOBAL_BYTES,
    GLOBAL_FRAGMENTS,
    HELLO_SD_MOUNTED,
    HELLO_SELF_CHECK_PASSED,
    MSG_GLOBAL_FRAGMENT,
    MSG_RUN_CONTROL,
    MSG_UPDATE_REPAIR,
    N_PARAMS,
    NO_SLOT,
    decode_frame,
    decode_global,
    decode_run_control,
    decode_update_repair,
    encode_alarm,
    encode_board_hello,
    encode_class_c_ready,
    encode_global_status,
    encode_replay_done,
    encode_update,
    fragment_update,
    quantize_model,
)
from run_outputs_v4 import MemoryOutputs  # noqa: E402
from study_plan import load_fl_modules, load_machine_order  # noqa: E402

GROUP = "tinyfl-v4-global"
RUN_ID = 0x4A5B6C01
HELLO_OK = HELLO_SD_MOUNTED | HELLO_SELF_CHECK_PASSED


def dev_eui_for(board_id: int) -> str:
    return f"70B3D57ED0{board_id:06X}"


def device_id_for(board_id: int) -> str:
    return "eui-" + dev_eui_for(board_id).lower()


def pseudo_delta(position: int, phase_index: int, round_: int) -> np.ndarray:
    """What a machine 'learned': deterministic, machine-specific, sane magnitude."""
    rng = np.random.default_rng([position, phase_index, round_, 7])
    return rng.normal(0.0, 0.05, N_PARAMS)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def thesis_boards(n_boards: int = 10, n_positions: int = 28) -> dict:
    """position p lives on board p mod n + 1 (v3's geometry)."""
    out: dict = {b: [] for b in range(1, n_boards + 1)}
    for p in range(n_positions):
        out[p % n_boards + 1].append(p)
    return {b: tuple(ps) for b, ps in out.items() if ps}


RX_WINDOW_S = 2.0               # a Class A board hears no multicast from its uplink until ~2 s after it
IDLE_READY_PERIOD_S = 180.0     # firmware: periodic CLASS_C_READY (or HELLO) while idle in Class C


def airtime(n: int) -> float:
    """SF7 / 125 kHz / CR4/5 airtime of an application payload of n bytes (13 B LoRaWAN overhead)."""
    payload_symbols = 8 + max(math.ceil((8 * (n + 13) - 4 * 7 + 28 + 16) / 28) * 5, 0)
    return (8 + 4.25 + payload_symbols) * 128 / 125000


class FakeBoard:
    def __init__(self, board_id: int, positions, sim: "Sim", flags: int = HELLO_OK,
                 stored_run_id: int = 0, run_complete: bool = False) -> None:
        self.board_id = board_id
        self.positions = tuple(sorted(positions))
        self.device_id = device_id_for(board_id)
        self.dev_eui = dev_eui_for(board_id)
        self.sim = sim
        self.flags = flags | (p4.HELLO_RUN_COMPLETE if run_complete else 0)
        self.run_complete = run_complete
        self.stored_run_id = stored_run_id
        self.adoptions = 0
        self.powered = True
        self.global_parts: dict[int, dict[int, bytes]] = {}
        self.done_slots: set = set()
        self.held: dict[int, object] = {}               # slot -> GlobalModel
        self.held_slot = 0
        self.finals: dict[int, object] = {}             # phase_index -> GlobalModel
        self.final_active: dict[int, tuple] = {}        # phase_index -> positions active in that phase
        self.outbox: dict[int, tuple] = {}              # position -> (seed_slot, frames)
        self.send_count: Counter = Counter()
        self.status_due: dict[int, float] = {}
        self.replay_seen: set = set()
        self.ended = False
        self.phase_index = -1
        self.round_ = 0
        # rev 4.3 radio model
        self.windows: list[tuple] = []                  # (start, end) intervals deaf to multicast
        self.next_tx = 0.0                              # duty cycle: earliest start of the next uplink
        self.last_uplink_t = sim.clock.now
        self.joined = False                             # has said hello at least once
        self.heard_run = False                          # CRC-valid GLOBAL / RUN_CONTROL since boot
        self.hello_count = 0
        self.epoch = 0                                  # bumps on power-off: queued frames never leave
        self.ready_pending = False
        self.active_positions: tuple = ()

    # -- uplink ---------------------------------------------------------------
    def uplink(self, frame: bytes, kind: str = "other", position: int = -1,
               fragment: int = -1) -> None:
        if not self.powered or self.ended:
            return                                      # after END a board sends nothing for this run
        if kind == "update":
            self.send_count[(position, fragment)] += 1
            if self.sim.uplink_drop(self.board_id, position, fragment,
                                    self.send_count[(position, fragment)],
                                    self.phase_index, self.round_):
                self.sim.uplinks_dropped += 1
                return
        sim = self.sim
        now = sim.clock.now
        if sim.duty_cycle:                              # 1 % duty cycle: 100 x airtime between frames
            toa = airtime(len(frame))
            start = max(now, self.next_tx)
            self.next_tx = start + toa * 100.0
            end = start + toa
        else:
            start = end = now
        self.windows.append((start, end + RX_WINDOW_S))
        self.last_uplink_t = max(self.last_uplink_t, end)
        epoch = self.epoch
        if end <= now:
            sim.controller.handle_uplink(self.device_id, 10, frame)
        else:
            def deliver(frame=frame, epoch=epoch):
                if self.powered and self.epoch == epoch:
                    sim.controller.handle_uplink(self.device_id, 10, frame)
            sim.schedule(end, deliver)

    def _ready_after_window(self) -> None:
        """Firmware: when the Class A windows close the board re-enters Class C and announces
        CLASS_C_READY (awaiting the next GLOBAL it does not hold), ~0.5 s later (testbed: 2.5 s)."""
        if self.ready_pending:
            return
        self.ready_pending = True
        epoch = self.epoch
        when = max((e for _, e in self.windows), default=self.sim.clock.now) + 0.5

        def send():
            self.ready_pending = False
            if self.powered and not self.ended and self.epoch == epoch:
                self.ready(self.held_slot + 1 if self.held_slot else NO_SLOT)
        self.sim.schedule(when, send)

    def in_window(self, now: float) -> bool:
        """Transmitting or inside RX1/RX2: a multicast now is missed."""
        self.windows = [w for w in self.windows if w[1] > now]
        return any(s <= now < e for s, e in self.windows)

    def send_hello(self) -> None:
        self.hello_count += 1
        if self.sim.hello_drop(self.board_id, self.hello_count):
            self.sim.hellos_dropped += 1
            self.last_uplink_t = self.sim.clock.now
            return
        self.uplink(encode_board_hello(self.board_id, 1, self.flags, self.stored_run_id,
                                       self.positions))

    def hello(self) -> None:
        self.powered = True                       # a board that says hello is on the air
        self.joined = True
        self.heard_run = False
        self.send_hello()
        self.ready(NO_SLOT)

    def ready(self, awaited: int) -> None:
        self.uplink(encode_class_c_ready(self.stored_run_id, awaited))

    def power_off(self) -> None:
        self.powered = False
        self.epoch += 1                                   # frames still queued for the air die with it

    def power_on(self, hello: bool = True) -> None:
        self.powered = True
        self.status_due.clear()
        if hello:
            self.hello()

    # -- downlink -----------------------------------------------------------------
    def receive_global(self, frame_bytes: bytes) -> None:
        if not self.powered:
            return                                  # (an ended board still hears a GLOBAL of another run)
        if self.in_window(self.sim.clock.now):
            self.sim.missed_in_window["global"] += 1
            self._ready_after_window()
            return
        frame = decode_frame(frame_bytes)
        slot = frame.round_id
        parts = self.global_parts.setdefault(slot, {})
        parts[frame.fragment_index] = frame.data
        self.status_due.setdefault(slot, self.sim.clock.now + self.sim.idle_timeout)
        if len(parts) < GLOBAL_FRAGMENTS or slot in self.done_slots:
            return
        payload = b"".join(parts[i] for i in range(GLOBAL_FRAGMENTS))
        validate_crc(payload, GLOBAL_BYTES)
        model = decode_global(payload)
        if self.ended and model.run_id == self.stored_run_id:
            return                                  # after END: silent for this run
        self.done_slots.add(slot)
        self.status_due.pop(slot, None)
        if model.run_id != self.stored_run_id:
            # 5.1: adopt on round 0, or on any GLOBAL when no run is held / the stored run is complete
            if not (model.round == 0 or self.stored_run_id == 0 or self.run_complete):
                return
            self.adopt(model.run_id)
        if slot <= self.held_slot:
            return                                  # not newer: ignored silently
        self.held[slot] = model
        self.held_slot = slot
        self.heard_run = True
        # rev 4.3: only the positions the GLOBAL marks as taking part are active in this phase
        # (a rev 4.2 board, legacy_rev42, uploads and replays everything it hosts)
        if self.sim.legacy_rev42:
            active = self.positions
        else:
            active = tuple(p for p in self.positions if model.participation.covers(p))
        self.active_positions = active
        self.uplink(encode_global_status(self.stored_run_id, slot, 0))
        if model.final:
            self.finals[model.phase_index] = model
            self.final_active[model.phase_index] = active
            self.ready(slot + 1)
            return
        if self.sim.crash(self.board_id, model.phase_index, model.round + 1):
            return                                  # took the GLOBAL, then went silent
        self.train_and_upload(model.phase_index, model.round + 1, slot)

    def adopt(self, run_id: int) -> None:
        """Archive and clear run state, store the new run_id (section 5.1)."""
        self.stored_run_id = run_id
        self.ended = False
        self.run_complete = False
        self.flags &= ~p4.HELLO_RUN_COMPLETE
        self.held.clear()
        self.held_slot = 0
        self.finals.clear()
        self.final_active.clear()
        self.outbox.clear()
        self.done_slots.clear()
        self.global_parts.clear()
        self.adoptions += 1

    def train_and_upload(self, phase_index: int, round_: int, seed_slot: int) -> None:
        self.phase_index, self.round_ = phase_index, round_
        self.send_count.clear()
        base_slot = seed_slot
        back = self.sim.stale_base(self.board_id, phase_index, round_)
        if back and (seed_slot - back) in self.held:
            base_slot = seed_slot - back
        self.outbox = {}
        for position in self.active_positions:
            delta, loss = self.sim.make_delta(self, position, phase_index, round_,
                                              self.held[base_slot])
            key = (phase_index, round_, position)
            self.sim.sent[key] = delta
            self.sim.sent_base[key] = base_slot
            self.sim.sent_true[key] = pseudo_delta(position, phase_index, round_)
            payload = encode_update(position, self.stored_run_id, phase_index, round_,
                                    base_slot, 100 + position, delta, loss)
            frames = fragment_update(payload, seed_slot)
            self.outbox[position] = (seed_slot, frames)
            for index, frame in enumerate(frames):
                self.uplink(frame, "update", position, index)
        self.ready(seed_slot + 1)

    def receive_repair(self, frame_bytes: bytes) -> None:
        if not self.powered:
            return
        frame = decode_frame(frame_bytes)
        assert frame.message_type == MSG_UPDATE_REPAIR
        req = decode_update_repair(frame)
        self.sim.repairs.append((self.board_id, req.position, req.missing_mask))
        entry = self.outbox.get(req.position)
        if entry is None or entry[0] != frame.round_id or req.run_id != self.stored_run_id:
            return
        seed_slot, frames = entry
        for index in range(GLOBAL_FRAGMENTS):
            if req.missing_mask >> index & 1:
                self.uplink(frames[index], "update", req.position, index)
        self.ready(seed_slot + 1)

    def receive_run_control(self, frame_bytes: bytes) -> None:
        if not self.powered or not self.joined:
            return                                   # off the air, or not booted yet
        if self.in_window(self.sim.clock.now):
            self.sim.missed_in_window["run_control"] += 1
            return
        frame = decode_frame(frame_bytes)
        control = decode_run_control(frame)
        if control.command == CMD_ADOPT:
            # rev 4.3 (3.4): every ADOPT makes the board send one HELLO (after adopting when the run
            # differs; also for the run it already holds), unless that run is its own and complete
            if control.run_id == self.stored_run_id and (self.ended or self.run_complete):
                return
            self.sim.adopts_heard[self.board_id] += 1
            if control.run_id != self.stored_run_id:
                self.adopt(control.run_id)
            self.heard_run = True
            if not self.sim.legacy_rev42:
                self.joined = True
                self.sim.adopt_hellos[self.board_id] += 1
                self.send_hello()
            return
        if self.ended:
            return                                   # a finished board is silent in Class C
        if control.run_id != self.stored_run_id:
            return                                   # boards ignore RUN_CONTROL for another run
        self.heard_run = True
        if control.command == CMD_END:
            self.ended = True                        # from now on: no uplink for this run
            self.run_complete = True
            self.flags |= p4.HELLO_RUN_COMPLETE
            self.sim.end_heard.append((self.board_id, self.sim.clock.now))
            return
        phases = [p for p in control.phases if p in self.finals]
        if not phases:
            return
        self.sim.replay_commands_heard[self.board_id] += 1
        if self.sim.replay_ignore(self.board_id, self.sim.replay_commands_heard[self.board_id]):
            return
        for p in phases:                             # one ALARM and one REPLAY_DONE per active (position, phase)
            for position in self.final_active.get(p, self.positions):
                scored, first_alarm = self.sim.replay_result(self, position, p, self.finals[p])
                if first_alarm is not None and not self.sim.alarm_drop(self.board_id, position, p):
                    self.uplink(encode_alarm(self.stored_run_id, position, p, first_alarm, 9.5, 3.0))
                self.uplink(encode_replay_done(
                    self.stored_run_id, position, p, scored, int(first_alarm is not None),
                    p4.NO_ALARM_INDEX if first_alarm is None else first_alarm))

    def periodic(self, now: float) -> None:
        if not self.powered or self.ended:
            return
        if self.joined and now - self.last_uplink_t >= IDLE_READY_PERIOD_S:
            # idle in Class C: the periodic CLASS_C_READY; until the run is heard (rev 4.3) a HELLO instead
            if not self.heard_run:
                self.send_hello()
            else:
                self.ready(self.held_slot + 1 if self.held_slot else NO_SLOT)
            self.last_uplink_t = now
        for slot, due in list(self.status_due.items()):
            if now >= due and slot not in self.done_slots:
                parts = self.global_parts.get(slot, {})
                missing = sum(1 << i for i in range(GLOBAL_FRAGMENTS) if i not in parts)
                self.status_due[slot] = now + self.sim.idle_timeout
                self.uplink(encode_global_status(self.stored_run_id, slot, missing))


DEFAULT_PARAMS = dict(
    class_c_guard_seconds=2.0, retry_guard_seconds=1.0, max_downlink_attempts=5,
    max_repair_rounds=3, max_global_repair_rounds=3, multicast_report_window_seconds=30.0,
    inflight_downlink_timeout_seconds=20.0, ready_wait_seconds=60.0, round_timeout_seconds=600.0,
    idle_phase_timeout_seconds=900.0, idle_retry_seconds=30.0, post_run_collect_seconds=300.0,
    replay_repeat_seconds=60.0, end_repeats=1, resume_repair_delay_seconds=5.0,
)


class Sim:
    def __init__(self, plan: RunPlan, boards: dict | None = None, stored_runs: dict | None = None,
                 uplink_drop=None, downlink_drop=None, crash=None, stale_base=None,
                 alarm_for=None, replay_ignore=None, idle_timeout: float = 120.0,
                 outputs=None, is_rostered=None, run_id: int = RUN_ID, start: bool = True,
                 sent_drop=None, duty_cycle: bool = False, legacy_rev42: bool = False,
                 hello_drop=None, alarm_drop=None, silent=None, **param_overrides) -> None:
        """``duty_cycle``: boards queue uplinks under the 1 % duty cycle (realistic round times).
        ``legacy_rev42``: boards ignore the participation mask and upload/replay every hosted position.
        ``hello_drop(board, n)`` / ``alarm_drop(board, position, phase)``: lose the n-th HELLO / an ALARM.
        ``silent``: board ids that completed an earlier run (``stored_runs`` says which; default 0x0D1DD00D)
        and have not rebooted: they send nothing and only answer an ADOPT (rev 4.3).
        Every board is deaf to multicast while transmitting and for ~2 s after each uplink (RX windows)."""
        self.plan = plan
        self.duty_cycle = duty_cycle
        self.legacy_rev42 = legacy_rev42
        self.hello_drop = hello_drop or (lambda board, n: False)
        self.alarm_drop = alarm_drop or (lambda board, position, phase: False)
        self.hellos_dropped = 0
        self.window_hits: list = []                 # (slot, board, t): a first multicast reached a board in its windows
        self.run_control_times: list = []           # (command, t) of every RUN_CONTROL multicast
        self.end_in_window: list = []               # (t, boards still in their windows) per END multicast
        self.missed_in_window: Counter = Counter()
        self.end_heard: list = []                   # (board, time) of every END a board received
        self.t_ending = None                        # simulated time the controller entered ENDING
        self.t_collecting = None
        board_map = boards if boards is not None else thesis_boards()
        # the run starts when every board of the scenario has said hello unless a test says otherwise
        param_overrides.setdefault("min_boards", len(board_map))
        self.sent_drop = sent_drop or (lambda device_id, frame, n: False)
        self.downlinks_seen = 0
        self.clock = Clock()
        self.published: list[tuple] = []
        self.events: list[tuple] = []
        self.event_t: list[float] = []
        self.cleared: list = []                     # (t, device) of every down/replace with an empty list
        self.replaced: list = []                    # (t, device, correlation) of every repair sent by down/replace
        self.out = outputs if outputs is not None else MemoryOutputs()
        self.sent: dict = {}
        self.sent_base: dict = {}
        self.sent_true: dict = {}
        self.repairs: list = []
        self.uplinks_dropped = 0
        self.idle_timeout = idle_timeout
        self.uplink_drop = uplink_drop or (lambda *a: False)
        self.downlink_drop = downlink_drop or (lambda *a: False)
        self.crash = crash or (lambda *a: False)
        self.stale_base = stale_base or (lambda *a: None)
        self.alarm_for = alarm_for or (lambda position, phase: False)
        self.replay_ignore = replay_ignore or (lambda board, n: False)
        self.replay_commands_heard: Counter = Counter()
        self.adopts_heard: Counter = Counter()
        self.adopt_hellos: Counter = Counter()      # HELLOs sent because an ADOPT was heard
        self.global_transmissions: Counter = Counter()
        self.run_controls: list = []
        self.is_rostered = is_rostered
        self.run_id = run_id
        params = dict(DEFAULT_PARAMS)
        params.update(param_overrides)
        self.params = ControllerParams(**params)
        self._fl = load_fl_modules()
        self._order = load_machine_order()[0]
        self.controller = self.make_controller()
        self.boards: dict[int, FakeBoard] = {}
        for board_id, positions in board_map.items():
            self.boards[board_id] = FakeBoard(
                board_id, positions, self, stored_run_id=(stored_runs or {}).get(board_id, 0))
        self.silent = set(silent or ())
        for board_id in self.silent:
            board = self.boards[board_id]
            if not board.stored_run_id:
                board.stored_run_id = 0x0D1DD00D
            board.ended = board.run_complete = True
            board.joined = True
            board.flags |= p4.HELLO_RUN_COMPLETE
        self._cursor = 0
        self._scheduled: list = []
        self.started = False
        if not start:
            self.started = True

    def _log(self, e, d, x) -> None:
        self.events.append((e, d, x))
        self.event_t.append(self.clock.now)

    def event_times(self, name: str, **where) -> list:
        """Simulated times of the events called ``name`` whose details contain ``where`` (a value that
        is a str matches by prefix)."""
        def ok(x):
            return all(str(x.get(k, "")).startswith(v) if isinstance(v, str) else x.get(k) == v
                       for k, v in where.items())
        return [t for (e, _, x), t in zip(self.events, self.event_t) if e == name and ok(x)]

    def _publish_replace(self, device_id, frame, correlation) -> None:
        """down/replace: a frame replaces the device's pending unicast downlink (and is delivered like
        any other repair), ``None`` clears it."""
        if frame is None:
            self.cleared.append((self.clock.now, device_id))
            return
        self.replaced.append((self.clock.now, device_id, correlation))
        self.published.append((device_id, frame, correlation))

    def make_controller(self, outputs=None) -> ControllerV4:
        return ControllerV4(
            self.plan, publish=lambda d, f, c: self.published.append((d, f, c)),
            publish_replace=self._publish_replace,
            log=self._log,
            multicast_device_id=GROUP, params=self.params,
            outputs=outputs if outputs is not None else self.out,
            is_rostered=self.is_rostered, fl_modules=self._fl, run_id=self.run_id,
            clock=self.clock, wall_clock=lambda: "2026-10-01T00:00:00+00:00")

    def resume_controller(self, checkpoint: dict, outputs=None) -> ControllerV4:
        """Replace the controller by one rebuilt from a checkpoint (the old one 'crashed')."""
        self.published_dropped = len(self.published) - self._cursor
        self._cursor = len(self.published)       # anything in flight died with the server
        self.controller = ControllerV4.resume(
            checkpoint, publish=lambda d, f, c: self.published.append((d, f, c)),
            publish_replace=self._publish_replace,
            log=self._log,
            multicast_device_id=GROUP, params=self.params,
            outputs=outputs if outputs is not None else self.out,
            is_rostered=self.is_rostered, fl_modules=self._fl,
            clock=self.clock, wall_clock=lambda: "2026-10-01T00:00:00+00:00")
        return self.controller

    # -- what the boards compute (override in a subclass to run real learning) ------------
    def make_delta(self, board: FakeBoard, position: int, phase_index: int, round_: int,
                   base_model) -> tuple:
        """(int8 delta QuantisedModel, train_loss) of one position's local training."""
        return quantize_model(pseudo_delta(position, phase_index, round_)), 0.01 * (position + 1)

    def replay_result(self, board: FakeBoard, position: int, phase_index: int,
                      final_model) -> tuple:
        """(acquisitions scored, first alarm acquisition index or None) of a REPLAY pass."""
        return 5000, (1234 + position if self.alarm_for(position, phase_index) else None)

    # -- helpers -------------------------------------------------------------------
    def names(self) -> list[str]:
        return [e for e, _, _ in self.events]

    def at(self, t_seconds: float, fn) -> None:
        """Run ``fn()`` once the simulated clock reaches start + t_seconds."""
        self.schedule(1000.0 + t_seconds, fn)

    def schedule(self, when: float, fn) -> None:
        self._seq = getattr(self, "_seq", 0) + 1
        self._scheduled.append((when, self._seq, fn))
        self._scheduled.sort(key=lambda x: (x[0], x[1]))

    def start(self, only=None) -> None:
        """Power up and HELLO the boards in ``only`` (default: all); the others stay off the air
        until a test calls their hello()."""
        for board in self.boards.values():
            if board.board_id in self.silent:
                continue                            # completed, not rebooted: waits for an ADOPT
            if only is None or board.board_id in only:
                board.hello()
            else:
                board.powered = False

    def step(self, dt: float = 1.0) -> None:
        while self._scheduled and self._scheduled[0][0] <= self.clock.now:
            _, _, fn = self._scheduled.pop(0)
            fn()
        self.controller.tick()
        stage = self.controller.stage
        if stage is Stage.COLLECTING and self.t_collecting is None:
            self.t_collecting = self.clock.now
        if stage in (Stage.ENDING, Stage.DONE) and self.t_ending is None:
            self.t_ending = self.clock.now
        while self._cursor < len(self.published):
            device_id, frame, correlation = self.published[self._cursor]
            self._cursor += 1
            self.downlinks_seen += 1
            if not self.sent_drop(device_id, frame, self.downlinks_seen):
                self.controller.handle_downlink_sent(device_id, [correlation])
            if device_id == GROUP:
                decoded = decode_frame(frame)
                if decoded.message_type == MSG_GLOBAL_FRAGMENT:
                    key = (decoded.round_id, decoded.fragment_index)
                    self.global_transmissions[key] += 1
                    count = self.global_transmissions[key]
                    g = self.controller._g
                    if count == 1 and decoded.fragment_index == 0 and g is not None:
                        # rev 4.3: the first multicast of a round never reaches a required board that is
                        # still transmitting or inside its RX windows
                        for dev in g.required:
                            board = next(b for b in self.boards.values() if b.dev_eui == dev)
                            if board.powered and board.in_window(self.clock.now):
                                self.window_hits.append((decoded.round_id, board.board_id, self.clock.now))
                    for board in list(self.boards.values()):
                        if not self.downlink_drop(board.board_id, decoded.round_id,
                                                  decoded.fragment_index, count):
                            board.receive_global(frame)
                else:
                    assert decoded.message_type == MSG_RUN_CONTROL
                    rc = decode_run_control(decoded)
                    self.run_controls.append(rc)
                    self.run_control_times.append((rc.command, self.clock.now))
                    if rc.command == CMD_END:
                        talking = [b.board_id for b in self.boards.values()
                                   if b.powered and not b.ended and b.in_window(self.clock.now)]
                        self.end_in_window.append((self.clock.now, talking))
                    for board in list(self.boards.values()):
                        board.receive_run_control(frame)
            else:
                board = next((b for b in self.boards.values() if b.device_id == device_id), None)
                if board is not None:
                    board.receive_repair(frame)
        self.clock.now += dt
        for board in list(self.boards.values()):
            board.periodic(self.clock.now)

    def run_until(self, predicate, max_steps: int = 40000) -> None:
        for _ in range(max_steps):
            if predicate():
                return
            self.step()
        raise AssertionError(
            f"did not converge; stage={self.controller.stage} "
            f"failure={self.controller.failure_reason!r} status={self.controller.status()} "
            f"last events={self.names()[-8:]}")

    def run_to_collecting(self, max_steps: int = 40000) -> None:
        self.start()
        self.run_until(lambda: self.controller.stage in (Stage.COLLECTING, Stage.DONE, Stage.FAILED),
                       max_steps)

    def run_to_done(self, max_steps: int = 60000) -> None:
        self.start()
        self.run_until(lambda: self.controller.stage in (Stage.DONE, Stage.FAILED), max_steps)

    # -- results ---------------------------------------------------------------------
    def masks(self, phase_index: int) -> dict:
        """The phase object of replay_masks.json (rev 4.1: ``phases`` is a list)."""
        return next(p for p in self.out.replay_masks["phases"] if p["phase_index"] == phase_index)

    def logged_global_codes(self, phase_index: int) -> list:
        rows = sorted((g for g in self.out.global_models if g["phase_index"] == phase_index),
                      key=lambda g: g["round"])
        return [np.array(g["codes"], dtype=np.int8) for g in rows]

    def round_records(self, phase_index: int) -> list:
        return [r for r in self.out.records("round") if r["phase_index"] == phase_index]
