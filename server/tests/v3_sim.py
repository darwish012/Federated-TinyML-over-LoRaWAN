"""A tiny discrete-time network for exercising HardwareExperimentController.

FakeBoards behave the way PROTOCOL_V3 says a board does: say hello, announce
Class C readiness, receive the multicast GLOBAL, acknowledge it, train (a
deterministic pseudo-delta per machine), send one 3-fragment UPDATE per machine
back-to-back in ascending position order, then announce readiness again. They
obey UPDATE_REPAIR by resending exactly the named fragments of the named
position. Uplink and downlink losses are injected by policy callables.

There is no radio, so a "downlink sent" event is confirmed immediately.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

from controller_v3 import HardwareExperimentController, Phase
from protocol_v3 import (
    GLOBAL_FRAGMENTS,
    HARDWARE_LEVELS,
    MSG_GLOBAL_FRAGMENT,
    MSG_UPDATE_FRAGMENT,
    MSG_UPDATE_REPAIR,
    N_PARAMS,
    HELLO_SD_MOUNTED,
    HELLO_SELF_CHECK_PASSED,
    decode_frame,
    decode_global,
    decode_repair_request,
    encode_board_hello,
    encode_class_c_ready,
    encode_global_status,
    encode_update,
    fragment_update,
    global_round_id,
    quantize_model,
    update_round_id,
)
from protocol import validate_crc
from protocol_v3 import GLOBAL_BYTES, board_of_position
from run_outputs import MemoryOutputs
from study_plan import load_fl_modules, load_machine_order, positions_on_board

GROUP = "tinyfl-v3-global"
RUN_ID = 4242


def device_id_for(board: int) -> str:
    return f"eui-70b3d57ed00{board:05x}"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def pseudo_delta(position: int, level_index: int, round_index: int) -> np.ndarray:
    """What a machine 'learned': deterministic, machine-specific, sane magnitude."""
    rng = np.random.default_rng([position, level_index, round_index, 7])
    return rng.normal(0.0, 0.05, N_PARAMS)


class FakeBoard:
    def __init__(self, number: int, sim: "Sim") -> None:
        self.number = number
        self.device_id = device_id_for(number)
        self.sim = sim
        self.global_parts: dict[int, dict[int, bytes]] = {}
        self.acked: set[int] = set()
        self.held_models: dict[tuple, object] = {}        # (level_index, round) -> GlobalModel
        self.update_frames: dict[int, list[bytes]] = {}
        self.send_count: Counter = Counter()             # (position, fragment) -> times sent
        self.status_due: dict[int, float] = {}
        self.level_index = -1
        self.round_index = -1

    # -- uplink ---------------------------------------------------------------
    def uplink(self, frame: bytes, kind: str = "other", position: int = -1,
               fragment: int = -1) -> None:
        if kind == "update":
            self.send_count[(position, fragment)] += 1
            if self.sim.uplink_drop(self.number, position, fragment,
                                    self.send_count[(position, fragment)],
                                    self.level_index, self.round_index):
                self.sim.uplinks_dropped += 1
                return
        self.sim.controller.handle_uplink(self.device_id, 10, frame)

    def hello(self, n_machines: int = 3, flags: int = HELLO_SD_MOUNTED | HELLO_SELF_CHECK_PASSED,
              run_id: int = 0):
        self.uplink(encode_board_hello(self.number, n_machines, 1, flags, run_id))

    def ready(self, slot: int = 0) -> None:
        """CLASS_C_READY; round_id is the slot of the GLOBAL the board waits for (3.6)."""
        self.uplink(encode_class_c_ready(slot, 5, 5))

    # -- downlink ---------------------------------------------------------------
    def receive_global(self, frame_bytes: bytes) -> None:
        frame = decode_frame(frame_bytes)
        header = frame.round_id
        parts = self.global_parts.setdefault(header, {})
        parts[frame.fragment_index] = frame.data
        self.status_due.setdefault(header, self.sim.clock.now + self.sim.idle_timeout)
        if len(parts) < GLOBAL_FRAGMENTS or header in self.acked:
            return
        payload = b"".join(parts[i] for i in range(GLOBAL_FRAGMENTS))
        validate_crc(payload, GLOBAL_BYTES)
        model = decode_global(payload)
        if model.run_id != self.sim.controller.run_id:
            return                                      # another run: a real board would reset
        self.acked.add(header)
        self.held_models[(model.level_index, model.round)] = model
        self.status_due.pop(header, None)
        self.uplink(encode_global_status(header, 0))
        if model.round < 12:
            if self.sim.crash(self.number, model.level_index, model.round):
                return                    # a board that took the global and then went silent
            self.train_and_upload(model.level_index, model.round)
            return
        # GLOBAL(L, 12) stored: now waiting for the next level's GLOBAL(L+1, 0).
        next_level = min(model.level_index + 1, len(HARDWARE_LEVELS) - 1)
        self.ready(global_round_id(next_level, 0))

    def train_and_upload(self, level_index: int, round_index: int) -> None:
        level = HARDWARE_LEVELS[level_index]
        header = update_round_id(level_index, round_index)
        self.level_index, self.round_index = level_index, round_index
        self.update_frames.clear()
        self.send_count.clear()
        base_g = self.sim.stale_base(self.number, level_index, round_index)
        if base_g is None:
            base_g = round_index                     # trains from the seed global
        for position in positions_on_board(self.number, level):
            v = pseudo_delta(position, level_index, round_index)
            delta = quantize_model(v)
            key = (level_index, round_index, position)
            self.sim.sent[key] = delta
            self.sim.sent_base[key] = base_g
            self.sim.sent_true[key] = v
            payload = encode_update(position, level_index, round_index, 100 + position,
                                    delta, 0.01 * (position + 1),
                                    run_id=self.sim.controller.run_id,
                                    base_slot=global_round_id(level_index, base_g))
            self.update_frames[position] = fragment_update(payload, header)
            for index, frame in enumerate(self.update_frames[position]):
                self.uplink(frame, "update", position, index)
        self.ready(header)              # after the last UPDATE fragment, waiting for GLOBAL(L, r)

    def receive_repair(self, frame_bytes: bytes) -> None:
        frame = decode_frame(frame_bytes)
        assert frame.message_type == MSG_UPDATE_REPAIR
        mask, position = decode_repair_request(frame)
        self.sim.repairs.append((self.number, position, mask))
        if position not in self.update_frames:
            return
        for index in range(GLOBAL_FRAGMENTS):
            if mask >> index & 1:
                self.uplink(self.update_frames[position][index], "update", position, index)
        self.ready(frame.round_id)

    def periodic(self, now: float) -> None:
        for header, due in list(self.status_due.items()):
            if now >= due and header not in self.acked:
                parts = self.global_parts.get(header, {})
                missing = sum(1 << i for i in range(GLOBAL_FRAGMENTS) if i not in parts)
                self.status_due[header] = now + self.sim.idle_timeout
                self.uplink(encode_global_status(header, missing))


class Sim:
    def __init__(self, levels=(4,), boards=None, expected_boards=None, seed=17070,
                 uplink_drop=None, downlink_drop=None, crash=None, idle_timeout=120.0,
                 outputs=None, stale_base=None, **controller_kwargs):
        self.clock = Clock()
        self.published: list[tuple] = []
        self.events: list[tuple] = []
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
        self.global_transmissions: Counter = Counter()
        board_numbers = list(boards or range(1, 11))
        params = dict(
            class_c_guard_seconds=2.0, retry_guard_seconds=1.0, max_downlink_attempts=5,
            max_repair_rounds=3, max_global_repair_rounds=3,
            multicast_report_window_seconds=30.0, inflight_downlink_timeout_seconds=20.0,
            ready_wait_seconds=60.0, round_timeout_seconds=600.0,
            post_run_collect_seconds=10.0, run_id=RUN_ID,
        )
        params.update(controller_kwargs)
        self.controller = HardwareExperimentController(
            levels=list(levels), seed=seed,
            expected_boards=expected_boards or len(board_numbers),
            publish=lambda d, f, c: self.published.append((d, f, c)),
            log=lambda e, d, x: self.events.append((e, d, x)),
            multicast_device_id=GROUP, outputs=self.out,
            fl_modules=load_fl_modules(), machine_order=load_machine_order()[0],
            clock=self.clock,
            wall_clock=lambda: "2026-09-30T00:00:00+00:00", **params,
        )
        self.boards = {n: FakeBoard(n, self) for n in board_numbers}
        self._cursor = 0

    def names(self) -> list[str]:
        return [e for e, _, _ in self.events]

    def start(self) -> None:
        for board in self.boards.values():
            board.hello()
            board.ready()

    def step(self, dt: float = 1.0) -> None:
        self.controller.tick()
        while self._cursor < len(self.published):
            device_id, frame, _ = self.published[self._cursor]
            self._cursor += 1
            self.controller.handle_downlink_sent(device_id)
            if device_id == GROUP:
                decoded = decode_frame(frame)
                assert decoded.message_type == MSG_GLOBAL_FRAGMENT
                self.global_transmissions[(decoded.round_id, decoded.fragment_index)] += 1
                count = self.global_transmissions[(decoded.round_id, decoded.fragment_index)]
                for board in self.boards.values():
                    if not self.downlink_drop(board.number, decoded.round_id,
                                              decoded.fragment_index, count):
                        board.receive_global(frame)
            else:
                board = next(b for b in self.boards.values() if b.device_id == device_id)
                board.receive_repair(frame)
        self.clock.now += dt
        for board in self.boards.values():
            board.periodic(self.clock.now)

    def run_until(self, predicate, max_steps: int = 20000) -> None:
        for _ in range(max_steps):
            if predicate():
                return
            self.step()
        raise AssertionError(
            f"did not converge; phase={self.controller.phase} "
            f"failure={self.controller.failure_reason!r} last events={self.names()[-8:]}"
        )

    def run_sweep(self, max_steps: int = 20000) -> None:
        self.start()
        self.run_until(lambda: self.controller.phase in (Phase.COLLECTING, Phase.DONE,
                                                          Phase.FAILED), max_steps)
