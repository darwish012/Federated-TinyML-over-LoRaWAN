"""Wire protocol v4: the general TinyFL production contract.

The single source of truth is ../PROTOCOL_V4.md; this module implements its
sections 2-3 for the server (and, for the tests and the simulator, the board
side of the same messages). Everything numerical is unchanged from v3: the
int8 Q_DYN_TENSOR codec is imported from protocol_v3, which is bit-identical to
research/simulation_completion_20260929/quant_fl.py (tests/test_protocol_v4.py
compares against that module again).

All multi-byte values are little-endian, packed, no padding. ``f32`` is
IEEE-754 binary32. CRC32 is zlib's (init 0, unsigned, little-endian trailer).

Frame (8 bytes):  magic 0xF1 | version 0x04 | u16 round_id | u8 frag_index |
                  u8 frag_count | u8 body_len (<= 213) | u8 type
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct

import numpy as np

from protocol import (
    CRC_BYTES,
    F_PORT,  # noqa: F401  (re-exported: the controller and tools import it from here)
    MAGIC,
    Frame,
    ProtocolError,
    append_crc,
    validate_crc,
)
# The numerical pipeline is shared with v3, not copied: one codec, one layout.
from protocol_v3 import (  # noqa: F401  (re-exported)
    DYN_FLOOR,
    DYN_HI,
    DYN_LO,
    LAYERS,
    N_PARAMS,
    N_TENSORS,
    SHAPES,
    TENSOR_SIZES,
    TENSOR_SLICES,
    ConflictingFragment,
    PartialMessage,
    QuantisedModel,
    dequantize_tensor,
    dynamic_scale,
    flatten,
    quantize_model,
    quantize_tensor,
    round_half_away_from_zero,
    unflatten,
)

VERSION = 0x04

# ---------------------------------------------------------------------------
# Message types (section 3)
# ---------------------------------------------------------------------------
MSG_UPDATE_FRAGMENT = 0x01   # up
MSG_CLASS_C_READY = 0x02     # up
MSG_GLOBAL_FRAGMENT = 0x03   # down, multicast
MSG_GLOBAL_STATUS = 0x04     # up
MSG_UPDATE_REPAIR = 0x05     # down, unicast
MSG_BOARD_HELLO = 0x06       # up
MSG_ALARM = 0x07             # up
MSG_SELF_CHECK = 0x08        # up
MSG_REPLAY_DONE = 0x09       # up
MSG_RUN_CONTROL = 0x0A       # down, multicast

UPLINK_TYPES = frozenset({
    MSG_UPDATE_FRAGMENT, MSG_CLASS_C_READY, MSG_GLOBAL_STATUS, MSG_BOARD_HELLO,
    MSG_ALARM, MSG_SELF_CHECK, MSG_REPLAY_DONE,
})
DOWNLINK_TYPES = frozenset({MSG_GLOBAL_FRAGMENT, MSG_UPDATE_REPAIR, MSG_RUN_CONTROL})
ALL_TYPES = UPLINK_TYPES | DOWNLINK_TYPES
FRAGMENT_TYPES = frozenset({MSG_UPDATE_FRAGMENT, MSG_GLOBAL_FRAGMENT})
CONTROL_TYPES = ALL_TYPES - FRAGMENT_TYPES

# ---------------------------------------------------------------------------
# Geometry and limits
# ---------------------------------------------------------------------------
HEADER_BYTES = 8
MAX_BODY_BYTES = 213
MAX_APPLICATION_BYTES = HEADER_BYTES + MAX_BODY_BYTES          # 221 (DR5)
NO_SLOT = 0xFFFF           # round_id of frames that refer to no slot; READY "any phase start"
NO_POSITION = 0xFFFF
MAX_POSITION = 0xFFFE
MAX_PHASE_INDEX = 254
MAX_SLOT = 0xFFFE
MAX_POSITIONS_PER_BOARD = 8

UPDATE_BYTES = 598
GLOBAL_BYTES = 624              # rev 4.3: + mask_base u16 + mask[32] before the CRC
UPDATE_CHUNK_BYTES = 211           # body = u16 position + <= 211 payload bytes
GLOBAL_CHUNK_BYTES = 213
UPDATE_FRAGMENTS = math.ceil(UPDATE_BYTES / UPDATE_CHUNK_BYTES)
GLOBAL_FRAGMENTS = math.ceil(GLOBAL_BYTES / GLOBAL_CHUNK_BYTES)
ALL_UPDATE_FRAGMENTS = (1 << UPDATE_FRAGMENTS) - 1
ALL_GLOBAL_FRAGMENTS = (1 << GLOBAL_FRAGMENTS) - 1

# UPDATE: 0 position u16, 2 run_id u32, 6 phase u8, 7 round u8, 8 base_slot u16,
#         10 n_train_rows u16, 12 scale[4], 28 delta[562], 590 loss f32, 594 crc32
_UPDATE_HEAD = struct.Struct("<HIBBHH4f")
_UPDATE_TAIL = struct.Struct("<f")
# GLOBAL: 0 run_id u32, 4 phase u8, 5 round u8, 6 rounds_in_phase u8, 7 flags u8,
#         8 scale[4], 24 weights[562], 586 crc32
_GLOBAL_HEAD = struct.Struct("<IBBBB4f")
_GLOBAL_MASK = struct.Struct("<H32s")          # mask_base, mask[32]: offset 586, before the CRC
_FRAME = struct.Struct("<BBHBBBB")
_READY = struct.Struct("<IH")
_STATUS = struct.Struct("<IHB")
_REPAIR = struct.Struct("<IHB")
_HELLO_HEAD = struct.Struct("<HBBIB")
_ALARM = struct.Struct("<IHBIff")
_SELF_CHECK = struct.Struct("<HHff")
_REPLAY_DONE = struct.Struct("<IHBIBI")
_RUN_CONTROL_HEAD = struct.Struct("<IBB")

assert _FRAME.size == HEADER_BYTES
assert _UPDATE_HEAD.size == 28 and _UPDATE_HEAD.size + N_PARAMS == 590
assert _UPDATE_HEAD.size + N_PARAMS + _UPDATE_TAIL.size + CRC_BYTES == UPDATE_BYTES
assert _GLOBAL_HEAD.size == 24 and _GLOBAL_MASK.size == 34
assert _GLOBAL_HEAD.size + N_PARAMS + _GLOBAL_MASK.size + CRC_BYTES == GLOBAL_BYTES
assert (_READY.size, _STATUS.size, _REPAIR.size, _HELLO_HEAD.size) == (6, 7, 7, 9)
assert (_ALARM.size, _SELF_CHECK.size, _REPLAY_DONE.size, _RUN_CONTROL_HEAD.size) == (19, 12, 16, 6)
assert UPDATE_FRAGMENTS == 3 and GLOBAL_FRAGMENTS == 3
assert MAX_APPLICATION_BYTES == 221
assert 2 + UPDATE_CHUNK_BYTES == MAX_BODY_BYTES

# BOARD_HELLO flag bits
HELLO_SD_MOUNTED = 0x01
HELLO_SELF_CHECK_PASSED = 0x02
HELLO_IMU_SOURCE = 0x04
HELLO_RUN_COMPLETE = 0x08

# GLOBAL flags
GLOBAL_FLAG_FINAL = 0x01
GLOBAL_FLAG_ALL_POSITIONS = 0x02   # rev 4.3: every position takes part (mask ignored; elastic)
MASK_WINDOW = 256                  # a planned phase's positions must fit one 256-wide window
NO_ALARM_INDEX = 0xFFFFFFFF        # REPLAY_DONE.first_alarm_index when nothing alarmed

# RUN_CONTROL commands (rev 4.1)
CMD_REPLAY = 1
CMD_END = 2
CMD_ADOPT = 3        # n_phases = 0: a board whose stored run differs adopts run_id (5.1)
RUN_CONTROL_COMMANDS = (CMD_REPLAY, CMD_END, CMD_ADOPT)
# 6 fixed bytes + phases must fit a 213-byte body; chunk a longer list across frames.
MAX_PHASES_PER_RUN_CONTROL = MAX_BODY_BYTES - _RUN_CONTROL_HEAD.size


# ---------------------------------------------------------------------------
# Frame layer
# ---------------------------------------------------------------------------
def encode_frame(message_type: int, round_id: int, fragment_index: int,
                 fragment_count: int, data: bytes = b"") -> bytes:
    if not 0 <= message_type <= 255:
        raise ProtocolError("message type must fit in one byte")
    if not 0 <= round_id <= 0xFFFF:
        raise ProtocolError("round_id must fit in u16")
    if not (0 <= fragment_index <= 255 and 0 <= fragment_count <= 255):
        raise ProtocolError("fragment index and count must fit in one byte")
    if len(data) > MAX_BODY_BYTES:
        raise ProtocolError(f"frame body of {len(data)} bytes exceeds {MAX_BODY_BYTES}")
    return _FRAME.pack(MAGIC, VERSION, round_id, fragment_index, fragment_count,
                       len(data), message_type) + bytes(data)


def decode_frame(payload: bytes) -> Frame:
    """Validate the header before anything is stored (section 2)."""
    if not HEADER_BYTES <= len(payload) <= MAX_APPLICATION_BYTES:
        raise ProtocolError(
            f"application frame length {len(payload)} is outside {HEADER_BYTES}..{MAX_APPLICATION_BYTES}")
    magic, version, round_id, index, count, body_len, message_type = _FRAME.unpack_from(payload, 0)
    if magic != MAGIC:
        raise ProtocolError("wrong experiment magic")
    if version != VERSION:
        raise ProtocolError(f"protocol version 0x{version:02X}; this server speaks v4 (0x04) only")
    if body_len > MAX_BODY_BYTES or len(payload) != HEADER_BYTES + body_len:
        raise ProtocolError("declared body length does not match the frame")
    if message_type not in ALL_TYPES:
        raise ProtocolError(f"unsupported message type 0x{message_type:02X}")
    if index >= count:
        raise ProtocolError(f"fragment index {index} is not below the count {count}")
    if message_type in FRAGMENT_TYPES and count != 3:
        raise ProtocolError(f"fragment count is {count}; expected 3")
    if message_type in CONTROL_TYPES and count != 1:
        raise ProtocolError(f"control frame declares {count} fragments; expected 1")
    return Frame(message_type, round_id, index, count, bytes(payload[HEADER_BYTES:]))


def _fragments(payload: bytes, message_type: int, round_id: int, chunk: int,
               prefix: bytes = b"") -> list[bytes]:
    count = math.ceil(len(payload) / chunk)
    return [
        encode_frame(message_type, round_id, i, count,
                     prefix + payload[i * chunk:(i + 1) * chunk])
        for i in range(count)
    ]


def fragment_update(payload: bytes, header_slot: int) -> list[bytes]:
    """UPDATE_FRAGMENT frames: body = u16 position + the next <= 211 bytes of the UPDATE.

    ``header_slot`` is the slot of the GLOBAL that opened the round (the seed).
    """
    if len(payload) != UPDATE_BYTES:
        raise ProtocolError(f"UPDATE payload has {len(payload)} bytes; expected {UPDATE_BYTES}")
    position = struct.unpack_from("<H", payload, 0)[0]
    return _fragments(payload, MSG_UPDATE_FRAGMENT, header_slot, UPDATE_CHUNK_BYTES,
                      struct.pack("<H", position))


def fragment_global(payload: bytes, slot: int) -> list[bytes]:
    if len(payload) != GLOBAL_BYTES:
        raise ProtocolError(f"GLOBAL payload has {len(payload)} bytes; expected {GLOBAL_BYTES}")
    return _fragments(payload, MSG_GLOBAL_FRAGMENT, slot, GLOBAL_CHUNK_BYTES)


def split_update_fragment(frame: Frame) -> tuple[int, Frame]:
    """(position, frame carrying only the payload chunk) of an UPDATE_FRAGMENT."""
    if frame.message_type != MSG_UPDATE_FRAGMENT:
        raise ProtocolError("not an UPDATE_FRAGMENT")
    if len(frame.data) < 2:
        raise ProtocolError("UPDATE fragment body is shorter than its u16 position prefix")
    position = struct.unpack_from("<H", frame.data, 0)[0]
    if position > MAX_POSITION:
        raise ProtocolError(f"UPDATE fragment position 0x{position:04X} is reserved")
    return position, Frame(frame.message_type, frame.round_id, frame.fragment_index,
                           frame.fragment_count, frame.data[2:])


def new_update_partial() -> PartialMessage:
    return PartialMessage(UPDATE_BYTES, UPDATE_FRAGMENTS, UPDATE_CHUNK_BYTES)


def new_global_partial() -> PartialMessage:
    return PartialMessage(GLOBAL_BYTES, GLOBAL_FRAGMENTS, GLOBAL_CHUNK_BYTES)


# ---------------------------------------------------------------------------
# Codec guards
# ---------------------------------------------------------------------------
def _check_codes(codes: np.ndarray, what: str) -> None:
    if int(codes.min()) < DYN_LO:
        raise ProtocolError(f"{what} carries int8 code -128; the convention is [-127, 127]")


def _check_scales(scales, what: str) -> None:
    for i, scale in enumerate(scales):
        if not (math.isfinite(scale) and scale > 0.0):
            raise ProtocolError(f"{what} scale[{i}] = {scale!r} is not finite and positive")


def _check_position(position: int, what: str) -> None:
    if not 0 <= position <= MAX_POSITION:
        raise ProtocolError(f"{what} position {position} is outside 0..{MAX_POSITION}")


# ---------------------------------------------------------------------------
# UPDATE (598 B) and GLOBAL (590 B)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Update:
    position: int
    run_id: int
    phase_index: int
    round: int                 # 1..R: the training round this update completes
    base_slot: int
    n_train_rows: int
    model: QuantisedModel      # int8 delta and its scales
    train_loss: float
    crc32: int


def encode_update(position: int, run_id: int, phase_index: int, round_: int,
                  base_slot: int, n_train_rows: int, delta: QuantisedModel,
                  train_loss: float) -> bytes:
    _check_position(position, "UPDATE")
    if not 0 <= run_id <= 0xFFFFFFFF:
        raise ProtocolError("run_id must fit in u32")
    if not 0 <= phase_index <= MAX_PHASE_INDEX:
        raise ProtocolError(f"phase_index {phase_index} is outside 0..{MAX_PHASE_INDEX}")
    if not 1 <= round_ <= 255:
        raise ProtocolError(f"round {round_} is outside 1..255")
    if not 0 <= base_slot <= 0xFFFF:
        raise ProtocolError("base_slot must fit in u16")
    n_train_rows = min(int(n_train_rows), 0xFFFF)      # saturates at 65535
    if n_train_rows < 0:
        raise ProtocolError("n_train_rows must not be negative")
    body = (_UPDATE_HEAD.pack(position, run_id, phase_index, round_, base_slot,
                              n_train_rows, *delta.scales)
            + delta.codes.tobytes() + _UPDATE_TAIL.pack(train_loss))
    return append_crc(body)


def decode_update(payload: bytes) -> Update:
    validate_crc(payload, UPDATE_BYTES)
    (position, run_id, phase_index, round_, base_slot, rows,
     s0, s1, s2, s3) = _UPDATE_HEAD.unpack_from(payload, 0)
    codes = np.frombuffer(payload, dtype=np.int8, count=N_PARAMS,
                          offset=_UPDATE_HEAD.size).copy()
    (loss,) = _UPDATE_TAIL.unpack_from(payload, _UPDATE_HEAD.size + N_PARAMS)
    crc = struct.unpack_from("<I", payload, UPDATE_BYTES - CRC_BYTES)[0]
    scales = (s0, s1, s2, s3)
    _check_position(position, "UPDATE")
    if phase_index > MAX_PHASE_INDEX:
        raise ProtocolError(f"UPDATE phase_index {phase_index} is outside 0..{MAX_PHASE_INDEX}")
    if round_ < 1:
        raise ProtocolError("UPDATE round must be >= 1 (round r completes training round r)")
    _check_scales(scales, "UPDATE")
    _check_codes(codes, "UPDATE")
    return Update(position, run_id, phase_index, round_, base_slot, rows,
                  QuantisedModel(codes=codes, scales=scales), loss, crc)


@dataclass(frozen=True)
class Participation:
    """Which positions take part in a phase (rev 4.3), as multicast in every GLOBAL of it.

    ``all`` (flags bit1) = every position takes part (elastic phases; the mask is ignored).
    Otherwise ``positions`` is the explicit set, which must fit one 256-wide window starting
    at ``mask_base``.
    """
    all: bool = True
    mask_base: int = 0
    positions: tuple = ()

    @classmethod
    def everyone(cls) -> "Participation":
        return cls(True, 0, ())

    @classmethod
    def of_positions(cls, positions) -> "Participation":
        positions = tuple(sorted(set(int(p) for p in positions)))
        if not positions:
            raise ProtocolError("a planned phase needs at least one position")
        for p in positions:
            _check_position(p, "participation")
        base = positions[0]
        if positions[-1] - base >= MASK_WINDOW:
            raise ProtocolError(
                f"positions {base}..{positions[-1]} do not fit one {MASK_WINDOW}-position window")
        return cls(False, base, positions)

    def covers(self, position: int) -> bool:
        return self.all or position in self.positions

    def mask_bytes(self) -> bytes:
        if self.all:
            return bytes(32)
        mask = bytearray(32)
        for p in self.positions:
            i = p - self.mask_base
            mask[i >> 3] |= 1 << (i & 7)
        return bytes(mask)

    @classmethod
    def from_wire(cls, all_positions: bool, mask_base: int, mask: bytes) -> "Participation":
        if all_positions:
            return cls(True, 0, ())
        positions = tuple(mask_base + 8 * j + i for j in range(32) for i in range(8)
                          if mask[j] >> i & 1 and mask_base + 8 * j + i <= MAX_POSITION)
        return cls(False, mask_base, positions)

    def to_dict(self) -> dict:
        return {"all": self.all, "mask_base": self.mask_base, "positions": list(self.positions)}

    @classmethod
    def from_dict(cls, d: dict) -> "Participation":
        return cls(bool(d["all"]), int(d["mask_base"]), tuple(d["positions"]))


@dataclass(frozen=True)
class GlobalModel:
    run_id: int
    phase_index: int
    round: int                 # 0 = initial model fl.init(seed); r = after round r
    rounds_in_phase: int
    flags: int
    model: QuantisedModel
    crc32: int
    participation: Participation = field(default_factory=Participation.everyone)

    @property
    def final(self) -> bool:
        return bool(self.flags & GLOBAL_FLAG_FINAL)


def encode_global(run_id: int, phase_index: int, round_: int, rounds_in_phase: int,
                  model: QuantisedModel, participation: Participation | None = None) -> bytes:
    if not 0 <= run_id <= 0xFFFFFFFF:
        raise ProtocolError("run_id must fit in u32")
    if not 0 <= phase_index <= MAX_PHASE_INDEX:
        raise ProtocolError(f"phase_index {phase_index} is outside 0..{MAX_PHASE_INDEX}")
    if not 1 <= rounds_in_phase <= 255 or not 0 <= round_ <= rounds_in_phase:
        raise ProtocolError(f"GLOBAL round {round_} is outside 0..{rounds_in_phase}")
    participation = participation or Participation.everyone()
    flags = GLOBAL_FLAG_FINAL if round_ == rounds_in_phase else 0
    if participation.all:
        flags |= GLOBAL_FLAG_ALL_POSITIONS
    body = (_GLOBAL_HEAD.pack(run_id, phase_index, round_, rounds_in_phase, flags, *model.scales)
            + model.codes.tobytes()
            + _GLOBAL_MASK.pack(participation.mask_base if not participation.all else 0,
                                participation.mask_bytes()))
    return append_crc(body)


def decode_global(payload: bytes) -> GlobalModel:
    validate_crc(payload, GLOBAL_BYTES)
    (run_id, phase_index, round_, rounds, flags,
     s0, s1, s2, s3) = _GLOBAL_HEAD.unpack_from(payload, 0)
    codes = np.frombuffer(payload, dtype=np.int8, count=N_PARAMS,
                          offset=_GLOBAL_HEAD.size).copy()
    crc = struct.unpack_from("<I", payload, GLOBAL_BYTES - CRC_BYTES)[0]
    scales = (s0, s1, s2, s3)
    if phase_index > MAX_PHASE_INDEX:
        raise ProtocolError(f"GLOBAL phase_index {phase_index} is outside 0..{MAX_PHASE_INDEX}")
    if rounds < 1 or round_ > rounds:
        raise ProtocolError(f"GLOBAL round {round_} is outside 0..{rounds}")
    if flags & ~(GLOBAL_FLAG_FINAL | GLOBAL_FLAG_ALL_POSITIONS):
        raise ProtocolError(f"GLOBAL flags 0x{flags:02X} set reserved bits")
    if bool(flags & GLOBAL_FLAG_FINAL) != (round_ == rounds):
        raise ProtocolError("GLOBAL final flag must be set exactly when round == rounds_in_phase")
    _check_scales(scales, "GLOBAL")
    _check_codes(codes, "GLOBAL")
    mask_base, mask = _GLOBAL_MASK.unpack_from(payload, _GLOBAL_HEAD.size + N_PARAMS)
    participation = Participation.from_wire(bool(flags & GLOBAL_FLAG_ALL_POSITIONS), mask_base, mask)
    return GlobalModel(run_id, phase_index, round_, rounds, flags,
                       QuantisedModel(codes, scales), crc, participation)


# ---------------------------------------------------------------------------
# Control bodies (section 3.3, exact sizes)
# ---------------------------------------------------------------------------
def _need(frame: Frame, message_type: int, size: int, name: str) -> None:
    if frame.message_type != message_type or len(frame.data) != size:
        raise ProtocolError(f"{name} carries exactly {size} bytes")


def _check_mask(mask: int, what: str, allow_zero: bool) -> None:
    if mask & ~ALL_GLOBAL_FRAGMENTS:
        raise ProtocolError(f"{what} missing_mask 0x{mask:02X} has bits outside 0..2")
    if mask == 0 and not allow_zero:
        raise ProtocolError(f"{what} missing_mask must not be empty")


@dataclass(frozen=True)
class ClassCReady:
    run_id: int
    awaited_slot: int


def encode_class_c_ready(run_id: int, awaited_slot: int) -> bytes:
    return encode_frame(MSG_CLASS_C_READY, awaited_slot, 0, 1,
                        _READY.pack(run_id, awaited_slot))


def decode_class_c_ready(frame: Frame) -> ClassCReady:
    _need(frame, MSG_CLASS_C_READY, _READY.size, "CLASS_C_READY")
    run_id, slot = _READY.unpack(frame.data)
    if frame.round_id != slot:
        raise ProtocolError(
            f"CLASS_C_READY header round_id {frame.round_id} != body awaited_slot {slot}")
    return ClassCReady(run_id, slot)


@dataclass(frozen=True)
class GlobalStatus:
    run_id: int
    slot: int
    missing_mask: int

    @property
    def complete(self) -> bool:
        return self.missing_mask == 0


def encode_global_status(run_id: int, slot: int, missing_mask: int) -> bytes:
    _check_mask(missing_mask, "GLOBAL_STATUS", allow_zero=True)
    return encode_frame(MSG_GLOBAL_STATUS, slot, 0, 1, _STATUS.pack(run_id, slot, missing_mask))


def decode_global_status(frame: Frame) -> GlobalStatus:
    _need(frame, MSG_GLOBAL_STATUS, _STATUS.size, "GLOBAL_STATUS")
    run_id, slot, mask = _STATUS.unpack(frame.data)
    _check_mask(mask, "GLOBAL_STATUS", allow_zero=True)
    if frame.round_id != slot:
        raise ProtocolError(f"GLOBAL_STATUS header round_id {frame.round_id} != body slot {slot}")
    return GlobalStatus(run_id, slot, mask)


@dataclass(frozen=True)
class UpdateRepair:
    run_id: int
    position: int
    missing_mask: int


def encode_update_repair(slot: int, run_id: int, position: int, missing_mask: int) -> bytes:
    """UPDATE_REPAIR; the header round_id is the slot of the round (its seed GLOBAL)."""
    _check_mask(missing_mask, "UPDATE_REPAIR", allow_zero=False)
    _check_position(position, "UPDATE_REPAIR")
    return encode_frame(MSG_UPDATE_REPAIR, slot, 0, 1,
                        _REPAIR.pack(run_id, position, missing_mask))


def decode_update_repair(frame: Frame) -> UpdateRepair:
    _need(frame, MSG_UPDATE_REPAIR, _REPAIR.size, "UPDATE_REPAIR")
    run_id, position, mask = _REPAIR.unpack(frame.data)
    _check_mask(mask, "UPDATE_REPAIR", allow_zero=False)
    _check_position(position, "UPDATE_REPAIR")
    return UpdateRepair(run_id, position, mask)


@dataclass(frozen=True)
class BoardHello:
    board_id: int
    fw_version: int
    flags: int
    stored_run_id: int          # 0 = none
    positions: tuple

    @property
    def sd_mounted(self) -> bool:
        return bool(self.flags & HELLO_SD_MOUNTED)

    @property
    def self_check_passed(self) -> bool:
        return bool(self.flags & HELLO_SELF_CHECK_PASSED)

    @property
    def imu_source(self) -> bool:
        return bool(self.flags & HELLO_IMU_SOURCE)

    @property
    def run_complete(self) -> bool:
        return bool(self.flags & HELLO_RUN_COMPLETE)

    @property
    def counted(self) -> bool:
        """Counts towards the start: bits 0 and 1, or bit 2 (IMU source)."""
        return self.imu_source or (self.sd_mounted and self.self_check_passed)


def encode_board_hello(board_id: int, fw_version: int, flags: int, stored_run_id: int,
                       positions) -> bytes:
    positions = tuple(int(p) for p in positions)
    if len(positions) > MAX_POSITIONS_PER_BOARD:
        raise ProtocolError(f"a board hosts at most {MAX_POSITIONS_PER_BOARD} positions")
    for p in positions:
        _check_position(p, "BOARD_HELLO")
    if not 0 <= board_id <= 0xFFFF:
        raise ProtocolError("board_id must fit in u16")
    body = _HELLO_HEAD.pack(board_id, fw_version, flags, stored_run_id, len(positions))
    body += b"".join(struct.pack("<H", p) for p in positions)
    return encode_frame(MSG_BOARD_HELLO, NO_SLOT, 0, 1, body)


def decode_board_hello(frame: Frame) -> BoardHello:
    if frame.message_type != MSG_BOARD_HELLO or len(frame.data) < _HELLO_HEAD.size:
        raise ProtocolError("BOARD_HELLO carries 9 + 2n bytes")
    board_id, fw, flags, stored, n = _HELLO_HEAD.unpack_from(frame.data, 0)
    if n > MAX_POSITIONS_PER_BOARD:
        raise ProtocolError(f"BOARD_HELLO declares {n} positions; the limit is {MAX_POSITIONS_PER_BOARD}")
    if len(frame.data) != _HELLO_HEAD.size + 2 * n:
        raise ProtocolError(f"BOARD_HELLO with n={n} must carry exactly {9 + 2 * n} bytes")
    positions = struct.unpack_from(f"<{n}H", frame.data, _HELLO_HEAD.size) if n else ()
    for p in positions:
        _check_position(p, "BOARD_HELLO")
    if len(set(positions)) != len(positions):
        raise ProtocolError("BOARD_HELLO lists a position twice")
    return BoardHello(board_id, fw, flags, stored, tuple(positions))


@dataclass(frozen=True)
class Alarm:
    run_id: int
    position: int
    phase_index: int
    acquisition_index: int
    score: float
    threshold: float


def encode_alarm(run_id: int, position: int, phase_index: int, acquisition_index: int,
                 score: float, threshold: float) -> bytes:
    _check_position(position, "ALARM")
    return encode_frame(MSG_ALARM, NO_SLOT, 0, 1, _ALARM.pack(
        run_id, position, phase_index, acquisition_index, score, threshold))


def decode_alarm(frame: Frame) -> Alarm:
    _need(frame, MSG_ALARM, _ALARM.size, "ALARM")
    alarm = Alarm(*_ALARM.unpack(frame.data))
    _check_position(alarm.position, "ALARM")
    if alarm.phase_index > MAX_PHASE_INDEX:
        raise ProtocolError(f"ALARM phase_index {alarm.phase_index} is outside 0..{MAX_PHASE_INDEX}")
    return alarm


@dataclass(frozen=True)
class SelfCheck:
    position: int
    rows_checked: int
    max_abs_diff: float
    median_abs_diff: float


def encode_self_check(position: int, rows_checked: int, max_abs_diff: float,
                      median_abs_diff: float) -> bytes:
    _check_position(position, "SELF_CHECK")
    return encode_frame(MSG_SELF_CHECK, NO_SLOT, 0, 1, _SELF_CHECK.pack(
        position, rows_checked, max_abs_diff, median_abs_diff))


def decode_self_check(frame: Frame) -> SelfCheck:
    _need(frame, MSG_SELF_CHECK, _SELF_CHECK.size, "SELF_CHECK")
    check = SelfCheck(*_SELF_CHECK.unpack(frame.data))
    _check_position(check.position, "SELF_CHECK")
    return check


@dataclass(frozen=True)
class ReplayDone:
    """One (position, phase) the board scored (rev 4.1: per phase, so all 255 phases confirm)."""
    run_id: int
    position: int
    phase_index: int
    acquisitions_scored: int
    alarm_latched: int              # 0 / 1
    first_alarm_index: int = NO_ALARM_INDEX      # rev 4.3; 0xFFFFFFFF = none

    @property
    def consistent(self) -> bool:
        """3.3: alarm_latched = 1 <=> first_alarm_index != 0xFFFFFFFF."""
        return (self.alarm_latched == 1) == (self.first_alarm_index != NO_ALARM_INDEX)


def encode_replay_done(run_id: int, position: int, phase_index: int,
                       acquisitions_scored: int, alarm_latched: int,
                       first_alarm_index: int | None = None) -> bytes:
    """``first_alarm_index`` defaults to "none" for alarm_latched 0 and to 0 otherwise.
    It is NOT checked against alarm_latched here, so tests can build a violating frame."""
    _check_position(position, "REPLAY_DONE")
    if not 0 <= phase_index <= MAX_PHASE_INDEX:
        raise ProtocolError(f"phase_index {phase_index} is outside 0..{MAX_PHASE_INDEX}")
    if alarm_latched not in (0, 1):
        raise ProtocolError("alarm_latched must be 0 or 1")
    if first_alarm_index is None:
        first_alarm_index = 0 if alarm_latched else NO_ALARM_INDEX
    return encode_frame(MSG_REPLAY_DONE, NO_SLOT, 0, 1, _REPLAY_DONE.pack(
        run_id, position, phase_index, acquisitions_scored, alarm_latched, first_alarm_index))


def decode_replay_done(frame: Frame) -> ReplayDone:
    _need(frame, MSG_REPLAY_DONE, _REPLAY_DONE.size, "REPLAY_DONE")
    done = ReplayDone(*_REPLAY_DONE.unpack(frame.data))
    _check_position(done.position, "REPLAY_DONE")
    if done.phase_index > MAX_PHASE_INDEX:
        raise ProtocolError(f"REPLAY_DONE phase_index {done.phase_index} is outside 0..{MAX_PHASE_INDEX}")
    if done.alarm_latched not in (0, 1):
        raise ProtocolError(f"REPLAY_DONE alarm_latched {done.alarm_latched} is not 0 or 1")
    return done


@dataclass(frozen=True)
class RunControl:
    run_id: int
    command: int
    phases: tuple


def encode_run_control(run_id: int, command: int, phases=()) -> bytes:
    phases = tuple(int(p) for p in phases)
    if command not in RUN_CONTROL_COMMANDS:
        raise ProtocolError(f"RUN_CONTROL command {command} is not 1 (REPLAY), 2 (END) or 3 (ADOPT)")
    if command == CMD_ADOPT and phases:
        raise ProtocolError("RUN_CONTROL ADOPT carries no phases (n_phases = 0)")
    if command == CMD_REPLAY and not phases:
        raise ProtocolError("RUN_CONTROL REPLAY must list at least one phase")
    if len(phases) > MAX_PHASES_PER_RUN_CONTROL:
        raise ProtocolError("RUN_CONTROL lists more phases than one frame carries")
    if any(not 0 <= p <= MAX_PHASE_INDEX for p in phases) or len(set(phases)) != len(phases):
        raise ProtocolError("RUN_CONTROL phases must be unique values in 0..254")
    return encode_frame(MSG_RUN_CONTROL, NO_SLOT, 0, 1,
                        _RUN_CONTROL_HEAD.pack(run_id, command, len(phases)) + bytes(phases))


def decode_run_control(frame: Frame) -> RunControl:
    if frame.message_type != MSG_RUN_CONTROL or len(frame.data) < _RUN_CONTROL_HEAD.size:
        raise ProtocolError("RUN_CONTROL carries 6 + n bytes")
    run_id, command, n = _RUN_CONTROL_HEAD.unpack_from(frame.data, 0)
    if len(frame.data) != _RUN_CONTROL_HEAD.size + n:
        raise ProtocolError("RUN_CONTROL body length does not match n_phases")
    if command not in RUN_CONTROL_COMMANDS:
        raise ProtocolError(f"RUN_CONTROL command {command} is not 1 (REPLAY), 2 (END) or 3 (ADOPT)")
    phases = tuple(frame.data[_RUN_CONTROL_HEAD.size:])
    if len(set(phases)) != len(phases) or any(p > MAX_PHASE_INDEX for p in phases):
        raise ProtocolError("RUN_CONTROL phases must be unique values in 0..254")
    if command == CMD_ADOPT and phases:
        raise ProtocolError("RUN_CONTROL ADOPT carries no phases (n_phases = 0)")
    if command == CMD_REPLAY and not phases:
        raise ProtocolError("RUN_CONTROL REPLAY must list at least one phase")
    return RunControl(run_id, command, phases)


# ---------------------------------------------------------------------------
# Helpers shared by the controller, the simulator and the tools
# ---------------------------------------------------------------------------
def dev_eui_of_device_id(device_id: str) -> str:
    """Upper-case 16-hex DevEUI of a TTS device_id ``eui-<hex>``; ProtocolError otherwise."""
    text = device_id[4:] if device_id.lower().startswith("eui-") else device_id
    text = text.strip()
    if len(text) != 16:
        raise ProtocolError(f"device id {device_id!r} is not eui-<16 hex digits>")
    try:
        bytes.fromhex(text)
    except ValueError as exc:
        raise ProtocolError(f"device id {device_id!r} is not eui-<16 hex digits>") from exc
    return text.upper()


def device_id_of_dev_eui(dev_eui: str) -> str:
    return f"eui-{dev_eui.lower()}"


__all__ = [name for name in dir() if not name.startswith("_")]
