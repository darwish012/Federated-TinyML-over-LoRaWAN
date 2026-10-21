"""Wire protocol v3 for the 10-board TinyFL federated-learning hardware experiment.

The single source of truth is ../PROTOCOL_V3.md; this module implements its
sections 2-4 for the server. Everything not stated there is unchanged from v2
(protocol.py, which is left intact), so the v2 helpers that are still correct
are imported from it rather than copied.

All multi-byte values are little-endian, ``f32`` is IEEE-754 binary32.

The int8 codec is the Q_DYN_TENSOR convention from
research/simulation_completion_20260929/quant_fl.py and must stay
bit-identical to it (tests/test_protocol_v3.py compares against that module):
per tensor ``scale = max(max|x| / 127, 1e-7)``, ``q = roundf(x / scale)``
with C roundf (half away from zero), clipped to [-127, 127].
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import numpy as np

from protocol import (
    CRC_BYTES,
    FRAGMENT_BODY_BYTES,
    HEADER_BYTES,
    MAGIC,
    MAX_APPLICATION_BYTES,
    F_PORT,
    Frame,
    ProtocolError,
    append_crc,
    validate_crc,
)

__all__ = [
    "VERSION", "MAGIC", "F_PORT", "ProtocolError", "Frame",
    "MSG_UPDATE_FRAGMENT", "MSG_CLASS_C_READY", "MSG_GLOBAL_FRAGMENT",
    "MSG_GLOBAL_STATUS", "MSG_UPDATE_REPAIR", "MSG_BOARD_HELLO", "MSG_ALARM",
    "MSG_SELF_CHECK", "MSG_REPLAY_DONE",
]

VERSION = 0x03

MSG_UPDATE_FRAGMENT = 0x01   # up
MSG_CLASS_C_READY = 0x02     # up, v2 unchanged
MSG_GLOBAL_FRAGMENT = 0x03   # down, multicast
MSG_GLOBAL_STATUS = 0x04     # up, u16 bitmap of GLOBAL fragments held (rev 3.2: no CRC)
MSG_UPDATE_REPAIR = 0x05     # down, u16 mask + u8 position
MSG_BOARD_HELLO = 0x06       # up
MSG_ALARM = 0x07             # up
MSG_SELF_CHECK = 0x08        # up
MSG_REPLAY_DONE = 0x09       # up (rev 3.2)

UPLINK_TYPES = frozenset({
    MSG_UPDATE_FRAGMENT, MSG_CLASS_C_READY, MSG_GLOBAL_STATUS,
    MSG_BOARD_HELLO, MSG_ALARM, MSG_SELF_CHECK, MSG_REPLAY_DONE,
})
SINGLE_FRAME_TYPES = frozenset({MSG_BOARD_HELLO, MSG_ALARM, MSG_SELF_CHECK, MSG_REPLAY_DONE})
FRAGMENT_TYPES = frozenset({MSG_UPDATE_FRAGMENT, MSG_GLOBAL_FRAGMENT})
ALL_TYPES = UPLINK_TYPES | {MSG_GLOBAL_FRAGMENT, MSG_UPDATE_REPAIR}

# ---------------------------------------------------------------------------
# Study geometry (protocol.json, amendment_hardware_scope.json)
# ---------------------------------------------------------------------------
LAYERS = (22, 12, 22)
N_PARAMS = 562
N_TENSORS = 4
SHAPES = ((22, 12), (12,), (12, 22), (22,))
TENSOR_SIZES = (264, 12, 264, 22)
_OFFSETS = (0, 264, 276, 540, 562)
TENSOR_SLICES = tuple(slice(_OFFSETS[i], _OFFSETS[i + 1]) for i in range(N_TENSORS))
assert sum(TENSOR_SIZES) == N_PARAMS

N_BOARDS = 10
ROUNDS_PER_LEVEL = 12
ROUND_SLOTS = ROUNDS_PER_LEVEL + 1      # round_id slots per level (g = 0..12)
NO_ROUND = 0xFF                         # round_id of HELLO / ALARM / SELF_CHECK
# level_index is the index into THIS canonical list, whichever subset of levels
# a particular sweep runs; boards' levels.csv uses the same indexing.
HARDWARE_LEVELS = (4, 8, 12, 16, 20, 24, 28)
MAX_POSITION = 53

DYN_LO, DYN_HI = -127, 127
DYN_FLOOR = 1e-7

# ---------------------------------------------------------------------------
# Message sizes
# ---------------------------------------------------------------------------
UPDATE_BYTES = 594          # rev 3.2
GLOBAL_BYTES = 586          # rev 3.2
UPDATE_DATA_BYTES = UPDATE_BYTES - CRC_BYTES
GLOBAL_DATA_BYTES = GLOBAL_BYTES - CRC_BYTES
UPDATE_CHUNK_BYTES = FRAGMENT_BODY_BYTES - 1     # rev 3.1: body = position byte + <= 213 bytes
UPDATE_FRAGMENTS = math.ceil(UPDATE_BYTES / UPDATE_CHUNK_BYTES)
GLOBAL_FRAGMENTS = math.ceil(GLOBAL_BYTES / FRAGMENT_BODY_BYTES)
ALL_UPDATE_FRAGMENTS = (1 << UPDATE_FRAGMENTS) - 1
ALL_GLOBAL_FRAGMENTS = (1 << GLOBAL_FRAGMENTS) - 1

# UPDATE: 0 position, 1 level_index, 2 round, 3 run_id u16, 5 base_slot, 6 n_train_rows u16,
#         8 scale[4], 24 delta[562], 586 train_loss, 590 crc32
_UPDATE_HEAD = struct.Struct("<BBBHBH4f")
_UPDATE_TAIL = struct.Struct("<f")             # train_loss at offset 586
# GLOBAL: 0 level_index, 1 round, 2 run_id u16, 4 scale[4], 20 weights[562], 582 crc32
_GLOBAL_HEAD = struct.Struct("<BBH4f")
_HELLO = struct.Struct("<BBBBH")               # board, n_machines, fw, flags, run_id
_ALARM = struct.Struct("<BBIff")
_SELF_CHECK = struct.Struct("<BHff")
_REPLAY_DONE = struct.Struct("<BBIB")          # position, level_mask, acquisitions_scored, alarms

assert _UPDATE_HEAD.size == 24 and _UPDATE_HEAD.size + N_PARAMS == 586
assert _UPDATE_HEAD.size + N_PARAMS + _UPDATE_TAIL.size + CRC_BYTES == UPDATE_BYTES == 594
assert _GLOBAL_HEAD.size == 20 and _GLOBAL_HEAD.size + N_PARAMS + CRC_BYTES == GLOBAL_BYTES == 586
assert (_HELLO.size, _ALARM.size, _SELF_CHECK.size, _REPLAY_DONE.size) == (6, 14, 11, 7)
assert UPDATE_FRAGMENTS == 3 and GLOBAL_FRAGMENTS == 3
assert MAX_APPLICATION_BYTES == 221

# BOARD_HELLO flag bits
HELLO_SD_MOUNTED = 0x01
HELLO_SELF_CHECK_PASSED = 0x02
HELLO_IMU_SOURCE = 0x04


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def board_of_position(position: int) -> int:
    """PROTOCOL_V3: position p runs on board p mod 10 + 1 (1..10)."""
    return position % N_BOARDS + 1


def level_index_of(level: int) -> int:
    try:
        return HARDWARE_LEVELS.index(level)
    except ValueError as exc:
        raise ValueError(
            f"level {level} is not one of the hardware levels {HARDWARE_LEVELS}"
        ) from exc


def global_round_id(level_index: int, g: int) -> int:
    """Frame-header round_id slot (rev 3.1): level_index * 13 + g, g = 0..12.

    13 slots per level, so GLOBAL(L, 12) and GLOBAL(L+1, 0) never share a
    value. For GLOBAL, g is the GLOBAL's round (0 = initial, 12 = final).
    """
    if not 0 <= level_index < len(HARDWARE_LEVELS) or not 0 <= g < ROUND_SLOTS:
        raise ProtocolError(f"slot (level_index {level_index}, g {g}) is out of range")
    return level_index * ROUND_SLOTS + g


def update_round_id(level_index: int, round_index: int) -> int:
    """Header slot of an UPDATE for 0-based payload round r0 (rev 3.1: g = r0 + 1).

    The UPDATE of round r (1..12) carries the slot of the GLOBAL that round
    produces; the payload's own ``round`` field stays 0..11.
    """
    if not 0 <= round_index < ROUNDS_PER_LEVEL:
        raise ProtocolError(f"round {round_index} is outside 0..11")
    return global_round_id(level_index, round_index + 1)


# ---------------------------------------------------------------------------
# Frame layer (VERSION 0x03)
# ---------------------------------------------------------------------------
def encode_frame(
    message_type: int,
    round_id: int,
    fragment_index: int,
    fragment_count: int,
    data: bytes = b"",
) -> bytes:
    values = (message_type, round_id, fragment_index, fragment_count, len(data))
    if any(not 0 <= value <= 255 for value in values):
        raise ProtocolError("all frame fields must fit in one byte")
    if len(data) > FRAGMENT_BODY_BYTES:
        raise ProtocolError("frame body exceeds 214 bytes")
    return bytes(
        (MAGIC, VERSION, message_type, round_id, fragment_index, fragment_count, len(data))
    ) + data


def decode_frame(payload: bytes) -> Frame:
    if not HEADER_BYTES <= len(payload) <= MAX_APPLICATION_BYTES:
        raise ProtocolError(f"application frame length {len(payload)} is outside 7..221")
    if payload[0] != MAGIC:
        raise ProtocolError("wrong experiment magic")
    if payload[1] != VERSION:
        raise ProtocolError(
            f"protocol version 0x{payload[1]:02X}; this server speaks v3 (0x03) only"
        )
    data_length = payload[6]
    if data_length > FRAGMENT_BODY_BYTES or len(payload) != HEADER_BYTES + data_length:
        raise ProtocolError("declared body length does not match the frame")
    frame = Frame(
        message_type=payload[2],
        round_id=payload[3],
        fragment_index=payload[4],
        fragment_count=payload[5],
        data=payload[HEADER_BYTES:],
    )
    if frame.message_type not in ALL_TYPES:
        raise ProtocolError(f"unsupported message type 0x{frame.message_type:02X}")
    if frame.fragment_index >= frame.fragment_count:
        raise ProtocolError(
            f"fragment index {frame.fragment_index} is not below the count {frame.fragment_count}"
        )
    if frame.message_type in FRAGMENT_TYPES and frame.fragment_count != 3:
        raise ProtocolError(f"fragment count is {frame.fragment_count}; expected 3")
    if frame.message_type in SINGLE_FRAME_TYPES and frame.fragment_count != 1:
        raise ProtocolError(f"single-frame message declares {frame.fragment_count} fragments")
    return frame


def fragment_payload(payload: bytes, message_type: int, round_id: int) -> list[bytes]:
    count = math.ceil(len(payload) / FRAGMENT_BODY_BYTES)
    if count > 255:
        raise ProtocolError("logical payload needs more than 255 fragments")
    return [
        encode_frame(
            message_type,
            round_id,
            index,
            count,
            payload[index * FRAGMENT_BODY_BYTES:(index + 1) * FRAGMENT_BODY_BYTES],
        )
        for index in range(count)
    ]


def fragment_update(payload: bytes, round_id: int) -> list[bytes]:
    """UPDATE_FRAGMENT frames (rev 3.1): each body is the position byte plus <= 213 payload bytes."""
    if len(payload) != UPDATE_BYTES:
        raise ProtocolError(f"UPDATE payload has {len(payload)} bytes; expected {UPDATE_BYTES}")
    position = payload[0]
    return [
        encode_frame(
            MSG_UPDATE_FRAGMENT,
            round_id,
            index,
            UPDATE_FRAGMENTS,
            bytes((position,)) + payload[index * UPDATE_CHUNK_BYTES:(index + 1) * UPDATE_CHUNK_BYTES],
        )
        for index in range(UPDATE_FRAGMENTS)
    ]


def split_update_fragment(frame: Frame) -> tuple[int, Frame]:
    """(position, frame carrying only the payload chunk) of an UPDATE_FRAGMENT."""
    if not frame.data:
        raise ProtocolError("UPDATE fragment has an empty body; it must start with the position")
    return frame.data[0], Frame(
        frame.message_type, frame.round_id, frame.fragment_index, frame.fragment_count,
        frame.data[1:],
    )


def expected_fragment_body(index: int, total_bytes: int, chunk_bytes: int = FRAGMENT_BODY_BYTES) -> int:
    """Payload bytes the index-th fragment of a total_bytes payload must carry."""
    return min(chunk_bytes, total_bytes - index * chunk_bytes)


class ConflictingFragment(ProtocolError):
    """A duplicate fragment whose bytes differ from the one already stored."""


class PartialMessage:
    """Payload chunks of one logical message, validated on arrival.

    Knows nothing about rounds or positions: the controller decides which
    partial a fragment belongs to (UPDATE fragments name their position) and
    only asks this class to police lengths. ``chunk_bytes`` is 214 for GLOBAL
    and 213 for UPDATE (one body byte is the position).
    """

    def __init__(self, total_bytes: int, fragment_count: int,
                 chunk_bytes: int = FRAGMENT_BODY_BYTES) -> None:
        self.total_bytes = total_bytes
        self.fragment_count = fragment_count
        self.chunk_bytes = chunk_bytes
        self.parts: dict[int, bytes] = {}

    def missing_mask(self) -> int:
        return sum(1 << i for i in range(self.fragment_count) if i not in self.parts)

    def add(self, frame: Frame) -> bytes | None:
        if frame.fragment_count != self.fragment_count:
            raise ProtocolError(
                f"fragment count is {frame.fragment_count}; expected {self.fragment_count}"
            )
        if not 0 <= frame.fragment_index < self.fragment_count:
            raise ProtocolError("fragment index is outside its declared count")
        want = expected_fragment_body(frame.fragment_index, self.total_bytes, self.chunk_bytes)
        if len(frame.data) != want:
            raise ProtocolError(
                f"fragment {frame.fragment_index} has {len(frame.data)} data bytes; "
                f"expected {want}"
            )
        previous = self.parts.get(frame.fragment_index)
        if previous is not None and previous != frame.data:
            raise ConflictingFragment("duplicate fragment has different bytes; not stored")
        self.parts[frame.fragment_index] = frame.data
        if len(self.parts) != self.fragment_count:
            return None
        return b"".join(self.parts[i] for i in range(self.fragment_count))


# ---------------------------------------------------------------------------
# int8 per-tensor codec (Q_DYN_TENSOR, bit-identical to quant_fl.py)
# ---------------------------------------------------------------------------
def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def dynamic_scale(vec: np.ndarray) -> float:
    """max|x|/127, floored at 1e-7 (quant_fl.dynamic_scale)."""
    vec = np.asarray(vec, dtype=float)
    max_abs = float(np.max(np.abs(vec))) if vec.size else 0.0
    return max(max_abs / 127.0, DYN_FLOOR)


def quantize_tensor(vec: np.ndarray) -> tuple[np.ndarray, float]:
    """One tensor -> (int8 codes, float64 scale), exactly quant_fl's expression."""
    vec = np.asarray(vec, dtype=float)
    scale = dynamic_scale(vec)
    codes = np.clip(round_half_away_from_zero(vec / scale), DYN_LO, DYN_HI)
    return codes.astype(np.int8), scale


def dequantize_tensor(codes: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(codes).astype(np.float64) * float(scale)


def flatten(params) -> np.ndarray:
    """(W1, b1, W2, b2) or an already-flat vector -> float64 vector of 562."""
    if isinstance(params, np.ndarray) and params.ndim == 1:
        flat = params.astype(float)
    else:
        flat = np.concatenate([np.asarray(p, dtype=float).ravel() for p in params])
    if flat.size != N_PARAMS:
        raise ValueError(f"model has {flat.size} parameters; expected {N_PARAMS}")
    return flat


def unflatten(vec: np.ndarray) -> tuple:
    vec = np.asarray(vec, dtype=float)
    return tuple(vec[TENSOR_SLICES[i]].reshape(SHAPES[i]) for i in range(N_TENSORS))


@dataclass(frozen=True)
class QuantisedModel:
    """562 int8 codes plus four per-tensor scales.

    ``scales`` are kept as computed (float64) after quantisation and as the
    float32 values that came off the wire after decoding; either way
    dequantize() multiplies the codes by exactly these numbers.
    """

    codes: np.ndarray
    scales: tuple

    def __post_init__(self) -> None:
        if self.codes.shape != (N_PARAMS,) or self.codes.dtype != np.int8:
            raise ValueError("codes must be an int8 vector of 562")
        if len(self.scales) != N_TENSORS:
            raise ValueError("a quantised model carries exactly four scales")

    def wire_scales(self) -> tuple:
        """The float32 scales as transmitted."""
        return tuple(float(np.float32(s)) for s in self.scales)

    def dequantize(self) -> np.ndarray:
        out = np.empty(N_PARAMS, dtype=np.float64)
        for i, sl in enumerate(TENSOR_SLICES):
            out[sl] = dequantize_tensor(self.codes[sl], self.scales[i])
        return out

    def dequantize_tuple(self) -> tuple:
        return unflatten(self.dequantize())

    def codes_sha256(self) -> str:
        import hashlib
        return hashlib.sha256(self.codes.tobytes()).hexdigest()


def quantize_model(params) -> QuantisedModel:
    flat = flatten(params)
    codes = np.empty(N_PARAMS, dtype=np.int8)
    scales = []
    for sl in TENSOR_SLICES:
        tensor_codes, scale = quantize_tensor(flat[sl])
        codes[sl] = tensor_codes
        scales.append(scale)
    return QuantisedModel(codes=codes, scales=tuple(scales))


def _check_codes(codes: np.ndarray, what: str) -> None:
    if int(codes.min()) < DYN_LO:
        raise ProtocolError(f"{what} carries int8 code -128; the convention is [-127, 127]")


def _check_scales(scales, what: str) -> None:
    for i, scale in enumerate(scales):
        if not (math.isfinite(scale) and scale > 0.0):
            raise ProtocolError(f"{what} scale[{i}] = {scale!r} is not finite and positive")


# ---------------------------------------------------------------------------
# UPDATE (591 B) and GLOBAL (584 B)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Update:
    position: int
    level_index: int
    round: int
    run_id: int
    base_slot: int
    n_train_rows: int
    model: QuantisedModel      # the int8 delta and its scales
    train_loss: float
    crc32: int


def encode_update(
    position: int,
    level_index: int,
    round_index: int,
    n_train_rows: int,
    delta: QuantisedModel,
    train_loss: float,
    run_id: int = 1,
    base_slot: int | None = None,
) -> bytes:
    """594-byte UPDATE. ``base_slot`` defaults to the slot of the GLOBAL that seeds the round."""
    if not 0 <= position <= MAX_POSITION:
        raise ProtocolError(f"position {position} is outside 0..{MAX_POSITION}")
    if not 0 <= level_index < len(HARDWARE_LEVELS):
        raise ProtocolError(f"level_index {level_index} is outside 0..6")
    if not 0 <= round_index < ROUNDS_PER_LEVEL:
        raise ProtocolError(f"round {round_index} is outside 0..11")
    if not 0 <= n_train_rows <= 0xFFFF:
        raise ProtocolError("n_train_rows must fit in u16")
    if not 0 <= run_id <= 0xFFFF:
        raise ProtocolError("run_id must fit in u16")
    if base_slot is None:
        base_slot = global_round_id(level_index, round_index)
    if not 0 <= base_slot <= 255:
        raise ProtocolError("base_slot must fit in one byte")
    body = (
        _UPDATE_HEAD.pack(position, level_index, round_index, run_id, base_slot,
                          n_train_rows, *delta.scales)
        + delta.codes.tobytes()
        + _UPDATE_TAIL.pack(train_loss)
    )
    return append_crc(body)


def decode_update(payload: bytes) -> Update:
    validate_crc(payload, UPDATE_BYTES)
    (position, level_index, round_index, run_id, base_slot, rows,
     s0, s1, s2, s3) = _UPDATE_HEAD.unpack_from(payload, 0)
    base = _UPDATE_HEAD.size
    codes = np.frombuffer(payload, dtype=np.int8, count=N_PARAMS, offset=base).copy()
    (loss,) = _UPDATE_TAIL.unpack_from(payload, base + N_PARAMS)
    crc = struct.unpack_from("<I", payload, UPDATE_BYTES - CRC_BYTES)[0]
    scales = (s0, s1, s2, s3)
    if position > MAX_POSITION:
        raise ProtocolError(f"UPDATE position {position} is outside 0..{MAX_POSITION}")
    if level_index >= len(HARDWARE_LEVELS):
        raise ProtocolError(f"UPDATE level_index {level_index} is outside 0..6")
    if round_index >= ROUNDS_PER_LEVEL:
        raise ProtocolError(f"UPDATE round {round_index} is outside 0..11")
    _check_scales(scales, "UPDATE")
    _check_codes(codes, "UPDATE")
    return Update(
        position=position,
        level_index=level_index,
        round=round_index,
        run_id=run_id,
        base_slot=base_slot,
        n_train_rows=rows,
        model=QuantisedModel(codes=codes, scales=scales),
        train_loss=loss,
        crc32=crc,
    )


@dataclass(frozen=True)
class GlobalModel:
    level_index: int
    round: int                 # 0 = initial model of the level, r = after round r
    run_id: int
    model: QuantisedModel
    crc32: int


def encode_global(level_index: int, round_index: int, model: QuantisedModel,
                  run_id: int = 1) -> bytes:
    if not 0 <= level_index < len(HARDWARE_LEVELS):
        raise ProtocolError(f"level_index {level_index} is outside 0..6")
    if not 0 <= round_index <= ROUNDS_PER_LEVEL:
        raise ProtocolError(f"GLOBAL round {round_index} is outside 0..12")
    if not 0 <= run_id <= 0xFFFF:
        raise ProtocolError("run_id must fit in u16")
    body = (_GLOBAL_HEAD.pack(level_index, round_index, run_id, *model.scales)
            + model.codes.tobytes())
    return append_crc(body)


def decode_global(payload: bytes) -> GlobalModel:
    validate_crc(payload, GLOBAL_BYTES)
    level_index, round_index, run_id, s0, s1, s2, s3 = _GLOBAL_HEAD.unpack_from(payload, 0)
    codes = np.frombuffer(
        payload, dtype=np.int8, count=N_PARAMS, offset=_GLOBAL_HEAD.size
    ).copy()
    crc = struct.unpack_from("<I", payload, GLOBAL_BYTES - CRC_BYTES)[0]
    scales = (s0, s1, s2, s3)
    _check_scales(scales, "GLOBAL")
    _check_codes(codes, "GLOBAL")
    return GlobalModel(level_index, round_index, run_id, QuantisedModel(codes, scales), crc)


# ---------------------------------------------------------------------------
# Control messages
# ---------------------------------------------------------------------------
def encode_repair_request(round_id: int, position: int, missing_mask: int) -> bytes:
    """UPDATE_REPAIR: the v2 2-byte missing-fragment mask plus the position byte."""
    if not 0 < missing_mask <= ALL_UPDATE_FRAGMENTS:
        raise ProtocolError(
            f"repair mask 0x{missing_mask:04X} is empty or names fragments "
            f"outside 0..{UPDATE_FRAGMENTS - 1}"
        )
    if not 0 <= position <= MAX_POSITION:
        raise ProtocolError(f"position {position} is outside 0..{MAX_POSITION}")
    return encode_frame(
        MSG_UPDATE_REPAIR,
        round_id,
        0,
        UPDATE_FRAGMENTS,
        missing_mask.to_bytes(2, "little") + bytes((position,)),
    )


def decode_repair_request(frame: Frame) -> tuple[int, int]:
    """(missing_mask, position) of an UPDATE_REPAIR frame."""
    if frame.message_type != MSG_UPDATE_REPAIR or len(frame.data) != 3:
        raise ProtocolError("UPDATE_REPAIR carries a 2-byte mask and a position byte (3 bytes)")
    mask = int.from_bytes(frame.data[:2], "little")
    if mask == 0 or mask & ~ALL_UPDATE_FRAGMENTS:
        raise ProtocolError(f"UPDATE_REPAIR mask 0x{mask:04X} is empty or has bits outside 0..2")
    return mask, frame.data[2]


def encode_class_c_ready(round_id: int, uplink_dr: int, rx2_dr: int) -> bytes:
    return encode_frame(
        MSG_CLASS_C_READY, round_id, 0, GLOBAL_FRAGMENTS, bytes((uplink_dr, rx2_dr))
    )


def encode_global_status(round_id: int, missing_mask: int) -> bytes:
    """GLOBAL_STATUS (rev 3.2): a 2-byte bitmap of the GLOBAL fragments still MISSING."""
    if not 0 <= missing_mask <= ALL_GLOBAL_FRAGMENTS:
        raise ProtocolError(f"bitmap 0x{missing_mask:04X} has bits outside 0..2")
    return encode_frame(
        MSG_GLOBAL_STATUS, round_id, 0, GLOBAL_FRAGMENTS, missing_mask.to_bytes(2, "little"),
    )


def decode_global_status(frame: Frame) -> int:
    if frame.message_type != MSG_GLOBAL_STATUS or len(frame.data) != 2:
        raise ProtocolError("GLOBAL_STATUS carries exactly a 2-byte bitmap")
    mask = int.from_bytes(frame.data, "little")
    if mask & ~ALL_GLOBAL_FRAGMENTS:
        raise ProtocolError(f"GLOBAL_STATUS bitmap 0x{mask:04X} has bits outside 0..2")
    return mask


@dataclass(frozen=True)
class BoardHello:
    board_number: int
    n_machines: int
    fw_version: int
    flags: int
    run_id: int = 0             # run_id held in the board's flash (0 = none)

    @property
    def sd_mounted(self) -> bool:
        return bool(self.flags & HELLO_SD_MOUNTED)

    @property
    def self_check_passed(self) -> bool:
        return bool(self.flags & HELLO_SELF_CHECK_PASSED)

    @property
    def imu_source(self) -> bool:
        return bool(self.flags & HELLO_IMU_SOURCE)


def encode_board_hello(board_number: int, n_machines: int, fw_version: int, flags: int,
                       run_id: int = 0) -> bytes:
    return encode_frame(
        MSG_BOARD_HELLO, NO_ROUND, 0, 1,
        _HELLO.pack(board_number, n_machines, fw_version, flags, run_id),
    )


def decode_board_hello(frame: Frame) -> BoardHello:
    if frame.message_type != MSG_BOARD_HELLO or len(frame.data) != _HELLO.size:
        raise ProtocolError("BOARD_HELLO carries exactly 6 bytes")
    board, machines, fw, flags, run_id = _HELLO.unpack(frame.data)
    return BoardHello(board, machines, fw, flags, run_id)


@dataclass(frozen=True)
class Alarm:
    position: int
    level_index: int
    acquisition_index: int
    score: float
    threshold: float


def encode_alarm(position: int, level_index: int, acquisition_index: int,
                 score: float, threshold: float) -> bytes:
    return encode_frame(
        MSG_ALARM, NO_ROUND, 0, 1,
        _ALARM.pack(position, level_index, acquisition_index, score, threshold),
    )


def decode_alarm(frame: Frame) -> Alarm:
    if frame.message_type != MSG_ALARM or len(frame.data) != _ALARM.size:
        raise ProtocolError("ALARM carries exactly 14 bytes")
    return Alarm(*_ALARM.unpack(frame.data))


@dataclass(frozen=True)
class SelfCheck:
    position: int
    rows_checked: int
    max_abs_diff: float
    median_abs_diff: float


def encode_self_check(position: int, rows_checked: int,
                      max_abs_diff: float, median_abs_diff: float) -> bytes:
    return encode_frame(
        MSG_SELF_CHECK, NO_ROUND, 0, 1,
        _SELF_CHECK.pack(position, rows_checked, max_abs_diff, median_abs_diff),
    )


def decode_self_check(frame: Frame) -> SelfCheck:
    if frame.message_type != MSG_SELF_CHECK or len(frame.data) != _SELF_CHECK.size:
        raise ProtocolError("SELF_CHECK carries exactly 11 bytes")
    return SelfCheck(*_SELF_CHECK.unpack(frame.data))


@dataclass(frozen=True)
class ReplayDone:
    position: int
    level_mask: int            # bit i = level_index i was scored
    acquisitions_scored: int
    alarms_latched: int

    def levels(self) -> list[int]:
        return [i for i in range(len(HARDWARE_LEVELS)) if self.level_mask >> i & 1]


def encode_replay_done(position: int, level_mask: int, acquisitions_scored: int,
                       alarms_latched: int) -> bytes:
    return encode_frame(
        MSG_REPLAY_DONE, NO_ROUND, 0, 1,
        _REPLAY_DONE.pack(position, level_mask, acquisitions_scored, alarms_latched),
    )


def decode_replay_done(frame: Frame) -> ReplayDone:
    if frame.message_type != MSG_REPLAY_DONE or len(frame.data) != _REPLAY_DONE.size:
        raise ProtocolError("REPLAY_DONE carries exactly 7 bytes")
    return ReplayDone(*_REPLAY_DONE.unpack(frame.data))
