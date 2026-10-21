"""Wire protocol for the two-LilyGO V91 transport experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import struct
import zlib


MAGIC = 0xF1
VERSION = 0x02
MSG_UPDATE_FRAGMENT = 0x01
MSG_CLASS_C_READY = 0x02
MSG_GLOBAL_FRAGMENT = 0x03
MSG_GLOBAL_STATUS = 0x04
# Downlink asking a node to resend the UPDATE fragments the server never got.
# Unconfirmed uplinks have no retransmission of their own, so without this a
# single lost fragment discards the whole 2202-byte update.
MSG_UPDATE_REPAIR = 0x05
F_PORT = 10

HEADER_BYTES = 7
FRAGMENT_BODY_BYTES = 214
MAX_APPLICATION_BYTES = HEADER_BYTES + FRAGMENT_BODY_BYTES
UPDATE_BYTES = 2202
GLOBAL_BYTES = 2241
CRC_BYTES = 4
UPDATE_DATA_BYTES = UPDATE_BYTES - CRC_BYTES
GLOBAL_DATA_BYTES = GLOBAL_BYTES - CRC_BYTES
UPDATE_FRAGMENTS = math.ceil(UPDATE_BYTES / FRAGMENT_BODY_BYTES)
GLOBAL_FRAGMENTS = math.ceil(GLOBAL_BYTES / FRAGMENT_BODY_BYTES)
ALL_GLOBAL_FRAGMENTS = (1 << GLOBAL_FRAGMENTS) - 1
ALL_UPDATE_FRAGMENTS = (1 << UPDATE_FRAGMENTS) - 1

assert MAX_APPLICATION_BYTES == 221
assert UPDATE_FRAGMENTS == 11
assert GLOBAL_FRAGMENTS == 11


class ProtocolError(ValueError):
    """The received application frame violates the experiment contract."""


@dataclass(frozen=True)
class Frame:
    message_type: int
    round_id: int
    fragment_index: int
    fragment_count: int
    data: bytes


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
        (
            MAGIC,
            VERSION,
            message_type,
            round_id,
            fragment_index,
            fragment_count,
            len(data),
        )
    ) + data


def decode_frame(payload: bytes) -> Frame:
    if not HEADER_BYTES <= len(payload) <= MAX_APPLICATION_BYTES:
        raise ProtocolError(f"application frame length {len(payload)} is outside 7..221")
    if payload[0] != MAGIC or payload[1] != VERSION:
        raise ProtocolError("wrong experiment magic or protocol version")
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
    if frame.message_type not in {
        MSG_UPDATE_FRAGMENT,
        MSG_CLASS_C_READY,
        MSG_GLOBAL_FRAGMENT,
        MSG_GLOBAL_STATUS,
        MSG_UPDATE_REPAIR,
    }:
        raise ProtocolError(f"unsupported message type 0x{frame.message_type:02X}")
    return frame


def encode_repair_request(round_id: int, missing_mask: int) -> bytes:
    """Build the downlink that names the UPDATE fragments still missing."""
    if not 0 < missing_mask <= ALL_UPDATE_FRAGMENTS:
        raise ProtocolError(
            f"repair mask 0x{missing_mask:04X} is empty or names fragments "
            f"outside 0..{UPDATE_FRAGMENTS - 1}"
        )
    return encode_frame(
        MSG_UPDATE_REPAIR,
        round_id,
        0,
        UPDATE_FRAGMENTS,
        missing_mask.to_bytes(2, "little"),
    )


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
            payload[index * FRAGMENT_BODY_BYTES : (index + 1) * FRAGMENT_BODY_BYTES],
        )
        for index in range(count)
    ]


def append_crc(payload_without_crc: bytes) -> bytes:
    return payload_without_crc + struct.pack("<I", zlib.crc32(payload_without_crc) & 0xFFFFFFFF)


def validate_crc(payload: bytes, expected_length: int) -> int:
    if len(payload) != expected_length or len(payload) < CRC_BYTES:
        raise ProtocolError(
            f"logical payload has {len(payload)} bytes; expected {expected_length}"
        )
    expected_crc = struct.unpack_from("<I", payload, len(payload) - CRC_BYTES)[0]
    actual_crc = zlib.crc32(payload[:-CRC_BYTES]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ProtocolError(
            f"logical payload CRC mismatch: expected {expected_crc:08X}, got {actual_crc:08X}"
        )
    return actual_crc


def _u32(value: int) -> int:
    return value & 0xFFFFFFFF


def dev_eui_from_device_id(device_id: str) -> bytes:
    """Recover the 8 DevEUI bytes (MSB first) from a TTS device_id."""
    text = device_id[4:] if device_id.startswith("eui-") else device_id
    try:
        dev_eui = bytes.fromhex(text)
    except ValueError as exc:
        raise ProtocolError(f"device id {device_id!r} is not eui-<16 hex digits>") from exc
    if len(dev_eui) != 8:
        raise ProtocolError(f"device id {device_id!r} does not carry 8 DevEUI bytes")
    return dev_eui


def node_seed(dev_eui: bytes) -> int:
    """FNV-1a over the DevEUI, MSB first. Mirrors nodeSeed() in the firmware.

    Seeding from the DevEUI rather than a compile-time node number is what
    lets every board run one identical binary.
    """
    seed = 0x811C9DC5
    for byte in dev_eui:
        seed = _u32((seed ^ byte) * 0x01000193)
    return seed


def synthetic_update_byte(seed: int, round_id: int, position: int) -> int:
    """Match the deterministic byte generator in the LilyGO firmware."""

    value = 0x9E3779B9
    value ^= _u32(seed * 0x85EBCA6B)
    value ^= _u32(round_id * 0xC2B2AE35)
    value = _u32(value + _u32(position * 0x27D4EB2D))
    value ^= value >> 16
    value = _u32(value * 0x7FEB352D)
    value ^= value >> 15
    value = _u32(value * 0x846CA68B)
    value ^= value >> 16
    return value & 0xFF


def make_synthetic_update(seed: int, round_id: int) -> bytes:
    data = bytes(
        synthetic_update_byte(seed, round_id, position)
        for position in range(UPDATE_DATA_BYTES)
    )
    return append_crc(data)


def expected_update_for_device(device_id: str, round_id: int) -> bytes:
    """The exact 2202 bytes this device should have sent, derived from its DevEUI."""
    return make_synthetic_update(node_seed(dev_eui_from_device_id(device_id)), round_id)


def make_global_payload(updates: dict[str, bytes], round_id: int) -> bytes:
    """Create a deterministic simulated aggregate derived from every update."""

    digest = hashlib.sha256()
    digest.update(b"tinyfl-v91-two-node-global\0")
    digest.update(bytes((round_id,)))
    for device_id in sorted(updates):
        encoded_id = device_id.encode("utf-8")
        digest.update(struct.pack("<H", len(encoded_id)))
        digest.update(encoded_id)
        digest.update(updates[device_id])
    seed = digest.digest()

    output = bytearray()
    counter = 0
    while len(output) < GLOBAL_DATA_BYTES:
        output.extend(hashlib.sha256(seed + struct.pack("<I", counter)).digest())
        counter += 1
    return append_crc(bytes(output[:GLOBAL_DATA_BYTES]))


@dataclass
class Reassembler:
    expected_type: int
    expected_round: int
    expected_count: int
    expected_bytes: int
    parts: dict[int, bytes] = field(default_factory=dict)

    def missing_mask(self) -> int:
        """Bitmask of fragment indices not yet received."""
        return sum(
            1 << index
            for index in range(self.expected_count)
            if index not in self.parts
        )

    def add(self, frame: Frame) -> bytes | None:
        if frame.message_type != self.expected_type or frame.round_id != self.expected_round:
            raise ProtocolError("fragment belongs to another message or round")
        if frame.fragment_count != self.expected_count:
            raise ProtocolError(
                f"fragment count is {frame.fragment_count}; expected {self.expected_count}"
            )
        if not 0 <= frame.fragment_index < self.expected_count:
            raise ProtocolError("fragment index is outside its declared count")
        offset = frame.fragment_index * FRAGMENT_BODY_BYTES
        expected_body = min(FRAGMENT_BODY_BYTES, self.expected_bytes - offset)
        if len(frame.data) != expected_body:
            raise ProtocolError(
                f"fragment {frame.fragment_index} has {len(frame.data)} data bytes; "
                f"expected {expected_body}"
            )
        previous = self.parts.get(frame.fragment_index)
        if previous is not None and previous != frame.data:
            raise ProtocolError("duplicate fragment changed its bytes")
        self.parts[frame.fragment_index] = frame.data
        if len(self.parts) != self.expected_count:
            return None
        payload = b"".join(self.parts[index] for index in range(self.expected_count))
        if len(payload) != self.expected_bytes:
            raise ProtocolError("reassembled payload length is wrong")
        return payload
