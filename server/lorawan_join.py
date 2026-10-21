"""Parse and cryptographically verify LoRaWAN OTAA join requests.

Ported from the Smart Building Management System (app/services/lorawan_join.py
and lorawan_identity.py, read only). Nothing here performs I/O.

The enrollment listener sees join requests from DevEUIs the network has never
met. A DevEUI is only trusted once it proves it holds this deployment's shared
AppKey: the join request carries a MIC that is an AES-128-CMAC over its own
body under that key, and only a holder of the key can produce one that
verifies. That check is the security boundary of auto-enrollment.

What the MIC does NOT give is replay protection: a captured join request stays
valid forever. That is acceptable here because replaying one can only enrol a
DevEUI that already held the AppKey (and could have joined by itself); the
Join Server's own DevNonce tracking rejects the replayed join.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import re

from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.cmac import CMAC

# PHYPayload of a join request (LoRaWAN 1.0.x section 6.2.4):
#   MHDR(1) | JoinEUI(8, LE) | DevEUI(8, LE) | DevNonce(2, LE) | MIC(4)
JOIN_REQUEST_LENGTH = 23
MHDR_MTYPE_MASK = 0xE0
MHDR_MTYPE_JOIN_REQUEST = 0x00
_MIC_LENGTH = 4


@dataclass(frozen=True)
class JoinRequest:
    """A parsed join request. EUIs are MSB-first upper-case hex, as TTS shows them."""

    join_eui: str
    dev_eui: str
    dev_nonce: int
    mic: bytes
    raw: bytes

    @property
    def signed_body(self) -> bytes:
        """The bytes the MIC covers: everything except the MIC itself."""
        return self.raw[:-_MIC_LENGTH]


def parse_join_request(payload: bytes) -> JoinRequest | None:
    """Decode a PHYPayload as a join request, or None if it is not one.

    Returns None instead of raising because it is fed every uplink the gateway
    hears: data frames and truncated frames are ordinary input.
    """
    if not isinstance(payload, (bytes, bytearray)):
        return None
    payload = bytes(payload)
    if len(payload) != JOIN_REQUEST_LENGTH:
        return None
    if (payload[0] & MHDR_MTYPE_MASK) != MHDR_MTYPE_JOIN_REQUEST:
        return None
    return JoinRequest(
        join_eui=payload[1:9][::-1].hex().upper(),     # little-endian on air
        dev_eui=payload[9:17][::-1].hex().upper(),
        dev_nonce=int.from_bytes(payload[17:19], "little"),
        mic=payload[19:23],
        raw=payload,
    )


def compute_join_request_mic(signed_body: bytes, app_key: bytes) -> bytes:
    """MIC = aes128_cmac(AppKey, MHDR | JoinEUI | DevEUI | DevNonce)[0:4]."""
    if len(app_key) != 16:
        raise ValueError("AppKey must be exactly 16 bytes")
    mac = CMAC(algorithms.AES(app_key))
    mac.update(signed_body)
    return mac.finalize()[:_MIC_LENGTH]


def verify_join_request_mic(request: JoinRequest, app_key: bytes) -> bool:
    """True when this join request was signed with ``app_key``.

    Constant-time comparison, and a malformed key is a failed check rather than
    an exception that could escape into the listener loop.
    """
    try:
        expected = compute_join_request_mic(request.signed_body, app_key)
    except Exception:  # noqa: BLE001
        return False
    return hmac.compare_digest(expected, request.mic)


def build_join_request(
    app_key: bytes, join_eui: str, dev_eui: str, dev_nonce: int, mhdr: int = 0x00
) -> bytes:
    """A correctly signed join request, the way a node would send it (tests, tools)."""
    body = (
        bytes([mhdr])
        + bytes.fromhex(normalize_hex(join_eui, 16, "join_eui"))[::-1]
        + bytes.fromhex(normalize_hex(dev_eui, 16, "dev_eui"))[::-1]
        + dev_nonce.to_bytes(2, "little")
    )
    return body + compute_join_request_mic(body, app_key)


def normalize_hex(value: str, expected_length: int, field_name: str) -> str:
    normalized = re.sub(r"[^0-9A-Fa-f]", "", str(value or "")).upper()
    if len(normalized) != expected_length:
        raise ValueError(
            f"{field_name} must contain exactly {expected_length} hexadecimal characters"
        )
    return normalized


def app_key_bytes(app_key_hex: str) -> bytes:
    """Decode the configured AppKey, rejecting anything that is not 16 bytes."""
    cleaned = "".join(str(app_key_hex or "").split()).replace("-", "").replace(":", "")
    try:
        raw = bytes.fromhex(cleaned)
    except ValueError as exc:
        raise ValueError("the AppKey must be valid hexadecimal") from exc
    if len(raw) != 16:
        raise ValueError("the AppKey must decode to exactly 16 bytes")
    return raw


def device_id_from_dev_eui(dev_eui: str) -> str:
    """TTS device id for this experiment: eui-<deveui lowercased>."""
    return f"eui-{normalize_hex(dev_eui, 16, 'dev_eui').lower()}"
