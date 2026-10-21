"""Per-device key derivation and the provisioning CRC (PROTOCOL_V4.md section 6).

    AppKey(DevEUI) = AES-128-ECB(master_key, DevEUI || 0x00 x 8)

DevEUI is the 8 raw bytes of the 16-hex-digit identifier, most significant byte first,
exactly as the board prints it in ``TFPROV HELLO`` and as TTS shows it. The master key
never leaves the operator PC and comes from an environment variable only; nothing in
this module logs or returns a key in an error message.
"""

from __future__ import annotations

import os
import re
import zlib

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

_HEX = re.compile(r"^[0-9A-Fa-f]+$")

# TFPROV OK <crc32 hex> covers the five stored values as raw bytes, in the order of the
# KEYS line: joineui(8) | appkey(16) | mc_devaddr(4) | mc_nwkskey(16) | mc_appskey(16).
KEYS_CRC_BYTES = 8 + 16 + 4 + 16 + 16        # 60 (PROTOCOL_V4.md 6.1 says "56"; see README)


class KeyError_(ValueError):
    """A key or identifier is missing or malformed (the message never contains key material)."""


def hex_bytes(value: str, nbytes: int, what: str) -> bytes:
    """Decode exactly ``nbytes`` bytes of hex; the error names ``what`` but never echoes the value."""
    text = "".join(str(value or "").split()).replace("-", "").replace(":", "")
    if len(text) != nbytes * 2 or not _HEX.match(text):
        raise KeyError_(f"{what} must be exactly {nbytes * 2} hexadecimal digits")
    return bytes.fromhex(text)


def master_key_from_env(env: dict | None = None, name: str = "TINYFL_MASTER_KEY") -> bytes:
    env = os.environ if env is None else env
    value = env.get(name, "")
    if not str(value).strip():
        raise KeyError_(f"set ${name} to the 32-hex-digit master key")
    return hex_bytes(value, 16, f"${name}")


def derive_app_key(master_key: bytes, dev_eui: str) -> bytes:
    """AES-128-ECB(master, DevEUI || 0x00 x 8): the device's AppKey."""
    if len(master_key) != 16:
        raise KeyError_("the master key must be 16 bytes")
    block = hex_bytes(dev_eui, 8, "DevEUI") + bytes(8)
    encryptor = Cipher(algorithms.AES(master_key), modes.ECB()).encryptor()
    return encryptor.update(block) + encryptor.finalize()


def provisioning_crc(joineui: bytes, appkey: bytes, mc_devaddr: bytes, mc_nwkskey: bytes,
                     mc_appskey: bytes) -> int:
    """zlib CRC32 of the raw bytes the board stores, in KEYS order."""
    if (len(joineui), len(appkey), len(mc_devaddr), len(mc_nwkskey), len(mc_appskey)) != (
            8, 16, 4, 16, 16):
        raise KeyError_("provisioning values have the wrong lengths")
    return zlib.crc32(joineui + appkey + mc_devaddr + mc_nwkskey + mc_appskey) & 0xFFFFFFFF
