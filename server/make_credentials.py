"""Convert TTS OTAA values into an MCCI-LMIC node_credentials.h file."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path
import re


def parse_hex(value: str, bytes_required: int, name: str) -> bytes:
    compact = re.sub(r"[^0-9A-Fa-f]", "", value)
    if len(compact) != bytes_required * 2:
        raise ValueError(f"{name} must contain exactly {bytes_required * 2} hex digits")
    return bytes.fromhex(compact)


def c_bytes(value: bytes) -> str:
    return ", ".join(f"0x{byte:02X}" for byte in value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--join-eui")
    parser.add_argument("--dev-eui")
    parser.add_argument("--app-key")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    join_eui = parse_hex(args.join_eui or input("JoinEUI: "), 8, "JoinEUI")
    dev_eui = parse_hex(args.dev_eui or input("DevEUI: "), 8, "DevEUI")
    app_key = parse_hex(
        args.app_key or getpass.getpass("AppKey (input hidden): "), 16, "AppKey"
    )
    text = f"""#pragma once

#define TINYFL_CREDENTIALS_CONFIGURED 1
#define TINYFL_JOIN_EUI {{{c_bytes(join_eui[::-1])}}}
#define TINYFL_DEV_EUI  {{{c_bytes(dev_eui[::-1])}}}
#define TINYFL_APP_KEY  {{{c_bytes(app_key)}}}
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
