"""Write (or clear) the Class C multicast session on both sides at once.

Multicast needs four values kept consistent across two files:

    firmware/tinyfl_node/lab_credentials.h   DevAddr + NwkSKey + AppSKey
    server/config.toml                       the group's TTS device id

Hand-editing these is exactly how the JoinEUI byte-order mistake happened, so
this script validates each field and writes both files itself. The two session
keys are read with getpass so they never reach shell history.

Unlike JoinEUI, NOTHING here is byte-reversed: DevAddr is written as a plain
32-bit number and the keys go in exactly as The Things Stack displays them.

    python configure_multicast.py --group-id tinyfl-v91-global
    python configure_multicast.py --disable
    python configure_multicast.py --show
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parent
CREDENTIALS = SERVER.parent / "firmware" / "tinyfl_node" / "lab_credentials.h"
CONFIG = SERVER / "config.toml"

DISABLED_DEVADDR = "0x00000000u"
ZERO_KEY = "{" + ", ".join(["0x00"] * 16) + "}"


def _clean_hex(text: str) -> str:
    """Strip the separators TTS and copy-paste tend to introduce."""
    return re.sub(r"(?i)^0x|[\s:,-]", "", text.strip())


def parse_devaddr(text: str) -> str:
    digits = _clean_hex(text)
    if not re.fullmatch(r"(?i)[0-9a-f]{8}", digits):
        raise ValueError(
            f"DevAddr must be 8 hex digits (4 bytes), got {len(digits)}: {text!r}"
        )
    return f"0x{digits.upper()}u"


def parse_key(text: str, name: str) -> str:
    digits = _clean_hex(text)
    if not re.fullmatch(r"(?i)[0-9a-f]{32}", digits):
        raise ValueError(
            f"{name} must be 32 hex digits (16 bytes), got {len(digits)}"
        )
    octets = [f"0x{digits[i:i + 2].upper()}" for i in range(0, 32, 2)]
    return "{" + ", ".join(octets) + "}"


def _replace_define(text: str, macro: str, value: str) -> str:
    pattern = re.compile(rf"^#define\s+{macro}\s+.*$", re.MULTILINE)
    if not pattern.search(text):
        raise SystemExit(f"{CREDENTIALS} has no #define {macro}")
    return pattern.sub(f"#define {macro} {value}", text, count=1)


def write_firmware(devaddr: str, nwkskey: str, appskey: str) -> None:
    text = CREDENTIALS.read_text(encoding="utf-8")
    text = _replace_define(text, "TINYFL_MULTICAST_DEVADDR", devaddr)
    text = _replace_define(text, "TINYFL_MULTICAST_NWKSKEY", nwkskey)
    text = _replace_define(text, "TINYFL_MULTICAST_APPSKEY", appskey)
    CREDENTIALS.write_text(text, encoding="utf-8", newline="\n")


def write_config(group_id: str, gateway_ids=None) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    pattern = re.compile(r'^multicast_device_id\s*=\s*".*"$', re.MULTILINE)
    if not pattern.search(text):
        raise SystemExit(f"{CONFIG} has no multicast_device_id line")
    text = pattern.sub(f'multicast_device_id = "{group_id}"', text, count=1)

    if gateway_ids is not None:
        # Writing the group id without a gateway produces a config that
        # fails at startup, so the two always move together.
        rendered = ", ".join(f'"{g}"' for g in gateway_ids)
        gw = re.compile(r"^multicast_gateway_ids\s*=\s*\[.*\]$", re.MULTILINE)
        if not gw.search(text):
            raise SystemExit(f"{CONFIG} has no multicast_gateway_ids line")
        text = gw.sub(f"multicast_gateway_ids = [{rendered}]", text, count=1)

    CONFIG.write_text(text, encoding="utf-8", newline="\n")


def show() -> None:
    creds = CREDENTIALS.read_text(encoding="utf-8")
    devaddr = re.search(r"^#define\s+TINYFL_MULTICAST_DEVADDR\s+(\S+)", creds,
                        re.MULTILINE)
    nwk = re.search(r"^#define\s+TINYFL_MULTICAST_NWKSKEY\s+(.*)$", creds,
                    re.MULTILINE)
    app = re.search(r"^#define\s+TINYFL_MULTICAST_APPSKEY\s+(.*)$", creds,
                    re.MULTILINE)
    group = re.search(r'^multicast_device_id\s*=\s*"(.*)"$',
                      CONFIG.read_text(encoding="utf-8"), re.MULTILINE)

    devaddr_value = devaddr.group(1) if devaddr else "?"
    enabled = devaddr_value not in ("0x00000000u", "0")
    print(f"firmware DevAddr    : {devaddr_value}")
    # Keys are never printed; only whether they look populated.
    for label, match in (("NwkSKey", nwk), ("AppSKey", app)):
        state = "set" if match and match.group(1).strip() != ZERO_KEY else "not set"
        print(f"firmware {label}    : {state}")
    print(f"server group id     : {group.group(1) if group else '?'!r}")
    print()
    print("MULTICAST IS " + ("ENABLED" if enabled else "DISABLED (unicast)"))
    if enabled:
        print("Both boards must be reflashed for the firmware side to take effect.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group-id",
                        help="the multicast group's device id in The Things Stack")
    parser.add_argument("--devaddr", help="multicast DevAddr, 8 hex digits")
    parser.add_argument("--gateway", action="append", metavar="GATEWAY_ID",
                        help="gateway to route group downlinks through; "
                             "repeat for several. Required with --group-id: "
                             "a group has no uplinks, so the network server "
                             "cannot otherwise find a downlink path.")
    parser.add_argument("--disable", action="store_true",
                        help="clear multicast and return to unicast")
    parser.add_argument("--show", action="store_true",
                        help="report the current state without changing it")
    args = parser.parse_args()

    if args.show:
        show()
        return 0

    if args.disable:
        write_firmware(DISABLED_DEVADDR, ZERO_KEY, ZERO_KEY)
        write_config("")
        print("Multicast disabled on both sides; reflash the boards.")
        return 0

    if not args.group_id:
        parser.error("--group-id is required (or use --disable / --show)")
    if not args.gateway:
        parser.error(
            "--gateway is required with --group-id. A multicast group never "
            "uplinks, so without a named gateway every downlink fails with "
            "no_downlink_path."
        )

    try:
        devaddr_text = args.devaddr or input("Multicast DevAddr (8 hex digits): ")
        devaddr = parse_devaddr(devaddr_text)
        nwkskey = parse_key(getpass.getpass("Multicast NwkSKey (32 hex digits): "),
                            "NwkSKey")
        appskey = parse_key(getpass.getpass("Multicast AppSKey (32 hex digits): "),
                            "AppSKey")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if nwkskey == appskey:
        # Not fatal -- some stacks allow it -- but it is far more often a paste
        # slip than a deliberate choice, and it fails as a silent decrypt error.
        print("WARNING: NwkSKey and AppSKey are identical; check you pasted both.")

    write_firmware(devaddr, nwkskey, appskey)
    write_config(args.group_id, args.gateway)
    print(f"Multicast configured for group {args.group_id!r}, DevAddr {devaddr}.")
    print("Keys were not echoed and are not in shell history.")
    print()
    print("Next: reflash EVERY board, then restart the server.")
    print("Expect 11 downlinks total instead of 11 per node.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
