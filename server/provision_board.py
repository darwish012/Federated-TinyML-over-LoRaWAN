"""PC side of board provisioning (PROTOCOL_V4.md section 6.1).

    python provision_board.py --port COM5

The board runs ``tinyfl_provision``. Over 115200 8N1 ASCII lines:

  1. board  -> ``TFPROV HELLO <fw_version> <deveui 16 hex, MSB first>`` (once per second)
  2. this   -> ``TFPROV KEYS <joineui 16 hex> <appkey 32 hex> <mc_devaddr 8 hex>
               <mc_nwkskey 32 hex> <mc_appskey 32 hex>``
  3. board  -> ``TFPROV OK <crc32 hex of the stored raw bytes>`` or ``TFPROV ERR <reason>``

The AppKey is derived here, AppKey = AES-128-ECB(master, DevEUI || 0x00 x 8), from
$TINYFL_MASTER_KEY; the shared multicast session keys come from $TINYFL_MC_DEVADDR,
$TINYFL_MC_NWKSKEY and $TINYFL_MC_APPSKEY; the JoinEUI from config.toml
([enrollment] join_eui). The tool verifies the board's CRC and only then appends the
DevEUI to the roster (roster.json, no keys). It never prints a key: not the master key,
not the AppKey, not the multicast keys, and a board's ERR text is scrubbed of any of them.

``--dry-run`` talks to a built-in fake board instead of a serial port and does not write
the roster (pass ``--roster`` explicitly to also exercise that).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
import tomllib
from typing import Callable

from keys_v4 import (
    KeyError_,
    derive_app_key,
    hex_bytes,
    master_key_from_env,
    provisioning_crc,
)
from roster_v4 import Roster, normalise_dev_eui

BAUD = 115200
HELLO_TIMEOUT_SECONDS = 130.0        # the board waits up to 120 s for KEYS
REPLY_TIMEOUT_SECONDS = 20.0
ENV_MC_DEVADDR = "TINYFL_MC_DEVADDR"
ENV_MC_NWKSKEY = "TINYFL_MC_NWKSKEY"
ENV_MC_APPSKEY = "TINYFL_MC_APPSKEY"


class ProvisionError(Exception):
    """Provisioning failed. The message never contains key material."""


@dataclass(frozen=True)
class Provisioned:
    dev_eui: str
    fw_version: str
    new_in_roster: bool | None       # None when the roster was not written


@dataclass(frozen=True)
class KeyMaterial:
    """Everything the board is given, as raw bytes."""
    joineui: bytes
    mc_devaddr: bytes
    mc_nwkskey: bytes
    mc_appskey: bytes

    @classmethod
    def from_env(cls, joineui_hex: str, env: dict | None = None) -> "KeyMaterial":
        env = os.environ if env is None else env
        try:
            return cls(
                joineui=hex_bytes(joineui_hex, 8, "JoinEUI ([enrollment] join_eui)"),
                mc_devaddr=hex_bytes(env.get(ENV_MC_DEVADDR, ""), 4, f"${ENV_MC_DEVADDR}"),
                mc_nwkskey=hex_bytes(env.get(ENV_MC_NWKSKEY, ""), 16, f"${ENV_MC_NWKSKEY}"),
                mc_appskey=hex_bytes(env.get(ENV_MC_APPSKEY, ""), 16, f"${ENV_MC_APPSKEY}"),
            )
        except KeyError_ as exc:
            raise ProvisionError(str(exc)) from None


def keys_line(joineui: bytes, appkey: bytes, mc: KeyMaterial) -> str:
    return "TFPROV KEYS " + " ".join(
        part.hex() for part in (joineui, appkey, mc.mc_devaddr, mc.mc_nwkskey, mc.mc_appskey)) + "\n"


def _scrub(text: str, secrets: list[bytes]) -> str:
    for secret in secrets:
        for rendering in (secret.hex(), secret.hex().upper()):
            text = text.replace(rendering, "<redacted>")
    return text


def _readline(port) -> str:
    raw = port.readline()
    return raw.decode("ascii", errors="replace").strip() if raw else ""


def read_hello(port, timeout: float, clock: Callable[[], float]) -> tuple[str, str]:
    """Wait for ``TFPROV HELLO <fw> <deveui>`` and return (fw_version, DevEUI upper-case hex)."""
    deadline = clock() + timeout
    while clock() < deadline:
        line = _readline(port)
        if not line.startswith("TFPROV"):
            continue                       # boot noise, other firmware output
        parts = line.split()
        if parts[1:2] == ["ERR"]:
            raise ProvisionError("the board reported an error before provisioning: "
                                 + " ".join(parts[2:]))
        if parts[1:2] == ["HELLO"] and len(parts) == 4:
            try:
                return parts[2], normalise_dev_eui(parts[3])
            except ValueError:
                raise ProvisionError("TFPROV HELLO carried a malformed DevEUI") from None
    raise ProvisionError(
        f"no TFPROV HELLO within {timeout:.0f} s: is tinyfl_provision flashed, the right port "
        "selected, and the board reset after the port was opened?")


def provision(port, material: KeyMaterial, master_key: bytes,
              roster: Roster | None = None, hello_timeout: float = HELLO_TIMEOUT_SECONDS,
              reply_timeout: float = REPLY_TIMEOUT_SECONDS, expected_dev_eui: str | None = None,
              out: Callable[[str], None] = print,
              clock: Callable[[], float] = time.monotonic) -> Provisioned:
    """Run the exchange on an open port; append to ``roster`` after the CRC verifies."""
    fw_version, dev_eui = read_hello(port, hello_timeout, clock)
    out(f"board found: DevEUI {dev_eui}, firmware {fw_version}")
    if expected_dev_eui is not None and normalise_dev_eui(expected_dev_eui) != dev_eui:
        raise ProvisionError(f"the board reports DevEUI {dev_eui}, not the expected "
                             f"{normalise_dev_eui(expected_dev_eui)}")
    appkey = derive_app_key(master_key, dev_eui)
    secrets = [master_key, appkey, material.mc_nwkskey, material.mc_appskey]
    expected_crc = provisioning_crc(material.joineui, appkey, material.mc_devaddr,
                                    material.mc_nwkskey, material.mc_appskey)
    port.write(keys_line(material.joineui, appkey, material).encode("ascii"))
    flush = getattr(port, "flush", None)
    if flush:
        flush()
    out("keys sent (not shown); waiting for the board to store and confirm them")

    deadline = clock() + reply_timeout
    while clock() < deadline:
        line = _readline(port)
        if not line.startswith("TFPROV"):
            continue
        parts = line.split()
        if parts[1:2] == ["ERR"]:
            reason = _scrub(" ".join(parts[2:]), secrets) or "no reason given"
            raise ProvisionError(f"the board refused the keys: {reason}")
        if parts[1:2] == ["OK"]:
            try:
                board_crc = int(parts[2], 16) if len(parts) >= 3 else None
            except ValueError:
                board_crc = None
            if board_crc is None:
                raise ProvisionError("TFPROV OK carried no valid CRC")
            if board_crc != expected_crc:
                raise ProvisionError(
                    "CRC mismatch: what the board stored is not what was sent; the board is NOT "
                    "added to the roster. Re-run provisioning")
            out("CRC verified: the board stored the keys")
            new = None
            if roster is not None:
                new = roster.add(dev_eui)
                out(f"roster updated: {dev_eui} {'added' if new else 'was already listed'} "
                    f"({len(roster)} devices)")
            return Provisioned(dev_eui, fw_version, new)
        # TFPROV HELLO repeats until the board has processed KEYS: ignore it
    raise ProvisionError(f"no TFPROV OK / ERR within {reply_timeout:.0f} s after sending the keys")


# ---------------------------------------------------------------------------
# A fake board for --dry-run and the tests
# ---------------------------------------------------------------------------
class FakeBoardSerial:
    """Speaks the board side of the exchange. ``fault`` injects a misbehaving board:

    None            behaves
    "badcrc"        replies OK with a CRC that does not match what it was sent
    "err"           replies ERR nvs_write_failed
    "err_leaks"     replies ERR with the AppKey hex inside the reason (must be scrubbed)
    "silent"        never prints HELLO
    "no_ok"         takes the keys and never confirms
    """

    def __init__(self, dev_eui: str = "70B3D57ED0000042", fw_version: str = "1",
                 fault: str | None = None, noise_lines: int = 2) -> None:
        self.dev_eui = normalise_dev_eui(dev_eui)
        self.fw_version = fw_version
        self.fault = fault
        self._noise = noise_lines
        self._outbox: list[str] = []
        self.received: list[str] = []
        self.stored: dict | None = None
        self.provisioned = False

    def readline(self) -> bytes:
        if self._outbox:
            return (self._outbox.pop(0) + "\n").encode("ascii")
        if self._noise > 0:
            self._noise -= 1
            return b"ets Jul 29 2019 12:21:46\n" if self._noise else b"rst:0x1 (POWERON_RESET)\n"
        if self.fault == "silent" or self.provisioned:
            return b""
        return f"TFPROV HELLO {self.fw_version} {self.dev_eui}\n".encode("ascii")

    def write(self, data: bytes) -> int:
        line = data.decode("ascii").strip()
        self.received.append("TFPROV KEYS <redacted>" if line.startswith("TFPROV KEYS") else line)
        parts = line.split()
        if parts[:2] != ["TFPROV", "KEYS"] or len(parts) != 7:
            self._outbox.append("TFPROV ERR bad_line")
            return len(data)
        try:
            raw = [bytes.fromhex(p) for p in parts[2:]]
        except ValueError:
            self._outbox.append("TFPROV ERR bad_hex")
            return len(data)
        if [len(r) for r in raw] != [8, 16, 4, 16, 16]:
            self._outbox.append("TFPROV ERR bad_length")
            return len(data)
        self.stored = dict(zip(("joineui", "appkey", "mcaddr", "mcnwk", "mcapp"), raw))
        crc = provisioning_crc(*raw)
        self.provisioned = True
        if self.fault == "err":
            self._outbox.append("TFPROV ERR nvs_write_failed")
        elif self.fault == "err_leaks":
            self._outbox.append(f"TFPROV ERR readback_mismatch {raw[1].hex()}")
        elif self.fault == "badcrc":
            self._outbox.append(f"TFPROV OK {crc ^ 0x1:08x}")
        elif self.fault == "no_ok":
            pass
        else:
            self._outbox.append(f"TFPROV OK {crc:08x}")
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _join_eui_from_config(path: Path) -> tuple[str, Path]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    join_eui = str(config.get("enrollment", {}).get("join_eui", ""))
    roster = Path(str(config.get("v4", {}).get("roster_path", "roster.json")))
    return join_eui, roster if roster.is_absolute() else path.resolve().parent / roster


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", help="serial port of the board, e.g. COM5")
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--roster", type=Path, default=None,
                        help="roster.json to append to (default: [v4] roster_path of the config)")
    parser.add_argument("--hello-timeout", type=float, default=HELLO_TIMEOUT_SECONDS)
    parser.add_argument("--expect-deveui", default=None,
                        help="abort unless the board reports this DevEUI")
    parser.add_argument("--dry-run", action="store_true",
                        help="use a built-in fake board; the roster is only written if --roster is given")
    parser.add_argument("--fake-deveui", default="70B3D57ED0000042")
    parser.add_argument("--fake-fault", default=None,
                        choices=["badcrc", "err", "err_leaks", "silent", "no_ok"])
    args = parser.parse_args(argv)

    try:
        join_eui, config_roster = _join_eui_from_config(args.config)
        material = KeyMaterial.from_env(join_eui)
        master = master_key_from_env()
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"cannot read {args.config}: {exc}", file=sys.stderr)
        return 2
    except (ProvisionError, KeyError_) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        port = FakeBoardSerial(args.fake_deveui, fault=args.fake_fault)
        roster = Roster(args.roster) if args.roster is not None else None
        print("DRY RUN: talking to a fake board; no serial port is opened"
              + ("" if roster is not None else ", roster not written"))
    else:
        if not args.port:
            print("error: --port is required (or use --dry-run)", file=sys.stderr)
            return 2
        try:
            import serial
        except ModuleNotFoundError:
            print("error: pyserial is not installed; run python -m pip install -r requirements.txt",
                  file=sys.stderr)
            return 2
        try:
            port = serial.Serial(args.port, args.baud, timeout=1)
        except serial.SerialException as exc:
            print(f"error: cannot open {args.port}: {exc}", file=sys.stderr)
            return 2
        roster = Roster(args.roster if args.roster is not None else config_roster)
    try:
        result = provision(port, material, master, roster, hello_timeout=args.hello_timeout,
                           expected_dev_eui=args.expect_deveui)
    except ProvisionError as exc:
        print(f"provisioning FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        port.close()
    print(f"provisioned {result.dev_eui}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
