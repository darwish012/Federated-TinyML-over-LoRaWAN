"""The roster: which DevEUIs this deployment authorises (PROTOCOL_V4.md section 6).

    {"devices": [{"dev_eui": "70B3D57ED0000001", "provisioned_utc": "2026-10-01T09:00:00+00:00"}]}

DevEUIs are 16 hex digits, most significant byte first (as the board prints them in
``TFPROV HELLO`` and as TTS shows them). The file holds no keys. The provisioning
tool appends to it; the server only reads it, and re-reads it whenever the file
changes, so a board can be provisioned while a run is in progress.

The roster, not a quota, is the authorisation: only a rostered DevEUI can be
registered in The Things Stack and only a rostered DevEUI can become a member of a run.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading

_HEX16 = re.compile(r"^[0-9A-Fa-f]{16}$")


class RosterError(ValueError):
    """roster.json is unreadable or malformed."""


def normalise_dev_eui(value: str) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("eui-"):
        text = text[4:]
    if not _HEX16.match(text):
        raise RosterError(f"DevEUI {value!r} must be exactly 16 hexadecimal digits")
    return text.upper()


class Roster:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._devices: dict[str, dict] = {}
        self._mtime: tuple | None = None
        self.reload(force=True)

    # -- reading ---------------------------------------------------------
    def reload(self, force: bool = False) -> bool:
        """Re-read the file if it changed. Returns True when the contents were reloaded."""
        with self._lock:
            try:
                stat = self.path.stat()
                mtime = (stat.st_mtime_ns, stat.st_size)     # size too: two writes can share a timestamp
            except FileNotFoundError:
                changed = bool(self._devices) or self._mtime is not None
                self._devices, self._mtime = {}, None
                return changed
            if not force and mtime == self._mtime:
                return False
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                devices = {}
                for entry in data["devices"]:
                    devices[normalise_dev_eui(entry["dev_eui"])] = dict(entry)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RosterError(f"{self.path} is not a valid roster: {exc}") from exc
            self._devices, self._mtime = devices, mtime
            return True

    def contains(self, dev_eui: str) -> bool:
        try:
            key = normalise_dev_eui(dev_eui)
        except RosterError:
            return False
        with self._lock:
            return key in self._devices

    def __contains__(self, dev_eui: str) -> bool:
        return self.contains(dev_eui)

    def dev_euis(self) -> list[str]:
        with self._lock:
            return sorted(self._devices)

    def __len__(self) -> int:
        with self._lock:
            return len(self._devices)

    # -- writing (provision_board.py) --------------------------------------
    def add(self, dev_eui: str, provisioned_utc: str | None = None) -> bool:
        """Append a DevEUI (idempotent). Returns True if it was not yet listed."""
        key = normalise_dev_eui(dev_eui)
        stamp = provisioned_utc or datetime.now(timezone.utc).isoformat()
        with self._lock:
            new = key not in self._devices
            self._devices[key] = {"dev_eui": key, "provisioned_utc": stamp}
            self._save_locked()
        return new

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"devices": [self._devices[k] for k in sorted(self._devices)]}
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, self.path)
        stat = self.path.stat()
        self._mtime = (stat.st_mtime_ns, stat.st_size)
