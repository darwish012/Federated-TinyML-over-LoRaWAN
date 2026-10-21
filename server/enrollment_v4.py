"""Hardened automatic enrollment for protocol v4 (PROTOCOL_V4.md section 6).

Evolves enrollment.py (v3, left intact) in these ways:

  * a DevEUI is registered only if it is in the ROSTER (roster.json, written by
    provision_board.py); the roster, not a quota, is the authorisation;
  * every device has its OWN AppKey: AES-128-ECB(master_key, DevEUI || 0x00 x 8), the
    master key coming from an environment variable only;
  * the JoinRequest must be exactly 23 bytes with MHDR 0x00 (MType JoinRequest), the
    JoinEUI must match, and the MIC (AES-CMAC over the first 19 bytes) is compared in
    constant time under that device's derived key;
  * seen (DevEUI, DevNonce) pairs are persisted and replays rejected;
  * registration is idempotent, and rolls back what THIS call created if a later stage
    (Join Server, Network Server, Application Server) fails;
  * bad-MIC, unknown-DevEUI and replay events are logged at a bounded rate, with a count
    of what was suppressed.

The existing TTS API calls are kept: identity POST, then JS / NS / AS PUT with Class C
and RX2 DR5 (enrollment.TtsRegistrationClient payloads). No key is ever logged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from enum import Enum
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable

from enrollment import (
    EnrollmentConfig,
    EnrollmentRegistry,
    JoinListener,
    Transport,
    TtsError,
    TtsRegistrationClient,
    extract_raw_payload,
    urllib_transport,
    MAX_WINDOW_SECONDS,
)
from keys_v4 import KeyError_, derive_app_key, master_key_from_env
from lorawan_join import (
    JOIN_REQUEST_LENGTH,
    JoinRequest,
    device_id_from_dev_eui,
    normalize_hex,
    verify_join_request_mic,
)
from roster_v4 import Roster

EventLogger = Callable[[str, str, dict], None]
STRICT_JOIN_MHDR = 0x00          # MType 000 (JoinRequest), RFU 000, Major 00 (LoRaWAN R1)


class Outcome(str, Enum):
    REGISTERED = "registered"
    ALREADY_REGISTERED = "already_registered"
    WINDOW_CLOSED = "window_closed"
    NOT_ROSTERED = "not_rostered"
    JOIN_EUI_MISMATCH = "join_eui_mismatch"
    BAD_MIC = "bad_mic"
    REPLAYED_DEV_NONCE = "replayed_dev_nonce"
    MISCONFIGURED = "misconfigured"
    REGISTRATION_FAILED = "registration_failed"


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    dev_eui: str
    detail: str = ""

    @property
    def enrolled(self) -> bool:
        return self.outcome is Outcome.REGISTERED


# ---------------------------------------------------------------------------
# Strict JoinRequest parsing
# ---------------------------------------------------------------------------
def classify_join_frame(payload) -> tuple[str, JoinRequest | None]:
    """("join", request) for a well-formed JoinRequest, ("malformed", None) for something that
    claims to be one (MType 000) but is not exactly 23 bytes with MHDR 0x00, ("other", None)."""
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        return "other", None
    payload = bytes(payload)
    if (payload[0] & 0xE0) != 0x00:
        return "other", None
    if len(payload) != JOIN_REQUEST_LENGTH or payload[0] != STRICT_JOIN_MHDR:
        return "malformed", None
    return "join", JoinRequest(
        join_eui=payload[1:9][::-1].hex().upper(),
        dev_eui=payload[9:17][::-1].hex().upper(),
        dev_nonce=int.from_bytes(payload[17:19], "little"),
        mic=payload[19:23], raw=payload)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EnrollmentConfigV4(EnrollmentConfig):
    master_key_env: str = "TINYFL_MASTER_KEY"
    nonce_store_path: str = "join_nonces.jsonl"
    bad_mic_log_interval_seconds: float = 60.0

    @classmethod
    def from_config(cls, config: dict) -> "EnrollmentConfigV4":      # type: ignore[override]
        base = EnrollmentConfig.from_config(config)
        section = dict(config.get("enrollment", {}))
        kwargs = {f.name: getattr(base, f.name) for f in fields(EnrollmentConfig)}
        kwargs.update(
            master_key_env=str(section.get("master_key_env", "TINYFL_MASTER_KEY")),
            nonce_store_path=str(section.get("nonce_store_path", "join_nonces.jsonl")),
            bad_mic_log_interval_seconds=float(section.get("bad_mic_log_interval_seconds", 60.0)),
        )
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Persisted DevNonce store
# ---------------------------------------------------------------------------
class JoinNonceStore:
    """(DevEUI, DevNonce) pairs already seen, appended to a JSON-lines file so a restart of
    the server does not forget them. A torn last line (crash) is ignored."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = None if path is None else Path(path)
        self._lock = threading.Lock()
        self._seen: set = set()
        self._handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                        self._seen.add((str(row["dev_eui"]).upper(), int(row["dev_nonce"])))
                    except (ValueError, KeyError, TypeError):
                        continue
            self._handle = self.path.open("a", encoding="utf-8")

    def seen(self, dev_eui: str, dev_nonce: int) -> bool:
        with self._lock:
            return (dev_eui.upper(), dev_nonce) in self._seen

    def add(self, dev_eui: str, dev_nonce: int, when: float | None = None) -> bool:
        """Record a pair. Returns False if it was already there."""
        key = (dev_eui.upper(), dev_nonce)
        with self._lock:
            if key in self._seen:
                return False
            self._seen.add(key)
            if self._handle is not None:
                self._handle.write(json.dumps(
                    {"dev_eui": key[0], "dev_nonce": dev_nonce, "t": int(when or time.time())}) + "\n")
                self._handle.flush()
                os.fsync(self._handle.fileno())
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


class RateLimitedLog:
    """At most one log line per key per interval, with the count of what was suppressed."""

    def __init__(self, log: EventLogger, interval_seconds: float,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.log = log
        self.interval = interval_seconds
        self.clock = clock
        self._state: dict = {}
        self._lock = threading.Lock()

    def __call__(self, event: str, device: str, details: dict, key=None) -> bool:
        key = (event, device) if key is None else key
        now = self.clock()
        with self._lock:
            last, suppressed = self._state.get(key, (None, 0))
            if last is not None and now - last < self.interval:
                self._state[key] = (last, suppressed + 1)
                return False
            self._state[key] = (now, 0)
        extra = {"suppressed_since_last": suppressed} if suppressed else {}
        self.log(event, device, {**details, **extra})
        return True


# ---------------------------------------------------------------------------
# TTS registration: idempotent, rolls back only what this call created
# ---------------------------------------------------------------------------
class TtsRegistrationClientV4(TtsRegistrationClient):
    """identity POST, then JS / NS / AS PUT (idempotent upserts).

    * identity 201: this call created the device, so a later failure deletes it again
      (TTS has no transaction across the four servers; a leftover identity would make every
      retry hit 409 id_taken).
    * identity 409: the device already exists (an earlier success, or a rollback that could
      not finish). The PUTs re-apply the keys and settings; nothing pre-existing is deleted
      if they fail.
    The AppKey is redacted from any error text.
    """

    def __init__(self, config: EnrollmentConfig, api_key: str,
                 transport: Transport = urllib_transport, timeout: float = 20.0) -> None:
        super().__init__(config, api_key, transport, timeout)
        self._redact: tuple = ()

    def _request(self, stage: str, method: str, path: str, body: dict | None = None) -> dict:
        try:
            return super()._request(stage, method, path, body)
        except TtsError as exc:
            text = exc.body
            for secret in self._redact:
                text = text.replace(secret, "<redacted>")
            raise TtsError(stage, exc.status_code, text) from None

    def register_device(self, device_id: str, dev_eui: str, join_eui: str,
                        app_key_hex: str) -> bool:
        """Returns True if the identity was created now, False if it already existed."""
        app = self.config.application_id
        self._redact = (app_key_hex.lower(), app_key_hex.upper())
        created = True
        try:
            self._request("identity", "POST", f"/api/v3/applications/{app}/devices",
                          self.identity_payload(device_id, dev_eui, join_eui))
        except TtsError as exc:
            if exc.status_code == 409 and exc.stage == "identity":
                created = False
            else:
                raise
        try:
            self._request("join_server", "PUT", f"/api/v3/js/applications/{app}/devices/{device_id}",
                          self.join_server_payload(device_id, dev_eui, join_eui, app_key_hex))
            self._request("network_server", "PUT",
                          f"/api/v3/ns/applications/{app}/devices/{device_id}",
                          self.network_server_payload(device_id, dev_eui, join_eui))
            self._request("application_server", "PUT",
                          f"/api/v3/as/applications/{app}/devices/{device_id}",
                          self.application_server_payload(device_id, dev_eui, join_eui))
        except Exception:
            if created:
                self.delete_device(device_id)
            raise
        finally:
            self._redact = ()
        return created


# ---------------------------------------------------------------------------
# Window + decision logic
# ---------------------------------------------------------------------------
@dataclass
class _Window:
    opened_by: str = ""
    opened_at: float = 0.0
    expires_at: float = 0.0
    registered: list = field(default_factory=list)
    rejected: dict = field(default_factory=dict)     # dev_eui -> bad-MIC attempts
    not_rostered: int = 0
    replays: int = 0


class EnrollmentManagerV4:
    def __init__(self, config: EnrollmentConfigV4, master_key: bytes, client,
                 registry: EnrollmentRegistry, roster: Roster, nonces: JoinNonceStore,
                 log: EventLogger, clock: Callable[[], float] = time.time) -> None:
        if len(master_key) != 16:
            raise KeyError_("the master key must be 16 bytes")
        self.config = config
        self._master_key = master_key
        self.client = client
        self.registry = registry
        self.roster = roster
        self.nonces = nonces
        self.log = log
        self.limited = RateLimitedLog(log, config.bad_mic_log_interval_seconds)
        self.clock = clock
        self._lock = threading.Lock()
        self._window = _Window()

    # -- window control --------------------------------------------------------
    def open_window(self, actor: str, seconds: int | None = None, quota: int | None = None) -> dict:
        """``quota`` is accepted for console compatibility and ignored: the roster is the authority."""
        del quota
        duration = max(1, min(int(seconds or self.config.window_seconds), MAX_WINDOW_SECONDS))
        now = self.clock()
        with self._lock:
            self._window = _Window(opened_by=str(actor or "unknown"), opened_at=now,
                                   expires_at=now + duration)
            status = self._status_locked(now)
        self.log("enrollment_window_opened", "server", {
            "opened_by": actor, "seconds": duration, "join_eui": self.config.join_eui,
            "rostered_devices": len(self.roster)})
        return status

    def close_window(self, actor: str = "") -> dict:
        with self._lock:
            was_open = self._window.expires_at > self.clock()
            self._window = _Window()
            status = self._status_locked(self.clock())
        if was_open:
            self.log("enrollment_window_closed", "server", {"closed_by": actor or "unknown"})
        return status

    def status(self) -> dict:
        with self._lock:
            return self._status_locked(self.clock())

    def _status_locked(self, now: float) -> dict:
        remaining = max(0, int(self._window.expires_at - now))
        w = self._window
        return {
            "open": remaining > 0, "opened_by": w.opened_by if remaining > 0 else "",
            "seconds_remaining": remaining, "registered": list(w.registered),
            "rejected_bad_mic": dict(w.rejected), "not_rostered": w.not_rostered,
            "replays": w.replays, "rostered_devices": len(self.roster),
        }

    # -- decision -------------------------------------------------------------------
    def evaluate(self, request: JoinRequest) -> Decision:
        """Decide on one join request, registering the device if (and only if) every check passes."""
        decision = self._decide(request)
        details = {"outcome": decision.outcome.value, "join_eui": request.join_eui,
                   "dev_nonce": request.dev_nonce, "detail": decision.detail}
        out = decision.outcome
        if out is Outcome.BAD_MIC:
            self.limited("enrollment_bad_mic", decision.dev_eui, details)
        elif out is Outcome.NOT_ROSTERED:
            self.limited("enrollment_not_rostered", decision.dev_eui, details)
        elif out is Outcome.REPLAYED_DEV_NONCE:
            self.limited("enrollment_replayed_dev_nonce", decision.dev_eui, details)
        elif out not in (Outcome.WINDOW_CLOSED, Outcome.JOIN_EUI_MISMATCH):
            # Window-closed and foreign JoinEUIs are the normal state of a gateway full of
            # other people's joins; logging every one would bury everything else.
            self.log(f"enrollment_{out.value}", decision.dev_eui, details)
        return decision

    def _decide(self, request: JoinRequest) -> Decision:
        dev_eui = request.dev_eui
        now = self.clock()
        with self._lock:
            if self._window.expires_at <= now:
                return Decision(Outcome.WINDOW_CLOSED, dev_eui, "no window is open")
        try:
            expected_join_eui = normalize_hex(self.config.join_eui, 16, "join_eui")
        except ValueError as exc:
            return Decision(Outcome.MISCONFIGURED, dev_eui, str(exc))
        if request.join_eui != expected_join_eui:
            return Decision(Outcome.JOIN_EUI_MISMATCH, dev_eui,
                            f"JoinEUI {request.join_eui} is not {expected_join_eui}")
        self.roster.reload()
        if not self.roster.contains(dev_eui):
            with self._lock:
                self._window.not_rostered += 1
            return Decision(Outcome.NOT_ROSTERED, dev_eui, "DevEUI is not in the roster")

        # --- the security boundary: MIC under THIS device's derived key ------------------
        try:
            key = derive_app_key(self._master_key, dev_eui)
        except KeyError_ as exc:
            return Decision(Outcome.MISCONFIGURED, dev_eui, str(exc))
        if not verify_join_request_mic(request, key):
            with self._lock:
                self._window.rejected[dev_eui] = self._window.rejected.get(dev_eui, 0) + 1
            return Decision(Outcome.BAD_MIC, dev_eui,
                            "MIC did not verify under this device's derived key")
        # --- replay protection, only for requests that passed the MIC ----------------------
        if not self.nonces.add(dev_eui, request.dev_nonce, now):
            with self._lock:
                self._window.replays += 1
            return Decision(Outcome.REPLAYED_DEV_NONCE, dev_eui,
                            f"DevNonce {request.dev_nonce} was already used by this DevEUI")
        if self.registry.contains(dev_eui):
            return Decision(Outcome.ALREADY_REGISTERED, dev_eui)
        return self._register(request, key)

    def _register(self, request: JoinRequest, key: bytes) -> Decision:
        dev_eui = request.dev_eui
        device_id = device_id_from_dev_eui(dev_eui)
        try:
            created = self.client.register_device(device_id, dev_eui, request.join_eui, key.hex())
        except TtsError as exc:
            return Decision(Outcome.REGISTRATION_FAILED, dev_eui, str(exc))
        except Exception as exc:  # noqa: BLE001 - TTS unreachable must not kill the listener
            return Decision(Outcome.REGISTRATION_FAILED, dev_eui, f"{type(exc).__name__}: {exc}")
        with self._lock:
            if dev_eui not in self._window.registered:
                self._window.registered.append(dev_eui)
            opened_by = self._window.opened_by
        self.registry.add(dev_eui, {
            "device_id": device_id, "join_eui": request.join_eui,
            "dev_nonce": request.dev_nonce, "mic_verified": True, "key": "derived",
            "enrolled_at_unix": int(self.clock()), "opened_by": opened_by,
            "source": "enrolled" if created is not False else "already_in_tts"})
        return Decision(Outcome.REGISTERED, dev_eui, device_id)


# ---------------------------------------------------------------------------
# Gateway event stream
# ---------------------------------------------------------------------------
class JoinListenerV4(JoinListener):
    """JoinListener with the strict 23-byte / MHDR 0x00 JoinRequest classification."""

    def handle_line(self, line: str):
        frame = extract_raw_payload(line)
        if frame is None:
            return None
        kind, request = classify_join_frame(frame)
        if kind == "malformed":
            self.manager.limited("enrollment_malformed_join", "gateway", {
                "length": len(frame), "mhdr": f"0x{frame[0]:02X}",
                "detail": "MType JoinRequest but not exactly 23 bytes with MHDR 0x00; ignored"})
            return None
        if request is None:
            return None
        return self.manager.evaluate(request)


def build_enrollment(config: dict, log: EventLogger, base_dir: Path, roster: Roster | None = None,
                     env: dict | None = None, transport: Transport = urllib_transport):
    """Create the manager and listener from config.toml and the environment (never from files)."""
    env = os.environ if env is None else env
    ecfg = EnrollmentConfigV4.from_config(config)
    missing = [name for name in (ecfg.gateway_api_key_env, ecfg.device_api_key_env)
               if not env.get(name, "").strip()]
    if missing:
        raise RuntimeError("enrollment needs these environment variables set: " + ", ".join(missing))
    master = master_key_from_env(env, ecfg.master_key_env)
    if roster is None:
        v4 = config.get("v4", {})
        roster_path = Path(str(v4.get("roster_path", "roster.json")))
        roster = Roster(roster_path if roster_path.is_absolute() else base_dir / roster_path)

    def resolve(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else base_dir / p

    manager = EnrollmentManagerV4(
        ecfg, master,
        TtsRegistrationClientV4(ecfg, env[ecfg.device_api_key_env].strip(), transport),
        EnrollmentRegistry(resolve(ecfg.registry_path)), roster,
        JoinNonceStore(resolve(ecfg.nonce_store_path)), log)
    listener = JoinListenerV4(manager, env[ecfg.gateway_api_key_env].strip(), log)
    return manager, listener


def console_event_log(event: str, device_id: str, details: dict) -> None:
    print(f"{time.strftime('%H:%M:%S')} {event:32s} {device_id:20s} {details}", flush=True)


def main() -> int:
    """Stand-alone enrollment: open a window and enrol rostered boards until it closes."""
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--window-minutes", type=float, default=None)
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    manager, listener = build_enrollment(config, console_event_log, args.config.resolve().parent)
    seconds = None if args.window_minutes is None else int(args.window_minutes * 60)
    manager.open_window("cli", seconds=seconds)
    listener.start()
    try:
        while manager.status()["open"] and listener.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        manager.close_window("operator")
        listener.stop()
    status = manager.status()
    print(json.dumps({"registered": status["registered"],
                      "rejected_bad_mic": status["rejected_bad_mic"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
