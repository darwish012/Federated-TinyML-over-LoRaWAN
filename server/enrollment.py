"""Automatic device enrollment: no board is registered by hand in The Things Stack.

Unknown devices never reach application scope, so a webhook or the application
MQTT topic cannot see them. Enrollment therefore reads GATEWAY traffic, the same
pattern as the Smart Building Management System (join_listener.py /
join_enrollment.py, read only), adapted to this server: no SQLite, a small JSON
registry, a window opened from the command line, and a quota.

A DevEUI is registered in TTS only when ALL of these hold:

  1. an operator opened an enrollment window (--enroll-minutes or the
     ``enroll`` command) and it has not expired,
  2. the window's quota is not exhausted,
  3. the join request's JoinEUI is this deployment's,
  4. the DevEUI is not already registered,
  5. the join request's MIC verifies under the shared AppKey (AES-128-CMAC).

Check 5 is the security boundary. Bad-MIC attempts are logged, listed in the
window status, and never registered. Nothing about the AppKey is ever logged.

Rights the two API keys need (see README):

    gateway key   RIGHT_GATEWAY_TRAFFIC_READ on the gateway(s)  (event stream)
    device key    view + edit end devices, write end-device keys (registration)
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable
import urllib.error
import urllib.request

from lorawan_join import (
    JoinRequest,
    app_key_bytes,
    device_id_from_dev_eui,
    normalize_hex,
    parse_join_request,
    verify_join_request_mic,
)

EventLogger = Callable[[str, str, dict], None]
Transport = Callable[[str, str, dict, "bytes | None", float], "tuple[int, bytes]"]

MAX_WINDOW_SECONDS = 6 * 3600
# Gateway events that carry no radio payload; skipped cheaply.
IGNORED_EVENT_PREFIXES = ("gs.status.", "gs.gateway.connection.", "events.stream.")


class Outcome(str, Enum):
    REGISTERED = "registered"
    ALREADY_REGISTERED = "already_registered"
    WINDOW_CLOSED = "window_closed"
    QUOTA_EXCEEDED = "quota_exceeded"
    JOIN_EUI_MISMATCH = "join_eui_mismatch"
    BAD_MIC = "bad_mic"
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
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class EnrollmentConfig:
    api_base_url: str = "http://localhost:1885"
    application_id: str = "tinyfl-v91-hw"
    join_eui: str = "0000000000000000"
    gateway_ids: tuple = ()
    cluster_address: str = ""
    window_seconds: int = 600
    quota: int = 10
    registry_path: str = "enrolled_devices.json"
    gateway_api_key_env: str = "TTS_GATEWAY_API_KEY"
    device_api_key_env: str = "TTS_DEVICE_API_KEY"
    app_key_env: str = "TINYFL_APP_KEY"
    # This experiment's MAC settings (README section 1, fix_multicast_rx2.py):
    lorawan_version: str = "MAC_V1_0_3"
    lorawan_phy_version: str = "PHY_V1_0_3_REV_A"
    frequency_plan_id: str = "EU_863_870_TTN"
    rx2_data_rate_index: str = "DATA_RATE_5"
    rx2_frequency: str = "869525000"

    @classmethod
    def from_config(cls, config: dict) -> "EnrollmentConfig":
        section = dict(config.get("enrollment", {}))
        tts = config.get("tts", {})
        experiment = config.get("experiment", {})
        defaults = cls()
        gateways = section.get("gateway_ids") or experiment.get("multicast_gateway_ids") or []
        cfg = cls(
            api_base_url=str(section.get("api_base_url")
                             or tts.get("console_url") or defaults.api_base_url).rstrip("/"),
            application_id=str(tts.get("application_id", defaults.application_id)),
            join_eui=str(section.get("join_eui", defaults.join_eui)),
            gateway_ids=tuple(str(g).strip() for g in gateways if str(g).strip()),
            cluster_address=str(section.get("cluster_address", "")).strip(),
            window_seconds=int(section.get("window_seconds", defaults.window_seconds)),
            quota=int(section.get("quota", defaults.quota)),
            registry_path=str(section.get("registry_path", defaults.registry_path)),
            gateway_api_key_env=str(section.get("gateway_api_key_env", defaults.gateway_api_key_env)),
            device_api_key_env=str(section.get("device_api_key_env", defaults.device_api_key_env)),
            app_key_env=str(section.get("app_key_env", defaults.app_key_env)),
            lorawan_version=str(section.get("lorawan_version", defaults.lorawan_version)),
            lorawan_phy_version=str(section.get("lorawan_phy_version", defaults.lorawan_phy_version)),
            frequency_plan_id=str(section.get("frequency_plan_id", defaults.frequency_plan_id)),
            rx2_data_rate_index=str(section.get("rx2_data_rate_index", defaults.rx2_data_rate_index)),
            rx2_frequency=str(section.get("rx2_frequency", defaults.rx2_frequency)),
        )
        normalize_hex(cfg.join_eui, 16, "enrollment.join_eui")
        if cfg.quota < 1 or cfg.window_seconds < 1:
            raise ValueError("enrollment.quota and enrollment.window_seconds must be positive")
        return cfg


# ---------------------------------------------------------------------------
# Registry of enrolled devices (small JSON file, no keys in it)
# ---------------------------------------------------------------------------
class EnrollmentRegistry:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._devices: dict[str, dict] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._devices = {k.upper(): v for k, v in data.get("devices", {}).items()}

    def contains(self, dev_eui: str) -> bool:
        with self._lock:
            return dev_eui.upper() in self._devices

    def add(self, dev_eui: str, record: dict) -> None:
        with self._lock:
            self._devices[dev_eui.upper()] = dict(record)
            self._save()

    def devices(self) -> dict[str, dict]:
        with self._lock:
            return {k: dict(v) for k, v in self._devices.items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps({"devices": self._devices}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.path)


# ---------------------------------------------------------------------------
# TTS registration client
# ---------------------------------------------------------------------------
class TtsError(Exception):
    def __init__(self, stage: str, status_code: int, body: str) -> None:
        super().__init__(f"TTS {stage} failed with HTTP {status_code}: {body[:300]}")
        self.stage = stage
        self.status_code = status_code
        self.body = body


def urllib_transport(method: str, url: str, headers: dict, body: bytes | None,
                     timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class TtsRegistrationClient:
    """Creates an end device in the Identity, Join, Network and Application Servers.

    Exact API calls (``{app}`` = application id, ``{id}`` = eui-<deveui>):

        POST /api/v3/applications/{app}/devices        Identity Server: ids, name, server addresses
        PUT  /api/v3/js/applications/{app}/devices/{id}  Join Server: AppKey
        PUT  /api/v3/ns/applications/{app}/devices/{id}  Network Server: MAC 1.0.3, RP 1.0.3 rev A,
                                                        EU868 plan, OTAA, Class C, desired RX2 DR5
        PUT  /api/v3/as/applications/{app}/devices/{id}  Application Server: link
        DELETE ... (all four)                          rollback if a later stage fails

    TTS has no transaction across the four, so a failure after the identity was
    created removes it again, otherwise the retry would hit 409 id_taken.
    """

    def __init__(self, config: EnrollmentConfig, api_key: str,
                 transport: Transport = urllib_transport, timeout: float = 20.0) -> None:
        self.config = config
        self.api_key = api_key
        self.transport = transport
        self.timeout = timeout
        # Where TTS thinks its own servers live. On a stack that calls itself
        # something other than the URL we connect to, registration fails with
        # network_server_address_mismatch unless cluster_address names it.
        self.host = (config.cluster_address
                     or config.api_base_url.replace("https://", "").replace("http://", "")
                     ).split("/")[0]

    def _request(self, stage: str, method: str, path: str, body: dict | None = None) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        data = None if body is None else json.dumps(body).encode("utf-8")
        status, raw = self.transport(method, self.config.api_base_url + path, headers, data,
                                     self.timeout)
        text = raw.decode("utf-8", errors="replace") if raw else ""
        if status >= 300:
            raise TtsError(stage, status, text)
        try:
            return json.loads(text) if text else {}
        except ValueError:
            return {}

    def _ids(self, device_id: str, dev_eui: str, join_eui: str) -> dict:
        return {"device_id": device_id, "dev_eui": dev_eui, "join_eui": join_eui}

    def identity_payload(self, device_id: str, dev_eui: str, join_eui: str) -> dict:
        c = self.config
        return {
            "end_device": {
                "ids": self._ids(device_id, dev_eui, join_eui),
                "name": device_id,
                "description": "TinyFL hardware experiment board (auto-enrolled)",
                "join_server_address": self.host,
                "network_server_address": self.host,
                "application_server_address": self.host,
                "lorawan_version": c.lorawan_version,
                "lorawan_phy_version": c.lorawan_phy_version,
                "frequency_plan_id": c.frequency_plan_id,
                "supports_join": True,
            },
            "field_mask": {"paths": [
                "name", "description", "join_server_address", "network_server_address",
                "application_server_address", "lorawan_version", "lorawan_phy_version",
                "frequency_plan_id", "supports_join",
            ]},
        }

    def join_server_payload(self, device_id: str, dev_eui: str, join_eui: str,
                            app_key_hex: str) -> dict:
        return {
            "end_device": {
                "ids": self._ids(device_id, dev_eui, join_eui),
                "network_server_address": self.host,
                "application_server_address": self.host,
                "root_keys": {"app_key": {"key": app_key_hex}},
            },
            "field_mask": {"paths": [
                "network_server_address", "application_server_address", "root_keys.app_key.key",
            ]},
        }

    def network_server_payload(self, device_id: str, dev_eui: str, join_eui: str) -> dict:
        c = self.config
        return {
            "end_device": {
                "ids": self._ids(device_id, dev_eui, join_eui),
                "lorawan_version": c.lorawan_version,
                "lorawan_phy_version": c.lorawan_phy_version,
                "frequency_plan_id": c.frequency_plan_id,
                "supports_join": True,
                "supports_class_c": True,
                "mac_settings": {
                    "desired_rx2_data_rate_index": c.rx2_data_rate_index,
                    "desired_rx2_frequency": c.rx2_frequency,
                },
            },
            "field_mask": {"paths": [
                "lorawan_version", "lorawan_phy_version", "frequency_plan_id",
                "supports_join", "supports_class_c",
                "mac_settings.desired_rx2_data_rate_index",
                "mac_settings.desired_rx2_frequency",
            ]},
        }

    def application_server_payload(self, device_id: str, dev_eui: str, join_eui: str) -> dict:
        return {
            "end_device": {"ids": self._ids(device_id, dev_eui, join_eui)},
            "field_mask": {"paths": []},
        }

    def register_device(self, device_id: str, dev_eui: str, join_eui: str,
                        app_key_hex: str) -> None:
        app = self.config.application_id
        self._request("identity", "POST", f"/api/v3/applications/{app}/devices",
                      self.identity_payload(device_id, dev_eui, join_eui))
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
            self.delete_device(device_id)
            raise

    def delete_device(self, device_id: str) -> None:
        app = self.config.application_id
        for path in (
            f"/api/v3/as/applications/{app}/devices/{device_id}",
            f"/api/v3/ns/applications/{app}/devices/{device_id}",
            f"/api/v3/js/applications/{app}/devices/{device_id}",
            f"/api/v3/applications/{app}/devices/{device_id}",
        ):
            try:
                self._request("rollback", "DELETE", path)
            except TtsError as exc:
                if exc.status_code != 404:
                    raise


# ---------------------------------------------------------------------------
# Window + decision logic
# ---------------------------------------------------------------------------
@dataclass
class _Window:
    opened_by: str = ""
    opened_at: float = 0.0
    expires_at: float = 0.0
    quota: int = 0
    registered: list = field(default_factory=list)
    rejected: dict = field(default_factory=dict)     # dev_eui -> bad-MIC attempt count


class EnrollmentManager:
    def __init__(
        self,
        config: EnrollmentConfig,
        app_key_hex: str,
        client,                       # anything with register_device(...)
        registry: EnrollmentRegistry,
        log: EventLogger,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self._app_key_hex = app_key_hex
        self.client = client
        self.registry = registry
        self.log = log
        self.clock = clock
        self._lock = threading.Lock()
        self._window = _Window()

    # -- window control ----------------------------------------------------
    def open_window(self, actor: str, seconds: int | None = None,
                    quota: int | None = None) -> dict:
        duration = max(1, min(int(seconds or self.config.window_seconds), MAX_WINDOW_SECONDS))
        limit = int(quota or self.config.quota)
        now = self.clock()
        with self._lock:
            self._window = _Window(opened_by=str(actor or "unknown"), opened_at=now,
                                   expires_at=now + duration, quota=limit)
            status = self._status_locked(now)
        self.log("enrollment_window_opened", "server",
                 {"opened_by": actor, "seconds": duration, "quota": limit,
                  "join_eui": self.config.join_eui})
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
        is_open = remaining > 0
        return {
            "open": is_open,
            "opened_by": self._window.opened_by if is_open else "",
            "seconds_remaining": remaining,
            "quota": self._window.quota,
            "quota_used": len(self._window.registered),
            "registered": list(self._window.registered),
            "rejected": dict(self._window.rejected),
        }

    # -- decision ------------------------------------------------------------
    def evaluate(self, request: JoinRequest) -> Decision:
        """Decide on one join request, registering the device if (and only if) all checks pass."""
        decision = self._decide(request)
        details = {"outcome": decision.outcome.value, "join_eui": request.join_eui,
                   "dev_nonce": request.dev_nonce, "detail": decision.detail}
        if decision.outcome is Outcome.BAD_MIC:
            self.log("enrollment_bad_mic", decision.dev_eui, details)
        elif decision.outcome is not Outcome.WINDOW_CLOSED:
            # Window-closed is the normal state of a gateway full of other people's
            # joins; logging every one would bury everything else.
            self.log(f"enrollment_{decision.outcome.value}", decision.dev_eui, details)
        return decision

    def _decide(self, request: JoinRequest) -> Decision:
        dev_eui = request.dev_eui
        now = self.clock()
        with self._lock:
            window = self._window
            if window.expires_at <= now:
                return Decision(Outcome.WINDOW_CLOSED, dev_eui, "no window is open")
            if len(window.registered) >= window.quota:
                return Decision(Outcome.QUOTA_EXCEEDED, dev_eui,
                                f"window quota of {window.quota} reached")

        try:
            expected_join_eui = normalize_hex(self.config.join_eui, 16, "join_eui")
            key = app_key_bytes(self._app_key_hex)
        except ValueError as exc:
            return Decision(Outcome.MISCONFIGURED, dev_eui, str(exc))
        if request.join_eui != expected_join_eui:
            return Decision(Outcome.JOIN_EUI_MISMATCH, dev_eui,
                            f"JoinEUI {request.join_eui} is not {expected_join_eui}")
        if self.registry.contains(dev_eui):
            return Decision(Outcome.ALREADY_REGISTERED, dev_eui)

        # --- the security boundary -------------------------------------------
        if not verify_join_request_mic(request, key):
            with self._lock:
                self._window.rejected[dev_eui] = self._window.rejected.get(dev_eui, 0) + 1
            return Decision(Outcome.BAD_MIC, dev_eui, "MIC did not verify against the shared AppKey")

        return self._register(request)

    def _register(self, request: JoinRequest) -> Decision:
        dev_eui = request.dev_eui
        device_id = device_id_from_dev_eui(dev_eui)
        try:
            self.client.register_device(device_id, dev_eui, request.join_eui, self._app_key_hex)
        except TtsError as exc:
            if exc.status_code == 409 and exc.stage == "identity":
                # Already in TTS but not in our registry (registered by hand, or a
                # previous run's registry was lost). It can join; leave it alone.
                self.registry.add(dev_eui, {"device_id": device_id, "join_eui": request.join_eui,
                                            "source": "already_in_tts"})
                return Decision(Outcome.ALREADY_REGISTERED, dev_eui, device_id)
            return Decision(Outcome.REGISTRATION_FAILED, dev_eui, str(exc))
        except Exception as exc:  # noqa: BLE001 - TTS unreachable must not kill the listener
            return Decision(Outcome.REGISTRATION_FAILED, dev_eui, f"{type(exc).__name__}: {exc}")

        with self._lock:
            if dev_eui not in self._window.registered:
                self._window.registered.append(dev_eui)
            opened_by = self._window.opened_by
        self.registry.add(dev_eui, {
            "device_id": device_id, "join_eui": request.join_eui,
            "dev_nonce": request.dev_nonce, "mic_verified": True,
            "enrolled_at_unix": int(self.clock()), "opened_by": opened_by,
            "source": "enrolled",
        })
        return Decision(Outcome.REGISTERED, dev_eui, device_id)


# ---------------------------------------------------------------------------
# Gateway event stream
# ---------------------------------------------------------------------------
def extract_raw_payload(line: str) -> bytes | None:
    """The radio frame inside one event-stream line, or None if it carries none."""
    text = line[5:].strip() if line.startswith("data:") else line.strip()
    try:
        event = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(event, dict):
        return None
    result = event.get("result", event)
    if not isinstance(result, dict):
        return None
    if str(result.get("name") or "").startswith(IGNORED_EVENT_PREFIXES):
        return None
    data = result.get("data") or {}
    if not isinstance(data, dict):
        return None
    # gs.up.receive wraps a GatewayUplinkMessage (data.message.raw_payload); other
    # event types put the frame at the top level. Both are accepted.
    message = data.get("message")
    raw = message.get("raw_payload") if isinstance(message, dict) else None
    raw = raw or data.get("raw_payload")
    if not raw:
        return None
    try:
        return base64.b64decode(raw)
    except Exception:  # noqa: BLE001
        return None


def open_event_stream(config: EnrollmentConfig, gateway_key: str, timeout: float = 30.0):
    """POST /api/v3/events for the configured gateways; returns the response object."""
    body = json.dumps({
        "identifiers": [{"gateway_ids": {"gateway_id": g}} for g in config.gateway_ids],
        "tail": 0,
    }).encode("utf-8")
    request = urllib.request.Request(
        config.api_base_url + "/api/v3/events", data=body, method="POST",
        headers={"Authorization": f"Bearer {gateway_key}", "Content-Type": "application/json",
                 "Accept": "text/event-stream"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


class JoinListener:
    """Streams gateway events while a window is open and feeds join requests to the manager.

    Its lifetime is the window's: the thread starts when a window opens and ends
    when it lapses, so between windows the server holds no gateway subscription.
    """

    def __init__(self, manager: EnrollmentManager, gateway_key: str, log: EventLogger,
                 opener=None) -> None:
        self.manager = manager
        self.gateway_key = gateway_key
        self.log = log
        self._opener = opener or (lambda: open_event_stream(manager.config, gateway_key))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if not self.manager.config.gateway_ids:
            self.log("enrollment_listener_not_started", "server",
                     {"reason": "no gateway ids: set enrollment.gateway_ids or "
                                "experiment.multicast_gateway_ids"})
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="join-enrollment-listener",
                                            daemon=True)
            self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def handle_line(self, line: str) -> Decision | None:
        """Process one event-stream line. Returns the decision if it was a join request."""
        frame = extract_raw_payload(line)
        if frame is None:
            return None
        request = parse_join_request(frame)
        if request is None:
            return None
        return self.manager.evaluate(request)

    def _run(self) -> None:
        self.log("enrollment_listener_started", "server",
                 {"gateways": list(self.manager.config.gateway_ids)})
        try:
            while not self._stop.is_set() and self.manager.status()["open"]:
                try:
                    self._consume()
                except urllib.error.HTTPError as exc:
                    hint = ""
                    if exc.code in (401, 403):
                        hint = ("the gateway API key lacks RIGHT_GATEWAY_TRAFFIC_READ "
                                "(console: 'Read gateway traffic')")
                    self.log("enrollment_stream_refused", "server",
                             {"http_status": exc.code, "hint": hint})
                    self._stop.wait(10)
                except Exception as exc:  # noqa: BLE001 - reconnect on any drop
                    self.log("enrollment_stream_dropped", "server",
                             {"error": type(exc).__name__})
                    self._stop.wait(3)
        finally:
            self.log("enrollment_listener_stopped", "server", {})

    def _consume(self) -> None:
        with self._opener() as response:
            for raw in response:
                if self._stop.is_set() or not self.manager.status()["open"]:
                    return
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                line = line.strip()
                if line:
                    self.handle_line(line)


# ---------------------------------------------------------------------------
# Wiring from config + standalone command
# ---------------------------------------------------------------------------
def build_enrollment(config: dict, log: EventLogger, base_dir: Path,
                     env: dict | None = None) -> tuple[EnrollmentManager, JoinListener]:
    """Create the manager and listener from config.toml and the environment.

    The three secrets come from environment variables only (never from files):
    the AppKey, a gateway key (traffic read) and a device key (registration).
    """
    env = os.environ if env is None else env
    ecfg = EnrollmentConfig.from_config(config)
    missing = [name for name in (ecfg.app_key_env, ecfg.gateway_api_key_env,
                                 ecfg.device_api_key_env) if not env.get(name, "").strip()]
    if missing:
        raise RuntimeError(
            "enrollment needs these environment variables set: " + ", ".join(missing)
        )
    app_key_bytes(env[ecfg.app_key_env])            # fail early on a malformed key
    registry_path = Path(ecfg.registry_path)
    if not registry_path.is_absolute():
        registry_path = base_dir / registry_path
    manager = EnrollmentManager(
        ecfg,
        env[ecfg.app_key_env].strip(),
        TtsRegistrationClient(ecfg, env[ecfg.device_api_key_env].strip()),
        EnrollmentRegistry(registry_path),
        log,
    )
    listener = JoinListener(manager, env[ecfg.gateway_api_key_env].strip(), log)
    return manager, listener


def console_event_log(event: str, device_id: str, details: dict) -> None:
    print(f"{time.strftime('%H:%M:%S')} {event:32s} {device_id:20s} {details}", flush=True)


def main() -> int:
    """Stand-alone enrollment: open a window and enrol boards until it closes."""
    import tomllib

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--window-minutes", type=float, default=None)
    parser.add_argument("--quota", type=int, default=None)
    args = parser.parse_args()
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    log = console_event_log
    manager, listener = build_enrollment(config, log, args.config.resolve().parent)
    seconds = None if args.window_minutes is None else int(args.window_minutes * 60)
    manager.open_window("cli", seconds=seconds, quota=args.quota)
    listener.start()
    try:
        while manager.status()["open"] and listener.is_running():
            time.sleep(1)
    except KeyboardInterrupt:
        manager.close_window("operator")
        listener.stop()
    status = manager.status()
    print(json.dumps({"registered": status["registered"], "rejected": status["rejected"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
