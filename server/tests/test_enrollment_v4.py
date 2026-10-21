"""Hardened enrollment: roster, per-device AppKey, strict JoinRequest, MIC, DevNonce replay,
idempotent registration with rollback, rate-limited logging and no key in any log or file."""
from __future__ import annotations

import base64
import json
import re

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import pytest

import enrollment_v4 as en
import lorawan_join
from enrollment import EnrollmentRegistry
from keys_v4 import KeyError_, derive_app_key, hex_bytes, master_key_from_env
from lorawan_join import build_join_request, device_id_from_dev_eui
from roster_v4 import Roster, RosterError, normalise_dev_eui

# Test-only values; no real key appears in this repository's tests.
MASTER = bytes(range(16))
OTHER_MASTER = bytes(range(16, 32))
JOIN_EUI = "70B3D57ED0010203"
DEV_A = "70B3D57ED0000001"
DEV_B = "70B3D57ED0000002"
DEV_C = "70B3D57ED0000003"          # never provisioned
APP = "tinyfl-v91-hw"


class FakeTts:
    """Stands in for the TTS HTTP API: remembers what exists and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.identities: set = set()
        self.js: dict = {}
        self.ns: set = set()
        self.as_: set = set()
        self.fail: dict = {}                # (METHOD, path fragment) -> (status, body)
        self.bodies: list[str] = []

    def __call__(self, method, url, headers, body, timeout):
        path = url.split("1885", 1)[-1]
        self.calls.append((method, path))
        payload = json.loads(body) if body else None
        if body:
            self.bodies.append(body.decode())
        for (m, fragment), (status, text) in list(self.fail.items()):
            if m == method and fragment in path:
                if self.fail_once:
                    del self.fail[(m, fragment)]
                return status, text
        device = path.rsplit("/", 1)[-1]
        if method == "POST" and path.endswith("/devices"):
            device = payload["end_device"]["ids"]["device_id"]
            if device in self.identities:
                return 409, b'{"code":6,"message":"error:pkg/identityserver:id_taken"}'
            self.identities.add(device)
            return 200, b"{}"
        if method == "PUT" and "/js/" in path:
            self.js[device] = payload["end_device"]["root_keys"]["app_key"]["key"]
        elif method == "PUT" and "/ns/" in path:
            self.ns.add(device)
        elif method == "PUT" and "/as/" in path:
            self.as_.add(device)
        elif method == "DELETE":
            if "/as/" in path:
                self.as_.discard(device)
            elif "/ns/" in path:
                self.ns.discard(device)
            elif "/js/" in path:
                self.js.pop(device, None)
            else:
                self.identities.discard(device)
        return 200, b"{}"

    fail_once = True

    def residue(self, device_id) -> bool:
        return device_id in self.identities or device_id in self.js or device_id in self.ns or device_id in self.as_


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path):
    roster = Roster(tmp_path / "roster.json")
    roster.add(DEV_A)
    roster.add(DEV_B)
    tts = FakeTts()
    events = []
    clock = Clock()
    cfg = en.EnrollmentConfigV4(application_id=APP, join_eui=JOIN_EUI, api_base_url="http://localhost:1885",
                                bad_mic_log_interval_seconds=60.0)
    nonce_path = tmp_path / "join_nonces.jsonl"
    registry_path = tmp_path / "enrolled_devices.json"
    manager = en.EnrollmentManagerV4(
        cfg, MASTER, en.TtsRegistrationClientV4(cfg, "device-api-key", tts), EnrollmentRegistry(registry_path),
        roster, en.JoinNonceStore(nonce_path), lambda e, d, x: events.append((e, d, x)), clock=clock)
    manager.limited.clock = clock
    manager.open_window("test", seconds=600)

    class Env:
        pass

    e = Env()
    e.manager, e.tts, e.events, e.clock, e.roster, e.cfg = manager, tts, events, clock, roster, cfg
    e.nonce_path, e.registry_path, e.tmp = nonce_path, registry_path, tmp_path
    return e


def _parse(frame: bytes):
    """A well-formed JoinRequest frame -> en.JoinRequest through the strict classifier."""
    kind, request = en.classify_join_frame(frame)
    assert kind == "join", kind
    return request


def join(dev_eui, nonce=1, key=None, join_eui=JOIN_EUI, master=MASTER):
    key = derive_app_key(master, dev_eui) if key is None else key
    return _parse(build_join_request(key, join_eui, dev_eui, nonce))





def names(env):
    return [e for e, _, _ in env.events]


# ---------------------------------------------------------------------------
# Per-device AppKey
# ---------------------------------------------------------------------------
def test_app_key_is_aes_ecb_of_deveui_and_eight_zero_bytes():
    block = bytes.fromhex(DEV_A) + bytes(8)
    enc = Cipher(algorithms.AES(MASTER), modes.ECB()).encryptor()
    assert derive_app_key(MASTER, DEV_A) == enc.update(block) + enc.finalize()
    assert len(derive_app_key(MASTER, DEV_A)) == 16
    assert derive_app_key(MASTER, DEV_A) != derive_app_key(MASTER, DEV_B)
    assert derive_app_key(MASTER, DEV_A) != derive_app_key(OTHER_MASTER, DEV_A)
    assert derive_app_key(MASTER, DEV_A.lower()) == derive_app_key(MASTER, DEV_A)
    # FIPS-197 appendix C.1 known answer: key 000102..0f, plaintext 00112233445566778899aabbccddeeff
    enc = Cipher(algorithms.AES(MASTER), modes.ECB()).encryptor()
    assert (enc.update(bytes.fromhex("00112233445566778899aabbccddeeff")) + enc.finalize()).hex() == \
        "69c4e0d86a7b0430d8cdb78070b4c55a"


def test_key_helpers_never_echo_a_value():
    secret = "ab" * 15 + "zz"
    for call in (lambda: hex_bytes(secret, 16, "the key"), lambda: master_key_from_env({"K": secret}, "K"),
                 lambda: master_key_from_env({}, "K"), lambda: derive_app_key(b"short", DEV_A),
                 lambda: derive_app_key(MASTER, "12345")):
        with pytest.raises(KeyError_) as err:
            call()
        assert secret not in str(err.value)
    assert master_key_from_env({"K": MASTER.hex()}, "K") == MASTER


# ---------------------------------------------------------------------------
# Strict JoinRequest classification
# ---------------------------------------------------------------------------
def test_classify_join_frame_is_exactly_23_bytes_with_mhdr_zero():
    good = build_join_request(derive_app_key(MASTER, DEV_A), JOIN_EUI, DEV_A, 9)
    assert len(good) == 23 and good[0] == 0
    kind, req = en.classify_join_frame(good)
    assert kind == "join" and (req.join_eui, req.dev_eui, req.dev_nonce) == (JOIN_EUI, DEV_A, 9)
    assert req.signed_body == good[:19] and req.mic == good[19:]
    for bad in (good[:-1], good + b"\x00", good[:22] + b"", b"\x00" * 24):
        assert en.classify_join_frame(bad)[0] == "malformed"
    assert en.classify_join_frame(bytes([0x01]) + good[1:])[0] == "malformed"       # RFU/major bits set
    assert en.classify_join_frame(bytes([0x04]) + good[1:])[0] == "malformed"
    for other in (bytes([0x40]) + good[1:], bytes([0x80]) + good[1:], b"", b"\xe0" + good[1:]):
        assert en.classify_join_frame(other)[0] == "other"                         # data frames, rejoin ...
    assert en.classify_join_frame(None)[0] == "other"


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------
def test_a_valid_join_under_the_derived_key_registers_the_device(env):
    decision = env.manager.evaluate(join(DEV_A))
    assert decision.outcome is en.Outcome.REGISTERED and decision.enrolled
    device_id = device_id_from_dev_eui(DEV_A)
    assert env.tts.calls == [
        ("POST", f"/api/v3/applications/{APP}/devices"),
        ("PUT", f"/api/v3/js/applications/{APP}/devices/{device_id}"),
        ("PUT", f"/api/v3/ns/applications/{APP}/devices/{device_id}"),
        ("PUT", f"/api/v3/as/applications/{APP}/devices/{device_id}"),
    ]
    assert env.tts.js[device_id] == derive_app_key(MASTER, DEV_A).hex()      # the DERIVED key is what TTS holds
    ns_body = json.loads(env.tts.bodies[2])["end_device"]
    assert ns_body["supports_class_c"] and ns_body["mac_settings"]["desired_rx2_data_rate_index"] == "DATA_RATE_5"
    assert ns_body["frequency_plan_id"] == "EU_863_870_TTN" and ns_body["lorawan_version"] == "MAC_V1_0_3"
    assert env.manager.registry.contains(DEV_A) and DEV_A in env.manager.status()["registered"]
    assert "enrollment_registered" in names(env)


def test_two_devices_get_two_different_keys(env):
    env.manager.evaluate(join(DEV_A))
    env.manager.evaluate(join(DEV_B))
    assert env.tts.js[device_id_from_dev_eui(DEV_A)] != env.tts.js[device_id_from_dev_eui(DEV_B)]


def test_a_device_not_in_the_roster_is_refused_even_with_a_valid_mic(env):
    decision = env.manager.evaluate(join(DEV_C))               # a correct MIC under DEV_C's derived key
    assert decision.outcome is en.Outcome.NOT_ROSTERED
    assert env.tts.calls == [] and not env.manager.registry.contains(DEV_C)
    assert env.manager.status()["not_rostered"] == 1 and "enrollment_not_rostered" in names(env)
    assert len(env.manager.nonces) == 0                         # nothing is recorded for an unauthorised device


def test_a_roster_entry_added_while_running_authorises_the_device(env):
    assert env.manager.evaluate(join(DEV_C)).outcome is en.Outcome.NOT_ROSTERED
    Roster(env.tmp / "roster.json").add(DEV_C)                  # provision_board.py appends to the same file
    assert env.manager.evaluate(join(DEV_C, nonce=2)).outcome is en.Outcome.REGISTERED


@pytest.mark.parametrize("make", [
    lambda: join(DEV_A, key=bytes(16)),                          # a wrong key
    lambda: join(DEV_A, master=OTHER_MASTER),                    # another deployment's master key
    lambda: join(DEV_A, key=derive_app_key(MASTER, DEV_B)),      # a neighbour's derived key
])
def test_a_bad_mic_is_refused_and_registers_nothing(env, make):
    decision = env.manager.evaluate(make())
    assert decision.outcome is en.Outcome.BAD_MIC
    assert env.tts.calls == [] and not env.manager.registry.contains(DEV_A)
    assert env.manager.status()["rejected_bad_mic"] == {DEV_A: 1}
    assert len(env.manager.nonces) == 0                         # a forged request cannot poison the nonce store


def test_the_mic_covers_the_first_19_bytes_and_is_compared_in_constant_time(env, monkeypatch):
    frame = bytearray(build_join_request(derive_app_key(MASTER, DEV_A), JOIN_EUI, DEV_A, 5))
    seen = []
    real = lorawan_join.hmac.compare_digest
    monkeypatch.setattr(lorawan_join.hmac, "compare_digest", lambda a, b: seen.append((a, b)) or real(a, b))
    for index in (1, 9, 17, 18):                                  # JoinEUI, DevEUI, DevNonce bytes of the signed body
        tampered = bytearray(frame)
        tampered[index] ^= 0x01
        request = _parse(bytes(tampered))
        if index == 9:                                            # keep the device rostered: flip within the DevEUI tail only
            continue
        assert env.manager.evaluate(request).outcome in (en.Outcome.BAD_MIC, en.Outcome.JOIN_EUI_MISMATCH)
    flipped_mic = bytearray(frame)
    flipped_mic[22] ^= 0x01
    assert env.manager.evaluate(_parse(bytes(flipped_mic))).outcome is en.Outcome.BAD_MIC
    assert seen and all(len(a) == 4 and len(b) == 4 for a, b in seen)    # 4-byte MICs through compare_digest
    expected = lorawan_join.compute_join_request_mic(bytes(frame[:19]), derive_app_key(MASTER, DEV_A))
    assert bytes(frame[19:]) == expected


def test_a_replayed_devnonce_is_rejected_and_persisted_across_restarts(env):
    request = join(DEV_A, nonce=77)
    assert env.manager.evaluate(request).outcome is en.Outcome.REGISTERED
    again = env.manager.evaluate(request)
    assert again.outcome is en.Outcome.REPLAYED_DEV_NONCE and "77" in again.detail
    assert len(env.tts.calls) == 4                                  # no second registration attempt
    assert env.manager.status()["replays"] == 1 and "enrollment_replayed_dev_nonce" in names(env)
    # a new process: the store is read back from disk, the same pair is still refused
    env.manager.nonces.close()
    store = en.JoinNonceStore(env.nonce_path)
    assert store.seen(DEV_A, 77) and not store.seen(DEV_A, 78) and not store.seen(DEV_B, 77)
    assert store.add(DEV_A, 77) is False and store.add(DEV_A, 78) is True
    store.close()
    lines = env.nonce_path.read_text().splitlines()
    assert {json.loads(x)["dev_nonce"] for x in lines} == {77, 78}
    # a torn last line (crash mid-write) does not lose the earlier ones
    env.nonce_path.write_text("\n".join(lines) + '\n{"dev_eui": "70B3')
    assert en.JoinNonceStore(env.nonce_path).seen(DEV_A, 77)


def test_a_fresh_devnonce_of_an_enrolled_device_is_idempotent(env):
    assert env.manager.evaluate(join(DEV_A, nonce=1)).outcome is en.Outcome.REGISTERED
    calls = len(env.tts.calls)
    assert env.manager.evaluate(join(DEV_A, nonce=2)).outcome is en.Outcome.ALREADY_REGISTERED   # a normal rejoin
    assert len(env.tts.calls) == calls


def test_window_and_joineui_checks(env):
    assert env.manager.evaluate(join(DEV_A, join_eui="0000000000000001")).outcome is en.Outcome.JOIN_EUI_MISMATCH
    env.manager.close_window("test")
    assert env.manager.evaluate(join(DEV_A)).outcome is en.Outcome.WINDOW_CLOSED
    assert env.tts.calls == [] and len(env.manager.nonces) == 0
    env.manager.open_window("test", seconds=30)
    env.clock.t += 31
    assert env.manager.evaluate(join(DEV_A)).outcome is en.Outcome.WINDOW_CLOSED
    # window-closed and foreign-JoinEUI traffic is the normal state of a busy gateway: not logged
    assert not [n for n in names(env) if n.startswith("enrollment_") and "window" not in n]


# ---------------------------------------------------------------------------
# Idempotent registration with rollback
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage, fragment", [("join_server", "/js/"), ("network_server", "/ns/"),
                                             ("application_server", "/as/")])
def test_a_failed_stage_rolls_back_what_this_call_created_and_a_retry_succeeds(env, stage, fragment):
    device_id = device_id_from_dev_eui(DEV_A)
    env.tts.fail[("PUT", fragment)] = (500, b'{"error":"boom"}')
    decision = env.manager.evaluate(join(DEV_A, nonce=1))
    assert decision.outcome is en.Outcome.REGISTRATION_FAILED and stage in decision.detail
    assert not env.manager.registry.contains(DEV_A)
    assert not env.tts.residue(device_id)                           # identity, JS, NS, AS: nothing left behind
    deletes = [c for c in env.tts.calls if c[0] == "DELETE"]
    assert [d[1].split("/")[3] for d in deletes] == ["as", "ns", "js", "applications"]
    retry = env.manager.evaluate(join(DEV_A, nonce=2))              # a fresh DevNonce: the board tries again
    assert retry.outcome is en.Outcome.REGISTERED
    assert env.tts.identities == {device_id} and env.tts.js[device_id] == derive_app_key(MASTER, DEV_A).hex()
    assert env.manager.registry.contains(DEV_A)


def test_an_existing_identity_is_reused_and_never_deleted_by_a_failed_retry(env):
    device_id = device_id_from_dev_eui(DEV_A)
    env.tts.identities.add(device_id)                                # registered by hand or an earlier success
    env.tts.fail[("PUT", "/js/")] = (500, b'{"error":"boom"}')
    decision = env.manager.evaluate(join(DEV_A, nonce=1))
    assert decision.outcome is en.Outcome.REGISTRATION_FAILED
    assert [c for c in env.tts.calls if c[0] == "DELETE"] == []      # nothing pre-existing is deleted
    assert device_id in env.tts.identities
    ok = env.manager.evaluate(join(DEV_A, nonce=2))                  # the PUTs re-apply keys and settings
    assert ok.outcome is en.Outcome.REGISTERED
    assert env.tts.js[device_id] == derive_app_key(MASTER, DEV_A).hex() and device_id in env.tts.ns
    entry = env.manager.registry.devices()[DEV_A]
    assert entry["source"] == "already_in_tts" and "key" not in json.dumps(entry).replace('"key": "derived"', "")


def test_tts_down_is_a_failed_registration_not_a_crash(env):
    def boom(*a):
        raise ConnectionError("TTS unreachable")

    env.manager.client.transport = boom
    decision = env.manager.evaluate(join(DEV_A))
    assert decision.outcome is en.Outcome.REGISTRATION_FAILED and "ConnectionError" in decision.detail
    assert not env.manager.registry.contains(DEV_A)


def test_an_error_body_that_echoes_the_key_is_redacted(env):
    key_hex = derive_app_key(MASTER, DEV_A).hex()
    env.tts.fail[("PUT", "/ns/")] = (400, f'{{"error":"bad key {key_hex}"}}'.encode())
    decision = env.manager.evaluate(join(DEV_A))
    assert decision.outcome is en.Outcome.REGISTRATION_FAILED
    assert key_hex not in decision.detail and "<redacted>" in decision.detail


# ---------------------------------------------------------------------------
# Rate-limited logging
# ---------------------------------------------------------------------------
def test_bad_mic_logging_is_rate_limited_with_a_suppressed_count(env):
    for n in range(50):
        env.manager.evaluate(join(DEV_A, key=bytes(16), nonce=n))
    logged = [x for e, d, x in env.events if e == "enrollment_bad_mic"]
    assert len(logged) == 1 and env.manager.status()["rejected_bad_mic"][DEV_A] == 50
    env.clock.t += 61
    env.manager.evaluate(join(DEV_A, key=bytes(16), nonce=99))
    logged = [x for e, d, x in env.events if e == "enrollment_bad_mic"]
    assert len(logged) == 2 and logged[1]["suppressed_since_last"] == 49
    # another device has its own budget
    env.manager.evaluate(join(DEV_B, key=bytes(16)))
    assert len([1 for e, d, x in env.events if e == "enrollment_bad_mic"]) == 3
    for n in range(20):
        env.manager.evaluate(join(DEV_C, nonce=n))
    assert len([1 for e, d, x in env.events if e == "enrollment_not_rostered"]) == 1


def test_no_key_ever_reaches_a_log_the_registry_or_the_nonce_store(env):
    secrets = {MASTER.hex(), derive_app_key(MASTER, DEV_A).hex(), derive_app_key(MASTER, DEV_B).hex()}
    env.manager.evaluate(join(DEV_A))
    env.manager.evaluate(join(DEV_A, key=bytes(16), nonce=3))
    env.manager.evaluate(join(DEV_B))
    env.tts.fail[("PUT", "/js/")] = (500, derive_app_key(MASTER, DEV_C).hex().encode())
    Roster(env.tmp / "roster.json").add(DEV_C)
    env.manager.evaluate(join(DEV_C, nonce=4))
    secrets.add(derive_app_key(MASTER, DEV_C).hex())
    text = json.dumps(env.events, default=str).lower()
    text += env.registry_path.read_text().lower() + env.nonce_path.read_text().lower()
    text += (env.tmp / "roster.json").read_text().lower()
    for secret in secrets:
        assert secret.lower() not in text
    assert not re.search(r"[0-9a-f]{32}", text.replace(DEV_A.lower(), "")), "a 32-hex string was logged"


# ---------------------------------------------------------------------------
# The listener, the roster file and the wiring
# ---------------------------------------------------------------------------
def event_line(frame: bytes) -> str:
    return "data: " + json.dumps({"result": {"name": "gs.up.receive", "data": {
        "message": {"raw_payload": base64.b64encode(frame).decode()}}}})


def test_the_listener_applies_the_strict_classification(env):
    listener = en.JoinListenerV4(env.manager, "gateway-key", lambda *a: None)
    good = build_join_request(derive_app_key(MASTER, DEV_A), JOIN_EUI, DEV_A, 1)
    assert listener.handle_line(event_line(good)).outcome is en.Outcome.REGISTERED
    assert listener.handle_line(event_line(good[:22])) is None                     # truncated: ignored, logged once
    assert listener.handle_line(event_line(good[:22])) is None
    assert [e for e, _, _ in env.events].count("enrollment_malformed_join") == 1
    data_frame = bytes([0x40]) + bytes(12)
    assert listener.handle_line(event_line(data_frame)) is None
    assert listener.handle_line("data: not json") is None
    assert listener.handle_line("") is None


def test_roster_file_round_trip_and_validation(tmp_path):
    path = tmp_path / "sub" / "roster.json"
    roster = Roster(path)
    assert len(roster) == 0 and not roster.contains(DEV_A)
    assert roster.add(DEV_A.lower(), "2026-10-01T00:00:00+00:00") is True
    assert roster.add(DEV_A) is False                                              # idempotent
    data = json.loads(path.read_text())
    assert [d["dev_eui"] for d in data["devices"]] == [DEV_A] and set(data["devices"][0]) == {"dev_eui", "provisioned_utc"}
    assert Roster(path).contains("eui-" + DEV_A.lower()) and Roster(path).contains(DEV_A.lower())
    for bad in ("123", "zz" * 8, ""):
        assert not roster.contains(bad)
        with pytest.raises(RosterError):
            normalise_dev_eui(bad)
    path.write_text('{"devices": [{"dev_eui": "nope"}]}')
    with pytest.raises(RosterError):
        Roster(path)
    path.write_text("not json")
    with pytest.raises(RosterError):
        Roster(path)


def test_build_enrollment_reads_secrets_from_the_environment_only(tmp_path):
    config = {"tts": {"application_id": APP, "console_url": "http://localhost:1885"},
              "experiment": {"multicast_gateway_ids": ["gw-1"]},
              "v4": {"roster_path": "roster.json"},
              "enrollment": {"join_eui": JOIN_EUI, "registry_path": "reg.json", "nonce_store_path": "n.jsonl"}}
    Roster(tmp_path / "roster.json").add(DEV_A)
    base_env = {"TINYFL_MASTER_KEY": MASTER.hex(), "TTS_GATEWAY_API_KEY": "g", "TTS_DEVICE_API_KEY": "d"}
    manager, listener = en.build_enrollment(config, lambda *a: None, tmp_path, env=base_env)
    assert manager.config.master_key_env == "TINYFL_MASTER_KEY" and manager.roster.contains(DEV_A)
    assert (tmp_path / "n.jsonl").exists()                                        # the nonce store is opened up front
    for missing in ("TTS_GATEWAY_API_KEY", "TTS_DEVICE_API_KEY"):
        broken = {k: v for k, v in base_env.items() if k != missing}
        with pytest.raises(RuntimeError, match=missing):
            en.build_enrollment(config, lambda *a: None, tmp_path, env=broken)
    with pytest.raises(KeyError_, match="TINYFL_MASTER_KEY"):
        en.build_enrollment(config, lambda *a: None, tmp_path, env={k: v for k, v in base_env.items() if k != "TINYFL_MASTER_KEY"})
    with pytest.raises(KeyError_) as err:
        en.build_enrollment(config, lambda *a: None, tmp_path, env=dict(base_env, TINYFL_MASTER_KEY="12" * 15))
    assert "12" * 15 not in str(err.value)


def test_quota_is_ignored_and_the_window_is_bounded(env):
    status = env.manager.open_window("console", seconds=10 ** 9, quota=1)
    assert status["open"] and status["seconds_remaining"] <= 6 * 3600
    assert env.manager.evaluate(join(DEV_A)).outcome is en.Outcome.REGISTERED
    assert env.manager.evaluate(join(DEV_B)).outcome is en.Outcome.REGISTERED      # the roster, not a quota, authorises
