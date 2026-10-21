"""Auto-enrollment: the MIC is the security boundary. No network: TTS is a recording fake."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re

import pytest
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.cmac import CMAC

import enrollment as en
from enrollment import (
    EnrollmentConfig,
    EnrollmentManager,
    EnrollmentRegistry,
    JoinListener,
    Outcome,
    TtsRegistrationClient,
)
from lorawan_join import (
    JOIN_REQUEST_LENGTH,
    app_key_bytes,
    build_join_request,
    compute_join_request_mic,
    device_id_from_dev_eui,
    parse_join_request,
    verify_join_request_mic,
)

SERVER = Path(__file__).resolve().parents[1]

# Throwaway values that exist only in this file (same convention as the Smart
# Building tests): a real AppKey must never appear in the repository.
APP_KEY_HEX = "0F1E2D3C4B5A69788796A5B4C3D2E1F0"
APP_KEY = bytes.fromhex(APP_KEY_HEX)
ATTACKER_KEY = bytes.fromhex("FFEEDDCCBBAA99887766554433221100")
JOIN_EUI = "70B3D57ED0000001"
DEV_EUI = "AABBCCDDEEFF0011"


# ---------------------------------------------------------------------------
# Known-answer vectors
# ---------------------------------------------------------------------------
RFC4493_KEY = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")


@pytest.mark.parametrize("message_hex, expected_hex", [
    ("", "bb1d6929e95937287fa37d129b756746"),
    ("6bc1bee22e409f96e93d7e117393172a", "070a16b46b4d4144f79bdd9dd04a287c"),
    ("6bc1bee22e409f96e93d7e117393172a" "ae2d8a571e03ac9c9eb76fac45af8e51" "30c81c46a35ce411",
     "dfa66747de9ae63030ca32611497c827"),
])
def test_cmac_primitive_matches_rfc4493(message_hex, expected_hex):
    """The primitive under every MIC, pinned to the published AES-CMAC answers."""
    got = compute_join_request_mic(bytes.fromhex(message_hex), RFC4493_KEY)
    assert got == bytes.fromhex(expected_hex)[:4]


# A JoinRequest assembled BY HAND from the LoRaWAN 1.0.x layout (MHDR | JoinEUI LE |
# DevEUI LE | DevNonce LE | MIC), its MIC computed with the cryptography library
# directly (not through lorawan_join) and pinned here as a constant.
KAT_BODY = bytes.fromhex("00" "010000d07ed5b370" "1100ffeeddccbbaa" "b2a1")
KAT_MIC = bytes.fromhex("d3028f7b")


def test_join_request_mic_known_answer():
    mac = CMAC(algorithms.AES(APP_KEY))
    mac.update(KAT_BODY)
    assert mac.finalize()[:4] == KAT_MIC                         # independent of our module
    assert compute_join_request_mic(KAT_BODY, APP_KEY) == KAT_MIC
    frame = KAT_BODY + KAT_MIC
    assert build_join_request(APP_KEY, JOIN_EUI, DEV_EUI, 0xA1B2) == frame
    request = parse_join_request(frame)
    assert (request.join_eui, request.dev_eui, request.dev_nonce) == (JOIN_EUI, DEV_EUI, 0xA1B2)
    assert request.mic == KAT_MIC and request.signed_body == KAT_BODY
    assert verify_join_request_mic(request, APP_KEY) is True
    assert verify_join_request_mic(request, ATTACKER_KEY) is False


def test_parses_the_real_captured_join_request_the_way_the_things_stack_does():
    """Frame and TTS decoding taken from the Smart Building tests (gs.up.receive of a TTGO LoRa32)."""
    request = parse_join_request(base64.b64decode("AAAAAAAAAAAAqwAA0H7Vs3B0nrvvgko="))
    assert request.join_eui == "0000000000000000"
    assert request.dev_eui == "70B3D57ED00000AB"
    assert request.dev_nonce == 0x9E74
    assert request.mic.hex().upper() == "BBEF824A"
    assert request.signed_body + request.mic == request.raw and len(request.signed_body) == 19


@pytest.mark.parametrize("payload", [b"", b"\x00" * 22, b"\x00" * 24, "text", None])
def test_non_join_payloads_are_not_parsed(payload):
    assert parse_join_request(payload) is None


@pytest.mark.parametrize("mhdr", [0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0])
def test_other_message_types_are_ignored(mhdr):
    assert parse_join_request(build_join_request(APP_KEY, JOIN_EUI, DEV_EUI, 1, mhdr=mhdr)) is None


@pytest.mark.parametrize("offset", [1, 9, 16, 17, 19, 22])
def test_any_tampered_byte_fails_the_mic(offset):
    frame = bytearray(build_join_request(APP_KEY, JOIN_EUI, DEV_EUI, 7))
    frame[offset] ^= 0x01
    assert verify_join_request_mic(parse_join_request(bytes(frame)), APP_KEY) is False


def test_malformed_keys_fail_closed():
    request = parse_join_request(build_join_request(APP_KEY, JOIN_EUI, DEV_EUI, 7))
    assert verify_join_request_mic(request, b"short") is False
    for bad in ("", None, "C830", "zz" * 16, "AB" * 15, "AB" * 17):
        with pytest.raises(ValueError):
            app_key_bytes(bad)
    assert app_key_bytes("0F:1E:2D:3C-4B5A69788796A5B4C3D2E1F0") == APP_KEY
    assert JOIN_REQUEST_LENGTH == 23


def test_device_id_is_eui_lowercase():
    assert device_id_from_dev_eui("AABBCCDDEEFF0011") == "eui-aabbccddeeff0011"
    assert device_id_from_dev_eui("aa:bb:cc:dd:ee:ff:00:11") == "eui-aabbccddeeff0011"


# ---------------------------------------------------------------------------
# A recording fake of The Things Stack
# ---------------------------------------------------------------------------
class FakeTts:
    def __init__(self, fail_at: str | None = None, already_exists: bool = False) -> None:
        self.calls: list[tuple] = []
        self.fail_at = fail_at
        self.already_exists = already_exists

    def transport(self, method, url, headers, body, timeout):
        payload = None if body is None else json.loads(body)
        self.calls.append((method, url.split("/api/v3", 1)[1], payload, headers))
        path = url
        if self.already_exists and method == "POST":
            return 409, b'{"code":6,"message":"error:pkg/identityserver:id_taken"}'
        stage = ("/js/" in path and "js") or ("/ns/" in path and "ns") or ("/as/" in path and "as")
        if self.fail_at and stage == self.fail_at and method == "PUT":
            return 500, b'{"message":"boom"}'
        return 200, b"{}"

    @property
    def paths(self):
        return [(m, p) for m, p, _, _ in self.calls]


def make_manager(tmp_path, tts=None, quota=3, clock=None, join_eui=JOIN_EUI, log=None):
    tts = tts or FakeTts()
    config = EnrollmentConfig(
        api_base_url="http://tts.example:1885", application_id="tinyfl-v91-hw",
        join_eui=join_eui, gateway_ids=("eui-00000000000000a1",),
        cluster_address="tts.example", quota=quota, window_seconds=600,
        registry_path=str(tmp_path / "enrolled.json"),
    )
    events = [] if log is None else log
    now = [1000.0]
    manager = EnrollmentManager(
        config, APP_KEY_HEX,
        TtsRegistrationClient(config, "fake-device-key", transport=tts.transport),
        EnrollmentRegistry(config.registry_path),
        lambda e, d, x: events.append((e, d, x)),
        clock=clock or (lambda: now[0]),
    )
    manager.events = events
    manager.tts = tts
    manager.now = now
    return manager


def request_for(dev_eui=DEV_EUI, key=APP_KEY, join_eui=JOIN_EUI, nonce=0x1234):
    return parse_join_request(build_join_request(key, join_eui, dev_eui, nonce))


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------
def test_a_valid_mic_registers_the_device_with_this_experiments_settings(tmp_path):
    manager = make_manager(tmp_path)
    manager.open_window("tester")
    decision = manager.evaluate(request_for())

    assert decision.outcome is Outcome.REGISTERED and decision.enrolled
    assert decision.detail == "eui-aabbccddeeff0011"
    calls = {(m, p.split("?")[0]): body for m, p, body, _ in manager.tts.calls}
    app = "tinyfl-v91-hw"
    device = "eui-aabbccddeeff0011"
    assert [(m, p) for m, p, _, _ in manager.tts.calls] == [
        ("POST", f"/applications/{app}/devices"),
        ("PUT", f"/js/applications/{app}/devices/{device}"),
        ("PUT", f"/ns/applications/{app}/devices/{device}"),
        ("PUT", f"/as/applications/{app}/devices/{device}"),
    ]
    identity = calls[("POST", f"/applications/{app}/devices")]["end_device"]
    assert identity["ids"] == {"device_id": device, "dev_eui": DEV_EUI, "join_eui": JOIN_EUI}
    assert identity["lorawan_version"] == "MAC_V1_0_3"
    assert identity["lorawan_phy_version"] == "PHY_V1_0_3_REV_A"
    assert identity["supports_join"] is True
    assert identity["join_server_address"] == "tts.example"

    join_server = calls[("PUT", f"/js/applications/{app}/devices/{device}")]["end_device"]
    assert join_server["root_keys"]["app_key"]["key"] == APP_KEY_HEX     # the shared AppKey

    network = calls[("PUT", f"/ns/applications/{app}/devices/{device}")]
    end_device = network["end_device"]
    assert end_device["lorawan_version"] == "MAC_V1_0_3"
    assert end_device["lorawan_phy_version"] == "PHY_V1_0_3_REV_A"
    assert end_device["frequency_plan_id"] == "EU_863_870_TTN"
    assert end_device["supports_join"] is True                              # OTAA
    assert end_device["supports_class_c"] is True
    assert end_device["mac_settings"] == {"desired_rx2_data_rate_index": "DATA_RATE_5",
                                          "desired_rx2_frequency": "869525000"}
    assert set(network["field_mask"]["paths"]) >= {
        "supports_class_c", "mac_settings.desired_rx2_data_rate_index",
        "mac_settings.desired_rx2_frequency", "lorawan_version", "lorawan_phy_version",
        "frequency_plan_id", "supports_join"}
    # every request is authenticated with the device key, and no other secret leaks
    assert all(h["Authorization"] == "Bearer fake-device-key" for _, _, _, h in manager.tts.calls)

    assert manager.status()["registered"] == [DEV_EUI] and manager.status()["quota_used"] == 1
    stored = json.loads((tmp_path / "enrolled.json").read_text())
    assert stored["devices"][DEV_EUI]["device_id"] == device
    assert stored["devices"][DEV_EUI]["mic_verified"] is True
    assert APP_KEY_HEX not in (tmp_path / "enrolled.json").read_text()
    assert APP_KEY_HEX not in json.dumps(manager.events)


def test_a_bad_mic_is_refused_logged_and_never_registered(tmp_path):
    manager = make_manager(tmp_path)
    manager.open_window("tester")
    decision = manager.evaluate(request_for(key=ATTACKER_KEY))

    assert decision.outcome is Outcome.BAD_MIC
    assert manager.tts.calls == [], "nothing may reach TTS"
    assert not manager.registry.contains(DEV_EUI)
    assert manager.status()["rejected"] == {DEV_EUI: 1}
    assert manager.status()["quota_used"] == 0
    names = [e for e, _, _ in manager.events]
    assert "enrollment_bad_mic" in names
    manager.evaluate(request_for(key=ATTACKER_KEY, nonce=2))
    assert manager.status()["rejected"] == {DEV_EUI: 2}


def test_a_replayed_frame_with_a_swapped_deveui_is_refused(tmp_path):
    manager = make_manager(tmp_path)
    manager.open_window("tester")
    forged = bytearray(build_join_request(APP_KEY, JOIN_EUI, DEV_EUI, 5))
    forged[9] ^= 0xFF                                       # another DevEUI, original MIC kept
    decision = manager.evaluate(parse_join_request(bytes(forged)))
    assert decision.outcome is Outcome.BAD_MIC and manager.tts.calls == []


def test_nothing_is_registered_while_the_window_is_closed(tmp_path):
    manager = make_manager(tmp_path)
    assert manager.evaluate(request_for()).outcome is Outcome.WINDOW_CLOSED
    manager.open_window("tester", seconds=60)
    manager.now[0] += 61                                    # lapses
    assert manager.evaluate(request_for()).outcome is Outcome.WINDOW_CLOSED
    manager.open_window("tester")
    manager.close_window("tester")
    assert manager.evaluate(request_for()).outcome is Outcome.WINDOW_CLOSED
    assert manager.tts.calls == []
    # a closed window is the normal state of a busy gateway and is not logged per frame
    assert not any(e == "enrollment_window_closed_attempt" for e, _, _ in manager.events)


def test_the_quota_bounds_one_window(tmp_path):
    manager = make_manager(tmp_path, quota=3)
    manager.open_window("tester")
    euis = [f"AABBCCDDEEFF00{i:02X}" for i in range(1, 5)]
    outcomes = [manager.evaluate(request_for(dev_eui=e)).outcome for e in euis]
    assert outcomes == [Outcome.REGISTERED] * 3 + [Outcome.QUOTA_EXCEEDED]
    assert len([c for c in manager.tts.calls if c[0] == "POST"]) == 3
    assert manager.status()["quota"] == 3
    # A bad MIC does not spend quota, and a fresh window resets it.
    manager.open_window("tester", quota=1)
    assert manager.evaluate(request_for(dev_eui=euis[3])).outcome is Outcome.REGISTERED
    assert manager.evaluate(request_for(dev_eui="AABBCCDDEEFF0099")).outcome is Outcome.QUOTA_EXCEEDED


def test_the_window_can_carry_its_own_quota_and_duration(tmp_path):
    manager = make_manager(tmp_path)
    status = manager.open_window("cli", seconds=120, quota=7)
    assert status["quota"] == 7 and 0 < status["seconds_remaining"] <= 120
    assert status["open"] and status["opened_by"] == "cli"
    assert manager.close_window("cli")["open"] is False


def test_another_networks_join_eui_is_ignored(tmp_path):
    manager = make_manager(tmp_path, join_eui="0000000000000000")
    manager.open_window("tester")
    decision = manager.evaluate(request_for())
    assert decision.outcome is Outcome.JOIN_EUI_MISMATCH and manager.tts.calls == []


def test_a_device_already_on_record_is_left_alone(tmp_path):
    manager = make_manager(tmp_path)
    manager.open_window("tester")
    assert manager.evaluate(request_for()).outcome is Outcome.REGISTERED
    calls = len(manager.tts.calls)
    assert manager.evaluate(request_for(nonce=99)).outcome is Outcome.ALREADY_REGISTERED
    assert len(manager.tts.calls) == calls


def test_the_registry_survives_a_restart(tmp_path):
    first = make_manager(tmp_path)
    first.open_window("tester")
    first.evaluate(request_for())
    second = make_manager(tmp_path)
    second.open_window("tester")
    assert second.evaluate(request_for()).outcome is Outcome.ALREADY_REGISTERED
    assert second.tts.calls == []


def test_a_device_already_in_tts_is_not_a_failure(tmp_path):
    manager = make_manager(tmp_path, tts=FakeTts(already_exists=True))
    manager.open_window("tester")
    decision = manager.evaluate(request_for())
    assert decision.outcome is Outcome.ALREADY_REGISTERED
    assert manager.status()["quota_used"] == 0
    assert manager.registry.devices()[DEV_EUI]["source"] == "already_in_tts"


@pytest.mark.parametrize("stage", ["js", "ns", "as"])
def test_a_failure_after_the_identity_rolls_the_partial_device_back(tmp_path, stage):
    manager = make_manager(tmp_path, tts=FakeTts(fail_at=stage))
    manager.open_window("tester")
    decision = manager.evaluate(request_for())
    assert decision.outcome is Outcome.REGISTRATION_FAILED
    deletes = [p for m, p in manager.tts.paths if m == "DELETE"]
    assert len(deletes) == 4 and deletes[-1].startswith("/applications/")
    assert manager.status()["quota_used"] == 0              # a failed attempt costs no quota
    assert not manager.registry.contains(DEV_EUI)


def test_an_unreachable_tts_does_not_raise(tmp_path):
    manager = make_manager(tmp_path)

    def boom(*args, **kwargs):
        raise ConnectionError("TTS unreachable")

    manager.client.transport = boom
    manager.open_window("tester")
    decision = manager.evaluate(request_for())
    assert decision.outcome is Outcome.REGISTRATION_FAILED and "unreachable" in decision.detail


def test_a_broken_app_key_fails_closed(tmp_path):
    manager = make_manager(tmp_path)
    manager._app_key_hex = "not-a-key"
    manager.open_window("tester")
    assert manager.evaluate(request_for()).outcome is Outcome.MISCONFIGURED
    assert manager.tts.calls == []


# ---------------------------------------------------------------------------
# Gateway event stream
# ---------------------------------------------------------------------------
def gateway_event(frame: bytes, name="gs.up.receive", shape="message") -> str:
    raw = base64.b64encode(frame).decode()
    data = {"message": {"raw_payload": raw}} if shape == "message" else {"raw_payload": raw}
    return "data: " + json.dumps({"result": {"name": name, "data": data}})


def test_listener_turns_gateway_events_into_decisions(tmp_path):
    manager = make_manager(tmp_path)
    manager.open_window("tester")
    listener = JoinListener(manager, "gateway-key", manager.log)
    join = build_join_request(APP_KEY, JOIN_EUI, DEV_EUI, 3)
    assert listener.handle_line(gateway_event(join)).outcome is Outcome.REGISTERED
    other = build_join_request(APP_KEY, JOIN_EUI, "AABBCCDDEEFF0022", 3)
    assert listener.handle_line(gateway_event(other, shape="top")).outcome is Outcome.REGISTERED
    # a data uplink, a status event, junk and keep-alives are not join requests
    data_frame = bytes([0x40]) + bytes(20)
    assert listener.handle_line(gateway_event(data_frame)) is None
    assert listener.handle_line(gateway_event(join, name="gs.status.receive")) is None
    assert listener.handle_line("not json") is None
    assert listener.handle_line('{"result": {"name": "x", "data": {}}}') is None
    bad = build_join_request(ATTACKER_KEY, JOIN_EUI, "AABBCCDDEEFF0033", 3)
    assert listener.handle_line(gateway_event(bad)).outcome is Outcome.BAD_MIC
    assert manager.status()["registered"] == [DEV_EUI, "AABBCCDDEEFF0022"]


def test_listener_thread_reads_a_stream_and_stops_when_the_window_closes(tmp_path):
    manager = make_manager(tmp_path)
    manager.open_window("tester")
    lines = [gateway_event(build_join_request(APP_KEY, JOIN_EUI, DEV_EUI, 3)).encode() + b"\n"]

    class Stream:
        def __enter__(self):
            return iter(lines)

        def __exit__(self, *exc):
            return False

    listener = JoinListener(manager, "gateway-key", manager.log, opener=lambda: Stream())
    assert listener.start()
    for _ in range(200):
        if manager.status()["registered"]:
            break
        import time
        time.sleep(0.01)
    assert manager.status()["registered"] == [DEV_EUI]
    manager.close_window("test")
    listener.stop()
    for _ in range(200):
        if not listener.is_running():
            break
        import time
        time.sleep(0.01)
    assert not listener.is_running()


def test_listener_needs_a_gateway(tmp_path):
    manager = make_manager(tmp_path)
    manager.config.gateway_ids = ()
    assert JoinListener(manager, "key", manager.log).start() is False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_example_config_carries_the_enrollment_fields_and_no_secret():
    import tomllib

    text = (SERVER / "config.toml.example").read_text(encoding="utf-8")
    config = tomllib.loads(text)
    section = config["enrollment"]
    for key in ("gateway_api_key_env", "join_eui", "window_seconds", "quota"):
        assert key in section, key
    parsed = EnrollmentConfig.from_config(config)
    assert parsed.quota == 10 and parsed.window_seconds > 0
    assert parsed.rx2_data_rate_index == "DATA_RATE_5" and parsed.rx2_frequency == "869525000"
    # the example must never hold a key: no 32-hex-digit runs, no NNSXS tokens
    assert not re.search(r"(?i)\b[0-9a-f]{32}\b", text)
    assert "NNSXS" not in text
    assert config["v3"]["levels"] == [4, 8, 12, 16, 20, 24, 28]


def test_build_enrollment_reads_secrets_from_the_environment_only(tmp_path):
    config = {"tts": {"application_id": "tinyfl-v91-hw"},
              "experiment": {"multicast_gateway_ids": ["eui-00000000000000a1"]},
              "enrollment": {"join_eui": JOIN_EUI, "registry_path": "reg.json"}}
    with pytest.raises(RuntimeError, match="TINYFL_APP_KEY"):
        en.build_enrollment(config, lambda *a: None, tmp_path, env={})
    env = {"TINYFL_APP_KEY": APP_KEY_HEX, "TTS_GATEWAY_API_KEY": "g", "TTS_DEVICE_API_KEY": "d"}
    manager, listener = en.build_enrollment(config, lambda *a: None, tmp_path, env=env)
    assert manager.config.gateway_ids == ("eui-00000000000000a1",)   # falls back to the multicast gateway
    assert manager.config.join_eui == JOIN_EUI
    assert manager.registry.path == tmp_path / "reg.json"
    env["TINYFL_APP_KEY"] = "short"
    with pytest.raises(ValueError):
        en.build_enrollment(config, lambda *a: None, tmp_path, env=env)
