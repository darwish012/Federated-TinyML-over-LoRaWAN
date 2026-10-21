"""MQTT glue of the v3 runner, without paho or a broker."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import protocol_v3 as p3
import tts_v3_server as srv

SERVER = Path(__file__).resolve().parents[1]


def uplink_event(frame: bytes | None, f_port=10, f_cnt=7) -> dict:
    message = {"f_port": f_port, "f_cnt": f_cnt,
               "settings": {"frequency": "868100000",
                            "data_rate": {"lora": {"spreading_factor": 7, "bandwidth": 125000}}},
               "rx_metadata": [{"rssi": -80, "snr": 9.5}]}
    if frame is not None:
        message["frm_payload"] = base64.b64encode(frame).decode()
    return {"uplink_message": message}


def test_decode_up_event_extracts_payload_fcnt_and_radio():
    frame = p3.encode_class_c_ready(3, 5, 5)
    decoded = srv.decode_up_event(uplink_event(frame))
    assert decoded["frame"] == frame and decoded["f_port"] == 10 and decoded["f_cnt"] == 7
    assert decoded["radio"] == {"spreading_factor": 7, "bandwidth": 125000,
                                "frequency": "868100000", "rssi": -80, "snr": 9.5}


def test_mac_only_uplink_has_no_payload_and_is_not_an_error():
    assert srv.decode_up_event(uplink_event(None)) is None


def test_missing_fcnt_is_allowed():
    event = uplink_event(p3.encode_class_c_ready(0, 5, 5))
    del event["uplink_message"]["f_cnt"]
    assert srv.decode_up_event(event)["f_cnt"] is None


def make_runner():
    runner = srv.TtsMqttV3Experiment.__new__(srv.TtsMqttV3Experiment)
    runner.topic_root = "v3/tinyfl-v91-hw"
    runner.events = []
    runner.log = lambda e, d, x: runner.events.append((e, d, x))
    calls = []
    runner.controller = SimpleNamespace(
        handle_uplink=lambda d, port, frame: calls.append(("up", d, port, frame)),
        handle_downlink_sent=lambda d: calls.append(("sent", d)),
        handle_downlink_failed=lambda d, r: calls.append(("failed", d, r)),
    )
    runner.calls = calls
    return runner


def message(topic: str, event: dict):
    return SimpleNamespace(topic=f"v3/tinyfl-v91-hw/devices/{topic}",
                           payload=json.dumps(event).encode())


def test_on_message_routes_uplinks_to_the_controller():
    runner = make_runner()
    frame = p3.encode_board_hello(4, 3, 1, 3)
    runner._on_message(None, None, message("eui-70b3d57ed0000004/up", uplink_event(frame, f_cnt=42)))
    assert runner.calls == [("up", "eui-70b3d57ed0000004", 10, frame)]
    # the frame counter is only logged; attribution no longer depends on it (rev 3.1)
    assert runner.events[0][0] == "radio_uplink" and runner.events[0][2]["f_cnt"] == 42


def test_on_message_routes_downlink_status_events():
    runner = make_runner()
    runner._on_message(None, None, message("tinyfl-v3-global/down/sent", {}))
    runner._on_message(None, None, message("tinyfl-v3-global/down/failed",
                                           {"downlink_failed": {"error": {"name": "x"}}}))
    assert runner.calls[0] == ("sent", "tinyfl-v3-global")
    assert runner.calls[1][:2] == ("failed", "tinyfl-v3-global")
    assert json.loads(runner.calls[1][2]) == {"error": {"name": "x"}}


def test_a_mac_only_uplink_is_logged_not_dispatched():
    runner = make_runner()
    runner._on_message(None, None, message("eui-0000000000000001/up", uplink_event(None)))
    assert runner.calls == [] and runner.events[0][0] == "mac_only_uplink"


def test_a_bad_event_is_logged_and_does_not_raise():
    runner = make_runner()
    runner._on_message(None, None, SimpleNamespace(
        topic="v3/tinyfl-v91-hw/devices/x/up", payload=b"{not json"))
    runner._on_message(None, None, SimpleNamespace(topic="v3/other/devices/x/up", payload=b"{}"))
    assert [e for e, _, _ in runner.events] == ["event_rejected", "event_rejected"]


def test_a_controller_bug_is_logged_with_its_traceback_not_swallowed():
    runner = make_runner()

    def explode(*args, **kwargs):
        raise RuntimeError("bug")

    runner.controller.handle_uplink = explode
    runner._on_message(None, None, message("eui-0000000000000001/up",
                                           uplink_event(p3.encode_class_c_ready(0, 5, 5))))
    assert runner.events[-1][0] == "controller_exception"
    assert "RuntimeError" in runner.events[-1][2]["traceback"]


def valid_config(tmp_path, **overrides):
    text = (SERVER / "config.toml.example").read_text(encoding="utf-8")
    text = text.replace('multicast_device_id = ""', 'multicast_device_id = "tinyfl-v3-global"')
    text = text.replace("multicast_gateway_ids = []",
                        'multicast_gateway_ids = ["eui-00000000000000a1"]')
    for old, new in overrides.items():
        text = text.replace(old, new)
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_accepts_a_multicast_configuration(tmp_path):
    config = srv.load_config(valid_config(tmp_path))
    assert config["experiment"]["multicast_device_id"] == "tinyfl-v3-global"
    assert config["v3"]["seed"] == 17070


def test_load_config_requires_multicast_and_a_gateway(tmp_path):
    example = SERVER / "config.toml.example"
    path = tmp_path / "config.toml"
    path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="multicast"):
        srv.load_config(path)
    text = example.read_text(encoding="utf-8").replace(
        'multicast_device_id = ""', 'multicast_device_id = "g"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="gateway"):
        srv.load_config(path)


def test_load_config_rejects_bad_levels_and_board_counts(tmp_path):
    with pytest.raises(ValueError):
        srv.load_config(valid_config(tmp_path, **{"levels = [4, 8, 12, 16, 20, 24, 28]": "levels = [8, 4]"}))
    with pytest.raises(ValueError):
        srv.load_config(valid_config(tmp_path, **{"expected_boards = 10": "expected_boards = 11"}))
