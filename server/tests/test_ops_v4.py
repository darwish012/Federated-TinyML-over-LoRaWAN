"""Operability: structured logs, the status line, the MQTT bridge (no network), graceful stop,
--resume and the command line."""
from __future__ import annotations

import base64
import io
import json
import threading
import time
import types

import pytest

import config_v4 as cv
import logging_v4 as lg
import protocol_v4 as p4
import tts_v4_server as srv
from roster_v4 import Roster
from v4_sim import HELLO_OK, device_id_for

APP = "tinyfl-v91-hw"
GROUP = "multicast-group-01"
GATEWAY = "eui-00000000000000a1"


def config_dict(tmp_path, **v4):
    values = {"mode": "elastic", "elastic_phases": 1, "rounds": 3, "min_boards": 1, "require_roster": False,
              "roster_path": str(tmp_path / "roster.json"), "status_line_seconds": 0}
    values.update(v4)
    return {"tts": {"application_id": APP, "tenant_id": "", "port": 1883},
            "experiment": {"multicast_device_id": GROUP, "multicast_gateway_ids": [GATEWAY],
                           "class_c_guard_seconds": 0.0},
            "v4": values, "enrollment": {"join_eui": "0000000000000000"}}


class FakeClient:
    """Just enough of paho's Client for the bridge; nothing connects anywhere."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def username_pw_set(self, *a):
        pass

    def tls_set(self, **kw):
        pass

    def tls_insecure_set(self, flag):
        pass

    def connect(self, *a):
        pass

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def subscribe(self, *a, **kw):
        pass


FAKE_MQTT = types.SimpleNamespace(Client=FakeClient, CallbackAPIVersion=types.SimpleNamespace(VERSION2=2),
                                  MQTTv311=4, MQTT_ERR_SUCCESS=0)
_REAL_SERVER = srv.TtsMqttV4Server


def make_server(cfg, directory, **kwargs):
    return _REAL_SERVER(cfg, directory, mqtt_module=FAKE_MQTT, **kwargs)


@pytest.fixture(autouse=True)
def _no_real_mqtt(monkeypatch):
    """main() builds its own server: give it the fake MQTT module as well."""
    monkeypatch.setattr(srv, "TtsMqttV4Server", lambda cfg, out, **kw: _REAL_SERVER(
        cfg, out, mqtt_module=FAKE_MQTT, **kw))


class Published:
    def __init__(self):
        self.messages = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.messages.append((topic, json.loads(payload)))
        return types.SimpleNamespace(rc=0)


@pytest.fixture
def server(tmp_path):
    cfg = cv.parse_config(config_dict(tmp_path), tmp_path)
    s = make_server(cfg, tmp_path / "run", env={"TTS_MQTT_API_KEY": "test-key"})
    s.sent = Published()
    s.client.publish = s.sent.publish
    s.client.connect = lambda *a: None
    s.client.loop_start = lambda: None
    s.client.loop_stop = lambda: None
    s.client.disconnect = lambda: None
    s.connected.set()
    yield s
    try:
        s.outputs.close()
        s.log.close()
    except Exception:  # noqa: BLE001
        pass


def up_message(board: int, frame: bytes):
    event = {"uplink_message": {"f_port": 10, "frm_payload": base64.b64encode(frame).decode(), "f_cnt": 1,
                                "settings": {"data_rate": {"lora": {"spreading_factor": 7, "bandwidth": 125000}}},
                                "rx_metadata": [{"rssi": -80, "snr": 5.0}]}}
    return types.SimpleNamespace(topic=f"v3/{APP}/devices/{device_id_for(board)}/up",
                                 payload=json.dumps(event).encode())


def hello_ready(server, board=1, position=0):
    server._on_message(None, None, up_message(board, p4.encode_board_hello(board, 1, HELLO_OK, 0, (position,))))
    server._on_message(None, None, up_message(board, p4.encode_class_c_ready(0, p4.NO_SLOT)))
    # rev 4.3: nothing is multicast within downlink_guard_seconds of a board's uplink, so let the
    # board's RX windows pass (the controller runs on the real clock here)
    for member in server.controller.membership.members.values():
        member.last_heard_t -= 30.0
        member.ready_since -= 30.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def test_json_event_log_writes_one_flushed_object_per_line(tmp_path):
    stream = io.StringIO()
    log = lg.JsonEventLog(tmp_path / "events.jsonl", stream=stream)
    log("round_complete", "server", {"round": 3, "arrived": {1, 2}})
    log("frame_rejected", "eui-1", {"error": "bad magic"})
    log("update_fragment", "eui-1", {"position": 4})                       # INFO, not a milestone: file only
    rows = [json.loads(s) for s in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["round_complete", "frame_rejected", "update_fragment"]
    assert [r["level"] for r in rows] == ["INFO", "WARN", "INFO"]
    assert set(rows[0]) == {"timestamp_utc", "level", "event", "device_id", "details"} and rows[0]["details"]["round"] == 3
    console = stream.getvalue().splitlines()
    assert len(console) == 2 and "round_complete" in console[0] and "WARN" in console[1]
    assert not any("update_fragment" in line for line in console)
    log.close()
    verbose = io.StringIO()
    lg.JsonEventLog(None, verbose=True, stream=verbose)("update_fragment", "x", {"position": 1})
    assert "update_fragment" in verbose.getvalue()


def test_the_log_redacts_anything_that_looks_like_key_material(tmp_path):
    log = lg.JsonEventLog(tmp_path / "e.jsonl", console=False)
    log("x", "d", {"app_key": "a" * 32, "nested": {"api_key": "b" * 32, "list": [{"nwkskey": "c"}]},
                   "master_key": "d" * 32, "token": "t", "api_key_env": "TTS_MQTT_API_KEY", "fine": 1})
    text = (tmp_path / "e.jsonl").read_text()
    assert ("a" * 32) not in text and ("b" * 32) not in text and ("d" * 32) not in text
    row = json.loads(text)["details"]
    assert row["app_key"] == "<redacted>" and row["nested"]["api_key"] == "<redacted>"
    assert row["nested"]["list"][0]["nwkskey"] == "<redacted>" and row["token"] == "<redacted>"
    assert row["api_key_env"] == "TTS_MQTT_API_KEY" and row["fine"] == 1          # env var NAMES are not secrets


def test_status_line_is_concise():
    line = lg.format_status({"stage": "upload", "phase": 2, "phases": 7, "round": 5, "rounds": 12, "active": 9,
                             "suspect": 1, "departed": 0, "bound": 2, "rostered": 12, "arrived_positions": 7,
                             "expected_positions": 12})
    assert line == ("[upload] phase 3/7 round 5/12 | boards active=9 suspect=1 departed=0 bound=2 rostered=12 "
                    "| arrived 7/12")
    waiting = lg.format_status({"stage": "waiting_start", "phase": None, "phases": 7, "round": None, "rounds": None,
                               "active": 0, "suspect": 0, "departed": 0, "bound": 1, "rostered": 3,
                               "arrived_positions": 0, "expected_positions": 0})
    assert waiting.startswith("[waiting_start] - | boards")


# ---------------------------------------------------------------------------
# The MQTT bridge
# ---------------------------------------------------------------------------
def test_uplink_events_reach_the_controller_and_downlinks_use_the_ttn_push_format(server):
    hello_ready(server)
    c = server.controller
    assert c.membership.get("70B3D57ED0000001").state.value == "BOUND"
    c.tick()                                                  # one downlink in flight at a time
    assert len(server.sent.messages) == 1
    corr = server.sent.messages[0][1]["downlinks"][0]["correlation_ids"]
    server._on_message(None, None, types.SimpleNamespace(      # TTS confirms it was sent (naming it) ...
        topic=f"v3/{APP}/devices/{GROUP}/down/sent",
        payload=json.dumps({"downlink_sent": {"correlation_ids": corr}}).encode()))
    c.tick()                                                  # ... and the first GLOBAL fragment follows
    assert [m[0] for m in server.sent.messages] == [f"v3/{APP}/devices/{GROUP}/down/push"] * 2
    first, second = (m[1]["downlinks"][0] for m in server.sent.messages)
    for body in (first, second):
        assert body["f_port"] == 10 and body["confirmed"] is False and body["priority"] == "NORMAL"
        assert len(body["correlation_ids"]) == 1
        # the multicast group never uplinks: the gateway is named explicitly
        assert body["class_b_c"] == {"gateways": [{"gateway_ids": {"gateway_id": GATEWAY}, "group_index": 0}]}
    for body, index in ((first, 0), (second, 1)):             # no board holds another run: no ADOPT first
        fragment = p4.decode_frame(base64.b64decode(body["frm_payload"]))
        assert fragment.message_type == p4.MSG_GLOBAL_FRAGMENT and fragment.round_id == 1
        assert fragment.fragment_index == index

def test_a_unicast_repair_downlink_names_the_board_and_has_no_gateway_block(server):
    hello_ready(server)
    c = server.controller
    server._publish_downlink(device_id_for(1), p4.encode_update_repair(5, c.run_id, 0, 0b101), "corr-1")
    topic, body = server.sent.messages[-1]
    assert topic == f"v3/{APP}/devices/{device_id_for(1)}/down/push"
    assert "class_b_c" not in body["downlinks"][0] and body["downlinks"][0]["correlation_ids"] == ["corr-1"]


def test_malformed_or_unknown_events_are_logged_not_fatal(server):
    c = server.controller
    server._on_message(None, None, types.SimpleNamespace(topic="other/root/devices/x/up", payload=b"{}"))
    server._on_message(None, None, types.SimpleNamespace(topic=f"v3/{APP}/devices/x/up", payload=b"not json"))
    server._on_message(None, None, types.SimpleNamespace(topic=f"v3/{APP}/devices/x/up", payload=b'{"nope": 1}'))
    server._on_message(None, None, types.SimpleNamespace(topic=f"v3/{APP}/devices/x/join", payload=b"{}"))
    mac_only = {"uplink_message": {"f_port": 0, "settings": {}}}
    server._on_message(None, None, types.SimpleNamespace(
        topic=f"v3/{APP}/devices/{device_id_for(1)}/up", payload=json.dumps(mac_only).encode()))
    server._on_message(None, None, up_message(1, b"\x00\x01\x02"))
    log = [json.loads(s) for s in (server.output_dir / "events.jsonl").read_text().splitlines()]
    assert [r["event"] for r in log].count("event_rejected") == 3
    assert c.counters["frames_rejected"] == 1 and not c.failed


def test_down_failed_with_a_deterministic_reason_stops_the_run(server):
    hello_ready(server)
    for _ in range(3):
        server.controller.tick()
    corr = server.sent.messages[0][1]["downlinks"][0]["correlation_ids"]
    reason = {"downlink_failed": {"downlink": {"correlation_ids": corr},
                                  "error": {"name": "no_downlink_path", "attributes": {}}}}
    server._on_message(None, None, types.SimpleNamespace(
        topic=f"v3/{APP}/devices/{GROUP}/down/failed", payload=json.dumps(reason).encode()))
    assert server.controller.failed and "no_downlink_path" in server.controller.failure_reason


# ---------------------------------------------------------------------------
# Console commands
# ---------------------------------------------------------------------------
def test_console_commands(server, capsys):
    hello_ready(server)
    Roster(server.cfg.roster_path).add("70B3D57ED00000FF")
    server.roster.reload()
    server._sync_roster()
    server._command(["status"])
    status = json.loads(capsys.readouterr().out)
    assert status["stage"] in ("downlink", "waiting_start") and status["bound"] == 1 and status["rostered"] == 1
    assert status["roster_devices"] == 1 and status["mqtt_connected"] is True and "counters" in status
    server._command(["roster"])
    rows = json.loads(capsys.readouterr().out)
    assert rows["70B3D57ED0000001"]["state"] == "BOUND" and rows["70B3D57ED00000FF"]["state"] == "ROSTERED"
    server._command(["checkpoint"])
    assert "checkpoint written" in capsys.readouterr().out and (server.output_dir / "checkpoint.json").exists()
    server._command(["frobnicate"])
    assert "commands:" in capsys.readouterr().out
    assert not server.stop_requested.is_set()
    server._command(["stop"])
    assert server.stop_requested.is_set()


def test_enroll_commands_need_the_enrollment_environment(server, capsys):
    server._command(["enroll-status"])                       # nothing opened yet: falls through to help
    assert "commands:" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="enrollment needs"):
        server.open_enrollment(1, "console")                 # the keys are not in the (empty) environment


# ---------------------------------------------------------------------------
# Graceful stop and --resume
# ---------------------------------------------------------------------------
def run_in_thread(server, until):
    result = {}
    thread = threading.Thread(target=lambda: result.update(code=server.run(None, interactive=False)), daemon=True)
    thread.start()
    deadline = time.time() + 20
    while time.time() < deadline and not until():
        time.sleep(0.05)
    server.request_stop()
    thread.join(timeout=20)
    assert not thread.is_alive()
    return result["code"]


def test_ctrl_c_checkpoints_and_exits_130_and_resume_continues_the_same_run(server, tmp_path):
    hello_ready(server)
    run_id = server.controller.run_id
    code = run_in_thread(server, lambda: server.controller.stage.value == "downlink" and len(server.sent.messages) >= 1)
    assert code == srv.EXIT_STOPPED == 130
    run_dir = server.output_dir
    ck = json.loads((run_dir / "checkpoint.json").read_text())
    assert ck["run_id"] == run_id and ck["stage"] == "downlink" and ck["plan"]["mode"] == "elastic"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["exit"] == "stopped_resumable" and summary["failed"] is False
    events = [json.loads(s)["event"] for s in (run_dir / "events.jsonl").read_text().splitlines()]
    assert "run_stopped_by_operator" in events and "run_summary" in events

    cfg = cv.parse_config(config_dict(tmp_path), tmp_path)
    second = make_server(cfg, run_dir, resume=True, env={"TTS_MQTT_API_KEY": "k"})
    try:
        assert second.controller.run_id == run_id and second.controller.stage.value == "downlink"
        assert second.controller.membership.get("70B3D57ED0000001").state.value == "BOUND"
        assert second.controller.slot_counter == ck["slot_counter"]
        events = [json.loads(s)["event"] for s in (run_dir / "events.jsonl").read_text().splitlines()]
        assert "resume_loaded" in events
    finally:
        second.outputs.close()
        second.log.close()


def test_resume_without_a_checkpoint_is_a_startup_error(tmp_path):
    cfg = cv.parse_config(config_dict(tmp_path), tmp_path)
    with pytest.raises(RuntimeError, match="no checkpoint.json"):
        make_server(cfg, tmp_path / "empty", resume=True, env={"TTS_MQTT_API_KEY": "k"})


def test_the_mqtt_key_is_required(tmp_path):
    cfg = cv.parse_config(config_dict(tmp_path), tmp_path)
    with pytest.raises(RuntimeError, match="TTS_MQTT_API_KEY"):
        make_server(cfg, tmp_path / "run", env={})


def test_a_missing_roster_is_refused_when_required(tmp_path):
    cfg = cv.parse_config(config_dict(tmp_path, require_roster=True), tmp_path)
    with pytest.raises(RuntimeError, match="provision at least one board"):
        make_server(cfg, tmp_path / "run", env={"TTS_MQTT_API_KEY": "k"})


def test_a_rostered_run_only_binds_rostered_boards(tmp_path):
    roster = Roster(tmp_path / "roster.json")
    roster.add("70B3D57ED0000001")
    cfg = cv.parse_config(config_dict(tmp_path, require_roster=True), tmp_path)
    s = make_server(cfg, tmp_path / "run", env={"TTS_MQTT_API_KEY": "k"})
    try:
        s._on_message(None, None, up_message(1, p4.encode_board_hello(1, 1, HELLO_OK, 0, (0,))))
        s._on_message(None, None, up_message(2, p4.encode_board_hello(2, 1, HELLO_OK, 0, (1,))))
        members = {m.dev_eui: m.state.value for m in s.controller.membership.members.values()}
        assert members == {"70B3D57ED0000001": "BOUND"}
        assert s.controller.counters["hello_rejected_not_rostered"] == 1
    finally:
        s.outputs.close()
        s.log.close()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def toml_text(d: dict) -> str:
    lines = []
    for section, values in d.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {json.dumps(value)}")
    return "\n".join(lines) + "\n"


@pytest.fixture
def cfg_file(tmp_path):
    path = tmp_path / "config.toml"
    cfg = config_dict(tmp_path)
    cfg["v4"]["roster_path"] = "roster.json"
    path.write_text(toml_text(cfg))
    return path


def test_main_check_config(cfg_file, monkeypatch, capsys):
    monkeypatch.setenv("TTS_MQTT_API_KEY", "x")
    assert srv.main(["--config", str(cfg_file), "--check-config"]) == 0
    out = capsys.readouterr().out
    assert "Configuration is valid" in out and "mode elastic" in out and "roster:" in out and "MISSING" in out
    monkeypatch.delenv("TTS_MQTT_API_KEY")
    assert srv.main(["--config", str(cfg_file), "--check-config"]) == 0
    out = capsys.readouterr().out
    assert "environment: set $TTS_MQTT_API_KEY" in out and "environment incomplete" in out


def test_main_check_config_planned_lists_the_thesis_phases(cfg_file, monkeypatch, capsys):
    text = cfg_file.read_text().replace('mode = "elastic"', 'mode = "planned"')
    cfg_file.write_text(text)
    monkeypatch.setenv("TTS_MQTT_API_KEY", "x")
    assert srv.main(["--config", str(cfg_file), "--check-config", "--phases", "4", "8"]) == 0
    out = capsys.readouterr().out
    assert "machine order from: machine_order_v2" in out and "2 phase(s)" in out and "first4x3" in out


def test_main_reports_config_errors_and_missing_files(tmp_path, cfg_file, monkeypatch, capsys):
    assert srv.main(["--config", str(tmp_path / "nope.toml")]) == srv.EXIT_CONFIG
    assert "not found" in capsys.readouterr().err
    bad = tmp_path / "bad.toml"
    bad.write_text('[tts]\napplication_id = ""\n[experiment]\n[v4]\nmode = "x"\n')
    assert srv.main(["--config", str(bad)]) == srv.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "tts.application_id" in err and "v4.mode" in err and "multicast_device_id" in err
    monkeypatch.delenv("TTS_MQTT_API_KEY", raising=False)
    assert srv.main(["--config", str(cfg_file)]) == srv.EXIT_CONFIG
    assert "TTS_MQTT_API_KEY" in capsys.readouterr().err


def test_main_refuses_to_start_a_new_run_over_an_existing_one_and_resume_needs_a_checkpoint(tmp_path, cfg_file,
                                                                                            monkeypatch, capsys):
    monkeypatch.setenv("TTS_MQTT_API_KEY", "x")
    existing = tmp_path / "old"
    existing.mkdir()
    (existing / "checkpoint.json").write_text("{}")
    assert srv.main(["--config", str(cfg_file), "--output", str(existing)]) == srv.EXIT_CONFIG
    assert "--resume" in capsys.readouterr().err
    assert srv.main(["--config", str(cfg_file), "--resume", str(tmp_path / "nothing")]) == srv.EXIT_CONFIG
    assert "checkpoint.json" in capsys.readouterr().err


def test_a_config_without_a_v4_section_runs_with_defaults_and_says_so(tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.toml"
    cfg = config_dict(tmp_path)
    del cfg["v4"]
    path.write_text(toml_text(cfg))
    monkeypatch.setenv("TTS_MQTT_API_KEY", "x")
    assert srv.main(["--config", str(path), "--check-config"]) == 0
    out = capsys.readouterr().out
    assert "no [v4] section" in out and "7 phase(s)" in out


def test_overrides_are_applied_before_validation(cfg_file):
    import tomllib
    raw = tomllib.loads(cfg_file.read_text())
    args = types.SimpleNamespace(mode="planned", min_boards=7, phases=[4, 8])
    cfg = cv.parse_config(srv.apply_overrides(raw, args), cfg_file.parent)
    assert cfg.mode == "planned" and cfg.params.min_boards == 7 and cfg.phases_raw == [4, 8]
