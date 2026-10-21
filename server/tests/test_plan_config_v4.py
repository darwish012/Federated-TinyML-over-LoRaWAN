"""Run plans, configuration validation, the shipped config.toml.example and the operator files."""
from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

import config_v4 as cv
import plan_v4 as pv
from study_plan import load_machine_order

SERVER = Path(__file__).resolve().parents[1]
EXAMPLE = SERVER / "config.toml.example"


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
def test_the_thesis_plan_is_seven_phases_of_twelve_rounds_from_machine_order_v2():
    order = load_machine_order()[0]
    plan = pv.thesis_plan(17070, machine_order=order)
    assert plan.mode == "planned" and len(plan.phases) == 7
    assert [len(p.positions) for p in plan.phases] == [4, 8, 12, 16, 20, 24, 28]
    assert all(p.rounds == 12 for p in plan.phases) and [p.index for p in plan.phases] == list(range(7))
    assert plan.phases[0].positions == (0, 1, 2, 3) and plan.phases[6].positions == tuple(range(28))
    assert plan.machine_of(0) == order[0] and plan.machine_of(27) == order[27] and plan.machine_of(99) is None
    assert plan.all_positions() == tuple(range(28))
    assert plan.phase(2).covers(11) and not plan.phase(2).covers(12)


def test_phase_specs_accept_integers_and_tables_and_report_every_error():
    specs, errors = pv.parse_phase_specs([4, {"first": 6, "rounds": 3, "name": "six"},
                                          {"positions": [9, 2, 5]}], default_rounds=12)
    assert errors == [] and [s.rounds for s in specs] == [12, 3, 12]
    assert specs[1].name == "six" and specs[1].positions == (0, 1, 2, 3, 4, 5)
    assert specs[2].positions == (2, 5, 9)
    _, errors = pv.parse_phase_specs([0, True, "x", {"first": 4, "positions": [1]}, {"first": -1}, {"positions": []},
                                      {"positions": [1, 1]}, {"first": 4, "rounds": 0}, {"first": 4, "bogus": 1},
                                      {"positions": [70000]}], 12)
    assert len(errors) >= 9
    assert any("v4.phases[3]" in e and "exactly one" in e for e in errors)
    _, errors = pv.parse_phase_specs([100], 12, machine_count=54)
    assert errors and "no machine" in errors[0]


def test_elastic_plan_has_no_position_list():
    plan = pv.elastic_plan(17070, phases=3, rounds=5)
    assert plan.mode == "elastic" and [p.positions for p in plan.phases] == [None] * 3
    assert plan.all_positions() == () and plan.phase(1).covers(60000) and plan.phase(2).rounds == 5
    with pytest.raises(ValueError):
        pv.elastic_plan(1, phases=0)
    with pytest.raises(ValueError):
        pv.elastic_plan(1, phases=256)


def test_plan_round_trips_through_a_checkpoint_dict():
    for plan in (pv.thesis_plan(1, machine_order=load_machine_order()[0]), pv.elastic_plan(2, 2, 4)):
        assert pv.RunPlan.from_dict(plan.to_dict()) == plan


def test_positions_without_board_per_phase():
    plan = pv.planned_plan([4, 8], 1, 2)
    assert pv.positions_without_board(plan, {0, 1, 2, 3, 4}) == {1: [5, 6, 7]}
    assert pv.positions_without_board(plan, set(range(8))) == {}
    assert pv.positions_without_board(pv.elastic_plan(1), set()) == {}


def test_plan_validation():
    with pytest.raises(ValueError):
        pv.RunPlan("sideways", (pv.PhaseSpec(0, "a", None, 1),), 1)
    with pytest.raises(ValueError):
        pv.RunPlan("planned", (), 1)
    with pytest.raises(ValueError):
        pv.RunPlan("planned", (pv.PhaseSpec(1, "a", (0,), 1),), 1)
    with pytest.raises(ValueError):
        pv.planned_plan([0], 1)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------
def valid_config(**v4):
    return {
        "tts": {"application_id": "tinyfl-v91-hw", "tenant_id": "", "port": 1883},
        "experiment": {"multicast_device_id": "multicast-group-01", "multicast_gateway_ids": ["eui-00000000000000a1"]},
        "v4": dict(v4),
        "enrollment": {"join_eui": "0000000000000000"},
    }


def test_a_minimal_valid_config_gives_the_documented_defaults():
    cfg = cv.parse_config(valid_config(), SERVER)
    assert cfg.mode == "planned" and cfg.seed == 17070 and cfg.rounds == 12
    assert cfg.phases_raw == [4, 8, 12, 16, 20, 24, 28]
    p = cfg.params
    assert (p.min_boards, p.start_after_seconds, p.require_all_positions, p.max_machines_per_board) == (1, 0, False, 8)
    assert (p.suspect_after_rounds, p.depart_after_rounds) == (2, 4)
    assert (p.round_timeout_seconds, p.ready_wait_seconds, p.idle_phase_timeout_seconds) == (900, 240, 3600)
    assert p.liveness_seconds == 600.0
    assert (p.post_run_collect_seconds, p.replay_repeat_seconds, p.end_repeats) == (1800, 300, 2)
    assert (p.class_c_guard_seconds, p.retry_guard_seconds, p.max_downlink_attempts, p.max_repair_rounds,
            p.multicast_report_window_seconds) == (15, 8, 5, 4, 180)
    assert cfg.roster_path == SERVER / "roster.json" and cfg.require_roster is True
    assert cfg.multicast_device_id == "multicast-group-01" and cfg.multicast_gateway_ids == ["eui-00000000000000a1"]
    plan = cfg.build_plan(load_machine_order()[0])
    assert plan.mode == "planned" and len(plan.phases) == 7


def test_elastic_config_builds_an_elastic_plan_and_experiment_timings_are_read():
    config = valid_config(mode="elastic", elastic_phases=2, rounds=4, min_boards=3, start_after_seconds=120,
                          round_timeout_seconds=60, suspect_after_rounds=1, depart_after_rounds=3)
    config["experiment"].update(class_c_guard_seconds=5.0, max_repair_rounds=2)
    cfg = cv.parse_config(config, SERVER)
    plan = cfg.build_plan()
    assert plan.mode == "elastic" and len(plan.phases) == 2 and plan.phases[0].rounds == 4
    assert cfg.params.min_boards == 3 and cfg.params.start_after_seconds == 120
    assert cfg.params.class_c_guard_seconds == 5.0 and cfg.params.max_repair_rounds == 2


def test_every_problem_is_reported_at_once_with_the_key_name():
    config = {"tts": {"application_id": "", "tenant_id": "x", "port": 99999},
              "experiment": {"multicast_device_id": "", "multicast_gateway_ids": []},
              "v4": {"mode": "sideways", "min_boards": 0, "depart_after_rounds": 1, "suspect_after_rounds": 3,
                     "round_timeout_seconds": -1, "end_repeats": 0, "typo_key": 1, "max_machines_per_board": 9,
                     "require_roster": "yes", "seed": "abc", "status_line_seconds": True},
              "enrollment": {"join_eui": "12", "mystery": 1, "window_seconds": 0}}
    with pytest.raises(cv.ConfigError) as err:
        cv.parse_config(config, SERVER)
    text = str(err.value)
    for needle in ("tts.application_id", "tts.tenant_id", "tts.port", "experiment.multicast_device_id",
                   "experiment.multicast_gateway_ids", "v4.mode", "v4.min_boards", "depart_after_rounds",
                   "round_timeout_seconds", "end_repeats", "typo_key", "max_machines_per_board",
                   "require_roster", "v4.seed", "status_line_seconds", "enrollment.join_eui", "mystery",
                   "window_seconds"):
        assert needle in text, needle
    assert len(err.value.errors) >= 17


@pytest.mark.parametrize("v4, needle", [
    ({"phases": []}, "non-empty list"),
    ({"phases": [4, {"first": 0}]}, "v4.phases[1]"),
    ({"mode": "elastic", "phases": [4]}, "elastic"),
    ({"mode": "elastic", "require_all_positions": True}, "planned"),
    ({"rounds": 0}, "v4.rounds"),
    ({"rounds": 256}, "v4.rounds"),
    ({"min_boards": 1.5}, "integer"),
    ({"liveness_seconds": 0}, "v4.liveness_seconds"),
    ({"liveness_seconds": -5}, "v4.liveness_seconds"),
    ({"liveness_seconds": "600"}, "v4.liveness_seconds"),
    ({"adopt_period_seconds": 0}, "v4.adopt_period_seconds"),
    ({"adopt_period_seconds": -1}, "v4.adopt_period_seconds"),
    ({"adopt_period_seconds": "120"}, "v4.adopt_period_seconds"),
    ({"downlink_guard_seconds": -1}, "v4.downlink_guard_seconds"),
    ({"end_repeat_seconds": 0}, "v4.end_repeat_seconds"),
    ({"end_quiet_seconds": 0}, "v4.end_quiet_seconds"),
    ({"end_quiet_seconds": 500, "end_linger_seconds": 400}, "end_linger_seconds"),
])
def test_specific_config_errors(v4, needle):
    with pytest.raises(cv.ConfigError) as err:
        cv.parse_config(valid_config(**v4), SERVER)
    assert needle in str(err.value)


def test_missing_sections_are_named_and_a_missing_v4_means_defaults():
    with pytest.raises(cv.ConfigError) as err:
        cv.parse_config({}, SERVER)
    assert "[tts]" in str(err.value) and "[experiment]" in str(err.value) and "[v4]" not in str(err.value)
    config = valid_config()
    del config["v4"]
    cfg = cv.parse_config(config, SERVER)
    assert cfg.has_v4_section is False and cfg.mode == "planned" and len(cfg.phases_raw) == 7
    with pytest.raises(cv.ConfigError, match=r"\[v4\] must be a table"):
        cv.parse_config(dict(valid_config(), v4=[1]), SERVER)


def test_load_config_reports_a_missing_or_broken_file(tmp_path):
    with pytest.raises(cv.ConfigError, match="not found"):
        cv.load_config(tmp_path / "nope.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml")
    with pytest.raises(cv.ConfigError, match="not valid TOML"):
        cv.load_config(bad)


def test_check_environment_names_variables_and_never_values():
    cfg = cv.parse_config(valid_config(), SERVER)
    assert cv.check_environment(cfg, {}) == ["set $TTS_MQTT_API_KEY to the TTS application MQTT API key"]
    assert cv.check_environment(cfg, {"TTS_MQTT_API_KEY": "x"}) == []
    problems = cv.check_environment(cfg, {"TTS_MQTT_API_KEY": "x", "TINYFL_MASTER_KEY": "not-hex-secret"},
                                    need_enrollment=True)
    text = " ".join(problems)
    assert "TTS_GATEWAY_API_KEY" in text and "TTS_DEVICE_API_KEY" in text and "32 hexadecimal" in text
    assert "not-hex-secret" not in text
    assert cv.check_environment(cfg, {"TTS_MQTT_API_KEY": "x", "TINYFL_MASTER_KEY": "00" * 16,
                                      "TTS_GATEWAY_API_KEY": "g", "TTS_DEVICE_API_KEY": "d"}, need_enrollment=True) == []


# ---------------------------------------------------------------------------
# The shipped example and the operator files
# ---------------------------------------------------------------------------
def test_config_example_documents_every_v4_and_enrollment_key_and_validates_once_edited():
    text = EXAMPLE.read_text()
    raw = tomllib.loads(text)
    assert set(raw["v4"]) == cv.V4_KEYS                                # every key, uncommented, with its default
    assert {"master_key_env", "nonce_store_path", "bad_mic_log_interval_seconds", "join_eui",
            "gateway_api_key_env", "device_api_key_env", "window_seconds", "registry_path"} <= set(raw["enrollment"])
    assert set(raw["enrollment"]) <= cv.ENROLLMENT_KEYS
    # the shipped example is deliberately incomplete (no multicast group / gateway): validation says so ...
    with pytest.raises(cv.ConfigError) as err:
        cv.parse_config(raw, SERVER)
    assert "multicast_device_id" in str(err.value) and "multicast_gateway_ids" in str(err.value)
    # ... and is valid once those two values are filled in
    raw["experiment"]["multicast_device_id"] = "multicast-group-01"
    raw["experiment"]["multicast_gateway_ids"] = ["eui-00000000000000a1"]
    cfg = cv.parse_config(raw, SERVER)
    assert cfg.mode == "planned" and cfg.params.min_boards == 1 and cfg.params.round_timeout_seconds == 900
    # the example's defaults equal the code's defaults
    from controller_v4 import ControllerParams
    default = ControllerParams()
    for key in ("suspect_after_rounds", "depart_after_rounds", "round_timeout_seconds", "ready_wait_seconds",
                "max_global_repair_rounds", "idle_phase_timeout_seconds", "idle_retry_seconds",
                "post_run_collect_seconds", "replay_repeat_seconds", "end_repeats", "resume_repair_delay_seconds",
                "inflight_downlink_timeout_seconds", "min_boards", "max_machines_per_board", "liveness_seconds",
                "downlink_guard_seconds", "end_repeat_seconds", "end_quiet_seconds", "end_linger_seconds",
                "adopt_period_seconds"):
        assert getattr(cfg.params, key) == getattr(default, key), key


def test_the_v3_sections_of_the_example_are_untouched_for_the_v3_server():
    raw = tomllib.loads(EXAMPLE.read_text())
    assert raw["v3"]["levels"] == [4, 8, 12, 16, 20, 24, 28] and raw["v3"]["expected_boards"] == 10
    assert raw["enrollment"]["app_key_env"] == "TINYFL_APP_KEY" and raw["enrollment"]["quota"] == 10


def test_requirements_and_run_script():
    requirements = (SERVER / "requirements.txt").read_text().lower()
    for package in ("paho-mqtt", "numpy", "cryptography", "pyserial", "pytest"):
        assert package in requirements
    script = (SERVER / "run_server_v4.ps1").read_text()
    for needle in ("tts_v4_server.py", "-Resume", "checkpoint.json", "-CheckConfig", "-EnrollMinutes",
                   "TTS_MQTT_API_KEY", "TINYFL_MASTER_KEY", "exit $code", "--resume"):
        assert needle in script, needle


def test_readme_has_the_v4_operator_guide():
    readme = (SERVER.parent / "README.md").read_text()
    assert "## v4 operator guide" in readme
    guide = readme.split("## v4 operator guide", 1)[1]
    for needle in ("provision_board.py", "roster.json", "TINYFL_MASTER_KEY", "planned", "elastic", "--resume",
                   "checkpoint.json", "replay_masks.json", "compare_hw_twin_v4.py", "Limits", "min_boards",
                   "RUN_CONTROL", "ADOPT"):
        assert needle in guide, needle
    for key in cv.V4_KEYS:
        assert key in guide, f"README v4 guide does not mention [v4] {key}"
