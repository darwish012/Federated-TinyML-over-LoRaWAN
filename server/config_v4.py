"""config.toml validation for the v4 server: every problem is reported at once, in plain words.

Sections read:

    [tts]          application_id, host, port, api_key_env, ... (as v3)
    [experiment]   multicast_device_id, multicast_gateway_ids and the downlink timings of v3
    [v4]           the run plan, start rules, membership and timeouts (see config.toml.example)
    [enrollment]   join_eui, master_key_env, window, registry and nonce stores

Nothing here reads or prints a secret; ``check_environment`` only reports whether a
named environment variable is set and well formed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import tomllib
from typing import Any

from controller_v4 import ControllerParams
from plan_v4 import DEFAULT_ROUNDS, MODES, RunPlan, elastic_plan, parse_phase_specs, planned_plan

SERVER_DIR = Path(__file__).resolve().parent
_HEX = re.compile(r"^[0-9A-Fa-f]+$")

V4_KEYS = {
    "mode", "seed", "rounds", "phases", "elastic_phases", "research_root",
    "min_boards", "start_after_seconds", "require_all_positions", "max_machines_per_board",
    "suspect_after_rounds", "depart_after_rounds", "round_timeout_seconds", "ready_wait_seconds",
    "max_global_repair_rounds", "idle_phase_timeout_seconds", "idle_retry_seconds",
    "post_run_collect_seconds", "replay_repeat_seconds", "end_repeats",
    "inflight_downlink_timeout_seconds", "resume_repair_delay_seconds", "run_timeout_minutes",
    "status_line_seconds", "roster_path", "require_roster", "liveness_seconds",
    "downlink_guard_seconds", "end_repeat_seconds", "end_quiet_seconds", "end_linger_seconds",
    "adopt_period_seconds", "prompt_repair_min_gap_seconds",
}
ENROLLMENT_KEYS = {
    "join_eui", "gateway_api_key_env", "device_api_key_env", "master_key_env", "gateway_ids",
    "window_seconds", "registry_path", "nonce_store_path", "bad_mic_log_interval_seconds",
    "cluster_address", "frequency_plan_id", "lorawan_version", "lorawan_phy_version",
    "rx2_data_rate_index", "rx2_frequency", "api_base_url", "app_key_env", "quota",
}
EXPERIMENT_TIMINGS = {
    "class_c_guard_seconds": 15.0, "retry_guard_seconds": 8.0, "max_downlink_attempts": 5,
    "max_repair_rounds": 4, "multicast_report_window_seconds": 180.0,
}


class ConfigError(ValueError):
    """The configuration is invalid; ``errors`` lists every problem."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("invalid configuration:\n  - " + "\n  - ".join(self.errors))


@dataclass
class V4Config:
    raw: dict
    mode: str
    seed: int
    rounds: int
    phases_raw: list
    elastic_phases: int
    research_root: str | None
    params: ControllerParams
    roster_path: Path
    require_roster: bool
    run_timeout_minutes: float
    status_line_seconds: float
    multicast_device_id: str
    multicast_gateway_ids: list = field(default_factory=list)
    has_v4_section: bool = True

    def build_plan(self, machine_order=None) -> RunPlan:
        if self.mode == "elastic":
            return elastic_plan(self.seed, self.elastic_phases, self.rounds)
        return planned_plan(self.phases_raw, self.seed, self.rounds, machine_order)


def _num(section: dict, key: str, default, errors: list, where: str, *, minimum=None,
         maximum=None, integer=False, exclusive_min=False):
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{where}.{key} must be a number, not {type(value).__name__}")
        return default
    if integer and (isinstance(value, float) and not value.is_integer()):
        errors.append(f"{where}.{key} must be an integer")
        return default
    if minimum is not None and (value <= minimum if exclusive_min else value < minimum):
        errors.append(f"{where}.{key} = {value} must be {'>' if exclusive_min else '>='} {minimum}")
    if maximum is not None and value > maximum:
        errors.append(f"{where}.{key} = {value} must be <= {maximum}")
    return int(value) if integer else float(value)


def _bool(section: dict, key: str, default: bool, errors: list, where: str) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        errors.append(f"{where}.{key} must be true or false")
        return default
    return value


def _hex(value, nbytes: int) -> bool:
    text = str(value or "")
    return len(text) == nbytes * 2 and bool(_HEX.match(text))


def parse_config(config: dict[str, Any], base_dir: Path | None = None) -> V4Config:
    """Validate a parsed config.toml and return a V4Config, or raise ConfigError."""
    base_dir = base_dir or SERVER_DIR
    errors: list[str] = []
    tts = config.get("tts")
    experiment = config.get("experiment")
    has_v4 = "v4" in config
    v4 = config.get("v4", {})            # a missing [v4] means "every default" (the thesis sweep)
    enrollment = config.get("enrollment", {})
    if not isinstance(tts, dict):
        errors.append("missing [tts] section")
        tts = {}
    if not isinstance(experiment, dict):
        errors.append("missing [experiment] section")
        experiment = {}
    if not isinstance(v4, dict):
        errors.append("[v4] must be a table (see config.toml.example)")
        v4 = {}

    # -- [tts] ---------------------------------------------------------------
    if not str(tts.get("application_id", "")).strip():
        errors.append("tts.application_id must name the TTS application")
    if tts.get("tenant_id", ""):
        errors.append("tts.tenant_id must be empty for this local Open Source deployment")
    _num(tts, "port", 1883, errors, "tts", minimum=1, maximum=65535, integer=True)
    if not str(tts.get("api_key_env", "TTS_MQTT_API_KEY")).strip():
        errors.append("tts.api_key_env must name an environment variable")

    # -- [experiment] ----------------------------------------------------------
    group = str(experiment.get("multicast_device_id", "")).strip()
    if not group:
        errors.append("experiment.multicast_device_id is empty: v4 delivers GLOBAL by multicast only "
                      "(python configure_multicast.py creates the group's config)")
    gateways = experiment.get("multicast_gateway_ids", [])
    if not isinstance(gateways, list) or not [g for g in gateways if str(g).strip()]:
        errors.append("experiment.multicast_gateway_ids is empty: a multicast group never uplinks, so "
                      "every downlink would fail with no_downlink_path; name the gateway(s)")
        gateways = []
    timings = {}
    for key, default in EXPERIMENT_TIMINGS.items():
        timings[key] = _num(experiment, key, default, errors, "experiment", minimum=0,
                            integer=isinstance(default, int))

    # -- [v4] ----------------------------------------------------------------------
    unknown = sorted(set(v4) - V4_KEYS)
    if unknown:
        errors.append(f"unknown [v4] keys {unknown}; allowed: {sorted(V4_KEYS)}")
    mode = v4.get("mode", "planned")
    if mode not in MODES:
        errors.append(f"v4.mode must be one of {list(MODES)}, not {mode!r}")
        mode = "planned"
    seed = _num(v4, "seed", 17070, errors, "v4", integer=True)
    rounds = _num(v4, "rounds", DEFAULT_ROUNDS, errors, "v4", minimum=1, maximum=255, integer=True)
    elastic_phases = _num(v4, "elastic_phases", 1, errors, "v4", minimum=1, maximum=255, integer=True)
    phases_raw = v4.get("phases", [4, 8, 12, 16, 20, 24, 28])
    if mode == "planned":
        if not isinstance(phases_raw, list) or not phases_raw:
            errors.append("v4.phases must be a non-empty list for mode = \"planned\" "
                          "(e.g. [4, 8, 12, 16, 20, 24, 28], first N positions each)")
            phases_raw = []
        else:
            _, phase_errors = parse_phase_specs(phases_raw, rounds)
            errors.extend(phase_errors)
            if len(phases_raw) > 255:
                errors.append("v4.phases lists more than 255 phases")
    elif "phases" in v4:
        errors.append("v4.phases applies to mode = \"planned\" only; elastic runs use v4.elastic_phases")

    params = ControllerParams()
    params.min_boards = _num(v4, "min_boards", 1, errors, "v4", minimum=1, integer=True)
    params.start_after_seconds = _num(v4, "start_after_seconds", 0.0, errors, "v4", minimum=0)
    params.require_all_positions = _bool(v4, "require_all_positions", False, errors, "v4")
    if params.require_all_positions and mode != "planned":
        errors.append("v4.require_all_positions only applies to mode = \"planned\"")
    params.max_machines_per_board = _num(v4, "max_machines_per_board", 8, errors, "v4",
                                         minimum=1, maximum=8, integer=True)
    params.suspect_after_rounds = _num(v4, "suspect_after_rounds", 2, errors, "v4",
                                       minimum=1, integer=True)
    params.depart_after_rounds = _num(v4, "depart_after_rounds", 4, errors, "v4",
                                      minimum=1, integer=True)
    if params.depart_after_rounds < params.suspect_after_rounds:
        errors.append("v4.depart_after_rounds must be >= v4.suspect_after_rounds")
    for key, default in (("round_timeout_seconds", 900.0), ("ready_wait_seconds", 240.0),
                         ("idle_phase_timeout_seconds", 3600.0), ("idle_retry_seconds", 120.0),
                         ("post_run_collect_seconds", 1800.0), ("replay_repeat_seconds", 300.0),
                         ("inflight_downlink_timeout_seconds", 45.0),
                         ("resume_repair_delay_seconds", 20.0), ("liveness_seconds", 600.0)):
        setattr(params, key, _num(v4, key, default, errors, "v4", minimum=0, exclusive_min=True))
    params.downlink_guard_seconds = _num(v4, "downlink_guard_seconds", 4.0, errors, "v4", minimum=0)
    for key, default in (("end_repeat_seconds", 30.0), ("end_quiet_seconds", 420.0),
                         ("end_linger_seconds", 1800.0)):
        setattr(params, key, _num(v4, key, default, errors, "v4", minimum=0, exclusive_min=True))
    params.prompt_repair_min_gap_seconds = _num(v4, "prompt_repair_min_gap_seconds", 30.0, errors, "v4",
                                                minimum=0)
    params.adopt_period_seconds = _num(v4, "adopt_period_seconds", 120.0, errors, "v4",
                                       minimum=0, exclusive_min=True)
    if params.end_linger_seconds < params.end_quiet_seconds:
        errors.append("v4.end_linger_seconds must be >= v4.end_quiet_seconds "
                      "(the linger limit ends an ENDING stage that never goes quiet)")
    params.max_global_repair_rounds = _num(v4, "max_global_repair_rounds", 4, errors, "v4",
                                           minimum=0, integer=True)
    params.end_repeats = _num(v4, "end_repeats", 2, errors, "v4", minimum=1, maximum=10, integer=True)
    for key, value in timings.items():
        setattr(params, key, value)
    run_timeout = _num(v4, "run_timeout_minutes", 900.0, errors, "v4", minimum=0, exclusive_min=True)
    status_seconds = _num(v4, "status_line_seconds", 30.0, errors, "v4", minimum=0)
    require_roster = _bool(v4, "require_roster", True, errors, "v4")
    roster_path = Path(str(v4.get("roster_path", "roster.json")))
    if not roster_path.is_absolute():
        roster_path = base_dir / roster_path

    # -- [enrollment] ----------------------------------------------------------------
    if not isinstance(enrollment, dict):
        errors.append("[enrollment] must be a table")
        enrollment = {}
    unknown = sorted(set(enrollment) - ENROLLMENT_KEYS)
    if unknown:
        errors.append(f"unknown [enrollment] keys {unknown}; allowed: {sorted(ENROLLMENT_KEYS)}")
    join_eui = str(enrollment.get("join_eui", "0000000000000000"))
    if not _hex(join_eui, 8):
        errors.append("enrollment.join_eui must be exactly 16 hexadecimal digits "
                      "(most significant byte first, as TTS shows it)")
    for key in ("master_key_env", "gateway_api_key_env", "device_api_key_env"):
        if key in enrollment and not str(enrollment[key]).strip():
            errors.append(f"enrollment.{key} must name an environment variable")
    _num(enrollment, "window_seconds", 600, errors, "enrollment", minimum=1, integer=True)
    _num(enrollment, "bad_mic_log_interval_seconds", 60.0, errors, "enrollment", minimum=0)

    if errors:
        raise ConfigError(errors)
    research_root = str(v4.get("research_root", "")).strip() or None
    return V4Config(
        raw=config, has_v4_section=has_v4, mode=mode, seed=seed, rounds=rounds, phases_raw=list(phases_raw),
        elastic_phases=elastic_phases, research_root=research_root, params=params,
        roster_path=roster_path, require_roster=require_roster, run_timeout_minutes=run_timeout,
        status_line_seconds=status_seconds, multicast_device_id=group,
        multicast_gateway_ids=[str(g).strip() for g in gateways if str(g).strip()])


def load_config(path: Path | str) -> V4Config:
    path = Path(path)
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError([f"{path} not found; copy config.toml.example to config.toml first"])
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError([f"{path} is not valid TOML: {exc}"])
    return parse_config(raw, path.resolve().parent)


def check_environment(config: V4Config, env: dict | None = None,
                      need_enrollment: bool = False) -> list[str]:
    """Problems with required environment variables. Reports names, never values."""
    env = os.environ if env is None else env
    enrollment = config.raw.get("enrollment", {})
    problems = []
    mqtt_key = config.raw.get("tts", {}).get("api_key_env", "TTS_MQTT_API_KEY")
    if not env.get(mqtt_key, "").strip():
        problems.append(f"set ${mqtt_key} to the TTS application MQTT API key")
    if need_enrollment:
        for key, default in (("master_key_env", "TINYFL_MASTER_KEY"),
                             ("gateway_api_key_env", "TTS_GATEWAY_API_KEY"),
                             ("device_api_key_env", "TTS_DEVICE_API_KEY")):
            name = enrollment.get(key, default)
            value = env.get(name, "").strip()
            if not value:
                problems.append(f"set ${name} (enrollment)")
            elif key == "master_key_env" and not _hex(value, 16):
                problems.append(f"${name} must be 32 hexadecimal digits (the 16-byte master key)")
    return problems
