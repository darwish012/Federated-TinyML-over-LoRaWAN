"""V43 phase driver: audit, cohort, grid, configurations, stats, comms.

V43 tests a threshold grid (no post-hoc winner) and three deployment
configurations (F-G/F-L/L-L) against V41R1's frozen streams, models and
scores, with the machine-life experiment (not the bearing stream) as the
primary statistical unit (V43_PREREGISTRATION.md). There is NO retraining and
NO new source data: every numeric result traces back to V41R1's already
hash-verified streams/models (reused transitively through
``raw_online_federated_v42.verify_frozen_v41r1_evidence``) and to V42
frozen2's own locked results (reused for gate G1's reproduction check).

Phases, in order (V43_PREREGISTRATION.md SS6): audit -> cohort -> grid ->
configurations -> stats -> comms -> (score, in the separate script
``raw_online_federated_v43_score.py``, mirroring V32/V42's driver/score
split).

The PREDICTION LOCK is ``v43_prediction_lock.json``, written at the end of
the ``configurations`` phase. Every phase before it (audit, cohort, grid,
configurations) must never read onset or device status -- only ``stats``
(the first post-lock phase) and ``score`` may. This mirrors V42's onset
boundary, just drawn one phase later because V43 explicitly separates
"predict" (grid + configurations) from "compute statistics that need onset"
(stats) in its own phase list.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

import raw_online_federated_v41 as v41base
import raw_online_federated_v42 as v42
import raw_online_federated_v42_calibration as calib
import raw_online_federated_v42_score as v42_score
import raw_online_federated_v43_cohort as cohort
import raw_online_federated_v43_comms as comms_mod
import raw_online_federated_v43_grid as grid
import raw_online_federated_v43_stats as stats_mod

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "raw_online_federated_v43_protocol.json"
V41R1_ROOT = v42.V41R1_ROOT
V42_FROZEN2_ROOT = RESULTS / "raw_online_federated_v42_frozen2"
EXPERIMENT = "V43"

sha256 = v42.sha256
read_json = v42.read_json
write_json = v42.write_json
write_csv = v42.write_csv


class IntegrityError(RuntimeError):
    """Raised for a frozen-dependency hash mismatch or an access-boundary
    violation (hard constraint, exit 3)."""


LAUNCHER_ARTIFACTS = frozenset({"v43_console_transcript.txt"})


def root_for(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("RunId must match [A-Za-z0-9_-]+")
    root = (RESULTS / f"raw_online_federated_v43_{run_id}").resolve()
    if RESULTS.resolve() not in root.parents:
        raise ValueError("V43 results must stay under results/")
    return root


def specification() -> dict:
    return read_json(PROTOCOL)


def dependency_paths() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL,
        "cohort_module": HERE / "raw_online_federated_v43_cohort.py",
        "grid_module": HERE / "raw_online_federated_v43_grid.py",
        "stats_module": HERE / "raw_online_federated_v43_stats.py",
        "comms_module": HERE / "raw_online_federated_v43_comms.py",
        "driver": Path(__file__).resolve(),
        "score_module": HERE / "raw_online_federated_v43_score.py",
        "cohort_tests": HERE / "raw_online_federated_v43_cohort_tests.py",
        "grid_tests": HERE / "raw_online_federated_v43_grid_tests.py",
        "stats_tests": HERE / "raw_online_federated_v43_stats_tests.py",
        "comms_tests": HERE / "raw_online_federated_v43_comms_tests.py",
        "tests": HERE / "raw_online_federated_v43_tests.py",
        "launcher": HERE / "start_v43_guarded.ps1",
        "design_doc": HERE / "V43_DESIGN_AND_RUN.md",
    }


# --------------------------------------------------------------------------
# Frozen-evidence verification: reused, not reimplemented. V42's own
# verifier already checks the entire V41R1 hash chain against V42's
# protocol; V43 only needs to additionally pin that V42's OWN files (and its
# frozen2 result artifacts, for gate G1) have not changed.
# --------------------------------------------------------------------------
def verify_frozen_v42_module_evidence() -> dict:
    spec = specification()
    expected = spec["reused_v42_module_sha256"]
    for name, digest in expected.items():
        path = HERE / name
        if not path.is_file() or sha256(path) != digest:
            raise IntegrityError(f"V43 reused V42 module changed since V43 was authored: {name}")
    return v42.verify_frozen_v41r1_evidence()


def verify_frozen_v42_frozen2_evidence() -> dict:
    spec = specification()
    expected = spec["reused_v42_frozen2_sha256"]
    paths = {name: V42_FROZEN2_ROOT / name for name in expected}
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != expected[name]:
            raise IntegrityError(f"V43 pinned V42 frozen2 artifact changed or missing: {name}")
    audit_lock = read_json(paths["v42_audit_lock.json"])
    if audit_lock.get("target_sensor_values_opened") is not False:
        raise IntegrityError("V42 frozen2 audit lock shows target sensor values were opened")
    return {"paths": paths}


# --------------------------------------------------------------------------
# Phase 0: audit
# --------------------------------------------------------------------------
def audit(root: Path) -> None:
    if root.exists():
        preexisting = sorted(path.name for path in root.iterdir() if path.name not in LAUNCHER_ARTIFACTS)
        if preexisting:
            raise IntegrityError(f"V43 RunId already contains artifacts: {root} ({preexisting})")
    specification()
    verify_frozen_v42_module_evidence()
    verify_frozen_v42_frozen2_evidence()
    deps = dependency_paths()
    missing = [str(path) for path in deps.values() if not path.is_file()]
    if missing:
        raise IntegrityError(f"missing V43 protected dependency: {missing}")

    # Structural check carried from V42 (memory.txt lesson): assert
    # guard_length >= persistence_window and fail loudly rather than let a
    # structurally-forced 0% guard rate pass unremarked. Geometry is
    # unchanged from V41R1/V42 (guard=4, persistence window=7).
    guard_check = calib.guard_latch_is_structurally_vacuous(calib.GUARD_SIZE)
    if guard_check["latch_based_guard_metrics_vacuous"]:
        print(
            "V43 WARNING: latch-based guard metrics are STRUCTURALLY VACUOUS at this geometry "
            f"(guard_length={guard_check['guard_length']}, persistence_window={guard_check['persistence_window']}, "
            f"complete_windows={guard_check['complete_windows']}) -- inherited unchanged from V41R1/V42; "
            "see V42_PREREGISTRATION.md SS5 'G2 erratum'.",
            flush=True,
        )

    write_json(
        root / "v43_audit_lock.json",
        {
            "experiment": EXPERIMENT,
            "immutable": True,
            "protocol_sha256": sha256(PROTOCOL),
            "dependency_sha256": {name: sha256(path) for name, path in deps.items()},
            "no_retraining": True,
            "source_clients": 40,
            "one_bearing_one_client": True,
            "target_streams_opened": False,
            "target_sensor_values_opened": False,
            "target_outcomes_opened": False,
            "onset_opened": False,
            "guard_structural_check": guard_check,
        },
    )
    print(f"V43 AUDIT ok: {root / 'v43_audit_lock.json'}", flush=True)


def require_audit(root: Path) -> dict:
    path = root / "v43_audit_lock.json"
    if not path.is_file():
        raise IntegrityError("V43 audit phase must run first")
    lock = read_json(path)
    if lock.get("protocol_sha256") != sha256(PROTOCOL):
        raise IntegrityError("V43 protocol changed after audit")
    for name, path2 in dependency_paths().items():
        if lock["dependency_sha256"].get(name) != sha256(path2):
            raise IntegrityError(f"V43 frozen dependency changed after audit: {name}")
    if lock.get("target_streams_opened") is not False or lock.get("onset_opened") is not False:
        raise IntegrityError("invalid V43 audit access state")
    verify_frozen_v42_module_evidence()
    verify_frozen_v42_frozen2_evidence()
    return lock


def source_roster() -> list[tuple[str, str, str]]:
    return v42.source_roster()


def load_device_stream(dataset: str, unit: str) -> tuple[np.ndarray, np.ndarray]:
    return v42.load_device_stream(dataset, unit)


def load_fold_bundle(dataset: str) -> dict:
    return v42.load_fold_bundle(dataset)


# --------------------------------------------------------------------------
# Phase 1: cohort (SS1.3 mandatory cohort table). No onset, no status, no
# score, no model -- structural bookkeeping only.
# --------------------------------------------------------------------------
def require_cohort_lock(root: Path) -> dict:
    require_audit(root)
    path = root / "v43_cohort_lock.json"
    if not path.is_file():
        raise IntegrityError("V43 cohort phase must run first")
    lock = read_json(path)
    if lock.get("protocol_sha256") != sha256(PROTOCOL):
        raise IntegrityError("V43 protocol changed after cohort")
    if lock.get("table_sha256") != sha256(root / "v43_cohort_table.csv"):
        raise IntegrityError("V43 cohort table changed after cohort lock")
    if lock.get("onset_opened") is not False:
        raise IntegrityError("invalid V43 cohort access state")
    return lock


def cohort_phase(root: Path) -> None:
    require_audit(root)
    roster_client_ids = [client_id for _, _, client_id in source_roster()]
    rows = cohort.cohort_table(roster_client_ids)
    totals = cohort.cohort_totals(rows)
    write_csv(root / "v43_cohort_table.csv", rows)
    write_json(root / "v43_cohort_table.json", {"rows": rows, "totals": totals})
    write_json(
        root / "v43_cohort_lock.json",
        {
            "protocol_sha256": sha256(PROTOCOL),
            "audit_lock_sha256": sha256(root / "v43_audit_lock.json"),
            "table_sha256": sha256(root / "v43_cohort_table.csv"),
            "totals": totals,
            "onset_opened": False,
        },
    )
    print(
        f"V43 COHORT: historical {totals['cohort_size_statement']}; "
        f"reused source roster {totals.get('streams_in_supplied_roster')} streams / "
        f"{totals.get('experiments_in_supplied_roster')} experiments",
        flush=True,
    )


# --------------------------------------------------------------------------
# Phase 2: grid (SS2.1 threshold grid + SS2.2 F-G/F-L/L-L + SS2.3 m
# sensitivity, all in one pass per device so each device's raw branch and
# local-PCA scores are computed exactly once -- V25 caching lesson).
# --------------------------------------------------------------------------
def require_grid_lock(root: Path) -> dict:
    require_cohort_lock(root)
    path = root / "v43_grid_lock.json"
    if not path.is_file():
        raise IntegrityError("V43 grid phase must run first")
    lock = read_json(path)
    if lock.get("protocol_sha256") != sha256(PROTOCOL):
        raise IntegrityError("V43 protocol changed after grid")
    if lock.get("predictions_sha256") != sha256(root / "v43_grid_predictions.csv"):
        raise IntegrityError("V43 grid predictions changed after grid lock")
    if lock.get("onset_opened") is not False:
        raise IntegrityError("invalid V43 grid access state")
    return lock


def grid_phase(root: Path) -> None:
    require_cohort_lock(root)
    spec = specification()
    pol = spec["frozen_v29b_policy"]
    weights = pol["weights"]
    strong = float(pol["strong"])
    vote_fraction = float(pol["vote_fraction"])
    epc_seed = int(spec["epc"]["seed"])

    rows: list[dict] = []
    bundle_cache: dict[str, dict] = {}
    for dataset, unit, client_id in source_roster():
        values, indices = load_device_stream(dataset, unit)
        if dataset not in bundle_cache:
            bundle_cache[dataset] = load_fold_bundle(dataset)
        bundle = bundle_cache[dataset]
        device_rows = grid.compute_device_predictions(
            dataset, unit, client_id, values, indices, bundle, weights, strong, vote_fraction, epc_seed
        )
        rows.extend(device_rows)
        print(f"V43 GRID {client_id}: {len(device_rows)} operating points", flush=True)

    write_csv(root / "v43_grid_predictions.csv", rows)
    write_json(
        root / "v43_grid_lock.json",
        {
            "protocol_sha256": sha256(PROTOCOL),
            "cohort_lock_sha256": sha256(root / "v43_cohort_lock.json"),
            "predictions_sha256": sha256(root / "v43_grid_predictions.csv"),
            "operating_points": len(rows),
            "onset_opened": False,
            "target_streams_opened": False,
        },
    )
    print(f"V43 GRID done: {len(rows)} operating points across 40 devices", flush=True)


# --------------------------------------------------------------------------
# Phase 3: configurations. Splits the grid predictions into the deployment
# grid (F-G/F-L/L-L) and the m-sensitivity leg, checks the m-sensitivity
# cohort really is fixed across sizes, and writes the PREDICTION LOCK --
# the last point before onset may be opened (SS5 G5).
# --------------------------------------------------------------------------
def require_prediction_lock(root: Path) -> dict:
    require_grid_lock(root)
    path = root / "v43_prediction_lock.json"
    if not path.is_file():
        raise IntegrityError("V43 configurations phase must run first")
    lock = read_json(path)
    if lock.get("grid_lock_sha256") != sha256(root / "v43_grid_lock.json"):
        raise IntegrityError("V43 grid lock changed after prediction lock")
    if lock.get("deployment_grid_sha256") != sha256(root / "v43_deployment_grid.csv"):
        raise IntegrityError("V43 deployment grid changed after prediction lock")
    if lock.get("m_sensitivity_sha256") != sha256(root / "v43_m_sensitivity.csv"):
        raise IntegrityError("V43 m-sensitivity table changed after prediction lock")
    return lock


def configurations_phase(root: Path) -> None:
    require_grid_lock(root)
    predictions = pd.read_csv(root / "v43_grid_predictions.csv")
    deployment = predictions[predictions["config"].isin(list(grid.CONFIGS))].reset_index(drop=True)
    sensitivity = predictions[predictions["config"] == grid.SENSITIVITY_CONFIG].reset_index(drop=True)
    if len(deployment) + len(sensitivity) != len(predictions):
        raise IntegrityError("V43 grid predictions contain an unrecognised config label")

    by_m = {m: set(sensitivity[sensitivity["calibration_points"] == m]["client_id"]) for m in grid.SENSITIVITY_M_GRID}
    if not (by_m[64] == by_m[128] == by_m[256]):
        raise IntegrityError("V43 m-sensitivity cohort is not fixed across calibration sizes (SS2.3)")

    write_csv(root / "v43_deployment_grid.csv", deployment.to_dict("records"))
    write_csv(root / "v43_m_sensitivity.csv", sensitivity.to_dict("records"))
    write_json(
        root / "v43_prediction_lock.json",
        {
            "protocol_sha256": sha256(PROTOCOL),
            "grid_lock_sha256": sha256(root / "v43_grid_lock.json"),
            "deployment_grid_sha256": sha256(root / "v43_deployment_grid.csv"),
            "m_sensitivity_sha256": sha256(root / "v43_m_sensitivity.csv"),
            "deployment_operating_points": int(len(deployment)),
            "m_sensitivity_operating_points": int(len(sensitivity)),
            "m_sensitivity_fixed_cohort_size": int(len(by_m[64])),
            "onset_opened": False,
            "target_streams_opened": False,
        },
    )
    print(
        f"V43 CONFIGURATIONS done: {len(deployment)} deployment points, {len(sensitivity)} m-sensitivity points, "
        f"fixed common cohort {len(by_m[64])} devices",
        flush=True,
    )


# --------------------------------------------------------------------------
# Phase 4: stats. The FIRST phase allowed to open onset/status evidence.
# Builds events (via v42_score.classify_event, reused not reimplemented),
# then the experiment-clustered bootstrap, onset-tolerance band, lead-time
# distributions, and the secondary negative-control enumeration.
# --------------------------------------------------------------------------
def _load_evidence() -> pd.DataFrame:
    spec = specification()
    path = HERE / spec["source_evidence_csv"]
    return pd.read_csv(path).set_index("client_id")


def build_events(predictions: pd.DataFrame) -> pd.DataFrame:
    """Merge locked predictions with evidence (status/EoL/onset), post-lock.

    Row classification is delegated to
    ``raw_online_federated_v42_score.classify_event`` -- reused, not
    reimplemented, so V43's "event" labels are defined identically to V42's.
    """
    evidence = _load_evidence()
    rows = []
    for record in predictions.to_dict("records"):
        client_id = record["client_id"]
        if client_id not in evidence.index:
            raise IntegrityError(f"no evidence row for {client_id}")
        meta = evidence.loc[client_id]
        status = str(meta["status"])
        acquisitions = int(meta["acquisitions"])
        onset_value = meta.get("first_sustained_onset_acquisition", np.nan)
        classified = v42_score.classify_event(status, record["first_alarm"], acquisitions, onset_value)
        rows.append(
            {
                **record,
                "status": status,
                "acquisitions_evidence": acquisitions,
                "first_sustained_onset_acquisition": onset_value,
                **classified,
            }
        )
    return pd.DataFrame(rows)


def require_stats_lock(root: Path) -> dict:
    require_prediction_lock(root)
    path = root / "v43_stats_lock.json"
    if not path.is_file():
        raise IntegrityError("V43 stats phase must run first")
    lock = read_json(path)
    if lock.get("protocol_sha256") != sha256(PROTOCOL):
        raise IntegrityError("V43 protocol changed after stats")
    artifacts = {
        "deployment_events_sha256": "v43_deployment_events.csv",
        "sensitivity_events_sha256": "v43_sensitivity_events.csv",
        "bootstrap_sha256": "v43_bootstrap_endpoints.csv",
        "matched_pair_sha256": "v43_matched_pair_pca.csv",
        "onset_tolerance_sha256": "v43_onset_tolerance_band.csv",
        "lead_time_sha256": "v43_lead_time.csv",
        "secondary_negative_controls_sha256": "v43_secondary_negative_controls.csv",
    }
    for key, filename in artifacts.items():
        if lock.get(key) != sha256(root / filename):
            raise IntegrityError(f"V43 stats artifact changed after stats lock: {filename}")
    return lock


def stats_phase(root: Path) -> None:
    require_prediction_lock(root)
    deployment = pd.read_csv(root / "v43_deployment_grid.csv")
    sensitivity = pd.read_csv(root / "v43_m_sensitivity.csv")
    deployment_events = build_events(deployment)
    sensitivity_events = build_events(sensitivity)
    deployment_events.to_csv(root / "v43_deployment_events.csv", index=False)
    sensitivity_events.to_csv(root / "v43_sensitivity_events.csv", index=False)

    bootstrap_rows: list[dict] = []
    for (config, z, epc), group in deployment_events.groupby(["config", "nominal_z", "epc"]):
        healthy = group[group["status"] == "healthy"].to_dict("records")
        failed = group[group["status"] == "failed"].to_dict("records")
        p1 = stats_mod.clustered_bootstrap(
            healthy, lambda subset: (float(np.mean([r["ever_alarmed"] for r in subset])) if subset else float("nan"))
        )
        p3 = stats_mod.clustered_bootstrap(
            failed, lambda subset: (float(np.mean([r["preterminal_detected"] for r in subset])) if subset else float("nan"))
        )
        bootstrap_rows.append(
            {
                "config": config, "nominal_z": float(z), "epc": bool(epc),
                "p1_point": p1["point"], "p1_ci_low": p1["ci_low"], "p1_ci_high": p1["ci_high"],
                "p1_experiments": p1["experiments"], "p1_streams": len(healthy),
                "p3_point": p3["point"], "p3_ci_low": p3["ci_low"], "p3_ci_high": p3["ci_high"],
                "p3_experiments": p3["experiments"], "p3_streams": len(failed),
            }
        )
    write_csv(root / "v43_bootstrap_endpoints.csv", bootstrap_rows)

    # SS2.2.1 (mandatory): the matched-algorithm pair F-L(PCA) minus
    # L-L(PCA) is the actual isolation of federated representation
    # learning -- F-L vs L-L is confounded (varies training locality AND
    # branch count at once) and must never be read as that isolation.
    # Paired per device, clustered on machine-life experiment, same as
    # every other bootstrap here.
    matched_pair_out: list[dict] = []
    for z, group in deployment_events.groupby("nominal_z"):
        fl_pca = group[group["config"] == "F-L(PCA)"]
        ll_pca = group[group["config"] == "L-L(PCA)"]
        fl_pca_healthy = fl_pca[fl_pca["status"] == "healthy"].to_dict("records")
        ll_pca_healthy = ll_pca[ll_pca["status"] == "healthy"].to_dict("records")
        fl_pca_failed = fl_pca[fl_pca["status"] == "failed"].to_dict("records")
        ll_pca_failed = ll_pca[ll_pca["status"] == "failed"].to_dict("records")
        p1_diff = stats_mod.matched_pair_bootstrap(fl_pca_healthy, ll_pca_healthy, "ever_alarmed")
        p3_diff = stats_mod.matched_pair_bootstrap(fl_pca_failed, ll_pca_failed, "preterminal_detected")
        matched_pair_out.append(
            {
                "nominal_z": float(z),
                "comparison": "F-L(PCA) minus L-L(PCA) -- isolates federated representation learning (SS2.2.1)",
                "p1_diff_point": p1_diff["point"], "p1_diff_ci_low": p1_diff["ci_low"], "p1_diff_ci_high": p1_diff["ci_high"],
                "p1_diff_paired_streams": p1_diff["paired_devices"], "p1_diff_experiments": p1_diff["experiments"],
                "p3_diff_point": p3_diff["point"], "p3_diff_ci_low": p3_diff["ci_low"], "p3_diff_ci_high": p3_diff["ci_high"],
                "p3_diff_paired_streams": p3_diff["paired_devices"], "p3_diff_experiments": p3_diff["experiments"],
            }
        )
    write_csv(root / "v43_matched_pair_pca.csv", matched_pair_out)

    tolerance_rows: list[dict] = []
    for (config, z, epc), group in deployment_events.groupby(["config", "nominal_z", "epc"]):
        for entry in stats_mod.onset_tolerance_band(group.to_dict("records")):
            tolerance_rows.append({"config": config, "nominal_z": float(z), "epc": bool(epc), **entry})
    write_csv(root / "v43_onset_tolerance_band.csv", tolerance_rows)

    lead_all: list[dict] = []
    for (config, z, epc), group in deployment_events.groupby(["config", "nominal_z", "epc"]):
        for entry in stats_mod.lead_time_rows(group.to_dict("records")):
            lead_all.append({"config": config, "nominal_z": float(z), "epc": bool(epc), **entry})
    write_csv(root / "v43_lead_time.csv", lead_all)

    lead_summary_rows: list[dict] = []
    if lead_all:
        lead_frame = pd.DataFrame(lead_all)
        for (config, z, epc), group in lead_frame.groupby(["config", "nominal_z", "epc"]):
            summary_acq = stats_mod.median_iqr(group["lead_time_acquisitions"].tolist())
            summary_sec = stats_mod.median_iqr(group["lead_time_seconds"].tolist())
            lead_summary_rows.append(
                {
                    "config": config, "nominal_z": float(z), "epc": bool(epc),
                    "acquisitions_median": summary_acq["median"], "acquisitions_q1": summary_acq["q1"],
                    "acquisitions_q3": summary_acq["q3"], "acquisitions_n": summary_acq["n"],
                    "seconds_median": summary_sec["median"], "seconds_q1": summary_sec["q1"],
                    "seconds_q3": summary_sec["q3"], "seconds_n": summary_sec["n"],
                }
            )
    write_csv(root / "v43_lead_time_summary.csv", lead_summary_rows)

    evidence = _load_evidence()
    negative_controls = stats_mod.secondary_negative_control_candidates(evidence)
    write_csv(root / "v43_secondary_negative_controls.csv", negative_controls)

    unique_rows = [{"client_id": cid} for cid in sorted(deployment_events["client_id"].unique())]
    g2_verified = stats_mod.verify_no_partial_experiment_across_resamples(unique_rows)
    if not g2_verified:
        raise IntegrityError("V43 gate G2 structural check failed: a bootstrap resample split a simultaneous experiment")

    write_json(
        root / "v43_stats_lock.json",
        {
            "protocol_sha256": sha256(PROTOCOL),
            "prediction_lock_sha256": sha256(root / "v43_prediction_lock.json"),
            "deployment_events_sha256": sha256(root / "v43_deployment_events.csv"),
            "sensitivity_events_sha256": sha256(root / "v43_sensitivity_events.csv"),
            "bootstrap_sha256": sha256(root / "v43_bootstrap_endpoints.csv"),
            "matched_pair_sha256": sha256(root / "v43_matched_pair_pca.csv"),
            "onset_tolerance_sha256": sha256(root / "v43_onset_tolerance_band.csv"),
            "lead_time_sha256": sha256(root / "v43_lead_time.csv"),
            "secondary_negative_controls_sha256": sha256(root / "v43_secondary_negative_controls.csv"),
            "g2_no_partial_experiment_verified": bool(g2_verified),
            "g2_resamples_checked": stats_mod.BOOTSTRAP_DRAWS,
            "onset_opened": True,
        },
    )
    print(f"V43 STATS done: G2 structural check (no partial experiment across {stats_mod.BOOTSTRAP_DRAWS} resamples) = {g2_verified}", flush=True)


# --------------------------------------------------------------------------
# Phase 5: comms (SS4, Leg 4). Independent of predictions/onset -- only
# needs the cohort table (node counts) and the frozen federated model
# bundle (payload size). Reuses raw_online_federated_v39_comm, not
# reimplemented.
# --------------------------------------------------------------------------
def require_comms_lock(root: Path) -> dict:
    path = root / "v43_comms_lock.json"
    if not path.is_file():
        raise IntegrityError("V43 comms phase must run first")
    lock = read_json(path)
    if lock.get("comms_sha256") != sha256(root / "v43_comms.json"):
        raise IntegrityError("V43 comms artifact changed after comms lock")
    return lock


def comms_phase(root: Path) -> None:
    require_cohort_lock(root)
    bundle = v41base.load_bundle(V41R1_ROOT / "models" / "all_source.npz")
    cohort_rows = read_json(root / "v43_cohort_table.json")["rows"]
    accounting = comms_mod.full_accounting(bundle, cohort_rows)
    write_json(root / "v43_comms.json", accounting)
    write_json(
        root / "v43_comms_lock.json",
        {
            "protocol_sha256": sha256(PROTOCOL),
            "cohort_lock_sha256": sha256(root / "v43_cohort_lock.json"),
            "comms_sha256": sha256(root / "v43_comms.json"),
            "onset_opened": False,
        },
    )
    print(
        f"V43 COMMS done: model_body_bytes={accounting['model_body_bytes']}, "
        f"bearing-client nodes={accounting['total_bearing_client_nodes']}, "
        f"physical nodes={accounting['total_physical_nodes']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("audit", "cohort", "grid", "configurations", "stats", "comms"),
    )
    args = parser.parse_args()
    root = root_for(args.run_id)
    actions = {
        "audit": audit,
        "cohort": cohort_phase,
        "grid": grid_phase,
        "configurations": configurations_phase,
        "stats": stats_phase,
        "comms": comms_phase,
    }
    try:
        actions[args.phase](root)
    except IntegrityError as error:
        print(f"V43 INTEGRITY FAILURE ({args.phase}): {error}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
