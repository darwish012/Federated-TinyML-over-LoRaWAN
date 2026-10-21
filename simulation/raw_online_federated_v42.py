"""V42 phase driver: audit, design, calibrate, predict-source, lock-source.

V42 repairs V41R1's premature-alarm failure by varying the score-calibration
sample size and (arm C) bootstrap-adjusting the alarm limit for estimation
error (V42_PREREGISTRATION.md). It reuses V41R1's frozen streams, trained
models and V29b triad UNCHANGED -- there is no training phase here at all.
The score-source phase (gate evaluation, reports, figures) lives in the
separate script ``raw_online_federated_v42_score.py``, mirroring V32's
driver/score split.

Phases, in order: audit -> design -> calibrate -> predict-source ->
lock-source -> (score-source, in raw_online_federated_v42_score.py).

Four arms per device (see raw_online_federated_v42_calibration.py for the
numerics):

    A_V41R1_BASELINE  8 score-calibration points, exact V41R1 reproduction
                       control -- computed by calling
                       raw_online_federated_v41.score_device() directly, not
                       by re-implementing it, so G5 is a genuine independent
                       check rather than a tautology.
    B_CALIB_SWEEP      calibration in {8, 16, 32, 64, 128}, limit rule unchanged
    C_EPC_ADJUSTED     same sweep, bootstrap-adjusted per-branch moderate limit
    D_SELF_STARTING    no fixed calibration window; each observation
                       standardized against all prior observations
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

import raw_online_federated_v25_policy as policy
import raw_online_federated_v29_models as models
import raw_online_federated_v41 as v41base
import raw_online_federated_v42_calibration as calib
import raw_online_federated_v42_design as design_mod

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "raw_online_federated_v42_protocol.json"
V41R1_ROOT = RESULTS / "raw_online_federated_v41r1_confirm1"
EXPERIMENT = "V42"


class IntegrityError(RuntimeError):
    """Raised for a frozen-dependency hash mismatch (hard constraint #1, exit 3)."""


# --------------------------------------------------------------------------
# House-style helpers (deliberately mirrored from raw_online_federated_v41,
# not reimplemented independently, so hashing is byte-identical).
# --------------------------------------------------------------------------
sha256 = v41base.sha256
read_json = v41base.read_json
write_json = v41base.write_json


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _safe_name(client_id: str) -> str:
    return client_id.replace("::", "__")


def root_for(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("RunId must match [A-Za-z0-9_-]+")
    root = (RESULTS / f"raw_online_federated_v42_{run_id}").resolve()
    if RESULTS.resolve() not in root.parents:
        raise ValueError("V42 results must stay under results/")
    return root


def specification() -> dict:
    value = read_json(PROTOCOL)
    original = read_json(v41base.V29B_LOCK)["selected_candidate"]
    fixed = value["frozen_v29b_policy"]
    for key in ("moderate", "strong", "vote_fraction", "k", "m", "weights"):
        if fixed[key] != original[key]:
            raise IntegrityError(f"V42 changed frozen V29b policy field {key}")
    return value


def _seed_for(base_seed: int, *parts: object) -> int:
    """Deterministic per-(device, branch, calibration-size) bootstrap seed.

    Uses hashlib (not Python's randomized built-in hash()) so the seed
    sequence is reproducible across processes and machines.
    """
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int((int(base_seed) + int.from_bytes(digest[:8], "big")) % (2**32 - 1))


# --------------------------------------------------------------------------
# Phase 0: audit
# --------------------------------------------------------------------------
def dependency_paths() -> dict[str, Path]:
    return {
        "protocol": PROTOCOL,
        "calibration_module": HERE / "raw_online_federated_v42_calibration.py",
        "design_module": HERE / "raw_online_federated_v42_design.py",
        "driver": Path(__file__).resolve(),
        "score_module": HERE / "raw_online_federated_v42_score.py",
        "calibration_tests": HERE / "raw_online_federated_v42_calibration_tests.py",
        "design_tests": HERE / "raw_online_federated_v42_design_tests.py",
        "tests": HERE / "raw_online_federated_v42_tests.py",
        "launcher": HERE / "start_v42_guarded.ps1",
        "design_doc": HERE / "V42_DESIGN_AND_RUN.md",
    }


def verify_frozen_v41r1_evidence() -> dict:
    """Re-verify V41R1's own lock files and the model/stream files they pin.

    This is hard constraint #1 from V42_PREREGISTRATION.md SS6: "Verify
    V41R1 model/stream hashes and reuse. Hash mismatch -> exit 3." Nothing
    here is retrained; every hash checked here is a hash of a file V41R1
    already produced or consumed.
    """
    spec = specification()
    expected_locks = spec["reused_v41r1_locks_sha256"]
    lock_files = {
        "v41_audit_lock.json": V41R1_ROOT / "v41_audit_lock.json",
        "v41_model_lock.json": V41R1_ROOT / "v41_model_lock.json",
        "v41_source_prediction_lock.json": V41R1_ROOT / "v41_source_prediction_lock.json",
        "v41_source_gate.json": V41R1_ROOT / "v41_source_gate.json",
        "v41_source_per_device.csv": V41R1_ROOT / "v41_source_per_device.csv",
        "training_trace.json": V41R1_ROOT / "training_trace.json",
    }
    for name, path in lock_files.items():
        if not path.is_file():
            raise IntegrityError(f"missing frozen V41R1 dependency: {path}")
        if sha256(path) != expected_locks[name]:
            raise IntegrityError(f"V41R1 frozen dependency changed since V42 was authored: {name}")

    audit_lock = read_json(lock_files["v41_audit_lock.json"])
    model_lock = read_json(lock_files["v41_model_lock.json"])
    if audit_lock.get("source_clients") != 40 or model_lock.get("source_clients") != 40:
        raise IntegrityError("V41R1 frozen source cohort is not the expected 40 clients")
    if audit_lock.get("target_sensor_values_opened") is not False:
        raise IntegrityError("V41R1 audit lock shows target sensor values were opened")

    expected_models = spec["reused_v41r1_model_sha256"]
    for name, digest in model_lock["models"].items():
        if expected_models.get(name) != digest:
            raise IntegrityError(f"V42 protocol model hash pin does not match V41R1's own model lock: {name}")
        path = V41R1_ROOT / "models" / name
        if not path.is_file() or sha256(path) != digest:
            raise IntegrityError(f"V41R1 frozen model file changed: {name}")

    for client_id, digest in audit_lock["source_stream_sha256"].items():
        dataset, unit = client_id.split("::", 1)
        path = v41base.V19_STREAMS / f"{dataset}__{unit}.npz"
        if not path.is_file() or sha256(path) != digest:
            raise IntegrityError(f"V19 source stream changed since V41R1's audit: {client_id}")

    return {"audit_lock": audit_lock, "model_lock": model_lock}


def guard_structural_check() -> dict:
    """Required structural check (V42_PREREGISTRATION.md SS5 "G2 erratum").

    Asserts guard_length >= persistence_window for every guard geometry V42
    deploys (the fixed-window B/C guard, always 4 acquisitions regardless of
    calibration size m, and the self-starting D guard, also 4). When the
    assertion does not hold -- it does not, at this geometry -- this prints
    a prominent warning rather than letting a structurally-forced 0% guard
    latch rate pass unremarked, and returns the flag recorded in
    v42_audit_lock.json.
    """
    fixed_window = calib.guard_latch_is_structurally_vacuous(calib.GUARD_SIZE)
    self_starting = calib.guard_latch_is_structurally_vacuous(
        calib.SELF_START_GUARD.stop - calib.SELF_START_GUARD.start
    )
    result = {
        "fixed_window_guard": fixed_window,
        "self_starting_guard": self_starting,
        "latch_based_guard_metrics_vacuous": bool(
            fixed_window["latch_based_guard_metrics_vacuous"] or self_starting["latch_based_guard_metrics_vacuous"]
        ),
    }
    if result["latch_based_guard_metrics_vacuous"]:
        print(
            "V42 WARNING: latch-based guard metrics are STRUCTURALLY VACUOUS at this geometry -- "
            f"guard_length={fixed_window['guard_length']}, persistence_window={fixed_window['persistence_window']}, "
            f"complete_windows={fixed_window['complete_windows']} (<=0 means the guard can never latch, at any "
            "threshold, for any arm or calibration size). Reported as 0% == arithmetically forced, not measured "
            "safety. G2 is redefined as the per-acquisition guard exceedance rate; see V42_PREREGISTRATION.md SS5.",
            flush=True,
        )
    return result


# Files the guarded launcher creates inside the result directory BEFORE the
# audit phase runs. They are not V42 scientific artifacts and must not trip the
# overwrite guard below -- `start_v42_guarded.ps1` does New-Item on the result
# root and then Start-Transcript into it, so by the time audit() executes the
# directory is already non-empty by construction.
LAUNCHER_ARTIFACTS = frozenset({"v42_console_transcript.txt"})


def audit(root: Path) -> None:
    if root.exists():
        preexisting = sorted(
            path.name for path in root.iterdir() if path.name not in LAUNCHER_ARTIFACTS
        )
        if preexisting:
            raise IntegrityError(
                f"V42 RunId already contains artifacts: {root} ({preexisting})"
            )
    specification()
    verify_frozen_v41r1_evidence()
    deps = dependency_paths()
    missing = [str(path) for path in deps.values() if not path.is_file()]
    if missing:
        raise IntegrityError(f"missing V42 protected dependency: {missing}")
    guard_check = guard_structural_check()
    write_json(
        root / "v42_audit_lock.json",
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
            "latch_based_guard_metrics_vacuous": guard_check["latch_based_guard_metrics_vacuous"],
        },
    )
    print(f"V42 AUDIT ok: {root / 'v42_audit_lock.json'}", flush=True)


def require_audit(root: Path) -> dict:
    path = root / "v42_audit_lock.json"
    if not path.is_file():
        raise IntegrityError("V42 audit phase must run first")
    lock = read_json(path)
    if lock.get("protocol_sha256") != sha256(PROTOCOL):
        raise IntegrityError("V42 protocol changed after audit")
    for name, path2 in dependency_paths().items():
        if lock["dependency_sha256"].get(name) != sha256(path2):
            raise IntegrityError(f"V42 frozen dependency changed after audit: {name}")
    if lock.get("target_streams_opened") is not False or lock.get("onset_opened") is not False:
        raise IntegrityError("invalid V42 audit access state")
    verify_frozen_v41r1_evidence()
    return lock


# --------------------------------------------------------------------------
# Device loading -- each device's stream and raw branch scores are loaded
# ONCE per phase invocation and reused across every arm/calibration size
# for that device within that phase (the V25 caching lesson).
# --------------------------------------------------------------------------
def source_roster() -> list[tuple[str, str, str]]:
    return v41base.source_roster()


def load_device_stream(dataset: str, unit: str) -> tuple[np.ndarray, np.ndarray]:
    return v41base.source_stream(dataset, unit)


def load_fold_bundle(dataset: str) -> dict:
    return v41base.load_bundle(V41R1_ROOT / "models" / f"fold_without_{dataset}.npz")


def raw_branch_scores(values: np.ndarray, bundle: dict) -> dict[str, np.ndarray]:
    normalized, _, _ = v41base.normalize_device(values)
    return v41base._raw_branch_scores(bundle, normalized)


# --------------------------------------------------------------------------
# Phase 2: calibrate (Section 3 EPC bootstrap + Section 4 eligibility ledger)
# --------------------------------------------------------------------------
def calibrate(root: Path) -> None:
    require_audit(root)
    require_design_lock(root)
    spec = specification()
    pol = spec["frozen_v29b_policy"]
    epc_cfg = spec["epc"]
    stream_lengths: dict[str, int] = {}
    epc_rows: list[dict] = []
    for dataset, unit, client_id in source_roster():
        values, _ = load_device_stream(dataset, unit)
        stream_lengths[client_id] = len(values)
        bundle = load_fold_bundle(dataset)
        raw = raw_branch_scores(values, bundle)
        for m in calib.CALIBRATION_SIZES:
            if len(values) < calib.required_acquisitions(str(m)):
                continue
            postcal = calib.postcal_slice(m)
            for branch in calib.BRANCHES:
                cal_values = raw[branch][postcal]
                seed = _seed_for(int(epc_cfg["seed"]), client_id, branch, m)
                result = calib.epc_adjusted_threshold(
                    cal_values,
                    moderate=float(pol["moderate"]),
                    seed=seed,
                    b=int(epc_cfg["B"]),
                    p=float(epc_cfg["p"]),
                )
                epc_rows.append(
                    {"client_id": client_id, "dataset": dataset, "branch": branch, "calibration_points": m, **result}
                )
        print(f"V42 CALIBRATE {client_id}", flush=True)
    elig_rows = calib.eligibility_rows(stream_lengths)
    write_csv(root / "v42_eligibility_ledger.csv", elig_rows)
    write_csv(root / "v42_epc_thresholds.csv", epc_rows)
    eligible_by_label = {
        label: sum(1 for row in elig_rows if row["calibration_label"] == label and row["eligible"])
        for label in calib.CALIBRATION_LABELS
    }
    write_json(
        root / "v42_calibration_lock.json",
        {
            "protocol_sha256": sha256(PROTOCOL),
            "audit_lock_sha256": sha256(root / "v42_audit_lock.json"),
            "design_lock_sha256": sha256(root / "v42_design_lock.json"),
            "eligibility_ledger_sha256": sha256(root / "v42_eligibility_ledger.csv"),
            "epc_thresholds_sha256": sha256(root / "v42_epc_thresholds.csv"),
            "eligible_devices_by_calibration_label": eligible_by_label,
            "source_devices": 40,
            "onset_opened": False,
        },
    )
    print(f"V42 CALIBRATE done: {eligible_by_label}", flush=True)


def require_design_lock(root: Path) -> dict:
    try:
        return design_mod.require_design_lock(root)
    except RuntimeError as error:
        raise IntegrityError(str(error)) from error


def require_calibration_lock(root: Path) -> dict:
    path = root / "v42_calibration_lock.json"
    if not path.is_file():
        raise IntegrityError("V42 calibrate phase must run first")
    lock = read_json(path)
    if lock.get("protocol_sha256") != sha256(PROTOCOL):
        raise IntegrityError("V42 protocol changed after calibration")
    if lock.get("eligibility_ledger_sha256") != sha256(root / "v42_eligibility_ledger.csv"):
        raise IntegrityError("V42 eligibility ledger changed after calibration lock")
    if lock.get("epc_thresholds_sha256") != sha256(root / "v42_epc_thresholds.csv"):
        raise IntegrityError("V42 EPC threshold table changed after calibration lock")
    if lock.get("onset_opened") is not False:
        raise IntegrityError("invalid V42 calibration access state")
    return lock


# --------------------------------------------------------------------------
# Phase 3: predict-source
# --------------------------------------------------------------------------
def _eligible(elig: pd.DataFrame, client_id: str, label: str) -> bool:
    row = elig[(elig["client_id"] == client_id) & (elig["calibration_label"] == label)]
    return bool(row.iloc[0]["eligible"]) if len(row) else False


def _summarize_point(
    client_id: str, dataset: str, arm: str, label: str,
    persistent: np.ndarray, raw: np.ndarray, guard_count: int, indices_full: np.ndarray, operation_start: int,
) -> dict:
    """Per-(device, operating point) summary row.

    Records BOTH guard figures per V42_PREREGISTRATION.md SS5's "G2
    erratum": ``guard_alarm`` (latch-based, structurally forced to False at
    this geometry -- see raw_online_federated_v42_calibration.guard_latch_is_structurally_vacuous)
    and ``guard_exceedances`` (per-acquisition count of the PRE-persistence
    ensemble condition, which is what the redefined G2 uses and can
    genuinely vary with calibration size and limit rule).
    """
    persistent = np.asarray(persistent, dtype=np.int8)
    raw = np.asarray(raw, dtype=bool)
    if len(raw) != len(persistent):
        raise IntegrityError(f"V42 raw/persistent length mismatch for {client_id} {arm} {label}")
    guard_primary = persistent[:guard_count]
    guard_raw = raw[:guard_count]
    operation = persistent[guard_count:]
    op_indices = indices_full[operation_start:]
    if len(operation) != len(op_indices):
        raise IntegrityError(f"V42 chronology mismatch for {client_id} {arm} {label}")
    positions = np.flatnonzero(operation)
    first_alarm = int(op_indices[positions[0]]) if len(positions) else -1
    return {
        "client_id": client_id,
        "dataset": dataset,
        "arm": arm,
        "calibration_label": label,
        "guard_count": int(guard_count),
        "guard_alarm": bool(np.any(guard_primary)),
        "guard_exceedances": int(np.sum(guard_raw)),
        "first_alarm": first_alarm,
        "operation_acquisitions": int(len(operation)),
        "acquisitions": int(len(indices_full)),
    }


def predict_source(root: Path) -> None:
    require_calibration_lock(root)
    spec = specification()
    pol = spec["frozen_v29b_policy"]
    weights = pol["weights"]
    moderate, strong, vote = float(pol["moderate"]), float(pol["strong"]), float(pol["vote_fraction"])
    elig = pd.read_csv(root / "v42_eligibility_ledger.csv")
    epc_table = pd.read_csv(root / "v42_epc_thresholds.csv")

    journal_hashes: dict[str, str] = {}
    summary_rows: list[dict] = []
    for dataset, unit, client_id in source_roster():
        values, indices = load_device_stream(dataset, unit)
        bundle = load_fold_bundle(dataset)
        journal_arrays: dict[str, object] = {"client_id": client_id, "dataset": dataset}
        raw = raw_branch_scores(values, bundle)

        # Shared m=8 zscores/ensemble computation, reused for both
        # B_CALIB_SWEEP@8 and arm A's guard-exceedance figure below: the two
        # are mathematically identical (calibration_tests.EnsembleEquivalenceTests),
        # so computing this once and reusing it is not an approximation.
        postcal_8 = calib.postcal_slice(8)
        guard_8 = calib.guard_slice(8)
        eval_slice_8 = slice(guard_8.start, len(values))
        zscores_8 = {
            branch: models.robust_z(raw[branch][postcal_8], raw[branch][eval_slice_8]) for branch in calib.BRANCHES
        }
        continuous_8, persistent_8, raw_exceed_8 = calib.ensemble_decision(zscores_8, weights, moderate, strong, vote)

        # Arm A: literal reuse of raw_online_federated_v41.score_device for
        # the deployed decision (persistent/first_alarm/continuous) -- not
        # reimplemented, so G5 is an independent reproduction check. Its
        # per-acquisition guard-exceedance figure (the redefined G2) reuses
        # raw_exceed_8 computed above -- the latch-based guard_primary V41
        # itself returns is structurally always empty (see the "G2 erratum").
        a_result = v41base.score_device(values, indices, bundle)
        a_persistent = np.concatenate([a_result["guard_primary"], a_result["primary_alarm"]]).astype(np.int8)
        journal_arrays["A8_persistent"] = a_persistent
        journal_arrays["A8_continuous"] = a_result["primary_continuous"]
        journal_arrays["A8_reported_first_alarm"] = np.asarray(a_result["first_alarm"])
        journal_arrays["A8_raw_exceedance"] = raw_exceed_8.astype(np.int8)
        summary_rows.append(
            _summarize_point(client_id, dataset, "A_V41R1_BASELINE", "8", a_persistent, raw_exceed_8, calib.GUARD_SIZE, indices, calib.op_start(8))
        )

        for m in calib.CALIBRATION_SIZES:
            label = str(m)
            if not _eligible(elig, client_id, label):
                continue
            postcal = calib.postcal_slice(m)
            guard = calib.guard_slice(m)
            opstart = calib.op_start(m)
            if m == 8:
                zscores, continuous_b, persistent_b, raw_exceed_b = zscores_8, continuous_8, persistent_8, raw_exceed_8
            else:
                eval_slice = slice(guard.start, len(values))
                zscores = {
                    branch: models.robust_z(raw[branch][postcal], raw[branch][eval_slice]) for branch in calib.BRANCHES
                }
                continuous_b, persistent_b, raw_exceed_b = calib.ensemble_decision(zscores, weights, moderate, strong, vote)
            journal_arrays[f"B{m}_persistent"] = persistent_b.astype(np.int8)
            journal_arrays[f"B{m}_continuous"] = continuous_b
            journal_arrays[f"B{m}_raw_exceedance"] = raw_exceed_b.astype(np.int8)
            summary_rows.append(
                _summarize_point(client_id, dataset, "B_CALIB_SWEEP", label, persistent_b, raw_exceed_b, guard.stop - guard.start, indices, opstart)
            )

            adjusted: dict[str, float] = {}
            for branch in calib.BRANCHES:
                sub = epc_table[
                    (epc_table["client_id"] == client_id)
                    & (epc_table["branch"] == branch)
                    & (epc_table["calibration_points"] == m)
                ]
                if len(sub) != 1:
                    raise IntegrityError(f"missing EPC threshold row for {client_id} {branch} m={m}")
                adjusted[branch] = float(sub.iloc[0]["adjusted_moderate_z"])
            continuous_c, persistent_c, raw_exceed_c = calib.ensemble_decision(zscores, weights, adjusted, strong, vote)
            journal_arrays[f"C{m}_persistent"] = persistent_c.astype(np.int8)
            journal_arrays[f"C{m}_continuous"] = continuous_c
            journal_arrays[f"C{m}_raw_exceedance"] = raw_exceed_c.astype(np.int8)
            summary_rows.append(
                _summarize_point(client_id, dataset, "C_EPC_ADJUSTED", label, persistent_c, raw_exceed_c, guard.stop - guard.start, indices, opstart)
            )

        if _eligible(elig, client_id, "self_starting"):
            guard_d = calib.SELF_START_GUARD
            opstart_d = calib.SELF_START_OP_START
            zscores_d = {}
            for branch in calib.BRANCHES:
                z_full = calib.self_starting_z(raw[branch])
                zscores_d[branch] = np.nan_to_num(z_full[guard_d.start :], nan=-1.0e6)
            continuous_d, persistent_d, raw_exceed_d = calib.ensemble_decision(zscores_d, weights, moderate, strong, vote)
            journal_arrays["D_persistent"] = persistent_d.astype(np.int8)
            journal_arrays["D_continuous"] = continuous_d
            journal_arrays["D_raw_exceedance"] = raw_exceed_d.astype(np.int8)
            summary_rows.append(
                _summarize_point(client_id, dataset, "D_SELF_STARTING", "self_starting", persistent_d, raw_exceed_d, guard_d.stop - guard_d.start, indices, opstart_d)
            )

        journal_path = root / "source_journals" / f"{_safe_name(client_id)}.npz"
        save_npz(journal_path, **journal_arrays)
        journal_hashes[client_id] = sha256(journal_path)
        print(f"V42 PREDICT {client_id}", flush=True)

    write_csv(root / "v42_source_predictions_summary.csv", summary_rows)
    write_json(
        root / "v42_source_prediction_manifest.json",
        {
            "protocol_sha256": sha256(PROTOCOL),
            "calibration_lock_sha256": sha256(root / "v42_calibration_lock.json"),
            "journal_sha256": journal_hashes,
            "summary_sha256": sha256(root / "v42_source_predictions_summary.csv"),
            "operating_points": len(summary_rows),
            "source_devices": 40,
            "onset_opened": False,
            "target_streams_opened": False,
        },
    )
    print(f"V42 PREDICT done: {len(summary_rows)} operating points across 40 devices", flush=True)


def require_prediction_manifest(root: Path) -> dict:
    require_calibration_lock(root)
    path = root / "v42_source_prediction_manifest.json"
    if not path.is_file():
        raise IntegrityError("V42 predict-source phase must run first")
    manifest = read_json(path)
    if manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise IntegrityError("V42 protocol changed after predict-source")
    if manifest.get("summary_sha256") != sha256(root / "v42_source_predictions_summary.csv"):
        raise IntegrityError("V42 prediction summary changed after predict-source")
    for client_id, digest in manifest["journal_sha256"].items():
        journal_path = root / "source_journals" / f"{_safe_name(client_id)}.npz"
        if not journal_path.is_file() or sha256(journal_path) != digest:
            raise IntegrityError(f"V42 source journal changed: {client_id}")
    if manifest.get("onset_opened") is not False or manifest.get("target_streams_opened") is not False:
        raise IntegrityError("invalid V42 prediction manifest access state")
    return manifest


# --------------------------------------------------------------------------
# Phase 4: lock-source
# --------------------------------------------------------------------------
def lock_source(root: Path) -> None:
    manifest = require_prediction_manifest(root)
    write_json(
        root / "v42_source_prediction_lock.json",
        {
            "protocol_sha256": sha256(PROTOCOL),
            "manifest_sha256": sha256(root / "v42_source_prediction_manifest.json"),
            "operating_points": manifest["operating_points"],
            "source_devices": 40,
            "onset_opened": False,
            "target_streams_opened": False,
        },
    )
    print(f"V42 LOCK-SOURCE ok: {root / 'v42_source_prediction_lock.json'}", flush=True)


def require_source_prediction_lock(root: Path) -> dict:
    manifest = require_prediction_manifest(root)
    path = root / "v42_source_prediction_lock.json"
    if not path.is_file():
        raise IntegrityError("V42 lock-source phase must run first")
    lock = read_json(path)
    if lock.get("manifest_sha256") != sha256(root / "v42_source_prediction_manifest.json"):
        raise IntegrityError("V42 prediction manifest changed after lock-source")
    if lock.get("onset_opened") is not False or lock.get("target_streams_opened") is not False:
        raise IntegrityError("invalid V42 source-prediction-lock access state")
    return lock


def design(root: Path) -> None:
    try:
        design_mod.design_phase(root)
    except RuntimeError as error:
        raise IntegrityError(str(error)) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("audit", "design", "calibrate", "predict-source", "lock-source"),
    )
    args = parser.parse_args()
    root = root_for(args.run_id)
    actions = {
        "audit": audit,
        "design": design,
        "calibrate": calibrate,
        "predict-source": predict_source,
        "lock-source": lock_source,
    }
    try:
        actions[args.phase](root)
    except IntegrityError as error:
        print(f"V42 INTEGRITY FAILURE ({args.phase}): {error}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
