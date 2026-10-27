"""V45 Section 4/9: the 30-candidate grid (5 nominal z x 6 persistence rules),
computed once per device from the frozen triad's cached branch scores.

ONSET BAN (V45_PREREGISTRATION.md Section 1): this module must never import,
call, or otherwise reference ``raw_online_federated_v41``'s post-lock
onset-interval proxy function (the one that shares a 4-of-7 persistence
structure with the deployed alarm -- see Section 1), and no onset quantity
may enter any value this module returns. Every metric here
-- ``ever_alarmed``, ``preterminal_detected``, ``lead_time_fraction_of_life``,
guard exceedance counts -- is a pure function of (stream length, first alarm
acquisition), never of an onset estimate. This is enforced mechanically by
``raw_online_federated_v45_onset_ban_tests.py``.

Reused, never reimplemented:

    raw_online_federated_v42.raw_branch_scores/load_fold_bundle   the frozen
        triad's per-branch raw scores and the leave-one-dataset-out /
        all_source model bundles
    raw_online_federated_v43_grid.zscores_at/epc_moderate          device-
        local z-scores and the EPC bootstrap-adjusted per-branch moderate
        limit, at m=64, generalized over an arbitrary nominal z (already
        built for V43's own z-grid -- V45 reuses it outright, not only at
        z=2.0)
    raw_online_federated_v42_calibration.ensemble_decision/
        guard_latch_is_structurally_vacuous/GUARD_SIZE/postcal_slice/
        guard_slice/op_start                                       the
        weighted moderate/strong ensemble condition and the guard structural
        check (V45_PREREGISTRATION.md Section 3)
    raw_online_federated_v25_policy.latch_stream/LatchConfig       the
        k-of-m persistence primitive, generalized here to the 6-rule grid
        instead of the single hardcoded 4-of-7 V42/V43/V44 use

No retraining, no raw extraction (V45_PREREGISTRATION.md Section 9): branch
scores are the three already-frozen triad outputs on an already-materialized
100-D feature stream, computed exactly once per device
(:func:`raw_branch_scores`, delegated outright to
``raw_online_federated_v42.raw_branch_scores``) and cached by the caller
across all 30 candidates -- this module's :func:`compute_device_candidates`
performs exactly one EPC bootstrap per (branch, z) -- 5 z-values x 3 branches
= 15 bootstrap draws per device -- and reuses that same raw ensemble-exceedance
stream across all 6 persistence rules for each z (no recomputation), per
Section 9's "no retraining, no raw extraction" and "computed once ... cached
... swept" requirement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import raw_online_federated_v25_policy as policy
import raw_online_federated_v42 as v42
import raw_online_federated_v42_calibration as calib
import raw_online_federated_v43_grid as v43grid

CALIBRATION_M = 64
Z_GRID: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5)
PERSISTENCE_GRID: tuple[tuple[str, int, int], ...] = (
    ("2-of-3", 2, 3),
    ("3-of-5", 3, 5),
    ("3-of-7", 3, 7),
    ("4-of-7", 4, 7),
    ("5-of-7", 5, 7),
    ("5-of-9", 5, 9),
)
CANDIDATE_COUNT = len(Z_GRID) * len(PERSISTENCE_GRID)
assert CANDIDATE_COUNT == 30

BRANCHES = calib.BRANCHES
GUARD_SIZE = calib.GUARD_SIZE


def candidate_id(nominal_z: float, persistence_label: str) -> str:
    return f"z{nominal_z:.1f}/{persistence_label}"


def all_candidates() -> list[tuple[float, str, int, int]]:
    """(nominal_z, persistence_label, k, m) for all 30 grid points, in a
    fixed, deterministic order (z-major, persistence-minor) -- 2.0/4-of-7 is
    simply the (z=2.0, label="4-of-7") entry, ordinary among the other 29
    (Section 4, gate G7)."""
    out = []
    for z in Z_GRID:
        for label, k, m in PERSISTENCE_GRID:
            out.append((float(z), label, int(k), int(m)))
    return out


def required_acquisitions() -> int:
    """Minimum stream length for one operational acquisition to exist at the
    frozen m=64 calibration window -- reused outright from
    ``raw_online_federated_v43_grid.required_acquisitions``."""
    return v43grid.required_acquisitions(CALIBRATION_M)


def eligible(n: int) -> bool:
    return int(n) >= required_acquisitions()


# --------------------------------------------------------------------------
# Model bundle loading -- dispatches on the cohort's precomputed
# bundle_dataset_key (raw_online_federated_v45_cohort.FOLD_TRAINED_DATASETS),
# reusing raw_online_federated_v42's own bundle loaders outright.
# --------------------------------------------------------------------------
def load_bundle(bundle_dataset_key: str) -> dict:
    if bundle_dataset_key == "all_source":
        return v42.v41base.load_bundle(v42.V41R1_ROOT / "models" / "all_source.npz")
    return v42.load_fold_bundle(bundle_dataset_key)


def raw_branch_scores(values: np.ndarray, bundle: dict) -> dict[str, np.ndarray]:
    return v42.raw_branch_scores(values, bundle)


# --------------------------------------------------------------------------
# Generalized persistence latch -- the only V45-specific addition over
# raw_online_federated_v42_calibration.k_of_m_latch, which hardcodes the
# single 4-of-7 LatchConfig. Delegates to the same underlying primitive
# (raw_online_federated_v25_policy.latch_stream), never reimplemented.
# --------------------------------------------------------------------------
def k_of_m_latch(flags: np.ndarray, k: int, m: int) -> np.ndarray:
    values = np.asarray(flags, dtype=np.float64).reshape(-1)
    return policy.latch_stream(values, 0.5, policy.LatchConfig(persistence_k=int(k), persistence_window=int(m)))


@dataclass(frozen=True)
class CandidateRow:
    client_id: str
    dataset: str
    experiment_id: str
    nominal_z: float
    persistence_label: str
    persistence_k: int
    persistence_window: int
    acquisitions: int
    guard_count: int
    guard_exceedances: int
    guard_latch_alarm: bool
    guard_latch_vacuous: bool
    first_alarm: int
    ever_alarmed: bool
    preterminal_detected: bool
    lead_time_acquisitions: int | None
    lead_time_fraction_of_life: float | None


def _row_dict(row: CandidateRow) -> dict:
    return {
        "candidate_id": candidate_id(row.nominal_z, row.persistence_label),
        "client_id": row.client_id,
        "dataset": row.dataset,
        "experiment_id": row.experiment_id,
        "nominal_z": row.nominal_z,
        "persistence_label": row.persistence_label,
        "persistence_k": row.persistence_k,
        "persistence_window": row.persistence_window,
        "acquisitions": row.acquisitions,
        "guard_count": row.guard_count,
        "guard_exceedances": row.guard_exceedances,
        "guard_latch_alarm": row.guard_latch_alarm,
        "guard_latch_vacuous": row.guard_latch_vacuous,
        "first_alarm": row.first_alarm,
        "ever_alarmed": row.ever_alarmed,
        "preterminal_detected": row.preterminal_detected,
        "lead_time_acquisitions": row.lead_time_acquisitions,
        "lead_time_fraction_of_life": row.lead_time_fraction_of_life,
    }


def _z_key(z: float) -> str:
    return f"{float(z):.1f}"


def compute_device_branch_cache(
    client_id: str,
    dataset: str,
    experiment_id: str,
    values: np.ndarray,
    indices: np.ndarray,
    bundle: dict,
    weights: dict[str, float],
    strong: float,
    vote_fraction: float,
    epc_seed: int,
) -> dict | None:
    """The expensive, EPC-bootstrap-bearing part of the pipeline, computed
    EXACTLY ONCE per device (the ``score-branches`` phase, V45_PREREGISTRATION.md
    Section 9's "compute ... once ... cache"): three raw branch scores, one
    z-score array per branch (calibration-window-dependent only, shared by
    every candidate z), and one pre-persistence ensemble exceedance boolean
    stream per nominal z (5 EPC bootstraps x 3 branches = 15 draws total).
    Returns ``None`` if the device is too short for the frozen m=64
    calibration window.

    The returned dict is npz-serializable by the phase driver
    (``raw_exceed_z1.5`` .. ``raw_exceed_z3.5`` boolean arrays plus scalar
    metadata) so the ``grid`` phase can sweep all 6 persistence rules per z
    -- 30 candidates total -- WITHOUT recomputing any branch score, z-score,
    or bootstrap draw (:func:`sweep_persistence`).
    """
    n = len(values)
    if not eligible(n):
        return None
    guard = v43grid.guard_slice(CALIBRATION_M)
    op_start = v43grid.op_start(CALIBRATION_M)
    guard_count = guard.stop - guard.start
    terminal = int(np.floor(0.9 * n))
    op_indices = indices[op_start:]

    raw = raw_branch_scores(values, bundle)
    # z-scores depend only on (raw, calibration window), never on the
    # candidate nominal z -- computed once and reused across all 5 z-values
    # (Section 9: "computed once per device ... cached").
    zscores = v43grid.zscores_at(raw, CALIBRATION_M, n)
    raw_exceed_by_z: dict[str, np.ndarray] = {}
    deployed_thresholds_by_z: dict[str, dict[str, float]] = {}
    for z in Z_GRID:
        moderate = v43grid.epc_moderate(raw, BRANCHES, CALIBRATION_M, z, client_id, epc_seed)
        _continuous, _persistent_unused, raw_exceed = calib.ensemble_decision(zscores, weights, moderate, strong, vote_fraction)
        raw_exceed = np.asarray(raw_exceed, dtype=bool)
        if len(raw_exceed) != n - guard.start:
            raise ValueError(f"V45 eval-window length mismatch for {client_id} z={z}")
        raw_exceed_by_z[_z_key(z)] = raw_exceed
        deployed_thresholds_by_z[_z_key(z)] = moderate

    return {
        "client_id": client_id,
        "dataset": dataset,
        "experiment_id": experiment_id,
        "acquisitions": int(n),
        "guard_count": int(guard_count),
        "terminal": int(terminal),
        "op_indices": op_indices,
        "raw_exceed_by_z": raw_exceed_by_z,
        "deployed_thresholds_by_z": deployed_thresholds_by_z,
    }


def sweep_persistence(cache: dict) -> list[dict]:
    """The cheap part (the ``grid`` phase): sweep all 6 persistence rules
    over each of the 5 already-cached raw-exceedance streams -- 30 candidate
    rows, no branch score, z-score, or bootstrap recomputation."""
    client_id = cache["client_id"]
    dataset = cache["dataset"]
    experiment_id = cache["experiment_id"]
    n = int(cache["acquisitions"])
    guard_count = int(cache["guard_count"])
    terminal = int(cache["terminal"])
    op_indices = np.asarray(cache["op_indices"])

    rows: list[dict] = []
    for z in Z_GRID:
        raw_exceed = np.asarray(cache["raw_exceed_by_z"][_z_key(z)], dtype=bool)
        for label, k, m in PERSISTENCE_GRID:
            persistent = k_of_m_latch(raw_exceed, k, m)
            guard_primary = persistent[:guard_count]
            guard_raw = raw_exceed[:guard_count]
            operation = persistent[guard_count:]
            if len(operation) != len(op_indices):
                raise ValueError(f"V45 chronology mismatch for {client_id} z={z} {label}")
            positions = np.flatnonzero(operation)
            first_alarm = int(op_indices[positions[0]]) if len(positions) else -1
            ever_alarmed = bool(first_alarm >= 0)
            preterminal_detected = bool(0 <= first_alarm < terminal)
            lead_acq = (n - first_alarm) if ever_alarmed else None
            lead_frac = (float(lead_acq) / float(n)) if ever_alarmed else None
            vacuity = calib.guard_latch_is_structurally_vacuous(guard_count, m)

            row = CandidateRow(
                client_id=client_id,
                dataset=dataset,
                experiment_id=experiment_id,
                nominal_z=float(z),
                persistence_label=label,
                persistence_k=int(k),
                persistence_window=int(m),
                acquisitions=int(n),
                guard_count=int(guard_count),
                guard_exceedances=int(np.sum(guard_raw)),
                guard_latch_alarm=bool(np.any(guard_primary)),
                guard_latch_vacuous=bool(vacuity["latch_based_guard_metrics_vacuous"]),
                first_alarm=first_alarm,
                ever_alarmed=ever_alarmed,
                preterminal_detected=preterminal_detected,
                lead_time_acquisitions=lead_acq,
                lead_time_fraction_of_life=lead_frac,
            )
            rows.append(_row_dict(row))
    if len(rows) != CANDIDATE_COUNT:
        raise RuntimeError(f"V45 grid must produce exactly {CANDIDATE_COUNT} candidate rows per eligible device")
    return rows


def compute_device_candidates(
    client_id: str,
    dataset: str,
    experiment_id: str,
    values: np.ndarray,
    indices: np.ndarray,
    bundle: dict,
    weights: dict[str, float],
    strong: float,
    vote_fraction: float,
    epc_seed: int,
) -> list[dict]:
    """All 30 candidate rows for one device in a single call -- a thin
    composition of :func:`compute_device_branch_cache` (score-branches) then
    :func:`sweep_persistence` (grid), for callers (tests, single-device
    smoke checks) that do not need the intermediate cache to persist to
    disk between two separate phase invocations."""
    cache = compute_device_branch_cache(
        client_id, dataset, experiment_id, values, indices, bundle, weights, strong, vote_fraction, epc_seed
    )
    if cache is None:
        return []
    return sweep_persistence(cache)


# --------------------------------------------------------------------------
# npz (de)serialization for the score-branches phase's on-disk cache.
# --------------------------------------------------------------------------
def save_branch_cache(path: Path, cache: dict) -> None:
    payload: dict[str, object] = {
        "client_id": np.asarray(cache["client_id"]),
        "dataset": np.asarray(cache["dataset"]),
        "experiment_id": np.asarray(cache["experiment_id"]),
        "acquisitions": np.asarray(cache["acquisitions"]),
        "guard_count": np.asarray(cache["guard_count"]),
        "terminal": np.asarray(cache["terminal"]),
        "op_indices": np.asarray(cache["op_indices"]),
    }
    for z in Z_GRID:
        key = _z_key(z)
        payload[f"raw_exceed_{key}"] = np.asarray(cache["raw_exceed_by_z"][key], dtype=bool)
        thresholds = cache["deployed_thresholds_by_z"][key]
        for branch in BRANCHES:
            payload[f"threshold_{key}_{branch}"] = np.asarray(float(thresholds[branch]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_branch_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        raw_exceed_by_z = {_z_key(z): np.asarray(archive[f"raw_exceed_{_z_key(z)}"], dtype=bool) for z in Z_GRID}
        deployed_thresholds_by_z = {
            _z_key(z): {branch: float(archive[f"threshold_{_z_key(z)}_{branch}"]) for branch in BRANCHES}
            for z in Z_GRID
        }
        return {
            "client_id": str(archive["client_id"]),
            "dataset": str(archive["dataset"]),
            "experiment_id": str(archive["experiment_id"]),
            "acquisitions": int(archive["acquisitions"]),
            "guard_count": int(archive["guard_count"]),
            "terminal": int(archive["terminal"]),
            "op_indices": np.asarray(archive["op_indices"]),
            "raw_exceed_by_z": raw_exceed_by_z,
            "deployed_thresholds_by_z": deployed_thresholds_by_z,
        }


__all__ = [
    "BRANCHES",
    "CALIBRATION_M",
    "CANDIDATE_COUNT",
    "GUARD_SIZE",
    "PERSISTENCE_GRID",
    "Z_GRID",
    "all_candidates",
    "candidate_id",
    "compute_device_branch_cache",
    "compute_device_candidates",
    "eligible",
    "k_of_m_latch",
    "load_bundle",
    "load_branch_cache",
    "raw_branch_scores",
    "save_branch_cache",
    "sweep_persistence",
    "required_acquisitions",
]
