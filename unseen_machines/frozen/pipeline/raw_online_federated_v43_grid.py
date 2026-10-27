"""V43 threshold grid (Section 2.1) and F-G/F-L/L-L deployment configurations
(Section 2.2), plus the m-sensitivity leg (Section 2.3).

Pure numerics on already-frozen V41R1 streams and models: no file writing, no
onset or status access, no target data. Every EPC bootstrap, eligibility, and
ensemble/latch primitive is IMPORTED from ``raw_online_federated_v42_calibration``
and ``raw_online_federated_v25_policy`` and never reimplemented here, per task
scope. This module only supplies the V43-specific numerics those primitives do
not already provide: the nominal-z grid, the three named deployment
configurations, and the m=256 sensitivity geometry.

SPEC-AMBIGUITY 1 (deployment configurations, SS2.2): the preregistration
defines F-G/F-L/L-L by MODEL and THRESHOLD STYLE, not by algorithm. The most
literal reading, reusing V42's own arms without inventing new mechanisms:

    F-G  federated triad model (raw_online_federated_v42.raw_branch_scores),
         NO EPC adjustment -> one GLOBAL nominal-z threshold shared by every
         device (V42 arm B_CALIB_SWEEP's mechanism, generalised across the
         full nominal-z grid instead of the single frozen value 2.0).
    F-L  federated triad model, WITH EPC adjustment -> a DEVICE-LOCAL
         bootstrap-adjusted threshold (V42 arm C_EPC_ADJUSTED's mechanism,
         generalised across the full nominal-z grid as the EPC bootstrap's
         starting/nominal moderate value).
    L-L  a purely local model (see raw_online_federated_v43_protocol.json
         "local_model"), WITH EPC adjustment (device-local), across the same
         nominal-z grid as F-L.

The preregistration's threshold grid (SS2.1: "each with and without EPC")
therefore maps onto exactly F-G (without) and F-L (with) at m=64; L-L extends
F-L's mechanism to the local model over the same grid. The fourth
combination -- a local model with a single GLOBAL (non-EPC) threshold -- is
NOT one of the three named configurations and is deliberately not computed
(task-scope instruction: do not add mechanisms the preregistration does not
describe).

SS2.2.1 (added after review, MANDATORY, not optional): F-L vs L-L is
CONFOUNDED as an isolation of federated representation learning -- F-L is the
three-branch V29b triad while the only available local comparator is
single-branch local PCA-16, so that comparison varies training locality AND
branch count simultaneously. SS2.2.1 requires a matched-algorithm pair that
varies training locality ALONE:

    F-L(PCA)  the FED_PCA16 branch ALONE (one of calib.BRANCHES, already
              computed for F-L -- no new branch or model), device-local EPC
              threshold, single-branch ensemble (weight 1.0), identical
              mechanism to L-L.
    L-L(PCA)  local PCA-16 ALONE, device-local EPC threshold -- numerically
              IDENTICAL to L-L (the "purely local model" chosen in
              SPEC-AMBIGUITY 1 already is local PCA-16), computed and
              reported under its own config label for a self-describing,
              symmetric matched-pair table rather than requiring the reader
              to infer L-L(PCA) == L-L.

F-L(PCA) reuses the SAME per-branch EPC-adjusted threshold and z-score arrays
already computed for F-L's three-branch ensemble (both are pure functions of
(client_id, branch, m), independent of which other branches are also being
adjusted) -- no extra bootstrap draw. L-L(PCA) reuses L-L's own threshold and
z-score arrays outright. This pair, not F-L vs L-L, is the isolation of
federated representation learning (SS2.2.1) -- reported explicitly as such
wherever it appears. F-L vs L-L is retained as the *deployment* comparison
(best shared configuration vs best local configuration) and is labelled
CONFOUNDED in every table it appears in, not only in prose.

SPEC-AMBIGUITY 2 (m=256 geometry, SS2.3): ``raw_online_federated_v42_calibration``
restricts its ``postcal_slice``/``guard_slice``/``op_start``/eligibility helpers
to V42's own fixed calibration-size enum ``{8, 16, 32, 64, 128}``, which does
not include 256. V43's m-sensitivity grid (SS2.3) requires m=256. This module
reimplements ONLY the trivial slice-index arithmetic for an arbitrary m (using
``calib.FIT_END``/``calib.GUARD_SIZE`` imported, not redefined, so the geometry
stays numerically identical to V42's for every m already in V42's enum) --
never the EPC bootstrap, eligibility computation, ensemble decision, or latch,
which remain imported from ``raw_online_federated_v42_calibration`` unchanged.
``raw_online_federated_v43_grid_tests.py`` asserts this local geometry agrees
with ``calib.postcal_slice``/``calib.guard_slice``/``calib.op_start`` exactly
for every m in V42's own enum.

SPEC-AMBIGUITY 3 (m-sensitivity threshold, SS2.3): SS2.3 names no threshold for
the m-sensitivity leg. The frozen V29b nominal moderate (2.0) is used, applied
identically across m so that only the calibration size varies -- the same
choice V42 made for its own primary m-sweep comparison.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

import raw_online_federated_v29_models as models
import raw_online_federated_v41 as v41base
import raw_online_federated_v42 as v42
import raw_online_federated_v42_calibration as calib

Z_GRID: tuple[float, ...] = (2.00, 2.25, 2.50, 2.75, 3.00)
PRIMARY_M = 64
SENSITIVITY_M_GRID: tuple[int, ...] = (64, 128, 256)
SENSITIVITY_Z = 2.0
DEPLOYMENT_CONFIGS: tuple[str, ...] = ("F-G", "F-L", "L-L")
MATCHED_PAIR_CONFIGS: tuple[str, ...] = ("F-L(PCA)", "L-L(PCA)")
CONFIGS: tuple[str, ...] = DEPLOYMENT_CONFIGS + MATCHED_PAIR_CONFIGS
SENSITIVITY_CONFIG = "F-L_M_SENSITIVITY"
LOCAL_BRANCH = "LOCAL_PCA16"
FED_PCA_BRANCH = "FED_PCA16"
EPC_B = calib.EPC_B
EPC_P = calib.EPC_P


# --------------------------------------------------------------------------
# Geometry (SPEC-AMBIGUITY 2): trivial slice arithmetic only, for m values
# outside V42's fixed enum (256). Uses calib.FIT_END/calib.GUARD_SIZE, never
# redefines them.
# --------------------------------------------------------------------------
def postcal_slice(m: int) -> slice:
    return slice(calib.FIT_END, calib.FIT_END + int(m))


def guard_slice(m: int) -> slice:
    start = calib.FIT_END + int(m)
    return slice(start, start + calib.GUARD_SIZE)


def op_start(m: int) -> int:
    start = calib.FIT_END + int(m)
    return start + calib.GUARD_SIZE


def required_acquisitions(m: int) -> int:
    return op_start(m) + 1


# --------------------------------------------------------------------------
# Local model (SPEC-AMBIGUITY 1): the already-computed per-device local-PCA
# comparator, a closed-form fit on that device's own 12-point FIT window --
# not a newly trained model, so "no retraining" is respected.
# --------------------------------------------------------------------------
def local_raw_score(values: np.ndarray) -> np.ndarray:
    normalized, _, _ = v41base.normalize_device(values)
    fit = normalized[calib.FIT]
    pca = models.fit_pca(fit, components=min(16, len(fit) - 1))
    return pca.score(normalized)


# --------------------------------------------------------------------------
# Z-score computation (device-local center/scale, shared by every arm) and
# EPC-adjusted moderate limits (delegated to calib.epc_adjusted_threshold,
# never reimplemented).
# --------------------------------------------------------------------------
def zscores_at(raw: Mapping[str, np.ndarray], m: int, n: int) -> dict[str, np.ndarray]:
    postcal = postcal_slice(m)
    guard = guard_slice(m)
    eval_slice = slice(guard.start, n)
    return {branch: models.robust_z(values[postcal], values[eval_slice]) for branch, values in raw.items()}


def epc_moderate(
    raw: Mapping[str, np.ndarray],
    branches: tuple[str, ...],
    m: int,
    z_nominal: float,
    client_id: str,
    epc_seed: int,
) -> dict[str, float]:
    """Per-branch EPC-adjusted moderate limit at nominal z, calibration size m.

    Reuses ``raw_online_federated_v42._seed_for`` verbatim so that at
    m=64, z_nominal=2.00 this reproduces V42's own arm C_EPC_ADJUSTED bootstrap
    draws bit-for-bit (required for gate G1) -- the seed depends only on
    (client_id, branch, m), never on z, matching V42's own convention.
    """
    postcal = postcal_slice(m)
    moderate: dict[str, float] = {}
    for branch in branches:
        cal_values = raw[branch][postcal]
        seed = v42._seed_for(int(epc_seed), client_id, branch, m)
        result = calib.epc_adjusted_threshold(cal_values, moderate=float(z_nominal), seed=seed, b=EPC_B, p=EPC_P)
        moderate[branch] = float(result["adjusted_moderate_z"])
    return moderate


def decide(
    zscores: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    moderate: Mapping[str, float] | float,
    strong: float,
    vote_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return calib.ensemble_decision(zscores, weights, moderate, strong, vote_fraction)


# --------------------------------------------------------------------------
# Per-(device, operating point) summary row -- mirrors
# raw_online_federated_v42._summarize_point's fields plus the config/z/epc
# tags V43 needs.
# --------------------------------------------------------------------------
def summarize_point(
    client_id: str, dataset: str, config: str, z_nominal: float, epc: bool, m: int,
    persistent: np.ndarray, raw_exceed: np.ndarray, indices: np.ndarray, n: int,
) -> dict:
    """``persistent``/``raw_exceed`` span the eval window ``[guard.start, n)``
    only (matching raw_online_federated_v42's own convention, NOT the full
    ``[0, n)`` stream) -- the first ``guard_count`` elements are the guard,
    everything after is the operational region."""
    guard = guard_slice(m)
    guard_count = guard.stop - guard.start
    persistent = np.asarray(persistent, dtype=np.int8)
    raw_exceed = np.asarray(raw_exceed, dtype=bool)
    if len(persistent) != len(raw_exceed):
        raise ValueError(f"persistent/raw_exceed length mismatch for {client_id} {config} z={z_nominal} m={m}")
    if len(persistent) != n - guard.start:
        raise ValueError(f"eval-window length mismatch for {client_id} {config} z={z_nominal} m={m}")
    guard_primary = persistent[:guard_count]
    guard_raw = raw_exceed[:guard_count]
    operation = persistent[guard_count:]
    op_indices = indices[op_start(m):]
    if len(operation) != len(op_indices):
        raise ValueError(f"chronology mismatch for {client_id} {config} z={z_nominal} m={m}")
    positions = np.flatnonzero(operation)
    first_alarm = int(op_indices[positions[0]]) if len(positions) else -1
    return {
        "client_id": client_id,
        "dataset": dataset,
        "config": config,
        "nominal_z": float(z_nominal),
        "epc": bool(epc),
        "calibration_points": int(m),
        "guard_count": int(guard_count),
        "guard_alarm": bool(np.any(guard_primary)),
        "guard_exceedances": int(np.sum(guard_raw)),
        "first_alarm": first_alarm,
        "operation_acquisitions": int(len(operation)),
        "acquisitions": int(n),
    }


# --------------------------------------------------------------------------
# Top-level per-device computation
# --------------------------------------------------------------------------
def compute_device_predictions(
    dataset: str, unit: str, client_id: str,
    values: np.ndarray, indices: np.ndarray, bundle: dict,
    weights: Mapping[str, float], strong: float, vote_fraction: float, epc_seed: int,
) -> list[dict]:
    n = len(values)
    rows: list[dict] = []
    raw_fed = v42.raw_branch_scores(values, bundle)
    raw_local = {LOCAL_BRANCH: local_raw_score(values)}
    local_weights = {LOCAL_BRANCH: 1.0}
    fed_pca_weights = {FED_PCA_BRANCH: 1.0}

    # Leg 1 (threshold grid) + Leg 2 (F-G/F-L/L-L, SS2.2) + the SS2.2.1
    # matched-algorithm pair (F-L(PCA)/L-L(PCA)), primary m=64.
    if n >= required_acquisitions(PRIMARY_M):
        z_fed_64 = zscores_at(raw_fed, PRIMARY_M, n)
        z_local_64 = zscores_at(raw_local, PRIMARY_M, n)
        for z in Z_GRID:
            _, persistent, raw_exceed = decide(z_fed_64, weights, float(z), strong, vote_fraction)
            rows.append(summarize_point(client_id, dataset, "F-G", z, False, PRIMARY_M, persistent, raw_exceed, indices, n))

            mod_fed = epc_moderate(raw_fed, calib.BRANCHES, PRIMARY_M, z, client_id, epc_seed)
            _, persistent, raw_exceed = decide(z_fed_64, weights, mod_fed, strong, vote_fraction)
            rows.append(summarize_point(client_id, dataset, "F-L", z, True, PRIMARY_M, persistent, raw_exceed, indices, n))

            mod_local = epc_moderate(raw_local, (LOCAL_BRANCH,), PRIMARY_M, z, client_id, epc_seed)
            _, persistent, raw_exceed = decide(z_local_64, local_weights, mod_local, strong, vote_fraction)
            rows.append(summarize_point(client_id, dataset, "L-L", z, True, PRIMARY_M, persistent, raw_exceed, indices, n))

            # SS2.2.1 matched-algorithm pair: FED_PCA16 alone vs local
            # PCA-16 alone, both single-branch, both device-local EPC.
            # Reuses mod_fed[FED_PCA_BRANCH] (already computed above as
            # part of F-L's three-branch adjustment -- the EPC seed formula
            # depends only on (client_id, branch, m), not on which other
            # branches are also adjusted, so this is not an approximation)
            # and z_fed_64[FED_PCA_BRANCH] rather than recomputing either.
            z_fed_pca_only = {FED_PCA_BRANCH: z_fed_64[FED_PCA_BRANCH]}
            mod_fed_pca_only = {FED_PCA_BRANCH: mod_fed[FED_PCA_BRANCH]}
            _, persistent, raw_exceed = decide(z_fed_pca_only, fed_pca_weights, mod_fed_pca_only, strong, vote_fraction)
            rows.append(summarize_point(client_id, dataset, "F-L(PCA)", z, True, PRIMARY_M, persistent, raw_exceed, indices, n))

            # L-L(PCA) reuses L-L's own z-scores/threshold outright -- L-L
            # IS local PCA-16 (SPEC-AMBIGUITY 1), so this is numerically
            # identical to the L-L row above, computed under its own config
            # label so the matched-pair table is self-describing.
            _, persistent, raw_exceed = decide(z_local_64, local_weights, mod_local, strong, vote_fraction)
            rows.append(summarize_point(client_id, dataset, "L-L(PCA)", z, True, PRIMARY_M, persistent, raw_exceed, indices, n))

    # Leg 3 (m sensitivity), F-L only, fixed common cohort enforced by the
    # caller (this function computes rows only when eligible at EVERY size).
    if n >= required_acquisitions(max(SENSITIVITY_M_GRID)):
        for m in SENSITIVITY_M_GRID:
            z_fed_m = zscores_at(raw_fed, m, n)
            mod_fed_m = epc_moderate(raw_fed, calib.BRANCHES, m, SENSITIVITY_Z, client_id, epc_seed)
            _, persistent, raw_exceed = decide(z_fed_m, weights, mod_fed_m, strong, vote_fraction)
            rows.append(
                summarize_point(client_id, dataset, SENSITIVITY_CONFIG, SENSITIVITY_Z, True, m, persistent, raw_exceed, indices, n)
            )

    return rows


__all__ = [
    "CONFIGS",
    "DEPLOYMENT_CONFIGS",
    "EPC_B",
    "EPC_P",
    "FED_PCA_BRANCH",
    "LOCAL_BRANCH",
    "MATCHED_PAIR_CONFIGS",
    "PRIMARY_M",
    "SENSITIVITY_CONFIG",
    "SENSITIVITY_M_GRID",
    "SENSITIVITY_Z",
    "Z_GRID",
    "compute_device_predictions",
    "decide",
    "epc_moderate",
    "guard_slice",
    "local_raw_score",
    "op_start",
    "postcal_slice",
    "required_acquisitions",
    "summarize_point",
    "zscores_at",
]
