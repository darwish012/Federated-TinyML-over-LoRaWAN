"""V42 calibration windows, EPC-adjusted limits, and self-starting standardization.

Implements V42_PREREGISTRATION.md Section 3 (the EPC bootstrap) and Section 4
(the eligibility ledger), plus the shared decision primitives (ensemble +
4-of-7 latch) that every arm's calibration output feeds into. This module is
pure numerics: no file I/O, no model training, no label or onset access, no
target data. It is imported by ``raw_online_federated_v42.py`` (the phase
driver, which supplies raw branch scores read from V41R1's frozen streams and
models) and exercised directly by ``raw_online_federated_v42_calibration_tests.py``.

The commissioning schedule is inherited unchanged from V41/V41R1 for its
fixed parts:

    INPUT_CAL  [0, 8)     input feature normalization only
    FIT        [8, 20)    local-PCA comparator fitting only
    POSTCAL(m) [20, 20+m) score-calibration window -- m is what V42 varies
    GUARD      [20+m, 20+m+4)  4 held-out acquisitions, fixed size
    OP_START(m) = 20 + m + 4

For m = 8 this reproduces V41R1's fixed 8-12-8-4 schedule exactly (arm
A_V41R1_BASELINE and arm B_CALIB_SWEEP at calibration size 8 are therefore
the same computation -- see raw_online_federated_v42.py).

Persistence is the 4-of-7 latch throughout (V42_PREREGISTRATION.md SS2). The
guard window is only 4 acquisitions long, one short of the 7 needed for the
persistence condition to ever be evaluated at all: this means the guard
segment is *structurally* incapable of latching regardless of score
behaviour, for every arm. That is not a V42 defect -- it reproduces exactly
why V41R1's guard was "already clean" -- but it is reported honestly rather
than presented as a validated safety property.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

import raw_online_federated_v25_policy as policy
import raw_online_federated_v29_models as models

# --------------------------------------------------------------------------
# Fixed commissioning geometry
# --------------------------------------------------------------------------
INPUT_CAL = slice(0, 8)
FIT = slice(8, 20)
FIT_END = 20
GUARD_SIZE = 4
CALIBRATION_SIZES: tuple[int, ...] = (8, 16, 32, 64, 128)
BRANCHES: tuple[str, ...] = ("FED_AE_RECON", "FED_PCA16", "FED_DEEP_SVDD")
LATCH = policy.LatchConfig(persistence_k=4, persistence_window=7)

# EPC bootstrap defaults (V42_PREREGISTRATION.md SS3)
EPC_B = 2000
EPC_P = 0.10

# Self-starting defaults (arm D_SELF_STARTING).
#
# SPEC-AMBIGUITY: the preregistration states only "no fixed calibration
# window; each observation standardized against all prior observations
# (Hawkins & Olwell self-starting)". It does not say where the running
# reference pool begins or how many prior points are required before the
# first evaluation. We take the most literal reading consistent with every
# fixed-window arm's shared 8-input/12-fit prefix: the growing reference
# pool begins at FIT_END (index 20 -- the same acquisition POSTCAL begins at
# for every fixed-window arm), and a point is evaluated only once at least
# SELF_START_MIN_HISTORY prior points exist in the pool (the minimum for a
# nondegenerate median/MAD). GUARD keeps its fixed 4-acquisition size and
# position [20, 24) so the G2 guard metric stays comparable across arms;
# OP_START is therefore 24 (i.e. m = 0, no separate waiting period).
SELF_START_MIN_HISTORY = 2
SELF_START_GUARD = slice(FIT_END, FIT_END + GUARD_SIZE)
SELF_START_OP_START = FIT_END + GUARD_SIZE


def postcal_slice(m: int) -> slice:
    """Score-calibration window [20, 20+m) for a B/C-arm calibration size."""
    if int(m) not in CALIBRATION_SIZES:
        raise ValueError(f"unsupported calibration size: {m}")
    return slice(FIT_END, FIT_END + int(m))


def guard_slice(m: int) -> slice:
    start = FIT_END + int(m)
    return slice(start, start + GUARD_SIZE)


def op_start(m: int) -> int:
    return FIT_END + int(m) + GUARD_SIZE


# --------------------------------------------------------------------------
# Persistence latch -- delegated to the frozen V25 primitive, never
# reimplemented (V42 task scope instruction).
# --------------------------------------------------------------------------
def k_of_m_latch(flags: np.ndarray) -> np.ndarray:
    """4-of-7 persistent latch over an already-boolean condition stream.

    Delegates to :func:`raw_online_federated_v25_policy.latch_stream`, fed a
    synthetic 0.0/1.0 "score" stream against threshold 0.5, so the
    windowed-count code path used throughout V25-V38 governs V42's
    persistence decision too (V42's own task scope explicitly forbids
    reimplementing the latch primitives).

    NOTE on a real semantic difference from ``raw_online_federated_v41.k_of_m``:
    V25's ``latch_stream`` is a PERMANENT latch -- once the k-of-m condition
    has held once, every later element is also True. V41's own ``k_of_m`` is
    a plain windowed condition that CAN return to False later if a
    subsequent trailing window no longer satisfies k-of-m. The two therefore
    are not elementwise-identical streams in general. They are, however,
    guaranteed to have the same FIRST True position (both ask "does this
    trailing window of length m have >= k hits", evaluated identically at
    every t), and every V42 metric (first_alarm, guard_alarm) only ever
    consults the first True position -- see
    raw_online_federated_v42_calibration_tests.EnsembleEquivalenceTests for
    the equivalence proof this reproduction relies on.
    """
    values = np.asarray(flags, dtype=np.float64).reshape(-1).astype(np.float64)
    return policy.latch_stream(values, 0.5, LATCH)


# --------------------------------------------------------------------------
# Section 3: the EPC-adjusted limit (arm C)
# --------------------------------------------------------------------------
def robust_limit(calibration: np.ndarray, moderate: float) -> tuple[float, float, float]:
    """(center, scale, raw-score limit) for one branch's calibration sample."""
    center, scale = models.scalar_center_scale(calibration)
    return float(center), float(scale), float(center) + float(moderate) * float(scale)


def epc_adjusted_threshold(
    calibration: np.ndarray,
    moderate: float,
    *,
    seed: int,
    b: int = EPC_B,
    p: float = EPC_P,
) -> dict[str, float]:
    """Bootstrap-adjusted z-threshold for one branch, one device (SS3).

    Nonparametric bootstrap: draw ``b`` resamples of size ``m = len(calibration)``
    WITH replacement from the calibration sample, recompute the robust
    location/scale and the resulting raw-score limit ``center* + moderate *
    scale*`` on each resample, and deploy the ``(1 - p)`` quantile of that
    bootstrap limit distribution. The deployed limit is then re-expressed as
    a z-threshold against the device's own (non-resampled) calibration
    center/scale, so every downstream step (ensemble, latch) is unchanged
    except for which z-threshold each branch compares against.

    This targets a DEVICE-LEVEL false-alarm probability, not a
    per-observation alpha: V38 set alpha = 0.05/0.10 per observation, which
    at alpha = 0.05 is a choice to false-alarm roughly every 20 observations.
    V42 instead asks for <= 5% probability of any false alarm over a
    device's own operational length.
    """
    cal = np.asarray(calibration, dtype=np.float64).reshape(-1)
    m = len(cal)
    if m < 2:
        raise ValueError("EPC bootstrap requires at least 2 calibration points")
    if not np.all(np.isfinite(cal)):
        raise ValueError("calibration sample must be finite")
    if not 0.0 < float(p) < 1.0:
        raise ValueError("p must be in (0, 1)")
    if int(b) < 1:
        raise ValueError("b must be a positive integer")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, m, size=(int(b), m))
    resampled = cal[draws]
    center_star = np.median(resampled, axis=1)
    mad_star = 1.4826 * np.median(np.abs(resampled - center_star[:, None]), axis=1)
    std_star = np.std(resampled, axis=1)
    scale_star = np.where(np.isfinite(mad_star) & (mad_star > 1e-6), mad_star, std_star)
    scale_star = np.where(np.isfinite(scale_star) & (scale_star > 1e-6), scale_star, 1e-6)
    limit_star = center_star + float(moderate) * scale_star
    deployed_limit = float(np.quantile(limit_star, 1.0 - float(p)))
    center, scale, nominal_limit = robust_limit(cal, moderate)
    adjusted_z_threshold = (deployed_limit - center) / scale
    return {
        "center": center,
        "scale": scale,
        "nominal_moderate_z": float(moderate),
        "nominal_limit": nominal_limit,
        "deployed_limit": deployed_limit,
        "adjusted_moderate_z": float(adjusted_z_threshold),
        "bootstrap_draws": int(b),
        "p": float(p),
        "m": int(m),
    }


# --------------------------------------------------------------------------
# Arm D: self-starting standardization
# --------------------------------------------------------------------------
def self_starting_z(
    raw: np.ndarray,
    start: int = FIT_END,
    min_history: int = SELF_START_MIN_HISTORY,
) -> np.ndarray:
    """Expanding-window robust z: standardize raw[t] against raw[start:t] only.

    Hawkins & Olwell self-starting: there is no fixed calibration window.
    The reference pool grows to include every prior observation regardless
    of that observation's own status (contamination included -- a known,
    expected property of self-starting charts, not a bug introduced here).
    Points before ``start + min_history`` are undefined (NaN) because fewer
    than ``min_history`` prior points exist yet.

    Reuses :func:`raw_online_federated_v29_models.robust_z` for the actual
    center/scale/clip computation at each step, so the numeric convention
    (median / 1.4826*MAD, clipped to [-12, 30]) is identical to every other
    arm's per-branch z-score, not a parallel formula.
    """
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    n = len(values)
    z = np.full(n, np.nan, dtype=np.float64)
    for t in range(int(start) + int(min_history), n):
        z[t] = float(models.robust_z(values[start:t], values[t : t + 1])[0])
    return z


# --------------------------------------------------------------------------
# Ensemble decision (V29b weighted moderate/strong rule), generalized to
# accept a per-branch moderate threshold for arm C.
# --------------------------------------------------------------------------
def ensemble_decision(
    zscores: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    moderate: Mapping[str, float] | float,
    strong: float,
    vote_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(continuous score, persistent alarm, raw exceedance) for the frozen V29b ensemble.

    Identical to ``raw_online_federated_v41.ensemble()`` when ``moderate`` is
    a single scalar shared by every branch (arms A/B/D); arm C instead
    passes a per-branch mapping of EPC-adjusted z-thresholds. ``strong`` is
    never adjusted by V42 -- only the moderate limit is the object of study
    (V42_PREREGISTRATION.md SS3); ``strong`` stays the frozen V29b value
    throughout, matching "Thresholds stay at V41R1's 2/8 except where arm C
    adjusts them" (SS2).

    ``raw`` is the PRE-persistence ensemble condition at each acquisition
    (strong OR moderate-vote), i.e. "does this acquisition exceed the
    deployed limit" with no 4-of-7 requirement applied. V41's own
    ``ensemble()`` computes this internally but never returns it; V42 must
    return it because it is exactly what the redefined G2 (per-acquisition
    guard exceedance rate, V42_PREREGISTRATION.md SS5 "G2 erratum") needs --
    the latch-based ``persistent`` guard figure is structurally zero at this
    geometry (guard=4 < persistence_window=7) and is not informative.
    """
    names = tuple(weights)
    w = np.asarray([weights[name] for name in names], dtype=np.float64)
    matrix = np.stack([np.asarray(zscores[name], dtype=np.float64) for name in names])
    if isinstance(moderate, Mapping):
        mod_arr = np.asarray([float(moderate[name]) for name in names], dtype=np.float64)
    else:
        mod_arr = np.full(len(names), float(moderate), dtype=np.float64)
    moderate_support = np.sum(w[:, None] * (matrix >= mod_arr[:, None]), axis=0)
    reliable = w >= 0.5 / len(names)
    strong_flag = np.any((matrix >= float(strong)) & reliable[:, None], axis=0)
    raw = strong_flag | (moderate_support >= float(vote_fraction))
    persistent = k_of_m_latch(raw)
    weighted = np.sum(w[:, None] * np.maximum(matrix, 0.0) / mod_arr[:, None], axis=0)
    strong_level = np.max(np.where(reliable[:, None], matrix / float(strong), -np.inf), axis=0)
    continuous = np.maximum(weighted, strong_level)
    return continuous, persistent, raw


def guard_latch_is_structurally_vacuous(
    guard_length: int, persistence_window: int = LATCH.persistence_window
) -> dict:
    """Required structural check (V42_PREREGISTRATION.md SS5, "G2 erratum").

    A persistence window of length ``persistence_window`` needs that many
    consecutive evaluated acquisitions before the k-of-m condition can ever
    hold even once. A guard of length ``guard_length`` contains
    ``guard_length - persistence_window + 1`` complete trailing windows.
    When that count is <= 0, the guard cannot latch at any threshold, for
    any arm, at this geometry -- a reported latch-based guard false-alarm
    rate of 0% is then arithmetically forced, not measured safety. This is
    the defect the erratum identifies in V41R1's own "guard was already
    clean" finding (and in V19's XJTU split, guard 0 acquisitions, rule
    10-of-12): both were vacuous by the same arithmetic, not validated.

    Must fail loudly, not silently emit a zero rate -- callers (the audit
    phase) print a prominent warning and record the result rather than
    letting a vacuous 0% pass unremarked.
    """
    complete_windows = int(guard_length) - int(persistence_window) + 1
    return {
        "guard_length": int(guard_length),
        "persistence_window": int(persistence_window),
        "complete_windows": complete_windows,
        "latch_based_guard_metrics_vacuous": bool(complete_windows <= 0),
    }


# --------------------------------------------------------------------------
# Section 4: eligibility ledger
# --------------------------------------------------------------------------
CALIBRATION_LABELS: tuple[str, ...] = tuple(str(m) for m in CALIBRATION_SIZES) + ("self_starting",)


def required_acquisitions(label: str) -> int:
    """Minimum stream length for one operational acquisition to exist."""
    if label == "self_starting":
        return SELF_START_OP_START + 1
    return op_start(int(label)) + 1


def eligibility_rows(stream_lengths: Mapping[str, int]) -> list[dict]:
    """Section 4 eligibility ledger. Counts only -- no status, EoL, or onset.

    For each calibration label (five fixed sizes plus self-starting) and
    every device, records whether the device's stream is long enough to
    supply that calibration window AND at least one post-commissioning
    operational acquisition. A device excluded at one label but present at
    another appears once per label -- never silently dropped.
    """
    rows = []
    for client_id, n in stream_lengths.items():
        dataset = client_id.split("::", 1)[0]
        for label in CALIBRATION_LABELS:
            required = required_acquisitions(label)
            eligible = int(n) >= required
            rows.append(
                {
                    "client_id": client_id,
                    "dataset": dataset,
                    "calibration_label": label,
                    "acquisitions": int(n),
                    "required_acquisitions": int(required),
                    "eligible": bool(eligible),
                    "reason": "eligible" if eligible else f"acquisitions {int(n)} < required {required}",
                }
            )
    return rows


__all__ = [
    "BRANCHES",
    "CALIBRATION_LABELS",
    "CALIBRATION_SIZES",
    "EPC_B",
    "EPC_P",
    "FIT",
    "FIT_END",
    "GUARD_SIZE",
    "INPUT_CAL",
    "LATCH",
    "SELF_START_GUARD",
    "SELF_START_MIN_HISTORY",
    "SELF_START_OP_START",
    "eligibility_rows",
    "ensemble_decision",
    "epc_adjusted_threshold",
    "guard_latch_is_structurally_vacuous",
    "guard_slice",
    "k_of_m_latch",
    "op_start",
    "postcal_slice",
    "required_acquisitions",
    "robust_limit",
    "self_starting_z",
]
