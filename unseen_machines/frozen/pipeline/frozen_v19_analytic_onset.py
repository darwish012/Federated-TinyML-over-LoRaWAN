"""Executable copy of the frozen V19 analytic vibration-departure proxy.

The formulas and constants in this module are transcribed from the locked
``bearing_eda_deep_analysis.ipynb`` onset cell.  This is a retrospective label
generator, not a deployable detector and not verified physical-damage truth.
It must never consume V64/V67 scores, alarms, states, or thresholds.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


FEATURES = (
    "rms_mean",
    "kurtosis_mean",
    "crest_factor_mean",
    "spectral_entropy_mean",
    "band_4_relative_power_mean",
)

CONFIG = {
    "fit_count": 40,
    "commissioning_count": 100,
    "reference_quantile": 0.99,
    "reference_mad_multiplier": 6.0,
    "persistence_k": 3,
    "persistence_m": 5,
    "smoothing_max_window": 21,
    "sudden_final_life_fraction": 0.20,
    "gradual_post_onset_spearman": 0.35,
    "gradual_post_onset_exceedance_fraction": 0.45,
    "commissioning_rms_change_limit": 0.25,
    "change_point_window_fraction": 0.05,
    "change_point_window_min": 5,
    "change_point_window_max": 50,
    "winsorized_local_z": 20.0,
}


def _robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(values - center), axis=0)
    q25 = np.nanquantile(values, 0.25, axis=0)
    q75 = np.nanquantile(values, 0.75, axis=0)
    iqr_scale = (q75 - q25) / 1.349
    standard_deviation = np.nanstd(values, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1e-12), mad, iqr_scale)
    scale = np.where(
        np.isfinite(scale) & (scale > 1e-12), scale, standard_deviation
    )
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return center, scale


def _safe_spearman(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.nanstd(values) <= 1e-12:
        return math.nan
    result = stats.spearmanr(
        np.arange(len(values), dtype=float), values, nan_policy="omit"
    )
    return float(result.statistic)


def _first_k_of_m(flags: np.ndarray, start: int, k: int, m: int) -> int | None:
    flags = np.asarray(flags, dtype=bool)
    if len(flags) < m:
        return None
    counts = (
        pd.Series(flags.astype(int))
        .rolling(m, min_periods=m)
        .sum()
        .to_numpy()
    )
    candidates = np.flatnonzero(
        (np.arange(len(flags)) >= max(start, m - 1)) & (counts >= k)
    )
    if not len(candidates):
        return None
    end = int(candidates[0])
    window_start = end - m + 1
    exceedances = np.flatnonzero(flags[window_start : end + 1])
    return int(window_start + exceedances[0])


def _strongest_positive_change(
    score: np.ndarray, start: int
) -> tuple[int | None, float, int]:
    score = np.asarray(score, dtype=float)
    n = len(score)
    window = int(
        np.clip(
            round(n * CONFIG["change_point_window_fraction"]),
            CONFIG["change_point_window_min"],
            CONFIG["change_point_window_max"],
        )
    )
    lower = max(int(start), window)
    upper = n - window
    if upper <= lower:
        return None, math.nan, window
    best_index: int | None = None
    best_delta = -math.inf
    for index in range(lower, upper + 1):
        before = np.nanmedian(score[index - window : index])
        after = np.nanmedian(score[index : index + window])
        delta = float(after - before)
        if np.isfinite(delta) and delta > best_delta:
            best_delta = delta
            best_index = index
    if best_index is None or best_delta <= 0:
        return None, float(best_delta), window
    return int(best_index), float(best_delta), window


def estimate(features: pd.DataFrame) -> dict[str, Any]:
    """Return the exact V19 proxy-onset diagnostics for one ordered lifecycle."""

    missing = [name for name in FEATURES if name not in features.columns]
    if missing:
        raise KeyError(f"Missing frozen V19 onset features: {missing}")
    n = int(len(features))
    minimum = CONFIG["commissioning_count"] + CONFIG["persistence_m"]
    if n < minimum:
        return {
            "analysis_status": "insufficient_post_commission_data",
            "observed_acquisitions": n,
            "minimum_required": minimum,
            "onset_index": None,
        }

    # Deliberately identical to V19. This is retrospective: bidirectional
    # interpolation may use a later value to fill an earlier missing value.
    matrix = (
        features.loc[:, FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .to_numpy(dtype=float)
    )
    center, scale = _robust_center_scale(matrix[: CONFIG["fit_count"]])
    local_z = (matrix - center) / scale
    local_z = np.clip(
        local_z,
        -CONFIG["winsorized_local_z"],
        CONFIG["winsorized_local_z"],
    )
    point_distance = np.sqrt(np.nanmean(local_z**2, axis=1))
    smoothing_window = int(
        np.clip(
            math.ceil(n * 0.01), 3, CONFIG["smoothing_max_window"]
        )
    )
    smoothed_distance = (
        pd.Series(point_distance)
        .rolling(smoothing_window, min_periods=1)
        .median()
        .to_numpy()
    )
    reference = smoothed_distance[: CONFIG["commissioning_count"]]
    reference_median = float(np.nanmedian(reference))
    reference_mad = float(
        1.4826 * np.nanmedian(np.abs(reference - reference_median))
    )
    quantile_threshold = float(
        np.nanquantile(reference, CONFIG["reference_quantile"])
    )
    robust_threshold = float(
        reference_median
        + CONFIG["reference_mad_multiplier"] * max(reference_mad, 1e-12)
    )
    threshold = max(quantile_threshold, robust_threshold)
    exceedance = smoothed_distance > threshold
    onset_index = _first_k_of_m(
        exceedance,
        CONFIG["commissioning_count"],
        CONFIG["persistence_k"],
        CONFIG["persistence_m"],
    )
    change_index, change_delta, change_window = _strongest_positive_change(
        smoothed_distance, CONFIG["commissioning_count"]
    )
    fit_rms = float(np.nanmedian(matrix[: CONFIG["fit_count"], 0]))
    guard_rms = float(np.nanmedian(matrix[80:100, 0]))
    commissioning_rms_change = float(
        guard_rms / max(abs(fit_rms), 1e-12) - 1.0
    )
    commissioning_unstable = bool(
        abs(commissioning_rms_change)
        > CONFIG["commissioning_rms_change_limit"]
    )

    result: dict[str, Any] = {
        "analysis_status": (
            "no_sustained_departure" if onset_index is None else "onset_estimated"
        ),
        "observed_acquisitions": n,
        "onset_index": onset_index,
        "onset_fraction": (
            None if onset_index is None else float((onset_index + 1) / n)
        ),
        "device_local_threshold": threshold,
        "threshold_quantile_arm": quantile_threshold,
        "threshold_robust_mad_arm": robust_threshold,
        "smoothing_window": smoothing_window,
        "commissioning_rms_change": commissioning_rms_change,
        "commissioning_unstable": commissioning_unstable,
        "strongest_change_index": change_index,
        "strongest_change_delta": change_delta,
        "change_point_window": change_window,
        "trajectory_regime": "no_sustained_departure",
        "onset_confidence": "low",
        "post_onset_spearman": None,
        "post_onset_exceedance_fraction": None,
        "healthy_degraded_centroid_distance": None,
    }
    if onset_index is None:
        return result

    onset_fraction = float((onset_index + 1) / n)
    post_score = smoothed_distance[onset_index:]
    post_spearman = _safe_spearman(post_score)
    post_exceedance_fraction = float(np.mean(exceedance[onset_index:]))
    healthy_centroid = np.nanmedian(local_z[: CONFIG["fit_count"]], axis=0)
    degraded_centroid = np.nanmedian(local_z[onset_index:], axis=0)
    centroid_distance = float(
        np.sqrt(np.nanmean((degraded_centroid - healthy_centroid) ** 2))
    )
    if commissioning_unstable and onset_fraction < 0.25:
        regime = "commissioning_shift_or_early_change"
    elif onset_fraction >= 1.0 - CONFIG["sudden_final_life_fraction"]:
        regime = "sudden_terminal"
    elif (
        np.isfinite(post_spearman)
        and post_spearman >= CONFIG["gradual_post_onset_spearman"]
        and post_exceedance_fraction
        >= CONFIG["gradual_post_onset_exceedance_fraction"]
    ):
        regime = "gradual_sustained"
    else:
        regime = "mixed_bursty"

    gap_fraction = (
        None
        if change_index is None
        else float(abs(change_index - onset_index) / max(n - 1, 1))
    )
    if change_index is None:
        confidence = "low"
    elif (
        gap_fraction is not None
        and gap_fraction <= 0.10
        and post_exceedance_fraction >= 0.60
        and not commissioning_unstable
    ):
        confidence = "high"
    elif gap_fraction is not None and gap_fraction <= 0.20:
        confidence = "medium"
    else:
        confidence = "low"
    result.update(
        {
            "trajectory_regime": regime,
            "onset_confidence": confidence,
            "post_onset_spearman": post_spearman,
            "post_onset_exceedance_fraction": post_exceedance_fraction,
            "healthy_degraded_centroid_distance": centroid_distance,
            "onset_change_point_gap_fraction": gap_fraction,
        }
    )
    return result


__all__ = ["CONFIG", "FEATURES", "estimate"]
