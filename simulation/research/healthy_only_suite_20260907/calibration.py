"""Healthy-only, causal calibration policies for the 2026-09-07 suite.

This module deliberately has no outcome/truth inputs.  Calibration is fitted
from the supplied training/calibration indices and prediction is chronological.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


WINDOW = 5
K = 3


def _indices(value: Sequence[int], n: int, name: str) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer index sequence")
    a = a.astype(int, copy=False)
    if len(a) == 0 or np.any(a < 0) or np.any(a >= n):
        raise ValueError(f"{name} contains an out-of-range or empty index")
    if len(np.unique(a)) != len(a):
        raise ValueError(f"{name} contains duplicate indices")
    return a


def _rolling_kth(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), -np.inf, dtype=float)
    if len(values) >= WINDOW:
        windows = np.lib.stride_tricks.sliding_window_view(values, WINDOW)
        out[WINDOW - 1 :] = np.partition(windows, WINDOW - K, axis=1)[:, WINDOW - K]
    return out


def _mad(values: np.ndarray) -> float:
    med = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - med)))


def predict(scores, train_indices, calibration_indices, mode) -> dict:
    """Fit a healthy-only policy and return full-length causal predictions.

    ``mode`` is one of ``point995``, ``block_guard``, or ``anchored_drift``.
    Thresholds are indexed by the original score stream.  Flags are suppressed
    before prediction starts and through the final calibration index.
    """
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("scores must be one-dimensional finite nonnegative values")
    n = len(x)
    train = _indices(train_indices, n, "train_indices")
    cal = _indices(calibration_indices, n, "calibration_indices")
    if np.intersect1d(train, cal).size:
        raise ValueError("training and calibration indices must be disjoint")
    if np.max(train) >= np.min(cal):
        raise ValueError("training indices must precede calibration indices")
    if len(cal) > 1 and not np.all(np.diff(cal) == 1):
        raise ValueError("calibration indices must be sorted and contiguous")
    if mode not in {"point995", "block_guard", "anchored_drift"}:
        raise ValueError(f"unknown calibration mode: {mode}")

    cal_values = x[cal]
    initial = float(np.quantile(cal_values, 0.995))
    cal_start, cal_end = int(np.min(cal)), int(np.max(cal))
    # The calibration event stream follows chronological calibration order.
    cal_event = _rolling_kth(cal_values)
    finite_event = cal_event[np.isfinite(cal_event)]
    blockmax = np.asarray(
        [np.max(finite_event[i : i + WINDOW])
         for i in range(0, len(finite_event) - WINDOW + 1, WINDOW)], dtype=float
    )
    insufficient = False
    if mode == "block_guard":
        if len(blockmax) < 4:
            threshold = initial
            insufficient = True
        else:
            center = float(np.median(blockmax))
            threshold = max(initial, float(np.quantile(blockmax, 0.95)) + _mad(blockmax))
    else:
        threshold = initial

    thresholds = np.full(n, threshold, dtype=float)
    flags = np.zeros(n, dtype=bool)
    adaptation_updates = 0
    stream = x[cal_start:]
    stats = _rolling_kth(stream)

    if mode != "anchored_drift":
        eligible = np.arange(cal_start, n) > cal_end
        flags[cal_start:] = (stats > threshold) & eligible
    else:
        zcal = np.log1p(np.maximum(cal_values, 0.0))
        baseline = float(np.median(zcal))
        baseline_mad = max(1.4826 * float(np.median(np.abs(zcal - baseline))), 1e-6)
        ewma = baseline
        prev_stat = -np.inf
        max_cap = 2.0 * initial
        for j, stat in enumerate(stats):
            t = cal_start + j
            current_threshold = max(initial, np.expm1(np.log1p(initial) + max(0.0, ewma - baseline)))
            if initial > 0:
                current_threshold = min(current_threshold, max_cap)
            else:
                current_threshold = 0.0
            thresholds[t] = float(current_threshold)
            alarm = bool(np.isfinite(stat) and stat > current_threshold)
            if t > cal_end:
                flags[t] = alarm
                z = float(np.log1p(max(x[t], 0.0)))
                if not alarm and not (np.isfinite(prev_stat) and prev_stat > current_threshold) and z <= baseline + 2 * baseline_mad:
                    ewma = float(np.clip(0.99 * ewma + 0.01 * z, baseline - baseline_mad, baseline + baseline_mad))
                    adaptation_updates += 1
            prev_stat = stat

    metadata = {
        "mode": mode,
        "calibration_start": cal_start,
        "calibration_end": cal_end,
        "training_count": int(len(train)),
        "calibration_count": int(len(cal)),
        "initial_point995": initial,
        "insufficient": bool(insufficient),
        "block_length": WINDOW,
        "block_count": int(len(blockmax)),
        "max_cap": float(2.0 * initial) if mode == "anchored_drift" else None,
        "static_threshold": None if mode == "anchored_drift" else float(threshold),
        "adaptation_update_count": int(adaptation_updates),
        "adaptation_guarantee": "none; anchored drift may absorb a slow fault",
        "finite_input_validated": True,
    }
    if mode == "anchored_drift":
        metadata.update({"alpha": 0.01, "baseline_median_log1p": baseline, "baseline_mad_log1p": baseline_mad})
    return {"flags": flags, "thresholds": thresholds, "metadata": metadata}
