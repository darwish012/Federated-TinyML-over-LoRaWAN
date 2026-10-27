"""Causal phase-sampled temporal alarm comparison (no outcome access)."""
from __future__ import annotations

import math
import numpy as np


def _idx(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    a = a.astype(int, copy=False)
    if len(a) == 0 or np.any(a < 0) or np.any(np.diff(a) <= 0):
        raise ValueError(f"{name} must be nonempty and strictly chronological")
    return a


def _validate(scores, train, cal):
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("scores must be finite, one-dimensional, nonnegative")
    if np.any(train >= len(x)) or np.any(cal >= len(x)):
        raise ValueError("indices out of range")
    if np.intersect1d(train, cal).size or np.max(train) >= np.min(cal):
        raise ValueError("training must precede and be disjoint from calibration")
    return x


def stride_from_history(train_indices, cal_indices, history_budget=100):
    train, cal = _idx(train_indices, "train_indices"), _idx(cal_indices, "cal_indices")
    if history_budget <= 0:
        raise ValueError("history_budget must be positive")
    if np.max(train) >= np.min(cal):
        raise ValueError("training must precede calibration")
    initial = max(1, math.ceil((int(np.max(cal)) + 1) / history_budget))
    stride = initial
    while stride > 1:
        nt = int(np.sum(train % stride == 0))
        nc = int(np.sum(cal % stride == 0))
        if nt >= 30 and nc >= 10:
            break
        stride = max(1, stride // 2)
    return int(stride)


def _rolling(values, window, k):
    out = np.full(len(values), -np.inf)
    if len(values) >= window:
        w = np.lib.stride_tricks.sliding_window_view(values, window)
        out[window - 1:] = np.partition(w, window - k, axis=1)[:, window - k]
    return out


def phase_vote(scores, train_indices, cal_indices, stride, window=21, k=13,
               multiplier=2.0, phases=5, votes=2):
    train, cal = _idx(train_indices, "train_indices"), _idx(cal_indices, "cal_indices")
    x = _validate(scores, train, cal)
    if (not isinstance(stride, (int, np.integer)) or not isinstance(window, (int, np.integer))
            or not isinstance(k, (int, np.integer)) or not isinstance(phases, (int, np.integer))
            or not isinstance(votes, (int, np.integer)) or stride < 1 or window < 1
            or not 1 <= k <= window or multiplier <= 0 or phases < 1 or votes < 1):
        raise ValueError("invalid cadence parameters")
    phase_count = min(int(phases), int(stride))
    offsets = np.unique(np.floor(np.arange(phase_count) * stride / phase_count).astype(int))
    required = min(int(votes), len(offsets))
    cal_end = int(np.max(cal))
    original_q = float(np.quantile(x[cal], .995))
    phase_refs, phase_decisions, phase_alarmcounts = [], [], []
    held = np.zeros((len(offsets), len(x)), dtype=bool)
    for p, offset in enumerate(offsets):
        retained = np.arange(offset, len(x), stride, dtype=int)
        hist = retained[np.isin(retained, np.concatenate((train, cal)))]
        hc = retained[np.isin(retained, cal)]
        refstats = _rolling(x[hist], window, k)
        finite = refstats[np.isfinite(refstats)]
        if len(finite):
            ref = float(np.max(finite))
        elif len(hc):
            ref = float(np.quantile(x[hc], .995))
        else:
            ref = original_q
        threshold = multiplier * ref
        phase_refs.append(ref)
        stat = _rolling(x[retained], window, k)
        decisions = (np.isfinite(stat) & (stat > threshold) & (retained > cal_end))
        # searchsorted selects the latest retained decision, so holds never see future data.
        latest = np.searchsorted(retained, np.arange(len(x)), side="right") - 1
        valid = latest >= 0
        held[p, valid] = decisions[latest[valid]]
        held[p, :cal_end + 1] = False
        phase_decisions.append(int(np.sum(retained > cal_end)))
        phase_alarmcounts.append(int(np.sum(decisions)))
    flags = (np.sum(held, axis=0) >= required)
    flags[:cal_end + 1] = False
    return {"flags": flags, "metadata": {"stride": int(stride), "offsets": offsets.tolist(),
            "votes": int(required), "reference_per_phase": phase_refs,
            "phase_decisioncounts": phase_decisions,
            "phase_alarmcounts": phase_alarmcounts}}


def dense_flags(scores, train_indices, cal_indices, window=21, k=13, multiplier=2.0):
    """Original dense full-score 21/13 history-max, multiplier-2 route."""
    train, cal = _idx(train_indices, "train_indices"), _idx(cal_indices, "cal_indices")
    x = _validate(scores, train, cal)
    hist = np.concatenate((train, cal))
    refstat = _rolling(x[hist], window, k)
    finite = refstat[np.isfinite(refstat)]
    ref = float(np.max(finite)) if len(finite) else float(np.quantile(x[cal], .995))
    threshold = multiplier * ref
    stat = _rolling(x, window, k)
    flags = (stat > threshold)
    flags[: int(np.max(cal)) + 1] = False
    return flags
