"""Pure causal V57 calibration-quality-routed impulse policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import raw_online_federated_v29_models as models

OP_START = 88
IMPULSE_CAL = slice(50, 84)
SCORE_CAL = slice(20, 84)
RECENT = 16
PRIOR = 128
TREND_THRESHOLD = 0.40
AE_FLOOR = 0.075
K = 6
WINDOW = 7
CLIP_LIMIT = 12.0
COLLAPSE_THRESHOLD = 0.20
IMPULSE_INDICES = np.asarray(
    [offset + index for offset in (0, 32, 64) for index in (2, 4, 5, 6, 7)],
    dtype=np.int64,
)


@dataclass(frozen=True)
class MachineTrace:
    n: int
    bearing_count: int
    clip_fraction: float
    collapse_route: bool
    impulse_trend: np.ndarray
    raw_watch_evidence: np.ndarray
    watch_alarm: np.ndarray
    final_alarm: np.ndarray


def _matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 100 or not np.all(np.isfinite(array)):
        raise ValueError("V57 expects a finite n x 100 feature stream")
    return array


def _robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    center = np.median(array, axis=0)
    mad = 1.4826 * np.median(np.abs(array - center), axis=0)
    std = np.std(array, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1e-6), mad, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return center, scale


def calibration_clip_fraction(features: np.ndarray) -> float:
    values = _matrix(features)
    if len(values) < SCORE_CAL.stop:
        raise ValueError("V57 calibration requires at least 84 acquisitions")
    center, scale = models.robust_center_scale(values[:8])
    preclip = np.abs((values[SCORE_CAL] - center) / scale)
    return float(np.mean(preclip >= CLIP_LIMIT))


def impulse_trend(features: np.ndarray) -> np.ndarray:
    values = _matrix(features)
    n = len(values)
    output = np.full(n, -np.inf, dtype=np.float64)
    if n < RECENT + PRIOR:
        return output
    impulse = values[:, IMPULSE_INDICES]
    center, scale = _robust_center_scale(impulse[IMPULSE_CAL])
    robust_z = np.abs((impulse - center) / scale)
    top = np.sort(np.log1p(robust_z), axis=1)[:, -10:]
    health = np.mean(top, axis=1)
    total = RECENT + PRIOR
    for index in range(total - 1, n):
        recent = health[index - RECENT + 1:index + 1]
        prior = health[index - total + 1:index - RECENT + 1]
        output[index] = float(np.median(recent) - np.median(prior))
    return output


def persistent_latch(raw: np.ndarray, *, k: int = K, window: int = WINDOW) -> np.ndarray:
    evidence = np.asarray(raw, dtype=bool).reshape(-1).copy()
    if not (0 < int(k) <= int(window)):
        raise ValueError("invalid k-of-window rule")
    evidence[:min(OP_START, len(evidence))] = False
    output = np.zeros(len(evidence), dtype=bool)
    latched = False
    for index in range(OP_START, len(evidence)):
        start = max(0, index - int(window) + 1)
        if int(np.sum(evidence[start:index + 1])) >= int(k):
            latched = True
        output[index] = latched
    return output


def first_alarm(alarm: np.ndarray) -> int:
    positions = np.flatnonzero(np.asarray(alarm, dtype=bool))
    return int(positions[0]) if len(positions) else -1


def evaluate_machine(
    feature_streams: Sequence[np.ndarray],
    ae_continuous_score: np.ndarray,
    v48_alarm: np.ndarray,
) -> MachineTrace:
    if not feature_streams:
        raise ValueError("machine must contain at least one bearing stream")
    arrays = [_matrix(values) for values in feature_streams]
    n = min([len(values) for values in arrays] + [len(ae_continuous_score), len(v48_alarm)])
    if n < SCORE_CAL.stop:
        raise ValueError("machine is not V57 eligible")
    trends = np.stack([impulse_trend(values[:n]) for values in arrays], axis=0)
    fused_trend = np.max(trends, axis=0)
    clip_fraction = max(calibration_clip_fraction(values[:n]) for values in arrays)
    collapse = bool(clip_fraction >= COLLAPSE_THRESHOLD)
    ae = np.asarray(ae_continuous_score, dtype=np.float64).reshape(-1)[:n]
    reference = np.asarray(v48_alarm, dtype=bool).reshape(-1)[:n]
    raw_watch = (fused_trend >= TREND_THRESHOLD) & (ae >= AE_FLOOR)
    raw_watch[:OP_START] = False
    watch = persistent_latch(raw_watch)
    # Under normal calibration the triad must be true at the same acquisitions
    # as fresh impulse evidence. A previously latched WATCH cannot be reused.
    final_raw = raw_watch if collapse else raw_watch & reference
    final = persistent_latch(final_raw)
    return MachineTrace(
        n=n,
        bearing_count=len(arrays),
        clip_fraction=float(clip_fraction),
        collapse_route=collapse,
        impulse_trend=fused_trend,
        raw_watch_evidence=raw_watch,
        watch_alarm=watch,
        final_alarm=final,
    )


__all__ = [
    "AE_FLOOR", "CLIP_LIMIT", "COLLAPSE_THRESHOLD", "IMPULSE_CAL",
    "IMPULSE_INDICES", "K", "MachineTrace", "OP_START", "PRIOR", "RECENT",
    "SCORE_CAL", "TREND_THRESHOLD", "WINDOW", "calibration_clip_fraction",
    "evaluate_machine", "first_alarm", "impulse_trend", "persistent_latch",
]
