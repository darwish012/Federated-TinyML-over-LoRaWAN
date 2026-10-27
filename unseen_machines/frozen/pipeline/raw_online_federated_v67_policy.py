"""Pure causal V67 literature-guided policy layered on frozen V65.

V67 adds two independent routes developed on the burned 66-device cohort:

* EARLY: short-history physical change plus strong AE evidence, available only
  during a fixed early-operation horizon and only when commissioning quality
  is eligible.  It never uses final stream length.
* TRIAD: for low-clip machines, a short-history physical change and AE score
  must coincide with the already-frozen conservative V48 reference.

Both routes use only present/past acquisitions and an 8-of-10 latch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import raw_online_federated_v57_policy as v57


@dataclass(frozen=True)
class Candidate:
    early_clip_min: float = 0.015
    early_physical: float = 0.30
    early_ae: float = 0.40
    early_deadline: int = 120
    triad_clip_max: float = 0.015
    triad_physical: float = 0.30
    triad_ae: float = 0.075
    k: int = 8
    window: int = 10

    @property
    def name(self) -> str:
        return "early_clip15_t30_ae40_until120__triad_clip15_t30_ae075__8of10"

    def validate(self) -> None:
        expected = Candidate()
        if self != expected:
            raise ValueError("V67 policy is fixed to the selected burned-data candidate")


def _matrix(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 100 or not np.all(np.isfinite(array)):
        raise ValueError("V67 expects finite n x 100 feature streams")
    return array


def short_impulse_change(features: np.ndarray, recent: int = 8, prior: int = 32) -> np.ndarray:
    """Backward-only median change of V57's log robust impulse health."""
    values = _matrix(features)
    impulse = values[:, v57.IMPULSE_INDICES]
    center, scale = v57._robust_center_scale(impulse[v57.IMPULSE_CAL])
    robust_z = np.abs((impulse - center) / scale)
    health = np.mean(np.sort(np.log1p(robust_z), axis=1)[:, -10:], axis=1)
    output = np.full(len(values), -np.inf, dtype=np.float64)
    total = int(recent) + int(prior)
    for index in range(total - 1, len(values)):
        recent_values = health[index - recent + 1:index + 1]
        prior_values = health[index - total + 1:index - recent + 1]
        output[index] = float(np.median(recent_values) - np.median(prior_values))
    return output


def fused_short_change(feature_streams: Sequence[np.ndarray]) -> np.ndarray:
    if not feature_streams:
        raise ValueError("at least one feature stream is required")
    arrays = [_matrix(values) for values in feature_streams]
    n = min(len(values) for values in arrays)
    return np.max(np.stack([short_impulse_change(values[:n]) for values in arrays]), axis=0)


def candidate_alarm(
    v65_alarm: np.ndarray,
    feature_streams: Sequence[np.ndarray],
    ae_continuous_score: np.ndarray,
    v48_alarm: np.ndarray,
    clip_fraction: float,
    candidate: Candidate = Candidate(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate.validate()
    base = np.asarray(v65_alarm, dtype=bool).reshape(-1)
    ae = np.asarray(ae_continuous_score, dtype=np.float64).reshape(-1)
    reference = np.asarray(v48_alarm, dtype=bool).reshape(-1)
    change = fused_short_change(feature_streams)
    if not (len(base) == len(ae) == len(reference) == len(change)):
        raise ValueError("V67 arrays must align")

    early_raw = (change >= candidate.early_physical) & (ae >= candidate.early_ae)
    early_raw[candidate.early_deadline + 1:] = False
    if float(clip_fraction) < candidate.early_clip_min:
        early_raw[:] = False
    early = v57.persistent_latch(early_raw, k=candidate.k, window=candidate.window)

    triad_raw = (
        (change >= candidate.triad_physical)
        & (ae >= candidate.triad_ae)
        & reference
    )
    if float(clip_fraction) > candidate.triad_clip_max:
        triad_raw[:] = False
    triad = v57.persistent_latch(triad_raw, k=candidate.k, window=candidate.window)
    return base | early | triad, early, triad, change


__all__ = ["Candidate", "candidate_alarm", "fused_short_change", "short_impulse_change"]
