"""Pure candidate policies for the V62 accumulated-corroboration study.

V62 never changes the normal-calibration route.  On collapse-routed machines it
adds a causal accumulated-evidence condition to V57's original hard evidence.
The module is intentionally free of file I/O so it can be unit tested and
hashed before any held-out stream is opened.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import raw_online_federated_v57_policy as v57


@dataclass(frozen=True, order=True)
class Candidate:
    trend_threshold: float
    history: int
    impulse_hits: int
    ae_hits: int

    @property
    def name(self) -> str:
        trend = f"{self.trend_threshold:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"trend_{trend}__h{self.history}__i{self.impulse_hits}__ae{self.ae_hits}"

    def validate(self) -> None:
        if not (0.0 < float(self.trend_threshold) < v57.TREND_THRESHOLD):
            raise ValueError("candidate threshold must be below V57's hard threshold")
        if not (1 <= int(self.impulse_hits) <= int(self.history)):
            raise ValueError("invalid impulse hit count")
        if not (1 <= int(self.ae_hits) <= int(self.history)):
            raise ValueError("invalid AE hit count")


def rolling_count(bits: np.ndarray, history: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.int64).reshape(-1)
    if int(history) <= 0:
        raise ValueError("history must be positive")
    cumulative = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(values)])
    index = np.arange(len(values), dtype=np.int64)
    starts = np.maximum(0, index - int(history) + 1)
    return cumulative[index + 1] - cumulative[starts]


def accumulated_raw(
    impulse_trend: np.ndarray,
    ae_continuous_score: np.ndarray,
    candidate: Candidate,
) -> np.ndarray:
    candidate.validate()
    trend = np.asarray(impulse_trend, dtype=np.float64).reshape(-1)
    ae = np.asarray(ae_continuous_score, dtype=np.float64).reshape(-1)
    if len(trend) != len(ae):
        raise ValueError("trend and AE arrays must align")
    impulse = np.isfinite(trend) & (trend >= float(candidate.trend_threshold))
    ae_active = np.isfinite(ae) & (ae >= v57.AE_FLOOR)
    raw = (
        (rolling_count(impulse, candidate.history) >= int(candidate.impulse_hits))
        & (rolling_count(ae_active, candidate.history) >= int(candidate.ae_hits))
    )
    # A full causal history is required.  This also preserves V57's OP_START.
    raw[: min(len(raw), max(v57.OP_START, candidate.history - 1))] = False
    return raw


def candidate_alarm(
    impulse_trend: np.ndarray,
    ae_continuous_score: np.ndarray,
    v57_raw_watch_evidence: np.ndarray,
    v57_final_alarm: np.ndarray,
    collapse_route: bool,
    candidate: Candidate,
) -> np.ndarray:
    base = np.asarray(v57_final_alarm, dtype=bool).reshape(-1)
    if not bool(collapse_route):
        # The normal route, including simultaneous V48 corroboration, is frozen.
        return base.copy()
    hard = np.asarray(v57_raw_watch_evidence, dtype=bool).reshape(-1)
    if len(hard) != len(base):
        raise ValueError("V57 evidence arrays must align")
    extra = accumulated_raw(impulse_trend, ae_continuous_score, candidate)
    if len(extra) != len(base):
        raise ValueError("candidate arrays must align")
    return v57.persistent_latch(hard | extra)


def candidate_grid() -> tuple[Candidate, ...]:
    result: list[Candidate] = []
    # Direct lower-threshold comparators reproduce the important threshold
    # trade-off with V57's original 6-of-7 persistence.
    for threshold in (0.15, 0.20, 0.25, 0.30):
        result.append(Candidate(threshold, 7, 6, 6))
    # Accumulated corroboration: moderate impulse evidence and persistent AE
    # evidence may occur at different acquisitions inside the causal history.
    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30):
        for history in (16, 32, 64, 96):
            for impulse_hits in (1, 2, 4, 8):
                if impulse_hits > history:
                    continue
                for ae_fraction in (0.50, 0.75, 0.90):
                    ae_hits = int(np.ceil(history * ae_fraction))
                    result.append(Candidate(threshold, history, impulse_hits, ae_hits))
    unique = {candidate.name: candidate for candidate in result}
    if len(unique) != len(result):
        raise AssertionError("duplicate V62 candidate names")
    return tuple(result)


__all__ = [
    "Candidate",
    "accumulated_raw",
    "candidate_alarm",
    "candidate_grid",
    "rolling_count",
]
