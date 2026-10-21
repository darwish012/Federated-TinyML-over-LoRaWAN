"""Pure causal V63 adaptive, confidence-gated rescue policy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import raw_online_federated_v57_policy as v57
import raw_online_federated_v62_policy as v62


ADAPTIVE_START = v57.RECENT + v57.PRIOR - 1


@dataclass(frozen=True, order=True)
class Candidate:
    rescue_clip_min: float
    calibration_length: int
    calibration_quantile: float
    calibration_margin: float
    threshold_floor: float
    history: int
    impulse_hits: int
    ae_hits: int

    @property
    def name(self) -> str:
        def number(value: float) -> str:
            return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        return (
            f"clip_{number(self.rescue_clip_min)}__cal{self.calibration_length}"
            f"q{number(self.calibration_quantile)}__m{number(self.calibration_margin)}"
            f"__floor{number(self.threshold_floor)}__h{self.history}"
            f"__i{self.impulse_hits}__ae{self.ae_hits}"
        )

    def validate(self) -> None:
        if self.rescue_clip_min not in (0.20, 0.25, 0.30):
            raise ValueError("rescue clip threshold is outside the frozen V60 grid")
        if self.calibration_length not in (24, 32, 48):
            raise ValueError("invalid adaptive calibration length")
        if self.calibration_quantile not in (0.90, 0.95, 1.0):
            raise ValueError("invalid adaptive calibration quantile")
        if self.calibration_margin not in (0.05, 0.075, 0.10):
            raise ValueError("invalid adaptive margin")
        if self.threshold_floor not in (0.15, 0.20):
            raise ValueError("invalid adaptive threshold floor")
        if not (1 <= self.impulse_hits <= self.history and 1 <= self.ae_hits <= self.history):
            raise ValueError("invalid accumulated-evidence counts")


def adaptive_threshold(impulse_trend: np.ndarray, candidate: Candidate) -> tuple[float, int]:
    candidate.validate()
    trend = np.asarray(impulse_trend, dtype=np.float64).reshape(-1)
    stop = ADAPTIVE_START + int(candidate.calibration_length)
    if len(trend) < stop:
        raise ValueError("stream is too short for adaptive calibration")
    calibration = trend[ADAPTIVE_START:stop]
    if not np.all(np.isfinite(calibration)):
        raise ValueError("adaptive calibration contains non-finite trend values")
    threshold = max(
        float(candidate.threshold_floor),
        float(np.quantile(calibration, candidate.calibration_quantile))
        + float(candidate.calibration_margin),
    )
    return float(threshold), stop


def candidate_alarm(
    impulse_trend: np.ndarray,
    ae_continuous_score: np.ndarray,
    v57_raw_watch_evidence: np.ndarray,
    v57_final_alarm: np.ndarray,
    collapse_route: bool,
    clip_fraction: float,
    candidate: Candidate,
) -> tuple[np.ndarray, float | None]:
    base = np.asarray(v57_final_alarm, dtype=bool).reshape(-1)
    if not bool(collapse_route) or float(clip_fraction) < float(candidate.rescue_clip_min):
        return base.copy(), None
    if len(base) < ADAPTIVE_START + int(candidate.calibration_length):
        # Structurally short machines cannot support the adaptive rescue.
        # Preserve their frozen V57 result instead of manufacturing a partial
        # calibration window.
        return base.copy(), None
    threshold, calibration_stop = adaptive_threshold(impulse_trend, candidate)
    trend = np.asarray(impulse_trend, dtype=np.float64).reshape(-1)
    ae = np.asarray(ae_continuous_score, dtype=np.float64).reshape(-1)
    hard = np.asarray(v57_raw_watch_evidence, dtype=bool).reshape(-1)
    if not (len(trend) == len(ae) == len(hard) == len(base)):
        raise ValueError("V63 arrays must align")
    extra = (
        v62.rolling_count(np.isfinite(trend) & (trend >= threshold), candidate.history)
        >= int(candidate.impulse_hits)
    ) & (
        v62.rolling_count(np.isfinite(ae) & (ae >= v57.AE_FLOOR), candidate.history)
        >= int(candidate.ae_hits)
    )
    extra[: min(len(extra), max(calibration_stop, candidate.history - 1, v57.OP_START))] = False
    return v57.persistent_latch(hard | extra), threshold


def candidate_grid() -> tuple[Candidate, ...]:
    shapes = ((32, 1, 24), (64, 1, 48), (64, 2, 48), (64, 4, 48), (64, 1, 58))
    result = []
    for clip in (0.20, 0.25, 0.30):
        for length in (24, 32, 48):
            for quantile in (0.90, 0.95, 1.0):
                for margin in (0.05, 0.075, 0.10):
                    for floor in (0.15, 0.20):
                        for history, impulse_hits, ae_hits in shapes:
                            result.append(Candidate(
                                clip, length, quantile, margin, floor,
                                history, impulse_hits, ae_hits,
                            ))
    if len(result) != 810 or len({item.name for item in result}) != 810:
        raise AssertionError("V63 candidate grid changed")
    return tuple(result)


__all__ = ["ADAPTIVE_START", "Candidate", "adaptive_threshold", "candidate_alarm", "candidate_grid"]
