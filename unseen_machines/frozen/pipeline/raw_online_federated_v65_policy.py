"""Pure V65 calibration-eligible WATCH maturation policy.

V65 leaves the frozen V64 alarm untouched and adds one normal-route fallback:
a sufficiently persistent V57 WATCH may authorize an alarm when the device's
commissioning-only clip fraction meets a predeclared eligibility floor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import raw_online_federated_v57_policy as v57


CLIP_MINIMA = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100)
PERSISTENCE = ((6, 7), (8, 10), (10, 12), (12, 16), (14, 20), (16, 24))


@dataclass(frozen=True, order=True)
class Candidate:
    clip_min: float
    watch_k: int
    watch_window: int

    @property
    def name(self) -> str:
        clip = f"{self.clip_min:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"clipmin_{clip}__watch_{self.watch_k}of{self.watch_window}"

    def validate(self) -> None:
        if float(self.clip_min) not in CLIP_MINIMA:
            raise ValueError("clip_min is outside the frozen V65 grid")
        if (int(self.watch_k), int(self.watch_window)) not in PERSISTENCE:
            raise ValueError("WATCH persistence is outside the frozen V65 grid")


def candidate_alarm(
    v64_alarm: np.ndarray,
    v57_raw_watch_evidence: np.ndarray,
    collapse_route: bool,
    clip_fraction: float,
    candidate: Candidate,
) -> np.ndarray:
    candidate.validate()
    base = np.asarray(v64_alarm, dtype=bool).reshape(-1)
    raw = np.asarray(v57_raw_watch_evidence, dtype=bool).reshape(-1)
    if len(base) != len(raw):
        raise ValueError("V64 and WATCH arrays must align")
    if bool(collapse_route) or float(clip_fraction) < float(candidate.clip_min):
        return base.copy()
    fallback = v57.persistent_latch(raw, k=candidate.watch_k, window=candidate.watch_window)
    return base | fallback


def candidate_grid() -> tuple[Candidate, ...]:
    result = tuple(Candidate(clip, k, window) for clip in CLIP_MINIMA for k, window in PERSISTENCE)
    if len(result) != 60 or len({item.name for item in result}) != 60:
        raise AssertionError("V65 candidate grid changed")
    return result


__all__ = ["CLIP_MINIMA", "PERSISTENCE", "Candidate", "candidate_alarm", "candidate_grid"]
