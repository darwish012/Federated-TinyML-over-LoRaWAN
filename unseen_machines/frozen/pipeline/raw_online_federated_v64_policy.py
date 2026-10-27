"""Pure V64 bounded temporal corroboration layered on a frozen V63 alarm."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import raw_online_federated_v57_policy as v57
import raw_online_federated_v62_policy as v62


@dataclass(frozen=True, order=True)
class Candidate:
    watch_reference_gap: int

    @property
    def name(self) -> str:
        return f"watch_reference_gap_{self.watch_reference_gap}"

    def validate(self) -> None:
        if self.watch_reference_gap not in (8, 16, 24, 32, 40, 48, 64, 96, 128):
            raise ValueError("gap is outside the frozen V64 grid")


def candidate_alarm(
    v63_alarm: np.ndarray,
    v57_raw_watch_evidence: np.ndarray,
    v48_alarm: np.ndarray,
    collapse_route: bool,
    candidate: Candidate,
) -> np.ndarray:
    candidate.validate()
    base = np.asarray(v63_alarm, dtype=bool).reshape(-1)
    if bool(collapse_route):
        return base.copy()
    raw = np.asarray(v57_raw_watch_evidence, dtype=bool).reshape(-1)
    reference = np.asarray(v48_alarm, dtype=bool).reshape(-1)
    if not (len(base) == len(raw) == len(reference)):
        raise ValueError("V64 arrays must align")
    watch = v57.persistent_latch(raw)
    new_watch = watch & ~np.concatenate([np.zeros(1, dtype=bool), watch[:-1]])
    recent_watch = v62.rolling_count(new_watch, candidate.watch_reference_gap + 1) >= 1
    authorization = reference & recent_watch
    authorization[: min(v57.OP_START, len(authorization))] = False
    return base | v57.persistent_latch(authorization)


def candidate_grid() -> tuple[Candidate, ...]:
    return tuple(Candidate(gap) for gap in (8, 16, 24, 32, 40, 48, 64, 96, 128))


__all__ = ["Candidate", "candidate_alarm", "candidate_grid"]
