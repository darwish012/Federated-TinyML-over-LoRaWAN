"""V25 Phase 3 alarm policy: a single scalar threshold with a 3-of-5 latch.

This module deliberately contains no model fitting, no rescoring, no labels,
no onset logic, and no target-vs-source branching. It operates only on
already-computed one-dimensional score streams (guard or operational) and a
scalar threshold ``H``.

The policy is the existing V19 persistence rule, isolated from every other
V19/V24 mechanism (no CUSUM, no adaptive location/scale, no trusted-normal
gate, no multiscale contrast blocks): latch permanently once at least
``persistence_k`` of the latest ``persistence_window`` acquisitions have
``score >= H``. A latch, once formed, never clears.

Threshold selection is cross-fitted (leave-one-device-out) against a
predeclared false-alarm-rate budget, estimated only from commissioning guard
streams -- healthy evidence already accepted throughout this project. Guard
streams may legitimately be empty (XJTU has no identifiable guard); every
function here handles that without crashing or dividing by zero.

The false-alarm rate is a *calibration statistic*, not the deployed policy:
it counts alarm **episodes** repeatedly within a guard stream (an episode
begins when the 3-of-5 condition first holds and ends once that condition
has been false for ``persistence_window`` consecutive acquisitions; a later
re-trigger is a new episode), via ``count_episodes``. This is deliberately
different from ``latch_stream``, which stays permanent once latched -- that
permanence is the deployed policy and is never altered by calibration.
The calibration set is **source guards only** (28 identifiable, 560 pooled
acquisitions; target guards are excluded so the deployed threshold is never
fit even partly on the held-out cohort). With only 28 guards totalling 560
acquisitions, one-episode-per-stream counting would collapse the whole
budget grid to two indistinguishable criteria; repeated-episode counting is
what gives the grid resolution. Which guards are passed in (source-only) is
the caller's responsibility -- this module only computes over whatever
streams it is given.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LatchConfig:
    """The 3-of-5 persistence rule, frozen and shared across arms/taxonomies."""

    persistence_k: int = 3
    persistence_window: int = 5

    def __post_init__(self) -> None:
        if self.persistence_window <= 0:
            raise ValueError("persistence_window must be positive")
        if self.persistence_k <= 0:
            raise ValueError("persistence_k must be positive")
        if self.persistence_k > self.persistence_window:
            raise ValueError("persistence_k cannot exceed persistence_window")


DEFAULT_LATCH = LatchConfig()

THRESHOLD_GRID = np.arange(2.0, 40.0 + 1e-9, 0.5, dtype=np.float64)
THRESHOLD_GRID.setflags(write=False)

# Aligned to the 28-source-guard / 560-acquisition lattice (target guards are
# excluded from calibration, per V25_PREREGISTRATION.md SS7.2): the finest
# distinguishable rate is 1 / 560 * 1000 = 1.786 episodes per 1,000, so these
# budgets mean exactly zero, at most one, and at most two episodes across all
# pooled source guard acquisitions -- not round numbers chosen independently
# of the evidence.
BUDGETS: tuple[float, ...] = (0.0, 1.8, 3.6)
MARGINS: tuple[float, ...] = (1.00, 1.25, 1.50)
PRIMARY_BUDGET: float = 1.8
PRIMARY_MARGIN: float = 1.25


def _validate_scores(scores: Sequence[float] | np.ndarray, *, name: str = "scores") -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if values.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def _validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not np.isfinite(value):
        raise ValueError("threshold must be finite")
    return value


def _validate_grid(grid: np.ndarray) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("grid must be a one-dimensional array")
    if values.size == 0:
        raise ValueError("grid must be non-empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("grid must be finite")
    return values


def latch_stream(
    scores: Sequence[float] | np.ndarray,
    threshold: float,
    config: LatchConfig = DEFAULT_LATCH,
) -> np.ndarray:
    """Boolean, permanent, causal 3-of-5-style latch over a score stream.

    Element ``t`` is ``True`` iff at some ``u <= t`` at least
    ``config.persistence_k`` of ``scores[u - config.persistence_window + 1 :
    u + 1]`` satisfied ``score >= threshold`` -- i.e. a *complete* trailing
    window of length ``persistence_window`` was required to ever latch,
    mirroring the frozen V19 ``CusumLatch``/``SecondaryLatch`` behaviour.
    Once latched, every later element is also ``True``. Element ``t`` depends
    only on ``scores[:t + 1]``, so truncating the input cannot change any
    earlier element.
    """

    values = _validate_scores(scores)
    threshold = _validate_threshold(threshold)
    n = len(values)
    window, k = config.persistence_window, config.persistence_k

    exceeds = values >= threshold
    hit_at = np.zeros(n, dtype=bool)
    if n >= window:
        cumulative = np.concatenate(([0], np.cumsum(exceeds.astype(np.int64))))
        counts = cumulative[window:] - cumulative[:-window]
        hit_at[window - 1:] = counts >= k

    latched = np.zeros(n, dtype=bool)
    positions = np.flatnonzero(hit_at)
    if len(positions):
        latched[int(positions[0]):] = True
    return latched


def first_latch_position(
    scores: Sequence[float] | np.ndarray,
    threshold: float,
    config: LatchConfig = DEFAULT_LATCH,
) -> int | None:
    """Positional index into ``scores`` of the first latched element, or None."""

    latched = latch_stream(scores, threshold, config)
    positions = np.flatnonzero(latched)
    return None if not len(positions) else int(positions[0])


def count_episodes(
    scores: Sequence[float] | np.ndarray,
    threshold: float,
    config: LatchConfig = DEFAULT_LATCH,
) -> int:
    """Count repeated 3-of-5 alarm episodes in one score stream.

    This is a **calibration statistic**, distinct from ``latch_stream``'s
    permanent deployed latch. An episode begins at the first acquisition
    where the ``persistence_k``-of-``persistence_window`` condition holds
    (the same windowed condition ``latch_stream`` uses, evaluated only once a
    complete trailing window is available) and ends once that condition has
    been false for ``persistence_window`` consecutive acquisitions; a later
    re-trigger counts as a new episode. ``latch_stream`` itself is never
    consulted or reused here so that its permanence is not accidentally
    disturbed.
    """

    values = _validate_scores(scores)
    threshold = _validate_threshold(threshold)
    n = len(values)
    window, k = config.persistence_window, config.persistence_k

    exceeds = values >= threshold
    condition = np.zeros(n, dtype=bool)
    if n >= window:
        cumulative = np.concatenate(([0], np.cumsum(exceeds.astype(np.int64))))
        counts = cumulative[window:] - cumulative[:-window]
        condition[window - 1:] = counts >= k

    episodes = 0
    active = False
    false_run = 0
    for hit in condition:
        if hit:
            if not active:
                episodes += 1
                active = True
            false_run = 0
        elif active:
            false_run += 1
            if false_run >= window:
                active = False
                false_run = 0
    return episodes


def false_alarm_rate(
    guard_streams: Sequence[np.ndarray],
    threshold: float,
    config: LatchConfig = DEFAULT_LATCH,
) -> float:
    """Pooled expected alarm episodes per 1,000 healthy guard acquisitions.

    Episodes are counted repeatedly within each guard stream via
    ``count_episodes`` (one-episode-per-stream counting is explicitly wrong
    here: with only 28 source guards totalling 560 acquisitions it would
    collapse the budget grid to two indistinguishable criteria). A guard
    stream may legitimately be empty (no identifiable guard, e.g. XJTU); such
    streams are skipped rather than passed to ``count_episodes`` or allowed
    to cause a division by zero. If every supplied stream is empty, no rate
    can be estimated and this raises ``ValueError``. This function does not
    decide which guards belong to the calibration set (source-only, per
    V25_PREREGISTRATION.md SS7.2) -- that filtering is the caller's
    responsibility.
    """

    streams = list(guard_streams)
    if len(streams) == 0:
        raise ValueError("guard_streams must be a non-empty sequence")
    threshold = _validate_threshold(threshold)

    total_acquisitions = 0
    total_episodes = 0
    for stream in streams:
        values = np.asarray(stream, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("each guard stream must be one-dimensional")
        if values.size == 0:
            continue  # e.g. XJTU: no identifiable guard -- skip, do not divide by zero
        if not np.all(np.isfinite(values)):
            raise ValueError("each guard stream must be finite")
        total_acquisitions += len(values)
        total_episodes += count_episodes(values, threshold, config)

    if total_acquisitions == 0:
        raise ValueError("no non-empty guard stream is available to estimate a false-alarm rate")
    return float(total_episodes) / float(total_acquisitions) * 1000.0


def select_threshold(
    guard_streams: Sequence[np.ndarray],
    budget: float,
    margin: float = 1.0,
    grid: np.ndarray = THRESHOLD_GRID,
    config: LatchConfig = DEFAULT_LATCH,
) -> float:
    """Smallest grid threshold with ``false_alarm_rate <= budget``, times ``margin``."""

    budget = float(budget)
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("budget must be finite and non-negative")
    margin = float(margin)
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    grid_values = np.sort(_validate_grid(grid))

    for candidate in grid_values:
        if false_alarm_rate(guard_streams, float(candidate), config) <= budget:
            return float(candidate) * margin
    raise ValueError(f"no grid value achieves false-alarm budget <= {budget}")


def leave_one_out_thresholds(
    guard_streams_by_device: Mapping[str, np.ndarray],
    budget: float,
    margin: float = 1.0,
    grid: np.ndarray = THRESHOLD_GRID,
    config: LatchConfig = DEFAULT_LATCH,
) -> dict:
    """Leave-one-device-out cross-fitted threshold check on guard streams.

    For each device ``d`` with a non-empty guard stream: compute
    ``H_(-d) = select_threshold(all guards except d, budget, margin)`` --
    which already has ``margin`` folded in -- then test whether ``d``'s own
    guard stream latches at ``H_(-d)``. Devices with no identifiable guard
    (empty stream, e.g. XJTU) cannot be held out and are excluded from both
    the per-device report and the counts.
    """

    if not isinstance(guard_streams_by_device, Mapping):
        raise ValueError("guard_streams_by_device must be a mapping")
    if len(guard_streams_by_device) == 0:
        raise ValueError("guard_streams_by_device must be a non-empty mapping")

    device_ids = list(guard_streams_by_device.keys())
    per_device: dict[str, dict[str, float | bool]] = {}
    loo_alarm_count = 0
    loo_device_count = 0

    for device_id in device_ids:
        own_stream = np.asarray(guard_streams_by_device[device_id], dtype=np.float64)
        if own_stream.ndim != 1:
            raise ValueError(f"guard stream for {device_id!r} must be one-dimensional")
        if own_stream.size == 0:
            continue  # no identifiable guard for this device (e.g. XJTU)
        if not np.all(np.isfinite(own_stream)):
            raise ValueError(f"guard stream for {device_id!r} must be finite")

        held_out_streams = [
            guard_streams_by_device[other] for other in device_ids if other != device_id
        ]
        held_out_threshold = select_threshold(held_out_streams, budget, margin, grid, config)
        latched = bool(np.any(latch_stream(own_stream, held_out_threshold, config)))

        per_device[device_id] = {"threshold": held_out_threshold, "latched": latched}
        loo_device_count += 1
        if latched:
            loo_alarm_count += 1

    if loo_device_count == 0:
        raise ValueError(
            "no device with a non-empty guard stream is available for leave-one-out estimation"
        )

    deployed_threshold = select_threshold(
        list(guard_streams_by_device.values()), budget, margin, grid, config
    )
    return {
        "per_device": per_device,
        "loo_alarm_count": loo_alarm_count,
        "loo_device_count": loo_device_count,
        "deployed_threshold": deployed_threshold,
    }


__all__ = [
    "BUDGETS",
    "DEFAULT_LATCH",
    "MARGINS",
    "PRIMARY_BUDGET",
    "PRIMARY_MARGIN",
    "THRESHOLD_GRID",
    "LatchConfig",
    "count_episodes",
    "false_alarm_rate",
    "first_latch_position",
    "latch_stream",
    "leave_one_out_thresholds",
    "select_threshold",
]
