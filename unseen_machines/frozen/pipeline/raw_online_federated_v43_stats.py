"""V43 Section 3.4 experiment-clustered bootstrap, Section 3.2 onset-tolerance
band, lead-time/detection-delay, and the Section 3.3 secondary negative-control
enumeration.

This module is the ONLY place besides ``raw_online_federated_v43_score.py``
that may read onset or status evidence -- callers must only invoke it AFTER
the V43 prediction lock exists (mirroring V42's onset-access boundary, just
drawn at the ``stats`` phase rather than only inside ``score``, per
V43_PREREGISTRATION.md's explicit ``audit -> cohort -> grid -> configurations
-> stats -> comms -> score`` phase order, §5 G5). It performs no file I/O of
its own; the phase driver supplies already-loaded rows/DataFrames.

Bootstrap clustering (SS3.4): resampling draws MACHINE-LIFE EXPERIMENTS (via
``raw_online_federated_v43_cohort.experiment_id``) with replacement, never
individual bearing rows -- an IMS experiment always enters or leaves a
resample with every one of its bearings present in the input rows. This is
enforced structurally by construction (the resample is built by unioning
whole per-experiment row groups), not merely by convention, and
``verify_no_partial_experiment_across_resamples`` re-derives the identical
draw sequence to check the invariant explicitly for gate G2.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import raw_online_federated_v43_cohort as cohort

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260805
CI_LOW, CI_HIGH = 0.025, 0.975
TAU_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)

# SS4/Leg 4 reuses these per-dataset acquisition-cadence assumptions,
# unchanged from raw_online_federated_v39_protocol.DRAFT.json
# "raw_telemetry_model.profiles" (not fabricated fresh here). HIT's cadence is
# undocumented there (null) and LDM-UPB is absent entirely -- both are
# reported as "physical time unavailable", never guessed.
CADENCE_SECONDS: dict[str, float | None] = {
    "ferrara": 300.0,
    "xjtu": 60.0,
    "pronostia": 10.0,
    "ims": 600.0,
    "hit": None,
    "ldm_upb": None,
}


# --------------------------------------------------------------------------
# SS3.4: experiment-clustered bootstrap
# --------------------------------------------------------------------------
def _group_by_experiment(rows: Sequence[dict], client_id_field: str = "client_id") -> dict[str, list[dict]]:
    by_experiment: dict[str, list[dict]] = {}
    for row in rows:
        eid = cohort.experiment_id(row[client_id_field])
        by_experiment.setdefault(eid, []).append(row)
    return by_experiment


def clustered_bootstrap(
    rows: Sequence[dict],
    value_fn: Callable[[list[dict]], float],
    *,
    client_id_field: str = "client_id",
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """95% percentile CI for ``value_fn`` over ``rows``, clustered on experiment.

    Each of ``draws`` resamples draws ``n_experiments`` experiment ids with
    replacement from the experiments PRESENT in ``rows`` and includes every
    row belonging to each drawn experiment (repeated in full if an experiment
    is drawn more than once) -- an experiment can never appear with only some
    of its member rows. ``value_fn`` may return NaN for a degenerate resample
    (e.g. no failed devices drawn); NaN draws are excluded from the quantile
    but counted in ``draws``.
    """
    rows = list(rows)
    point = value_fn(rows)
    by_experiment = _group_by_experiment(rows, client_id_field)
    experiment_ids = sorted(by_experiment)
    n_exp = len(experiment_ids)
    if n_exp == 0:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"), "draws": 0, "experiments": 0, "finite_draws": 0}
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(draws), dtype=np.float64)
    for i in range(int(draws)):
        chosen = rng.integers(0, n_exp, size=n_exp)
        resample_rows: list[dict] = []
        for j in chosen:
            resample_rows.extend(by_experiment[experiment_ids[j]])
        values[i] = value_fn(resample_rows)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"point": point, "ci_low": float("nan"), "ci_high": float("nan"), "draws": int(draws), "experiments": n_exp, "finite_draws": 0}
    return {
        "point": point,
        "ci_low": float(np.quantile(finite, CI_LOW)),
        "ci_high": float(np.quantile(finite, CI_HIGH)),
        "draws": int(draws),
        "experiments": n_exp,
        "finite_draws": int(len(finite)),
    }


def matched_pair_rows(
    rows_a: Sequence[dict], rows_b: Sequence[dict], value_field: str, *, client_id_field: str = "client_id",
) -> list[dict]:
    """Pair rows from two operating-point subsets (e.g. F-L(PCA) and
    L-L(PCA) at the same nominal z) by client_id and compute the per-device
    difference ``value_a - value_b``. Only devices present in BOTH subsets
    are included -- a genuine paired comparison, per
    V43_PREREGISTRATION.md SS2.2.1's matched-algorithm pair. ``value_field``
    values are cast to float so a boolean field (e.g. ``ever_alarmed``,
    ``preterminal_detected``) works directly (True/False -> 1.0/0.0).
    """
    by_b = {row[client_id_field]: row for row in rows_b}
    paired = []
    for row_a in rows_a:
        cid = row_a[client_id_field]
        if cid in by_b:
            paired.append({client_id_field: cid, "diff": float(row_a[value_field]) - float(by_b[cid][value_field])})
    return paired


def matched_pair_bootstrap(
    rows_a: Sequence[dict], rows_b: Sequence[dict], value_field: str, *,
    client_id_field: str = "client_id", draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Experiment-clustered 95% CI on the mean paired difference (A - B),
    for the SS2.2.1 matched-algorithm pair F-L(PCA) minus L-L(PCA). Reuses
    :func:`clustered_bootstrap` outright (paired rows are clustered on
    experiment exactly like every other bootstrap in this module) rather
    than a parallel implementation.
    """
    paired = matched_pair_rows(rows_a, rows_b, value_field, client_id_field=client_id_field)
    result = clustered_bootstrap(
        paired,
        lambda subset: (float(np.mean([r["diff"] for r in subset])) if subset else float("nan")),
        client_id_field=client_id_field, draws=draws, seed=seed,
    )
    result["paired_devices"] = len(paired)
    return result


def verify_no_partial_experiment_across_resamples(
    rows: Sequence[dict],
    *,
    client_id_field: str = "client_id",
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> bool:
    """Gate G2: re-derives :func:`clustered_bootstrap`'s exact draw sequence
    and checks, for every one of ``draws`` resamples (not a sample of them),
    that no experiment ever appears with only some of its member rows.
    """
    rows = list(rows)
    by_experiment = _group_by_experiment(rows, client_id_field)
    experiment_ids = sorted(by_experiment)
    n_exp = len(experiment_ids)
    if n_exp == 0:
        return True
    universe_client_ids = [row[client_id_field] for members in by_experiment.values() for row in members]
    rng = np.random.default_rng(int(seed))
    for _ in range(int(draws)):
        chosen = rng.integers(0, n_exp, size=n_exp)
        selected: set[str] = set()
        for j in chosen:
            selected.update(row[client_id_field] for row in by_experiment[experiment_ids[j]])
        if not cohort.validate_no_partial_experiment(selected, universe_client_ids):
            return False
    return True


# --------------------------------------------------------------------------
# SS3.2: onset-tolerance band. classify_event_with_tolerance(tau=0.0)
# reproduces raw_online_federated_v42_score.classify_event's "event" field
# exactly (raw_online_federated_v43_stats_tests.py asserts this equivalence).
# --------------------------------------------------------------------------
def classify_event_with_tolerance(
    status: str, first_alarm: int, acquisitions: int, onset_value: float, tau: float,
) -> str:
    """One of missed/premature/valid/late/not_lifecycle, at onset tolerance tau.

    ``tau`` widens the valid boundary backward from the annotated onset by
    ``tau`` of the device's own remaining life at onset (remaining life taken
    as ``acquisitions - onset``, i.e. to the device's own recorded end of
    life, not to the 90% terminal-window cutoff -- SPEC-AMBIGUITY: SS3.2 does
    not define "remaining life" precisely; this is the most literal physical
    reading). An alarm at or after ``onset - tau * remaining_life`` (and
    still before the terminal window) counts as valid at that tolerance,
    instead of premature.
    """
    terminal = int(math.floor(0.9 * int(acquisitions)))
    first_alarm = int(first_alarm)
    onset_available = status == "failed" and pd.notna(onset_value)
    if not onset_available:
        return "not_lifecycle"
    onset = int(onset_value)
    remaining_life = max(int(acquisitions) - onset, 0)
    valid_lower_bound = onset - float(tau) * remaining_life
    if first_alarm < 0:
        return "missed"
    if first_alarm < valid_lower_bound:
        return "premature"
    if first_alarm < terminal:
        return "valid"
    return "late"


def onset_tolerance_band(rows: Sequence[dict]) -> list[dict]:
    """Per (arm-key, tau) counts of premature/valid/late/missed among failed,
    onset-available devices. ``rows`` must already carry ``status``,
    ``first_alarm``, ``acquisitions_evidence``, and
    ``first_sustained_onset_acquisition`` (i.e. be post-merge event rows).
    """
    out = []
    for tau in TAU_GRID:
        events = [
            classify_event_with_tolerance(
                row["status"], row["first_alarm"], row["acquisitions_evidence"],
                row.get("first_sustained_onset_acquisition", float("nan")), tau,
            )
            for row in rows
            if row["status"] == "failed"
        ]
        counts = {name: events.count(name) for name in ("premature", "valid", "late", "missed", "not_lifecycle")}
        onset_available = sum(1 for name in events if name != "not_lifecycle")
        out.append({
            "tau": tau,
            "failed_devices": len(events),
            "onset_available_devices": onset_available,
            **counts,
            "premature_fraction_of_onset_available": counts["premature"] / max(onset_available, 1),
            "valid_fraction_of_onset_available": counts["valid"] / max(onset_available, 1),
        })
    return out


# --------------------------------------------------------------------------
# Lead time / detection delay (onset-independent: first alarm -> end of life)
# --------------------------------------------------------------------------
def lead_time_rows(rows: Sequence[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row["status"] != "failed" or int(row["first_alarm"]) < 0:
            continue
        dataset = cohort.dataset_of(row["client_id"])
        cadence = CADENCE_SECONDS.get(dataset)
        acquisitions_total = int(row["acquisitions_evidence"])
        lead_acquisitions = acquisitions_total - int(row["first_alarm"])
        lead_seconds = float(lead_acquisitions) * cadence if cadence is not None else None
        out.append({
            "client_id": row["client_id"],
            "dataset": dataset,
            "lead_time_acquisitions": int(lead_acquisitions),
            "lead_time_seconds": lead_seconds,
            "cadence_seconds_assumption": cadence,
        })
    return out


def median_iqr(values: Sequence[float | None]) -> dict:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if len(array) == 0:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan"), "n": 0}
    return {
        "median": float(np.median(array)),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
        "n": int(len(array)),
    }


# --------------------------------------------------------------------------
# SS3.3: secondary negative-control enumeration, at the machine-life
# experiment unit. Two kinds: documented-healthy devices (already primary,
# repeated here for completeness) and the clearly-stable pre-onset window of
# every onset-available failed device (never a censored/confounded device).
# --------------------------------------------------------------------------
def secondary_negative_control_candidates(evidence: pd.DataFrame) -> list[dict]:
    rows = []
    for client_id, meta in evidence.iterrows():
        status = str(meta["status"])
        eid = cohort.experiment_id(client_id)
        if status == "healthy":
            rows.append({
                "client_id": client_id, "experiment_id": eid, "dataset": cohort.dataset_of(client_id),
                "kind": "documented_healthy_device", "candidate_negative_acquisitions": int(meta["acquisitions"]),
            })
        elif status == "failed":
            onset_value = meta.get("first_sustained_onset_acquisition", float("nan"))
            if pd.notna(onset_value) and int(onset_value) > 0:
                rows.append({
                    "client_id": client_id, "experiment_id": eid, "dataset": cohort.dataset_of(client_id),
                    "kind": "pre_onset_window_of_failed_device",
                    "candidate_negative_acquisitions": int(onset_value),
                })
        # censored (IMS) and fixed_fault (HIT) devices are deliberately never
        # added here -- confounded/not a lifecycle transition, per SS3.3/G4.
    return rows


def secondary_negative_control_summary(rows: list[dict]) -> dict:
    experiments = {row["experiment_id"] for row in rows}
    return {
        "candidate_rows": len(rows),
        "candidate_experiments": len(experiments),
        "documented_healthy_experiments": len({r["experiment_id"] for r in rows if r["kind"] == "documented_healthy_device"}),
        "pre_onset_window_experiments": len({r["experiment_id"] for r in rows if r["kind"] == "pre_onset_window_of_failed_device"}),
    }


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "CADENCE_SECONDS",
    "TAU_GRID",
    "classify_event_with_tolerance",
    "clustered_bootstrap",
    "lead_time_rows",
    "matched_pair_bootstrap",
    "matched_pair_rows",
    "median_iqr",
    "onset_tolerance_band",
    "secondary_negative_control_candidates",
    "secondary_negative_control_summary",
    "verify_no_partial_experiment_across_resamples",
]
