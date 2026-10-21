"""V60 pure functions: eligibility correction, collapse-route provenance,
threshold sensitivity, and confound-breaking-evidence checks.

Nothing in this module touches the filesystem. Every function takes plain
Python/numpy data structures already loaded by the driver
(raw_online_federated_v60.py) and returns plain data structures the driver
serializes. This keeps the producer (these functions) testable in isolation
and lets a contract test feed a real producer's output into a real consumer.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

import raw_online_federated_v46_stats as stats
import raw_online_federated_v57_policy as policy

# A machine needs RECENT+PRIOR acquisitions before impulse_trend can return a
# finite value anywhere; below that raw_watch_evidence is always False.
BLIND_MIN_ACQUISITIONS = int(policy.RECENT + policy.PRIOR)
DEPLOYED_COLLAPSE_THRESHOLD = float(policy.COLLAPSE_THRESHOLD)
SWEEP_THRESHOLDS: tuple[float, ...] = (0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
ONSET_TOLERANCE = 0.0
TRANSITION_TOLERANCE = 0.10
MID_BAND = (0.05, 0.20)

ONSET_CATEGORIES = ("valid", "premature", "late", "missed", "unavailable", "blind")
TRANSITION_CATEGORIES = ("valid", "premature", "missed", "blind")
SCA_CATEGORIES = ("valid", "premature", "missed", "control_alarm", "control_ok")

GROUND_TRUTH_ONSET = "onset"
GROUND_TRUTH_TRANSITION = "transition"
GROUND_TRUTH_SCA = "sca"
GROUND_TRUTH_HEALTHY_CONTROL = "healthy_control"
GROUND_TRUTH_NO_TRUTH = "no_truth"


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def is_structurally_blind(n: int) -> bool:
    """A machine that can never produce finite impulse-trend evidence."""
    return int(n) < BLIND_MIN_ACQUISITIONS


def classify_onset(first_alarm: int, onset, n: int, tolerance: float = ONSET_TOLERANCE) -> str:
    """Reproduces V57's own onset-timing classification (see
    raw_online_federated_v57.py:_strict) without importing that driver."""
    if _is_missing(onset):
        return "unavailable"
    terminal = int(np.floor(0.9 * int(n)))
    return stats.onset_event(int(first_alarm), float(onset), terminal, int(n), float(tolerance))


def classify_transition(first_alarm: int, transition, n: int, tolerance: float = TRANSITION_TOLERANCE) -> str:
    """Mirrors V58's transition-tolerance classification, collapsed to the
    shared valid/premature/missed vocabulary (no separate 'late' bucket for
    a fixed published transition point)."""
    if _is_missing(transition):
        return "unavailable"
    first_alarm = int(first_alarm)
    if first_alarm < 0:
        return "missed"
    lower = float(transition) - float(tolerance) * int(n)
    return "premature" if first_alarm < lower else "valid"


def classify_sca(first_alarm: int, first_fault: int, is_control: bool) -> str:
    """Mirrors V57's SCA staged-diagnostic classification."""
    first_alarm = int(first_alarm)
    if bool(is_control):
        return "control_alarm" if first_alarm >= 0 else "control_ok"
    if first_alarm < 0:
        return "missed"
    return "premature" if first_alarm < int(first_fault) else "valid"


def classify_healthy_control(first_alarm: int) -> str:
    return "false_alarm" if int(first_alarm) >= 0 else "control_ok"


def corrected_category(original: str, *, structurally_blind: bool, blind_eligible_policy: bool) -> str:
    """Recode 'missed' to 'blind' only when the policy's own evidence
    mechanism (impulse_trend) structurally cannot fire for this machine.
    Policies that do not depend on impulse_trend (V51, V48) keep their
    original category even for a structurally blind machine, because a
    genuine miss there is not an artifact of insufficient history."""
    if blind_eligible_policy and structurally_blind and original == "missed":
        return "blind"
    return original


def tally(categories: Iterable[str], buckets: Sequence[str] = ONSET_CATEGORIES) -> dict:
    values = list(categories)
    return {bucket: sum(1 for value in values if value == bucket) for bucket in buckets}


def eligibility_rows(historical_records: Sequence[Mapping], v48_first_by_machine: Mapping[str, int]) -> list[dict]:
    """Phase 1 producer.

    historical_records: one dict per historical run_to_failure machine, each
    with at least machine_id, dataset, n, onset, watch_timing, alarm_timing
    (as already computed and locked by V57's score phase).
    v48_first_by_machine: machine_id -> first V48 alarm index, read from the
    stored machine_predictions npz 'v48_alarm' array.
    """
    rows = []
    for record in historical_records:
        machine_id = str(record["machine_id"])
        n = int(record["n"])
        onset = record.get("onset")
        blind = is_structurally_blind(n)
        watch_original = str(record["watch_timing"])
        alarm_original = str(record["alarm_timing"])
        v48_first = int(v48_first_by_machine[machine_id])
        v48_timing = classify_onset(v48_first, onset, n)
        rows.append({
            "machine_id": machine_id,
            "dataset": str(record["dataset"]),
            "n": n,
            "onset": None if _is_missing(onset) else float(onset),
            "onset_known": not _is_missing(onset),
            "structurally_blind": blind,
            "watch_timing_original": watch_original,
            "watch_timing_corrected": corrected_category(watch_original, structurally_blind=blind, blind_eligible_policy=True),
            "alarm_timing_original": alarm_original,
            "alarm_timing_corrected": corrected_category(alarm_original, structurally_blind=blind, blind_eligible_policy=True),
            "v48_first": v48_first,
            "v48_timing": v48_timing,
        })
    return rows


def eligibility_tallies(rows: Sequence[Mapping], v51_tally: Mapping[str, int]) -> dict:
    """Phase 1 consumer: aggregate corrected tallies from eligibility_rows'
    output. V51's own aggregate tally is carried through unchanged (it is not
    structurally blind-affected: V51 does not depend on impulse_trend), so it
    is passed in directly from the already-locked V57 report rather than
    recomputed here."""
    watch = tally([row["watch_timing_corrected"] for row in rows])
    alarm = tally([row["alarm_timing_corrected"] for row in rows])
    v48 = tally([row["v48_timing"] for row in rows])
    v51 = {bucket: int(v51_tally.get(bucket, 0)) for bucket in ONSET_CATEGORIES if bucket != "blind"}
    v51["blind"] = 0
    return {
        "total_machines": len(rows),
        "onset_known": sum(1 for row in rows if row["onset_known"]),
        "structurally_blind": sum(1 for row in rows if row["structurally_blind"]),
        "v57_evaluable": sum(1 for row in rows if row["onset_known"] and not row["structurally_blind"]),
        "watch": watch,
        "alarm": alarm,
        "v51": v51,
        "v48": v48,
    }


def collapse_inventory_rows(raw_rows: Sequence[Mapping]) -> list[dict]:
    """Phase 2 producer. raw_rows: one dict per machine-prediction across all
    panels (historical, sca, v52_complete, v58), each with machine_id,
    dataset, panel, n, clip_fraction, collapse_route, ground_truth
    (one of the GROUND_TRUTH_* constants), and truth (the onset/transition/
    (first_fault,is_control)/None payload appropriate to ground_truth)."""
    truth_bearing = {GROUND_TRUTH_ONSET, GROUND_TRUTH_TRANSITION, GROUND_TRUTH_SCA}
    rows = []
    for raw in raw_rows:
        ground_truth = str(raw["ground_truth"])
        n = int(raw["n"])
        evaluable = False
        if ground_truth == GROUND_TRUTH_ONSET:
            evaluable = not _is_missing(raw.get("truth"))
        elif ground_truth in (GROUND_TRUTH_TRANSITION, GROUND_TRUTH_SCA):
            evaluable = True
        rows.append({
            "machine_id": str(raw["machine_id"]),
            "dataset": str(raw["dataset"]),
            "panel": str(raw["panel"]),
            "n": n,
            "structurally_blind": is_structurally_blind(n),
            "clip_fraction": float(raw["clip_fraction"]),
            "collapse_route": bool(raw["collapse_route"]),
            "ground_truth": ground_truth,
            "evaluable": evaluable and ground_truth in truth_bearing,
        })
    return rows


def crosstab_by_dataset(rows: Sequence[Mapping], *, evaluable_only: bool = False) -> dict:
    """Phase 2 consumer: collapse_route==True counts by dataset."""
    counts: dict[str, int] = {}
    for row in rows:
        if evaluable_only and not row["evaluable"]:
            continue
        if bool(row["collapse_route"]):
            dataset = str(row["dataset"])
            counts[dataset] = counts.get(dataset, 0) + 1
    return dict(sorted(counts.items()))


def mid_band_rows(rows: Sequence[Mapping], lower: float = MID_BAND[0], upper: float = MID_BAND[1]) -> list[dict]:
    """The clip in [0.05, 0.20) band: below the deployed collapse threshold
    but not trivially small."""
    return [dict(row) for row in rows if lower <= float(row["clip_fraction"]) < upper]


def reclassify_final_alarm(raw_watch_evidence, v48_alarm, clip_fraction: float, threshold: float) -> np.ndarray:
    """Exactly the rule specified for Phase 3: derive the final alarm from
    the stored per-acquisition arrays only, using raw_online_federated_v57_policy's
    own persistent_latch (imported, not re-implemented)."""
    watch = np.asarray(raw_watch_evidence, dtype=bool)
    reference = np.asarray(v48_alarm, dtype=bool)
    final_raw = watch if float(clip_fraction) >= float(threshold) else (watch & reference)
    return policy.persistent_latch(final_raw)


def sweep_machine(
    *, raw_watch_evidence, v48_alarm, clip_fraction: float, n: int,
    ground_truth: str, truth, thresholds: Sequence[float] = SWEEP_THRESHOLDS,
) -> dict[float, str]:
    """Phase 3 producer for a single machine: category per swept threshold."""
    blind = is_structurally_blind(n)
    result: dict[float, str] = {}
    for threshold in thresholds:
        final_alarm = reclassify_final_alarm(raw_watch_evidence, v48_alarm, clip_fraction, threshold)
        first = policy.first_alarm(final_alarm)
        if ground_truth == GROUND_TRUTH_ONSET:
            category = classify_onset(first, truth, n)
            category = corrected_category(category, structurally_blind=blind, blind_eligible_policy=True)
        elif ground_truth == GROUND_TRUTH_TRANSITION:
            category = classify_transition(first, truth, n)
            category = corrected_category(category, structurally_blind=blind, blind_eligible_policy=True)
        elif ground_truth == GROUND_TRUTH_SCA:
            first_fault, is_control = truth
            category = classify_sca(first, first_fault, is_control)
        elif ground_truth == GROUND_TRUTH_HEALTHY_CONTROL:
            category = classify_healthy_control(first)
        else:
            category = "alarmed" if first >= 0 else "quiet"
        result[float(threshold)] = category
    return result


def sweep_tallies(
    per_machine: Mapping[str, Mapping[float, str]],
    meta_by_machine: Mapping[str, Mapping],
    thresholds: Sequence[float] = SWEEP_THRESHOLDS,
    buckets: Sequence[str] = ONSET_CATEGORIES,
) -> dict:
    """Phase 3 consumer: aggregate and per-dataset tallies from
    sweep_machine's real per-machine output."""
    datasets = sorted({str(meta_by_machine[machine]["dataset"]) for machine in per_machine})
    out: dict[float, dict] = {}
    for threshold in thresholds:
        categories = [per_machine[machine][float(threshold)] for machine in per_machine]
        per_dataset = {}
        for dataset in datasets:
            subset = [
                per_machine[machine][float(threshold)]
                for machine in per_machine
                if str(meta_by_machine[machine]["dataset"]) == dataset
            ]
            per_dataset[dataset] = tally(subset, buckets)
        out[float(threshold)] = {
            "is_deployed": bool(float(threshold) == DEPLOYED_COLLAPSE_THRESHOLD),
            "aggregate": tally(categories, buckets),
            "per_dataset": per_dataset,
        }
    return out


def confound_breaking_candidates(inventory_rows: Sequence[Mapping], deployed_threshold: float = DEPLOYED_COLLAPSE_THRESHOLD) -> list[dict]:
    """Phase 4: machines that would supply the missing evidence -- non-LDM,
    n>=144, clip>=deployed_threshold, and evaluable (onset or transition
    known)."""
    candidates = []
    for row in inventory_rows:
        if str(row["dataset"]) == "ldm_upb":
            continue
        if float(row["clip_fraction"]) < float(deployed_threshold):
            continue
        if int(row["n"]) < BLIND_MIN_ACQUISITIONS:
            continue
        if not bool(row["evaluable"]):
            continue
        candidates.append(dict(row))
    return candidates


__all__ = [
    "BLIND_MIN_ACQUISITIONS", "DEPLOYED_COLLAPSE_THRESHOLD", "SWEEP_THRESHOLDS",
    "ONSET_TOLERANCE", "TRANSITION_TOLERANCE", "MID_BAND",
    "ONSET_CATEGORIES", "TRANSITION_CATEGORIES", "SCA_CATEGORIES",
    "GROUND_TRUTH_ONSET", "GROUND_TRUTH_TRANSITION", "GROUND_TRUTH_SCA",
    "GROUND_TRUTH_HEALTHY_CONTROL", "GROUND_TRUTH_NO_TRUTH",
    "is_structurally_blind", "classify_onset", "classify_transition",
    "classify_sca", "classify_healthy_control", "corrected_category", "tally",
    "eligibility_rows", "eligibility_tallies",
    "collapse_inventory_rows", "crosstab_by_dataset", "mid_band_rows",
    "reclassify_final_alarm", "sweep_machine", "sweep_tallies",
    "confound_breaking_candidates",
]
