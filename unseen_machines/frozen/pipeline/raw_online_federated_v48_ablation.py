"""V48 branch-subset scoring and onset-blind selection primitives."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import raw_online_federated_v29_models as models
import raw_online_federated_v42_calibration as calibration
import raw_online_federated_v43_grid as grid
import raw_online_federated_v45_cohort as cohort
import raw_online_federated_v46_machine as v46machine

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "raw_online_federated_v48_protocol.json"
BRANCH_ORDER = ("FED_AE_RECON", "FED_PCA16", "FED_DEEP_SVDD")
SUBSET_ORDER = ("AE", "PCA", "SVDD", "AE_PCA", "AE_SVDD", "PCA_SVDD", "AE_PCA_SVDD")
REFERENCE = "AE_PCA_SVDD"


def specification() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def subset_branches(name: str) -> tuple[str, ...]:
    return tuple(specification()["subsets"][name])


def normalized_weights(name: str) -> dict[str, float]:
    branches = subset_branches(name)
    base = specification()["base_weights"]
    total = sum(float(base[branch]) for branch in branches)
    return {branch: float(base[branch]) / total for branch in branches}


def branch_parameter_counts(bundle: dict) -> dict[str, int]:
    return {
        "FED_PCA16": int(bundle["pca"].center.size + bundle["pca"].basis.size),
        "FED_AE_RECON": int(sum(weight.size for weight in bundle["ae"].weights) + bundle["ae"].head_weight.size + 1),
        "FED_DEEP_SVDD": int(bundle["svdd"].center.size + sum(weight.size for weight in bundle["svdd"].weights)),
    }


def subset_parameter_count(bundle: dict, name: str) -> int:
    counts = branch_parameter_counts(bundle)
    return int(sum(counts[branch] for branch in subset_branches(name)))


@dataclass(frozen=True)
class PreparedBearing:
    client_id: str
    n: int
    calibration_m: int
    guard_count: int
    op_start: int
    zscores: Mapping[str, np.ndarray]
    thresholds: Mapping[str, float]


def prepare_bearing(row: dict, bundle: dict, calibration_m: int, epc_seed: int) -> PreparedBearing | None:
    values, _ = cohort.load_stream(row)
    n = len(values)
    if n <= grid.op_start(calibration_m):
        return None
    center, scale = models.robust_center_scale(values[calibration.INPUT_CAL])
    normalized = models.normalize(values, center, scale)
    raw = {
        "FED_AE_RECON": bundle["ae"].reconstruction_score(normalized),
        "FED_PCA16": bundle["pca"].score(normalized),
        "FED_DEEP_SVDD": bundle["svdd"].score(normalized),
    }
    zscores = grid.zscores_at(raw, calibration_m, n)
    cfg = specification()["decision"]
    thresholds = grid.epc_moderate(raw, BRANCH_ORDER, calibration_m, float(cfg["nominal_z"]), str(row["client_id"]), epc_seed)
    guard = grid.guard_slice(calibration_m)
    return PreparedBearing(
        client_id=str(row["client_id"]), n=n, calibration_m=int(calibration_m),
        guard_count=int(guard.stop - guard.start), op_start=int(grid.op_start(calibration_m)),
        zscores=zscores, thresholds=thresholds,
    )


def subset_evidence(prepared: PreparedBearing, name: str) -> v46machine.BearingEvidence:
    branches = subset_branches(name)
    weights = normalized_weights(name)
    zscores = {branch: prepared.zscores[branch] for branch in branches}
    thresholds = {branch: prepared.thresholds[branch] for branch in branches}
    cfg = specification()["decision"]
    continuous, _persistent, raw_exceed = calibration.ensemble_decision(
        zscores, weights, thresholds, float(cfg["strong_z"]), float(cfg["vote_fraction"])
    )
    return v46machine.BearingEvidence(
        client_id=prepared.client_id, n=prepared.n, calibration_m=prepared.calibration_m,
        guard_count=prepared.guard_count, op_start=prepared.op_start,
        continuous=np.asarray(continuous, dtype=float), raw_exceed=np.asarray(raw_exceed, dtype=bool),
    )


def select_subset(records: Sequence[dict], heldout_dataset: str | None) -> dict:
    spec = specification()["selection"]
    development = [row for row in records if heldout_dataset is None or row["dataset"] != heldout_dataset]
    by_subset = {name: {row["machine_id"]: row for row in development if row["subset"] == name} for name in SUBSET_ORDER}
    reference = by_subset[REFERENCE]
    rtf_ids = sorted(key for key, row in reference.items() if row["role"] == cohort.RUN_TO_FAILURE)
    healthy_ids = sorted(key for key, row in reference.items() if row["role"] == cohort.HEALTHY_CONTROL)
    reference_healthy = sum(bool(reference[key]["ever_alarmed"]) for key in healthy_ids)
    candidates = []
    for order, name in enumerate(SUBSET_ORDER):
        challenger = by_subset[name]
        common_rtf = [key for key in rtf_ids if key in challenger]
        common_healthy = [key for key in healthy_ids if key in challenger]
        paired = v46machine.bootstrap_paired_difference([
            (bool(challenger[key]["preterminal_detected"]), bool(reference[key]["preterminal_detected"]))
            for key in common_rtf
        ], int(spec["bootstrap_draws"]), int(spec["bootstrap_seed"]) + order)
        healthy_alarms = sum(bool(challenger[key]["ever_alarmed"]) for key in common_healthy)
        p3 = float(np.mean([challenger[key]["preterminal_detected"] for key in common_rtf])) if common_rtf else None
        early = float(np.mean([challenger[key]["early_operation_alarm"] for key in common_rtf])) if common_rtf else None
        body = int(next(iter(challenger.values()))["model_body_bytes"]) if challenger else None
        qualifies = name == REFERENCE or bool(
            healthy_alarms <= reference_healthy and paired["ci_low"] is not None
            and float(paired["ci_low"]) >= float(spec["preterminal_noninferiority_margin"])
        )
        candidates.append({
            "subset": name, "order": order, "qualifies": qualifies, "heldout_dataset": heldout_dataset,
            "development_rtf_machines": len(common_rtf), "development_healthy_machines": len(common_healthy),
            "healthy_alarms": int(healthy_alarms), "reference_healthy_alarms": int(reference_healthy),
            "preterminal_detection_rate": p3, "early_operation_alarm_rate": early,
            "model_body_bytes": body, "paired_difference": paired,
        })
    feasible = [row for row in candidates if row["qualifies"]]
    feasible.sort(key=lambda row: (
        float("inf") if row["early_operation_alarm_rate"] is None else row["early_operation_alarm_rate"],
        float("inf") if row["model_body_bytes"] is None else row["model_body_bytes"],
        -(row["preterminal_detection_rate"] if row["preterminal_detection_rate"] is not None else -1.0),
        row["order"],
    ))
    return {
        "heldout_dataset": heldout_dataset,
        "development_datasets": sorted({str(row["dataset"]) for row in development}),
        "selected_subset": feasible[0]["subset"], "candidates": candidates,
    }


__all__ = [
    "BRANCH_ORDER", "REFERENCE", "SUBSET_ORDER", "PreparedBearing", "branch_parameter_counts",
    "normalized_weights", "prepare_bearing", "select_subset", "specification", "subset_branches",
    "subset_evidence", "subset_parameter_count",
]
