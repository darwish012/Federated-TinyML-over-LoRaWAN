"""V46 machine-aware training, calibration, and fusion primitives.

This module never reads onset/status evidence.  A machine-life experiment is
one federated client.  Bearings remain separate inside the client and are
averaged before the machine contributes one server update.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import raw_online_federated_v29_models as models
import raw_online_federated_v41 as v41
import raw_online_federated_v42_calibration as calib
import raw_online_federated_v43_grid as v43grid
import raw_online_federated_v45_cohort as cohort
import raw_online_federated_v45_grid as v45grid

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "raw_online_federated_v46_protocol.json"

BRANCHES = ("FED_AE_RECON", "FED_PCA16", "FED_DEEP_SVDD")
PRIMARY_M = 64
ADAPTIVE_M = (64, 32, 16, 8)
FUSIONS = (
    "bearing_latch_then_any",
    "any_then_machine_latch",
    "corroborated_then_machine_latch",
)


class IntegrityFailure(RuntimeError):
    pass


def specification() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int(base) + int.from_bytes(digest[:4], "little") % 1_000_000_000


def group_rows_by_machine(rows: Sequence[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["experiment_id"]), []).append(row)
    return {key: sorted(value, key=lambda item: item["client_id"]) for key, value in sorted(groups.items())}


def validate_machine_partition(groups: Mapping[str, Sequence[dict]]) -> None:
    seen: set[str] = set()
    for machine_id, rows in groups.items():
        if not rows:
            raise IntegrityFailure(f"empty machine group: {machine_id}")
        if any(str(row["experiment_id"]) != machine_id for row in rows):
            raise IntegrityFailure(f"mixed experiment in machine group: {machine_id}")
        datasets = {str(row["dataset"]) for row in rows}
        if len(datasets) != 1:
            raise IntegrityFailure(f"machine spans datasets: {machine_id}")
        for row in rows:
            client_id = str(row["client_id"])
            if client_id in seen:
                raise IntegrityFailure(f"bearing appears twice: {client_id}")
            seen.add(client_id)


def _normalised_fit(row: dict) -> np.ndarray:
    values, _ = cohort.load_stream(row)
    if len(values) <= calib.INPUT_CAL.stop:
        raise IntegrityFailure(f"too short even for input calibration: {row['client_id']}")
    center, scale = models.robust_center_scale(values[calib.INPUT_CAL])
    normalised = models.normalize(values, center, scale)
    stop = min(calib.FIT.stop, len(normalised))
    fit = np.asarray(normalised[calib.FIT.start:stop], dtype=np.float32)
    if len(fit) < 2:
        raise IntegrityFailure(f"too short for local fit: {row['client_id']}")
    return fit


def training_machines(rows: Sequence[dict], holdout_dataset: str | None = None) -> dict[str, list[np.ndarray]]:
    allowed = {cohort.RUN_TO_FAILURE, cohort.HEALTHY_CONTROL}
    selected = [
        row for row in rows
        if row["role"] in allowed and (holdout_dataset is None or row["dataset"] != holdout_dataset)
    ]
    groups = group_rows_by_machine(selected)
    validate_machine_partition(groups)
    return {machine_id: [_normalised_fit(row) for row in members] for machine_id, members in groups.items()}


def fit_hierarchical_pca(machine_bearings: Mapping[str, Sequence[np.ndarray]], components: int = 16) -> models.PCAModel:
    """Equal bearing weight inside machine, then equal machine weight."""
    if not machine_bearings:
        raise ValueError("hierarchical PCA requires machines")
    machine_means = []
    machine_seconds = []
    for bearings in machine_bearings.values():
        if not bearings:
            raise ValueError("machine has no bearings")
        means, seconds = [], []
        for values in bearings:
            x = np.asarray(values, dtype=np.float64)
            means.append(np.mean(x, axis=0))
            seconds.append((x.T @ x) / len(x))
        machine_means.append(np.mean(np.stack(means), axis=0))
        machine_seconds.append(np.mean(np.stack(seconds), axis=0))
    global_mean = np.mean(np.stack(machine_means), axis=0)
    second = np.mean(np.stack(machine_seconds), axis=0)
    covariance = second - np.outer(global_mean, global_mean)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][: int(components)]
    return models.PCAModel(global_mean.astype(np.float32), eigenvectors[:, order].T.astype(np.float32))


def _machine_local_ae(
    global_model: models.AEModel,
    machine_id: str,
    bearings: Sequence[np.ndarray],
    round_index: int,
    cfg: Mapping[str, float | int],
) -> tuple[models.AEModel, list[dict]]:
    locals_, diagnostics = [], []
    for bearing_index, values in enumerate(bearings):
        local, diagnostic = models.train_ae_local(
            global_model,
            values,
            epochs=int(cfg["local_epochs"]),
            batch_size=int(cfg["batch_size"]),
            learning_rate=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
            seed=stable_seed(f"ae:{machine_id}:{bearing_index}:{round_index}", int(cfg["seed"])),
        )
        locals_.append(local)
        diagnostics.append(diagnostic)
    return models._mean_ae(locals_), diagnostics


def train_hierarchical_ae(
    initial: models.AEModel,
    machine_bearings: Mapping[str, Sequence[np.ndarray]],
    cfg: Mapping[str, float | int],
) -> tuple[models.AEModel, list[dict]]:
    model = initial
    trace: list[dict] = []
    for round_index in range(int(cfg["rounds"])):
        machine_models, diagnostics = [], []
        for machine_id, bearings in sorted(machine_bearings.items()):
            local, local_diag = _machine_local_ae(model, machine_id, bearings, round_index, cfg)
            machine_models.append(local)
            diagnostics.extend(local_diag)
        model = models._mean_ae(machine_models)
        trace.append({
            "round": round_index + 1,
            "machine_clients": len(machine_models),
            "bearing_updates": len(diagnostics),
            "mean_reconstruction_loss": float(np.mean([d["reconstruction_loss"] for d in diagnostics])),
        })
    return model, trace


def _hierarchical_svdd_center(initial: models.SVDDModel, machine_bearings: Mapping[str, Sequence[np.ndarray]]) -> np.ndarray:
    machine_centers = []
    for bearings in machine_bearings.values():
        bearing_centers = [np.mean(initial.embed(values), axis=0) for values in bearings]
        machine_centers.append(np.mean(np.stack(bearing_centers), axis=0))
    center = np.mean(np.stack(machine_centers), axis=0).astype(np.float32)
    small = np.abs(center) < 0.1
    center[small & (center < 0)] = -0.1
    center[small & (center >= 0)] = 0.1
    return center


def train_hierarchical_svdd(
    initial: models.SVDDModel,
    machine_bearings: Mapping[str, Sequence[np.ndarray]],
    cfg: Mapping[str, float | int],
) -> tuple[models.SVDDModel, list[dict]]:
    model = models.SVDDModel(initial.weights, _hierarchical_svdd_center(initial, machine_bearings))
    trace: list[dict] = []
    for round_index in range(int(cfg["rounds"])):
        machine_models, diagnostics = [], []
        for machine_id, bearings in sorted(machine_bearings.items()):
            bearing_models = []
            for bearing_index, values in enumerate(bearings):
                local, diagnostic = models.train_svdd_local(
                    model,
                    values,
                    epochs=int(cfg["local_epochs"]),
                    batch_size=int(cfg["batch_size"]),
                    learning_rate=float(cfg["learning_rate"]),
                    weight_decay=float(cfg["weight_decay"]),
                    seed=stable_seed(f"svdd:{machine_id}:{bearing_index}:{round_index}", int(cfg["seed"])),
                    synthetic_margin=None,
                    variance_floor=float(cfg["svdd_minimum_latent_variance"]),
                )
                bearing_models.append(local)
                diagnostics.append(diagnostic)
            machine_models.append(models._mean_svdd(bearing_models, model.center))
        model = models._mean_svdd(machine_models, model.center)
        trace.append({
            "round": round_index + 1,
            "machine_clients": len(machine_models),
            "bearing_updates": len(diagnostics),
            "mean_loss": float(np.mean([d["loss"] for d in diagnostics])),
        })
    return model, trace


def train_hierarchical_bundle(initial_bundle: dict, machine_bearings: Mapping[str, Sequence[np.ndarray]]) -> tuple[dict, dict]:
    cfg = specification()["hierarchical_finetune"]
    pca = fit_hierarchical_pca(machine_bearings, int(cfg["pca_components"]))
    ae, ae_trace = train_hierarchical_ae(initial_bundle["ae"], machine_bearings, cfg)
    svdd, svdd_trace = train_hierarchical_svdd(initial_bundle["svdd"], machine_bearings, cfg)
    bundle = {"pca": pca, "ae": ae, "svdd": svdd}
    trace = {
        "machine_clients": len(machine_bearings),
        "bearing_streams": int(sum(len(v) for v in machine_bearings.values())),
        "bearings_per_machine": {key: len(value) for key, value in sorted(machine_bearings.items())},
        "ae_empirical_normal_nuisance": True,
        "augmentation_applied_to_calibration_or_evaluation": False,
        "synthetic_faults_in_primary_svdd_or_alarm": False,
        "ae": ae_trace,
        "svdd": svdd_trace,
    }
    return bundle, trace


def save_bundle(path: Path, bundle: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v41.save_bundle(path, bundle)


def load_bundle(path: Path) -> dict:
    return v41.load_bundle(path)


def adaptive_calibration_size(n: int) -> int | None:
    for m in ADAPTIVE_M:
        if n > v43grid.op_start(m):
            return int(m)
    return None


@dataclass(frozen=True)
class BearingEvidence:
    client_id: str
    n: int
    calibration_m: int
    guard_count: int
    op_start: int
    continuous: np.ndarray
    raw_exceed: np.ndarray


def score_bearing(row: dict, bundle: dict, calibration_m: int, epc_seed: int) -> BearingEvidence | None:
    values, _ = cohort.load_stream(row)
    n = len(values)
    if n <= v43grid.op_start(calibration_m):
        return None
    center, scale = models.robust_center_scale(values[calib.INPUT_CAL])
    normalised = models.normalize(values, center, scale)
    raw = {
        "FED_AE_RECON": bundle["ae"].reconstruction_score(normalised),
        "FED_PCA16": bundle["pca"].score(normalised),
        "FED_DEEP_SVDD": bundle["svdd"].score(normalised),
    }
    zscores = v43grid.zscores_at(raw, calibration_m, n)
    alarm_cfg = specification()["alarm"]
    thresholds = v43grid.epc_moderate(
        raw,
        BRANCHES,
        calibration_m,
        float(alarm_cfg["nominal_z"]),
        str(row["client_id"]),
        int(epc_seed),
    )
    continuous, _persistent_unused, raw_exceed = calib.ensemble_decision(
        zscores,
        alarm_cfg["weights"],
        thresholds,
        float(alarm_cfg["strong"]),
        float(alarm_cfg["vote_fraction"]),
    )
    guard = v43grid.guard_slice(calibration_m)
    return BearingEvidence(
        client_id=str(row["client_id"]),
        n=n,
        calibration_m=int(calibration_m),
        guard_count=int(guard.stop - guard.start),
        op_start=int(v43grid.op_start(calibration_m)),
        continuous=np.asarray(continuous, dtype=np.float64),
        raw_exceed=np.asarray(raw_exceed, dtype=bool),
    )


def _latch(flags: np.ndarray) -> np.ndarray:
    cfg = specification()["alarm"]
    return np.asarray(v45grid.k_of_m_latch(flags, int(cfg["persistence_k"]), int(cfg["persistence_window"])), dtype=bool)


def fuse_machine(evidence: Sequence[BearingEvidence], fusion: str) -> dict:
    if fusion not in FUSIONS:
        raise ValueError(f"unknown fusion: {fusion}")
    if not evidence:
        raise ValueError("cannot fuse no bearings")
    n_values = {item.n for item in evidence}
    m_values = {item.calibration_m for item in evidence}
    starts = {item.op_start for item in evidence}
    lengths = {len(item.raw_exceed) for item in evidence}
    continuous_lengths = {len(item.continuous) for item in evidence}
    if len(n_values) != 1 or len(m_values) != 1 or len(starts) != 1 or len(lengths) != 1 or len(continuous_lengths) != 1:
        raise IntegrityFailure("simultaneous bearings are not aligned")
    matrix = np.stack([item.raw_exceed for item in evidence], axis=0)
    continuous_matrix = np.stack([item.continuous for item in evidence], axis=0)
    if fusion == "bearing_latch_then_any":
        eval_alarm = np.any(np.stack([_latch(row) for row in matrix], axis=0), axis=0)
        fused_raw = np.any(matrix, axis=0)
        fused_continuous = np.max(continuous_matrix, axis=0)
    elif fusion == "any_then_machine_latch":
        fused_raw = np.any(matrix, axis=0)
        eval_alarm = _latch(fused_raw)
        fused_continuous = np.max(continuous_matrix, axis=0)
    else:
        required = 1 if len(evidence) == 1 else 2
        fused_raw = np.sum(matrix, axis=0) >= required
        eval_alarm = _latch(fused_raw)
        ordered = np.sort(continuous_matrix, axis=0)
        fused_continuous = np.mean(ordered[-required:], axis=0)
    guard_count = evidence[0].guard_count
    operation_alarm = np.asarray(eval_alarm[guard_count:], dtype=bool)
    n = evidence[0].n
    op_start = evidence[0].op_start
    full_alarm = np.zeros(n, dtype=bool)
    full_alarm[op_start:op_start + len(operation_alarm)] = operation_alarm
    full_score = np.zeros(n, dtype=np.float64)
    operation_score = np.asarray(fused_continuous[guard_count:], dtype=np.float64)
    full_score[op_start:op_start + len(operation_score)] = operation_score
    first = int(np.flatnonzero(full_alarm)[0]) if np.any(full_alarm) else -1
    terminal = int(np.floor(0.9 * n))
    guard_raw = np.asarray(fused_raw[:guard_count], dtype=bool)
    return {
        "fusion": fusion,
        "bearing_count": len(evidence),
        "calibration_m": evidence[0].calibration_m,
        "n": n,
        "op_start": op_start,
        "terminal": terminal,
        "guard_count": guard_count,
        "guard_exceedances": int(np.sum(guard_raw)),
        "guard_exceedance_rate": float(np.mean(guard_raw)) if guard_count else float("nan"),
        "first_alarm": first,
        "ever_alarmed": bool(first >= 0),
        "preterminal_detected": bool(0 <= first < terminal),
        "lead_acquisitions": int(n - 1 - first) if first >= 0 else None,
        "lead_fraction": float((n - 1 - first) / max(n - 1, 1)) if first >= 0 else None,
        "alarm": full_alarm,
        "score": full_score,
        "bearing_ids": [item.client_id for item in evidence],
        "bearing_raw_exceed": matrix,
    }


def bootstrap_paired_difference(
    pairs: Sequence[tuple[bool, bool]], draws: int = 10000, seed: int = 20260806
) -> dict:
    if not pairs:
        return {"difference": None, "ci_low": None, "ci_high": None, "n": 0}
    array = np.asarray(pairs, dtype=np.float64)
    differences = array[:, 0] - array[:, 1]
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(draws), len(array)))
    sampled = np.mean(differences[indices], axis=1)
    return {
        "difference": float(np.mean(differences)),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
        "n": int(len(array)),
    }


def select_fusion(records: Sequence[dict]) -> dict:
    """Select without onset/status. Records are primary m=64 machine rows."""
    spec = specification()["selection"]
    reference_rows = {
        row["machine_id"]: row for row in records
        if row["arm"] == "reference" and row["fusion"] == FUSIONS[0]
    }
    rtf_ids = [key for key, row in reference_rows.items() if row["role"] == cohort.RUN_TO_FAILURE]
    healthy_ids = [key for key, row in reference_rows.items() if row["role"] == cohort.HEALTHY_CONTROL]
    reference_p1 = sum(bool(reference_rows[key]["ever_alarmed"]) for key in healthy_ids)
    candidates = []
    for order, fusion in enumerate(FUSIONS):
        challenger = {
            row["machine_id"]: row for row in records
            if row["arm"] == "machine_hierarchical" and row["fusion"] == fusion
        }
        common_rtf = [key for key in rtf_ids if key in challenger]
        common_healthy = [key for key in healthy_ids if key in challenger]
        paired = [
            (bool(challenger[key]["preterminal_detected"]), bool(reference_rows[key]["preterminal_detected"]))
            for key in common_rtf
        ]
        diff = bootstrap_paired_difference(paired, int(spec["bootstrap_draws"]), int(spec["bootstrap_seed"]))
        p1 = sum(bool(challenger[key]["ever_alarmed"]) for key in common_healthy)
        guard_values = [float(challenger[key]["guard_exceedance_rate"]) for key in challenger]
        p3 = float(np.mean([challenger[key]["preterminal_detected"] for key in common_rtf])) if common_rtf else float("nan")
        leads = [challenger[key]["lead_fraction"] for key in common_rtf if challenger[key]["lead_fraction"] is not None]
        qualifies = bool(
            p1 <= reference_p1
            and diff["ci_low"] is not None
            and float(diff["ci_low"]) >= float(spec["p3_noninferiority_margin"])
        )
        candidates.append({
            "fusion": fusion,
            "order": order,
            "qualifies": qualifies,
            "p1_alarms": int(p1),
            "p1_reference_alarms": int(reference_p1),
            "p3": p3,
            "paired_difference": diff,
            "mean_guard_exceedance_rate": float(np.mean(guard_values)) if guard_values else None,
            "median_lead_fraction": float(np.median(leads)) if leads else None,
            "rtf_machines": len(common_rtf),
            "rtf_streams": int(sum(int(challenger[key].get("stream_count", 1)) for key in common_rtf)),
            "healthy_machines": len(common_healthy),
            "healthy_streams": int(sum(int(challenger[key].get("stream_count", 1)) for key in common_healthy)),
        })
    feasible = [row for row in candidates if row["qualifies"]]
    feasible.sort(key=lambda row: (
        float("inf") if row["mean_guard_exceedance_rate"] is None else row["mean_guard_exceedance_rate"],
        -row["p3"],
        -(row["median_lead_fraction"] if row["median_lead_fraction"] is not None else -1.0),
        row["order"],
    ))
    return {"selected_fusion": feasible[0]["fusion"] if feasible else None, "candidates": candidates}


__all__ = [
    "ADAPTIVE_M", "BRANCHES", "FUSIONS", "IntegrityFailure", "PRIMARY_M",
    "BearingEvidence", "adaptive_calibration_size", "bootstrap_paired_difference",
    "fit_hierarchical_pca", "fuse_machine", "group_rows_by_machine", "load_bundle",
    "save_bundle", "score_bearing", "select_fusion", "specification",
    "train_hierarchical_bundle", "training_machines", "validate_machine_partition",
]
