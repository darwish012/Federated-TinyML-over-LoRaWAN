"""V41: frozen V29b triad trained label-blind, then tested once on fresh DLR.

The module intentionally contains training, prediction, locking, and post-lock
scoring as explicit phases.  Target MAT values cannot be opened before the
source gate and all-source model lock exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

import raw_online_federated_v29_models as models
import raw_online_federated_v41_data as fresh


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PROTOCOL = HERE / "raw_online_federated_v41_protocol.DRAFT.json"
V12_PROTOCOL = HERE / "raw_online_federated_v12_protocol.json"
V19_STREAMS = HERE / "results" / "raw_online_federated_v19" / "feature_streams" / "source_train"
SOURCE_EVIDENCE = HERE / "results" / "bearing_eda_deep" / "tables" / "device_local_onset_estimates.csv"
V29B_LOCK = HERE / "results" / "raw_online_federated_v29b_v29b1" / "v29b_source_policy_lock.json"
FEATURE_DIM = 100
INPUT_CAL, FIT, POSTCAL, GUARD, OP_START = slice(0, 8), slice(8, 20), slice(20, 28), slice(28, 32), 32
BRANCHES = ("FED_AE_RECON", "FED_PCA16", "FED_DEEP_SVDD")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def root_for(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("RunId must match [A-Za-z0-9_-]+")
    root = (RESULTS / f"raw_online_federated_v41_{run_id}").resolve()
    if RESULTS.resolve() not in root.parents:
        raise ValueError("V41 results must stay under results")
    return root


def specification() -> dict:
    value = read_json(PROTOCOL)
    original = read_json(V29B_LOCK)["selected_candidate"]
    fixed = value["frozen_v29b_policy"]
    for key in ("moderate", "strong", "vote_fraction", "k", "m", "weights"):
        if fixed[key] != original[key]:
            raise RuntimeError(f"V41 changed frozen V29b policy field {key}")
    return value


def source_roster() -> list[tuple[str, str, str]]:
    inherited = read_json(V12_PROTOCOL)["train_clients"]
    rows = [(dataset, unit, f"{dataset}::{unit}") for dataset, units in inherited.items() for unit in units]
    if len(rows) != 40 or len({row[2] for row in rows}) != 40:
        raise RuntimeError("V41 source roster must be 40 unique clients")
    return rows


def source_stream(dataset: str, unit: str) -> tuple[np.ndarray, np.ndarray]:
    path = V19_STREAMS / f"{dataset}__{unit}.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["features"], dtype=np.float32)
        indices = np.asarray(archive["acquisition_index"], dtype=np.int32)
    if (
        values.ndim != 2 or values.shape[1] != FEATURE_DIM or len(values) <= OP_START
        or not np.all(np.isfinite(values))
        or not np.array_equal(indices, np.arange(len(values), dtype=np.int32))
    ):
        raise RuntimeError(f"invalid source stream {dataset}::{unit}")
    return values, indices


def normalize_device(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center, scale = models.robust_center_scale(values[INPUT_CAL])
    return models.normalize(values, center, scale), center, scale


def _train_bundle(client_fit: list[np.ndarray], seed: int) -> tuple[dict, dict]:
    config = specification()["federated_models"]
    pca = models.fit_federated_pca(client_fit, int(config["pca_components"]))
    ae, ae_trace = models.train_federated_ae(
        client_fit, rounds=int(config["rounds"]), local_epochs=int(config["local_epochs"]),
        batch_size=int(config["batch_size"]), learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]), seed=int(seed),
    )
    svdd, svdd_trace = models.train_federated_svdd(
        client_fit, rounds=int(config["rounds"]), local_epochs=int(config["local_epochs"]),
        batch_size=int(config["batch_size"]), learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]), seed=int(seed), synthetic_margin=None,
        variance_floor=float(config["svdd_minimum_latent_variance"]),
    )
    bundle = {"pca": pca, "ae": ae, "svdd": svdd}
    return bundle, {"ae": ae_trace, "svdd": svdd_trace, "clients": len(client_fit)}


def save_bundle(path: Path, bundle: dict) -> None:
    payload: dict[str, np.ndarray] = {
        "pca_center": bundle["pca"].center, "pca_basis": bundle["pca"].basis,
        "ae_head_weight": bundle["ae"].head_weight,
        "ae_head_bias": np.asarray(bundle["ae"].head_bias),
        "svdd_center": bundle["svdd"].center,
    }
    payload.update({f"ae_w{i}": value for i, value in enumerate(bundle["ae"].weights)})
    payload.update({f"svdd_w{i}": value for i, value in enumerate(bundle["svdd"].weights)})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def load_bundle(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        pca = models.PCAModel(np.asarray(archive["pca_center"]), np.asarray(archive["pca_basis"]))
        ae_weights = tuple(np.asarray(archive[f"ae_w{i}"], dtype=np.float32) for i in range(4))
        ae = models.AEModel(ae_weights, np.asarray(archive["ae_head_weight"]), float(archive["ae_head_bias"]))
        svdd_weights = tuple(np.asarray(archive[f"svdd_w{i}"], dtype=np.float32) for i in range(2))
        svdd = models.SVDDModel(svdd_weights, np.asarray(archive["svdd_center"]))
    return {"pca": pca, "ae": ae, "svdd": svdd}


def audit(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("V41 RunId already contains artifacts")
    spec = specification()
    source_hashes = {}
    for dataset, unit, client_id in source_roster():
        source_stream(dataset, unit)
        source_hashes[client_id] = sha256(V19_STREAMS / f"{dataset}__{unit}.npz")
    # No target MAT file is deserialized here. Presence is deliberately optional
    # so the protocol can be frozen while downloads are still in progress.
    dependency_paths = {
        "protocol": PROTOCOL,
        "runner": Path(__file__).resolve(),
        "data_adapter": Path(fresh.__file__).resolve(),
        "model_primitives": Path(models.__file__).resolve(),
        "tests": HERE / "raw_online_federated_v41_tests.py",
        "launcher": HERE / "start_v41_guarded.ps1",
        "organizer": HERE / "prepare_v41_dlr.ps1",
    }
    write_json(root / "v41_audit_lock.json", {
        "experiment": "V41", "immutable": True, "protocol_sha256": sha256(PROTOCOL),
        "v29b_policy_sha256": sha256(V29B_LOCK), "source_stream_sha256": source_hashes,
        "dependency_sha256": {name: sha256(path) for name, path in dependency_paths.items()},
        "source_clients": 40, "fresh_target_clients_declared": 6,
        "one_bearing_one_client": True, "target_streams_opened": False,
        "target_sensor_values_opened": False, "target_outcomes_opened": False,
        "onset_opened": False, "fresh_target_ids": spec["official_dataset"]["fresh_clients"],
    })


def require_audit(root: Path) -> dict:
    lock = read_json(root / "v41_audit_lock.json")
    if lock["protocol_sha256"] != sha256(PROTOCOL) or lock["target_sensor_values_opened"]:
        raise RuntimeError("invalid V41 audit lock")
    current = {
        "protocol": PROTOCOL, "runner": Path(__file__).resolve(),
        "data_adapter": Path(fresh.__file__).resolve(), "model_primitives": Path(models.__file__).resolve(),
        "tests": HERE / "raw_online_federated_v41_tests.py", "launcher": HERE / "start_v41_guarded.ps1",
        "organizer": HERE / "prepare_v41_dlr.ps1",
    }
    for name, path in current.items():
        if lock["dependency_sha256"].get(name) != sha256(path):
            raise RuntimeError(f"V41 frozen dependency changed after audit: {name}")
    if lock.get("v29b_policy_sha256") != sha256(V29B_LOCK):
        raise RuntimeError("frozen V29b policy changed after V41 audit")
    for client_id, digest in lock.get("source_stream_sha256", {}).items():
        dataset, unit = client_id.split("::", 1)
        if sha256(V19_STREAMS / f"{dataset}__{unit}.npz") != digest:
            raise RuntimeError(f"V41 source stream changed after audit: {client_id}")
    return lock


def train_source(root: Path) -> None:
    require_audit(root)
    if (root / "target_streams").exists():
        raise RuntimeError("target streams exist before source training")
    rows = source_roster()
    normalized: dict[str, np.ndarray] = {}
    datasets: dict[str, str] = {}
    for dataset, unit, client_id in rows:
        values, _ = source_stream(dataset, unit)
        normalized[client_id] = normalize_device(values)[0][FIT]
        datasets[client_id] = dataset
    model_hashes, traces = {}, {}
    model_dir = root / "models"
    for excluded in sorted(set(datasets.values())):
        clients = [normalized[cid] for cid in sorted(normalized) if datasets[cid] != excluded]
        bundle, trace = _train_bundle(clients, specification()["federated_models"]["seed"])
        path = model_dir / f"fold_without_{excluded}.npz"
        save_bundle(path, bundle)
        model_hashes[path.name] = sha256(path)
        traces[path.stem] = trace
        print(f"V41 TRAIN {path.stem}: {len(clients)} equal-weight clients", flush=True)
    final, trace = _train_bundle([normalized[cid] for cid in sorted(normalized)], specification()["federated_models"]["seed"])
    final_path = model_dir / "all_source.npz"
    save_bundle(final_path, final)
    model_hashes[final_path.name] = sha256(final_path)
    traces[final_path.stem] = trace
    write_json(root / "training_trace.json", traces)
    write_json(root / "v41_model_lock.json", {
        "experiment": "V41", "protocol_sha256": sha256(PROTOCOL),
        "source_clients": 40, "equal_client_weight": True,
        "models": model_hashes, "target_streams_opened": False,
        "target_sensor_values_opened": False, "target_outcomes_opened": False,
        "onset_opened": False,
    })


def require_model_lock(root: Path) -> dict:
    require_audit(root)
    lock = read_json(root / "v41_model_lock.json")
    if lock["protocol_sha256"] != sha256(PROTOCOL) or lock["target_sensor_values_opened"]:
        raise RuntimeError("invalid V41 model lock")
    for name, digest in lock["models"].items():
        if sha256(root / "models" / name) != digest:
            raise RuntimeError(f"V41 model changed: {name}")
    return lock


def k_of_m(flags: np.ndarray, k: int = 4, m: int = 7) -> np.ndarray:
    values = np.asarray(flags, dtype=np.int8).reshape(-1)
    result = np.zeros(len(values), dtype=bool)
    if len(values) >= m:
        cumulative = np.concatenate([[0], np.cumsum(values, dtype=np.int64)])
        result[m - 1:] = cumulative[m:] - cumulative[:-m] >= k
    return result


def ensemble(zscores: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    policy = specification()["frozen_v29b_policy"]
    names = tuple(policy["weights"])
    weights = np.asarray([policy["weights"][name] for name in names], dtype=np.float64)
    matrix = np.stack([zscores[name] for name in names])
    moderate_support = np.sum(weights[:, None] * (matrix >= policy["moderate"]), axis=0)
    reliable = weights >= 0.5 / len(names)
    strong = np.any((matrix >= policy["strong"]) & reliable[:, None], axis=0)
    raw = strong | (moderate_support >= policy["vote_fraction"])
    persistent = k_of_m(raw, int(policy["k"]), int(policy["m"]))
    weighted = np.sum(weights[:, None] * np.maximum(matrix, 0.0), axis=0) / policy["moderate"]
    strong_level = np.max(np.where(reliable[:, None], matrix / policy["strong"], -np.inf), axis=0)
    return np.maximum(weighted, strong_level), persistent


def _raw_branch_scores(bundle: dict, values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "FED_AE_RECON": bundle["ae"].reconstruction_score(values),
        "FED_PCA16": bundle["pca"].score(values),
        "FED_DEEP_SVDD": bundle["svdd"].score(values),
    }


def _single_alarm(calibration: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = models.robust_z(calibration, values)
    return z, k_of_m(z >= 2.0)


def score_device(values: np.ndarray, indices: np.ndarray, bundle: dict) -> dict[str, np.ndarray | int]:
    normalized, _, _ = normalize_device(values)
    raw = _raw_branch_scores(bundle, normalized)
    postcal = {name: raw[name][POSTCAL] for name in BRANCHES}
    eval_slice = slice(GUARD.start, len(values))
    zscores = {name: models.robust_z(postcal[name], raw[name][eval_slice]) for name in BRANCHES}
    continuous, alarm = ensemble(zscores)
    guard_count = GUARD.stop - GUARD.start

    # Fixed secondary comparators; none can alter the primary decision.
    local_pca = models.fit_pca(normalized[FIT], components=min(16, len(normalized[FIT]) - 1))
    local_raw = local_pca.score(normalized)
    local_z, local_alarm = _single_alarm(local_raw[POSTCAL], local_raw[eval_slice])
    random_scores = np.median(np.stack([
        models.initial_ae(seed).reconstruction_score(normalized)
        for seed in models.RANDOM_SEEDS
    ]), axis=0)
    random_z, random_alarm = _single_alarm(random_scores[POSTCAL], random_scores[eval_slice])
    input_raw = np.log1p(np.mean(normalized * normalized, axis=1))
    input_z, input_alarm = _single_alarm(input_raw[POSTCAL], input_raw[eval_slice])
    simple_raw = np.log1p(np.mean(normalized[:, models.SIMPLE_RULE_INDICES] ** 2, axis=1))
    simple_z, simple_alarm = _single_alarm(simple_raw[POSTCAL], simple_raw[eval_slice])

    return {
        "guard_indices": indices[GUARD], "guard_primary": alarm[:guard_count].astype(np.int8),
        "operation_indices": indices[OP_START:], "primary_continuous": continuous[guard_count:],
        "primary_alarm": alarm[guard_count:].astype(np.int8),
        "first_alarm": int(indices[OP_START:][np.flatnonzero(alarm[guard_count:])[0]]) if np.any(alarm[guard_count:]) else -1,
        "branch_z": np.stack([zscores[name][guard_count:] for name in BRANCHES]),
        "local_pca_z": local_z[guard_count:], "local_pca_alarm": local_alarm[guard_count:].astype(np.int8),
        "random_ae_z": random_z[guard_count:], "random_ae_alarm": random_alarm[guard_count:].astype(np.int8),
        "input_profile_z": input_z[guard_count:], "input_profile_alarm": input_alarm[guard_count:].astype(np.int8),
        "simple_rule_z": simple_z[guard_count:], "simple_rule_alarm": simple_alarm[guard_count:].astype(np.int8),
    }


def save_journal(path: Path, client_id: str, result: dict, stream_hash: str, model_hash: str) -> None:
    payload = {key: np.asarray(value) for key, value in result.items()}
    payload.update({
        "client_id": np.asarray(client_id), "stream_sha256": np.asarray(stream_hash),
        "model_sha256": np.asarray(model_hash), "protocol_sha256": np.asarray(sha256(PROTOCOL)),
        "labels_opened": np.asarray(False), "onset_opened": np.asarray(False),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def predict_source(root: Path) -> None:
    require_model_lock(root)
    hashes = {}
    for dataset, unit, client_id in source_roster():
        stream_path = V19_STREAMS / f"{dataset}__{unit}.npz"
        model_path = root / "models" / f"fold_without_{dataset}.npz"
        values, indices = source_stream(dataset, unit)
        journal = root / "source_journals" / f"{dataset}__{unit}.npz"
        save_journal(journal, client_id, score_device(values, indices, load_bundle(model_path)), sha256(stream_path), sha256(model_path))
        hashes[client_id] = sha256(journal)
        print(f"V41 SOURCE PREDICT {client_id}", flush=True)
    write_json(root / "v41_source_prediction_lock.json", {
        "protocol_sha256": sha256(PROTOCOL), "journals": hashes,
        "target_streams_opened": False, "target_sensor_values_opened": False,
        "target_outcomes_opened": False, "onset_opened": False,
    })


def _auprc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8); score = np.asarray(score, dtype=float)
    positives = int(np.sum(y))
    if positives == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable"); labels = y[order]
    tp = np.cumsum(labels); fp = np.cumsum(1 - labels)
    return float(np.sum((tp / np.maximum(tp + fp, 1))[labels == 1]) / positives)


def _metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.int8); pred = np.asarray(pred, dtype=np.int8)
    tp, tn = int(np.sum((y == 1) & (pred == 1))), int(np.sum((y == 0) & (pred == 0)))
    fp, fn = int(np.sum((y == 0) & (pred == 1))), int(np.sum((y == 1) & (pred == 0)))
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); specificity = tn / max(tn + fp, 1)
    return {"n": len(y), "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": (tp + tn) / max(len(y), 1), "balanced_accuracy": 0.5 * (recall + specificity),
            "specificity": specificity, "precision": precision, "recall": recall,
            "F1": 2 * precision * recall / max(precision + recall, 1e-15), "AUPRC": _auprc(y, score)}


def _load_journal(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if bool(archive["labels_opened"]) or bool(archive["onset_opened"]):
            raise RuntimeError("non-blind V41 journal")
        return {key: np.asarray(archive[key]) for key in archive.files}


def score_source(root: Path) -> None:
    lock = read_json(root / "v41_source_prediction_lock.json")
    for client_id, digest in lock["journals"].items():
        dataset, unit = client_id.split("::", 1)
        if sha256(root / "source_journals" / f"{dataset}__{unit}.npz") != digest:
            raise RuntimeError(f"source journal changed: {client_id}")
    evidence = pd.read_csv(SOURCE_EVIDENCE).set_index("client_id")
    ys, ps, ss, rows, guards = [], [], [], [], []
    for _, _, client_id in source_roster():
        dataset, unit = client_id.split("::", 1)
        journal = _load_journal(root / "source_journals" / f"{dataset}__{unit}.npz")
        meta = evidence.loc[client_id]
        indices = journal["operation_indices"].astype(int); pred = journal["primary_alarm"].astype(np.int8)
        first = int(journal["first_alarm"]); guards.append(float(np.mean(journal["guard_primary"])))
        onset = meta.get("first_sustained_onset_acquisition", np.nan)
        event = "not_lifecycle"
        if meta["status"] == "failed" and pd.notna(onset):
            onset = int(onset); terminal = int(math.floor(0.9 * int(meta["acquisitions"])))
            y = (indices >= onset).astype(np.int8); ys.append(y); ps.append(pred); ss.append(journal["primary_continuous"])
            event = "missed" if first < 0 else "premature" if first < onset else "valid" if first < terminal else "late"
        elif meta["status"] == "healthy":
            event = "control_alarm" if first >= 0 else "control_clean"
        rows.append({"client_id": client_id, "dataset": dataset, "status": meta["status"], "first_alarm": first, "event": event, "guard_false_alarm_rate": guards[-1]})
    failed = [row for row in rows if row["event"] in {"missed", "premature", "valid", "late"}]
    controls = [row for row in rows if row["event"].startswith("control_")]
    pooled = _metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(ss))
    gate_values = {
        "pointwise_F1": pooled["F1"],
        "preterminal_detection_rate": sum(row["event"] in {"premature", "valid"} for row in failed) / max(len(failed), 1),
        "interval_consistent_detection_rate": sum(row["event"] == "valid" for row in failed) / max(len(failed), 1),
        "documented_healthy_control_alarm_rate": sum(row["event"] == "control_alarm" for row in controls) / max(len(controls), 1),
        "mean_guard_false_alarm_rate": float(np.mean(guards)),
    }
    limits = specification()["source_release_gate"]
    conditions = {
        "pointwise_F1": gate_values["pointwise_F1"] >= limits["pointwise_F1_minimum"],
        "preterminal_detection_rate": gate_values["preterminal_detection_rate"] >= limits["preterminal_detection_rate_minimum"],
        "interval_consistent_detection_rate": gate_values["interval_consistent_detection_rate"] >= limits["interval_consistent_detection_rate_minimum"],
        "documented_healthy_control_alarm_rate": gate_values["documented_healthy_control_alarm_rate"] <= limits["documented_healthy_control_alarm_rate_maximum"],
        "mean_guard_false_alarm_rate": gate_values["mean_guard_false_alarm_rate"] <= limits["mean_guard_false_alarm_rate_maximum"],
    }
    payload = {"source_prediction_lock_sha256": sha256(root / "v41_source_prediction_lock.json"), "pointwise": pooled,
               "event_counts": {name: sum(row["event"] == name for row in rows) for name in ("premature", "valid", "late", "missed", "control_alarm", "control_clean")},
               "gate_values": gate_values, "gate_conditions": conditions, "passed": all(conditions.values()),
               "target_streams_opened": False, "target_sensor_values_opened": False}
    write_json(root / "v41_source_gate.json", payload)
    pd.DataFrame(rows).to_csv(root / "v41_source_per_device.csv", index=False)
    if not payload["passed"]:
        print("V41 SOURCE SAFETY STOP: gate failed; fresh target sensor values remain unopened", flush=True)
    else:
        print("V41 SOURCE GATE PASSED: fresh target stream phase is authorized", flush=True)


def require_source_release(root: Path) -> dict:
    gate = read_json(root / "v41_source_gate.json")
    if not gate.get("passed") or gate["source_prediction_lock_sha256"] != sha256(root / "v41_source_prediction_lock.json"):
        raise RuntimeError("V41 source gate did not authorize fresh target access")
    require_model_lock(root)
    return gate


def streams_target(root: Path) -> None:
    require_source_release(root)
    manifests = []
    for device in fresh.target_devices():
        path = root / "target_streams" / f"dlr__{device.unit}.npz"
        manifests.append(fresh.save_feature_stream(path, device))
        print(f"V41 TARGET STREAM {device.client_id}: {manifests[-1]['acquisitions']} acquisitions", flush=True)
    write_json(root / "v41_target_stream_lock.json", {
        "protocol_sha256": sha256(PROTOCOL), "clients": manifests,
        "source_gate_sha256": sha256(root / "v41_source_gate.json"),
        "target_sensor_values_opened": True, "target_outcomes_opened": False, "onset_opened": False,
    })


def predict_target(root: Path) -> None:
    require_source_release(root)
    stream_lock = read_json(root / "v41_target_stream_lock.json")
    model_path = root / "models" / "all_source.npz"; bundle = load_bundle(model_path)
    hashes = {}
    for device in fresh.target_devices():
        stream_path = root / "target_streams" / f"dlr__{device.unit}.npz"
        values, indices = fresh.load_feature_stream(stream_path)
        journal = root / "target_journals" / f"dlr__{device.unit}.npz"
        save_journal(journal, device.client_id, score_device(values, indices, bundle), sha256(stream_path), sha256(model_path))
        hashes[device.client_id] = sha256(journal)
        print(f"V41 TARGET PREDICT {device.client_id}", flush=True)
    write_json(root / "v41_target_prediction_lock.json", {
        "protocol_sha256": sha256(PROTOCOL), "journals": hashes,
        "target_stream_lock_sha256": sha256(root / "v41_target_stream_lock.json"),
        "target_policy_tuning": False, "target_rescue_rerun": False,
        "target_outcomes_opened": False, "onset_opened": False,
    })


def _health_trajectory(values: np.ndarray) -> np.ndarray:
    normalized, _, _ = normalize_device(values)
    return np.sqrt(np.mean(normalized * normalized, axis=1)).astype(np.float64)


def analytical_onset_interval(values: np.ndarray) -> dict[str, object]:
    """Fixed post-lock analytical proxy; accepts input features, never detector scores."""
    h = _health_trajectory(values); n = len(h); start = OP_START
    baseline = h[:OP_START]; center, scale = models.scalar_center_scale(baseline)
    z = (h - center) / scale
    candidates: list[int] = []
    sustained = np.flatnonzero(k_of_m(z >= 4.0, 4, 7) & (np.arange(n) >= start))
    if len(sustained): candidates.append(int(sustained[0]))
    ewma = np.zeros(n); state = 0.0
    for i, value in enumerate(z):
        state = 0.2 * float(value) + 0.8 * state; ewma[i] = state
    found = np.flatnonzero(k_of_m(ewma >= 3.0, 4, 7) & (np.arange(n) >= start))
    if len(found): candidates.append(int(found[0]))
    window = max(5, min(20, n // 20)); effect = np.zeros(n)
    for i in range(start + window, n):
        effect[i] = (float(np.median(h[i-window+1:i+1])) - center) / scale
    found = np.flatnonzero(k_of_m(effect >= 3.0, 4, 7) & (np.arange(n) >= start))
    if len(found): candidates.append(int(found[0]))
    upper = max(start + 1, int(math.floor(0.9 * n)))
    if upper - start >= 10:
        scores = []
        for cut in range(start + 5, upper):
            before = h[max(0, cut - window):cut]; after = h[cut:min(n, cut + window)]
            scores.append((float(np.median(after) - np.median(before)) / scale, cut))
        best_score, best_cut = max(scores)
        if best_score >= 3.0: candidates.append(int(best_cut))
    candidates = sorted(set(candidates))
    if not candidates:
        return {"available": False, "methods": [], "lower": None, "consensus": None, "upper": None}
    return {"available": True, "methods": candidates, "lower": min(candidates),
            "consensus": int(round(float(np.median(candidates)))), "upper": max(candidates)}


def _bootstrap_event(rows: list[dict], seed: int = 20260804, draws: int = 5000) -> dict[str, list[float]]:
    if not rows: return {}
    rng = np.random.default_rng(seed); values = []
    for _ in range(draws):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        values.append(np.mean([row["interval_consistent"] for row in sample]))
    return {"interval_consistent_rate_95_CI": [float(np.quantile(values, .025)), float(np.quantile(values, .975))]}


def score_target(root: Path) -> None:
    lock_path = root / "v41_target_prediction_lock.json"; lock = read_json(lock_path)
    if lock["target_policy_tuning"] or lock["target_rescue_rerun"]:
        raise RuntimeError("V41 target lock permits tuning")
    method_fields = {
        "FED_TRIAD_PRIMARY": ("primary_alarm", "primary_continuous"),
        "LOCAL_PCA16": ("local_pca_alarm", "local_pca_z"),
        "RANDOM_AE_MEDIAN": ("random_ae_alarm", "random_ae_z"),
        "INPUT_PROFILE_RULE": ("input_profile_alarm", "input_profile_z"),
        "RMS_KURTOSIS_CREST_RULE": ("simple_rule_alarm", "simple_rule_z"),
    }
    rows, method_vectors = [], {name: {"y": [], "p": [], "s": [], "events": []} for name in method_fields}
    for device in fresh.target_devices():
        journal_path = root / "target_journals" / f"dlr__{device.unit}.npz"
        if sha256(journal_path) != lock["journals"][device.client_id]:
            raise RuntimeError(f"target journal changed: {device.client_id}")
        journal = _load_journal(journal_path)
        values, _ = fresh.load_feature_stream(root / "target_streams" / f"dlr__{device.unit}.npz")
        onset = analytical_onset_interval(values); indices = journal["operation_indices"].astype(int)
        first = int(journal["first_alarm"]); terminal = int(math.floor(0.9 * len(values)))
        row = {"client_id": device.client_id, "test": device.test, "bearing": device.bearing,
               "acquisitions": len(values), "first_alarm": first,
               "guard_clean": not bool(np.any(journal["guard_primary"])), **{f"onset_{k}": v for k, v in onset.items()}}
        if onset["available"]:
            consensus = int(onset["consensus"]); tolerance = max(5, int(math.ceil(.05 * (len(values) - consensus))))
            y = (indices >= consensus).astype(np.int8)
            row["interval_consistent"] = bool(first >= int(onset["lower"]) and first <= int(onset["upper"]) + tolerance and first < terminal)
            row["preterminal_detected"] = bool(first >= 0 and first < terminal)
            row["premature_before_lower"] = bool(0 <= first < int(onset["lower"]))
            row["missed"] = bool(first < 0); row["detection_delay_from_consensus"] = first - consensus if first >= 0 else ""
        else:
            row.update({"interval_consistent": False, "preterminal_detected": bool(0 <= first < terminal),
                        "premature_before_lower": False, "missed": first < 0, "detection_delay_from_consensus": ""})
        for method, (alarm_field, score_field) in method_fields.items():
            pred = journal[alarm_field].astype(np.int8); score = journal[score_field].astype(float)
            positions = np.flatnonzero(pred); method_first = int(indices[positions[0]]) if len(positions) else -1
            row[f"{method}_first_alarm"] = method_first
            if onset["available"]:
                method_vectors[method]["y"].append(y); method_vectors[method]["p"].append(pred); method_vectors[method]["s"].append(score)
                event = "missed" if method_first < 0 else "premature" if method_first < int(onset["lower"]) else "valid" if method_first <= int(onset["upper"]) + tolerance and method_first < terminal else "late"
                method_vectors[method]["events"].append(event)
        rows.append(row)
    eligible = [row for row in rows if row["onset_available"]]
    method_summary = {}
    for method, vectors in method_vectors.items():
        pointwise = _metrics(np.concatenate(vectors["y"]), np.concatenate(vectors["p"]), np.concatenate(vectors["s"])) if vectors["y"] else {}
        method_summary[method] = {"pointwise": pointwise, "event_counts": {name: vectors["events"].count(name) for name in ("premature", "valid", "late", "missed")}}
    pooled = method_summary["FED_TRIAD_PRIMARY"]["pointwise"]
    pairs = []
    for test in (2, 3, 5):
        pair = [row for row in rows if row["test"] == test]
        pairs.append({"test": test, "either_bearing_interval_consistent": any(row["interval_consistent"] for row in pair),
                      "either_bearing_preterminal": any(row["preterminal_detected"] for row in pair),
                      "both_guards_clean": all(row["guard_clean"] for row in pair)})
    summary = {
        "classification": "single frozen prospective run on six previously unopened DLR bearings",
        "prediction_lock_sha256": sha256(lock_path), "pointwise_pooled_analytical_consensus": pooled,
        "method_comparison": method_summary,
        "onset_eligible_devices": len(eligible), "fresh_devices": 6,
        "guard_clean": sum(row["guard_clean"] for row in rows),
        "interval_consistent": sum(row["interval_consistent"] for row in eligible),
        "preterminal_detected": sum(row["preterminal_detected"] for row in rows),
        "premature_before_analytical_lower_bound": sum(row["premature_before_lower"] for row in eligible),
        "missed": sum(row["missed"] for row in rows), "device_cluster_bootstrap": _bootstrap_event(eligible),
        "pair_level": pairs, "target_policy_tuning": False, "target_rescue_rerun": False,
        "onset_caveat": "analytical consensus is an exploratory proxy derived after prediction lock, not physical ground truth"
    }
    write_json(root / "v41_final_metrics.json", summary)
    pd.DataFrame(rows).to_csv(root / "v41_target_per_device.csv", index=False)
    lines = ["# V41 fresh DLR confirmation results", "", summary["classification"] + ".", "",
             "The alarm policy, source models, source gate, and target prediction journals were locked before the analytical onset proxy was computed. No target tuning or rescue rerun is permitted.", "",
             "## Primary results", "", "| Metric | Result |", "|---|---:|",
             f"| Fresh physical-bearing clients | 6 |", f"| Clean commissioning guards | {summary['guard_clean']}/6 |",
             f"| Analytical-onset eligible | {len(eligible)}/6 |",
             f"| Interval-consistent detections | {summary['interval_consistent']}/{len(eligible)} |",
             f"| Preterminal detections | {summary['preterminal_detected']}/6 |",
             f"| Premature before analytical lower bound | {summary['premature_before_analytical_lower_bound']}/{len(eligible)} |",
             f"| Missed | {summary['missed']}/6 |"]
    if pooled:
        for name in ("accuracy", "balanced_accuracy", "specificity", "precision", "recall", "F1", "AUPRC"):
            lines.append(f"| Pointwise {name} | {100 * pooled[name]:.2f}% |")
    lines += ["", "## Frozen comparator results", "", "| Method | F1 | AUPRC | Valid | Premature | Late | Missed |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, result in method_summary.items():
        point = result["pointwise"]; events = result["event_counts"]
        lines.append(f"| {method} | {100*point['F1']:.2f}% | {100*point['AUPRC']:.2f}% | {events['valid']} | {events['premature']} | {events['late']} | {events['missed']} |")
    lines += ["", "## Scientific boundary", "", summary["onset_caveat"] + ". DLR test termination is pair-level evidence and does not identify which bearing initiated failure; both bearing-level and pair-level results are therefore retained."]
    (root / "V41_FINAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(root / "V41_FINAL_RESULTS.md", flush=True)


def verify(root: Path) -> None:
    source = read_json(root / "v41_source_gate.json")
    if not source["passed"]:
        print("V41 verified source safety stop; target remained sealed", flush=True); return
    final = read_json(root / "v41_final_metrics.json")
    if final["target_policy_tuning"] or final["target_rescue_rerun"] or final["fresh_devices"] != 6:
        raise RuntimeError("invalid V41 final integrity")
    print("V41 integrity verified: six fresh clients, no target tuning or rescue", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True, choices=("audit", "train-source", "predict-source", "score-source", "streams-target", "predict-target", "score-target", "verify"))
    args = parser.parse_args(); root = root_for(args.run_id)
    actions = {"audit": audit, "train-source": train_source, "predict-source": predict_source,
               "score-source": score_source, "streams-target": streams_target, "predict-target": predict_target,
               "score-target": score_target, "verify": verify}
    actions[args.phase](root)


if __name__ == "__main__":
    main()
