"""Thesis-grade device-training and frozen-server federated anomaly study.

This runner models datasets stored on physical LilyGO SD cards. Every device is
a fixed installation site assigned to exactly one prepared dataset/client
domain for its entire lifetime; a device never alternates between datasets.
The existing test partitions never enter a device training stream and remain
untouched at the server until every federated round ends.  Deployment scoring
then uses a fixed operator-confirmed healthy prefix for calibration and applies
trusted-only predict-then-adapt updates without changing model weights.

The default ``thesis`` profile evaluates six controlled variants. It isolates
FedAvg/FedAdam/FedYogi server aggregation at rank 1, then compares rank-1 LoRA
with rank-2 rank-stabilized LoRA and trusted-only learning with train-all. This
provides interpretable ablations without an impractical full-factorial sweep.

Two device-side learning policies appear in those variants:

``trusted``
    Every window is predicted, but only confidently normal windows may update
    TinyFL, online normalization, calibration, or replay.
``all``
    Every device-pool window produces a TinyFL training candidate. Online
    calibration remains trusted-only so anomaly thresholds are not deliberately
    contaminated. This is the literal train-all comparison requested for the
    study.

Labels are used by a post-partition protocol audit and attached to records for
final metrics. They are never read by allocation, normalization, calibration,
model training, federation, replay, thresholding, or alarms.

Example::

    python realtime_federated_simulation.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_DIR = SCRIPT_DIR.parent
PYTHON_TOOLS_DIR = RELEASE_DIR / "python_tools"
DATASETS_ROOT = RELEASE_DIR.parent / "datasets"
if str(PYTHON_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_TOOLS_DIR))

from cwru_eval import classification_metrics, load_domain, load_normalization  # noqa: E402
from improved_comprehensive_experiment import (  # noqa: E402
    ALL_DATASETS,
    DISPLAY_NAMES,
    OnlineWelford,
    _prepared_to_raw,
)
from learning_failure_experiments import (  # noqa: E402
    ABLoRA,
    DenseAE,
    LARGER_DIMS,
    _flat_aggregate_fedavg,
    _init_base_weights,
    _sparse_int8_reconstruction,
    load_base_npz,
    save_base_npz,
    stable_seed,
)
from run_full_simulation import CLIENTS, validate_domain  # noqa: E402
from tiny_ae import dequantize_dynamic, quantize_dynamic  # noqa: E402


LDM_DATASET = "ldm_upb"
LDM_MACHINES = ("B06", "B13", "B14", "B15")
LDM_FEATURE_ROOT = (
    SCRIPT_DIR
    / "results"
    / "ldm_reserved_cohort_external_frozen1"
    / "features"
)
LDM_FEATURE_LOCK_ROOT = (
    SCRIPT_DIR
    / "results"
    / "ldm_reserved_cohort_external_frozen1"
    / "feature_locks"
)
LDM_TRANSITION_PATH = SCRIPT_DIR / "ldm_reserved_cohort_transitions_after_lock.json"
LDM_EXTENSION_PROTOCOL_PATH = (
    SCRIPT_DIR / "realtime_federated_ldm_extension_protocol.json"
)
LDM_EXTENSION_PROTOCOL_V2_PATH = (
    SCRIPT_DIR / "realtime_federated_ldm_extension_protocol_v2.json"
)
LDM_EXTENSION_PROTOCOL_V3_PATH = (
    SCRIPT_DIR / "realtime_federated_ldm_extension_protocol_v3.json"
)
LDM_EXTENSION_PROTOCOL_V4_PATH = (
    SCRIPT_DIR / "realtime_federated_ldm_extension_protocol_v4.json"
)
LDM_EXTENSION_PROTOCOL_V4R1_PATH = (
    SCRIPT_DIR / "realtime_federated_ldm_extension_protocol_v4r1.json"
)
LDM_EXTENSION_PROTOCOL_V4R2_PATH = (
    SCRIPT_DIR / "realtime_federated_ldm_extension_protocol_v4r2.json"
)
LDM_EXTENSION_PROTOCOL_V5_PATH = (
    SCRIPT_DIR / "realtime_federated_ldm_extension_protocol_v5.json"
)
LDM_DENSE_TRAIN_WINDOWS = 1_400
LDM_OPERATION_WINDOWS = 1_000
LDM_ENVELOPE_CALIBRATION_WINDOWS = 1_000
VALID_LDM_SPLIT_STRATEGIES = {
    "fixed_prefix_v1",
    "broad_healthy_v2",
    "known_healthy_envelope_v3",
    "stratified_healthy_v4",
    "stratified_healthy_v4r1",
    "stratified_healthy_v4r2",
    "stratified_healthy_v5_condition_aware",
}

RUN_TO_FAILURE_DATASETS = {"ims", "xjtu", "pronostia", LDM_DATASET}
# IMS and PRONOSTIA are progressive, non-recovering degradation studies.  Their
# early faults are weak in individual 512-sample windows, so a causal trend
# detector is appropriate.  XJTU is intentionally excluded: its labelled
# normal phase contains large operating drift, for which an irreversible latch
# would be unsafe.  This policy is fixed by study protocol, never selected from
# a held-out label during evaluation.
PROGNOSTIC_LATCH_DATASETS = {"ims", "pronostia", LDM_DATASET}
VALID_POLICIES = {"trusted", "all"}
VALID_SERVER_OPTIMIZERS = {"fedavg", "fedadam", "fedyogi"}
VALID_STUDY_PROFILES = {"quick", "quick4", "recommended", "thesis"}


@dataclass(frozen=True)
class RealtimeConfig:
    datasets: tuple[str, ...] = ALL_DATASETS
    seeds: tuple[int, ...] = (11, 29, 47, 71, 97)
    study_profile: str = "thesis"
    num_devices: int = 10
    include_ldm_clients: bool = False
    ldm_split_strategy: str = "fixed_prefix_v1"
    calibration_windows: int = 100
    deployment_calibration_windows: int = 200
    deployment_calibration_fraction: float = 0.20
    minimum_calibration_windows: int = 20
    commissioning_batch: int = 60
    commissioning_local_epochs: int = 2
    commissioning_lr: float = 0.01
    operation_block: int = 30
    anchor_replay_size: int = 64
    recent_replay_size: int = 64
    minimum_replay_size: int = 32
    local_epochs: int = 1
    local_lr: float = 0.005
    fedprox_mu: float = 0.001
    personal_prox_mu: float = 0.01
    personal_rate: float = 0.10
    personal_norm_limit: float = 4.0
    normalizer_alpha: float = 0.001
    score_buffer_size: int = 128
    ae_fusion_weight: float = 0.75
    mahalanobis_fusion_weight: float = 0.40
    run_to_failure_mahalanobis_weight: float = 0.70
    covariance_shrinkage: float = 0.05
    prognostic_latch: bool = True
    prognostic_window: int = 20
    prognostic_history: int = 15
    prognostic_required: int = 10
    prognostic_sigma: float = 6.0
    prognostic_score_buffer_size: int = 256
    prognostic_covariance_shrinkage: float = 0.10
    trust_quantile: float = 0.90
    alarm_quantile: float = 0.995
    trust_sigma: float = 2.0
    alarm_sigma: float = 4.0
    temporal_window: int = 5
    temporal_required: int = 3
    recovery_required: int = 8
    cusum_decay: float = 0.90
    cusum_allowance: float = 0.50
    cusum_limit: float = 8.0
    cusum_maximum: float = 16.0
    cusum_clip: float = 3.0
    minimum_clients_per_round: int = 1
    lora_alpha: float = 0.5
    server_learning_rate: float = 0.005
    server_beta1: float = 0.9
    server_beta2: float = 0.99
    server_tau: float = 0.001
    firmware_compression: bool = True
    sparsity_k: int = 32
    error_feedback: bool = True


@dataclass(frozen=True)
class StudyVariant:
    """One controlled thesis ablation; avoids an impractical full factorial."""

    name: str
    policy: str
    server_optimizer: str
    rank: int
    rank_stabilized: bool
    noise_sigma: float = 0.0


def study_variants(profile: str) -> tuple[StudyVariant, ...]:
    variants = {
        "quick": (
            StudyVariant("fedadam_rs_r2_trusted", "trusted", "fedadam", 2, True),
        ),
        "quick4": (
            StudyVariant(
                "fedadam_rs_r4_denoise", "trusted", "fedadam", 4, True, 0.02
            ),
        ),
        "recommended": (
            StudyVariant("fedadam_rs_r2_trusted", "trusted", "fedadam", 2, True),
            StudyVariant("fedadam_rs_r2_all", "all", "fedadam", 2, True),
            StudyVariant(
                "fedadam_rs_r4_denoise", "trusted", "fedadam", 4, True, 0.02
            ),
        ),
        "thesis": (
            # Optimizer ablation at identical adapter/policy settings.
            StudyVariant("fedavg_r1_trusted", "trusted", "fedavg", 1, False),
            StudyVariant("fedadam_r1_trusted", "trusted", "fedadam", 1, False),
            StudyVariant("fedyogi_r1_trusted", "trusted", "fedyogi", 1, False),
            # Adapter-rank and train-all ablations around the recommended stack.
            StudyVariant("fedadam_rs_r2_trusted", "trusted", "fedadam", 2, True),
            StudyVariant("fedadam_rs_r2_all", "all", "fedadam", 2, True),
            StudyVariant(
                "fedadam_rs_r4_denoise", "trusted", "fedadam", 4, True, 0.02
            ),
        ),
    }
    if profile not in variants:
        raise ValueError(f"unknown study profile: {profile}")
    return variants[profile]


def _constructor_alpha(config: RealtimeConfig, variant: StudyVariant) -> float:
    # ABLoRA internally applies constructor_alpha/rank. Multiplying the user
    # alpha by sqrt(rank) therefore yields rsLoRA's alpha/sqrt(rank).
    if variant.rank_stabilized:
        return float(config.lora_alpha * np.sqrt(float(variant.rank)))
    return float(config.lora_alpha)


def _study_signature(config: RealtimeConfig) -> str:
    relevant = {
        "dense_schema": 5,
        "site_layout": "one_dataset_client_domain_per_device_synchronized_v2",
        "datasets": config.datasets,
        "num_devices": config.num_devices,
        "include_ldm_clients": config.include_ldm_clients,
        "ldm_split_strategy": config.ldm_split_strategy,
        "ldm_extension_schema": (
            config.ldm_split_strategy
            if config.include_ldm_clients
            else None
        ),
        "calibration_windows": config.calibration_windows,
        "commissioning_batch": config.commissioning_batch,
        "commissioning_local_epochs": config.commissioning_local_epochs,
        "commissioning_lr": config.commissioning_lr,
        "architecture": LARGER_DIMS,
    }
    encoded = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class StreamWindow:
    dataset: str
    logical_client: str
    raw_features: np.ndarray
    split: str
    label: int
    source_file: str
    start_sample: int
    window_id: str

    @property
    def domain_key(self) -> str:
        return f"{self.dataset}:{self.logical_client}"


@dataclass
class DeviceDomainStream:
    device_id: str
    dataset: str
    logical_client: str
    initial_mean: np.ndarray
    initial_std: np.ndarray
    calibration_seed: list[StreamWindow]
    train: list[StreamWindow]
    operation: list[StreamWindow]
    server_test: list[StreamWindow]
    calibration_replay_allowed: bool = True

    @property
    def domain_key(self) -> str:
        return f"{self.dataset}:{self.logical_client}"

    @property
    def context_key(self) -> str:
        return f"{self.device_id}:{self.domain_key}"


@dataclass
class DeviceShard:
    device_id: str
    streams: dict[str, DeviceDomainStream]


def _natural_parts(value: str) -> tuple[tuple[int, object], ...]:
    """Natural source order: timestamps and numbered files sort correctly."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
        if part != ""
    )


def _window_sort_key(window: StreamWindow) -> tuple[object, ...]:
    return (*_natural_parts(window.source_file), int(window.start_sample), window.window_id)


def _client_display_name(dataset: str, logical_client: str) -> str:
    if dataset == LDM_DATASET:
        return f"LDM-UPB {logical_client}"
    return DISPLAY_NAMES[(dataset, logical_client)]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _prepared_bin_dir(dataset: str) -> Path:
    canonical = DATASETS_ROOT / dataset / f"{dataset}_bin"
    if canonical.is_dir():
        return canonical
    legacy = PYTHON_TOOLS_DIR / f"{dataset}_bin"
    if legacy.is_dir():
        return legacy
    raise FileNotFoundError(
        f"missing prepared dataset directory; checked {canonical} and {legacy}"
    )


def _ldm_protocol_path(strategy: str) -> Path:
    paths = {
        "fixed_prefix_v1": LDM_EXTENSION_PROTOCOL_PATH,
        "broad_healthy_v2": LDM_EXTENSION_PROTOCOL_V2_PATH,
        "known_healthy_envelope_v3": LDM_EXTENSION_PROTOCOL_V3_PATH,
        "stratified_healthy_v4": LDM_EXTENSION_PROTOCOL_V4_PATH,
        "stratified_healthy_v4r1": LDM_EXTENSION_PROTOCOL_V4R1_PATH,
        "stratified_healthy_v4r2": LDM_EXTENSION_PROTOCOL_V4R2_PATH,
        "stratified_healthy_v5_condition_aware": LDM_EXTENSION_PROTOCOL_V5_PATH,
    }
    try:
        return paths[strategy]
    except KeyError as exc:
        raise ValueError(f"unsupported LDM split strategy: {strategy}") from exc


def _rows_as_windows(
    dataset: str,
    logical_client: str,
    values: np.ndarray,
    rows: list[dict[str, object]],
    prepared_mean: np.ndarray,
    prepared_std: np.ndarray,
) -> list[StreamWindow]:
    raw = _prepared_to_raw(values, prepared_mean, prepared_std)
    return [
        StreamWindow(
            dataset=dataset,
            logical_client=logical_client,
            raw_features=np.asarray(features, dtype=np.float32),
            split=str(row["split"]),
            label=int(row["label"]),
            source_file=str(row["source_file"]),
            start_sample=int(row["start_sample"]),
            window_id=str(row["window_id"]),
        )
        for features, row in zip(raw, rows)
    ]


def _operation_order(
    dataset: str, by_split: dict[str, list[StreamWindow]]
) -> list[StreamWindow]:
    """Restore causal order without using the shuffled prepared row order."""
    if dataset in RUN_TO_FAILURE_DATASETS:
        combined = [
            window
            for split in ("train", "calibration", "val", "test")
            for window in by_split.get(split, [])
        ]
        return sorted(combined, key=_window_sort_key)
    return [
        window
        for split in ("train", "calibration", "val", "test")
        for window in sorted(by_split.get(split, []), key=_window_sort_key)
    ]


def _fixed_site_assignments(
    datasets: tuple[str, ...], num_devices: int
) -> tuple[tuple[str, str, str], ...]:
    """Map each physical device to one immutable dataset/client site."""
    if num_devices < 1:
        raise ValueError("num_devices must be positive")
    available_sites = [
        (dataset, logical_client)
        for dataset in datasets
        for logical_client in CLIENTS
    ]
    if num_devices > len(available_sites):
        raise ValueError(
            f"{num_devices} devices requested but {len(available_sites)} fixed "
            "dataset/client sites are available; add datasets/sites rather "
            "than mixing several domains on one device"
        )
    return tuple(
        (f"device_{index + 1}", dataset, logical_client)
        for index, (dataset, logical_client) in enumerate(
            available_sites[:num_devices]
        )
    )


def _initial_statistics(
    windows: list[StreamWindow], config: RealtimeConfig, context: str
) -> tuple[np.ndarray, np.ndarray, list[StreamWindow]]:
    count = min(config.calibration_windows, len(windows))
    if count < config.minimum_calibration_windows:
        raise ValueError(
            f"{context} has only {len(windows)} known-healthy training windows; "
            f"at least {config.minimum_calibration_windows} are required"
        )
    seed_windows = sorted(windows, key=_window_sort_key)[:count]
    estimator = OnlineWelford(width=12)
    for window in seed_windows:
        estimator.update(window.raw_features)
    initial_mean, initial_std = estimator.statistics()
    initial_std = np.where(np.asarray(initial_std) < 1e-6, 1.0, initial_std)
    return (
        np.asarray(initial_mean, dtype=np.float32),
        np.asarray(initial_std, dtype=np.float32),
        seed_windows,
    )


def _statistics_from_calibration_seed(
    windows: list[StreamWindow], config: RealtimeConfig, context: str
) -> tuple[np.ndarray, np.ndarray]:
    if len(windows) < config.minimum_calibration_windows:
        raise ValueError(
            f"{context} has only {len(windows)} calibration-only healthy windows; "
            f"at least {config.minimum_calibration_windows} are required"
        )
    estimator = OnlineWelford(width=12)
    for window in windows:
        estimator.update(window.raw_features)
    initial_mean, initial_std = estimator.statistics()
    initial_std = np.where(np.asarray(initial_std) < 1e-6, 1.0, initial_std)
    return (
        np.asarray(initial_mean, dtype=np.float32),
        np.asarray(initial_std, dtype=np.float32),
    )


def _representative_calibration_windows(
    windows: list[StreamWindow], capacity: int
) -> list[StreamWindow]:
    """Bound calibration memory without collapsing it to the chronological tail."""
    ordered = sorted(windows, key=_window_sort_key)
    if capacity <= 0:
        raise ValueError("calibration capacity must be positive")
    if len(ordered) <= capacity:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, capacity, dtype=np.int64)
    if len(np.unique(positions)) != capacity:
        raise ValueError("representative calibration positions are not unique")
    return [ordered[int(position)] for position in positions]


def _ldm_split_boundaries(
    n: int,
    transition: int,
    calibration_windows: int,
    dense_train_windows: int = LDM_DENSE_TRAIN_WINDOWS,
    operation_windows: int = LDM_OPERATION_WINDOWS,
) -> dict[str, tuple[int, int]]:
    """Fixed chronological LDM split; labels never choose train/calibration sizes."""
    values = (n, transition, calibration_windows, dense_train_windows, operation_windows)
    if any(int(value) <= 0 for value in values):
        raise ValueError("LDM split sizes and transition must be positive")
    calibration_end = int(calibration_windows)
    train_end = calibration_end + int(dense_train_windows)
    operation_end = train_end + int(operation_windows)
    if not operation_end < int(transition) < int(n):
        raise ValueError(
            "LDM chronology must retain healthy server-test windows and a "
            "non-empty post-transition test segment"
        )
    return {
        "calibration_seed": (0, calibration_end),
        "dense_train": (calibration_end, train_end),
        "healthy_operation": (train_end, operation_end),
        "server_test": (operation_end, int(n)),
        "server_healthy": (operation_end, int(transition)),
        "server_unhealthy": (int(transition), int(n)),
    }


def _evenly_spaced_indices(start: int, end: int, count: int) -> np.ndarray:
    """Select a deterministic chronological subset from [start, end)."""
    if count <= 0 or end - start < count:
        raise ValueError("evenly spaced LDM range is too short for requested count")
    indices = np.linspace(start, end - 1, count, dtype=np.int64)
    if len(np.unique(indices)) != count:
        raise ValueError("evenly spaced LDM indices are not unique")
    return indices


def _ldm_split_indices(
    n: int,
    transition: int,
    calibration_windows: int,
    strategy: str,
) -> dict[str, np.ndarray]:
    if strategy not in VALID_LDM_SPLIT_STRATEGIES:
        raise ValueError(f"unsupported LDM split strategy: {strategy}")
    if strategy == "fixed_prefix_v1":
        boundaries = _ldm_split_boundaries(n, transition, calibration_windows)
        return {
            key: np.arange(start, end, dtype=np.int64)
            for key, (start, end) in boundaries.items()
        }

    if strategy in {
        "stratified_healthy_v4r2",
        "stratified_healthy_v5_condition_aware",
    }:
        healthy = np.arange(transition, dtype=np.int64)
        shuffled = np.random.default_rng(20_260_901).permutation(healthy)
        calibration_count = int(np.floor(0.20 * len(healthy)))
        healthy_test_count = int(np.floor(0.20 * len(healthy)))
        calibration = np.sort(shuffled[:calibration_count])
        server_healthy = np.sort(
            shuffled[
                calibration_count : calibration_count + healthy_test_count
            ]
        )
        training_pool = np.sort(
            shuffled[calibration_count + healthy_test_count :]
        )
        dense_positions = _evenly_spaced_indices(
            0, len(training_pool), LDM_DENSE_TRAIN_WINDOWS
        )
        dense_train = training_pool[dense_positions]
        training_pool = np.setdiff1d(
            training_pool, dense_train, assume_unique=True
        )
        operation_positions = _evenly_spaced_indices(
            0, len(training_pool), LDM_OPERATION_WINDOWS
        )
        healthy_operation = training_pool[operation_positions]
        server_unhealthy = np.arange(transition, n, dtype=np.int64)
        server_test = np.concatenate((server_healthy, server_unhealthy))
        groups = (
            calibration,
            dense_train,
            healthy_operation,
            server_healthy,
            server_unhealthy,
        )
        combined = np.concatenate(groups)
        if len(np.unique(combined)) != len(combined):
            raise ValueError("fixed-shuffle LDM partitions overlap")
        if any(np.any(group >= transition) for group in groups[:4]):
            raise ValueError("fixed-shuffle LDM healthy partition crosses transition")
        return {
            "calibration_seed": calibration,
            "dense_train": dense_train,
            "healthy_operation": healthy_operation,
            "server_test": server_test,
            "server_healthy": server_healthy,
            "server_unhealthy": server_unhealthy,
        }

    if strategy in {"stratified_healthy_v4", "stratified_healthy_v4r1"}:
        healthy = np.arange(transition, dtype=np.int64)
        calibration = healthy[healthy % 5 == 0]
        server_healthy = healthy[healthy % 5 == 1]
        training_pool = healthy[np.isin(healthy % 5, (2, 3, 4))]
        dense_positions = _evenly_spaced_indices(
            0, len(training_pool), LDM_DENSE_TRAIN_WINDOWS
        )
        dense_train = training_pool[dense_positions]
        training_pool = np.setdiff1d(
            training_pool, dense_train, assume_unique=True
        )
        operation_positions = _evenly_spaced_indices(
            0, len(training_pool), LDM_OPERATION_WINDOWS
        )
        healthy_operation = training_pool[operation_positions]
        server_unhealthy = np.arange(transition, n, dtype=np.int64)
        server_test = np.concatenate((server_healthy, server_unhealthy))
        groups = (
            calibration,
            dense_train,
            healthy_operation,
            server_healthy,
            server_unhealthy,
        )
        combined = np.concatenate(groups)
        if len(np.unique(combined)) != len(combined):
            raise ValueError("stratified LDM partitions overlap")
        if any(np.any(group >= transition) for group in groups[:4]):
            raise ValueError("stratified LDM healthy partition crosses transition")
        if not np.array_equal(server_unhealthy, np.arange(transition, n)):
            raise ValueError("stratified LDM test omits unhealthy acquisitions")
        return {
            "calibration_seed": calibration,
            "dense_train": dense_train,
            "healthy_operation": healthy_operation,
            "server_test": server_test,
            "server_healthy": server_healthy,
            "server_unhealthy": server_unhealthy,
        }

    if strategy == "known_healthy_envelope_v3":
        server_start = int(np.floor(0.80 * transition))
        calibration = _evenly_spaced_indices(
            0,
            server_start,
            LDM_ENVELOPE_CALIBRATION_WINDOWS,
        )
        available = np.setdiff1d(
            np.arange(server_start, dtype=np.int64),
            calibration,
            assume_unique=True,
        )
        dense_positions = _evenly_spaced_indices(
            0, len(available), LDM_DENSE_TRAIN_WINDOWS
        )
        dense_train = available[dense_positions]
        available = np.setdiff1d(
            available, dense_train, assume_unique=True
        )
        operation_positions = _evenly_spaced_indices(
            0, len(available), LDM_OPERATION_WINDOWS
        )
        healthy_operation = available[operation_positions]
        server_test = np.arange(server_start, n, dtype=np.int64)
        device_indices = np.concatenate(
            (calibration, dense_train, healthy_operation)
        )
        if len(np.unique(device_indices)) != len(device_indices):
            raise ValueError("known-healthy LDM envelope partitions overlap")
        if np.any(device_indices >= server_start):
            raise ValueError("known-healthy LDM envelope reaches the server test")
        return {
            "calibration_seed": calibration,
            "dense_train": dense_train,
            "healthy_operation": healthy_operation,
            "server_test": server_test,
            "server_healthy": server_test[server_test < transition],
            "server_unhealthy": server_test[server_test >= transition],
        }

    operation_start = int(np.floor(0.60 * transition))
    server_start = int(np.floor(0.80 * transition))
    calibration = np.arange(calibration_windows, dtype=np.int64)
    dense_train = _evenly_spaced_indices(
        calibration_windows,
        operation_start,
        LDM_DENSE_TRAIN_WINDOWS,
    )
    healthy_operation = _evenly_spaced_indices(
        operation_start,
        server_start,
        LDM_OPERATION_WINDOWS,
    )
    server_test = np.arange(server_start, n, dtype=np.int64)
    groups = (calibration, dense_train, healthy_operation, server_test)
    if any(len(group) == 0 for group in groups):
        raise ValueError("broad LDM split produced an empty segment")
    if not (
        int(calibration[-1]) < int(dense_train[0])
        and int(dense_train[-1]) < int(healthy_operation[0])
        and int(healthy_operation[-1]) < int(server_test[0])
        and int(server_test[0]) < transition < n
    ):
        raise ValueError("broad LDM split is not strictly chronological")
    if any(np.any(group >= transition) for group in groups[:3]):
        raise ValueError("broad LDM client pool contains post-transition data")
    return {
        "calibration_seed": calibration,
        "dense_train": dense_train,
        "healthy_operation": healthy_operation,
        "server_test": server_test,
        "server_healthy": server_test[server_test < transition],
        "server_unhealthy": server_test[server_test >= transition],
    }


def _load_ldm_participating_clients(
    config: RealtimeConfig,
    start_device_index: int,
    machine_count: int,
) -> tuple[
    dict[str, DeviceShard], list[dict[str, object]], list[dict[str, object]]
]:
    """Load fixed-count healthy client pools and future LDM test chronologies."""
    if machine_count < 0 or machine_count > len(LDM_MACHINES):
        raise ValueError(f"LDM participating-client count must be 0..{len(LDM_MACHINES)}")
    if machine_count == 0:
        return {}, [], []
    protocol_path = _ldm_protocol_path(config.ldm_split_strategy)
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"missing fixed LDM extension protocol: {protocol_path}"
        )
    extension_protocol = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )
    expected_schema = {
        "fixed_prefix_v1": "realtime_federated_ldm_participating_client_temporal_v1",
        "broad_healthy_v2": "realtime_federated_ldm_participating_client_temporal_v2",
        "known_healthy_envelope_v3": "realtime_federated_ldm_participating_client_temporal_v3",
        "stratified_healthy_v4": "realtime_federated_ldm_participating_client_temporal_v4",
        "stratified_healthy_v4r1": "realtime_federated_ldm_participating_client_temporal_v4r1",
        "stratified_healthy_v4r2": "realtime_federated_ldm_participating_client_temporal_v4r2",
        "stratified_healthy_v5_condition_aware": "realtime_federated_ldm_participating_client_temporal_v5",
    }.get(config.ldm_split_strategy)
    if extension_protocol.get("schema") != expected_schema:
        raise ValueError("unexpected LDM extension protocol schema")
    if config.calibration_windows != 100:
        raise ValueError(
            "the fixed LDM extension requires exactly 100 calibration-only windows"
        )
    expected_transition_hash = str(
        extension_protocol["transition_file"]["sha256"]
    ).upper()
    if _sha256_file(LDM_TRANSITION_PATH) != expected_transition_hash:
        raise ValueError("LDM transition file hash drift from extension protocol")
    transition_payload = json.loads(LDM_TRANSITION_PATH.read_text(encoding="utf-8"))
    transitions = transition_payload.get("transitions", {})
    if set(transitions) != set(LDM_MACHINES):
        raise ValueError("LDM transition file must bind exactly B06/B13/B14/B15")

    shards: dict[str, DeviceShard] = {}
    allocation: list[dict[str, object]] = []
    leakage: list[dict[str, object]] = []
    for offset, machine in enumerate(LDM_MACHINES[:machine_count]):
        device_id = f"device_{start_device_index + offset}"
        feature_path = LDM_FEATURE_ROOT / f"ldm_upb__{machine}.npz"
        lock_path = LDM_FEATURE_LOCK_ROOT / f"{machine}.json"
        if not feature_path.is_file() or not lock_path.is_file():
            raise FileNotFoundError(f"missing locked LDM feature artifact for {machine}")
        feature_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        feature_hash = _sha256_file(feature_path)
        expected_feature_hash = str(
            extension_protocol["locked_feature_artifacts"][machine]
        ).upper()
        if feature_hash != expected_feature_hash:
            raise ValueError(f"LDM feature hash drift from extension protocol for {machine}")
        if feature_hash != str(feature_lock["feature_sha256"]).upper():
            raise ValueError(f"locked LDM feature hash drift for {machine}")
        with np.load(feature_path, allow_pickle=False) as payload:
            acquisition_index = np.asarray(payload["acquisition_index"], dtype=np.int64)
            full_features = np.asarray(payload["features"], dtype=np.float32)
            client_id = str(np.asarray(payload["client_id"]).item())
        if client_id != f"ldm_upb::{machine}":
            raise ValueError(f"unexpected LDM client identity for {machine}: {client_id}")
        if full_features.ndim != 2 or full_features.shape[1] != 100:
            raise ValueError(f"invalid locked LDM feature matrix for {machine}: {full_features.shape}")
        if not np.array_equal(
            acquisition_index, np.arange(len(full_features), dtype=np.int64)
        ):
            raise ValueError(f"non-contiguous LDM acquisition chronology for {machine}")
        projected = np.asarray(full_features[:, :12], dtype=np.float32)
        if not np.all(np.isfinite(projected)):
            raise ValueError(f"non-finite LDM time-statistic projection for {machine}")

        transition = int(transitions[machine])
        split_indices = _ldm_split_indices(
            len(projected),
            transition,
            config.calibration_windows,
            config.ldm_split_strategy,
        )

        def make_windows(indices: np.ndarray, split: str) -> list[StreamWindow]:
            return [
                StreamWindow(
                    dataset=LDM_DATASET,
                    logical_client=machine,
                    raw_features=projected[index].copy(),
                    split=split,
                    label=int(index >= transition),
                    source_file=f"{machine}/acquisition_{index:06d}",
                    start_sample=index,
                    window_id=f"ldm_upb:{machine}:{index:06d}",
                )
                for index in np.asarray(indices, dtype=np.int64).tolist()
            ]

        calibration_seed = make_windows(split_indices["calibration_seed"], "calibration")
        train = make_windows(split_indices["dense_train"], "train")
        operation = make_windows(split_indices["healthy_operation"], "train")
        server_test = make_windows(split_indices["server_test"], "test")
        if any(window.label for window in calibration_seed + train + operation):
            raise ValueError(f"unhealthy LDM data entered a client pool for {machine}")
        if not all(
            window.label == int(window.start_sample >= transition)
            for window in server_test
        ):
            raise ValueError(f"LDM server-test labels are inconsistent for {machine}")
        initial_mean, initial_std = _statistics_from_calibration_seed(
            calibration_seed, config, f"{device_id}/{LDM_DATASET}/{machine}"
        )
        stream = DeviceDomainStream(
            device_id=device_id,
            dataset=LDM_DATASET,
            logical_client=machine,
            initial_mean=initial_mean,
            initial_std=initial_std,
            calibration_seed=calibration_seed,
            train=train,
            operation=operation,
            server_test=server_test,
            calibration_replay_allowed=False,
        )
        shards[device_id] = DeviceShard(device_id, {stream.domain_key: stream})
        allocation.append(
            {
                "assignment_status": "ACTIVE",
                "site_index": start_device_index + offset,
                "device_id": device_id,
                "fixed_site": True,
                "assigned_domain_count": 1,
                "dataset": LDM_DATASET,
                "logical_client": machine,
                "client_name": _client_display_name(LDM_DATASET, machine),
                "train_windows": len(train),
                "operation_windows": len(operation),
                "device_pool_windows": len(train) + len(operation),
                "server_test_windows": len(server_test),
                "server_healthy_windows": len(split_indices["server_healthy"]),
                "server_unhealthy_windows": len(split_indices["server_unhealthy"]),
                "calibration_seed_windows": len(calibration_seed),
                "calibration_replay_allowed": False,
                "published_transition": transition,
                "ldm_split_strategy": config.ldm_split_strategy,
                "training_coverage_last_index": int(operation[-1].start_sample),
                "server_test_first_index": int(server_test[0].start_sample),
                "chronological_future_test": (
                    config.ldm_split_strategy
                    not in {
                        "stratified_healthy_v4",
                        "stratified_healthy_v4r1",
                        "stratified_healthy_v4r2",
                        "stratified_healthy_v5_condition_aware",
                    }
                ),
                "feature_projection": "locked_v12_mean_time_statistics_coordinates_0_11",
                "train_sources": len(train),
                "operation_sources": len(operation),
                "server_sources": len(server_test),
            }
        )
        leakage.append(
            {
                "device_id": device_id,
                "fixed_site": True,
                "dataset": LDM_DATASET,
                "logical_client": machine,
                "device_windows": len(calibration_seed) + len(train) + len(operation),
                "server_windows": len(server_test),
                "window_id_overlap": 0,
                "source_file_overlap": 0,
                "calibration_train_overlap": 0,
                "unhealthy_device_windows": 0,
                "status": "PASS",
            }
        )
    return shards, allocation, leakage


def load_device_study(
    config: RealtimeConfig,
) -> tuple[dict[str, DeviceShard], list[dict[str, object]], list[dict[str, object]]]:
    """Build one fixed domain per SD card plus an untouched server holdout."""
    base_capacity = len(config.datasets) * len(CLIENTS)
    base_count = min(config.num_devices, base_capacity)
    ldm_count = config.num_devices - base_count
    if ldm_count and not config.include_ldm_clients:
        raise ValueError(
            "more devices than prepared base sites requires include_ldm_clients"
        )
    assignments = _fixed_site_assignments(config.datasets, base_count)
    shards = {
        device_id: DeviceShard(device_id, {})
        for device_id, _, _ in assignments
    }
    allocation_audit: list[dict[str, object]] = []
    leakage_audit: list[dict[str, object]] = []
    normalization_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for site_index, (device_id, dataset, logical_client) in enumerate(
        assignments, start=1
    ):
        bin_dir = _prepared_bin_dir(dataset)
        if dataset not in normalization_cache:
            normalization_cache[dataset] = load_normalization(bin_dir)
        prepared_mean, prepared_std = normalization_cache[dataset]
        domain = load_domain(logical_client, bin_dir)
        validate_domain(dataset, logical_client, domain)
        by_split = {
            split: _rows_as_windows(
                dataset,
                logical_client,
                np.asarray(domain[split], dtype=np.float32),
                list(domain[f"{split}_rows"]),
                prepared_mean,
                prepared_std,
            )
            for split in ("train", "calibration", "val", "test")
        }

        # The original test sources define the server holdout. If a fault
        # source also contributed guarded validation windows, its complete
        # source remains at the server to prevent identity leakage.
        original_test_sources = {
            window.source_file for window in by_split["test"]
        }
        all_windows = [
            window
            for split in ("train", "calibration", "val", "test")
            for window in by_split[split]
        ]
        server_windows = [
            window
            for window in all_windows
            if window.source_file in original_test_sources
        ]
        device_windows = [
            window
            for window in all_windows
            if window.source_file not in original_test_sources
        ]
        device_ids = {window.window_id for window in device_windows}
        server_ids = {window.window_id for window in server_windows}
        device_sources = {window.source_file for window in device_windows}
        server_sources = {window.source_file for window in server_windows}
        id_overlap = sorted(device_ids & server_ids)
        source_overlap = sorted(device_sources & server_sources)
        leakage_audit.append(
            {
                "device_id": device_id,
                "fixed_site": True,
                "dataset": dataset,
                "logical_client": logical_client,
                "device_windows": len(device_windows),
                "server_windows": len(server_windows),
                "window_id_overlap": len(id_overlap),
                "source_file_overlap": len(source_overlap),
                "status": "PASS" if not id_overlap and not source_overlap else "FAIL",
            }
        )
        if id_overlap or source_overlap:
            details = (id_overlap or source_overlap)[:5]
            raise ValueError(
                f"server leakage for {device_id}/{dataset}/{logical_client}: {details}"
            )

        train = sorted(
            [window for window in device_windows if window.split == "train"],
            key=_window_sort_key,
        )
        operation = _operation_order(
            dataset,
            {
                "calibration": [
                    window
                    for window in device_windows
                    if window.split == "calibration"
                ],
                "val": [
                    window for window in device_windows if window.split == "val"
                ],
            },
        )
        server_test = _operation_order(
            dataset,
            {
                split: [
                    window for window in server_windows if window.split == split
                ]
                for split in ("train", "calibration", "val", "test")
            },
        )
        initial_mean, initial_std, calibration_seed = _initial_statistics(
            train, config, f"{device_id}/{dataset}/{logical_client}"
        )
        stream = DeviceDomainStream(
            device_id=device_id,
            dataset=dataset,
            logical_client=logical_client,
            initial_mean=initial_mean,
            initial_std=initial_std,
            calibration_seed=calibration_seed,
            train=train,
            operation=operation,
            server_test=server_test,
        )
        shards[device_id].streams[stream.domain_key] = stream
        allocation_audit.append(
            {
                "assignment_status": "ACTIVE",
                "site_index": site_index,
                "device_id": device_id,
                "fixed_site": True,
                "assigned_domain_count": 1,
                "dataset": dataset,
                "logical_client": logical_client,
                "client_name": _client_display_name(dataset, logical_client),
                "train_windows": len(train),
                "operation_windows": len(operation),
                "device_pool_windows": len(train) + len(operation),
                "server_test_windows": len(server_test),
                "calibration_seed_windows": len(calibration_seed),
                "train_sources": len({window.source_file for window in train}),
                "operation_sources": len(
                    {window.source_file for window in operation}
                ),
                "server_sources": len(
                    {window.source_file for window in server_test}
                ),
            }
        )

    active_pairs = {(dataset, client) for _, dataset, client in assignments}
    for site_index, (dataset, logical_client) in enumerate(
        (
            (dataset, logical_client)
            for dataset in config.datasets
            for logical_client in CLIENTS
        ),
        start=1,
    ):
        if (dataset, logical_client) in active_pairs:
            continue
        allocation_audit.append(
            {
                "assignment_status": "INACTIVE",
                "site_index": site_index,
                "device_id": "unassigned",
                "fixed_site": True,
                "assigned_domain_count": 0,
                "dataset": dataset,
                "logical_client": logical_client,
                "client_name": _client_display_name(dataset, logical_client),
                "reason": "device count does not activate this fixed site",
            }
        )

    ldm_shards, ldm_allocation, ldm_leakage = _load_ldm_participating_clients(
        config, base_count + 1, ldm_count
    )
    shards.update(ldm_shards)
    allocation_audit.extend(ldm_allocation)
    leakage_audit.extend(ldm_leakage)
    if config.include_ldm_clients:
        for offset, machine in enumerate(LDM_MACHINES[ldm_count:], start=ldm_count):
            allocation_audit.append(
                {
                    "assignment_status": "INACTIVE",
                    "site_index": base_count + offset + 1,
                    "device_id": "unassigned",
                    "fixed_site": True,
                    "assigned_domain_count": 0,
                    "dataset": LDM_DATASET,
                    "logical_client": machine,
                    "client_name": _client_display_name(LDM_DATASET, machine),
                    "reason": "device count does not activate this LDM participating client",
                }
            )

    for device_id, shard in shards.items():
        if len(shard.streams) != 1:
            raise RuntimeError(
                f"{device_id} must own exactly one fixed domain; found "
                f"{len(shard.streams)}"
            )
    return shards, allocation_audit, leakage_audit


class AdaptiveNormalizer:
    """Slow client-local statistics, updated only when explicitly authorized."""

    def __init__(self, initial_mean: np.ndarray, initial_std: np.ndarray, alpha: float):
        self.mean = np.asarray(initial_mean, dtype=np.float64).copy()
        self.variance = np.maximum(np.asarray(initial_std, dtype=np.float64) ** 2, 1e-12)
        self.alpha = float(alpha)

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(self.variance, 1e-12))
        std = np.where(std < 1e-6, 1.0, std)
        return ((np.asarray(raw, dtype=np.float64) - self.mean) / std).astype(np.float32)

    def update(self, raw: np.ndarray) -> None:
        value = np.asarray(raw, dtype=np.float64)
        delta = value - self.mean
        self.mean += self.alpha * delta
        self.variance = (1.0 - self.alpha) * (
            self.variance + self.alpha * delta * delta
        )


class RobustScoreBuffer:
    """Fixed-memory robust calibration state suitable for a small device."""

    def __init__(self, capacity: int):
        self.values: deque[float] = deque(maxlen=int(capacity))

    def append(self, value: float) -> None:
        if np.isfinite(value):
            self.values.append(float(value))

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.append(float(value))

    def location_scale(self) -> tuple[float, float]:
        values = np.asarray(self.values, dtype=np.float64)
        if len(values) < 2:
            return 0.0, 1.0
        location = float(np.median(values))
        mad_scale = 1.4826 * float(np.median(np.abs(values - location)))
        if mad_scale > 1e-8:
            return location, mad_scale
        std_scale = float(np.std(values, ddof=1))
        return location, max(std_scale, 1e-8)

    def standardized(self, value: float) -> float:
        location, scale = self.location_scale()
        return float((value - location) / scale)

    def quantile(self, probability: float) -> float:
        if not self.values:
            raise ValueError("score buffer is empty")
        return float(np.quantile(np.asarray(self.values, dtype=np.float64), probability))

    def copy(self) -> "RobustScoreBuffer":
        cloned = RobustScoreBuffer(self.values.maxlen or len(self.values) or 1)
        cloned.extend(self.values)
        return cloned


class CalibrationState:
    """Fixed-site online normalization, thresholds, and alarm state."""

    def __init__(
        self,
        dataset: str,
        initial_mean: np.ndarray,
        initial_std: np.ndarray,
        config: RealtimeConfig,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.normalizer = AdaptiveNormalizer(
            initial_mean, initial_std, config.normalizer_alpha
        )
        self.ae_scores = RobustScoreBuffer(config.score_buffer_size)
        self.magnitude_scores = RobustScoreBuffer(config.score_buffer_size)
        self.mahalanobis_scores = RobustScoreBuffer(config.score_buffer_size)
        self.combined_scores = RobustScoreBuffer(config.score_buffer_size)
        self.inverse_covariance = np.eye(len(initial_mean), dtype=np.float64)
        self.ldm_condition_aware = bool(
            dataset == LDM_DATASET
            and config.ldm_split_strategy
            == "stratified_healthy_v5_condition_aware"
        )
        self.prognostic_enabled = bool(
            config.prognostic_latch
            and dataset in PROGNOSTIC_LATCH_DATASETS
            and not self.ldm_condition_aware
        )
        self.prognostic_center = np.zeros(len(initial_mean), dtype=np.float64)
        self.prognostic_normalizer_mean = np.asarray(
            initial_mean, dtype=np.float64
        ).copy()
        self.prognostic_normalizer_std = np.maximum(
            np.asarray(initial_std, dtype=np.float64), 1e-6
        )
        self.prognostic_inverse_covariance = np.eye(
            len(initial_mean), dtype=np.float64
        )
        self.prognostic_scores = RobustScoreBuffer(
            config.prognostic_score_buffer_size
        )
        self.prognostic_feature_history: deque[np.ndarray] = deque(
            maxlen=config.prognostic_window
        )
        self.prognostic_alarm_history: deque[int] = deque(
            maxlen=config.prognostic_history
        )
        self.prognostic_latched = False
        self.mahalanobis_weight = (
            config.run_to_failure_mahalanobis_weight
            if dataset in RUN_TO_FAILURE_DATASETS
            else config.mahalanobis_fusion_weight
        )
        self.alarm_history: deque[int] = deque(maxlen=config.temporal_window)
        self.cusum = 0.0
        self.cusum_reference = config.cusum_allowance
        self.alarm_latched = False
        self.recovery_count = 0
        self.last_source: str | None = None

    def _combine(
        self,
        ae_score: float,
        magnitude_score: float,
        mahalanobis_score: float,
        weight: float,
    ) -> float:
        neural_score = float(
            weight * self.ae_scores.standardized(ae_score)
            + (1.0 - weight)
            * self.magnitude_scores.standardized(magnitude_score)
        )
        return float(
            (1.0 - self.mahalanobis_weight) * neural_score
            + self.mahalanobis_weight
            * self.mahalanobis_scores.standardized(mahalanobis_score)
        )

    def seed_scores(
        self,
        model: ABLoRA,
        windows: list[StreamWindow],
        weight: float,
    ) -> None:
        self.recalibrate_scores(
            model,
            [window.raw_features for window in windows],
            weight,
        )

    def recalibrate_scores(
        self,
        model: ABLoRA,
        raw_features: Iterable[np.ndarray],
        weight: float,
    ) -> None:
        """Replace score calibration using healthy data and the current model.

        Model updates change the reconstruction-error scale. Keeping scores
        produced by older adapters makes the threshold mathematically stale,
        so all three fixed-memory buffers are rebuilt atomically here.
        """
        normalized_values = np.asarray(
            [self.normalizer.normalize(raw) for raw in raw_features],
            dtype=np.float64,
        )
        if not len(normalized_values):
            raise ValueError("online calibration produced no samples")
        covariance = np.cov(normalized_values, rowvar=False)
        if np.ndim(covariance) != 2:
            covariance = np.eye(normalized_values.shape[1], dtype=np.float64)
        shrinkage = self.config.covariance_shrinkage
        covariance = (
            (1.0 - shrinkage) * np.asarray(covariance, dtype=np.float64)
            + shrinkage * np.eye(normalized_values.shape[1], dtype=np.float64)
        )
        inverse_covariance = np.linalg.pinv(covariance, rcond=1e-6)

        # A rolling feature mean suppresses window noise and exposes slow,
        # persistent degradation.  The Hotelling-style score is fitted only
        # on the operator-confirmed healthy commissioning sequence.  It needs
        # 12 means + a 12x12 inverse covariance and is therefore practical on
        # an ESP32/LilyGO.  Partial means are excluded from calibration.
        prognostic_vectors: list[np.ndarray] = []
        rolling: deque[np.ndarray] = deque(maxlen=self.config.prognostic_window)
        for normalized in normalized_values:
            rolling.append(np.asarray(normalized, dtype=np.float64))
            prognostic_vectors.append(
                np.mean(np.asarray(rolling, dtype=np.float64), axis=0)
            )
        full_start = min(
            self.config.prognostic_window - 1,
            max(0, len(prognostic_vectors) - 1),
        )
        full_vectors = np.asarray(prognostic_vectors[full_start:], dtype=np.float64)
        prognostic_center = np.mean(full_vectors, axis=0)
        prognostic_covariance = np.cov(full_vectors, rowvar=False)
        if np.ndim(prognostic_covariance) != 2:
            prognostic_covariance = np.eye(
                normalized_values.shape[1], dtype=np.float64
            )
        prognostic_shrinkage = self.config.prognostic_covariance_shrinkage
        prognostic_covariance = (
            (1.0 - prognostic_shrinkage)
            * np.asarray(prognostic_covariance, dtype=np.float64)
            + prognostic_shrinkage
            * np.eye(normalized_values.shape[1], dtype=np.float64)
        )
        prognostic_inverse_covariance = np.linalg.pinv(
            prognostic_covariance, rcond=1e-6
        )
        # Threshold calibration includes the healthy startup transients from
        # partial rolling windows.  The covariance itself is still fitted only
        # on full windows.  Keeping startup scores makes the irreversible latch
        # conservative and matches the state sequence the firmware experiences
        # after a cold boot.
        prognostic_values = [
            float(
                (vector - prognostic_center)
                @ prognostic_inverse_covariance
                @ (vector - prognostic_center)
            )
            for vector in prognostic_vectors
        ]
        ae_values: list[float] = []
        magnitude_values: list[float] = []
        mahalanobis_values: list[float] = []
        for normalized in normalized_values:
            ae_values.append(float(model.losses(normalized[None, :])[0]))
            magnitude_values.append(float(np.mean(normalized * normalized)))
            mahalanobis_values.append(
                float(normalized @ inverse_covariance @ normalized)
            )
        new_ae = RobustScoreBuffer(self.config.score_buffer_size)
        new_magnitude = RobustScoreBuffer(self.config.score_buffer_size)
        new_mahalanobis = RobustScoreBuffer(self.config.score_buffer_size)
        new_combined = RobustScoreBuffer(self.config.score_buffer_size)
        new_ae.extend(ae_values)
        new_magnitude.extend(magnitude_values)
        new_mahalanobis.extend(mahalanobis_values)
        for ae, magnitude, mahalanobis in zip(
            ae_values, magnitude_values, mahalanobis_values
        ):
            neural = float(
                weight * new_ae.standardized(ae)
                + (1.0 - weight) * new_magnitude.standardized(magnitude)
            )
            new_combined.append(
                (1.0 - self.mahalanobis_weight) * neural
                + self.mahalanobis_weight
                * new_mahalanobis.standardized(mahalanobis)
            )
        if not new_combined.values:
            raise ValueError("online calibration produced no finite scores")
        self.ae_scores = new_ae
        self.magnitude_scores = new_magnitude
        self.mahalanobis_scores = new_mahalanobis
        self.combined_scores = new_combined
        self.inverse_covariance = inverse_covariance
        self.prognostic_normalizer_mean = self.normalizer.mean.copy()
        self.prognostic_normalizer_std = np.sqrt(
            np.maximum(self.normalizer.variance, 1e-12)
        )
        self.prognostic_center = prognostic_center
        self.prognostic_inverse_covariance = prognostic_inverse_covariance
        new_prognostic = RobustScoreBuffer(
            self.config.prognostic_score_buffer_size
        )
        new_prognostic.extend(prognostic_values)
        self.prognostic_scores = new_prognostic
        location, scale = self.combined_scores.location_scale()
        healthy_standardized = np.clip(
            (np.asarray(self.combined_scores.values, dtype=np.float64) - location)
            / max(scale, 1e-8),
            -self.config.cusum_clip,
            self.config.cusum_clip,
        )
        # The empirical healthy mean increment is exactly -allowance.  This
        # preserves Page-CUSUM sensitivity to persistent sub-threshold shifts
        # without recreating the positive-drift false-alarm defect.
        self.cusum_reference = float(
            np.mean(healthy_standardized) + self.config.cusum_allowance
        )
        self.reset_temporal()
        self.prognostic_feature_history.extend(
            np.asarray(normalized, dtype=np.float64)
            for normalized in normalized_values[
                -self.config.prognostic_window :
            ]
        )

    def reset_temporal(self) -> None:
        self.alarm_history.clear()
        self.prognostic_alarm_history.clear()
        self.cusum = 0.0
        self.alarm_latched = False
        self.prognostic_latched = False
        self.recovery_count = 0

    def _thresholds(self) -> tuple[float, float, float, float]:
        location, scale = self.combined_scores.location_scale()
        trust_threshold = min(
            self.combined_scores.quantile(self.config.trust_quantile),
            location + self.config.trust_sigma * scale,
        )
        # A single accepted outlier must not make the detector permanently
        # insensitive.  Cap the empirical tail by the robust MAD boundary,
        # while preserving a full-scale gap above the trust boundary.
        if self.ldm_condition_aware:
            # The primary V5 alarm uses an empirical healthy quantile matched
            # to the declared specificity gate. The existing 3-of-5 temporal
            # rule supplies persistence; no target fault score enters this
            # calibration.
            alarm_threshold = max(
                self.combined_scores.quantile(0.90), trust_threshold
            )
        else:
            alarm_threshold = max(
                min(
                    self.combined_scores.quantile(self.config.alarm_quantile),
                    location + self.config.alarm_sigma * scale,
                ),
                trust_threshold + scale,
            )
        return trust_threshold, alarm_threshold, location, scale

    def _prognostic_threshold(self) -> tuple[float, float, float]:
        """Return a conservative threshold for correlated rolling scores.

        Rolling Hotelling scores can have a near-zero MAD even when a few
        healthy commissioning points span a wider tail.  A 10% standard-
        deviation floor prevents that numerical concentration from causing an
        irreversible early latch.  This floor is computed from healthy scores
        only and was fixed before the confirmation runs.
        """
        values = np.asarray(self.prognostic_scores.values, dtype=np.float64)
        if not len(values):
            raise ValueError("prognostic score buffer is empty")
        location = float(np.median(values))
        mad_scale = 1.4826 * float(np.median(np.abs(values - location)))
        std_floor = (
            0.10 * float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
        scale = max(mad_scale, std_floor, 1e-8)
        threshold = min(
            float(np.quantile(values, self.config.alarm_quantile)),
            location + self.config.prognostic_sigma * scale,
        )
        return threshold, location, scale

    def observe(
        self,
        window: StreamWindow,
        model: ABLoRA,
        *,
        allow_calibration: bool,
        weight: float,
        confirmed_normal: bool = False,
    ) -> dict[str, object]:
        if (
            self.dataset not in RUN_TO_FAILURE_DATASETS
            and self.last_source is not None
            and window.source_file != self.last_source
        ):
            self.reset_temporal()
        self.last_source = window.source_file

        normalized = self.normalizer.normalize(window.raw_features)
        ae_score = float(model.losses(normalized[None, :])[0])
        magnitude_score = float(np.mean(normalized * normalized))
        mahalanobis_score = float(
            normalized.astype(np.float64)
            @ self.inverse_covariance
            @ normalized.astype(np.float64)
        )
        combined_score = self._combine(
            ae_score, magnitude_score, mahalanobis_score, weight
        )
        trust_threshold, alarm_threshold, location, scale = self._thresholds()

        prognostic_normalized = (
            np.asarray(window.raw_features, dtype=np.float64)
            - self.prognostic_normalizer_mean
        ) / self.prognostic_normalizer_std
        self.prognostic_feature_history.append(prognostic_normalized)
        prognostic_vector = np.mean(
            np.asarray(self.prognostic_feature_history, dtype=np.float64), axis=0
        )
        prognostic_delta = prognostic_vector - self.prognostic_center
        prognostic_score = float(
            prognostic_delta
            @ self.prognostic_inverse_covariance
            @ prognostic_delta
        )
        (
            prognostic_threshold,
            prognostic_location,
            prognostic_scale,
        ) = self._prognostic_threshold()
        prognostic_raw_alarm = bool(
            self.prognostic_enabled
            and prognostic_score > prognostic_threshold
        )
        self.prognostic_alarm_history.append(int(prognostic_raw_alarm))
        prognostic_event = bool(
            self.prognostic_enabled
            and sum(self.prognostic_alarm_history)
            >= self.config.prognostic_required
        )
        if prognostic_event:
            self.prognostic_latched = True

        raw_alarm = combined_score > alarm_threshold
        self.alarm_history.append(int(raw_alarm))
        window_alarm = sum(self.alarm_history) >= self.config.temporal_required
        standardized = float(
            np.clip(
                (combined_score - location) / max(scale, 1e-8),
                -self.config.cusum_clip,
                self.config.cusum_clip,
            )
        )
        # Boundary excess remains a useful point-alarm diagnostic, while the
        # sequential detector uses an empirically calibrated Page reference.
        boundary_excess = float(
            np.clip(
                (combined_score - alarm_threshold) / max(scale, 1e-8),
                -self.config.cusum_clip,
                self.config.cusum_clip,
            )
        )
        cusum_increment = standardized - self.cusum_reference
        self.cusum = float(
            np.clip(
                max(
                    0.0,
                    self.config.cusum_decay * self.cusum + cusum_increment,
                ),
                0.0,
                self.config.cusum_maximum,
            )
        )
        cusum_alarm = bool(
            not self.ldm_condition_aware
            and self.cusum >= self.config.cusum_limit
        )
        event_alarm = bool(
            window_alarm or cusum_alarm or self.prognostic_latched
        )

        if event_alarm:
            self.alarm_latched = True
            self.recovery_count = 0
        elif self.alarm_latched:
            if combined_score <= trust_threshold:
                self.recovery_count += 1
                if self.recovery_count >= self.config.recovery_required:
                    self.reset_temporal()
            else:
                self.recovery_count = 0

        # ``alarm`` is the current per-window decision used for metrics.
        # ``alarm_latched`` is retained separately for the real operational
        # state and blocks online learning until explicit recovery.
        alarm = event_alarm
        trusted = bool(
            combined_score <= trust_threshold and not self.alarm_latched
        )
        uncertain = bool(not trusted and not alarm)
        # The calibration split represents the operator-confirmed healthy
        # installation period. It is identified by protocol, not by reading a
        # fault label. This grace phase lets a new physical domain establish
        # its live baseline without latching a false alarm.
        if confirmed_normal:
            self.reset_temporal()
            alarm = False
            trusted = True
            uncertain = False
        if allow_calibration and trusted:
            self.normalizer.update(window.raw_features)
            self.ae_scores.append(ae_score)
            self.magnitude_scores.append(magnitude_score)
            self.mahalanobis_scores.append(mahalanobis_score)
            self.combined_scores.append(combined_score)

        return {
            "ae_score": ae_score,
            "magnitude_score": magnitude_score,
            "mahalanobis_score": mahalanobis_score,
            "prognostic_score": prognostic_score,
            "prognostic_threshold": prognostic_threshold,
            "prognostic_location": prognostic_location,
            "prognostic_scale": prognostic_scale,
            "prognostic_raw_alarm": prognostic_raw_alarm,
            "prognostic_event": prognostic_event,
            "prognostic_latched": bool(self.prognostic_latched),
            "combined_score": combined_score,
            "trust_threshold": trust_threshold,
            "alarm_threshold": alarm_threshold,
            "trusted": trusted,
            "uncertain": uncertain,
            "raw_alarm": bool(raw_alarm),
            "window_alarm": bool(window_alarm),
            "cusum_alarm": bool(cusum_alarm),
            "alarm": alarm,
            "alarm_latched": bool(self.alarm_latched),
            "standardized_score": standardized,
            "cusum_boundary_excess": boundary_excess,
            "cusum_increment": cusum_increment,
            "cusum_reference": self.cusum_reference,
            "cusum": self.cusum,
            "recovery_count": self.recovery_count,
        }

    def frozen_copy(self) -> "CalibrationState":
        std = np.sqrt(np.maximum(self.normalizer.variance, 1e-12))
        cloned = CalibrationState(
            self.dataset,
            self.normalizer.mean.copy(),
            std,
            self.config,
        )
        cloned.normalizer.variance = self.normalizer.variance.copy()
        cloned.ae_scores = self.ae_scores.copy()
        cloned.magnitude_scores = self.magnitude_scores.copy()
        cloned.mahalanobis_scores = self.mahalanobis_scores.copy()
        cloned.combined_scores = self.combined_scores.copy()
        cloned.inverse_covariance = self.inverse_covariance.copy()
        cloned.prognostic_enabled = self.prognostic_enabled
        cloned.prognostic_normalizer_mean = (
            self.prognostic_normalizer_mean.copy()
        )
        cloned.prognostic_normalizer_std = (
            self.prognostic_normalizer_std.copy()
        )
        cloned.prognostic_center = self.prognostic_center.copy()
        cloned.prognostic_inverse_covariance = (
            self.prognostic_inverse_covariance.copy()
        )
        cloned.prognostic_scores = self.prognostic_scores.copy()
        cloned.prognostic_feature_history.extend(
            value.copy() for value in self.prognostic_feature_history
        )
        cloned.mahalanobis_weight = self.mahalanobis_weight
        cloned.cusum_reference = self.cusum_reference
        cloned.reset_temporal()
        cloned.last_source = None
        return cloned


@dataclass(frozen=True)
class ReplayItem:
    domain_key: str
    raw_features: np.ndarray


class RuntimeDevice:
    """One fixed-site LilyGO: one shared adapter and one local calibrator."""

    def __init__(
        self,
        shard: DeviceShard,
        base_weights: list[np.ndarray],
        global_adapter: np.ndarray,
        config: RealtimeConfig,
        seed: int,
        variant: StudyVariant,
    ) -> None:
        self.shard = shard
        self.base_weights = base_weights
        self.config = config
        self.variant = variant
        self.policy = variant.policy
        self.personal_adapter = np.zeros_like(global_adapter, dtype=np.float32)
        self.compression_residual = np.zeros_like(global_adapter, dtype=np.float32)
        self.pending_personal_candidate: np.ndarray | None = None
        self.last_compression_error_norm = 0.0
        self.rng = np.random.default_rng(
            stable_seed(seed, f"full-runtime:{variant.name}:{shard.device_id}")
        )
        self.contexts: dict[str, CalibrationState] = {}
        calibration_items: list[ReplayItem] = []
        anchor_seed_items: list[ReplayItem] = []
        calibration_model = self.model(global_adapter, personalized=True)
        for domain_key, stream in shard.streams.items():
            score_calibration_windows = stream.calibration_seed
            if (
                stream.dataset == LDM_DATASET
                and config.ldm_split_strategy
                in {
                    "stratified_healthy_v4r1",
                    "stratified_healthy_v4r2",
                    "stratified_healthy_v5_condition_aware",
                }
            ):
                score_calibration_windows = _representative_calibration_windows(
                    stream.calibration_seed, config.score_buffer_size
                )
            state = CalibrationState(
                stream.dataset, stream.initial_mean, stream.initial_std, config
            )
            state.seed_scores(
                calibration_model,
                score_calibration_windows,
                config.ae_fusion_weight,
            )
            self.contexts[domain_key] = state
            current_items = [
                ReplayItem(domain_key, window.raw_features.copy())
                for window in score_calibration_windows
            ]
            calibration_items.extend(current_items)
            if stream.calibration_replay_allowed:
                anchor_seed_items.extend(current_items)

        # These immutable operator-confirmed healthy samples are kept
        # separately from trainable replay. They recalibrate score thresholds
        # whenever the effective model changes.
        self.calibration_items: deque[ReplayItem] = deque(
            calibration_items,
            maxlen=config.score_buffer_size,
        )

        anchor_count = min(config.anchor_replay_size, len(anchor_seed_items))
        if anchor_count:
            indices = self.rng.choice(
                len(anchor_seed_items), anchor_count, replace=False
            )
            self.anchor_items = [
                anchor_seed_items[int(index)] for index in indices
            ]
        else:
            self.anchor_items = []
        self.recent_items: deque[ReplayItem] = deque(
            maxlen=config.recent_replay_size
        )
        self.pending_training_windows = 0
        self.counters: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def model(self, global_adapter: np.ndarray, *, personalized: bool) -> ABLoRA:
        model = ABLoRA(
            LARGER_DIMS,
            self.variant.rank,
            self.base_weights,
            train_a=False,
            alpha=_constructor_alpha(self.config, self.variant),
        )
        effective = np.asarray(global_adapter, dtype=np.float32)
        if personalized:
            effective = effective + self.personal_adapter
        model.set_trainable(effective, reset_optimizer=True)
        return model

    def process_block(
        self,
        windows: list[StreamWindow],
        global_adapter: np.ndarray,
        stream_round: int,
        seed: int,
    ) -> tuple[list[dict[str, object]], np.ndarray | None, int, int]:
        model = self.model(global_adapter, personalized=True)
        records: list[dict[str, object]] = []
        newly_selected = 0
        for window in windows:
            state = self.contexts[window.domain_key]
            decision = state.observe(
                window,
                model,
                allow_calibration=True,
                confirmed_normal=window.split == "calibration",
                weight=self.config.ae_fusion_weight,
            )
            if window.split == "calibration":
                self.calibration_items.append(
                    ReplayItem(window.domain_key, window.raw_features.copy())
                )
            gradient_selected = bool(
                self.policy == "all" or decision["trusted"]
            )
            if gradient_selected:
                self.recent_items.append(
                    ReplayItem(window.domain_key, window.raw_features.copy())
                )
                newly_selected += 1
                self.pending_training_windows += 1

            counters = self.counters[window.domain_key]
            counters["seen"] += 1
            counters["gradient_selected"] += int(gradient_selected)
            counters["trusted"] += int(bool(decision["trusted"]))
            counters["uncertain"] += int(bool(decision["uncertain"]))
            counters["alarms"] += int(bool(decision["alarm"]))
            counters["latched"] += int(bool(decision["alarm_latched"]))
            records.append(
                {
                    "scope": "device_prequential",
                    "seed": seed,
                    "study_variant": self.variant.name,
                    "policy": self.policy,
                    "server_optimizer": self.variant.server_optimizer,
                    "lora_rank": self.variant.rank,
                    "rank_stabilized": self.variant.rank_stabilized,
                    "device_id": self.shard.device_id,
                    "dataset": window.dataset,
                    "logical_client": window.logical_client,
                    "client_name": _client_display_name(
                        window.dataset, window.logical_client
                    ),
                    "stream_round": stream_round,
                    "split": window.split,
                    "window_id": window.window_id,
                    "source_file": window.source_file,
                    "start_sample": window.start_sample,
                    "label": window.label,
                    "gradient_selected": gradient_selected,
                    **decision,
                }
            )

        update, update_weight = self.local_update(global_adapter)
        return records, update, update_weight, newly_selected

    def local_update(
        self, global_adapter: np.ndarray
    ) -> tuple[np.ndarray | None, int]:
        replay = self.anchor_items + list(self.recent_items)
        if (
            self.pending_training_windows <= 0
            or len(replay) < self.config.minimum_replay_size
        ):
            return None, 0
        samples = np.asarray(
            [
                self.contexts[item.domain_key].normalizer.normalize(
                    item.raw_features
                )
                for item in replay
            ],
            dtype=np.float32,
        )
        samples = samples[self.rng.permutation(len(samples))]
        # The federated model begins exactly at the broadcast global adapter.
        # Its FedProx term is included in the gradient before AdamW so it also
        # participates in the optimizer's moment estimates.
        model = self.model(global_adapter, personalized=False)
        global_start = model.get_trainable().copy()
        for _ in range(self.config.local_epochs):
            for sample in samples:
                model.train_step(
                    sample,
                    self.config.local_lr,
                    model_input=(
                        sample
                        + np.float32(self.variant.noise_sigma)
                        * self.rng.standard_normal(sample.shape).astype(np.float32)
                        if self.variant.noise_sigma > 0.0
                        else None
                    ),
                    proximal_center=global_start,
                    proximal_mu=self.config.fedprox_mu,
                )
        delta = model.get_trainable() - global_start

        # Ditto-style local personalization: optimize a separate local adapter
        # and regularize it explicitly toward the global broadcast. It is never
        # aggregated, so global and personalized objectives remain separated.
        personal_model = self.model(global_adapter, personalized=True)
        for _ in range(self.config.local_epochs):
            for sample in samples:
                personal_model.train_step(
                    sample,
                    self.config.local_lr,
                    model_input=(
                        sample
                        + np.float32(self.variant.noise_sigma)
                        * self.rng.standard_normal(sample.shape).astype(np.float32)
                        if self.variant.noise_sigma > 0.0
                        else None
                    ),
                    proximal_center=np.asarray(global_adapter, dtype=np.float32),
                    proximal_mu=self.config.personal_prox_mu,
                )
        self.pending_personal_candidate = personal_model.get_trainable().copy()
        update_weight = self.pending_training_windows
        self.pending_training_windows = 0
        if self.config.firmware_compression:
            corrected = delta
            if self.config.error_feedback:
                corrected = corrected + self.compression_residual
            compressed, _ = _sparse_int8_reconstruction(
                np.zeros_like(corrected), corrected, self.config.sparsity_k
            )
            if self.config.error_feedback:
                self.compression_residual = np.asarray(
                    corrected - compressed, dtype=np.float32
                )
            delta = compressed
        self.last_compression_error_norm = float(
            np.linalg.norm(self.compression_residual)
        )
        return np.asarray(delta, dtype=np.float32), update_weight

    def accept_global(self, new_global_adapter: np.ndarray) -> None:
        if self.pending_personal_candidate is not None:
            target_residual = self.pending_personal_candidate - np.asarray(
                new_global_adapter, dtype=np.float32
            )
            rate = np.float32(self.config.personal_rate)
            self.personal_adapter = (
                (np.float32(1.0) - rate) * self.personal_adapter
                + rate * target_residual
            ).astype(np.float32)
            self.pending_personal_candidate = None
            norm = float(np.linalg.norm(self.personal_adapter))
            if norm > self.config.personal_norm_limit:
                self.personal_adapter *= np.float32(
                    self.config.personal_norm_limit / max(norm, 1e-12)
                )

        if not self.contexts:
            return
        current_model = self.model(new_global_adapter, personalized=True)
        calibration_by_domain: dict[str, list[np.ndarray]] = defaultdict(list)
        for item in self.calibration_items:
            calibration_by_domain[item.domain_key].append(item.raw_features)
        for domain_key, state in self.contexts.items():
            state.recalibrate_scores(
                current_model,
                calibration_by_domain[domain_key],
                self.config.ae_fusion_weight,
            )


def _take_balanced_block(
    shard: DeviceShard,
    positions: dict[str, int],
    field: str,
    limit: int,
) -> list[StreamWindow]:
    """Take a chronological block; fixed-site shards contain one domain."""
    output: list[StreamWindow] = []
    keys = sorted(shard.streams)
    while len(output) < limit:
        progressed = False
        for domain_key in keys:
            sequence = getattr(shard.streams[domain_key], field)
            position = positions[domain_key]
            if position >= len(sequence):
                continue
            output.append(sequence[position])
            positions[domain_key] = position + 1
            progressed = True
            if len(output) >= limit:
                break
        if not progressed:
            break
    return output


def _synchronized_round_blocks(
    sequence: list[StreamWindow], round_count: int
) -> list[list[StreamWindow]]:
    """Partition one chronological site stream across shared rounds once."""
    if round_count <= 0:
        raise ValueError("round_count must be positive")
    quotient, remainder = divmod(len(sequence), round_count)
    blocks: list[list[StreamWindow]] = []
    position = 0
    for round_index in range(round_count):
        size = quotient + int(round_index < remainder)
        blocks.append(sequence[position : position + size])
        position += size
    if position != len(sequence):
        raise RuntimeError("synchronized partition did not consume the sequence")
    return blocks


def _normalize_static(
    windows: list[StreamWindow], shard: DeviceShard
) -> np.ndarray:
    samples = []
    for window in windows:
        stream = shard.streams[window.domain_key]
        samples.append(
            (window.raw_features - stream.initial_mean) / stream.initial_std
        )
    return np.asarray(samples, dtype=np.float32)


def federated_commissioning(
    shards: dict[str, DeviceShard], config: RealtimeConfig
) -> tuple[list[np.ndarray], list[dict[str, object]], list[dict[str, object]]]:
    """Train the dense base until every known-normal device window is consumed."""
    initial = _init_base_weights(LARGER_DIMS)
    global_flat = DenseAE(LARGER_DIMS, weights=initial).get_trainable()
    sequences = {
        device_id: next(iter(shard.streams.values())).train
        for device_id, shard in shards.items()
    }
    round_count = max(
        (len(sequence) + config.commissioning_batch - 1)
        // config.commissioning_batch
        for sequence in sequences.values()
    )
    round_blocks = {
        device_id: _synchronized_round_blocks(sequence, round_count)
        for device_id, sequence in sequences.items()
    }
    records: list[dict[str, object]] = []
    for round_index in range(round_count):
        contributions: list[tuple[np.ndarray, int]] = []
        for device_id, shard in shards.items():
            batch_windows = round_blocks[device_id][round_index]
            if not batch_windows:
                continue
            batch = _normalize_static(batch_windows, shard)
            local = DenseAE(LARGER_DIMS, weights=initial)
            local.set_trainable(global_flat, reset_optimizer=True)
            before = float(np.mean(local.losses(batch)))
            local.train(
                batch,
                epochs=config.commissioning_local_epochs,
                lr=config.commissioning_lr,
            )
            after = float(np.mean(local.losses(batch)))
            local_flat = local.get_trainable()
            contributions.append((local_flat, len(batch)))
            records.append(
                {
                    "stage": "dense_commissioning",
                    "round": round_index,
                    "scheduled_rounds": round_count,
                    "device_id": device_id,
                    "samples": len(batch),
                    "sample_visits": len(batch)
                    * config.commissioning_local_epochs,
                    "loss_before": before,
                    "loss_after": after,
                    "update_norm": float(
                        np.linalg.norm(local_flat - global_flat)
                    ),
                }
            )
        if len(contributions) >= config.minimum_clients_per_round:
            global_flat = _flat_aggregate_fedavg(
                global_flat, contributions, len(contributions)
            )

    trained = DenseAE(LARGER_DIMS, weights=initial)
    trained.set_trainable(global_flat, reset_optimizer=True)
    audit: list[dict[str, object]] = []
    for device_id, shard in shards.items():
        for domain_key, stream in shard.streams.items():
            consumed = sum(len(block) for block in round_blocks[device_id])
            expected = len(stream.train)
            if consumed != expected:
                raise RuntimeError(
                    f"commissioning did not consume {device_id}/{domain_key}: "
                    f"{consumed} of {expected}"
                )
            audit.append(
                {
                    "stage": "dense_commissioning",
                    "device_id": device_id,
                    "dataset": stream.dataset,
                    "logical_client": stream.logical_client,
                    "expected_windows": expected,
                    "seen_windows": consumed,
                    "gradient_selected_windows": consumed,
                    "complete": consumed == expected,
                }
            )
    return [weight.copy() for weight in trained.W], records, audit


def _aggregate_operation_updates(
    updates: dict[str, tuple[np.ndarray, int]],
) -> np.ndarray:
    total = sum(samples for _, samples in updates.values())
    aggregate = np.zeros_like(next(iter(updates.values()))[0], dtype=np.float32)
    for delta, samples in updates.values():
        aggregate += np.asarray(delta, dtype=np.float32) * np.float32(samples)
    return aggregate / np.float32(total)


class AdaptiveServerOptimizer:
    """FedAvg/FedAdam/FedYogi state held entirely on the PC server."""

    def __init__(
        self, mode: str, size: int, config: RealtimeConfig
    ) -> None:
        if mode not in VALID_SERVER_OPTIMIZERS:
            raise ValueError(f"unsupported server optimizer: {mode}")
        self.mode = mode
        self.config = config
        self.m = np.zeros(size, dtype=np.float32)
        self.v = np.zeros(size, dtype=np.float32)
        self.t = 0

    def step(self, global_adapter: np.ndarray, delta: np.ndarray) -> np.ndarray:
        update = np.asarray(delta, dtype=np.float32)
        if self.mode == "fedavg":
            return (np.asarray(global_adapter, dtype=np.float32) + update).astype(
                np.float32
            )
        beta1 = np.float32(self.config.server_beta1)
        beta2 = np.float32(self.config.server_beta2)
        self.t += 1
        self.m = beta1 * self.m + (np.float32(1.0) - beta1) * update
        squared = update * update
        if self.mode == "fedadam":
            self.v = beta2 * self.v + (np.float32(1.0) - beta2) * squared
        else:
            # Yogi prevents the second moment from growing monotonically when
            # heterogeneous clients occasionally produce a large update.
            self.v = self.v - (np.float32(1.0) - beta2) * squared * np.sign(
                self.v - squared
            )
            self.v = np.maximum(self.v, np.float32(0.0))
        m_hat = self.m / np.float32(1.0 - self.config.server_beta1 ** self.t)
        v_hat = self.v / np.float32(1.0 - self.config.server_beta2 ** self.t)
        server_step = np.float32(self.config.server_learning_rate) * m_hat / (
            np.sqrt(v_hat) + np.float32(self.config.server_tau)
        )
        return (
            np.asarray(global_adapter, dtype=np.float32) + server_step
        ).astype(np.float32)

    def state_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "t": self.t,
            "m": self.m.copy(),
            "v": self.v.copy(),
        }


def run_variant(
    shards: dict[str, DeviceShard],
    base_weights: list[np.ndarray],
    config: RealtimeConfig,
    seed: int,
    variant: StudyVariant,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, RuntimeDevice],
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    adapter_size = ABLoRA(
        LARGER_DIMS,
        variant.rank,
        base_weights,
        train_a=False,
        alpha=_constructor_alpha(config, variant),
    ).get_trainable().size
    master_adapter = np.zeros(adapter_size, dtype=np.float32)
    broadcast_adapter = np.zeros(adapter_size, dtype=np.float32)
    server_optimizer = AdaptiveServerOptimizer(
        variant.server_optimizer, adapter_size, config
    )
    runtimes = {
        device_id: RuntimeDevice(
            shard, base_weights, broadcast_adapter, config, seed, variant
        )
        for device_id, shard in shards.items()
    }
    operation_sequences = {
        device_id: next(iter(shard.streams.values())).operation
        for device_id, shard in shards.items()
    }
    round_count = max(
        (len(sequence) + config.operation_block - 1) // config.operation_block
        for sequence in operation_sequences.values()
    )
    operation_blocks = {
        device_id: _synchronized_round_blocks(sequence, round_count)
        for device_id, sequence in operation_sequences.items()
    }
    window_records: list[dict[str, object]] = []
    federation_records: list[dict[str, object]] = []
    for stream_round in range(round_count):
        updates: dict[str, tuple[np.ndarray, int]] = {}
        selected_this_round = 0
        for device_id, runtime in runtimes.items():
            block = operation_blocks[device_id][stream_round]
            if not block:
                continue
            records, update, update_weight, selected = runtime.process_block(
                block, broadcast_adapter, stream_round, seed
            )
            selected_this_round += selected
            window_records.extend(records)
            if update is not None and update_weight > 0:
                updates[device_id] = (update, update_weight)

        shared_delta = np.zeros_like(master_adapter)
        federated = len(updates) >= config.minimum_clients_per_round
        if federated:
            shared_delta = _aggregate_operation_updates(updates)
            master_adapter = server_optimizer.step(master_adapter, shared_delta)
            if config.firmware_compression:
                scale, packed = quantize_dynamic(master_adapter)
                broadcast_adapter = dequantize_dynamic(packed, scale).astype(
                    np.float32
                )
            else:
                broadcast_adapter = master_adapter.copy()
        # Commit local personalization and recalibrate every device against
        # its exact effective model, even in a round where the global server
        # update is skipped.
        for runtime in runtimes.values():
            runtime.accept_global(broadcast_adapter)

        retained = min(config.sparsity_k, len(broadcast_adapter))
        compression_errors = [
            runtimes[device_id].last_compression_error_norm
            for device_id in updates
        ]
        if updates:
            uplink_bytes_per_participant = (
                12 + 3 * retained
                if config.firmware_compression
                else 4 * len(broadcast_adapter)
            )
        else:
            uplink_bytes_per_participant = 0
        broadcast_bytes_per_device = (
            12 + len(broadcast_adapter)
            if config.firmware_compression and federated
            else 4 * len(broadcast_adapter) if federated else 0
        )
        broadcast_recipients = len(runtimes) if federated else 0
        uplink_total_bytes = uplink_bytes_per_participant * len(updates)
        broadcast_total_bytes = (
            broadcast_bytes_per_device * broadcast_recipients
        )
        federation_records.append(
            {
                "stage": "online_lora",
                "seed": seed,
                "study_variant": variant.name,
                "policy": variant.policy,
                "server_optimizer": variant.server_optimizer,
                "lora_rank": variant.rank,
                "rank_stabilized": variant.rank_stabilized,
                "noise_sigma": variant.noise_sigma,
                "stream_round": stream_round,
                "scheduled_rounds": round_count,
                "configured_devices": len(runtimes),
                "participating_devices": len(updates),
                "participation_fraction": float(
                    len(updates) / max(len(runtimes), 1)
                ),
                "federated": federated,
                "new_gradient_windows": selected_this_round,
                "shared_update_norm": float(np.linalg.norm(shared_delta)),
                "server_moment_norm": float(np.linalg.norm(server_optimizer.m)),
                "server_second_moment_norm": float(
                    np.linalg.norm(server_optimizer.v)
                ),
                "master_adapter_norm": float(np.linalg.norm(master_adapter)),
                "broadcast_adapter_norm": float(
                    np.linalg.norm(broadcast_adapter)
                ),
                "broadcast_quantization_error": float(
                    np.linalg.norm(master_adapter - broadcast_adapter)
                ),
                "mean_uplink_error_feedback_norm": float(
                    mean(compression_errors) if compression_errors else 0.0
                ),
                "uplink_bytes_per_participant": uplink_bytes_per_participant,
                "uplink_total_bytes": uplink_total_bytes,
                "broadcast_bytes_per_device": broadcast_bytes_per_device,
                "broadcast_recipients": broadcast_recipients,
                "broadcast_total_bytes": broadcast_total_bytes,
                "round_total_communication_bytes": (
                    uplink_total_bytes + broadcast_total_bytes
                ),
            }
        )

    audit: list[dict[str, object]] = []
    for device_id, runtime in runtimes.items():
        for domain_key, stream in runtime.shard.streams.items():
            expected = len(stream.operation)
            seen = runtime.counters[domain_key]["seen"]
            selected = runtime.counters[domain_key]["gradient_selected"]
            if seen != expected:
                raise RuntimeError(
                    f"operation did not consume {device_id}/{domain_key}: "
                    f"{seen} of {expected}"
                )
            if variant.policy == "all" and selected != expected:
                raise RuntimeError(
                    f"train-all skipped gradients for {device_id}/{domain_key}: "
                    f"{selected} of {expected}"
                )
            audit.append(
                {
                    "stage": "online_lora",
                    "seed": seed,
                    "study_variant": variant.name,
                    "policy": variant.policy,
                    "server_optimizer": variant.server_optimizer,
                    "lora_rank": variant.rank,
                    "rank_stabilized": variant.rank_stabilized,
                    "device_id": device_id,
                    "dataset": stream.dataset,
                    "logical_client": stream.logical_client,
                    "expected_windows": expected,
                    "seen_windows": seen,
                    "gradient_selected_windows": selected,
                    "trusted_windows": runtime.counters[domain_key]["trusted"],
                    "uncertain_windows": runtime.counters[domain_key]["uncertain"],
                    "alarm_windows": runtime.counters[domain_key]["alarms"],
                    "latched_windows": runtime.counters[domain_key]["latched"],
                    "complete": seen == expected,
                }
            )
    return (
        window_records,
        federation_records,
        audit,
        runtimes,
        master_adapter,
        broadcast_adapter,
        server_optimizer.state_dict(),
    )


def _new_calibrated_state(
    stream: DeviceDomainStream,
    model: ABLoRA,
    config: RealtimeConfig,
    weight: float,
    mahalanobis_weight: float | None = None,
) -> CalibrationState:
    state = CalibrationState(
        stream.dataset, stream.initial_mean, stream.initial_std, config
    )
    if mahalanobis_weight is not None:
        state.mahalanobis_weight = mahalanobis_weight
    state.seed_scores(model, stream.calibration_seed, weight)
    return state


def _deployment_calibration_count(
    stream: DeviceDomainStream, config: RealtimeConfig
) -> int:
    proportional = int(
        np.ceil(config.deployment_calibration_fraction * len(stream.server_test))
    )
    requested = max(config.deployment_calibration_windows, proportional)
    return min(
        requested,
        len(stream.server_test) - config.minimum_calibration_windows,
    )


def _deployment_calibrated_state(
    stream: DeviceDomainStream,
    model: ABLoRA,
    config: RealtimeConfig,
    weight: float,
    mahalanobis_weight: float | None = None,
) -> tuple[CalibrationState, list[StreamWindow]]:
    """Commission an unseen deployment stream before prequential scoring.

    The fixed chronological prefix is selected without inspecting labels.  A
    separate protocol audit proves that it is operator-confirmed healthy and
    that the remaining stream still contains both normal and fault windows.
    """
    count = _deployment_calibration_count(stream, config)
    if count < config.minimum_calibration_windows:
        raise ValueError(
            f"{stream.domain_key} has only {count} deployment calibration windows"
        )
    calibration = stream.server_test[:count]
    estimator = OnlineWelford(width=12)
    for window in calibration:
        estimator.update(window.raw_features)
    initial_mean, initial_std = estimator.statistics()
    initial_std = np.where(np.asarray(initial_std) < 1e-6, 1.0, initial_std)
    state = CalibrationState(
        stream.dataset,
        np.asarray(initial_mean, dtype=np.float32),
        np.asarray(initial_std, dtype=np.float32),
        config,
    )
    if mahalanobis_weight is not None:
        state.mahalanobis_weight = mahalanobis_weight
    state.seed_scores(model, calibration, weight)
    return state, stream.server_test[count:]


def _evaluate_stream(
    *,
    stream: DeviceDomainStream,
    state: CalibrationState,
    model: ABLoRA,
    weight: float,
    seed: int,
    study_variant: str,
    policy: str,
    model_variant: str,
    server_optimizer: str,
    lora_rank: int,
    rank_stabilized: bool,
    windows: list[StreamWindow] | None = None,
    allow_calibration: bool = False,
    evaluation_protocol: str = "frozen_cross_source",
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    evaluation_windows = stream.server_test if windows is None else windows
    for order_index, window in enumerate(evaluation_windows):
        decision = state.observe(
            window,
            model,
            allow_calibration=allow_calibration,
            weight=weight,
        )
        records.append(
            {
                "scope": "frozen_server_test",
                "seed": seed,
                "study_variant": study_variant,
                "policy": policy,
                "model_variant": model_variant,
                "server_optimizer": server_optimizer,
                "lora_rank": lora_rank,
                "rank_stabilized": rank_stabilized,
                "evaluation_protocol": evaluation_protocol,
                "deployment_calibration_windows": (
                    len(stream.server_test) - len(evaluation_windows)
                    if evaluation_protocol == "deployment_prequential"
                    else 0
                ),
                "device_id": stream.device_id,
                "dataset": stream.dataset,
                "logical_client": stream.logical_client,
                "client_name": _client_display_name(
                    stream.dataset, stream.logical_client
                ),
                "split": "server_test",
                "order_index": order_index,
                "window_id": window.window_id,
                "source_file": window.source_file,
                "start_sample": window.start_sample,
                "label": window.label,
                **decision,
            }
        )
    return records


def evaluate_baselines(
    shards: dict[str, DeviceShard],
    config: RealtimeConfig,
    seed: int,
) -> list[dict[str, object]]:
    initial_base = _init_base_weights(LARGER_DIMS)
    zero_adapter = ABLoRA(
        LARGER_DIMS,
        1,
        initial_base,
        train_a=False,
        alpha=config.lora_alpha,
    ).get_trainable()
    records: list[dict[str, object]] = []
    for shard in shards.values():
        for stream in shard.streams.values():
            for variant, weight in (
                ("untrained_autoencoder", config.ae_fusion_weight),
                ("magnitude_only", 0.0),
            ):
                model = ABLoRA(
                    LARGER_DIMS,
                    1,
                    initial_base,
                    train_a=False,
                    alpha=config.lora_alpha,
                )
                model.set_trainable(zero_adapter, reset_optimizer=True)
                state, evaluation_windows = _deployment_calibrated_state(
                    stream, model, config, weight, mahalanobis_weight=0.0
                )
                records.extend(
                    _evaluate_stream(
                        stream=stream,
                        state=state,
                        model=model,
                        weight=weight,
                        seed=seed,
                        study_variant="baseline",
                        policy="baseline",
                        model_variant=variant,
                        server_optimizer="none",
                        lora_rank=1,
                        rank_stabilized=False,
                        windows=evaluation_windows,
                        allow_calibration=True,
                        evaluation_protocol="deployment_prequential",
                    )
                )
    return records


def evaluate_final_models(
    shards: dict[str, DeviceShard],
    runtimes: dict[str, RuntimeDevice],
    broadcast_adapter: np.ndarray,
    config: RealtimeConfig,
    seed: int,
    variant: StudyVariant,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for device_id, shard in shards.items():
        runtime = runtimes[device_id]
        for domain_key, stream in shard.streams.items():
            for variant_name, personalized in (
                ("final_global", False),
                ("final_personalized", True),
            ):
                model = runtime.model(
                    broadcast_adapter, personalized=personalized
                )
                frozen_state = runtime.contexts[domain_key].frozen_copy()
                frozen_state.recalibrate_scores(
                    model,
                    [
                        item.raw_features
                        for item in runtime.calibration_items
                        if item.domain_key == domain_key
                    ],
                    config.ae_fusion_weight,
                )
                records.extend(
                    _evaluate_stream(
                        stream=stream,
                        state=frozen_state,
                        model=model,
                        weight=config.ae_fusion_weight,
                        seed=seed,
                        study_variant=variant.name,
                        policy=variant.policy,
                        model_variant=f"{variant_name}_frozen",
                        server_optimizer=variant.server_optimizer,
                        lora_rank=variant.rank,
                        rank_stabilized=variant.rank_stabilized,
                    )
                )
                deployment_state, evaluation_windows = _deployment_calibrated_state(
                    stream, model, config, config.ae_fusion_weight
                )
                records.extend(
                    _evaluate_stream(
                        stream=stream,
                        state=deployment_state,
                        model=model,
                        weight=config.ae_fusion_weight,
                        seed=seed,
                        study_variant=variant.name,
                        policy=variant.policy,
                        model_variant=variant_name,
                        server_optimizer=variant.server_optimizer,
                        lora_rank=variant.rank,
                        rank_stabilized=variant.rank_stabilized,
                        windows=evaluation_windows,
                        allow_calibration=True,
                        evaluation_protocol="deployment_prequential",
                    )
                )
    return records


def _alarm_metrics(
    labels: np.ndarray, alarms: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(alarms, dtype=np.int8)
    tp = int(np.sum((predicted == 1) & (truth == 1)))
    fp = int(np.sum((predicted == 1) & (truth == 0)))
    fn = int(np.sum((predicted == 0) & (truth == 1)))
    tn = int(np.sum((predicted == 0) & (truth == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    if set(truth.tolist()) == {0, 1}:
        ranking = classification_metrics(truth, scores, float("inf"))
        roc_auc = float(ranking["roc_auc"])
        pr_auc = float(ranking["pr_auc"])
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def _detection_delay(rows: list[dict[str, object]]) -> float:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["device_id"])].append(row)
    delays: list[float] = []
    for group in groups.values():
        first_fault = next(
            (index for index, row in enumerate(group) if int(row["label"]) == 1),
            None,
        )
        if first_fault is None:
            continue
        first_alarm = next(
            (
                index
                for index in range(first_fault, len(group))
                if bool(group[index]["alarm"])
            ),
            None,
        )
        delays.append(
            float(
                (first_alarm - first_fault)
                if first_alarm is not None
                else len(group) - first_fault
            )
        )
    return float(mean(delays)) if delays else float("nan")


def _temporal_alarm_metrics(
    rows: list[dict[str, object]],
) -> dict[str, float | int]:
    """Report event alarms separately from the operational alarm latch."""
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["device_id"])].append(row)

    pre_fault_windows = 0
    pre_fault_alarm_windows = 0
    false_alarm_events = 0
    devices_with_fault = 0
    devices_latched_before_fault = 0
    latched_windows = 0
    for group in groups.values():
        latched_windows += sum(
            bool(row.get("alarm_latched", row["alarm"])) for row in group
        )
        first_fault = next(
            (index for index, row in enumerate(group) if int(row["label"]) == 1),
            None,
        )
        prefix_end = first_fault if first_fault is not None else len(group)
        prefix = group[:prefix_end]
        pre_fault_windows += len(prefix)
        pre_fault_alarm_windows += sum(bool(row["alarm"]) for row in prefix)
        previous_alarm = False
        for row in prefix:
            current_alarm = bool(row["alarm"])
            false_alarm_events += int(current_alarm and not previous_alarm)
            previous_alarm = current_alarm
        if first_fault is not None:
            devices_with_fault += 1
            if first_fault > 0 and bool(
                group[first_fault - 1].get(
                    "alarm_latched", group[first_fault - 1]["alarm"]
                )
            ):
                devices_latched_before_fault += 1

    total_windows = len(rows)
    return {
        "pre_fault_windows": pre_fault_windows,
        "pre_fault_alarm_windows": pre_fault_alarm_windows,
        "pre_fault_alarm_fraction": (
            pre_fault_alarm_windows / pre_fault_windows
            if pre_fault_windows
            else float("nan")
        ),
        "false_alarm_events": false_alarm_events,
        "latched_fraction": (
            latched_windows / total_windows if total_windows else float("nan")
        ),
        "devices_latched_before_fault": devices_latched_before_fault,
        "devices_with_fault": devices_with_fault,
    }


def summarize_server_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[
        tuple[int, str, str, str, str, str], list[dict[str, object]]
    ] = defaultdict(list)
    for row in records:
        key = (
            int(row["seed"]),
            str(row["study_variant"]),
            str(row["policy"]),
            str(row["model_variant"]),
            str(row["dataset"]),
            str(row["logical_client"]),
        )
        groups[key].append(row)
    output: list[dict[str, object]] = []
    for (seed, study_variant, policy, model_variant, dataset, logical_client), rows in sorted(
        groups.items()
    ):
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
        alarms = np.asarray([bool(row["alarm"]) for row in rows], dtype=np.int8)
        scores = np.asarray([float(row["combined_score"]) for row in rows])
        output.append(
            {
                "seed": seed,
                "study_variant": study_variant,
                "policy": policy,
                "model_variant": model_variant,
                "evaluation_protocol": rows[0]["evaluation_protocol"],
                "deployment_calibration_windows": rows[0][
                    "deployment_calibration_windows"
                ],
                "server_optimizer": rows[0]["server_optimizer"],
                "lora_rank": rows[0]["lora_rank"],
                "rank_stabilized": rows[0]["rank_stabilized"],
                "dataset": dataset,
                "logical_client": logical_client,
                "client_name": _client_display_name(dataset, logical_client),
                "windows": len(rows),
                "trusted_fraction": float(
                    np.mean([bool(row["trusted"]) for row in rows])
                ),
                "uncertain_fraction": float(
                    np.mean([bool(row["uncertain"]) for row in rows])
                ),
                "detection_delay_windows": _detection_delay(rows),
                **_temporal_alarm_metrics(rows),
                **_alarm_metrics(labels, alarms, scores),
            }
        )
    return output


def summarize_prequential_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[
        tuple[int, str, str, str, str, str], list[dict[str, object]]
    ] = defaultdict(list)
    for row in records:
        key = (
            int(row["seed"]),
            str(row["study_variant"]),
            str(row["policy"]),
            str(row["dataset"]),
            str(row["logical_client"]),
            str(row["split"]),
        )
        groups[key].append(row)
    output: list[dict[str, object]] = []
    for (seed, study_variant, policy, dataset, logical_client, split), rows in sorted(groups.items()):
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
        alarms = np.asarray([bool(row["alarm"]) for row in rows], dtype=np.int8)
        scores = np.asarray([float(row["combined_score"]) for row in rows])
        output.append(
            {
                "seed": seed,
                "study_variant": study_variant,
                "policy": policy,
                "server_optimizer": rows[0]["server_optimizer"],
                "lora_rank": rows[0]["lora_rank"],
                "rank_stabilized": rows[0]["rank_stabilized"],
                "dataset": dataset,
                "logical_client": logical_client,
                "client_name": _client_display_name(dataset, logical_client),
                "split": split,
                "windows": len(rows),
                "gradient_selected_fraction": float(
                    np.mean([bool(row["gradient_selected"]) for row in rows])
                ),
                "trusted_fraction": float(
                    np.mean([bool(row["trusted"]) for row in rows])
                ),
                "maximum_cusum": float(max(float(row["cusum"]) for row in rows)),
                **_temporal_alarm_metrics(rows),
                **_alarm_metrics(labels, alarms, scores),
            }
        )
    return output


def aggregate_seed_metrics(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[
        tuple[str, str, str, str, str], list[dict[str, object]]
    ] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["study_variant"]),
                str(row["policy"]),
                str(row["model_variant"]),
                str(row["dataset"]),
                str(row["logical_client"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    metrics = (
        "precision",
        "recall",
        "f1",
        "fpr",
        "roc_auc",
        "pr_auc",
        "trusted_fraction",
        "uncertain_fraction",
        "detection_delay_windows",
        "pre_fault_alarm_fraction",
        "false_alarm_events",
        "latched_fraction",
        "devices_latched_before_fault",
    )
    for (study_variant, policy, model_variant, dataset, logical_client), group in sorted(groups.items()):
        item: dict[str, object] = {
            "study_variant": study_variant,
            "policy": policy,
            "model_variant": model_variant,
            "evaluation_protocol": group[0].get(
                "evaluation_protocol", "device_prequential"
            ),
            "server_optimizer": group[0]["server_optimizer"],
            "lora_rank": group[0]["lora_rank"],
            "rank_stabilized": group[0]["rank_stabilized"],
            "dataset": dataset,
            "logical_client": logical_client,
            "client_name": _client_display_name(dataset, logical_client),
            "seeds": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            average, deviation, ci95 = _mean_std_ci95(finite)
            item[f"{metric}_mean"] = average
            item[f"{metric}_std"] = deviation
            item[f"{metric}_ci95"] = ci95
            item[f"{metric}_n"] = len(finite)
        output.append(item)
    return output


def _t_critical_95(sample_count: int) -> float:
    """Two-sided 95% Student-t critical value without a SciPy dependency."""
    values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    if sample_count <= 1:
        return float("nan")
    return values.get(sample_count, 1.960)


def _mean_std_ci95(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan"), float("nan")
    average = float(np.mean(finite))
    if len(finite) == 1:
        return average, 0.0, float("nan")
    deviation = float(np.std(finite, ddof=1))
    ci95 = float(_t_critical_95(len(finite)) * deviation / np.sqrt(len(finite)))
    return average, deviation, ci95


def build_paired_comparisons(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Paired seed deltas for the controlled ablations and baseline gains."""
    contrasts = (
        (
            "fedadam_minus_fedavg_r1_global",
            "fedadam_r1_trusted",
            "final_global",
            "fedavg_r1_trusted",
            "final_global",
        ),
        (
            "fedadam_minus_fedavg_r1_personalized",
            "fedadam_r1_trusted",
            "final_personalized",
            "fedavg_r1_trusted",
            "final_personalized",
        ),
        (
            "fedyogi_minus_fedadam_r1_global",
            "fedyogi_r1_trusted",
            "final_global",
            "fedadam_r1_trusted",
            "final_global",
        ),
        (
            "fedyogi_minus_fedadam_r1_personalized",
            "fedyogi_r1_trusted",
            "final_personalized",
            "fedadam_r1_trusted",
            "final_personalized",
        ),
        (
            "rank2_minus_rank1_fedadam_global",
            "fedadam_rs_r2_trusted",
            "final_global",
            "fedadam_r1_trusted",
            "final_global",
        ),
        (
            "rank2_minus_rank1_fedadam_personalized",
            "fedadam_rs_r2_trusted",
            "final_personalized",
            "fedadam_r1_trusted",
            "final_personalized",
        ),
        (
            "rank4_denoise_minus_rank2_global",
            "fedadam_rs_r4_denoise",
            "final_global",
            "fedadam_rs_r2_trusted",
            "final_global",
        ),
        (
            "rank4_denoise_minus_rank2_personalized",
            "fedadam_rs_r4_denoise",
            "final_personalized",
            "fedadam_rs_r2_trusted",
            "final_personalized",
        ),
        (
            "train_all_minus_trusted_rank2_global",
            "fedadam_rs_r2_all",
            "final_global",
            "fedadam_rs_r2_trusted",
            "final_global",
        ),
        (
            "train_all_minus_trusted_rank2_personalized",
            "fedadam_rs_r2_all",
            "final_personalized",
            "fedadam_rs_r2_trusted",
            "final_personalized",
        ),
        (
            "recommended_global_minus_untrained",
            "fedadam_rs_r2_trusted",
            "final_global",
            "baseline",
            "untrained_autoencoder",
        ),
        (
            "recommended_global_minus_magnitude_only",
            "fedadam_rs_r2_trusted",
            "final_global",
            "baseline",
            "magnitude_only",
        ),
    )
    indexed: dict[tuple[str, str, str, str, int], dict[str, object]] = {}
    domains: set[tuple[str, str]] = set()
    for row in rows:
        dataset = str(row["dataset"])
        logical_client = str(row["logical_client"])
        key = (
            str(row["study_variant"]),
            str(row["model_variant"]),
            dataset,
            logical_client,
            int(row["seed"]),
        )
        indexed[key] = row
        domains.add((dataset, logical_client))

    metrics = ("f1", "precision", "recall", "fpr", "roc_auc", "pr_auc")
    output: list[dict[str, object]] = []
    available_seeds = sorted({int(row["seed"]) for row in rows})
    for (
        contrast,
        treatment_variant,
        treatment_model,
        control_variant,
        control_model,
    ) in contrasts:
        for dataset, logical_client in sorted(domains):
            paired_seeds = [
                seed
                for seed in available_seeds
                if (
                    treatment_variant,
                    treatment_model,
                    dataset,
                    logical_client,
                    seed,
                )
                in indexed
                and (
                    control_variant,
                    control_model,
                    dataset,
                    logical_client,
                    seed,
                )
                in indexed
            ]
            if not paired_seeds:
                continue
            item: dict[str, object] = {
                "contrast": contrast,
                "treatment_variant": treatment_variant,
                "treatment_model": treatment_model,
                "control_variant": control_variant,
                "control_model": control_model,
                "dataset": dataset,
                "logical_client": logical_client,
                "client_name": _client_display_name(dataset, logical_client),
                "paired_seeds": len(paired_seeds),
            }
            for metric in metrics:
                differences = np.asarray(
                    [
                        float(
                            indexed[
                                (
                                    treatment_variant,
                                    treatment_model,
                                    dataset,
                                    logical_client,
                                    seed,
                                )
                            ][metric]
                        )
                        - float(
                            indexed[
                                (
                                    control_variant,
                                    control_model,
                                    dataset,
                                    logical_client,
                                    seed,
                                )
                            ][metric]
                        )
                        for seed in paired_seeds
                    ],
                    dtype=np.float64,
                )
                average, deviation, ci95 = _mean_std_ci95(differences)
                item[f"delta_{metric}_mean"] = average
                item[f"delta_{metric}_std"] = deviation
                item[f"delta_{metric}_ci95"] = ci95
            output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    fields = sorted({field for row in rows for field in row})
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: object) -> None:
    """Replace JSON only after the temporary file has been fully written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: object) -> bool:
    """Interpret both in-memory booleans and booleans reloaded from CSV."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass"}
    return bool(value)


def build_protocol_audit(
    shards: dict[str, DeviceShard],
    config: RealtimeConfig,
) -> tuple[list[dict[str, object]], list[str]]:
    """Post-partition label audit; results never enter a device decision."""
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    server_groups: dict[tuple[str, str], list[StreamWindow]] = defaultdict(list)
    for device_id, shard in shards.items():
        for stream in shard.streams.values():
            calibration = [
                window
                for window in stream.operation
                if window.split == "calibration"
            ]
            normal = sum(window.label == 0 for window in calibration)
            fault = sum(window.label == 1 for window in calibration)
            status = "PASS" if fault == 0 else "FAIL"
            rows.append(
                {
                    "audit": "operator_confirmed_calibration",
                    "device_id": device_id,
                    "dataset": stream.dataset,
                    "logical_client": stream.logical_client,
                    "windows": len(calibration),
                    "normal_windows": normal,
                    "fault_windows": fault,
                    "status": status,
                }
            )
            if status == "FAIL":
                failures.append(
                    f"{device_id}/{stream.domain_key} operator-confirmed calibration contains faults"
                )
            deployment_count = _deployment_calibration_count(stream, config)
            deployment_calibration = stream.server_test[:deployment_count]
            deployment_evaluation = stream.server_test[deployment_count:]
            calibration_faults = sum(
                window.label == 1 for window in deployment_calibration
            )
            evaluation_normal = sum(
                window.label == 0 for window in deployment_evaluation
            )
            evaluation_fault = sum(
                window.label == 1 for window in deployment_evaluation
            )
            deployment_status = (
                "PASS"
                if len(deployment_calibration)
                >= config.minimum_calibration_windows
                and calibration_faults == 0
                and evaluation_normal > 0
                and evaluation_fault > 0
                else "FAIL"
            )
            rows.append(
                {
                    "audit": "deployment_prequential_protocol",
                    "device_id": device_id,
                    "dataset": stream.dataset,
                    "logical_client": stream.logical_client,
                    "windows": len(deployment_calibration),
                    "normal_windows": len(deployment_calibration)
                    - calibration_faults,
                    "fault_windows": calibration_faults,
                    "evaluation_normal_windows": evaluation_normal,
                    "evaluation_fault_windows": evaluation_fault,
                    "status": deployment_status,
                }
            )
            if deployment_status == "FAIL":
                failures.append(
                    f"{device_id}/{stream.domain_key} invalid deployment calibration/evaluation chronology"
                )
            server_groups[(stream.dataset, stream.logical_client)].extend(
                stream.server_test
            )
    for (dataset, logical_client), windows in sorted(server_groups.items()):
        normal = sum(window.label == 0 for window in windows)
        fault = sum(window.label == 1 for window in windows)
        status = "PASS" if normal > 0 and fault > 0 else "FAIL"
        rows.append(
            {
                "audit": "server_holdout_class_coverage",
                "device_id": "server",
                "dataset": dataset,
                "logical_client": logical_client,
                "windows": len(windows),
                "normal_windows": normal,
                "fault_windows": fault,
                "status": status,
            }
        )
        if status == "FAIL":
            failures.append(
                f"{dataset}/{logical_client} server holdout needs normal and fault windows"
            )
    return rows, failures


def _run_signature(
    config: RealtimeConfig, variant: StudyVariant | None, seed: int
) -> str:
    payload = {
        "config": asdict(config),
        "variant": asdict(variant) if variant is not None else "baseline",
        "seed": seed,
        "code_schema": 21,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _buffer_as_fixed(buffer: RobustScoreBuffer, capacity: int) -> tuple[np.ndarray, int]:
    values = np.asarray(buffer.values, dtype=np.float32)
    output = np.zeros(capacity, dtype=np.float32)
    count = min(len(values), capacity)
    if count:
        output[:count] = values[-count:]
    return output, count


def export_lilygo_states(
    run_dir: Path,
    runtimes: dict[str, RuntimeDevice],
    base_weights: list[np.ndarray],
    master_adapter: np.ndarray,
    broadcast_adapter: np.ndarray,
    server_state: dict[str, object],
    config: RealtimeConfig,
    variant: StudyVariant,
    seed: int,
) -> None:
    """Export fixed-shape arrays needed to continue on real LilyGO devices."""
    export_dir = run_dir / "lilygo_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    server_meta = {
        "schema": 3,
        "seed": seed,
        "variant": asdict(variant),
        "dims": LARGER_DIMS,
        "dtype": "float32 little-endian",
        "global_adapter_values": len(broadcast_adapter),
        "server_optimizer": server_state["mode"],
        "server_learning_rate": config.server_learning_rate,
        "server_beta1": config.server_beta1,
        "server_beta2": config.server_beta2,
        "server_tau": config.server_tau,
        "firmware_compression": config.firmware_compression,
        "error_feedback": config.error_feedback,
    }
    np.savez(
        export_dir / "server_state.npz",
        master_adapter=np.asarray(master_adapter, dtype=np.float32),
        broadcast_adapter=np.asarray(broadcast_adapter, dtype=np.float32),
        server_m=np.asarray(server_state["m"], dtype=np.float32),
        server_v=np.asarray(server_state["v"], dtype=np.float32),
        server_t=np.asarray([int(server_state["t"])], dtype=np.int32),
        metadata_utf8=np.frombuffer(
            json.dumps(server_meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
        ),
    )

    manifest: dict[str, object] = {"server": server_meta, "devices": []}
    for device_id, runtime in sorted(runtimes.items()):
        domain_keys = sorted(runtime.contexts)
        means = []
        variances = []
        thresholds = []
        ae_buffers = []
        magnitude_buffers = []
        mahalanobis_buffers = []
        inverse_covariances = []
        prognostic_centers = []
        prognostic_normalizer_means = []
        prognostic_normalizer_stds = []
        prognostic_inverse_covariances = []
        prognostic_buffers = []
        prognostic_buffer_counts = []
        prognostic_feature_histories = []
        prognostic_feature_history_counts = []
        combined_buffers = []
        buffer_counts = []
        temporal_state = []
        for domain_key in domain_keys:
            state = runtime.contexts[domain_key]
            trust, alarm, _, _ = state._thresholds()
            ae_values, ae_count = _buffer_as_fixed(
                state.ae_scores, config.score_buffer_size
            )
            magnitude_values, magnitude_count = _buffer_as_fixed(
                state.magnitude_scores, config.score_buffer_size
            )
            mahalanobis_values, mahalanobis_count = _buffer_as_fixed(
                state.mahalanobis_scores, config.score_buffer_size
            )
            prognostic_values, prognostic_count = _buffer_as_fixed(
                state.prognostic_scores,
                config.prognostic_score_buffer_size,
            )
            feature_history = np.zeros(
                (config.prognostic_window, len(state.prognostic_center)),
                dtype=np.float32,
            )
            history_values = np.asarray(
                state.prognostic_feature_history, dtype=np.float32
            )
            history_count = min(len(history_values), config.prognostic_window)
            if history_count:
                feature_history[:history_count] = history_values[-history_count:]
            combined_values, combined_count = _buffer_as_fixed(
                state.combined_scores, config.score_buffer_size
            )
            means.append(state.normalizer.mean)
            variances.append(state.normalizer.variance)
            thresholds.append((trust, alarm, state.cusum_reference))
            ae_buffers.append(ae_values)
            magnitude_buffers.append(magnitude_values)
            mahalanobis_buffers.append(mahalanobis_values)
            inverse_covariances.append(state.inverse_covariance)
            prognostic_centers.append(state.prognostic_center)
            prognostic_normalizer_means.append(
                state.prognostic_normalizer_mean
            )
            prognostic_normalizer_stds.append(
                state.prognostic_normalizer_std
            )
            prognostic_inverse_covariances.append(
                state.prognostic_inverse_covariance
            )
            prognostic_buffers.append(prognostic_values)
            prognostic_buffer_counts.append(prognostic_count)
            prognostic_feature_histories.append(feature_history)
            prognostic_feature_history_counts.append(history_count)
            combined_buffers.append(combined_values)
            buffer_counts.append(
                (ae_count, magnitude_count, mahalanobis_count, combined_count)
            )
            temporal_state.append(
                (
                    state.cusum,
                    float(state.recovery_count),
                    float(state.alarm_latched),
                    float(state.prognostic_latched),
                )
            )

        reference_model = runtime.model(broadcast_adapter, personalized=False)
        arrays: dict[str, np.ndarray] = {
            "global_B_flat": np.asarray(broadcast_adapter, dtype=np.float32),
            "personal_B_flat": np.asarray(runtime.personal_adapter, dtype=np.float32),
            "compression_residual": np.asarray(
                runtime.compression_residual, dtype=np.float32
            ),
            "normalizer_means": np.asarray(means, dtype=np.float32),
            "normalizer_variances": np.asarray(variances, dtype=np.float32),
            "score_thresholds": np.asarray(thresholds, dtype=np.float32),
            "ae_score_buffers": np.asarray(ae_buffers, dtype=np.float32),
            "magnitude_score_buffers": np.asarray(
                magnitude_buffers, dtype=np.float32
            ),
            "mahalanobis_score_buffers": np.asarray(
                mahalanobis_buffers, dtype=np.float32
            ),
            "inverse_covariances": np.asarray(
                inverse_covariances, dtype=np.float32
            ),
            "prognostic_centers": np.asarray(
                prognostic_centers, dtype=np.float32
            ),
            "prognostic_normalizer_means": np.asarray(
                prognostic_normalizer_means, dtype=np.float32
            ),
            "prognostic_normalizer_stds": np.asarray(
                prognostic_normalizer_stds, dtype=np.float32
            ),
            "prognostic_inverse_covariances": np.asarray(
                prognostic_inverse_covariances, dtype=np.float32
            ),
            "prognostic_score_buffers": np.asarray(
                prognostic_buffers, dtype=np.float32
            ),
            "prognostic_score_buffer_counts": np.asarray(
                prognostic_buffer_counts, dtype=np.int32
            ),
            "prognostic_feature_histories": np.asarray(
                prognostic_feature_histories, dtype=np.float32
            ),
            "prognostic_feature_history_counts": np.asarray(
                prognostic_feature_history_counts, dtype=np.int32
            ),
            "combined_score_buffers": np.asarray(
                combined_buffers, dtype=np.float32
            ),
            "score_buffer_counts": np.asarray(buffer_counts, dtype=np.int32),
            "temporal_state": np.asarray(temporal_state, dtype=np.float32),
        }
        for index, weight in enumerate(base_weights):
            arrays[f"W{index}"] = np.asarray(weight, dtype=np.float32)
        for index, fixed_a in enumerate(reference_model.A):
            arrays[f"A{index}"] = np.asarray(fixed_a, dtype=np.float32)
        device_meta = {
            "device_id": device_id,
            "domains": domain_keys,
            "dims": LARGER_DIMS,
            "rank": variant.rank,
            "rank_stabilized": variant.rank_stabilized,
            "constructor_alpha": _constructor_alpha(config, variant),
            "effective_lora_scaling": float(
                _constructor_alpha(config, variant) / variant.rank
            ),
            "A_shapes": [list(array.shape) for array in reference_model.A],
            "B_shapes": [list(shape) for shape in reference_model.trainable_shapes()],
            "score_buffer_size": config.score_buffer_size,
            "prognostic_enabled": [
                bool(runtime.contexts[key].prognostic_enabled)
                for key in domain_keys
            ],
            "prognostic_window": config.prognostic_window,
            "prognostic_history": config.prognostic_history,
            "prognostic_required": config.prognostic_required,
            "prognostic_sigma": config.prognostic_sigma,
            "prognostic_score_buffer_size": (
                config.prognostic_score_buffer_size
            ),
            "score_components": [
                "autoencoder",
                "magnitude",
                "mahalanobis",
                "prognostic_hotelling",
                "combined",
            ],
            "threshold_columns": ["trust", "alarm", "cusum_reference"],
            "temporal_columns": [
                "cusum",
                "recovery_count",
                "alarm_latched",
                "prognostic_latched",
            ],
        }
        arrays["metadata_utf8"] = np.frombuffer(
            json.dumps(device_meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
        np.savez(export_dir / f"{device_id}_state.npz", **arrays)
        manifest["devices"].append(device_meta)
    _write_json_atomic(export_dir / "lilygo_state_manifest.json", manifest)


def load_completed_run(
    run_dir: Path, expected_signature: str
) -> dict[str, object] | None:
    checkpoint_path = run_dir / "checkpoint.json"
    summary_path = run_dir / "run_summary.json"
    if not checkpoint_path.exists() or not summary_path.exists():
        return None
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        checkpoint.get("status") != "complete"
        or checkpoint.get("signature") != expected_signature
    ):
        return None
    required = {
        "federation_records",
        "training_audit",
        "prequential_metrics",
        "server_metrics",
    }
    if not isinstance(summary, dict) or not required.issubset(summary):
        return None
    if not all(isinstance(summary[key], list) for key in required):
        return None
    return summary


def save_completed_run(
    run_dir: Path,
    signature: str,
    *,
    device_records: list[dict[str, object]],
    server_records: list[dict[str, object]],
    federation_records: list[dict[str, object]],
    training_audit: list[dict[str, object]],
    prequential_metrics: list[dict[str, object]],
    server_metrics: list[dict[str, object]],
) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "device_predictions.csv", device_records)
    write_csv(run_dir / "server_predictions.csv", server_records)
    write_csv(run_dir / "federation_rounds.csv", federation_records)
    write_csv(run_dir / "training_audit.csv", training_audit)
    write_csv(run_dir / "prequential_metrics.csv", prequential_metrics)
    write_csv(run_dir / "server_metrics.csv", server_metrics)
    summary: dict[str, object] = {
        "federation_records": federation_records,
        "training_audit": training_audit,
        "prequential_metrics": prequential_metrics,
        "server_metrics": server_metrics,
    }
    _write_json_atomic(run_dir / "run_summary.json", summary)
    # Written last: only a fully persisted run may be resumed/skipped.
    _write_json_atomic(
        run_dir / "checkpoint.json",
        {"status": "complete", "signature": signature},
    )
    return summary


def write_report(
    path: Path,
    config: RealtimeConfig,
    summary: list[dict[str, object]],
    training_audit: list[dict[str, object]],
    paired_comparisons: list[dict[str, object]],
) -> None:
    lines = [
        "# Thesis Federated Device/Server Simulation",
        "",
        "Every active LilyGO remained at one fixed dataset/client site. All device-side windows were consumed before held-out deployment evaluation.",
        "Main model rows use a fixed operator-confirmed healthy prefix followed by predict-then-adapt prequential evaluation. Rows ending in `_frozen` are the stricter zero-adaptation cross-source diagnostic.",
        "Completed run checkpoints and LilyGO state exports are stored under `runs/`.",
        "",
        "## Held-out deployment and frozen diagnostic results",
        "",
        "| Study variant | Model | Client | F1 (95% CI) | FPR | Recall | AUROC | Delay |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['study_variant']} | {row['model_variant']} | {row['client_name']} | "
            f"{100*float(row['f1_mean']):.2f}% +/- {100*float(row['f1_ci95']):.2f} | "
            f"{100*float(row['fpr_mean']):.2f}% | "
            f"{100*float(row['recall_mean']):.2f}% | "
            f"{100*float(row['roc_auc_mean']):.2f}% | "
            f"{float(row['detection_delay_windows_mean']):.1f} |"
        )
    incomplete = [
        row for row in training_audit if not _as_bool(row.get("complete", True))
    ]
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            f"- Controlled paired rows: {len(paired_comparisons)}",
            "- Each delta is treatment minus control on the same seed and client.",
            "- `full_paired_comparisons.csv` reports sample SD and a two-sided Student-t 95% confidence interval.",
            "- A delta interval excluding zero is evidence of a consistent difference; it is not automatically practical significance.",
            "",
            "## Integrity checks",
            "",
            f"- Incomplete device streams: {len(incomplete)}",
            "- The fixed deployment prefix calibrates normalization and thresholds only; it never updates model weights.",
            "- Remaining deployment windows are predicted before trusted-only online calibration updates.",
            "- `_frozen` rows never update normalization, thresholds, replay, or weights.",
            "- Source files are allocated as indivisible groups.",
            "- The protocol audit checks calibration contamination and server class coverage.",
            "- Online alarms use bounded, decaying CUSUM with explicit recovery.",
            "- IMS/FEMTO additionally use a fixed 20-window causal Hotelling trend alarm fitted only on the healthy commissioning prefix; its irreversible latch is disabled for XJTU and laboratory datasets.",
            "- The prognostic reference normalization is frozen after commissioning so trusted online adaptation cannot learn away progressive degradation.",
            "- Labels are used only for the protocol audit and post-hoc metrics.",
            "- Global and personalized optimization are separate objectives.",
            "- The recommended hybrid adds a shrinkage Mahalanobis score calibrated only on healthy deployment data.",
            "- Client Top-K compression uses persistent error feedback.",
            "- The PC keeps the float32 adaptive-optimizer master state; devices receive quantized adapters.",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(asdict(config), indent=2),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_datasets(value: str) -> tuple[str, ...]:
    datasets = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(datasets) - set(ALL_DATASETS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown datasets: {', '.join(unknown)}")
    if not datasets:
        raise argparse.ArgumentTypeError("at least one dataset is required")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_seeds, default=(11, 29, 47, 71, 97))
    parser.add_argument("--datasets", type=parse_datasets, default=ALL_DATASETS)
    parser.add_argument(
        "--study-profile",
        choices=sorted(VALID_STUDY_PROFILES),
        default="thesis",
        help="quick=rank-2 reference; quick4=rank-4 hybrid; recommended=rank/policy candidates; thesis=full controlled ablations",
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=10,
        help="number of fixed sites to activate in dataset/client order",
    )
    parser.add_argument(
        "--include-ldm-clients",
        action="store_true",
        help=(
            "allow up to four additional LDM B06/B13/B14/B15 virtual clients "
            "after the prepared fixed sites"
        ),
    )
    parser.add_argument(
        "--ldm-split-strategy",
        choices=sorted(VALID_LDM_SPLIT_STRATEGIES),
        default="fixed_prefix_v1",
        help=(
            "fixed_prefix_v1 preserves the negative 2,500-prefix screen; "
            "broad_healthy_v2 samples healthy training through 80% of the "
            "pre-transition chronology and reserves the final 20% plus all "
            "unhealthy acquisitions for server testing; "
            "known_healthy_envelope_v3 also spreads calibration across the "
            "known-healthy operating envelope; stratified_healthy_v4 makes "
            "disjoint deterministic healthy train/calibration/test subsets "
            "across the full published healthy envelope; "
            "stratified_healthy_v4r1 corrects its calibration buffer to keep "
            "evenly spaced representatives instead of only the chronological tail; "
            "stratified_healthy_v4r2 replaces modulo allocation with a fixed-seed "
            "shuffle to prevent periodic operating-cycle aliasing; "
            "stratified_healthy_v5_condition_aware uses a calibration-only "
            "90th-percentile 3-of-5 alarm without LDM CUSUM or trend latching"
        ),
    )
    parser.add_argument("--calibration-windows", type=int, default=100)
    parser.add_argument(
        "--deployment-calibration-windows",
        type=int,
        default=200,
        help="minimum chronological healthy prefix excluded from held-out metrics",
    )
    parser.add_argument(
        "--deployment-calibration-fraction",
        type=float,
        default=0.20,
        help="minimum fraction of each held-out chronology used for commissioning",
    )
    parser.add_argument("--operation-block", type=int, default=30)
    parser.add_argument("--replay-size", type=int, default=128)
    parser.add_argument("--local-lr", type=float, default=0.005)
    parser.add_argument("--fedprox-mu", type=float, default=0.001)
    parser.add_argument("--personal-prox-mu", type=float, default=0.01)
    parser.add_argument("--personal-rate", type=float, default=0.10)
    parser.add_argument("--server-learning-rate", type=float, default=0.005)
    parser.add_argument("--ae-weight", type=float, default=0.75)
    parser.add_argument("--mahalanobis-weight", type=float, default=0.40)
    parser.add_argument(
        "--run-to-failure-mahalanobis-weight", type=float, default=0.70
    )
    parser.add_argument("--covariance-shrinkage", type=float, default=0.05)
    parser.add_argument(
        "--no-prognostic-latch",
        action="store_true",
        help="disable the irreversible IMS/FEMTO causal degradation alarm",
    )
    parser.add_argument("--prognostic-window", type=int, default=20)
    parser.add_argument("--prognostic-history", type=int, default=15)
    parser.add_argument("--prognostic-required", type=int, default=10)
    parser.add_argument("--prognostic-sigma", type=float, default=6.0)
    parser.add_argument(
        "--prognostic-score-buffer-size", type=int, default=256
    )
    parser.add_argument(
        "--prognostic-covariance-shrinkage", type=float, default=0.10
    )
    parser.add_argument(
        "--no-firmware-compression",
        action="store_true",
        help="disable Top-K/int8 communication for a float32 reference",
    )
    parser.add_argument(
        "--no-error-feedback",
        action="store_true",
        help="disable the client residual that returns dropped Top-K coordinates in later rounds",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore compatible completed run checkpoints and recompute them",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "thesis_realtime_federated_v2",
    )
    args = parser.parse_args()

    if args.num_devices < 2:
        parser.error("--num-devices must be at least 2")
    if args.ldm_split_strategy != "fixed_prefix_v1" and not args.include_ldm_clients:
        parser.error("--ldm-split-strategy requires --include-ldm-clients")
    available_sites = len(args.datasets) * len(CLIENTS)
    if args.include_ldm_clients:
        available_sites += len(LDM_MACHINES)
    if args.num_devices > available_sites:
        parser.error(
            f"--num-devices cannot exceed {available_sites} for the selected "
            "datasets because every device must own one unique fixed site"
        )
    if args.calibration_windows < 20:
        parser.error("--calibration-windows must be at least 20")
    if args.deployment_calibration_windows < 20:
        parser.error("--deployment-calibration-windows must be at least 20")
    if not 0.0 < args.deployment_calibration_fraction < 0.5:
        parser.error("--deployment-calibration-fraction must be between 0 and 0.5")
    if args.operation_block < 1:
        parser.error("--operation-block must be positive")
    if args.replay_size < 32 or args.replay_size % 2:
        parser.error("--replay-size must be an even number of at least 32")
    if args.operation_block > args.replay_size // 2:
        parser.error(
            "--operation-block cannot exceed half of --replay-size; otherwise "
            "new train-all windows could leave replay before their local update"
        )
    if not 0.0 <= args.personal_rate <= 1.0:
        parser.error("--personal-rate must be between 0 and 1")
    if args.fedprox_mu < 0.0 or args.personal_prox_mu < 0.0:
        parser.error("FedProx coefficients must be non-negative")
    if args.server_learning_rate <= 0.0:
        parser.error("--server-learning-rate must be positive")
    if not 0.0 <= args.ae_weight <= 1.0:
        parser.error("--ae-weight must be between 0 and 1")
    if not 0.0 <= args.mahalanobis_weight <= 1.0:
        parser.error("--mahalanobis-weight must be between 0 and 1")
    if not 0.0 <= args.run_to_failure_mahalanobis_weight <= 1.0:
        parser.error("--run-to-failure-mahalanobis-weight must be between 0 and 1")
    if not 0.0 < args.covariance_shrinkage <= 1.0:
        parser.error("--covariance-shrinkage must be in (0, 1]")
    if args.prognostic_window < 2:
        parser.error("--prognostic-window must be at least 2")
    if args.prognostic_history < 1:
        parser.error("--prognostic-history must be positive")
    if not 1 <= args.prognostic_required <= args.prognostic_history:
        parser.error(
            "--prognostic-required must be between 1 and --prognostic-history"
        )
    if args.prognostic_sigma <= 0.0:
        parser.error("--prognostic-sigma must be positive")
    if args.prognostic_score_buffer_size < 32:
        parser.error("--prognostic-score-buffer-size must be at least 32")
    if not 0.0 < args.prognostic_covariance_shrinkage <= 1.0:
        parser.error("--prognostic-covariance-shrinkage must be in (0, 1]")

    config = RealtimeConfig(
        datasets=tuple(args.datasets),
        seeds=tuple(args.seeds),
        study_profile=args.study_profile,
        num_devices=args.num_devices,
        include_ldm_clients=args.include_ldm_clients,
        ldm_split_strategy=args.ldm_split_strategy,
        calibration_windows=args.calibration_windows,
        deployment_calibration_windows=args.deployment_calibration_windows,
        deployment_calibration_fraction=args.deployment_calibration_fraction,
        operation_block=args.operation_block,
        anchor_replay_size=args.replay_size // 2,
        recent_replay_size=args.replay_size // 2,
        minimum_replay_size=max(16, args.replay_size // 4),
        local_lr=args.local_lr,
        fedprox_mu=args.fedprox_mu,
        personal_prox_mu=args.personal_prox_mu,
        personal_rate=args.personal_rate,
        server_learning_rate=args.server_learning_rate,
        ae_fusion_weight=args.ae_weight,
        mahalanobis_fusion_weight=args.mahalanobis_weight,
        run_to_failure_mahalanobis_weight=(
            args.run_to_failure_mahalanobis_weight
        ),
        covariance_shrinkage=args.covariance_shrinkage,
        prognostic_latch=not args.no_prognostic_latch,
        prognostic_window=args.prognostic_window,
        prognostic_history=args.prognostic_history,
        prognostic_required=args.prognostic_required,
        prognostic_sigma=args.prognostic_sigma,
        prognostic_score_buffer_size=args.prognostic_score_buffer_size,
        prognostic_covariance_shrinkage=(
            args.prognostic_covariance_shrinkage
        ),
        firmware_compression=not args.no_firmware_compression,
        error_feedback=not args.no_error_feedback,
    )
    variants = study_variants(config.study_profile)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building one fixed dataset/client site per LilyGO and server holdout...")
    shards, allocation_audit, leakage_audit = load_device_study(config)
    write_csv(output_dir / "full_data_allocation_audit.csv", allocation_audit)
    write_csv(output_dir / "full_leakage_audit.csv", leakage_audit)
    protocol_audit, protocol_failures = build_protocol_audit(shards, config)
    write_csv(output_dir / "full_protocol_audit.csv", protocol_audit)
    if protocol_failures:
        raise ValueError(
            "experimental protocol audit failed: " + "; ".join(protocol_failures)
        )

    base_path = output_dir / "full_federated_base.npz"
    base_signature = _study_signature(config)
    commissioning_rounds_path = output_dir / "dense_commissioning_rounds.csv"
    commissioning_audit_path = output_dir / "dense_commissioning_audit.csv"
    reuse_base = False
    if base_path.exists() and not args.no_resume:
        try:
            dims, candidate_weights, base_meta = load_base_npz(base_path)
        except (OSError, ValueError, KeyError, TypeError):
            print("Dense commissioning checkpoint is unreadable; rebuilding it.")
        else:
            if (
                dims == LARGER_DIMS
                and base_meta.get("base_signature") == base_signature
            ):
                candidate_records = read_csv(commissioning_rounds_path)
                candidate_audit = read_csv(commissioning_audit_path)
                audit_complete = bool(candidate_audit) and all(
                    _as_bool(row.get("complete", False))
                    for row in candidate_audit
                )
                if candidate_records and audit_complete:
                    base_weights = candidate_weights
                    commissioning_records = candidate_records
                    commissioning_audit = candidate_audit
                    reuse_base = True
                    print("Reusing compatible dense commissioning checkpoint.")
                else:
                    print(
                        "Dense checkpoint audit is missing or incomplete; "
                        "rebuilding it."
                    )
    if not reuse_base:
        print("Federated dense commissioning over every device train window...")
        base_weights, commissioning_records, commissioning_audit = federated_commissioning(
            shards, config
        )
        save_base_npz(
            base_path,
            LARGER_DIMS,
            base_weights,
            {
                "mode": "complete device commissioning before frozen server test",
                "base_signature": base_signature,
                "config": asdict(config),
            },
        )
        write_csv(commissioning_rounds_path, commissioning_records)
        write_csv(commissioning_audit_path, commissioning_audit)

    all_server_metrics: list[dict[str, object]] = []
    all_prequential_metrics: list[dict[str, object]] = []
    all_federation_records: list[dict[str, object]] = list(commissioning_records)
    all_training_audit: list[dict[str, object]] = list(commissioning_audit)

    for seed in config.seeds:
        print(f"\n=== Seed {seed}: controlled thesis variants ===")
        for variant in variants:
            run_dir = output_dir / "runs" / variant.name / f"seed_{seed}"
            signature = _run_signature(config, variant, seed)
            completed = None if args.no_resume else load_completed_run(
                run_dir, signature
            )
            if completed is not None:
                print(f"--- {variant.name}: checkpoint complete, skipping")
                all_federation_records.extend(completed["federation_records"])
                all_training_audit.extend(completed["training_audit"])
                all_prequential_metrics.extend(completed["prequential_metrics"])
                all_server_metrics.extend(completed["server_metrics"])
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoint.json").unlink(missing_ok=True)
            print(f"--- {variant.name}: consume complete device operation pool ---")
            started = time.perf_counter()
            (
                device_records,
                federation,
                audit,
                runtimes,
                master_adapter,
                broadcast_adapter,
                server_state,
            ) = run_variant(
                shards, base_weights, config, seed, variant
            )
            server_records = evaluate_final_models(
                shards,
                runtimes,
                broadcast_adapter,
                config,
                seed,
                variant,
            )
            prequential_metrics = summarize_prequential_records(device_records)
            server_metrics = summarize_server_records(server_records)
            export_lilygo_states(
                run_dir,
                runtimes,
                base_weights,
                master_adapter,
                broadcast_adapter,
                server_state,
                config,
                variant,
                seed,
            )
            completed = save_completed_run(
                run_dir,
                signature,
                device_records=device_records,
                server_records=server_records,
                federation_records=federation,
                training_audit=audit,
                prequential_metrics=prequential_metrics,
                server_metrics=server_metrics,
            )
            all_federation_records.extend(completed["federation_records"])
            all_training_audit.extend(completed["training_audit"])
            all_prequential_metrics.extend(completed["prequential_metrics"])
            all_server_metrics.extend(completed["server_metrics"])
            print(
                f"{variant.name} complete in {time.perf_counter() - started:.1f}s"
            )

        baseline_dir = output_dir / "runs" / "baseline" / f"seed_{seed}"
        baseline_signature = _run_signature(config, None, seed)
        baseline = None if args.no_resume else load_completed_run(
            baseline_dir, baseline_signature
        )
        if baseline is None:
            baseline_dir.mkdir(parents=True, exist_ok=True)
            (baseline_dir / "checkpoint.json").unlink(missing_ok=True)
            print(f"=== Seed {seed}: deployment baselines ===")
            baseline_records = evaluate_baselines(shards, config, seed)
            baseline_metrics = summarize_server_records(baseline_records)
            baseline = save_completed_run(
                baseline_dir,
                baseline_signature,
                device_records=[],
                server_records=baseline_records,
                federation_records=[],
                training_audit=[],
                prequential_metrics=[],
                server_metrics=baseline_metrics,
            )
        else:
            print(f"=== Seed {seed}: baseline checkpoint complete, skipping ===")
        all_server_metrics.extend(baseline["server_metrics"])

    server_summary = aggregate_seed_metrics(all_server_metrics)
    paired_comparisons = build_paired_comparisons(all_server_metrics)

    write_csv(output_dir / "full_federation_rounds.csv", all_federation_records)
    write_csv(output_dir / "full_training_audit.csv", all_training_audit)
    write_csv(output_dir / "full_prequential_metrics.csv", all_prequential_metrics)
    write_csv(output_dir / "full_server_per_seed_metrics.csv", all_server_metrics)
    write_csv(output_dir / "full_server_summary.csv", server_summary)
    write_csv(output_dir / "full_paired_comparisons.csv", paired_comparisons)
    payload = {
        "study": "Fixed-site SD-card TinyFL followed by frozen server holdout",
        "provenance": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dense_base_signature": base_signature,
            "ldm_extension_protocol": (
                str(_ldm_protocol_path(config.ldm_split_strategy).resolve())
                if config.include_ldm_clients
                else None
            ),
            "ldm_extension_protocol_sha256": (
                _sha256_file(_ldm_protocol_path(config.ldm_split_strategy))
                if config.include_ldm_clients
                else None
            ),
            "ldm_transition_sha256": (
                _sha256_file(LDM_TRANSITION_PATH)
                if config.include_ldm_clients
                else None
            ),
        },
        "config": asdict(config),
        "study_variants": [asdict(variant) for variant in variants],
        "fixed_site_assignments": [
            {
                "device_id": device_id,
                "dataset": stream.dataset,
                "logical_client": stream.logical_client,
                "client_name": _client_display_name(
                    stream.dataset, stream.logical_client
                ),
            }
            for device_id, shard in sorted(shards.items())
            for stream in shard.streams.values()
        ],
        "server_summary": server_summary,
        "paired_comparisons": paired_comparisons,
        "label_usage": "post-hoc protocol audit and metrics only; never model or alarm decisions",
        "device_layout": "one immutable dataset/client domain per LilyGO",
        "device_pool": "train + calibration + val for active fixed sites",
        "server_holdout": "test only",
        "limitations": [
            "Prepared inputs are 12-feature windows, not a live accelerometer stream.",
            "Source allocation simulates SD cards but cannot measure physical radio loss, energy, or timing.",
            "Each public dataset/client domain is treated as one fixed physical site proxy.",
            "Adaptive server optimization is simulated on the PC; only compressed FFA-LoRA B adapters are deployed to devices.",
            *(
                [
                    "B06/B13/B14/B15 are post-outcome participating-client temporal development, not independent external validation.",
                    "The LDM clients use a disclosed 12-coordinate projection from locked 100-feature V12 acquisition vectors.",
                    "Devices 11-14 are virtual simulation clients; only ten physical LilyGO boards are available.",
                ]
                if config.include_ldm_clients
                else []
            ),
        ],
    }
    _write_json_atomic(output_dir / "full_results.json", payload)
    write_report(
        output_dir / "FULL_FEDERATED_REPORT.md",
        config,
        server_summary,
        all_training_audit,
        paired_comparisons,
    )
    print("\nFull simulation complete")
    print(f"  Report: {output_dir / 'FULL_FEDERATED_REPORT.md'}")


if __name__ == "__main__":
    main()
