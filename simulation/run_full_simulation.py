"""Run the complete TinyFedLoRA simulation study across all prepared datasets.

This file intentionally reads only the prepared ``*_bin`` folders. It never reads
or downloads the raw CWRU, XJTU-SY, IMS, or PRONOSTIA archives.

The study answers three levels of questions:

1. Data difficulty: do simple statistical thresholds already solve the task?
2. Learning value: do training, federation, and a larger model add value?
3. Deployment value: does the lightweight firmware-equivalent LoRA path retain
   enough anomaly sensitivity after Top-K sparsification and int8 quantization?

It also tests the cross-dataset question:

    Can one shared autoencoder learn normal bearing behaviour across four
    different datasets without losing anomaly sensitivity?

Run this script from any directory. By default it uses five deterministic data
ordering seeds and writes JSON, CSV, and Markdown reports beside this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_DIR = SCRIPT_DIR.parent
PYTHON_TOOLS_DIR = RELEASE_DIR / "python_tools"
if str(PYTHON_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_TOOLS_DIR))

from compare_architectures import GeneralLoRAAutoencoder  # noqa: E402
from cwru_eval import classification_metrics, load_domain  # noqa: E402
from tiny_ae import (  # noqa: E402
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
    ADAM_WEIGHT_DECAY,
    BASE_SEED,
    _lcg_values,
    dequantize_dynamic,
    quantize_dynamic,
)


DATASETS = ("cwru", "xjtu", "ims", "pronostia")
CLIENTS = ("client_a", "client_b")
CLIENT_DISPLAY_NAMES = {
    ("cwru", "client_a"): "DE",
    ("cwru", "client_b"): "FE",
    ("xjtu", "client_a"): "Bearing1_1",
    ("xjtu", "client_b"): "Bearing1_2",
    ("ims", "client_a"): "IMS_Odd",
    ("ims", "client_b"): "IMS_Even",
    ("pronostia", "client_a"): "FEMTO_1_1",
    ("pronostia", "client_b"): "FEMTO_1_2",
}

SMALL_DIMS = (12, 16, 8, 4, 8, 16, 12)
BIGGER_DIMS = (12, 24, 12, 4, 12, 24, 12)
LORA_RANKS = (1, 2, 4)

# Backward-compatible aliases used by a few comparison helpers below.
LIGHT_DIMS = SMALL_DIMS
LIGHT_RANK = 2
LARGE_DIMS = BIGGER_DIMS
LARGE_RANK = 4

FEATURE_BASELINES = {
    "baseline_rms": 0,
    "baseline_peak_to_peak": 1,
    "baseline_kurtosis": 2,
    "baseline_stddev": 9,
}

RESEARCH_QUESTIONS = (
    "Are anomalies already clearly separable using simple statistical features?",
    "Does a trained autoencoder outperform an untrained autoencoder and simple baselines?",
    "Does federated training improve anomaly detection compared with local-only training?",
    "How does federated training compare with centralized training under an equal training budget?",
    "Can one shared autoencoder learn normal bearing behaviour across four different datasets without losing anomaly sensitivity?",
    "Does a shared multi-dataset model perform better or worse than dataset-specific models?",
    "Which small/bigger architecture and LoRA rank (1, 2, or 4) gives the best accuracy/size tradeoff?",
    "How much performance is lost through Top-K sparsification and signed-int8 quantization?",
    "Does federation particularly help weaker clients or heterogeneous operating conditions?",
    "Are results consistent across multiple data-ordering seeds, datasets, and clients?",
    "Which datasets and clients remain difficult?",
    "Is the lightweight federated model ready to justify LilyGO hardware deployment?",
)


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    dims: tuple[int, ...]
    rank: int

    @property
    def slug(self) -> str:
        return f"{self.architecture}_r{self.rank}"

    @property
    def lora_parameters(self) -> int:
        return sum(out_dim * self.rank for out_dim in self.dims[1:])

    @property
    def dense_parameters(self) -> int:
        return sum(
            in_dim * out_dim for in_dim, out_dim in zip(self.dims[:-1], self.dims[1:])
        )

    @property
    def lora_a_parameters(self) -> int:
        return sum(in_dim * self.rank for in_dim in self.dims[:-1])

    @property
    def estimated_training_state_bytes(self) -> int:
        # Frozen W + frozen A + trainable B + Adam m/v for B. Runtime activation
        # buffers are small for both architectures and are not included here.
        floats = (
            self.dense_parameters
            + self.lora_a_parameters
            + 3 * self.lora_parameters
        )
        return 4 * floats

    @property
    def index_bytes(self) -> int:
        return 1 if self.lora_parameters <= 256 else 2

    @property
    def update_packet_bytes(self) -> int:
        # Current 87-byte packet has 32 one-byte indices. A model above 256
        # parameters needs uint16 indices, adding another 32 bytes.
        return 87 + 32 * (self.index_bytes - 1)

    @property
    def global_packet_bytes(self) -> int:
        # Fixed metadata/CRC is 16 bytes; every LoRA B value is signed int8.
        return 16 + self.lora_parameters

    @property
    def fits_single_lora_packet(self) -> bool:
        return self.update_packet_bytes <= 255 and self.global_packet_bytes <= 255

    @property
    def fits_lilygo_training_memory(self) -> bool:
        # Conservative guard; the ESP32 has far more than this available for the
        # complete sketch, and the largest tested state is only about 11 KiB.
        return self.estimated_training_state_bytes < 64 * 1024


MODEL_SPECS = tuple(
    ModelSpec(architecture, dims, rank)
    for architecture, dims in (("small", SMALL_DIMS), ("bigger", BIGGER_DIMS))
    for rank in LORA_RANKS
)


@dataclass(frozen=True)
class ExperimentConfig:
    datasets: tuple[str, ...] = DATASETS
    seeds: tuple[int, ...] = (11, 29, 47, 71, 97)
    rounds: int = 3
    samples_per_round: int = 30
    local_epochs: int = 8
    learning_rate: float = 0.02
    sparsity_k: int = 32
    fedadam_eta: float = 0.01
    fedadam_beta1: float = 0.9
    fedadam_beta2: float = 0.99
    fedadam_tau: float = 1e-3
    update_packet_bytes: int = 87
    global_packet_bytes: int = 144

    @property
    def training_samples_per_client(self) -> int:
        return self.rounds * self.samples_per_round


class DenseAutoencoder:
    """Full-weight baseline with the same bias-free architecture as TinyFedLoRA.

    This is deliberately a comparison baseline. The deployable methods train only
    LoRA B matrices; this model trains every dense W matrix.
    """

    def __init__(self, dims: tuple[int, ...]) -> None:
        if len(dims) != 7 or dims[0] != dims[-1]:
            raise ValueError("the dense baseline expects a symmetric six-layer autoencoder")
        self.dims = dims
        self.num_layers = len(dims) - 1
        scales = (0.5, 0.5, 0.75, 1.0, 0.75, 0.5)
        state = BASE_SEED
        self.W: list[np.ndarray] = []
        for in_dim, out_dim, scale in zip(dims[:-1], dims[1:], scales):
            values, state = _lcg_values(state, out_dim * in_dim, scale)
            self.W.append(values.reshape(out_dim, in_dim).astype(np.float32))
        self.m = [np.zeros_like(matrix) for matrix in self.W]
        self.v = [np.zeros_like(matrix) for matrix in self.W]
        self.t = 0

    def forward(self, sample: np.ndarray, with_cache: bool = False):
        activations = [np.asarray(sample, dtype=np.float32)]
        preactivations: list[np.ndarray] = []
        current = activations[0]
        for layer, weights in enumerate(self.W):
            z = weights @ current
            current = z if layer == self.num_layers - 1 else np.maximum(z, 0.0)
            preactivations.append(z)
            activations.append(current)
        loss = float(np.mean((current - activations[0]) ** 2))
        return (loss, activations, preactivations) if with_cache else loss

    def train_step(self, sample: np.ndarray, learning_rate: float) -> float:
        loss, activations, preactivations = self.forward(sample, with_cache=True)
        self.t += 1
        delta = np.float32(2.0 / self.dims[0]) * (activations[-1] - activations[0])
        gradients: list[np.ndarray | None] = [None] * self.num_layers
        for layer in range(self.num_layers - 1, -1, -1):
            gradients[layer] = np.outer(delta, activations[layer]).astype(np.float32)
            if layer > 0:
                delta = self.W[layer].T @ delta
                delta = delta * (preactivations[layer - 1] > 0.0)

        for layer, gradient in enumerate(gradients):
            assert gradient is not None
            clipped = np.clip(gradient, -1.0, 1.0).astype(np.float32)
            self.m[layer] = ADAM_BETA1 * self.m[layer] + (1.0 - ADAM_BETA1) * clipped
            self.v[layer] = ADAM_BETA2 * self.v[layer] + (1.0 - ADAM_BETA2) * clipped * clipped
            m_hat = self.m[layer] / (1.0 - float(ADAM_BETA1) ** self.t)
            v_hat = self.v[layer] / (1.0 - float(ADAM_BETA2) ** self.t)
            update = m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)
            self.W[layer] -= np.float32(learning_rate) * (
                update + ADAM_WEIGHT_DECAY * self.W[layer]
            )
        return loss

    def train(self, samples: np.ndarray, epochs: int, learning_rate: float) -> list[float]:
        losses: list[float] = []
        for _ in range(epochs):
            epoch_losses = [self.train_step(sample, learning_rate) for sample in samples]
            losses.append(float(np.mean(epoch_losses)))
        return losses

    def losses(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray([self.forward(sample) for sample in samples], dtype=np.float32)


def stable_seed(seed: int, name: str) -> int:
    """Derive a repeatable NumPy seed without Python's randomized hash()."""
    return (int(seed) + zlib.crc32(name.encode("utf-8"))) & 0xFFFFFFFF


def lora_parameter_count(dims: tuple[int, ...], rank: int) -> int:
    return sum(out_dim * rank for out_dim in dims[1:])


def dense_parameter_count(dims: tuple[int, ...]) -> int:
    return sum(in_dim * out_dim for in_dim, out_dim in zip(dims[:-1], dims[1:]))


def client_key(dataset: str, client: str) -> str:
    return f"{dataset}:{client}"


def validate_domain(dataset: str, client: str, domain: dict[str, object]) -> None:
    """Fail early on leakage, malformed features, or unusable evaluation splits."""
    split_ids: dict[str, set[str]] = {}
    for split in ("train", "calibration", "val", "test"):
        values = np.asarray(domain[split], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 12 or values.shape[0] == 0:
            raise ValueError(f"{dataset}/{client} {split} must be a non-empty N x 12 matrix")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{dataset}/{client} {split} contains non-finite normalized features")
        rows = domain[f"{split}_rows"]
        if len(rows) != len(values):
            raise ValueError(f"{dataset}/{client} {split} manifest/feature length mismatch")
        split_ids[split] = {str(row["window_id"]) for row in rows}
        if len(split_ids[split]) != len(rows):
            raise ValueError(f"{dataset}/{client} {split} contains duplicate window IDs")

    names = tuple(split_ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise ValueError(
                    f"{dataset}/{client} leakage: {left} and {right} share {len(overlap)} windows"
                )

    normal_sources_by_split = {
        split: {
            str(row["source_file"])
            for row in domain[f"{split}_rows"]
            if int(row["label"]) == 0
        }
        for split in ("train", "calibration", "val", "test")
    }

    # Training recordings must be independent from every evaluation split, and
    # the final normal test recording must be independent from calibration and
    # validation. CWRU intentionally uses different, guard-separated windows of
    # 99.mat for calibration and validation, so that one source-file reuse is
    # allowed only when the physical sample intervals do not overlap.
    forbidden_source_pairs = (
        ("train", "calibration"),
        ("train", "val"),
        ("train", "test"),
        ("calibration", "test"),
        ("val", "test"),
    )
    for left, right in forbidden_source_pairs:
        overlap = normal_sources_by_split[left] & normal_sources_by_split[right]
        if overlap:
            raise ValueError(
                f"{dataset}/{client} normal-source leakage between {left} and {right}: "
                f"{sorted(overlap)[:3]}"
            )

    shared_cal_val_sources = (
        normal_sources_by_split["calibration"] & normal_sources_by_split["val"]
    )
    for source in shared_cal_val_sources:
        calibration_starts = sorted(
            int(row["start_sample"])
            for row in domain["calibration_rows"]
            if int(row["label"]) == 0 and str(row["source_file"]) == source
        )
        validation_starts = sorted(
            int(row["start_sample"])
            for row in domain["val_rows"]
            if int(row["label"]) == 0 and str(row["source_file"]) == source
        )
        left_index = 0
        right_index = 0
        while left_index < len(calibration_starts) and right_index < len(validation_starts):
            calibration_start = calibration_starts[left_index]
            validation_start = validation_starts[right_index]
            if calibration_start + 512 <= validation_start:
                left_index += 1
            elif validation_start + 512 <= calibration_start:
                right_index += 1
            else:
                raise ValueError(
                    f"{dataset}/{client} calibration/validation windows overlap in {source}: "
                    f"starts {calibration_start} and {validation_start}"
                )

    for label_name in ("val_labels", "labels"):
        labels = np.asarray(domain[label_name], dtype=np.int8)
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError(f"{dataset}/{client} {label_name} must contain normal and fault samples")


def load_all_domains(config: ExperimentConfig) -> dict[str, dict[str, dict[str, object]]]:
    domains: dict[str, dict[str, dict[str, object]]] = {}
    for dataset in config.datasets:
        bin_dir = PYTHON_TOOLS_DIR / f"{dataset}_bin"
        if not bin_dir.is_dir():
            raise FileNotFoundError(f"missing prepared directory: {bin_dir}")
        domains[dataset] = {}
        for client in CLIENTS:
            domain = load_domain(client, bin_dir)
            validate_domain(dataset, client, domain)
            if len(domain["train"]) < config.training_samples_per_client:
                raise ValueError(
                    f"{dataset}/{client} needs at least {config.training_samples_per_client} "
                    f"training windows; found {len(domain['train'])}"
                )
            domains[dataset][client] = domain
    return domains


def select_training_windows(
    domains: dict[str, dict[str, dict[str, object]]], config: ExperimentConfig, seed: int
) -> dict[str, np.ndarray]:
    """Select the same balanced per-client training budget for every method."""
    selected: dict[str, np.ndarray] = {}
    count = config.training_samples_per_client
    for dataset in config.datasets:
        for client in CLIENTS:
            key = client_key(dataset, client)
            values = np.asarray(domains[dataset][client]["train"], dtype=np.float32)
            rng = np.random.default_rng(stable_seed(seed, key))
            indices = rng.permutation(len(values))[:count]
            selected[key] = values[indices]
    return selected


def ordered_stack(arrays: Iterable[np.ndarray], seed: int, name: str) -> np.ndarray:
    values = np.vstack(tuple(arrays)).astype(np.float32)
    rng = np.random.default_rng(stable_seed(seed, name))
    return values[rng.permutation(len(values))]


def threshold_value(scores: np.ndarray, method: str) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("threshold calibration requires at least two finite scores")
    if method == "3-sigma":
        return float(values.mean() + 3.0 * values.std(ddof=1))
    if method == "99th":
        return float(np.percentile(values, 99.0))
    if method == "99.5th":
        return float(np.percentile(values, 99.5))
    if method == "median_mad":
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median + 3.0 * 1.4826 * mad
    raise ValueError(f"unknown threshold method: {method}")


def choose_threshold(
    calibration_scores: np.ndarray, validation_scores: np.ndarray, validation_labels: np.ndarray
) -> tuple[str, float, dict[str, float | int]]:
    """Choose a threshold on validation only, then freeze it before final test."""
    best: tuple[tuple[float, float, float], str, float, dict[str, float | int]] | None = None
    for method in ("3-sigma", "99th", "99.5th", "median_mad"):
        threshold = threshold_value(calibration_scores, method)
        metrics = classification_metrics(validation_labels, validation_scores, threshold)
        objective = (float(metrics["f1"]), -float(metrics["fpr"]), float(metrics["recall"]))
        if best is None or objective > best[0]:
            best = (objective, method, threshold, metrics)
    assert best is not None
    return best[1], best[2], best[3]


def evaluate_scores(
    *,
    dataset: str,
    client: str,
    seed: int,
    method: str,
    family: str,
    training_scope: str,
    domain: dict[str, object],
    scorer: Callable[[np.ndarray], np.ndarray],
    trainable_parameters: int,
    total_model_parameters: int,
    rank: int | None,
    deployable: bool,
    training_seconds: float,
    uplink_bytes: int,
    downlink_bytes: int,
    architecture: str = "",
    model_dims: str = "",
    fits_lilygo_training_memory: bool = True,
    fits_single_lora_packet: bool = True,
    global_packet_bytes: int = 0,
    update_packet_bytes: int = 0,
    notes: str = "",
) -> dict[str, object]:
    calibration_scores = np.asarray(scorer(np.asarray(domain["calibration"])), dtype=np.float32)
    validation_scores = np.asarray(scorer(np.asarray(domain["val"])), dtype=np.float32)
    test_scores = np.asarray(scorer(np.asarray(domain["test"])), dtype=np.float32)
    threshold_method, threshold, validation_metrics = choose_threshold(
        calibration_scores,
        validation_scores,
        np.asarray(domain["val_labels"], dtype=np.int8),
    )
    labels = np.asarray(domain["labels"], dtype=np.int8)
    metrics = classification_metrics(labels, test_scores, threshold)
    normal = test_scores[labels == 0]
    fault = test_scores[labels == 1]
    return {
        "dataset": dataset,
        "client": client,
        "client_name": CLIENT_DISPLAY_NAMES[(dataset, client)],
        "seed": seed,
        "method": method,
        "family": family,
        "training_scope": training_scope,
        "architecture": architecture,
        "model_dims": model_dims,
        "threshold_method": threshold_method,
        "threshold": float(threshold),
        "validation_f1": float(validation_metrics["f1"]),
        "validation_fpr": float(validation_metrics["fpr"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tn": int(metrics["tn"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "fpr": float(metrics["fpr"]),
        "roc_auc": float(metrics["roc_auc"]),
        "pr_auc": float(metrics["pr_auc"]),
        "normal_mean_score": float(np.mean(normal)),
        "fault_mean_score": float(np.mean(fault)),
        "score_separation": float(np.mean(fault) / max(float(np.mean(normal)), 1e-12)),
        "trainable_parameters": trainable_parameters,
        "total_model_parameters": total_model_parameters,
        "lora_rank": rank if rank is not None else "",
        "deployable_on_current_firmware": deployable,
        "fits_lilygo_training_memory": fits_lilygo_training_memory,
        "fits_single_lora_packet": fits_single_lora_packet,
        "requires_fragmentation": not fits_single_lora_packet,
        "global_packet_bytes": global_packet_bytes,
        "update_packet_bytes": update_packet_bytes,
        "uplink_bytes": uplink_bytes,
        "downlink_bytes": downlink_bytes,
        "training_seconds": float(training_seconds),
        "notes": notes,
    }


def train_lora_centralized(
    dims: tuple[int, ...], rank: int, samples: np.ndarray, config: ExperimentConfig
) -> tuple[GeneralLoRAAutoencoder, float]:
    model = GeneralLoRAAutoencoder(dims, rank)
    started = time.perf_counter()
    model.train(samples, epochs=config.local_epochs, learning_rate=config.learning_rate)
    return model, time.perf_counter() - started


def train_dense_centralized(
    dims: tuple[int, ...], samples: np.ndarray, config: ExperimentConfig
) -> tuple[DenseAutoencoder, float]:
    model = DenseAutoencoder(dims)
    started = time.perf_counter()
    model.train(samples, epochs=config.local_epochs, learning_rate=config.learning_rate)
    return model, time.perf_counter() - started


def firmware_sparse_reconstruction(
    global_weights: np.ndarray, local_weights: np.ndarray, sparsity_k: int
) -> np.ndarray:
    """Rebuild the local model as the server does from a Top-K int8 UPDATE.

    Stable descending magnitude order matches the firmware selection sort's tie
    behaviour. Packet ordering does not affect reconstruction, but this keeps the
    simulator faithful to the client implementation.
    """
    delta = np.asarray(local_weights - global_weights, dtype=np.float32)
    indices = np.argsort(-np.abs(delta), kind="stable")[:sparsity_k]
    scale, packed = quantize_dynamic(delta[indices])
    reconstructed = np.asarray(global_weights, dtype=np.float32).copy()
    reconstructed[indices] += dequantize_dynamic(packed, scale)
    return reconstructed


def run_fedadam(
    *,
    dims: tuple[int, ...],
    rank: int,
    client_samples: dict[str, np.ndarray],
    config: ExperimentConfig,
    firmware_compression: bool,
) -> tuple[dict[str, GeneralLoRAAutoencoder], float]:
    """Train one global LoRA model using sample-weighted FedAdam."""
    count = lora_parameter_count(dims, rank)
    global_weights = np.zeros(count, dtype=np.float32)
    adam_m = np.zeros(count, dtype=np.float32)
    adam_v = np.zeros(count, dtype=np.float32)
    clients = {key: GeneralLoRAAutoencoder(dims, rank) for key in client_samples}
    started = time.perf_counter()

    for round_index in range(config.rounds):
        reconstructed_models: list[tuple[np.ndarray, int]] = []
        start = round_index * config.samples_per_round
        end = start + config.samples_per_round
        for key, model in clients.items():
            model.set_weights(global_weights, reset_optimizer=True)
            batch = client_samples[key][start:end]
            if len(batch) != config.samples_per_round:
                raise ValueError(f"{key} has an incomplete federated batch in round {round_index}")
            model.train(batch, epochs=config.local_epochs, learning_rate=config.learning_rate)
            local_weights = model.get_weights()
            if firmware_compression:
                reconstructed = firmware_sparse_reconstruction(
                    global_weights, local_weights, config.sparsity_k
                )
            else:
                reconstructed = np.asarray(local_weights, dtype=np.float32)
            reconstructed_models.append((reconstructed, len(batch)))

        total_samples = sum(samples for _, samples in reconstructed_models)
        weighted_average = sum(
            (weights * np.float32(samples) for weights, samples in reconstructed_models),
            start=np.zeros(count, dtype=np.float32),
        ) / np.float32(total_samples)
        aggregate_delta = weighted_average - global_weights
        step = round_index + 1
        adam_m = (
            config.fedadam_beta1 * adam_m
            + (1.0 - config.fedadam_beta1) * aggregate_delta
        ).astype(np.float32)
        adam_v = (
            config.fedadam_beta2 * adam_v
            + (1.0 - config.fedadam_beta2) * aggregate_delta * aggregate_delta
        ).astype(np.float32)
        m_hat = adam_m / np.float32(1.0 - config.fedadam_beta1**step)
        v_hat = adam_v / np.float32(1.0 - config.fedadam_beta2**step)
        global_weights = global_weights + np.float32(config.fedadam_eta) * m_hat / (
            np.sqrt(v_hat) + np.float32(config.fedadam_tau)
        )
        if firmware_compression:
            scale, packed = quantize_dynamic(global_weights)
            global_weights = dequantize_dynamic(packed, scale)
        global_weights = np.asarray(global_weights, dtype=np.float32)

    elapsed = time.perf_counter() - started
    for model in clients.values():
        model.set_weights(global_weights, reset_optimizer=True)
    return clients, elapsed


def append_baselines(
    records: list[dict[str, object]],
    domains: dict[str, dict[str, dict[str, object]]],
    config: ExperimentConfig,
    seed: int,
) -> None:
    for dataset in config.datasets:
        for client in CLIENTS:
            domain = domains[dataset][client]
            for method, feature_index in FEATURE_BASELINES.items():
                records.append(
                    evaluate_scores(
                        dataset=dataset,
                        client=client,
                        seed=seed,
                        method=method,
                        family="statistical_baseline",
                        training_scope="none",
                        domain=domain,
                        scorer=lambda values, index=feature_index: np.abs(values[:, index]),
                        trainable_parameters=0,
                        total_model_parameters=0,
                        rank=None,
                        deployable=True,
                        training_seconds=0.0,
                        uplink_bytes=0,
                        downlink_bytes=0,
                    )
                )


def append_untrained(
    records: list[dict[str, object]],
    domains: dict[str, dict[str, dict[str, object]]],
    config: ExperimentConfig,
    seed: int,
) -> None:
    model = GeneralLoRAAutoencoder(LIGHT_DIMS, LIGHT_RANK)
    for dataset in config.datasets:
        for client in CLIENTS:
            records.append(
                evaluate_scores(
                    dataset=dataset,
                    client=client,
                    seed=seed,
                    method="untrained_lora_light",
                    family="autoencoder",
                    training_scope="none",
                    domain=domains[dataset][client],
                    scorer=model.losses,
                    trainable_parameters=lora_parameter_count(LIGHT_DIMS, LIGHT_RANK),
                    total_model_parameters=dense_parameter_count(LIGHT_DIMS),
                    rank=LIGHT_RANK,
                    deployable=True,
                    training_seconds=0.0,
                    uplink_bytes=0,
                    downlink_bytes=0,
                    notes="Frozen base and zero LoRA B matrices; no training.",
                )
            )


def append_dataset_specific_models(
    records: list[dict[str, object]],
    domains: dict[str, dict[str, dict[str, object]]],
    selected: dict[str, np.ndarray],
    config: ExperimentConfig,
    seed: int,
) -> None:
    light_lora_count = lora_parameter_count(LIGHT_DIMS, LIGHT_RANK)
    light_dense_count = dense_parameter_count(LIGHT_DIMS)
    large_lora_count = lora_parameter_count(LARGE_DIMS, LARGE_RANK)
    large_dense_count = dense_parameter_count(LARGE_DIMS)

    for dataset in config.datasets:
        print(f"  Training dataset-specific comparisons: {dataset.upper()}")
        keys = [client_key(dataset, client) for client in CLIENTS]

        # Local-only LoRA: each client sees exactly 90 selected healthy windows once
        # per epoch, the same number of sample presentations as federated training.
        for client, key in zip(CLIENTS, keys):
            model, elapsed = train_lora_centralized(
                LIGHT_DIMS, LIGHT_RANK, selected[key], config
            )
            records.append(
                evaluate_scores(
                    dataset=dataset,
                    client=client,
                    seed=seed,
                    method="local_lora_light",
                    family="autoencoder",
                    training_scope="single_client",
                    domain=domains[dataset][client],
                    scorer=model.losses,
                    trainable_parameters=light_lora_count,
                    total_model_parameters=light_dense_count,
                    rank=LIGHT_RANK,
                    deployable=True,
                    training_seconds=elapsed,
                    uplink_bytes=0,
                    downlink_bytes=0,
                )
            )

        combined = ordered_stack(
            (selected[key] for key in keys), seed, f"{dataset}:centralized"
        )
        central_lora, central_lora_seconds = train_lora_centralized(
            LIGHT_DIMS, LIGHT_RANK, combined, config
        )
        central_dense, central_dense_seconds = train_dense_centralized(
            LIGHT_DIMS, combined, config
        )

        fed_float, fed_float_seconds = run_fedadam(
            dims=LIGHT_DIMS,
            rank=LIGHT_RANK,
            client_samples={key: selected[key] for key in keys},
            config=config,
            firmware_compression=False,
        )
        fed_firmware, fed_firmware_seconds = run_fedadam(
            dims=LIGHT_DIMS,
            rank=LIGHT_RANK,
            client_samples={key: selected[key] for key in keys},
            config=config,
            firmware_compression=True,
        )
        fed_large, fed_large_seconds = run_fedadam(
            dims=LARGE_DIMS,
            rank=LARGE_RANK,
            client_samples={key: selected[key] for key in keys},
            config=config,
            firmware_compression=False,
        )

        for client, key in zip(CLIENTS, keys):
            common = {
                "dataset": dataset,
                "client": client,
                "seed": seed,
                "domain": domains[dataset][client],
            }
            records.append(
                evaluate_scores(
                    **common,
                    method="centralized_lora_light",
                    family="autoencoder",
                    training_scope="two_clients_same_dataset",
                    scorer=central_lora.losses,
                    trainable_parameters=light_lora_count,
                    total_model_parameters=light_dense_count,
                    rank=LIGHT_RANK,
                    deployable=False,
                    training_seconds=central_lora_seconds,
                    uplink_bytes=0,
                    downlink_bytes=0,
                    notes="Centralized research upper bound; no radio communication.",
                )
            )
            records.append(
                evaluate_scores(
                    **common,
                    method="centralized_full_light",
                    family="full_weight_autoencoder",
                    training_scope="two_clients_same_dataset",
                    scorer=central_dense.losses,
                    trainable_parameters=light_dense_count,
                    total_model_parameters=light_dense_count,
                    rank=None,
                    deployable=False,
                    training_seconds=central_dense_seconds,
                    uplink_bytes=0,
                    downlink_bytes=0,
                    notes="All dense W matrices trained; comparison baseline only.",
                )
            )
            records.append(
                evaluate_scores(
                    **common,
                    method="federated_float_fedadam_lora_light",
                    family="autoencoder",
                    training_scope="two_clients_same_dataset",
                    scorer=fed_float[key].losses,
                    trainable_parameters=light_lora_count,
                    total_model_parameters=light_dense_count,
                    rank=LIGHT_RANK,
                    deployable=False,
                    training_seconds=fed_float_seconds,
                    uplink_bytes=0,
                    downlink_bytes=0,
                    notes="FedAdam without Top-K or int8; isolates compression cost.",
                )
            )
            records.append(
                evaluate_scores(
                    **common,
                    method="federated_firmware_fedadam_lora_light",
                    family="autoencoder",
                    training_scope="two_clients_same_dataset",
                    scorer=fed_firmware[key].losses,
                    trainable_parameters=light_lora_count,
                    total_model_parameters=light_dense_count,
                    rank=LIGHT_RANK,
                    deployable=True,
                    training_seconds=fed_firmware_seconds,
                    uplink_bytes=config.rounds * config.update_packet_bytes,
                    downlink_bytes=config.rounds * config.global_packet_bytes,
                    notes="FedAdam + Top-K 32 + signed-int8 UPDATE and GLOBAL quantization.",
                )
            )
            records.append(
                evaluate_scores(
                    **common,
                    method="dataset_federated_float_fedadam_lora_large",
                    family="autoencoder",
                    training_scope="two_clients_same_dataset",
                    scorer=fed_large[key].losses,
                    trainable_parameters=large_lora_count,
                    total_model_parameters=large_dense_count,
                    rank=LARGE_RANK,
                    deployable=False,
                    training_seconds=fed_large_seconds,
                    uplink_bytes=0,
                    downlink_bytes=0,
                    notes="Large rank-4 dataset-specific reference; current packets cannot address 464 parameters.",
                )
            )


def append_shared_large_models(
    records: list[dict[str, object]],
    domains: dict[str, dict[str, dict[str, object]]],
    selected: dict[str, np.ndarray],
    config: ExperimentConfig,
    seed: int,
) -> None:
    """Train one large LoRA autoencoder across all eight dataset/client domains."""
    print("  Training shared large model across all selected datasets/clients")
    all_keys = [client_key(dataset, client) for dataset in config.datasets for client in CLIENTS]
    shared_samples = {key: selected[key] for key in all_keys}
    combined = ordered_stack(
        (shared_samples[key] for key in all_keys), seed, "all_datasets:centralized_large"
    )
    central, central_seconds = train_lora_centralized(
        LARGE_DIMS, LARGE_RANK, combined, config
    )
    federated, federated_seconds = run_fedadam(
        dims=LARGE_DIMS,
        rank=LARGE_RANK,
        client_samples=shared_samples,
        config=config,
        firmware_compression=False,
    )
    lora_count = lora_parameter_count(LARGE_DIMS, LARGE_RANK)
    dense_count = dense_parameter_count(LARGE_DIMS)

    for dataset in config.datasets:
        for client in CLIENTS:
            key = client_key(dataset, client)
            common = {
                "dataset": dataset,
                "client": client,
                "seed": seed,
                "family": "shared_multi_dataset_autoencoder",
                "training_scope": "all_four_datasets_eight_clients",
                "domain": domains[dataset][client],
                "trainable_parameters": lora_count,
                "total_model_parameters": dense_count,
                "rank": LARGE_RANK,
                "deployable": False,
                "uplink_bytes": 0,
                "downlink_bytes": 0,
            }
            records.append(
                evaluate_scores(
                    **common,
                    method="shared_centralized_lora_large",
                    scorer=central.losses,
                    training_seconds=central_seconds,
                    notes="One balanced large LoRA model trained centrally across all eight clients.",
                )
            )
            records.append(
                evaluate_scores(
                    **common,
                    method="shared_federated_float_fedadam_lora_large",
                    scorer=federated[key].losses,
                    training_seconds=federated_seconds,
                    notes="One large float FedAdam LoRA model shared by all eight clients.",
                )
            )


def sweep_method(stage: str, spec: ModelSpec) -> str:
    return f"{stage}_{spec.slug}"


def spec_metadata(spec: ModelSpec) -> dict[str, object]:
    return {
        "architecture": spec.architecture,
        "model_dims": "->".join(map(str, spec.dims)),
        "trainable_parameters": spec.lora_parameters,
        "total_model_parameters": spec.dense_parameters,
        "rank": spec.rank,
        "fits_lilygo_training_memory": spec.fits_lilygo_training_memory,
        "fits_single_lora_packet": spec.fits_single_lora_packet,
        "global_packet_bytes": spec.global_packet_bytes,
        "update_packet_bytes": spec.update_packet_bytes,
    }


def append_architecture_rank_sweep(
    records: list[dict[str, object]],
    domains: dict[str, dict[str, dict[str, object]]],
    selected: dict[str, np.ndarray],
    config: ExperimentConfig,
    seed: int,
) -> None:
    """Run the complete small/bigger x rank-1/2/4 LoRA experiment matrix."""

    # A full-weight centralized reference is rank-independent, so run it once
    # per architecture rather than repeating the same experiment three times.
    for architecture, dims in (("small", SMALL_DIMS), ("bigger", BIGGER_DIMS)):
        for dataset in config.datasets:
            keys = [client_key(dataset, client) for client in CLIENTS]
            combined = ordered_stack(
                (selected[key] for key in keys),
                seed,
                f"{dataset}:centralized_full:{architecture}",
            )
            dense_model, elapsed = train_dense_centralized(dims, combined, config)
            for client in CLIENTS:
                records.append(
                    evaluate_scores(
                        dataset=dataset,
                        client=client,
                        seed=seed,
                        method=f"centralized_full_{architecture}",
                        family="full_weight_autoencoder",
                        training_scope="two_clients_same_dataset",
                        domain=domains[dataset][client],
                        scorer=dense_model.losses,
                        trainable_parameters=dense_parameter_count(dims),
                        total_model_parameters=dense_parameter_count(dims),
                        rank=None,
                        deployable=False,
                        training_seconds=elapsed,
                        uplink_bytes=0,
                        downlink_bytes=0,
                        architecture=architecture,
                        model_dims="->".join(map(str, dims)),
                        notes="All dense W matrices trained; rank-independent research baseline.",
                    )
                )

    for spec in MODEL_SPECS:
        metadata = spec_metadata(spec)
        fragmentation_note = (
            " Fits one LoRa packet."
            if spec.fits_single_lora_packet
            else " Compression is simulated, but real LoRa deployment requires fragmentation."
        )
        print(
            f"  Sweep {spec.slug}: {metadata['model_dims']}, "
            f"LoRA={spec.lora_parameters}, GLOBAL={spec.global_packet_bytes} bytes"
        )

        # Untrained model: establishes whether architecture alone, rather than
        # learning, creates anomaly separation.
        untrained = GeneralLoRAAutoencoder(spec.dims, spec.rank)
        for dataset in config.datasets:
            for client in CLIENTS:
                records.append(
                    evaluate_scores(
                        dataset=dataset,
                        client=client,
                        seed=seed,
                        method=sweep_method("untrained", spec),
                        family="autoencoder",
                        training_scope="none",
                        domain=domains[dataset][client],
                        scorer=untrained.losses,
                        deployable=False,
                        training_seconds=0.0,
                        uplink_bytes=0,
                        downlink_bytes=0,
                        notes="Frozen base with zero LoRA B matrices." + fragmentation_note,
                        **metadata,
                    )
                )

        # Dataset-specific local, centralized, float FedAdam, and compressed
        # FedAdam comparisons under exactly the same per-client sample budget.
        for dataset in config.datasets:
            keys = [client_key(dataset, client) for client in CLIENTS]
            local_models: dict[str, GeneralLoRAAutoencoder] = {}
            local_seconds: dict[str, float] = {}
            for client, key in zip(CLIENTS, keys):
                local_models[key], local_seconds[key] = train_lora_centralized(
                    spec.dims, spec.rank, selected[key], config
                )

            combined = ordered_stack(
                (selected[key] for key in keys), seed, f"{dataset}:centralized:{spec.slug}"
            )
            centralized, centralized_seconds = train_lora_centralized(
                spec.dims, spec.rank, combined, config
            )
            fed_float, fed_float_seconds = run_fedadam(
                dims=spec.dims,
                rank=spec.rank,
                client_samples={key: selected[key] for key in keys},
                config=config,
                firmware_compression=False,
            )
            fed_firmware, fed_firmware_seconds = run_fedadam(
                dims=spec.dims,
                rank=spec.rank,
                client_samples={key: selected[key] for key in keys},
                config=config,
                firmware_compression=True,
            )

            for client, key in zip(CLIENTS, keys):
                common = {
                    "dataset": dataset,
                    "client": client,
                    "seed": seed,
                    "family": "autoencoder",
                    "domain": domains[dataset][client],
                    **metadata,
                }
                records.append(
                    evaluate_scores(
                        **common,
                        method=sweep_method("local", spec),
                        training_scope="single_client",
                        scorer=local_models[key].losses,
                        deployable=False,
                        training_seconds=local_seconds[key],
                        uplink_bytes=0,
                        downlink_bytes=0,
                        notes="Local-only LoRA training." + fragmentation_note,
                    )
                )
                records.append(
                    evaluate_scores(
                        **common,
                        method=sweep_method("centralized_lora", spec),
                        training_scope="two_clients_same_dataset",
                        scorer=centralized.losses,
                        deployable=False,
                        training_seconds=centralized_seconds,
                        uplink_bytes=0,
                        downlink_bytes=0,
                        notes="Compute-matched centralized LoRA reference." + fragmentation_note,
                    )
                )
                records.append(
                    evaluate_scores(
                        **common,
                        method=sweep_method("federated_float", spec),
                        training_scope="two_clients_same_dataset",
                        scorer=fed_float[key].losses,
                        deployable=False,
                        training_seconds=fed_float_seconds,
                        uplink_bytes=0,
                        downlink_bytes=0,
                        notes="Float FedAdam reference without radio compression." + fragmentation_note,
                    )
                )
                records.append(
                    evaluate_scores(
                        **common,
                        method=sweep_method("federated_firmware", spec),
                        training_scope="two_clients_same_dataset",
                        scorer=fed_firmware[key].losses,
                        deployable=(spec.slug == "small_r2"),
                        training_seconds=fed_firmware_seconds,
                        uplink_bytes=config.rounds * spec.update_packet_bytes,
                        downlink_bytes=config.rounds * spec.global_packet_bytes,
                        notes=(
                            "FedAdam + Top-K 32 + signed-int8 UPDATE/GLOBAL quantization."
                            + fragmentation_note
                        ),
                    )
                )

        # One model shared by all selected datasets and all eight clients.
        all_keys = [
            client_key(dataset, client)
            for dataset in config.datasets
            for client in CLIENTS
        ]
        all_samples = {key: selected[key] for key in all_keys}
        combined_all = ordered_stack(
            (all_samples[key] for key in all_keys), seed, f"shared:{spec.slug}"
        )
        shared_centralized, shared_centralized_seconds = train_lora_centralized(
            spec.dims, spec.rank, combined_all, config
        )
        shared_float, shared_float_seconds = run_fedadam(
            dims=spec.dims,
            rank=spec.rank,
            client_samples=all_samples,
            config=config,
            firmware_compression=False,
        )
        shared_firmware, shared_firmware_seconds = run_fedadam(
            dims=spec.dims,
            rank=spec.rank,
            client_samples=all_samples,
            config=config,
            firmware_compression=True,
        )
        for dataset in config.datasets:
            for client in CLIENTS:
                key = client_key(dataset, client)
                common = {
                    "dataset": dataset,
                    "client": client,
                    "seed": seed,
                    "family": "shared_multi_dataset_autoencoder",
                    "training_scope": "all_selected_datasets_and_clients",
                    "domain": domains[dataset][client],
                    "deployable": False,
                    **metadata,
                }
                records.append(
                    evaluate_scores(
                        **common,
                        method=sweep_method("shared_centralized", spec),
                        scorer=shared_centralized.losses,
                        training_seconds=shared_centralized_seconds,
                        uplink_bytes=0,
                        downlink_bytes=0,
                        notes="One centralized LoRA model shared across all clients." + fragmentation_note,
                    )
                )
                records.append(
                    evaluate_scores(
                        **common,
                        method=sweep_method("shared_federated_float", spec),
                        scorer=shared_float[key].losses,
                        training_seconds=shared_float_seconds,
                        uplink_bytes=0,
                        downlink_bytes=0,
                        notes="One float FedAdam LoRA model shared across all clients." + fragmentation_note,
                    )
                )
                records.append(
                    evaluate_scores(
                        **common,
                        method=sweep_method("shared_federated_firmware", spec),
                        scorer=shared_firmware[key].losses,
                        training_seconds=shared_firmware_seconds,
                        uplink_bytes=config.rounds * spec.update_packet_bytes,
                        downlink_bytes=config.rounds * spec.global_packet_bytes,
                        notes=(
                            "Shared FedAdam with Top-K/int8 compression."
                            + fragmentation_note
                        ),
                    )
                )


SUMMARY_METRICS = (
    "validation_f1",
    "validation_fpr",
    "precision",
    "recall",
    "f1",
    "fpr",
    "roc_auc",
    "pr_auc",
    "normal_mean_score",
    "fault_mean_score",
    "score_separation",
    "training_seconds",
)


def summarize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["dataset"]), str(record["client"]), str(record["method"]))].append(record)

    summary: list[dict[str, object]] = []
    for (dataset, client, method), rows in sorted(grouped.items()):
        first = rows[0]
        item: dict[str, object] = {
            "dataset": dataset,
            "client": client,
            "client_name": first["client_name"],
            "method": method,
            "family": first["family"],
            "training_scope": first["training_scope"],
            "architecture": first["architecture"],
            "model_dims": first["model_dims"],
            "seeds": len(rows),
            "trainable_parameters": first["trainable_parameters"],
            "total_model_parameters": first["total_model_parameters"],
            "lora_rank": first["lora_rank"],
            "deployable_on_current_firmware": first["deployable_on_current_firmware"],
            "fits_lilygo_training_memory": first["fits_lilygo_training_memory"],
            "fits_single_lora_packet": first["fits_single_lora_packet"],
            "requires_fragmentation": first["requires_fragmentation"],
            "global_packet_bytes": first["global_packet_bytes"],
            "update_packet_bytes": first["update_packet_bytes"],
            "uplink_bytes_per_client": first["uplink_bytes"],
            "downlink_bytes_per_client": first["downlink_bytes"],
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        methods = sorted({str(row["threshold_method"]) for row in rows})
        item["selected_threshold_methods"] = ",".join(methods)
        summary.append(item)
    return summary


def summary_lookup(summary: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {
        (str(row["dataset"]), str(row["client"]), str(row["method"])): row
        for row in summary
    }


def status_from_delta(delta: float, supported: float = 0.02) -> str:
    if delta >= supported:
        return "Supported"
    if delta > -supported:
        return "Partially supported"
    return "Not supported"


def average_metric(
    lookup: dict[tuple[str, str, str], dict[str, object]],
    config: ExperimentConfig,
    method: str,
    metric: str,
) -> float:
    values = [
        float(lookup[(dataset, client, method)][f"{metric}_mean"])
        for dataset in config.datasets
        for client in CLIENTS
    ]
    return mean(values)


def build_question_assessments(
    summary: list[dict[str, object]], config: ExperimentConfig
) -> list[dict[str, str]]:
    lookup = summary_lookup(summary)
    pairs = [(dataset, client) for dataset in config.datasets for client in CLIENTS]
    baseline_methods = tuple(FEATURE_BASELINES)
    firmware_method = "federated_firmware_fedadam_lora_light"
    float_method = "federated_float_fedadam_lora_light"
    local_method = "local_lora_light"
    central_method = "centralized_lora_light"
    untrained_method = "untrained_lora_light"
    dataset_large = "dataset_federated_float_fedadam_lora_large"
    shared_large = "shared_federated_float_fedadam_lora_large"

    best_baseline_f1: dict[tuple[str, str], float] = {}
    best_baseline_name: dict[tuple[str, str], str] = {}
    for pair in pairs:
        candidates = [
            (method, float(lookup[(*pair, method)]["f1_mean"])) for method in baseline_methods
        ]
        best_name, best_value = max(candidates, key=lambda item: item[1])
        best_baseline_f1[pair] = best_value
        best_baseline_name[pair] = best_name

    easy_pairs = [
        pair
        for pair in pairs
        if best_baseline_f1[pair]
        >= float(lookup[(*pair, firmware_method)]["f1_mean"]) - 0.02
    ]
    training_delta = mean(
        float(lookup[(*pair, firmware_method)]["f1_mean"])
        - float(lookup[(*pair, untrained_method)]["f1_mean"])
        for pair in pairs
    )
    federation_delta = mean(
        float(lookup[(*pair, firmware_method)]["f1_mean"])
        - float(lookup[(*pair, local_method)]["f1_mean"])
        for pair in pairs
    )
    central_delta = mean(
        float(lookup[(*pair, float_method)]["f1_mean"])
        - float(lookup[(*pair, central_method)]["f1_mean"])
        for pair in pairs
    )
    shared_f1_delta = average_metric(lookup, config, shared_large, "f1") - average_metric(
        lookup, config, dataset_large, "f1"
    )
    shared_fpr_delta = average_metric(lookup, config, shared_large, "fpr") - average_metric(
        lookup, config, dataset_large, "fpr"
    )
    light_size_delta = average_metric(lookup, config, float_method, "f1") - average_metric(
        lookup, config, dataset_large, "f1"
    )
    quantization_delta = average_metric(lookup, config, firmware_method, "f1") - average_metric(
        lookup, config, float_method, "f1"
    )
    quantization_fpr_delta = average_metric(lookup, config, firmware_method, "fpr") - average_metric(
        lookup, config, float_method, "fpr"
    )
    helped_pairs = [
        pair
        for pair in pairs
        if float(lookup[(*pair, firmware_method)]["f1_mean"])
        > float(lookup[(*pair, local_method)]["f1_mean"]) + 0.02
    ]
    maximum_seed_std = max(
        float(lookup[(*pair, firmware_method)]["f1_std"]) for pair in pairs
    )
    difficult_pairs = [
        pair
        for pair in pairs
        if float(lookup[(*pair, firmware_method)]["f1_mean"]) < 0.80
        or float(lookup[(*pair, firmware_method)]["fpr_mean"]) > 0.10
    ]
    deployment_passes = [
        pair
        for pair in pairs
        if float(lookup[(*pair, firmware_method)]["f1_mean"]) >= 0.80
        and float(lookup[(*pair, firmware_method)]["fpr_mean"]) <= 0.10
    ]

    pair_text = lambda values: ", ".join(f"{d}/{CLIENT_DISPLAY_NAMES[(d, c)]}" for d, c in values) or "none"
    return [
        {
            "question": RESEARCH_QUESTIONS[0],
            "status": (
                "Supported"
                if len(easy_pairs) == len(pairs)
                else "Partially supported"
                if easy_pairs
                else "Not supported"
            ),
            "evidence": (
                f"A best one-feature baseline was within 2 F1 points of firmware FedAdam on "
                f"{len(easy_pairs)}/{len(pairs)} clients: {pair_text(easy_pairs)}."
            ),
        },
        {
            "question": RESEARCH_QUESTIONS[1],
            "status": status_from_delta(training_delta),
            "evidence": f"Mean firmware-trained minus untrained F1 = {training_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[2],
            "status": status_from_delta(federation_delta),
            "evidence": f"Mean firmware FedAdam minus local-only F1 = {federation_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[3],
            "status": status_from_delta(central_delta),
            "evidence": f"Mean float FedAdam minus compute-matched centralized LoRA F1 = {central_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[4],
            "status": (
                "Supported"
                if shared_f1_delta >= -0.02 and shared_fpr_delta <= 0.02
                else "Partially supported"
                if shared_f1_delta >= -0.05 and shared_fpr_delta <= 0.05
                else "Not supported"
            ),
            "evidence": (
                f"Shared-large minus dataset-large F1 = {shared_f1_delta:+.4f}; "
                f"FPR change = {shared_fpr_delta:+.4f}."
            ),
        },
        {
            "question": RESEARCH_QUESTIONS[5],
            "status": status_from_delta(shared_f1_delta),
            "evidence": f"Shared-large minus dataset-specific-large mean F1 = {shared_f1_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[6],
            "status": "Supported" if light_size_delta >= -0.02 else "Partially supported" if light_size_delta >= -0.05 else "Not supported",
            "evidence": f"Float lightweight minus float large mean F1 = {light_size_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[7],
            "status": "Supported" if quantization_delta >= -0.02 and quantization_fpr_delta <= 0.02 else "Partially supported" if quantization_delta >= -0.05 else "Not supported",
            "evidence": (
                f"Firmware compression changes mean F1 by {quantization_delta:+.4f} and "
                f"mean FPR by {quantization_fpr_delta:+.4f}."
            ),
        },
        {
            "question": RESEARCH_QUESTIONS[8],
            "status": "Supported" if helped_pairs else "Not supported",
            "evidence": f"Federation improved F1 by more than 2 points on {len(helped_pairs)}/{len(pairs)} clients: {pair_text(helped_pairs)}.",
        },
        {
            "question": RESEARCH_QUESTIONS[9],
            "status": "Supported" if maximum_seed_std <= 0.03 else "Partially supported" if maximum_seed_std <= 0.07 else "Not supported",
            "evidence": f"Maximum firmware-model F1 standard deviation across seeds = {maximum_seed_std:.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[10],
            "status": "Supported" if difficult_pairs else "Not supported",
            "evidence": f"Clients with F1 < 0.80 or FPR > 0.10: {pair_text(difficult_pairs)}.",
        },
        {
            "question": RESEARCH_QUESTIONS[11],
            "status": "Supported" if len(deployment_passes) == len(pairs) else "Partially supported" if deployment_passes else "Not supported",
            "evidence": (
                f"Suggested gate (F1 >= 0.80 and FPR <= 0.10) passed on "
                f"{len(deployment_passes)}/{len(pairs)} clients: {pair_text(deployment_passes)}."
            ),
        },
    ]


def build_sweep_question_assessments(
    summary: list[dict[str, object]], config: ExperimentConfig
) -> tuple[list[dict[str, str]], ModelSpec]:
    """Assess the research questions using the complete six-configuration sweep."""
    lookup = summary_lookup(summary)
    pairs = [(dataset, client) for dataset in config.datasets for client in CLIENTS]

    def method_average(method: str, metric: str) -> float:
        return mean(
            float(lookup[(*pair, method)][f"{metric}_mean"])
            for pair in pairs
        )

    def firmware_method(spec: ModelSpec) -> str:
        return sweep_method("federated_firmware", spec)

    # Rank configurations by mean F1, then prefer lower FPR on a tie.
    best_spec = max(
        MODEL_SPECS,
        key=lambda spec: (
            method_average(firmware_method(spec), "f1"),
            -method_average(firmware_method(spec), "fpr"),
        ),
    )
    best_firmware = firmware_method(best_spec)
    best_untrained = sweep_method("untrained", best_spec)
    best_local = sweep_method("local", best_spec)
    best_centralized = sweep_method("centralized_lora", best_spec)
    best_float = sweep_method("federated_float", best_spec)
    best_shared_float = sweep_method("shared_federated_float", best_spec)

    best_baseline: dict[tuple[str, str], tuple[str, dict[str, object]]] = {}
    best_firmware_by_pair: dict[tuple[str, str], tuple[ModelSpec, dict[str, object]]] = {}
    for pair in pairs:
        baseline_name = max(
            FEATURE_BASELINES,
            key=lambda method: (
                float(lookup[(*pair, method)]["f1_mean"]),
                -float(lookup[(*pair, method)]["fpr_mean"]),
            ),
        )
        best_baseline[pair] = (baseline_name, lookup[(*pair, baseline_name)])
        pair_spec = max(
            MODEL_SPECS,
            key=lambda spec: (
                float(lookup[(*pair, firmware_method(spec))]["f1_mean"]),
                -float(lookup[(*pair, firmware_method(spec))]["fpr_mean"]),
            ),
        )
        best_firmware_by_pair[pair] = (
            pair_spec,
            lookup[(*pair, firmware_method(pair_spec))],
        )

    easy_pairs = []
    trained_better_pairs = []
    for pair in pairs:
        _, baseline = best_baseline[pair]
        pair_spec, trained = best_firmware_by_pair[pair]
        untrained = lookup[(*pair, sweep_method("untrained", pair_spec))]
        if (
            float(baseline["f1_mean"]) >= 0.80
            and float(baseline["fpr_mean"]) <= 0.10
            and float(baseline["f1_mean"]) >= float(trained["f1_mean"]) - 0.02
        ):
            easy_pairs.append(pair)
        if float(trained["f1_mean"]) > max(
            float(untrained["f1_mean"]), float(baseline["f1_mean"])
        ) + 0.02:
            trained_better_pairs.append(pair)

    training_delta = method_average(best_firmware, "f1") - method_average(
        best_untrained, "f1"
    )
    federation_delta = method_average(best_firmware, "f1") - method_average(
        best_local, "f1"
    )
    centralized_delta = method_average(best_float, "f1") - method_average(
        best_centralized, "f1"
    )
    shared_f1_delta = method_average(best_shared_float, "f1") - method_average(
        best_float, "f1"
    )
    shared_fpr_delta = method_average(best_shared_float, "fpr") - method_average(
        best_float, "fpr"
    )
    quantization_delta = method_average(best_firmware, "f1") - method_average(
        best_float, "f1"
    )
    quantization_fpr_delta = method_average(best_firmware, "fpr") - method_average(
        best_float, "fpr"
    )

    best_small = max(
        (spec for spec in MODEL_SPECS if spec.architecture == "small"),
        key=lambda spec: method_average(firmware_method(spec), "f1"),
    )
    best_bigger = max(
        (spec for spec in MODEL_SPECS if spec.architecture == "bigger"),
        key=lambda spec: method_average(firmware_method(spec), "f1"),
    )
    size_delta = method_average(firmware_method(best_bigger), "f1") - method_average(
        firmware_method(best_small), "f1"
    )

    helped_pairs = [
        pair
        for pair in pairs
        if float(lookup[(*pair, best_firmware)]["f1_mean"])
        > float(lookup[(*pair, best_local)]["f1_mean"]) + 0.02
    ]
    maximum_seed_std = max(
        float(lookup[(*pair, best_firmware)]["f1_std"]) for pair in pairs
    )
    difficult_pairs = [
        pair
        for pair in pairs
        if float(best_firmware_by_pair[pair][1]["f1_mean"]) < 0.80
        or float(best_firmware_by_pair[pair][1]["fpr_mean"]) > 0.10
    ]
    deployment_passes = [
        pair
        for pair in pairs
        if float(lookup[(*pair, best_firmware)]["f1_mean"]) >= 0.80
        and float(lookup[(*pair, best_firmware)]["fpr_mean"]) <= 0.10
    ]

    def pair_text(values: Iterable[tuple[str, str]]) -> str:
        rendered = [f"{dataset}/{CLIENT_DISPLAY_NAMES[(dataset, client)]}" for dataset, client in values]
        return ", ".join(rendered) if rendered else "none"

    configuration_evidence = (
        f"Best mean-F1 configuration: {best_spec.slug} "
        f"({best_spec.lora_parameters} LoRA values, "
        f"GLOBAL {best_spec.global_packet_bytes} bytes, "
        f"single-packet={best_spec.fits_single_lora_packet})."
    )
    assessments = [
        {
            "question": RESEARCH_QUESTIONS[0],
            "status": "Supported" if len(easy_pairs) == len(pairs) else "Partially supported" if easy_pairs else "Not supported",
            "evidence": f"Strict easy-data rule passed on {len(easy_pairs)}/{len(pairs)} clients: {pair_text(easy_pairs)}.",
        },
        {
            "question": RESEARCH_QUESTIONS[1],
            "status": "Supported" if len(trained_better_pairs) == len(pairs) else "Partially supported" if trained_better_pairs else "Not supported",
            "evidence": (
                f"A trained configuration beat both its untrained version and the best one-feature baseline by >2 F1 points on "
                f"{len(trained_better_pairs)}/{len(pairs)} clients: {pair_text(trained_better_pairs)}. "
                f"For {best_spec.slug}, trained-untrained mean F1={training_delta:+.4f}."
            ),
        },
        {
            "question": RESEARCH_QUESTIONS[2],
            "status": status_from_delta(federation_delta),
            "evidence": f"For {best_spec.slug}, firmware FedAdam minus local-only mean F1={federation_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[3],
            "status": status_from_delta(centralized_delta),
            "evidence": f"For {best_spec.slug}, float FedAdam minus compute-matched centralized LoRA mean F1={centralized_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[4],
            "status": "Supported" if shared_f1_delta >= -0.02 and shared_fpr_delta <= 0.02 else "Partially supported" if shared_f1_delta >= -0.05 and shared_fpr_delta <= 0.05 else "Not supported",
            "evidence": f"For {best_spec.slug}, shared minus dataset-specific F1={shared_f1_delta:+.4f}, FPR={shared_fpr_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[5],
            "status": status_from_delta(shared_f1_delta),
            "evidence": f"For {best_spec.slug}, shared minus dataset-specific mean F1={shared_f1_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[6],
            "status": "Supported" if best_spec.fits_single_lora_packet else "Partially supported",
            "evidence": configuration_evidence + f" Best-bigger minus best-small mean F1={size_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[7],
            "status": "Supported" if quantization_delta >= -0.02 and quantization_fpr_delta <= 0.02 else "Partially supported" if quantization_delta >= -0.05 else "Not supported",
            "evidence": f"For {best_spec.slug}, compression changes mean F1={quantization_delta:+.4f}, FPR={quantization_fpr_delta:+.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[8],
            "status": "Supported" if helped_pairs else "Not supported",
            "evidence": f"Best overall configuration improved >2 F1 points over local training on {len(helped_pairs)}/{len(pairs)} clients: {pair_text(helped_pairs)}.",
        },
        {
            "question": RESEARCH_QUESTIONS[9],
            "status": "Supported" if maximum_seed_std <= 0.03 else "Partially supported" if maximum_seed_std <= 0.07 else "Not supported",
            "evidence": f"Maximum {best_spec.slug} firmware F1 standard deviation={maximum_seed_std:.4f}.",
        },
        {
            "question": RESEARCH_QUESTIONS[10],
            "status": "Supported" if difficult_pairs else "Not supported",
            "evidence": f"Still difficult after choosing each client's best configuration: {pair_text(difficult_pairs)}.",
        },
        {
            "question": RESEARCH_QUESTIONS[11],
            "status": "Supported" if len(deployment_passes) == len(pairs) and best_spec.fits_single_lora_packet else "Partially supported" if deployment_passes else "Not supported",
            "evidence": f"{best_spec.slug} passed F1>=0.80/FPR<=0.10 on {len(deployment_passes)}/{len(pairs)} clients: {pair_text(deployment_passes)}.",
        },
    ]
    return assessments, best_spec


RUN_FIELDS = (
    "dataset",
    "client",
    "client_name",
    "seed",
    "method",
    "family",
    "training_scope",
    "architecture",
    "model_dims",
    "threshold_method",
    "threshold",
    "validation_f1",
    "validation_fpr",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "fpr",
    "roc_auc",
    "pr_auc",
    "normal_mean_score",
    "fault_mean_score",
    "score_separation",
    "trainable_parameters",
    "total_model_parameters",
    "lora_rank",
    "deployable_on_current_firmware",
    "fits_lilygo_training_memory",
    "fits_single_lora_packet",
    "requires_fragmentation",
    "global_packet_bytes",
    "update_packet_bytes",
    "uplink_bytes",
    "downlink_bytes",
    "training_seconds",
    "notes",
)


def write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(fields) if fields is not None else list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def write_markdown_report(
    path: Path,
    config: ExperimentConfig,
    summary: list[dict[str, object]],
    assessments: list[dict[str, str]],
) -> None:
    lookup = summary_lookup(summary)
    core_methods = (
        "baseline_rms",
        "baseline_peak_to_peak",
        "untrained_lora_light",
        "local_lora_light",
        "centralized_lora_light",
        "federated_float_fedadam_lora_light",
        "federated_firmware_fedadam_lora_light",
        "dataset_federated_float_fedadam_lora_large",
        "shared_federated_float_fedadam_lora_large",
    )
    lines = [
        "# Full TinyFedLoRA Simulation Report",
        "",
        "This report was generated entirely from prepared feature binaries. Raw dataset files were not read.",
        "",
        "## Configuration",
        "",
        f"- Datasets: {', '.join(config.datasets)}",
        f"- Seeds: {', '.join(map(str, config.seeds))}",
        f"- Training budget: {config.rounds} rounds x {config.samples_per_round} samples/client/round x {config.local_epochs} epochs",
        f"- Lightweight LoRA: {LIGHT_DIMS}, rank {LIGHT_RANK}, {lora_parameter_count(LIGHT_DIMS, LIGHT_RANK)} trainable values",
        f"- Large LoRA: {LARGE_DIMS}, rank {LARGE_RANK}, {lora_parameter_count(LARGE_DIMS, LARGE_RANK)} trainable values",
        "- Thresholds are selected on validation data and frozen before final-test evaluation.",
        "",
        "## Core results (mean across seeds)",
        "",
        "| Dataset | Client | Method | F1 | FPR | Recall | PR-AUC |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset in config.datasets:
        for client in CLIENTS:
            for method in core_methods:
                row = lookup[(dataset, client, method)]
                lines.append(
                    f"| {dataset.upper()} | {row['client_name']} | {method} | "
                    f"{100*float(row['f1_mean']):.2f}% +/- {100*float(row['f1_std']):.2f}% | "
                    f"{100*float(row['fpr_mean']):.2f}% | {100*float(row['recall_mean']):.2f}% | "
                    f"{100*float(row['pr_auc_mean']):.2f}% |"
                )

    lines.extend(["", "## Research-question assessment", ""])
    for index, assessment in enumerate(assessments, start=1):
        lines.extend(
            [
                f"### {index}. {assessment['question']}",
                "",
                f"**{assessment['status']}** — {assessment['evidence']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation rule",
            "",
            "A high F1 score alone is not sufficient for deployment. Inspect false-positive rate, recall, seed variation, and the simple baselines. A dataset where a one-feature threshold matches the autoencoder is evidence that the data is intrinsically easy, not evidence that the autoencoder caused the separation.",
            "",
            "The shared large model is a research experiment and is not compatible with the current 128-value/uint8-index LilyGO packet format.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sweep_markdown_report(
    path: Path,
    config: ExperimentConfig,
    summary: list[dict[str, object]],
    assessments: list[dict[str, str]],
    best_spec: ModelSpec,
) -> None:
    lookup = summary_lookup(summary)
    pairs = [(dataset, client) for dataset in config.datasets for client in CLIENTS]

    def average_for(method: str, metric: str) -> float:
        return mean(
            float(lookup[(*pair, method)][f"{metric}_mean"])
            for pair in pairs
        )

    lines = [
        "# Full TinyFedLoRA Architecture/Rank Sweep",
        "",
        "This report was generated from prepared feature binaries only. Raw dataset archives were not read.",
        "",
        "## Configuration",
        "",
        f"- Datasets: {', '.join(config.datasets)}",
        f"- Clients: {2 * len(config.datasets)}",
        f"- Seeds: {', '.join(map(str, config.seeds))}",
        f"- Training budget: {config.rounds} rounds x {config.samples_per_round} samples/client/round x {config.local_epochs} epochs",
        "- Architectures: small 12->16->8->4->8->16->12; bigger 12->24->12->4->12->24->12",
        "- LoRA ranks: 1, 2, 4 for both architectures (six configurations)",
        "- Threshold method is chosen on validation data and frozen before final-test evaluation.",
        "",
        "## Architecture/rank comparison",
        "",
        "| Configuration | LoRA values | Est. train state | UPDATE | GLOBAL | One packet | Mean firmware F1 | Mean FPR | Trained-untrained F1 | Quantization F1 cost | Readiness clients |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for spec in MODEL_SPECS:
        firmware = sweep_method("federated_firmware", spec)
        untrained = sweep_method("untrained", spec)
        float_method = sweep_method("federated_float", spec)
        firmware_f1 = average_for(firmware, "f1")
        firmware_fpr = average_for(firmware, "fpr")
        training_delta = firmware_f1 - average_for(untrained, "f1")
        quantization_delta = firmware_f1 - average_for(float_method, "f1")
        ready = sum(
            float(lookup[(*pair, firmware)]["f1_mean"]) >= 0.80
            and float(lookup[(*pair, firmware)]["fpr_mean"]) <= 0.10
            for pair in pairs
        )
        lines.append(
            f"| {spec.slug} | {spec.lora_parameters} | "
            f"{spec.estimated_training_state_bytes / 1024:.1f} KiB | "
            f"{spec.update_packet_bytes} B | {spec.global_packet_bytes} B | "
            f"{'yes' if spec.fits_single_lora_packet else 'no'} | "
            f"{100*firmware_f1:.2f}% | {100*firmware_fpr:.2f}% | "
            f"{100*training_delta:+.2f} pp | {100*quantization_delta:+.2f} pp | "
            f"{ready}/{len(pairs)} |"
        )

    lines.extend(
        [
            "",
            f"Best overall mean-F1 configuration: **{best_spec.slug}**.",
            "",
            "Rank-4 models fit LilyGO training memory, but their GLOBAL packets exceed LoRa's 255-byte single-packet limit. The bigger rank-4 model also needs two-byte Top-K indices. Their compression accuracy is simulated, while real deployment would require fragmentation and protocol changes.",
            "",
            "## Best compressed LoRA configuration per client",
            "",
            "| Dataset | Client | Best configuration | F1 | FPR | Recall | Simple baseline | Baseline F1 |",
            "|---|---|---|---:|---:|---:|---|---:|",
        ]
    )
    for dataset, client in pairs:
        spec = max(
            MODEL_SPECS,
            key=lambda candidate: (
                float(lookup[(dataset, client, sweep_method('federated_firmware', candidate))]["f1_mean"]),
                -float(lookup[(dataset, client, sweep_method('federated_firmware', candidate))]["fpr_mean"]),
            ),
        )
        result = lookup[(dataset, client, sweep_method("federated_firmware", spec))]
        baseline_name = max(
            FEATURE_BASELINES,
            key=lambda method: (
                float(lookup[(dataset, client, method)]["f1_mean"]),
                -float(lookup[(dataset, client, method)]["fpr_mean"]),
            ),
        )
        baseline = lookup[(dataset, client, baseline_name)]
        lines.append(
            f"| {dataset.upper()} | {CLIENT_DISPLAY_NAMES[(dataset, client)]} | {spec.slug} | "
            f"{100*float(result['f1_mean']):.2f}% | {100*float(result['fpr_mean']):.2f}% | "
            f"{100*float(result['recall_mean']):.2f}% | {baseline_name} | "
            f"{100*float(baseline['f1_mean']):.2f}% |"
        )

    lines.extend(["", "## Research-question assessment", ""])
    for index, assessment in enumerate(assessments, start=1):
        lines.extend(
            [
                f"### {index}. {assessment['question']}",
                "",
                f"**{assessment['status']}** - {assessment['evidence']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "Compare trained models against their matching untrained architecture/rank and against the best one-feature baseline. A higher-capacity untrained model performing well is not evidence that LoRA training helped.",
            "",
            "A model is hardware-promising only when F1, false-positive rate, stability, memory, and packet constraints are all acceptable. Detailed per-seed results are available in full_simulation_runs.csv.",
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
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown datasets: {', '.join(unknown)}")
    if not datasets:
        raise argparse.ArgumentTypeError("at least one dataset is required")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=(11, 29, 47, 71, 97),
        help="comma-separated data-ordering seeds (default: 11,29,47,71,97)",
    )
    parser.add_argument(
        "--datasets",
        type=parse_datasets,
        default=DATASETS,
        help="comma-separated prepared datasets (default: all four)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="directory for CSV, JSON, and Markdown outputs",
    )
    args = parser.parse_args()

    config = ExperimentConfig(datasets=tuple(args.datasets), seeds=tuple(args.seeds))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and validating prepared datasets (raw archives are never opened)...")
    domains = load_all_domains(config)
    records: list[dict[str, object]] = []

    for seed in config.seeds:
        print(f"\n=== Seed {seed} ===")
        selected = select_training_windows(domains, config, seed)
        append_baselines(records, domains, config, seed)
        append_architecture_rank_sweep(records, domains, selected, config, seed)

    summary = summarize_records(records)
    assessments, best_spec = build_sweep_question_assessments(summary, config)
    payload = {
        "study": "Full TinyFedLoRA small/bigger architecture and rank sweep",
        "config": asdict(config),
        "model_sweep": [
            {
                "slug": spec.slug,
                "architecture": spec.architecture,
                "dims": spec.dims,
                "rank": spec.rank,
                "lora_parameters": spec.lora_parameters,
                "dense_parameters": spec.dense_parameters,
                "estimated_training_state_bytes": spec.estimated_training_state_bytes,
                "update_packet_bytes": spec.update_packet_bytes,
                "global_packet_bytes": spec.global_packet_bytes,
                "fits_lilygo_training_memory": spec.fits_lilygo_training_memory,
                "fits_single_lora_packet": spec.fits_single_lora_packet,
            }
            for spec in MODEL_SPECS
        ],
        "best_mean_f1_configuration": best_spec.slug,
        "research_questions": list(RESEARCH_QUESTIONS),
        "question_assessments": assessments,
        "summary": summary,
        "runs": records,
        "limitations": [
            "Run-to-failure fault labels depend on the prepared manifests and their declared time boundaries.",
            "Seeds vary balanced training-window selection/order; firmware frozen W/A initialization remains deterministic.",
            "All six configurations fit estimated LilyGO training memory.",
            "Rank-4 GLOBAL models exceed one 255-byte LoRa packet; bigger rank 4 also requires uint16 Top-K indices.",
            "Only small rank 2 matches the currently compiled 128-value firmware without protocol/model edits.",
            "Simulation cannot establish real radio reliability, SD-card reliability, timing, energy use, or thermal behaviour.",
        ],
    }

    json_path = output_dir / "full_simulation_results.json"
    runs_path = output_dir / "full_simulation_runs.csv"
    summary_path = output_dir / "full_simulation_summary.csv"
    report_path = output_dir / "FULL_SIMULATION_REPORT.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(runs_path, records, RUN_FIELDS)
    write_csv(summary_path, summary)
    write_sweep_markdown_report(report_path, config, summary, assessments, best_spec)

    print("\nSimulation complete")
    print(f"  Full JSON: {json_path}")
    print(f"  Runs CSV:  {runs_path}")
    print(f"  Summary:   {summary_path}")
    print(f"  Report:    {report_path}")


if __name__ == "__main__":
    main()
