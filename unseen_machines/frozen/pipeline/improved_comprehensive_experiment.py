"""Scientifically controlled comprehensive federated-AE experiment.

This replaces the scratch ``final_comprehensive_table.py`` workflow without
overwriting it.  It fixes the important causes of the weak/contradictory table:

* every trainable model starts from a genuinely federated pretrained base;
* LoRA training uses real client rounds rather than one centralized stack;
* B-only AdamW keeps its state throughout local training;
* denoising means noisy input -> clean target;
* model, score-fusion weight, and threshold are selected on validation only;
* test labels are used once, after selection;
* IMS labels are never rewritten to manufacture a "corrected" F1 score.

Only prepared ``*_bin`` folders are read.  Raw archives are never opened.

Example (full five-seed study):

    python improved_comprehensive_experiment.py --seeds 11,29,47,71,97

The script is intentionally not imported by the firmware.  It writes a new
``results/improved_comprehensive`` directory and preserves all old results.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
if str(RELEASE_DIR) not in sys.path:
    sys.path.insert(0, str(RELEASE_DIR))

from cwru_eval import classification_metrics, load_domain, load_normalization  # noqa: E402
from learning_failure_experiments import (  # noqa: E402
    ABLoRA,
    DenseAE,
    LARGER_DIMS,
    WindowCycle,
    _init_base_weights,
    federated_train_dense,
    federated_train_lora,
    save_base_npz,
    stable_seed,
)
from run_full_simulation import (  # noqa: E402
    CLIENTS,
    ExperimentConfig,
    choose_threshold,
    validate_domain,
)


ALL_DATASETS = ("cwru", "xjtu", "ims", "pronostia", "paderborn")
DISPLAY_NAMES = {
    ("cwru", "client_a"): "CWRU DE",
    ("cwru", "client_b"): "CWRU FE",
    ("xjtu", "client_a"): "XJTU Bearing 1_1",
    ("xjtu", "client_b"): "XJTU Bearing 1_2",
    ("ims", "client_a"): "IMS Odd",
    ("ims", "client_b"): "IMS Even",
    ("pronostia", "client_a"): "FEMTO Bearing 1_1",
    ("pronostia", "client_b"): "FEMTO Bearing 1_2",
    ("paderborn", "client_a"): "PU Client A (Outer)",
    ("paderborn", "client_b"): "PU Client B (Inner)",
}

FUSION_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class StudyConfig:
    datasets: tuple[str, ...] = ALL_DATASETS
    seeds: tuple[int, ...] = (11, 29, 47, 71, 97)
    warmup_windows: int = 1000
    warmup_fraction: float = 0.70
    base_rounds: int = 20
    base_samples_per_round: int = 60
    base_local_epochs: int = 2
    base_client_lr: float = 0.01
    adaptation_rounds: int = 20
    adaptation_samples_per_round: int = 60
    adaptation_local_epochs: int = 2
    adaptation_client_lr: float = 0.01
    lora_alpha: float = 0.5
    lora_rank: int = 1
    denoising_sigma: float = 0.05
    server_algorithm: str = "fedadam"
    fedadam_eta: float = 0.005
    firmware_compression: bool = False
    sparsity_k: int = 32


class OnlineWelford:
    """Numerically stable per-client normalization learned from warmup only."""

    def __init__(self, width: int = 12) -> None:
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, sample: np.ndarray) -> None:
        value = np.asarray(sample, dtype=np.float64)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    def statistics(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("online normalization requires at least two warmup windows")
        variance = self.m2 / (self.count - 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        std = np.where(std < 1e-6, 1.0, std)
        return self.mean.astype(np.float32), std.astype(np.float32)


def _prepared_to_raw(
    prepared: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    return (
        np.asarray(prepared, dtype=np.float32) * np.asarray(stds, dtype=np.float32)
        + np.asarray(means, dtype=np.float32)
    ).astype(np.float32)


def _online_normalize(
    raw: np.ndarray, mean_values: np.ndarray, std_values: np.ndarray
) -> np.ndarray:
    return ((np.asarray(raw, dtype=np.float32) - mean_values) / std_values).astype(
        np.float32
    )


def _warmup_count(
    total: int,
    requested: int,
    warmup_fraction: float,
    adaptation_batch_size: int,
) -> int:
    """Choose a warmup count while retaining meaningful adaptation data.

    ``requested`` remains a cap (normally 1,000), while ``warmup_fraction``
    defaults to 70%. At least one complete adaptation batch is retained when
    the client has enough windows; for smaller synthetic/test clients, at least
    two warmup and two adaptation windows are retained.
    """
    if total < 4:
        raise ValueError("each client needs at least four healthy training windows")
    if requested < 2:
        raise ValueError("warmup window cap must be at least two")
    if not 0.0 < warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be strictly between 0 and 1")
    if adaptation_batch_size < 1:
        raise ValueError("adaptation_batch_size must be positive")

    fractional_count = int(np.floor(total * warmup_fraction))
    reserved_adaptation = min(adaptation_batch_size, total - 2)
    return min(int(requested), fractional_count, total - reserved_adaptation)


def load_online_domains(config: StudyConfig):
    """Load prepared data, validate splits, then apply warmup-only normalization.

    Returns two domain trees with identical evaluation splits:
    ``pretrain_domains`` trains the dense base on warmup windows; ``adapt_domains``
    trains LoRA/dense continuation only on post-warmup healthy windows.
    """
    pretrain_domains: dict[str, dict[str, dict[str, object]]] = {}
    adapt_domains: dict[str, dict[str, dict[str, object]]] = {}
    normalization_rows: list[dict[str, object]] = []

    for dataset in config.datasets:
        bin_dir = PYTHON_TOOLS_DIR / f"{dataset}_bin"
        if not bin_dir.is_dir():
            raise FileNotFoundError(f"missing prepared dataset directory: {bin_dir}")
        prepared_means, prepared_stds = load_normalization(bin_dir)
        pretrain_domains[dataset] = {}
        adapt_domains[dataset] = {}

        for client in CLIENTS:
            original = load_domain(client, bin_dir)
            validate_domain(dataset, client, original)
            raw_splits = {
                split: _prepared_to_raw(
                    np.asarray(original[split], dtype=np.float32),
                    prepared_means,
                    prepared_stds,
                )
                for split in ("train", "calibration", "val", "test")
            }
            count = _warmup_count(
                len(raw_splits["train"]),
                config.warmup_windows,
                config.warmup_fraction,
                config.adaptation_samples_per_round,
            )
            warmup_raw = raw_splits["train"][:count]
            adaptation_raw = raw_splits["train"][count:]

            estimator = OnlineWelford(width=warmup_raw.shape[1])
            for sample in warmup_raw:
                estimator.update(sample)
            online_mean, online_std = estimator.statistics()

            normalized = {
                split: _online_normalize(values, online_mean, online_std)
                for split, values in raw_splits.items()
            }
            warmup_normalized = normalized["train"][:count]
            adaptation_normalized = normalized["train"][count:]

            common = dict(original)
            common.update(
                {
                    "calibration": normalized["calibration"],
                    "val": normalized["val"],
                    "test": normalized["test"],
                }
            )
            pretrain = dict(common)
            pretrain["train"] = warmup_normalized
            adapt = dict(common)
            adapt["train"] = adaptation_normalized
            pretrain_domains[dataset][client] = pretrain
            adapt_domains[dataset][client] = adapt
            normalization_rows.append(
                {
                    "dataset": dataset,
                    "client": client,
                    "client_name": DISPLAY_NAMES[(dataset, client)],
                    "warmup_windows": count,
                    "adaptation_windows": len(adaptation_normalized),
                    "online_mean": online_mean.tolist(),
                    "online_std": online_std.tolist(),
                }
            )

    return pretrain_domains, adapt_domains, normalization_rows


def _cycles_for(
    domains: dict[str, dict[str, dict[str, object]]],
    datasets: Iterable[str],
    seed: int,
    name: str,
) -> dict[str, WindowCycle]:
    return {
        f"{dataset}:{client}": WindowCycle(
            np.asarray(domains[dataset][client]["train"], dtype=np.float32),
            stable_seed(seed, f"{name}:{dataset}:{client}"),
        )
        for dataset in datasets
        for client in CLIENTS
    }


def pretrain_shared_base(
    domains: dict[str, dict[str, dict[str, object]]],
    config: StudyConfig,
    seed: int,
) -> list[np.ndarray]:
    training_config = ExperimentConfig(
        datasets=config.datasets,
        rounds=config.base_rounds,
        samples_per_round=config.base_samples_per_round,
        local_epochs=config.base_local_epochs,
        learning_rate=config.base_client_lr,
        fedadam_eta=config.fedadam_eta,
    )
    initial = _init_base_weights(LARGER_DIMS)
    flat, factory, _ = federated_train_dense(
        dims=LARGER_DIMS,
        base_weights=initial,
        datasets=config.datasets,
        domains=domains,
        config=training_config,
        seed=seed,
        algorithm=config.server_algorithm,
        cycles=_cycles_for(domains, config.datasets, seed, "base_pretrain"),
        per_round=config.base_samples_per_round,
        epochs=config.base_local_epochs,
        lr=config.base_client_lr,
    )
    return [weight.copy() for weight in factory(flat).W]


def _adaptation_config(config: StudyConfig, datasets: tuple[str, ...]) -> ExperimentConfig:
    return ExperimentConfig(
        datasets=datasets,
        rounds=config.adaptation_rounds,
        samples_per_round=config.adaptation_samples_per_round,
        local_epochs=config.adaptation_local_epochs,
        learning_rate=config.adaptation_client_lr,
        fedadam_eta=config.fedadam_eta,
        sparsity_k=config.sparsity_k,
    )


def train_lora_scope(
    domains,
    datasets: tuple[str, ...],
    base_weights: list[np.ndarray],
    config: StudyConfig,
    seed: int,
    *,
    noise_sigma: float,
    name: str,
) -> ABLoRA:
    training_config = _adaptation_config(config, datasets)
    flat, factory, _ = federated_train_lora(
        dims=LARGER_DIMS,
        rank=config.lora_rank,
        base_weights=base_weights,
        train_a=False,
        alpha=config.lora_alpha,
        datasets=datasets,
        domains=domains,
        config=training_config,
        seed=seed,
        algorithm=config.server_algorithm,
        cycles=_cycles_for(domains, datasets, seed, name),
        noise_sigma=noise_sigma,
        firmware_compression=config.firmware_compression,
        sparsity_k=config.sparsity_k,
    )
    return factory(flat)


def train_dense_scope(
    domains,
    datasets: tuple[str, ...],
    base_weights: list[np.ndarray],
    config: StudyConfig,
    seed: int,
    *,
    name: str,
) -> DenseAE:
    training_config = _adaptation_config(config, datasets)
    flat, factory, _ = federated_train_dense(
        dims=LARGER_DIMS,
        base_weights=base_weights,
        datasets=datasets,
        domains=domains,
        config=training_config,
        seed=seed,
        algorithm=config.server_algorithm,
        cycles=_cycles_for(domains, datasets, seed, name),
        per_round=config.adaptation_samples_per_round,
        epochs=config.adaptation_local_epochs,
        lr=config.adaptation_client_lr,
    )
    return factory(flat)


def train_candidates(domains, base_weights, config: StudyConfig, seed: int):
    """Train shared and dataset-specific candidates for one seed."""
    candidates: dict[str, dict[str, object]] = {
        "untrained": {
            "model": DenseAE(LARGER_DIMS),
            "scope": "none",
            "learned": False,
        },
        "frozen_pretrained": {
            "model": DenseAE(LARGER_DIMS, weights=base_weights),
            "scope": "all_clients_base_pretraining",
            "learned": True,
        },
    }

    all_datasets = tuple(config.datasets)
    candidates["lora_generalized"] = {
        "model": train_lora_scope(
            domains, all_datasets, base_weights, config, seed,
            noise_sigma=0.0, name="lora_generalized",
        ),
        "scope": "all_clients",
        "learned": True,
    }
    candidates["lora_denoising_generalized"] = {
        "model": train_lora_scope(
            domains, all_datasets, base_weights, config, seed,
            noise_sigma=config.denoising_sigma,
            name="lora_denoising_generalized",
        ),
        "scope": "all_clients",
        "learned": True,
    }
    candidates["dense_generalized"] = {
        "model": train_dense_scope(
            domains, all_datasets, base_weights, config, seed,
            name="dense_generalized",
        ),
        "scope": "all_clients",
        "learned": True,
    }

    for dataset in config.datasets:
        scope = (dataset,)
        candidates[f"lora_specialized:{dataset}"] = {
            "model": train_lora_scope(
                domains, scope, base_weights, config, seed,
                noise_sigma=0.0, name=f"lora_specialized:{dataset}",
            ),
            "scope": dataset,
            "learned": True,
        }
        candidates[f"lora_denoising_specialized:{dataset}"] = {
            "model": train_lora_scope(
                domains, scope, base_weights, config, seed,
                noise_sigma=config.denoising_sigma,
                name=f"lora_denoising_specialized:{dataset}",
            ),
            "scope": dataset,
            "learned": True,
        }
        candidates[f"dense_specialized:{dataset}"] = {
            "model": train_dense_scope(
                domains, scope, base_weights, config, seed,
                name=f"dense_specialized:{dataset}",
            ),
            "scope": dataset,
            "learned": True,
        }
    return candidates


def _robust_scale(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.float64)
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    scale = max(1.4826 * mad, float(np.std(reference, ddof=1)), 1e-8)
    return ((np.asarray(values, dtype=np.float64) - median) / scale).astype(np.float64)


def magnitude_scores(features: np.ndarray) -> np.ndarray:
    """Tiny high-SNR safety score: mean squared online Z magnitude."""
    values = np.asarray(features, dtype=np.float32)
    return np.mean(values * values, axis=1).astype(np.float32)


def _fused_scores(
    model,
    calibration: np.ndarray,
    values: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    calibration_ae = model.losses(calibration)
    values_ae = model.losses(values)
    calibration_magnitude = magnitude_scores(calibration)
    values_magnitude = magnitude_scores(values)
    calibration_fused = (
        weight * _robust_scale(calibration_ae, calibration_ae)
        + (1.0 - weight)
        * _robust_scale(calibration_magnitude, calibration_magnitude)
    )
    values_fused = (
        weight * _robust_scale(calibration_ae, values_ae)
        + (1.0 - weight)
        * _robust_scale(calibration_magnitude, values_magnitude)
    )
    return calibration_fused, values_fused


def _candidate_names_for_dataset(dataset: str) -> tuple[str, ...]:
    return (
        "untrained",
        "frozen_pretrained",
        "lora_generalized",
        "lora_denoising_generalized",
        "dense_generalized",
        f"lora_specialized:{dataset}",
        f"lora_denoising_specialized:{dataset}",
        f"dense_specialized:{dataset}",
    )


def evaluate_candidates(domains, candidates, config: StudyConfig, seed: int):
    """Select fusion/threshold on validation, then evaluate held-out test once."""
    run_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for dataset in config.datasets:
        for client in CLIENTS:
            domain = domains[dataset][client]
            calibration = np.asarray(domain["calibration"], dtype=np.float32)
            validation = np.asarray(domain["val"], dtype=np.float32)
            test = np.asarray(domain["test"], dtype=np.float32)
            validation_labels = np.asarray(domain["val_labels"], dtype=np.int8)
            test_labels = np.asarray(domain["labels"], dtype=np.int8)
            best: tuple[tuple[float, float, float, int], dict[str, object]] | None = None

            for candidate_name in _candidate_names_for_dataset(dataset):
                candidate = candidates[candidate_name]
                model = candidate["model"]
                for fusion_weight in FUSION_WEIGHTS:
                    calibration_scores, validation_scores = _fused_scores(
                        model, calibration, validation, fusion_weight
                    )
                    threshold_method, threshold, validation_metrics = choose_threshold(
                        calibration_scores, validation_scores, validation_labels
                    )
                    _, test_scores = _fused_scores(
                        model, calibration, test, fusion_weight
                    )
                    test_metrics = classification_metrics(
                        test_labels, test_scores, threshold
                    )
                    row: dict[str, object] = {
                        "seed": seed,
                        "dataset": dataset,
                        "client": client,
                        "client_name": DISPLAY_NAMES[(dataset, client)],
                        "candidate": candidate_name,
                        "training_scope": candidate["scope"],
                        "learned": candidate["learned"],
                        "fusion_weight_ae": fusion_weight,
                        "fusion_weight_magnitude": 1.0 - fusion_weight,
                        "threshold_method": threshold_method,
                        "threshold": threshold,
                        "validation_f1": float(validation_metrics["f1"]),
                        "validation_fpr": float(validation_metrics["fpr"]),
                        "validation_recall": float(validation_metrics["recall"]),
                        **{key: test_metrics[key] for key in (
                            "tp", "fp", "fn", "tn", "precision", "recall",
                            "f1", "fpr", "roc_auc", "pr_auc"
                        )},
                    }
                    run_rows.append(row)
                    selection_key = (
                        float(validation_metrics["f1"]),
                        -float(validation_metrics["fpr"]),
                        float(validation_metrics["recall"]),
                        int(candidate["learned"]),
                    )
                    if best is None or selection_key > best[0]:
                        best = (selection_key, row)

            assert best is not None
            selected = dict(best[1])
            selected["selection"] = "validation_winner"
            selected_rows.append(selected)
    return run_rows, selected_rows


def summarize_selected(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["client"])), []).append(row)
    summary: list[dict[str, object]] = []
    for (dataset, client), group in sorted(groups.items()):
        item: dict[str, object] = {
            "dataset": dataset,
            "client": client,
            "client_name": DISPLAY_NAMES[(dataset, client)],
            "seeds": len(group),
            "selected_candidates": ",".join(
                sorted({str(row["candidate"]) for row in group})
            ),
        }
        for metric in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"):
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = float(mean(values))
            item[f"{metric}_std"] = float(pstdev(values))
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    path: Path,
    config: StudyConfig,
    summary: list[dict[str, object]],
) -> None:
    lines = [
        "# Improved Comprehensive Federated-AE Results",
        "",
        "All model/fusion/threshold choices were made on validation data. Test labels were used only for the final metrics.",
        "",
        "| Client | Selected model(s) across seeds | F1 | FPR | Recall | AUROC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['client_name']} | {row['selected_candidates']} | "
            f"{100*float(row['f1_mean']):.2f}% +/- {100*float(row['f1_std']):.2f} | "
            f"{100*float(row['fpr_mean']):.2f}% | "
            f"{100*float(row['recall_mean']):.2f}% | "
            f"{100*float(row['roc_auc_mean']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Scientific safeguards",
            "",
            "- Online normalization uses each client's healthy warmup windows only.",
            "- Base pretraining and adaptation keep raw client windows local to the simulated client.",
            "- The score fusion uses only AE reconstruction error and mean squared online-Z magnitude.",
            "- A fusion weight of zero is an explicit magnitude-only safety fallback for easy high-SNR faults.",
            "- IMS labels are the prepared manifest labels. No samples are relabeled after seeing test scores.",
            "- The arbitrary `corr_labels[:600] = 1` result is intentionally excluded because it turns the complete 468-window IMS test split into faults and cannot support an F1 claim.",
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
    parser.add_argument("--warmup-windows", type=int, default=1000)
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.70,
        help=(
            "maximum fraction of each client's healthy training windows used "
            "for online-normalization warmup (default: 0.70)"
        ),
    )
    parser.add_argument("--base-rounds", type=int, default=20)
    parser.add_argument("--adaptation-rounds", type=int, default=20)
    parser.add_argument("--samples-per-round", type=int, default=60)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--client-lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--fedadam-eta", type=float, default=0.005)
    parser.add_argument("--denoising-sigma", type=float, default=0.05)
    parser.add_argument(
        "--firmware-compression", action="store_true",
        help="apply Top-K 32 and signed-int8 to LoRA updates/global broadcasts",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=SCRIPT_DIR / "results" / "improved_comprehensive",
    )
    args = parser.parse_args()

    if args.warmup_windows < 2:
        parser.error("--warmup-windows must be at least 2")
    if not 0.0 < args.warmup_fraction < 1.0:
        parser.error("--warmup-fraction must be strictly between 0 and 1")
    for name in ("base_rounds", "adaptation_rounds", "samples_per_round", "local_epochs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    config = StudyConfig(
        datasets=tuple(args.datasets),
        seeds=tuple(args.seeds),
        warmup_windows=args.warmup_windows,
        warmup_fraction=args.warmup_fraction,
        base_rounds=args.base_rounds,
        base_samples_per_round=args.samples_per_round,
        base_local_epochs=args.local_epochs,
        base_client_lr=args.client_lr,
        adaptation_rounds=args.adaptation_rounds,
        adaptation_samples_per_round=args.samples_per_round,
        adaptation_local_epochs=args.local_epochs,
        adaptation_client_lr=args.client_lr,
        lora_alpha=args.alpha,
        denoising_sigma=args.denoising_sigma,
        fedadam_eta=args.fedadam_eta,
        firmware_compression=args.firmware_compression,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prepared datasets and building warmup-only normalization...")
    pretrain_domains, adapt_domains, normalization_rows = load_online_domains(config)
    # Write the partition audit before any training. If a later stage fails, the
    # exact warmup/adaptation counts and statistics remain available for review.
    write_csv(output_dir / "normalization_audit.csv", normalization_rows)
    all_runs: list[dict[str, object]] = []
    all_selected: list[dict[str, object]] = []

    for seed in config.seeds:
        print(f"\n=== Seed {seed}: federated base pretraining ===")
        started = time.perf_counter()
        base_weights = pretrain_shared_base(pretrain_domains, config, seed)
        save_base_npz(
            output_dir / f"pretrained_base_seed_{seed}.npz",
            LARGER_DIMS,
            base_weights,
            {
                "seed": seed,
                "datasets": list(config.datasets),
                "normalization": "per-client warmup-only Welford",
                "config": asdict(config),
            },
        )
        print(f"Base ready in {time.perf_counter() - started:.1f}s")
        print("Training specialized/generalized standard and denoising candidates...")
        candidates = train_candidates(adapt_domains, base_weights, config, seed)
        seed_runs, seed_selected = evaluate_candidates(
            adapt_domains, candidates, config, seed
        )
        all_runs.extend(seed_runs)
        all_selected.extend(seed_selected)

    summary = summarize_selected(all_selected)
    write_csv(output_dir / "candidate_validation_and_test_runs.csv", all_runs)
    write_csv(output_dir / "validation_selected_runs.csv", all_selected)
    write_csv(output_dir / "improved_comprehensive_summary.csv", summary)
    payload = {
        "study": "Improved comprehensive federated autoencoder experiment",
        "config": asdict(config),
        "selection_policy": (
            "candidate, AE/magnitude fusion weight, and calibration threshold "
            "method selected using validation F1/FPR/recall only"
        ),
        "summary": summary,
        "selected_runs": all_selected,
        "limitations": [
            "PC simulation does not measure physical radio, energy, SD, or timing reliability.",
            "Federated learning keeps raw samples local but does not by itself guarantee update privacy.",
            "Feature insufficiency may remain for subtle faults; no test-driven relabeling is allowed.",
        ],
    }
    (output_dir / "improved_comprehensive_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    build_report(output_dir / "IMPROVED_COMPREHENSIVE_REPORT.md", config, summary)

    print("\nExperiment complete")
    print(f"  Output directory: {output_dir}")
    print(f"  Report: {output_dir / 'IMPROVED_COMPREHENSIVE_REPORT.md'}")


if __name__ == "__main__":
    main()
