"""Comparative evaluation of lightweight and original models, centralized, local, and baseline detectors.

Performs threshold ablation on validation data, and then compares metrics on the held-out test split.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from cwru_eval import load_domain, classification_metrics, threshold_from_calibration
from tiny_ae import (
    _lcg_values,
    _round_away_from_zero,
    quantize_dynamic,
    dequantize_dynamic,
    BASE_SEED,
    LORA_A_SEED,
    LORA_A_SCALE,
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
    ADAM_WEIGHT_DECAY,
    INPUT_SIZE,
)

# Constants
DEFAULT_LR = 0.02
LOCAL_EPOCHS = 8
TRAIN_SAMPLES_PER_ROUND = 30
TARGET_FEDERATED_ROUNDS = 3
LORA_ALPHA = np.float32(0.5)
SPARSITY_K = 32


class GeneralLoRAAutoencoder:
    """Generalized LoRA Autoencoder with arbitrary architecture shape."""
    def __init__(self, dims: tuple[int, ...], rank: int) -> None:
        self.dims = dims
        self.rank = rank
        self.num_layers = len(dims) - 1
        
        # Frozen base W and LoRA A initialization
        self.W = []
        state = BASE_SEED
        scales = (0.5, 0.5, 0.75, 1.0, 0.75, 0.5)
        for in_dim, out_dim, scale in zip(dims[:-1], dims[1:], scales[:self.num_layers]):
            values, state = _lcg_values(state, out_dim * in_dim, scale)
            self.W.append(values.reshape(out_dim, in_dim))

        self.A = []
        state = LORA_A_SEED
        for in_dim in dims[:-1]:
            values, state = _lcg_values(state, rank * in_dim, LORA_A_SCALE)
            self.A.append(values.reshape(rank, in_dim))
            
        # Trainable B weights
        self.B = [
            np.zeros((out_dim, rank), dtype=np.float32)
            for out_dim in dims[1:]
        ]
        self.m = [np.zeros_like(matrix) for matrix in self.B]
        self.v = [np.zeros_like(matrix) for matrix in self.B]
        self.t = 0
        
    def get_weights(self) -> np.ndarray:
        return np.concatenate([matrix.ravel() for matrix in self.B]).astype(np.float32)

    def set_weights(self, flat_weights: np.ndarray, reset_optimizer: bool = False) -> None:
        offset = 0
        for index, matrix in enumerate(self.B):
            size = matrix.size
            self.B[index] = flat_weights[offset : offset + size].reshape(matrix.shape).copy()
            offset += size
        if reset_optimizer:
            self.m = [np.zeros_like(matrix) for matrix in self.B]
            self.v = [np.zeros_like(matrix) for matrix in self.B]
            self.t = 0

    def forward(self, sample: np.ndarray, with_cache: bool = False):
        activations = [sample]
        preactivations = []
        projections = []
        current = sample
        for layer, (base, proj_a, proj_b) in enumerate(zip(self.W, self.A, self.B)):
            low_rank_input = proj_a @ current
            z = base @ current + LORA_ALPHA * (proj_b @ low_rank_input)
            current = z if layer == self.num_layers - 1 else np.maximum(z, 0.0)
            projections.append(low_rank_input)
            preactivations.append(z)
            activations.append(current)

        loss = float(np.mean((current - sample) ** 2))
        if with_cache:
            class ForwardResult:
                def __init__(self, loss, activations, preactivations, projections):
                    self.loss = loss
                    self.activations = activations
                    self.preactivations = preactivations
                    self.projections = projections
            return ForwardResult(loss, activations, preactivations, projections)
        return loss

    def train_step(self, sample: np.ndarray, learning_rate: float = DEFAULT_LR) -> float:
        cache = self.forward(sample, with_cache=True)
        self.t += 1
        lr = np.float32(learning_rate)

        delta = np.float32(2.0 / INPUT_SIZE) * (cache.activations[-1] - cache.activations[0])
        gradients = [None] * self.num_layers
        for layer in range(self.num_layers - 1, -1, -1):
            gradients[layer] = LORA_ALPHA * np.outer(delta, cache.projections[layer])
            if layer == 0:
                break
            effective = self.W[layer] + LORA_ALPHA * (self.B[layer] @ self.A[layer])
            delta = effective.T @ delta
            delta = delta * (cache.preactivations[layer - 1] > 0.0)
            
        # AdamW updates
        for layer in range(self.num_layers):
            g = np.clip(gradients[layer], -1.0, 1.0).astype(np.float32)
            self.m[layer] = ADAM_BETA1 * self.m[layer] + (1.0 - ADAM_BETA1) * g
            self.v[layer] = ADAM_BETA2 * self.v[layer] + (1.0 - ADAM_BETA2) * g * g
            m_hat = self.m[layer] / (1.0 - float(ADAM_BETA1) ** self.t)
            v_hat = self.v[layer] / (1.0 - float(ADAM_BETA2) ** self.t)
            update = m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)
            self.B[layer] -= lr * (update + ADAM_WEIGHT_DECAY * self.B[layer])
        return cache.loss

    def train(self, samples: np.ndarray, epochs: int = 1, learning_rate: float = DEFAULT_LR) -> list[float]:
        losses = []
        for _ in range(epochs):
            epoch_losses = [self.train_step(sample, learning_rate) for sample in samples]
            losses.append(float(np.mean(epoch_losses)))
        return losses

    def losses(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray([self.forward(sample) for sample in samples], dtype=np.float32)


# Simulation aggregator matching server
class ServerFedAdam:
    def __init__(self, num_weights: int) -> None:
        self.num_weights = num_weights
        self.global_weights = np.zeros(num_weights, dtype=np.float32)
        self.adam_m = np.zeros(num_weights, dtype=np.float32)
        self.adam_v = np.zeros(num_weights, dtype=np.float32)
        self.adam_step = 0

    def aggregate(self, updates: list[dict[str, object]]) -> np.ndarray:
        weighted_average = np.zeros(self.num_weights, dtype=np.float32)
        total_samples = 0
        for u in updates:
            n = u["samples"]
            total_samples += n
            weighted_average += u["weights"] * n
        weighted_average /= total_samples
        
        delta = weighted_average - self.global_weights
        self.adam_step += 1
        self.adam_m = 0.9 * self.adam_m + 0.1 * delta
        self.adam_v = 0.99 * self.adam_v + 0.01 * delta * delta
        
        m_hat = self.adam_m / (1.0 - 0.9 ** self.adam_step)
        v_hat = self.adam_v / (1.0 - 0.99 ** self.adam_step)
        self.global_weights += 0.01 * m_hat / (np.sqrt(v_hat) + 1e-3)
        
        # Broadcast quantization
        maximum = np.max(np.abs(self.global_weights))
        scale = max(float(maximum) / 127.0, 1e-7)
        qweights = np.clip(_round_away_from_zero(self.global_weights / scale), -127.0, 127.0).astype(np.int8)
        self.global_weights = qweights.astype(np.float32) * scale
        return self.global_weights


# Simulates federated learning for a specific architecture size
def run_federated_sim(dims: tuple[int, ...], rank: int, domains: dict[str, dict[str, object]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    clients = {
        "DE": GeneralLoRAAutoencoder(dims, rank),
        "FE": GeneralLoRAAutoencoder(dims, rank),
    }
    num_weights = sum(out_dim * rank for out_dim in dims[1:])
    server = ServerFedAdam(num_weights)
    
    global_model = server.global_weights.copy()
    cursors = {"DE": 0, "FE": 0}
    
    for round_idx in range(TARGET_FEDERATED_ROUNDS):
        updates = []
        for slug, client in clients.items():
            client.model = client  # general mapping
            client.set_weights(global_model, reset_optimizer=True)
            
            # Read block slice
            start = cursors[slug]
            end = start + TRAIN_SAMPLES_PER_ROUND
            train_slice = domains[slug]["train"][start:end]
            cursors[slug] = end
            
            client.train(train_slice, epochs=LOCAL_EPOCHS, learning_rate=DEFAULT_LR)
            
            diffs = client.get_weights() - global_model
            abs_diffs = np.abs(diffs)
            top_k_indices = np.argsort(-abs_diffs)[:SPARSITY_K]
            top_k_indices.sort()
            
            top_k_weights = diffs[top_k_indices]
            scale, qweights = quantize_dynamic(top_k_weights)
            dequantized = dequantize_dynamic(qweights, scale)
            
            reconstructed = global_model.copy()
            reconstructed[top_k_indices] += dequantized
            updates.append({"weights": reconstructed, "samples": len(train_slice)})
            
        global_model = server.aggregate(updates)
        
    # Evaluate calibration and test losses for each client
    cal_losses = {}
    test_losses = {}
    val_losses = {}
    for slug, client in clients.items():
        client.set_weights(global_model, reset_optimizer=True)
        cal_losses[slug] = client.losses(domains[slug]["calibration"])
        val_losses[slug] = client.losses(domains[slug]["val"])
        test_losses[slug] = client.losses(domains[slug]["test"])
        
    return cal_losses, val_losses, test_losses


def run_local_only_sim(dims: tuple[int, ...], rank: int, domains: dict[str, dict[str, object]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    cal_losses = {}
    val_losses = {}
    test_losses = {}
    for slug in ("DE", "FE"):
        client = GeneralLoRAAutoencoder(dims, rank)
        # Trains locally on all 90 blocks used by federated rounds
        train_data = domains[slug]["train"][:90]
        client.train(train_data, epochs=LOCAL_EPOCHS, learning_rate=DEFAULT_LR)
        cal_losses[slug] = client.losses(domains[slug]["calibration"])
        val_losses[slug] = client.losses(domains[slug]["val"])
        test_losses[slug] = client.losses(domains[slug]["test"])
    return cal_losses, val_losses, test_losses


def run_centralized_sim(dims: tuple[int, ...], rank: int, domains: dict[str, dict[str, object]]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Centralized model trains on combined data from both clients
    central = GeneralLoRAAutoencoder(dims, rank)
    train_data = np.vstack([domains["DE"]["train"][:90], domains["FE"]["train"][:90]])
    # Permute to represent standard centralized shuffle
    train_data = np.random.default_rng(20260714).permutation(train_data)
    central.train(train_data, epochs=LOCAL_EPOCHS, learning_rate=DEFAULT_LR)
    
    cal_losses = {}
    val_losses = {}
    test_losses = {}
    for slug in ("DE", "FE"):
        cal_losses[slug] = central.losses(domains[slug]["calibration"])
        val_losses[slug] = central.losses(domains[slug]["val"])
        test_losses[slug] = central.losses(domains[slug]["test"])
    return cal_losses, val_losses, test_losses


# Simple detector baselines
class SimpleFeatureDetector:
    def __init__(self, feature_idx: int) -> None:
        self.idx = feature_idx
    def get_scores(self, data: np.ndarray) -> np.ndarray:
        # Since data is normalized in load_domain, these are standard scaled features
        # Anomaly score is simply the feature value (or absolute value)
        return np.abs(data[:, self.idx])


# Threshold computation helper for ablation
def get_threshold(scores: np.ndarray, method: str) -> float:
    if method == "3-sigma":
        return float(np.mean(scores) + 3.0 * np.std(scores, ddof=1))
    elif method == "99th":
        return float(np.percentile(scores, 99.0))
    elif method == "99.5th":
        return float(np.percentile(scores, 99.5))
    elif method == "median_mad":
        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        scaled_mad = 1.4826 * mad
        return float(median + 3.0 * scaled_mad)
    else:
        raise ValueError(f"Unknown threshold method {method}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cwru", "xjtu", "ims", "pronostia"], default="cwru")
    args = parser.parse_args()

    print(f"Loading domains for {args.dataset.upper()} dataset...")
    bin_dir = Path(__file__).parent / f"{args.dataset}_bin"
    domains = {
        "DE": load_domain("client_a", bin_dir),
        "FE": load_domain("client_b", bin_dir),
    }
    
    # 1. THRESHOLD ABLATION ON VALIDATION SPLIT
    print("\n=======================================================")
    print("PHASE 1: Threshold Ablation on Validation Split (Lightweight Model)")
    print("=======================================================")
    
    cal_losses, val_losses, _ = run_federated_sim((12, 16, 8, 4, 8, 16, 12), 2, domains)
    
    methods = ["3-sigma", "99th", "99.5th", "median_mad"]
    best_method = "3-sigma"
    best_avg_f1 = 0.0
    
    for method in methods:
        print(f"\nThreshold Method: {method}")
        f1_sum = 0.0
        for slug in ("DE", "FE"):
            t = get_threshold(cal_losses[slug], method)
            metrics = classification_metrics(domains[slug]["val_labels"], val_losses[slug], t)
            f1_sum += metrics["f1"]
            print(f"  {slug}: Threshold={t:.6f}, F1={metrics['f1']*100:.2f}%, FPR={metrics['fpr']*100:.2f}%")
        avg_f1 = f1_sum / 2.0
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_method = method
            
    print(f"\nSelected Threshold Method for Final Evaluation: {best_method} (Avg Val F1: {best_avg_f1*100:.2f}%)")
    
    # 2. RUN AND COMPARE ARCHITECTURES ON FINAL HELD-OUT TEST SPLIT
    print("\n=======================================================")
    print("PHASE 2: Architectural Performance Comparison (Final Test Split)")
    print("=======================================================")
    
    results = {}
    
    # 2.1 Lightweight Federated
    cal, _, test = run_federated_sim((12, 16, 8, 4, 8, 16, 12), 2, domains)
    results["FedAdam-Lightweight"] = {"cal": cal, "test": test, "weights": 128, "bytes": 87 * 3}
    
    # 2.2 Original Federated
    cal, _, test = run_federated_sim((12, 32, 16, 8, 16, 32, 12), 4, domains)
    # Original rank 4 model has sum(out_dim * 4) = 464 weights
    results["FedAdam-Original"] = {"cal": cal, "test": test, "weights": 464, "bytes": 285 * 3} # Update size for 464 rank 4 requires fragmentation or larger packets
    
    # 2.3 Local-Only (Lightweight)
    cal, _, test = run_local_only_sim((12, 16, 8, 4, 8, 16, 12), 2, domains)
    results["Local-Lightweight"] = {"cal": cal, "test": test, "weights": 128, "bytes": 0}
    
    # 2.4 Centralized (Lightweight)
    cal, _, test = run_centralized_sim((12, 16, 8, 4, 8, 16, 12), 2, domains)
    results["Centralized"] = {"cal": cal, "test": test, "weights": 128, "bytes": 0}
    
    # 2.5 Simple Baselines (RMS & Peak-to-Peak)
    rms_det = SimpleFeatureDetector(0) # RMS
    p2p_det = SimpleFeatureDetector(1) # Peak-to-Peak
    
    results["Baseline-RMS"] = {
        "cal": {slug: rms_det.get_scores(domains[slug]["calibration"]) for slug in ("DE", "FE")},
        "test": {slug: rms_det.get_scores(domains[slug]["test"]) for slug in ("DE", "FE")},
        "weights": 0, "bytes": 0
    }
    results["Baseline-P2P"] = {
        "cal": {slug: p2p_det.get_scores(domains[slug]["calibration"]) for slug in ("DE", "FE")},
        "test": {slug: p2p_det.get_scores(domains[slug]["test"]) for slug in ("DE", "FE")},
        "weights": 0, "bytes": 0
    }

    # Evaluate all models on final test split for both 3-sigma and median_mad methods
    for method in ("3-sigma", "median_mad"):
        summary = []
        for name, r in results.items():
            for slug in ("DE", "FE"):
                t = get_threshold(r["cal"][slug], method)
                m = classification_metrics(domains[slug]["labels"], r["test"][slug], t)
                summary.append({
                    "Model": name,
                    "Domain": slug,
                    "F1-Score (%)": f"{m['f1']*100:.2f}%",
                    "Precision (%)": f"{m['precision']*100:.2f}%",
                    "Recall (%)": f"{m['recall']*100:.2f}%",
                    "FPR (%)": f"{m['fpr']*100:.2f}%",
                    "Weights": r["weights"],
                    "Tx Bytes": r["bytes"],
                })
                
        # Format table for console output
        print(f"\nComparative Architecture Results (Threshold Method: {method}):")
        print(f"{'Model':<22} | {'Domain':<6} | {'F1':<8} | {'Precision':<10} | {'Recall':<8} | {'FPR':<8} | {'Weights':<8} | {'Tx Bytes':<8}")
        print("-" * 92)
        for row in summary:
            print(f"{row['Model']:<22} | {row['Domain']:<6} | {row['F1-Score (%)']:<8} | {row['Precision (%)']:<10} | {row['Recall (%)']:<8} | {row['FPR (%)']:<8} | {row['Weights']:<8} | {row['Tx Bytes']:<8}")

        # Write comparative metrics to output
        output_path = Path(__file__).parent / f"architecture_comparison_{method}_{args.dataset}.json"
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Comparative metrics written to {output_path}")


if __name__ == "__main__":
    main()
