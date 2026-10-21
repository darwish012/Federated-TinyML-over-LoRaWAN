"""Federated Autoencoder Learning-Rescue Experiment (PC-only).

This is a SEPARATE experiment runner that sits beside ``run_full_simulation.py``.
It does NOT modify the existing simulator or its results. It reuses the existing
prepared-data loader (``cwru_eval.load_domain``) and leakage validation
(``validate_domain`` imported from the full simulator) so it reads only the
prepared ``*_bin`` folders and never opens raw dataset archives.

It diagnoses why federated LoRA training currently produces almost no change, by
asking one primary question:

    Can training produce a repeatable improvement over the matching untrained
    model on the four difficult clients (CWRU FE, IMS Odd, IMS Even, PRONOSTIA
    FEMTO 1_2)?

The CLI mirrors the rescue plan:

    python learning_failure_experiments.py --stage all \\
        --seeds 11,29,47,71,97 --datasets cwru,xjtu,ims,pronostia \\
        --output-dir results/learning_failure

Supported stages: diagnostics, base_pretraining, lora_ablation, optimization,
denoising, drift, all (plus ``checks`` which runs the built-in correctness
tests). It writes:

    learning_diagnostics.csv
    learning_ablation_runs.csv
    learning_ablation_summary.csv
    pretrained_base.npz
    LEARNING_FAILURE_REPORT.md
    learning_failure_decisions.json

Run the standalone acceptance tests with ``python -m unittest learning_failure_tests``
from this directory, or ``python learning_failure_experiments.py --stage checks``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable, Sequence

import numpy as np

# --- locate shared python_tools so we reuse the existing prepared-data loader --
SCRIPT_DIR = Path(__file__).resolve().parent
RELEASE_DIR = SCRIPT_DIR.parent
PYTHON_TOOLS_DIR = RELEASE_DIR / "python_tools"
if str(PYTHON_TOOLS_DIR) not in sys.path:  # pragma: no cover - path setup
    sys.path.insert(0, str(PYTHON_TOOLS_DIR))

from compare_architectures import GeneralLoRAAutoencoder  # noqa: E402
from cwru_eval import classification_metrics, load_domain  # noqa: E402
from run_full_simulation import (  # noqa: E402
    BIGGER_DIMS,
    CLIENTS,
    CLIENT_DISPLAY_NAMES,
    DATASETS,
    DenseAutoencoder,
    ExperimentConfig,
    client_key,
    choose_threshold,
    evaluate_scores,
    load_all_domains,
    ordered_stack,
    stable_seed,
    validate_domain,
)
from tiny_ae import (  # noqa: E402
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
    ADAM_WEIGHT_DECAY,
    BASE_SEED,
    LORA_A_SCALE,
    LORA_A_SEED,
    _lcg_values,
    dequantize_dynamic,
    quantize_dynamic,
)


# --------------------------------------------------------------------------- #
# Architecture and the set of "difficult" clients the rescue targets.
# --------------------------------------------------------------------------- #
LARGER_DIMS = BIGGER_DIMS  # 12->24->12->4->12->24->12 (ESP32-feasible per plan)
DEFAULT_RANK = 1            # rank-1 initially (per rescue plan)
INPUT_SIZE = 12

# Difficult-client group from the plan.
DIFFICULT_CLIENTS = {
    ("cwru", "client_b"): "CWRU FE",
    ("ims", "client_a"): "IMS Odd",
    ("ims", "client_b"): "IMS Even",
    ("pronostia", "client_b"): "PRONOSTIA FEMTO 1_2",
}

NUM_BASE_SCALES = (0.5, 0.5, 0.75, 1.0, 0.75, 0.5)
STAGES = ("diagnostics", "base_pretraining", "lora_ablation",
          "optimization", "denoising", "drift")

SCREENING_SEED = 11
ALL_SEEDS = (11, 29, 47, 71, 97)

# Firmware packet budget used by the deployment gate.
MAX_LORA_PACKET_BYTES = 255


# --------------------------------------------------------------------------- #
# Functional core: forward + analytic gradients, dtype-parametric.
# --------------------------------------------------------------------------- #
def _ae_forward(
    sample: np.ndarray,
    weights: Sequence[np.ndarray],
    *,
    target: np.ndarray | None = None,
    lora_a: Sequence[np.ndarray] | None = None,
    lora_b: Sequence[np.ndarray] | None = None,
    lora_alpha: float = 0.0,
    dtype=np.float32,
):
    """Functional forward pass shared by all dense and LoRA models.

    ``weights`` are the dense base W matrices. When ``lora_a``/``lora_b`` are
    supplied the effective weight of layer ``i`` is ``W[i] + alpha * (B[i]@A[i])``.
    Returns ``(loss, activations, preactivations, projections)`` so the same
    routine powers training and the analytic backward pass. ``dtype`` defaults to
    float32 (training) but float64 is used by the gradient-check tests.
    """
    x = np.asarray(sample, dtype=dtype)
    activations = [x]
    preactivations: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    current = x
    num_layers = len(weights)
    last_layer = num_layers - 1
    use_lora = lora_a is not None and lora_b is not None and lora_alpha != 0.0
    for layer in range(num_layers):
        z = weights[layer] @ current
        if use_lora:
            low_rank = lora_a[layer] @ current
            z = z + np.asarray(lora_alpha, dtype=dtype) * (lora_b[layer] @ low_rank)
            projections.append(low_rank)
        else:
            projections.append(np.zeros(weights[layer].shape[0], dtype=dtype))
        current = z if layer == last_layer else np.maximum(z, np.zeros((), dtype=dtype))
        current = np.asarray(current, dtype=dtype)
        preactivations.append(z)
        activations.append(current)
    reconstruction_target = (
        activations[0] if target is None else np.asarray(target, dtype=dtype)
    )
    if reconstruction_target.shape != activations[-1].shape:
        raise ValueError(
            "autoencoder target must have the same shape as the reconstruction"
        )
    diff = activations[-1] - reconstruction_target
    loss = float(np.asarray(np.mean(diff * diff), dtype=dtype))
    return loss, activations, preactivations, projections


def _analytic_gradients(
    sample: np.ndarray,
    weights: Sequence[np.ndarray],
    *,
    target: np.ndarray | None = None,
    lora_a: Sequence[np.ndarray] | None = None,
    lora_b: Sequence[np.ndarray] | None = None,
    lora_alpha: float = 0.0,
    dtype=np.float32,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Analytic gradients for W, A and B for one sample (mean-squared error).

    Returns ``(grad_W, grad_A, grad_B)``. When LoRA is disabled (the dense-only
    case) ``grad_A`` and ``grad_B`` are zero-shaped placeholders. ``grad_W`` is
    always returned so a frozen-W LoRA model reports zero base gradients.
    """
    loss, activations, preactivations, projections = _ae_forward(
        sample, weights, target=target, lora_a=lora_a, lora_b=lora_b,
        lora_alpha=lora_alpha,
        dtype=dtype,
    )
    num_layers = len(weights)
    reconstruction_target = (
        activations[0] if target is None else np.asarray(target, dtype=dtype)
    )
    delta = np.asarray(2.0 / reconstruction_target.size, dtype=dtype) * (
        activations[-1] - reconstruction_target
    )
    grad_W: list[np.ndarray] = [
        np.zeros_like(weights[i], dtype=dtype) for i in range(num_layers)
    ]
    grad_B_rev: list[np.ndarray] = []
    grad_A_rev: list[np.ndarray] = []
    use_lora = lora_a is not None and lora_b is not None and lora_alpha != 0.0
    for layer in range(num_layers - 1, -1, -1):
        # d loss / d W[layer] = outer(delta, input to layer)
        grad_W[layer] = np.outer(delta, activations[layer]).astype(dtype)
        if use_lora:
            # z = W x + alpha * B (A x)  ->  d z / d B = alpha * (A x) outer(delta)
            grad_B_rev.append(
                (np.asarray(lora_alpha, dtype=dtype)
                 * np.outer(delta, projections[layer])).astype(dtype)
            )
            # d z / d A = alpha * B^T delta  outer(x)
            grad_A_rev.append(
                (np.asarray(lora_alpha, dtype=dtype)
                 * np.outer(lora_b[layer].T @ delta, activations[layer])).astype(dtype)
            )
        if layer > 0:
            effective = weights[layer].astype(dtype)
            if use_lora:
                effective = effective + np.asarray(lora_alpha, dtype=dtype) * (
                    lora_b[layer] @ lora_a[layer]
                )
            delta = effective.T @ delta
            delta = delta * (preactivations[layer - 1] > 0.0)
            delta = np.asarray(delta, dtype=dtype)
    grad_B = list(reversed(grad_B_rev))
    grad_A = list(reversed(grad_A_rev))
    if not use_lora:
        grad_A = [np.zeros_like(a, dtype=dtype) for a in lora_a] if lora_a is not None else []
        grad_B = [np.zeros_like(b, dtype=dtype) for b in lora_b] if lora_b is not None else []
    return grad_W, grad_A, grad_B


# --------------------------------------------------------------------------- #
# Models. Each exposes dense W init, flat get/set of its trainable tensors,
# a losses() scorer, and a train_step(). They share the functional core above.
# --------------------------------------------------------------------------- #
def _init_base_weights(dims: tuple[int, ...]) -> list[np.ndarray]:
    """Firmware-equivalent LCG base-W init (deterministic, matches TinyLoRA)."""
    weights: list[np.ndarray] = []
    state = BASE_SEED
    for in_dim, out_dim, scale in zip(
        dims[:-1], dims[1:], NUM_BASE_SCALES[: len(dims) - 1]
    ):
        values, state = _lcg_values(state, out_dim * in_dim, scale)
        weights.append(values.reshape(out_dim, in_dim).astype(np.float32))
    return weights


def _init_lora_a(dims: tuple[int, ...], rank: int) -> list[np.ndarray]:
    """LoRA A factors using the firmware LORA_A_SEED deterministic draws."""
    projections: list[np.ndarray] = []
    state = LORA_A_SEED
    for in_dim in dims[:-1]:
        values, state = _lcg_values(state, rank * in_dim, LORA_A_SCALE)
        projections.append(values.reshape(rank, in_dim).astype(np.float32))
    return projections


@dataclass
class _AdamState:
    """Per-tensor AdamW state mirrored on the firmware optimizer."""

    shapes: tuple[tuple[int, ...], ...]
    m: list[np.ndarray] = field(default_factory=list)
    v: list[np.ndarray] = field(default_factory=list)
    t: int = 0

    @classmethod
    def for_tensors(cls, tensors: Sequence[np.ndarray]) -> "_AdamState":
        return cls(shapes=tuple(t.shape for t in tensors))

    def reset(self) -> None:
        self.m = [np.zeros(s, dtype=np.float32) for s in self.shapes]
        self.v = [np.zeros(s, dtype=np.float32) for s in self.shapes]
        self.t = 0

    def step(self, tensors: list[np.ndarray], grads: list[np.ndarray],
             lr: float) -> list[np.ndarray]:
        if not self.m:
            self.m = [np.zeros_like(g) for g in grads]
            self.v = [np.zeros_like(g) for g in grads]
        self.t += 1
        out = []
        for i, (t, g) in enumerate(zip(tensors, grads)):
            g = np.clip(g, -1.0, 1.0).astype(np.float32)
            self.m[i] = ADAM_BETA1 * self.m[i] + (1.0 - ADAM_BETA1) * g
            self.v[i] = ADAM_BETA2 * self.v[i] + (1.0 - ADAM_BETA2) * (g * g)
            m_hat = self.m[i] / (1.0 - float(ADAM_BETA1) ** self.t)
            v_hat = self.v[i] / (1.0 - float(ADAM_BETA2) ** self.t)
            update = m_hat / (np.sqrt(v_hat) + ADAM_EPSILON)
            new_t = t.astype(np.float32) - np.float32(lr) * (
                update + ADAM_WEIGHT_DECAY * t.astype(np.float32)
            )
            out.append(new_t.astype(np.float32))
        return out


class DenseAE:
    """Full-weight (trainable W) bias-free autoencoder. Trains every dense W."""

    def __init__(self, dims: tuple[int, ...], weights: list[np.ndarray] | None = None):
        self.dims = dims
        self.num_layers = len(dims) - 1
        self.W = [w.copy() for w in (weights if weights is not None else _init_base_weights(dims))]
        self._opt = _AdamState.for_tensors(self.W)

    # trainable tensor plumbing -------------------------------------------------
    def trainable_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(w.shape for w in self.W)

    def get_trainable(self) -> np.ndarray:
        return np.concatenate([w.ravel() for w in self.W]).astype(np.float32)

    def set_trainable(self, flat: np.ndarray, reset_optimizer: bool = False) -> None:
        flat = np.asarray(flat, dtype=np.float32)
        offset = 0
        for i, w in enumerate(self.W):
            self.W[i] = flat[offset:offset + w.size].reshape(w.shape).copy()
            offset += w.size
        if reset_optimizer:
            self._opt.reset()

    def parameter_count(self) -> int:
        return sum(w.size for w in self.W)

    # inference / training ------------------------------------------------------
    def losses(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray([_ae_forward(s, self.W)[0] for s in samples], dtype=np.float32)

    def train_step(self, sample: np.ndarray, lr: float) -> float:
        loss, _, _, _ = _ae_forward(sample, self.W)
        grad_W, _, _ = _analytic_gradients(sample, self.W)
        self.W = self._opt.step(self.W, grad_W, lr)
        return loss

    def train(self, samples: np.ndarray, epochs: int, lr: float) -> list[float]:
        losses = []
        for _ in range(epochs):
            losses.append(float(np.mean([self.train_step(s, lr) for s in samples])))
        return losses


class ABLoRA:
    """LoRA adapter over a FROZEN base: trains A and/or B at ``rank``.

    ``train_a`` selects whether A is also updated. Standard LoRA init is used:
    small deterministic random A, zero B, scaling ``alpha/rank``. ``base_weights``
    are frozen by reference (never updated by this model).
    """

    def __init__(
        self,
        dims: tuple[int, ...],
        rank: int,
        base_weights: list[np.ndarray],
        *,
        train_a: bool,
        alpha: float = 1.0,
        lora_a: list[np.ndarray] | None = None,
        lora_b: list[np.ndarray] | None = None,
    ):
        self.dims = dims
        self.rank = rank
        self.num_layers = len(dims) - 1
        self.W = base_weights  # frozen, shared by reference
        self.train_a = train_a
        self.alpha = np.float32(float(alpha) / float(rank))
        self.A = [a.copy() for a in (lora_a if lora_a is not None else _init_lora_a(dims, rank))]
        self.B = [b.copy() for b in (lora_b if lora_b is not None
                                     else [np.zeros((out, rank), dtype=np.float32)
                                           for out in dims[1:]])]
        # Size optimizer state to exactly the trainable tensors.  The previous
        # B-only path rebuilt Adam on every sample, silently discarding momentum
        # and bias-correction history.  Keeping one persistent B optimizer is
        # required for the firmware-equivalent AdamW update.
        self._opt = _AdamState.for_tensors(self.trainable_tensors)

    @property
    def trainable_tensors(self) -> list[np.ndarray]:
        return (self.A + self.B) if self.train_a else self.B

    def trainable_shapes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(t.shape for t in self.trainable_tensors)

    def get_trainable(self) -> np.ndarray:
        return np.concatenate([t.ravel() for t in self.trainable_tensors]).astype(np.float32)

    def set_trainable(self, flat: np.ndarray, reset_optimizer: bool = False) -> None:
        flat = np.asarray(flat, dtype=np.float32)
        offset = 0
        for t in self.trainable_tensors:
            n = t.size
            t[...] = flat[offset:offset + n].reshape(t.shape)
            offset += n
        if reset_optimizer:
            self._opt.reset()

    def parameter_count(self) -> int:
        return sum(t.size for t in self.trainable_tensors)

    def lora_only_parameter_count(self) -> int:
        # B only, for the 255-byte packet gate
        return sum(b.size for b in self.B)

    def losses(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray(
            [_ae_forward(s, self.W, lora_a=self.A, lora_b=self.B,
                         lora_alpha=self.alpha)[0] for s in samples],
            dtype=np.float32,
        )

    def train_step(
        self,
        sample: np.ndarray,
        lr: float,
        *,
        model_input: np.ndarray | None = None,
        proximal_center: np.ndarray | None = None,
        proximal_mu: float = 0.0,
    ) -> float:
        """Train one sample, optionally with an exact FedProx parameter term.

        ``proximal_center`` is the flattened trainable adapter received from the
        server.  Adding ``mu * (theta - center)`` to the gradient before AdamW
        is the actual gradient of ``mu/2 * ||theta-center||^2``.  The previous
        real-time runner applied a correction after AdamW, which was only an
        approximation and did not participate in Adam's moment estimates.
        """
        if proximal_mu < 0.0:
            raise ValueError("proximal_mu must be non-negative")
        clean_target = np.asarray(sample, dtype=np.float32)
        actual_input = clean_target if model_input is None else np.asarray(
            model_input, dtype=np.float32
        )
        loss, _, _, _ = _ae_forward(
            actual_input,
            self.W,
            target=clean_target,
            lora_a=self.A,
            lora_b=self.B,
            lora_alpha=self.alpha,
        )
        grad_W, grad_A, grad_B = _analytic_gradients(
            actual_input,
            self.W,
            target=clean_target,
            lora_a=self.A,
            lora_b=self.B,
            lora_alpha=self.alpha,
        )
        if proximal_center is not None and proximal_mu > 0.0:
            center = np.asarray(proximal_center, dtype=np.float32).ravel()
            trainable = self.trainable_tensors
            expected = sum(tensor.size for tensor in trainable)
            if center.size != expected:
                raise ValueError(
                    f"proximal_center has {center.size} values; expected {expected}"
                )
            prox_gradients = (grad_A + grad_B) if self.train_a else grad_B
            offset = 0
            for tensor, gradient in zip(trainable, prox_gradients):
                target = center[offset : offset + tensor.size].reshape(tensor.shape)
                gradient += np.float32(proximal_mu) * (tensor - target)
                offset += tensor.size
        if self.train_a:
            self.A, self.B = self._apply_opt_ab(self.A, grad_A, self.B, grad_B, lr)
        else:
            # A frozen: backprop through A for an accurate B gradient, but only
            # update B. Optimizer state is persistently sized to the B tensors.
            self.B = self._apply_opt_b_only(self.B, grad_B, lr)
        return loss

    def _apply_opt_ab(self, A, grad_A, B, grad_B, lr):
        combined = self._opt.step(list(A) + list(B), list(grad_A) + list(grad_B), lr)
        half = len(A)
        return combined[:half], combined[half:]

    def _apply_opt_b_only(self, B, grad_B, lr):
        return self._opt.step(list(B), list(grad_B), lr)

    def train(
        self,
        samples: np.ndarray,
        epochs: int,
        lr: float,
        *,
        noise_sigma: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[float]:
        """Run ``epochs`` of mean per-sample-loss training over ``samples``."""
        if noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        if noise_sigma > 0.0 and rng is None:
            raise ValueError("a deterministic rng is required for denoising LoRA")
        losses = []
        for _ in range(epochs):
            epoch_losses = []
            for sample in samples:
                model_input = None
                if noise_sigma > 0.0:
                    model_input = sample + np.float32(noise_sigma) * rng.standard_normal(
                        sample.shape
                    ).astype(np.float32)
                epoch_losses.append(
                    self.train_step(sample, lr, model_input=model_input)
                )
            losses.append(float(np.mean(epoch_losses)))
        return losses


class DenoisingAE:
    """DenseAE wrapper that corrupts the input with Gaussian noise and trains
    the model to reconstruct the CLEAN input (denoising objective).

    Detection still scores the clean reconstruction error.
    """

    def __init__(self, dims: tuple[int, ...], sigma: float, seed: int,
                 weights: list[np.ndarray] | None = None):
        self.core = DenseAE(dims, weights=weights)
        self.sigma = float(sigma)
        self._rng = np.random.default_rng(seed)

    def __getattr__(self, name):
        # Delegate parameter plumbing to the dense core.
        return getattr(self.core, name)

    @property
    def dims(self) -> tuple[int, ...]:
        return self.core.dims

    def losses(self, samples: np.ndarray) -> np.ndarray:
        # Scoring uses clean inputs (no noise).
        return self.core.losses(samples)

    def train_step(self, sample: np.ndarray, lr: float) -> float:
        noisy = sample + self.sigma * self._rng.standard_normal(sample.shape).astype(np.float32)
        loss, _, _, _ = _ae_forward(noisy, self.core.W, target=sample)
        grad_W, _, _ = _analytic_gradients(noisy, self.core.W, target=sample)
        self.core.W = self.core._opt.step(self.core.W, grad_W, lr)
        return loss

    def train(self, samples: np.ndarray, epochs: int, lr: float) -> list[float]:
        return [float(np.mean([self.train_step(s, lr) for s in samples]))
                for _ in range(epochs)]


# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #
class WindowCycle:
    """Cycle through a client's healthy training windows without replacement.

    Reshuffles only after exhaustion, exactly as the plan requires.
    """

    def __init__(self, windows: np.ndarray, seed: int):
        self.windows = np.asarray(windows, dtype=np.float32)
        if len(self.windows) == 0:
            raise ValueError("WindowCycle requires at least one window")
        self._rng = np.random.default_rng(seed)
        self._order = self._rng.permutation(len(self.windows))
        self._pos = 0

    def next(self, count: int) -> np.ndarray:
        if count <= 0:
            raise ValueError("requested window count must be positive")
        chunks: list[np.ndarray] = []
        collected = 0
        while collected < count:
            remaining = count - collected
            avail = len(self._order) - self._pos
            take = min(remaining, avail)
            idx = self._order[self._pos:self._pos + take]
            chunks.append(self.windows[idx])
            collected += take
            self._pos += take
            if self._pos >= len(self._order):
                # exhausted -> reshuffle for the next pass
                self._pos = 0
                self._order = self._rng.permutation(len(self.windows))
        return np.concatenate(chunks)


def select_pretrain_windows(domains, datasets: tuple[str, ...],
                            per_round: int, seed: int) -> dict[str, "WindowCycle"]:
    cycles: dict[str, WindowCycle] = {}
    for ds in datasets:
        for client in CLIENTS:
            key = client_key(ds, client)
            train = np.asarray(domains[ds][client]["train"], dtype=np.float32)
            cycles[key] = WindowCycle(train, stable_seed(seed, key))
    return cycles


# --------------------------------------------------------------------------- #
# Federated training engines (dense and LoRA) for FedAvg / FedAdam
# --------------------------------------------------------------------------- #
def _flat_aggregate_fedavg(global_flat, contributions, count):
    total = sum(s for _, s in contributions)
    acc = np.zeros_like(global_flat)
    for w, s in contributions:
        acc += np.asarray(w, dtype=np.float32) * np.float32(s)
    return (acc / np.float32(total)).astype(np.float32)


def _flat_aggregate_fedadam(global_flat, contributions, count, opt, cfg):
    weighted = _flat_aggregate_fedavg(global_flat, contributions, count)
    delta = (weighted - global_flat).astype(np.float32)
    opt["t"] += 1
    opt["m"][:] = cfg.fedadam_beta1 * opt["m"] + (1.0 - cfg.fedadam_beta1) * delta
    opt["v"][:] = cfg.fedadam_beta2 * opt["v"] + (1.0 - cfg.fedadam_beta2) * (delta * delta)
    m_hat = opt["m"] / (1.0 - cfg.fedadam_beta1 ** opt["t"])
    v_hat = opt["v"] / (1.0 - cfg.fedadam_beta2 ** opt["t"])
    step = cfg.fedadam_eta * m_hat / (np.sqrt(v_hat) + cfg.fedadam_tau)
    return (global_flat + step.astype(np.float32)).astype(np.float32)


def _make_fedadam_state(n):
    return {"m": np.zeros(n, dtype=np.float32),
            "v": np.zeros(n, dtype=np.float32), "t": 0}


def federated_train_dense(
    *, dims, base_weights, datasets, domains, config, seed, algorithm,
    cycles=None, per_round=None, epochs=None, lr=None, capture_diagnostics=False,
):
    """Federated full-dense training. Returns ``(global_flat, factory, diag_rows)``.

    ``factory(global_flat)`` returns a fresh DenseAE with that global state so
    evaluation reuses the same code path. ``base_weights`` is the initialization
    every client starts each round from (then the global trainable weights
    overwrite W).
    """
    rounds = config.rounds
    per_round = config.samples_per_round if per_round is None else per_round
    epochs = config.local_epochs if epochs is None else epochs
    lr = config.learning_rate if lr is None else lr
    keys = [client_key(ds, c) for ds in datasets for c in CLIENTS]
    prototype = DenseAE(dims, weights=base_weights)
    n = prototype.parameter_count()
    global_flat = prototype.get_trainable()
    server_opt = _make_fedadam_state(n)
    diag_rows = []
    for r in range(rounds):
        contributions = []
        update_flats = []
        for key in keys:
            model = DenseAE(dims, weights=base_weights)
            model.set_trainable(global_flat, reset_optimizer=True)
            if cycles:
                batch = cycles[key].next(per_round)
            else:
                batch = _fallback_batch(domains, key, per_round, r, config.samples_per_round)
            before = float(np.mean(model.losses(batch)))
            model.train(batch, epochs=epochs, lr=lr)
            after = float(np.mean(model.losses(batch)))
            local_flat = model.get_trainable()
            update_flats.append(local_flat - global_flat)
            contributions.append((local_flat, len(batch)))
            if capture_diagnostics:
                diag_rows.append(_dense_diag_row(
                    key, r, before, after, model.W, update_flats[-1]))
        if algorithm == "fedavg":
            global_flat = _flat_aggregate_fedavg(global_flat, contributions, len(keys))
        else:
            global_flat = _flat_aggregate_fedadam(
                global_flat, contributions, len(keys), server_opt, config)
        if capture_diagnostics:
            _annotate_cosine(diag_rows, update_flats, r)

    def factory(flat):
        m = DenseAE(dims, weights=base_weights)
        m.set_trainable(flat, reset_optimizer=True)
        return m

    return global_flat, factory, (diag_rows if capture_diagnostics else [])


def _sparse_int8_reconstruction(
    global_flat: np.ndarray,
    local_flat: np.ndarray,
    sparsity_k: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Reconstruct a client model from a Top-K signed-int8 delta."""
    delta = np.asarray(local_flat - global_flat, dtype=np.float32)
    k = min(int(sparsity_k), delta.size)
    indices = np.argsort(-np.abs(delta), kind="stable")[:k]
    scale, packed = quantize_dynamic(delta[indices])
    reconstructed = np.asarray(global_flat, dtype=np.float32).copy()
    reconstructed[indices] += dequantize_dynamic(packed, scale)
    denominator = max(float(np.linalg.norm(delta)), 1e-12)
    error = float(np.linalg.norm(reconstructed - local_flat)) / denominator
    return reconstructed, {
        "topk_retained_fraction": float(k / max(delta.size, 1)),
        "relative_compression_error": error,
        "quantized_zero_fraction": float(np.mean(packed == 0)),
    }


def federated_train_lora(
    *,
    dims,
    rank,
    base_weights,
    train_a,
    alpha,
    datasets,
    domains,
    config,
    seed,
    algorithm,
    cycles=None,
    per_round=None,
    epochs=None,
    lr=None,
    noise_sigma=0.0,
    firmware_compression=False,
    sparsity_k=32,
    capture_diagnostics=False,
):
    """Train one shared LoRA adapter with genuine client/server federation.

    Local AdamW state persists throughout each client's local epoch block and is
    reset only when the next global model is received.  Optional compression
    mirrors the Top-K/int8 radio path, but screening should normally use float32.
    """
    rounds = config.rounds
    per_round = config.samples_per_round if per_round is None else per_round
    epochs = config.local_epochs if epochs is None else epochs
    lr = config.learning_rate if lr is None else lr
    keys = [client_key(ds, c) for ds in datasets for c in CLIENTS]
    prototype = ABLoRA(
        dims, rank, base_weights, train_a=train_a, alpha=alpha
    )
    global_flat = prototype.get_trainable()
    server_opt = _make_fedadam_state(global_flat.size)
    diag_rows: list[dict] = []

    for round_index in range(rounds):
        contributions: list[tuple[np.ndarray, int]] = []
        update_flats: list[np.ndarray] = []
        for key in keys:
            model = ABLoRA(
                dims, rank, base_weights, train_a=train_a, alpha=alpha
            )
            model.set_trainable(global_flat, reset_optimizer=True)
            if cycles:
                batch = cycles[key].next(per_round)
            else:
                batch = _fallback_batch(
                    domains, key, per_round, round_index, config.samples_per_round
                )
            before = float(np.mean(model.losses(batch)))
            local_rng = np.random.default_rng(
                stable_seed(seed, f"{key}:round:{round_index}:noise")
            )
            model.train(
                batch,
                epochs=epochs,
                lr=lr,
                noise_sigma=float(noise_sigma),
                rng=local_rng if noise_sigma > 0.0 else None,
            )
            after = float(np.mean(model.losses(batch)))
            local_flat = model.get_trainable()
            compression_stats = {
                "topk_retained_fraction": 1.0,
                "relative_compression_error": 0.0,
                "quantized_zero_fraction": 0.0,
            }
            if firmware_compression:
                local_flat, compression_stats = _sparse_int8_reconstruction(
                    global_flat, local_flat, sparsity_k
                )
            update = local_flat - global_flat
            update_flats.append(update)
            contributions.append((local_flat, len(batch)))
            if capture_diagnostics:
                diag_rows.append(
                    {
                        "client_key": key,
                        "round": round_index,
                        "loss_before": before,
                        "loss_after": after,
                        "update_norm": float(np.linalg.norm(update)),
                        "A_norm": float(
                            np.sqrt(sum(np.sum(a * a) for a in model.A))
                        ),
                        "B_norm": float(
                            np.sqrt(sum(np.sum(b * b) for b in model.B))
                        ),
                        **compression_stats,
                    }
                )

        if algorithm == "fedavg":
            global_flat = _flat_aggregate_fedavg(
                global_flat, contributions, len(keys)
            )
        elif algorithm == "fedadam":
            global_flat = _flat_aggregate_fedadam(
                global_flat, contributions, len(keys), server_opt, config
            )
        else:
            raise ValueError(f"unknown federated algorithm: {algorithm}")

        if firmware_compression:
            scale, packed = quantize_dynamic(global_flat)
            global_flat = dequantize_dynamic(packed, scale).astype(np.float32)
        if capture_diagnostics:
            _annotate_cosine(diag_rows, update_flats, round_index)

    def factory(flat):
        model = ABLoRA(
            dims, rank, base_weights, train_a=train_a, alpha=alpha
        )
        model.set_trainable(flat, reset_optimizer=True)
        return model

    return global_flat, factory, (diag_rows if capture_diagnostics else [])


def _fallback_batch(domains, key, per_round, r, default):
    ds, client = key.split(":")
    train = np.asarray(domains[ds][client]["train"], dtype=np.float32)
    start = (r * default) % max(1, len(train) - per_round)
    return train[start:start + per_round]


def _dense_diag_row(key, r, before, after, W, update):
    w_norm = float(np.sqrt(sum(np.sum(w * w) for w in W)))
    g_norm = float(np.sqrt(np.sum(update * update)))
    return {
        "client_key": key, "round": r,
        "loss_before": before, "loss_after": after,
        "W_norm": w_norm, "update_norm": g_norm,
        "update_flat": update,  # used transiently for cosine, dropped at write
    }


def _annotate_cosine(rows, update_flats, r):
    """Add pairwise cosine similarity of client updates to the round's rows."""
    round_rows = [row for row in rows if row["round"] == r]
    if len(update_flats) < 2:
        for row in round_rows:
            row["pairwise_cosine_mean"] = float("nan")
        return
    a = update_flats[0]
    cosines = []
    for b in update_flats[1:]:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosines.append(float(np.dot(a, b) / denom) if denom > 1e-12 else float("nan"))
    for row in round_rows:
        row["pairwise_cosine_mean"] = float(mean(cosines)) if cosines else float("nan")


# --------------------------------------------------------------------------- #
# Base pretraining + .npz checkpoint
# --------------------------------------------------------------------------- #
@dataclass
class PretrainConfig:
    rounds: int = 20
    samples_per_round: int = 60
    local_epochs: int = 2
    learning_rate: float = 0.02
    algorithm: str = "fedavg"  # or fedadam


def pretrain_base(domains, datasets, dims, config: PretrainConfig, seed: int,
                  algorithm: str = "fedavg") -> tuple[list[np.ndarray], dict]:
    base_weights = _init_base_weights(dims)
    exp_cfg = ExperimentConfig(
        datasets=datasets,
        rounds=config.rounds,
        samples_per_round=config.samples_per_round,
        local_epochs=config.local_epochs,
        learning_rate=config.learning_rate,
    )
    cycles = select_pretrain_windows(
        domains, datasets, config.samples_per_round, seed)
    global_flat, factory, _ = federated_train_dense(
        dims=dims, base_weights=base_weights, datasets=datasets, domains=domains,
        config=exp_cfg, seed=seed, algorithm=algorithm,
        cycles=cycles, per_round=config.samples_per_round,
        epochs=config.local_epochs, lr=config.learning_rate,
    )
    trained = factory(global_flat)
    fingerprint = normalization_fingerprint(domains, datasets)
    meta = {
        "architecture": list(dims),
        "rounds": config.rounds,
        "samples_per_round": config.samples_per_round,
        "local_epochs": config.local_epochs,
        "learning_rate": config.learning_rate,
        "algorithm": algorithm,
        "seed": int(seed),
        "datasets": list(datasets),
        "normalization_means": fingerprint[0].tolist(),
        "normalization_stds": fingerprint[1].tolist(),
        "created_utc": int(time.time()),
    }
    return [w.copy() for w in trained.W], meta


def save_base_npz(path: Path, dims, weights, meta: dict) -> None:
    arrays = {f"W{i}": w for i, w in enumerate(weights)}
    arrays["__meta__"] = np.array(json.dumps(meta), dtype=object)
    arrays["__dims__"] = np.array(dims, dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def load_base_npz(path: Path) -> tuple[tuple[int, ...], list[np.ndarray], dict]:
    data = np.load(path, allow_pickle=True)
    dims = tuple(int(x) for x in data["__dims__"])
    weights = [data[f"W{i}"].astype(np.float32) for i in range(len(dims) - 1)]
    meta = json.loads(str(data["__meta__"].item()))
    return dims, weights, meta


def normalization_fingerprint(domains, datasets) -> tuple[np.ndarray, np.ndarray]:
    """Recover the prepared per-feature means/stds used for Z-scoring.

    The prepared ``*_bin/normalization.json`` is the source of truth; we read it
    directly so the checkpoint records the exact normalization the base expects.
    """
    means_acc = []
    stds_acc = []
    for ds in datasets:
        norm_path = PYTHON_TOOLS_DIR / f"{ds}_bin" / "normalization.json"
        payload = json.loads(norm_path.read_text(encoding="utf-8"))
        means_acc.append(np.asarray(payload["means"], dtype=np.float32))
        stds_acc.append(np.asarray(payload["stds"], dtype=np.float32))
    return (np.mean(means_acc, axis=0).astype(np.float32),
            np.mean(stds_acc, axis=0).astype(np.float32))


# --------------------------------------------------------------------------- #
# Stage 1: learning diagnostics
# --------------------------------------------------------------------------- #
def gradient_check(*, dims, rank, alpha, dtype=np.float64, eps=1e-5):
    """Finite-difference gradient checks for dense W, LoRA A and LoRA B.

    Returns a dict of relative errors. Used by stage 1 and the acceptance tests.
    """
    rng = np.random.default_rng(123)
    sample = rng.standard_normal(INPUT_SIZE).astype(dtype)
    W = [w.astype(dtype) for w in _init_base_weights(dims)]
    A = [a.astype(dtype) for a in _init_lora_a(dims, rank)]
    B = [rng.standard_normal((o, rank)).astype(dtype) * 0.05
         for o in dims[1:]]

    gW, gA, gB = _analytic_gradients(sample, W, lora_a=A, lora_b=B,
                                     lora_alpha=alpha, dtype=dtype)

    def fd_for(ana_flat, getter_setter, size):
        ana_flat = np.asarray(ana_flat, dtype=dtype).ravel()
        errs = []
        idxs = np.linspace(0, size - 1, min(12, size)).astype(int)
        for idx in idxs:
            orig = getter_setter("get", idx)
            getter_setter("set", idx, orig + eps)
            lp, _, _, _ = _ae_forward(sample, W, lora_a=A, lora_b=B,
                                      lora_alpha=alpha, dtype=dtype)
            getter_setter("set", idx, orig - eps)
            lm, _, _, _ = _ae_forward(sample, W, lora_a=A, lora_b=B,
                                      lora_alpha=alpha, dtype=dtype)
            getter_setter("set", idx, orig)
            numeric = (lp - lm) / (2 * eps)
            denom = max(1e-12, abs(numeric) + abs(ana_flat[idx]))
            errs.append(abs(numeric - ana_flat[idx]) / denom)
        return float(mean(errs)) if errs else float("nan")

    def make_accessor(tensor, grad_for_tensor):
        flat = tensor.ravel()
        ana_flat = np.asarray(grad_for_tensor, dtype=dtype).ravel()

        def access(mode, idx, value=None):
            if mode == "get":
                return float(flat[idx])
            flat[idx] = value
            # write back into the structure used by _ae_forward
            tensor[...] = flat.reshape(tensor.shape)
            return None

        return ana_flat, access, flat.size

    a_W, acc_W, n_W = make_accessor(W[2], gW[2])
    a_A, acc_A, n_A = make_accessor(A[0], gA[0])
    a_B, acc_B, n_B = make_accessor(B[2], gB[2])
    return {
        "W_max_rel_err": fd_for(a_W, acc_W, n_W),
        "A_max_rel_err": fd_for(a_A, acc_A, n_A),
        "B_max_rel_err": fd_for(a_B, acc_B, n_B),
    }


def overfit_check(dims, n_samples=16, epochs=200, lr=0.02, seed=7) -> dict:
    """Confirm the dense AE can deliberately overfit 16 healthy samples."""
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal((n_samples, INPUT_SIZE)).astype(np.float32) * 0.5
    model = DenseAE(dims)
    before = float(np.mean(model.losses(samples)))
    model.train(samples, epochs=epochs, lr=lr)
    after = float(np.mean(model.losses(samples)))
    return {"before": before, "after": after,
            "reduction_pct": 100.0 * (before - after) / max(before, 1e-12)}


def diagnostics_stage(domains, datasets, dims, rank, alpha, seed) -> tuple[list[dict], dict]:
    """Capture per-client/round diagnostics during one dense federated run."""
    print(f"  [diagnostics] gradient checks (rank={rank}, alpha={alpha:.2f})...")
    gc = gradient_check(dims=dims, rank=rank, alpha=alpha)
    print(f"    rel-err W={gc['W_max_rel_err']:.4g} "
          f"A={gc['A_max_rel_err']:.4g} B={gc['B_max_rel_err']:.4g}")
    print(f"  [diagnostics] overfit 16 samples...")
    of = overfit_check(dims)
    print(f"    before={of['before']:.4f} after={of['after']:.4f} "
          f"reduction={of['reduction_pct']:.1f}%")

    base_weights = _init_base_weights(dims)
    exp_cfg = ExperimentConfig(datasets=datasets, rounds=3,
                               samples_per_round=30, local_epochs=4,
                               learning_rate=0.02)
    cycles = select_pretrain_windows(domains, datasets, 30, seed)
    _, _, rows = federated_train_dense(
        dims=dims, base_weights=base_weights, datasets=datasets, domains=domains,
        config=exp_cfg, seed=seed, algorithm="fedadam", cycles=cycles,
        per_round=30, epochs=4, lr=0.02, capture_diagnostics=True)

    # strip the transient full update vector before returning for CSV safety
    for row in rows:
        row.pop("update_flat", None)
    return rows, {"gradient_check": gc, "overfit": of}


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate_model_on_domain(model, dataset, client, seed, method, family,
                             scope, domain, *, rank=None, alpha=None,
                             trainable_params=0, total_params=0, seconds=0.0,
                             uplink=0, downlink=0, deployable=False,
                             global_bytes=0, update_bytes=0, notes="") -> dict:
    return evaluate_scores(
        dataset=dataset, client=client, seed=seed, method=method, family=family,
        training_scope=scope, domain=domain, scorer=model.losses,
        trainable_parameters=trainable_params, total_model_parameters=total_params,
        rank=rank, deployable=deployable, training_seconds=seconds,
        uplink_bytes=uplink, downlink_bytes=downlink,
        architecture="lora_ab" if isinstance(model, ABLoRA) else "dense",
        model_dims="->".join(map(str, model.dims)),
        global_packet_bytes=global_bytes, update_packet_bytes=update_bytes,
        notes=notes,
    )


def lora_packet_sizes(dims, rank, train_a: bool, sparsity_k=32) -> tuple[int, int]:
    """Estimate UPDATE/GLOBAL serialized bytes for a LoRA model.

    GLOBAL carries all trainable LoRA values signed-int8; UPDATE carries Top-K
    deltas. Packet metadata matches the existing 87-byte baseline layout.
    """
    b_count = sum(o * rank for o in dims[1:])
    a_count = sum(i * rank for i in dims[:-1]) if train_a else 0
    trainable = b_count + a_count
    metadata = 16
    global_bytes = metadata + trainable
    index_bytes = 2 if trainable > 256 else 1
    update_bytes = 87 if trainable <= 256 else 87 + min(sparsity_k, trainable) * (index_bytes - 1)
    return int(update_bytes), int(global_bytes)


# --------------------------------------------------------------------------- #
# Stage 2: controls + base pretraining
# --------------------------------------------------------------------------- #
def base_pretraining_stage(domains, datasets, dims, seed,
                           output_dir: Path, pretrain_cfg: PretrainConfig,
                           algorithm="fedavg") -> tuple[list[np.ndarray], dict]:
    print(f"  [base_pretraining] federated {algorithm}, "
          f"{pretrain_cfg.rounds}r x {pretrain_cfg.samples_per_round}s x "
          f"{pretrain_cfg.local_epochs}e")
    weights, meta = pretrain_base(domains, datasets, dims, pretrain_cfg, seed,
                                  algorithm=algorithm)
    # report validation loss reduction vs untrained base
    untrained = DenseAE(dims)
    val_before, val_after = [], []
    for ds in datasets:
        for client in CLIENTS:
            val = np.asarray(domains[ds][client]["val"], dtype=np.float32)
            mask = np.asarray(domains[ds][client]["val_labels"], dtype=np.int8) == 0
            val = val[mask]
            if len(val):
                val_before.append(float(np.mean(untrained.losses(val))))
                val_after.append(
                    float(np.mean(DenseAE(dims, weights=weights).losses(val))))
    vb = float(mean(val_before)) if val_before else float("nan")
    va = float(mean(val_after)) if val_after else float("nan")
    meta["validation_loss_untrained"] = vb
    meta["validation_loss_pretrained"] = va
    print(f"    healthy val loss: untrained={vb:.4f} pretrained={va:.4f} "
          f"(reduction {100.0*(vb-va)/max(vb,1e-12):.1f}%)")
    npz_path = output_dir / "pretrained_base.npz"
    save_base_npz(npz_path, dims, weights, meta)
    print(f"    saved base -> {npz_path}")
    return weights, meta


# --------------------------------------------------------------------------- #
# Stage 3: LoRA ablation over the pretrained base
# --------------------------------------------------------------------------- #
def lora_ablation_stage(domains, datasets, dims, base_weights, rank, alpha_grid,
                        seed, config: ExperimentConfig) -> list[dict]:
    rows: list[dict] = []
    pairs = [(ds, c) for ds in datasets for c in CLIENTS]
    base_count = sum(w.size for w in base_weights)

    # 1) frozen pretrained base, no adaptation
    frozen = DenseAE(dims, weights=base_weights)
    for ds, client in pairs:
        rows.append(evaluate_model_on_domain(
            frozen, ds, client, seed, "lora_frozen_pretrained", "frozen_base",
            "none", domains[ds][client], rank=rank,
            trainable_params=0, total_params=base_count,
            notes="frozen pretrained base, no LoRA"))

    # 2) completely untrained random base (true no-learning control)
    random_base = _init_base_weights(dims)
    random_frozen = ABLoRA(
        dims, rank, random_base, train_a=False, alpha=1.0
    )
    for ds, client in pairs:
        rows.append(evaluate_model_on_domain(
            random_frozen, ds, client, seed, "lora_frozen_random",
            "untrained_control", "none", domains[ds][client], rank=rank,
            trainable_params=0, total_params=base_count,
            notes="random frozen base with zero LoRA B; no training"))

    # 3) current random frozen base + genuinely federated B-only LoRA.
    for alpha in alpha_grid:
        random_cycles = select_pretrain_windows(
            domains, datasets, config.samples_per_round,
            stable_seed(seed, f"random_base_alpha_{alpha}"),
        )
        flat, factory, _ = federated_train_lora(
            dims=dims, rank=rank, base_weights=random_base, train_a=False,
            alpha=alpha, datasets=datasets, domains=domains, config=config,
            seed=seed, algorithm="fedadam", cycles=random_cycles,
        )
        trained_random = factory(flat)
        for ds, client in pairs:
            ub, gb = lora_packet_sizes(dims, rank, train_a=False)
            rows.append(evaluate_model_on_domain(
                trained_random, ds, client, seed,
                f"lora_bonly_randombase_a{alpha:.1f}", "lora_b_only",
                "federated_frozen_random_base", domains[ds][client],
                rank=rank, alpha=alpha,
                trainable_params=trained_random.parameter_count(),
                total_params=base_count, deployable=(gb <= MAX_LORA_PACKET_BYTES),
                global_bytes=gb, update_bytes=ub,
                notes="federated B-only LoRA on random frozen base"))

    # 4) pretrained base + B-only LoRA, and 5) pretrained base + A+B LoRA.
    # Each client performs local training and the server aggregates every round;
    # the previous implementation accidentally trained one centralized stack.
    for alpha in alpha_grid:
        for train_a, suffix, family, scope in [
            (False, "bonly", "lora_b_only", "pretrained_base_b_only"),
            (True, "ab", "lora_ab", "pretrained_base_ab"),
        ]:
            cycles = select_pretrain_windows(
                domains, datasets, config.samples_per_round,
                stable_seed(seed, f"pretrained_{suffix}_alpha_{alpha}"),
            )
            t0 = time.perf_counter()
            flat, factory, _ = federated_train_lora(
                dims=dims, rank=rank, base_weights=base_weights,
                train_a=train_a, alpha=alpha, datasets=datasets,
                domains=domains, config=config, seed=seed,
                algorithm="fedadam", cycles=cycles,
            )
            seconds = time.perf_counter() - t0
            model = factory(flat)
            ub, gb = lora_packet_sizes(dims, rank, train_a=train_a)
            for ds, client in pairs:
                rows.append(evaluate_model_on_domain(
                    model, ds, client, seed,
                    f"lora_{suffix}_pretrained_a{alpha:.1f}", family, scope,
                    domains[ds][client], rank=rank, alpha=alpha,
                    trainable_params=model.parameter_count(), total_params=base_count,
                    seconds=seconds, deployable=(gb <= MAX_LORA_PACKET_BYTES),
                    global_bytes=gb, update_bytes=ub,
                    notes=(f"federated {'A+B' if train_a else 'B-only'} "
                           "LoRA on pretrained base")))

    # 6) continued full-dense federated training (upper bound)
    weights, factory, _ = federated_train_dense(
        dims=dims, base_weights=base_weights, datasets=datasets, domains=domains,
        config=config, seed=seed, algorithm="fedadam")
    cont = factory(weights)
    for ds, client in pairs:
        rows.append(evaluate_model_on_domain(
            cont, ds, client, seed, "continued_dense_fedadam", "full_weight_ae",
            "continued_dense_federated", domains[ds][client],
            trainable_params=base_count, total_params=base_count,
            notes="continued full-dense federated training (upper bound)"))
    return rows


# --------------------------------------------------------------------------- #
# Stage 4: sequential optimization tuning (seed 11 screening)
# --------------------------------------------------------------------------- #
def _val_loss_for(model, domains, datasets):
    """Mean healthy-validation reconstruction loss across clients."""
    vals = []
    for ds in datasets:
        for client in CLIENTS:
            val = np.asarray(domains[ds][client]["val"], dtype=np.float32)
            mask = np.asarray(domains[ds][client]["val_labels"], dtype=np.int8) == 0
            val = val[mask]
            if len(val):
                vals.append(float(np.mean(model.losses(val))))
    return float(mean(vals)) if vals else float("nan")


def _validation_objective(model, domains, datasets) -> dict[str, float]:
    """Validation-only detection objective; test labels are never inspected."""
    difficult_f1: list[float] = []
    difficult_fpr: list[float] = []
    all_f1: list[float] = []
    for ds in datasets:
        for client in CLIENTS:
            domain = domains[ds][client]
            calibration_scores = model.losses(
                np.asarray(domain["calibration"], dtype=np.float32)
            )
            validation_scores = model.losses(
                np.asarray(domain["val"], dtype=np.float32)
            )
            _, _, metrics = choose_threshold(
                calibration_scores,
                validation_scores,
                np.asarray(domain["val_labels"], dtype=np.int8),
            )
            all_f1.append(float(metrics["f1"]))
            if (ds, client) in DIFFICULT_CLIENTS:
                difficult_f1.append(float(metrics["f1"]))
                difficult_fpr.append(float(metrics["fpr"]))
    target_f1 = difficult_f1 if difficult_f1 else all_f1
    target_fpr = difficult_fpr if difficult_fpr else [0.0]
    return {
        "hard_validation_f1": float(mean(target_f1)),
        "hard_validation_fpr": float(mean(target_fpr)),
        "all_validation_f1": float(mean(all_f1)),
        "healthy_validation_loss": _val_loss_for(model, domains, datasets),
    }


def _screen_lora_candidate(
    *, domains, datasets, dims, base_weights, rank, seed, rounds,
    samples_per_round, local_epochs, lr, alpha, algorithm, fedadam_eta=0.01,
):
    config = ExperimentConfig(
        datasets=datasets,
        rounds=rounds,
        samples_per_round=samples_per_round,
        local_epochs=local_epochs,
        learning_rate=lr,
        fedadam_eta=fedadam_eta,
    )
    cycles = select_pretrain_windows(
        domains, datasets, samples_per_round,
        stable_seed(seed, "optimization_shared_order"),
    )
    flat, factory, _ = federated_train_lora(
        dims=dims,
        rank=rank,
        base_weights=base_weights,
        train_a=False,
        alpha=alpha,
        datasets=datasets,
        domains=domains,
        config=config,
        seed=seed,
        algorithm=algorithm,
        cycles=cycles,
    )
    model = factory(flat)
    return model, _validation_objective(model, domains, datasets)


def _candidate_sort_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        metrics["hard_validation_f1"],
        -metrics["hard_validation_fpr"],
        metrics["all_validation_f1"],
        -metrics["healthy_validation_loss"],
    )


def optimization_stage(domains, datasets, dims, base_weights, rank,
                       seed=SCREENING_SEED) -> list[dict]:
    """Sequentially tune genuine federated B-only LoRA on validation data."""
    trace: list[dict] = []

    budgets = [
        ("3r_30s_8e", 3, 30, 8),
        ("10r_60s_4e", 10, 60, 4),
        ("20r_60s_2e", 20, 60, 2),
    ]
    best_budget = None
    for name, rounds, spr, epochs in budgets:
        _, metrics = _screen_lora_candidate(
            domains=domains, datasets=datasets, dims=dims,
            base_weights=base_weights, rank=rank, seed=seed, rounds=rounds,
            samples_per_round=spr, local_epochs=epochs, lr=0.02, alpha=1.0,
            algorithm="fedavg",
        )
        trace.append({"step": "budget", "config": name, "rounds": rounds,
                      "samples": spr, "epochs": epochs, **metrics})
        if best_budget is None or _candidate_sort_key(metrics) > best_budget[1]:
            best_budget = ((rounds, spr, epochs), _candidate_sort_key(metrics))
    rounds, spr, epochs = best_budget[0]

    best_lr = None
    for lr in (0.002, 0.01, 0.02):
        _, metrics = _screen_lora_candidate(
            domains=domains, datasets=datasets, dims=dims,
            base_weights=base_weights, rank=rank, seed=seed, rounds=rounds,
            samples_per_round=spr, local_epochs=epochs, lr=lr, alpha=1.0,
            algorithm="fedavg",
        )
        trace.append({"step": "lr", "config": f"lr_{lr}", **metrics})
        if best_lr is None or _candidate_sort_key(metrics) > best_lr[1]:
            best_lr = (lr, _candidate_sort_key(metrics))
    lr = best_lr[0]

    best_alpha = None
    for alpha in (0.5, 1.0, 2.0):
        _, metrics = _screen_lora_candidate(
            domains=domains, datasets=datasets, dims=dims,
            base_weights=base_weights, rank=rank, seed=seed, rounds=rounds,
            samples_per_round=spr, local_epochs=epochs, lr=lr, alpha=alpha,
            algorithm="fedavg",
        )
        trace.append({"step": "alpha", "config": f"alpha_{alpha}", **metrics})
        if best_alpha is None or _candidate_sort_key(metrics) > best_alpha[1]:
            best_alpha = (alpha, _candidate_sort_key(metrics))
    alpha = best_alpha[0]

    best_algo = None
    server_candidates = (
        ("fedavg", 0.0),
        ("fedadam", 0.001),
        ("fedadam", 0.005),
        ("fedadam", 0.01),
    )
    for algo, eta in server_candidates:
        _, metrics = _screen_lora_candidate(
            domains=domains, datasets=datasets, dims=dims,
            base_weights=base_weights, rank=rank, seed=seed, rounds=rounds,
            samples_per_round=spr, local_epochs=epochs, lr=lr, alpha=alpha,
            algorithm=algo, fedadam_eta=eta if algo == "fedadam" else 0.01,
        )
        name = algo if algo == "fedavg" else f"fedadam_eta_{eta}"
        trace.append({"step": "server", "config": name,
                      "algorithm": algo, "fedadam_eta": eta, **metrics})
        if best_algo is None or _candidate_sort_key(metrics) > best_algo[2]:
            best_algo = (algo, eta, _candidate_sort_key(metrics), metrics)

    trace.append({"step": "final", "config": "best",
                  "rounds": rounds, "samples": spr, "epochs": epochs,
                  "lr": lr, "alpha": alpha, "algorithm": best_algo[0],
                  "fedadam_eta": best_algo[1], **best_algo[3]})
    print(f"  [optimization] best -> budget={rounds}r/{spr}s/{epochs}e "
          f"lr={lr} alpha={alpha} algo={best_algo[0]} "
          f"hard-val-F1={best_algo[3]['hard_validation_f1']:.4f}")
    return trace


# --------------------------------------------------------------------------- #
# Stage 5a: denoising AE vs PCA baseline
# --------------------------------------------------------------------------- #
def pca_reconstruction_error(train: np.ndarray, eval_data: np.ndarray,
                             n_components: int = 6) -> np.ndarray:
    """Per-sample reconstruction error using PCA fit on healthy train data."""
    mean = train.mean(axis=0)
    centered = train - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    comps = Vt[:n_components]
    proj = (eval_data - mean) @ comps.T
    recon = proj @ comps + mean
    return np.mean((eval_data - recon) ** 2, axis=1).astype(np.float32)


def denoising_stage(domains, datasets, dims, base_weights, sigmas,
                    seed, config: ExperimentConfig) -> list[dict]:
    rows: list[dict] = []
    pairs = [(ds, c) for ds in datasets for c in CLIENTS]
    for sigma in sigmas:
        combined = ordered_stack(
            (np.asarray(domains[ds][c]["train"][:config.training_samples_per_client],
                        dtype=np.float32) for ds, c in pairs),
            seed, f"denoise_sigma_{sigma}")
        model = DenoisingAE(dims, sigma=sigma, seed=seed, weights=base_weights)
        model.train(combined, epochs=config.local_epochs, lr=config.learning_rate)
        for ds, client in pairs:
            rows.append(evaluate_model_on_domain(
                model, ds, client, seed, f"denoising_ae_sigma{sigma}",
                "denoising_ae", "pretrained_base_denoising",
                domains[ds][client], total_params=sum(w.size for w in base_weights),
                notes=f"denoising AE sigma={sigma}"))
    # PCA baseline
    for ds, client in pairs:
        train = np.asarray(domains[ds][client]["train"], dtype=np.float32)
        cal = np.asarray(domains[ds][client]["calibration"], dtype=np.float32)
        test = np.asarray(domains[ds][client]["test"], dtype=np.float32)
        scores = pca_reconstruction_error(train, test)
        cal_scores = pca_reconstruction_error(train, cal)
        threshold = float(np.mean(cal_scores) + 3.0 * np.std(cal_scores, ddof=1))
        labels = np.asarray(domains[ds][client]["labels"], dtype=np.int8)
        metrics = classification_metrics(labels, scores, threshold)
        rows.append({"dataset": ds, "client": client,
                     "client_name": CLIENT_DISPLAY_NAMES[(ds, client)],
                     "seed": seed, "method": "pca_reconstruction",
                     "family": "pca_baseline", "training_scope": "single_client",
                     "threshold_method": "3-sigma", "threshold": threshold,
                     **{k: metrics[k] for k in ("tp", "fp", "fn", "tn", "precision",
                                                "recall", "f1", "fpr", "roc_auc",
                                                "pr_auc")}})
    return rows


# --------------------------------------------------------------------------- #
# Stage 5b: drift handling (EWMA + 3-of-5 windowing)
# --------------------------------------------------------------------------- #
def ewma_drift_scores(test_scores: np.ndarray, labels: np.ndarray,
                      calibration_scores: np.ndarray, *, alpha=0.05,
                      window=5, required=3) -> dict:
    """EWMA baseline updated only from windows accepted as normal, with a
    3-of-5 consecutive-window alarm rule."""
    base = float(np.mean(calibration_scores))
    spread = float(np.std(calibration_scores, ddof=1)) if len(calibration_scores) > 1 else 1e-6
    threshold = base + 3.0 * spread
    level = base
    alarm_count = 0
    tp = fp = 0
    history: list[int] = []
    for s, lab in zip(test_scores, labels):
        is_alarm = s > threshold
        history.append(int(is_alarm))
        if len(history) > window:
            history.pop(0)
        fired = (sum(history) >= required)
        if fired:
            alarm_count += 1
            if lab == 1:
                tp += 1
            else:
                fp += 1
        # freeze adaptation when suspected anomaly
        if not is_alarm:
            level = (1 - alpha) * level + alpha * float(s)
            threshold = level + 3.0 * spread
    return {"ewma_alarm_count": alarm_count, "ewma_tp": tp, "ewma_fp": fp,
            "window": window, "required": required, "alpha": alpha}


def drift_stage(domains, datasets, dims, base_weights, seed,
                config: ExperimentConfig) -> list[dict]:
    rows: list[dict] = []
    chronological = {"ims", "xjtu", "pronostia"}
    pairs = [(ds, c) for ds in datasets for c in CLIENTS]
    combined = ordered_stack(
        (np.asarray(domains[ds][c]["train"][:config.training_samples_per_client],
                    dtype=np.float32) for ds, c in pairs),
        seed, "drift_model")
    model = ABLoRA(dims, DEFAULT_RANK, base_weights, train_a=True, alpha=1.0)
    model.train(combined, epochs=config.local_epochs, lr=config.learning_rate)
    for ds, client in pairs:
        test = np.asarray(domains[ds][client]["test"], dtype=np.float32)
        cal = np.asarray(domains[ds][client]["calibration"], dtype=np.float32)
        labels = np.asarray(domains[ds][client]["labels"], dtype=np.int8)
        scores = model.losses(test)
        cal_scores = model.losses(cal)
        ewma = ewma_drift_scores(scores, labels, cal_scores)
        rows.append({"dataset": ds, "client": client,
                     "client_name": CLIENT_DISPLAY_NAMES[(ds, client)],
                     "seed": seed, "method": "ewma_3of5_drift",
                     "family": "drift_handling",
                     "chronological_preserved": ds in chronological,
                     **ewma})
    return rows


# --------------------------------------------------------------------------- #
# CSV / report writers
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(fields) if fields else sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})


def summarize_ablation(runs: list[dict]) -> list[dict]:
    """Aggregate ablation runs across seeds into a summary."""
    from statistics import pstdev
    grouped: dict[tuple, list[dict]] = {}
    for r in runs:
        grouped.setdefault((r["dataset"], r["client"], r["method"]), []).append(r)
    summary = []
    for (ds, client, method), rows in sorted(grouped.items()):
        item = {"dataset": ds, "client": client, "method": method,
                "seeds": len(rows), "family": rows[0]["family"],
                "training_scope": rows[0]["training_scope"]}
        for metric in ("f1", "fpr", "precision", "recall", "roc_auc", "pr_auc"):
            vals = [float(row[metric]) for row in rows if metric in row]
            if vals:
                item[f"{metric}_mean"] = mean(vals)
                item[f"{metric}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        summary.append(item)
    return summary


def build_report(path: Path, payload: dict) -> None:
    lines = [
        "# Learning-Failure Rescue Report",
        "",
        "This experiment is PC-only and reuses the existing prepared-data loader",
        "and leakage validation. It does not modify the original simulator.",
        "",
        "## Configuration",
        "",
        f"- Datasets: {', '.join(payload['config']['datasets'])}",
        f"- Seeds: {', '.join(map(str, payload['config']['seeds']))}",
        f"- Architecture: {payload['config']['dims']}",
        f"- LoRA rank: {payload['config']['rank']}",
        f"- Difficult clients: {', '.join(DIFFICULT_CLIENTS.values())}",
        "",
        "## Correctness checks",
        "",
    ]
    for key, value in payload["checks"].items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Stage outcomes", ""])
    for stage, outcome in payload["decisions"].items():
        lines.append(f"- **{stage}**: {outcome}")
    lines.extend([
        "",
        "## Decisions and rejection reasons",
        "",
        "See ``learning_failure_decisions.json`` for the machine-readable record of",
        "every configuration tried, which advanced, and which were rejected.",
        "",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Built-in correctness checks (also exposed via learning_failure_tests.py)
# --------------------------------------------------------------------------- #
def run_correctness_checks(dims=LARGER_DIMS, rank=DEFAULT_RANK) -> dict:
    """Return a dict of pass/fail booleans for the acceptance criteria."""
    out: dict[str, object] = {}

    # 1. finite-difference gradient checks for dense W, LoRA A and LoRA B
    gc = gradient_check(dims=dims, rank=rank, alpha=1.0)
    out["gradient_check_W"] = gc["W_max_rel_err"] < 0.02
    out["gradient_check_A"] = gc["A_max_rel_err"] < 0.02
    out["gradient_check_B"] = gc["B_max_rel_err"] < 0.02
    out["_gradient_rel_errs"] = gc

    # 2. overfit 16 healthy samples (>=50% reconstruction-loss reduction)
    of = overfit_check(dims)
    out["overfit_16_samples"] = of["reduction_pct"] >= 50.0
    out["_overfit"] = of

    # 3. W stays unchanged during frozen-base LoRA training
    base = _init_base_weights(dims)
    base_snapshot = [w.copy() for w in base]
    model = ABLoRA(dims, rank, base, train_a=True, alpha=1.0)
    rng = np.random.default_rng(0)
    model.train(rng.standard_normal((8, INPUT_SIZE)).astype(np.float32),
                epochs=3, lr=0.02)
    out["W_frozen_during_lora"] = all(
        np.array_equal(a, b) for a, b in zip(base_snapshot, model.W))

    # 4. both A and B change during A+B training
    a_before = [a.copy() for a in model.A]
    b_before = [b.copy() for b in model.B]
    model.train(rng.standard_normal((8, INPUT_SIZE)).astype(np.float32),
                epochs=3, lr=0.02)
    out["A_changes_in_ab_training"] = not all(
        np.array_equal(a, b) for a, b in zip(a_before, model.A))
    out["B_changes_in_ab_training"] = not all(
        np.array_equal(a, b) for a, b in zip(b_before, model.B))

    # 5. weighted federated aggregation matches a manual calculation
    manual = (0.3 * np.array([1.0, 2.0]) + 0.7 * np.array([3.0, 4.0]))
    got = _flat_aggregate_fedavg(
        np.zeros(2),
        [(np.array([1.0, 2.0]), 3), (np.array([3.0, 4.0]), 7)],
        2)
    out["weighted_aggregation_matches"] = bool(np.allclose(manual, got))

    # 6. identical results for repeated runs with the same seed
    s = 42
    data = np.random.default_rng(s).standard_normal((8, INPUT_SIZE)).astype(np.float32)
    m1 = DenseAE(dims)
    m1.train(data, epochs=2, lr=0.02)
    w1 = m1.get_trainable().copy()
    m2 = DenseAE(dims)
    m2.train(data, epochs=2, lr=0.02)
    out["seed_reproducibility"] = bool(np.array_equal(w1, m2.get_trainable()))

    # 7. final compressed packets remain <= 255 bytes (rank-1 LoRA, B-only)
    ub, gb = lora_packet_sizes(dims, rank, train_a=False)
    out["rank1_packet_le_255_bytes"] = (ub <= MAX_LORA_PACKET_BYTES
                                        and gb <= MAX_LORA_PACKET_BYTES)
    out["_packet_bytes"] = {"update": ub, "global": gb}
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(s.strip()) for s in value.split(",") if s.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def parse_datasets(value: str) -> tuple[str, ...]:
    datasets = tuple(s.strip().lower() for s in value.split(",") if s.strip())
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown datasets: {', '.join(unknown)}")
    if not datasets:
        raise argparse.ArgumentTypeError("at least one dataset is required")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=("all", "checks") + STAGES,
                        help="experiment stage to run (default: all)")
    parser.add_argument("--seeds", type=parse_seeds, default=ALL_SEEDS,
                        help="comma-separated seeds (default 11,29,47,71,97)")
    parser.add_argument("--datasets", type=parse_datasets, default=DATASETS,
                        help="comma-separated datasets (default: all four)")
    parser.add_argument("--output-dir", type=Path,
                        default=SCRIPT_DIR / "results" / "learning_failure",
                        help="directory for outputs")
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--alpha-grid", type=str, default="0.5,1.0,2.0",
                        help="comma-separated LoRA alpha values")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "checks":
        checks = run_correctness_checks(rank=args.rank)
        print(json.dumps({k: v for k, v in checks.items()
                          if not k.startswith("_")}, indent=2))
        failed = [k for k, v in checks.items() if v is False]
        print("\nFAILED:", ", ".join(failed) if failed else "none")
        return

    config = ExperimentConfig(datasets=args.datasets, seeds=args.seeds)
    alpha_grid = tuple(float(a) for a in args.alpha_grid.split(","))
    print("Loading and validating prepared datasets (raw archives never opened)...")
    domains = load_all_domains(config)

    decisions: dict[str, str] = {}
    diag_rows: list[dict] = []
    ablation_runs: list[dict] = []
    base_weights = _init_base_weights(LARGER_DIMS)

    do_all = (args.stage == "all")
    stages_to_run = STAGES if do_all else (args.stage,)

    for seed in (args.seeds if do_all else (args.seeds[0],)):
        print(f"\n=== Seed {seed} ===")
        if "diagnostics" in stages_to_run:
            print("Stage: diagnostics")
            rows, info = diagnostics_stage(domains, args.datasets, LARGER_DIMS,
                                           args.rank, 1.0, seed)
            diag_rows.extend(rows)
            decisions["diagnostics"] = (
                f"gradient rel-err W={info['gradient_check']['W_max_rel_err']:.4g}, "
                f"overfit reduction={info['overfit']['reduction_pct']:.1f}%")
        if "base_pretraining" in stages_to_run and seed == args.seeds[0]:
            print("Stage: base_pretraining")
            pretrain_cfg = PretrainConfig()
            base_weights, meta = base_pretraining_stage(
                domains, args.datasets, LARGER_DIMS, seed, output_dir, pretrain_cfg)
            decisions["base_pretraining"] = (
                f"healthy val loss {meta['validation_loss_untrained']:.4f} -> "
                f"{meta['validation_loss_pretrained']:.4f}")
        if "lora_ablation" in stages_to_run:
            print("Stage: lora_ablation")
            ablation_runs.extend(lora_ablation_stage(
                domains, args.datasets, LARGER_DIMS, base_weights, args.rank,
                alpha_grid, seed, config))
        if "optimization" in stages_to_run and seed == args.seeds[0]:
            print("Stage: optimization")
            trace = optimization_stage(domains, args.datasets, LARGER_DIMS,
                                       base_weights, args.rank, seed=SCREENING_SEED)
            final = [t for t in trace if t["step"] == "final"][0]
            decisions["optimization"] = (
                f"best {final['rounds']}r/{final['samples']}s/{final['epochs']}e "
                f"lr={final['lr']} alpha={final['alpha']} algo={final['algorithm']}")
            (output_dir / "optimization_trace.json").write_text(
                json.dumps(trace, indent=2), encoding="utf-8")
        if "denoising" in stages_to_run and seed == args.seeds[0]:
            print("Stage: denoising")
            denoise_rows = denoising_stage(domains, args.datasets, LARGER_DIMS,
                                           base_weights, (0.01, 0.05), seed, config)
            write_csv(output_dir / "denoising_runs.csv", denoise_rows)
            decisions["denoising"] = "denoising AE (sigma 0.01, 0.05) + PCA baseline recorded"
        if "drift" in stages_to_run and seed == args.seeds[0]:
            print("Stage: drift")
            drift_rows = drift_stage(domains, args.datasets, LARGER_DIMS,
                                     base_weights, seed, config)
            (output_dir / "drift_runs.json").write_text(
                json.dumps(drift_rows, indent=2), encoding="utf-8")
            decisions["drift"] = "EWMA + 3-of-5 window alarms recorded"

    # --- write outputs ---------------------------------------------------------
    if diag_rows:
        write_csv(output_dir / "learning_diagnostics.csv", diag_rows)
    if ablation_runs:
        write_csv(output_dir / "learning_ablation_runs.csv", ablation_runs)
        write_csv(output_dir / "learning_ablation_summary.csv",
                  summarize_ablation(ablation_runs))

    payload = {
        "study": "Federated Autoencoder Learning-Rescue Experiment",
        "config": {"datasets": list(args.datasets), "seeds": list(args.seeds),
                   "dims": list(LARGER_DIMS), "rank": args.rank,
                   "alpha_grid": list(alpha_grid)},
        "decisions": decisions,
        "checks": {k: v for k, v in run_correctness_checks(rank=args.rank).items()
                   if not k.startswith("_")},
    }
    build_report(output_dir / "LEARNING_FAILURE_REPORT.md", payload)
    (output_dir / "learning_failure_decisions.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    print("\nLearning-failure experiment complete")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    main()
