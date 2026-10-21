"""Shared ESP32/Python definition of the lightweight LoRA autoencoder.

The constants and initialization order in this module are a contract with
``firmware_p2p_client/nn.cpp``.  Keep changes synchronized.  All matrices are
row-major and the only trainable/transmitted values are the 128 LoRA B values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

INPUT_SIZE = 12
LAYER_DIMS = (12, 16, 8, 4, 8, 16, 12)
LORA_RANK = 2
LORA_ALPHA = np.float32(0.5)
NUM_LAYERS = 6
NUM_WEIGHTS = sum(out_dim * LORA_RANK for out_dim in LAYER_DIMS[1:])

BASE_SEED = 0x4E4E4D31
LORA_A_SEED = 0x4C4F5241
BASE_SCALES = (0.5, 0.5, 0.75, 1.0, 0.75, 0.5)
LORA_A_SCALE = 0.5

ADAM_BETA1 = np.float32(0.9)
ADAM_BETA2 = np.float32(0.999)
ADAM_EPSILON = np.float32(1e-8)
ADAM_WEIGHT_DECAY = np.float32(0.001)
DEFAULT_LR = np.float32(0.02)


def _lcg_values(seed: int, count: int, scale: float) -> tuple[np.ndarray, int]:
    """Return firmware-equivalent signed LCG draws and the final state."""
    state = int(seed) & 0xFFFFFFFF
    out = np.empty(count, dtype=np.float32)
    scale32 = np.float32(scale)
    denominator = np.float32(16777216.0)  # 2**24
    for index in range(count):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        fraction = np.float32(state >> 8) / denominator
        signed = np.float32(np.float32(2.0) * fraction - np.float32(1.0))
        out[index] = np.float32(signed * scale32)
    return out, state


def frozen_parameters() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Create frozen base W and LoRA A matrices in firmware fill order."""
    weights: list[np.ndarray] = []
    state = BASE_SEED
    for in_dim, out_dim, scale in zip(LAYER_DIMS[:-1], LAYER_DIMS[1:], BASE_SCALES):
        values, state = _lcg_values(state, out_dim * in_dim, scale)
        weights.append(values.reshape(out_dim, in_dim))

    projections: list[np.ndarray] = []
    state = LORA_A_SEED
    for in_dim in LAYER_DIMS[:-1]:
        values, state = _lcg_values(state, LORA_RANK * in_dim, LORA_A_SCALE)
        projections.append(values.reshape(LORA_RANK, in_dim))
    return weights, projections


def _round_away_from_zero(values: np.ndarray) -> np.ndarray:
    """Match C/C++ ``roundf`` (NumPy's round uses ties-to-even)."""
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5))


def quantize_dynamic(weights: Sequence[float] | np.ndarray) -> tuple[np.float32, np.ndarray]:
    """Pack B values with the firmware's per-packet dynamic signed-int8 scale."""
    values = np.asarray(weights, dtype=np.float32)
    maximum = np.max(np.abs(values)) if values.size else np.float32(0.0)
    scale = np.float32(max(float(maximum) / 127.0, 1e-7))
    rounded = _round_away_from_zero(values / scale)
    packed = np.clip(rounded, -127.0, 127.0).astype(np.int8)
    return scale, packed


def dequantize_dynamic(packed: Sequence[int] | np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(packed, dtype=np.int8).astype(np.float32) * np.float32(scale)


def broadcast_equivalent(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the values a client actually obtains after a radio broadcast."""
    scale, packed = quantize_dynamic(weights)
    return dequantize_dynamic(packed, scale)


@dataclass
class ForwardResult:
    loss: float
    activations: list[np.ndarray]
    preactivations: list[np.ndarray]
    projections: list[np.ndarray]


class TinyLoRAAutoencoder:
    """12-16-8-4-8-16-12 AE with frozen W/A and trainable rank-2 B."""

    def __init__(self) -> None:
        self.W, self.A = frozen_parameters()
        self.B = [
            np.zeros((out_dim, LORA_RANK), dtype=np.float32)
            for out_dim in LAYER_DIMS[1:]
        ]
        self.m = [np.zeros_like(matrix) for matrix in self.B]
        self.v = [np.zeros_like(matrix) for matrix in self.B]
        self.t = 0

    def copy(self) -> "TinyLoRAAutoencoder":
        clone = TinyLoRAAutoencoder()
        clone.B = [matrix.copy() for matrix in self.B]
        clone.m = [matrix.copy() for matrix in self.m]
        clone.v = [matrix.copy() for matrix in self.v]
        clone.t = self.t
        return clone

    def get_weights(self) -> np.ndarray:
        return np.concatenate([matrix.ravel() for matrix in self.B]).astype(np.float32)

    def set_weights(self, flat_weights: Sequence[float] | np.ndarray, reset_optimizer: bool = False) -> None:
        flat = np.asarray(flat_weights, dtype=np.float32)
        if flat.size != NUM_WEIGHTS:
            raise ValueError(f"expected {NUM_WEIGHTS} B weights, got {flat.size}")
        offset = 0
        for index, matrix in enumerate(self.B):
            size = matrix.size
            self.B[index] = flat[offset : offset + size].reshape(matrix.shape).copy()
            offset += size
        if reset_optimizer:
            self.m = [np.zeros_like(matrix) for matrix in self.B]
            self.v = [np.zeros_like(matrix) for matrix in self.B]
            self.t = 0

    def set_broadcast_weights(self, flat_weights: Sequence[float] | np.ndarray) -> None:
        self.set_weights(broadcast_equivalent(flat_weights), reset_optimizer=True)

    def forward(self, sample: Sequence[float] | np.ndarray, with_cache: bool = False):
        x = np.asarray(sample, dtype=np.float32)
        if x.shape != (INPUT_SIZE,):
            raise ValueError(f"expected one {INPUT_SIZE}-feature sample, got {x.shape}")

        activations = [x]
        preactivations: list[np.ndarray] = []
        projections: list[np.ndarray] = []
        current = x
        for layer, (base, proj_a, proj_b) in enumerate(zip(self.W, self.A, self.B)):
            low_rank_input = proj_a @ current
            z = base @ current + LORA_ALPHA * (proj_b @ low_rank_input)
            z = np.asarray(z, dtype=np.float32)
            current = z if layer == NUM_LAYERS - 1 else np.maximum(z, np.float32(0.0))
            current = np.asarray(current, dtype=np.float32)
            projections.append(low_rank_input)
            preactivations.append(z)
            activations.append(current)

        loss = float(np.mean((current - x) ** 2, dtype=np.float32))
        if with_cache:
            return ForwardResult(loss, activations, preactivations, projections)
        return loss

    def _adamw_update(self, layer: int, gradient: np.ndarray, learning_rate: np.float32) -> None:
        gradient = np.clip(gradient, -1.0, 1.0).astype(np.float32)
        self.m[layer] = ADAM_BETA1 * self.m[layer] + (np.float32(1.0) - ADAM_BETA1) * gradient
        self.v[layer] = ADAM_BETA2 * self.v[layer] + (np.float32(1.0) - ADAM_BETA2) * gradient * gradient
        first_hat = self.m[layer] / np.float32(1.0 - float(ADAM_BETA1) ** self.t)
        second_hat = self.v[layer] / np.float32(1.0 - float(ADAM_BETA2) ** self.t)
        update = first_hat / (np.sqrt(second_hat) + ADAM_EPSILON)
        self.B[layer] -= learning_rate * (update + ADAM_WEIGHT_DECAY * self.B[layer])
        self.B[layer] = self.B[layer].astype(np.float32)

    def train_step(self, sample: Sequence[float] | np.ndarray, learning_rate: float = float(DEFAULT_LR)) -> float:
        """Mirror firmware's reverse pass, including alpha-scaled B gradients.

        All deltas are computed against one immutable pre-update model.  This
        avoids the subtle error of propagating through B6 after B6 has already
        been updated while calculating the B5 gradient.
        """
        cache = self.forward(sample, with_cache=True)
        self.t += 1
        lr = np.float32(learning_rate)

        delta = np.float32(2.0 / INPUT_SIZE) * (cache.activations[-1] - cache.activations[0])
        gradients: list[np.ndarray | None] = [None] * NUM_LAYERS
        for layer in range(NUM_LAYERS - 1, -1, -1):
            gradients[layer] = LORA_ALPHA * np.outer(delta, cache.projections[layer])
            if layer == 0:
                break
            effective = self.W[layer] + LORA_ALPHA * (self.B[layer] @ self.A[layer])
            delta = effective.T @ delta
            delta = delta * (cache.preactivations[layer - 1] > 0.0)
            delta = np.asarray(delta, dtype=np.float32)
        for layer, gradient in enumerate(gradients):
            assert gradient is not None
            self._adamw_update(layer, gradient, lr)
        return cache.loss

    def train(self, samples: np.ndarray, epochs: int = 1, learning_rate: float = float(DEFAULT_LR)) -> list[float]:
        samples = np.asarray(samples, dtype=np.float32)
        losses: list[float] = []
        for _ in range(epochs):
            epoch_losses = [self.train_step(sample, learning_rate) for sample in samples]
            losses.append(float(np.mean(epoch_losses)))
        return losses

    def losses(self, samples: Iterable[Sequence[float]] | np.ndarray) -> np.ndarray:
        return np.asarray([self.forward(sample) for sample in samples], dtype=np.float32)


assert NUM_WEIGHTS == 128
