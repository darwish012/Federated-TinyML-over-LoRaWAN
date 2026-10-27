"""V27 variable-sample equal-client federated anomaly detector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import learning_failure_experiments as lfe
import raw_online_federated_v12 as raw_v12
from learning_failure_experiments import DenseAE

FEATURE_DIM = 100
AE_DIMS = (100, 32, 8, 32, 100)
INPUT_CLIP = 12.0
HEAD_DIM = 3


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIM or len(array) < 1:
        raise ValueError("V27 calibration must be a nonempty 100-D matrix")
    center = np.nanmedian(array, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(array - center), axis=0)
    std = np.nanstd(array, axis=0)
    scale = np.where(np.isfinite(mad) & (mad > 1e-6), mad, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    center = np.where(np.isfinite(center), center, 0.0)
    return center.astype(np.float32), scale.astype(np.float32)


def normalize(
    values: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    array = (np.asarray(values, dtype=np.float32) - center) / scale
    return np.clip(array, -INPUT_CLIP, INPUT_CLIP).astype(np.float32)


def scalar_center_scale(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        raise ValueError("empty score calibration")
    center = float(np.nanmedian(array))
    mad = 1.4826 * float(np.nanmedian(np.abs(array - center)))
    std = float(np.nanstd(array))
    return center, max(mad if np.isfinite(mad) else 0.0, std * 0.1, 1e-6)


def reconstruct(model: DenseAE, values: np.ndarray) -> np.ndarray:
    output = []
    for value in np.asarray(values, dtype=np.float32):
        _, activations, _, _ = lfe._ae_forward(value, model.W)
        output.append(np.asarray(activations[-1], dtype=np.float32))
    return np.asarray(output, dtype=np.float32)


def head_features(model: DenseAE, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    rebuilt = reconstruct(model, array)
    residual = np.abs(array - rebuilt)
    return np.column_stack(
        [
            np.log1p(np.mean(residual * residual, axis=1)),
            np.mean(np.abs(array), axis=1),
            np.max(np.abs(array), axis=1),
        ]
    ).astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def empirical_normal_nuisance(
    clean: np.ndarray, *, seed: int
) -> np.ndarray:
    """Health-preserving noise bootstrapped from measured local differences."""

    values = np.asarray(clean, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    if len(values) >= 2:
        differences = np.diff(values, axis=0)
    else:
        differences = rng.normal(0.0, 0.05, size=(1, FEATURE_DIM))
    result = []
    for value in values:
        residual = differences[int(rng.integers(0, len(differences)))]
        scale = float(rng.uniform(0.20, 0.60))
        result.append(np.clip(value + scale * residual, -INPUT_CLIP, INPUT_CLIP))
    return np.asarray(result, dtype=np.float32)


def generalized_pseudo_anomalies(
    clean: np.ndarray, *, seed: int
) -> np.ndarray:
    """Create unlabeled generalized corruptions, never reconstruction targets."""

    values = np.asarray(clean, dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    result = []
    for index, value in enumerate(values):
        first = values[int(rng.integers(0, len(values)))]
        second = values[int(rng.integers(0, len(values)))]
        gamma = float(rng.uniform(2.0, 4.0))
        pseudo = value + gamma * (first - second)
        count = max(5, int(round(FEATURE_DIM * rng.uniform(0.10, 0.30))))
        coordinates = rng.choice(FEATURE_DIM, size=count, replace=False)
        pseudo[coordinates] += rng.standard_t(df=3, size=count) * float(
            rng.uniform(2.0, 5.0)
        )
        result.append(np.clip(pseudo, -INPUT_CLIP, INPUT_CLIP))
    return np.asarray(result, dtype=np.float32)


class MeanGradientAE(DenseAE):
    def mean_step(
        self, model_input: np.ndarray, target: np.ndarray, learning_rate: float
    ) -> float:
        inputs = np.asarray(model_input, dtype=np.float32)
        targets = np.asarray(target, dtype=np.float32)
        if inputs.shape != targets.shape or inputs.ndim != 2:
            raise ValueError("V27 AE input/target matrices must match")
        gradient_sums = [np.zeros_like(weight) for weight in self.W]
        losses = []
        for value, expected in zip(inputs, targets):
            loss, _, _, _ = lfe._ae_forward(value, self.W, target=expected)
            gradients, _, _ = lfe._analytic_gradients(
                value, self.W, target=expected
            )
            losses.append(float(loss))
            for layer, gradient in enumerate(gradients):
                gradient_sums[layer] += gradient
        mean_gradients = [
            gradient / max(len(inputs), 1) for gradient in gradient_sums
        ]
        self.W = self._opt.step(self.W, mean_gradients, float(learning_rate))
        return float(np.mean(losses))


@dataclass(frozen=True)
class ClientData:
    client_id: str
    calibration: np.ndarray
    local_fitting: np.ndarray

    def normalized(self) -> np.ndarray:
        center, scale = robust_center_scale(self.calibration)
        return normalize(self.local_fitting, center, scale)


@dataclass(frozen=True)
class FederatedModel:
    autoencoder: DenseAE
    head_weight: np.ndarray
    head_bias: float


def _head_step(
    weight: np.ndarray,
    bias: float,
    features: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
) -> tuple[np.ndarray, float, float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    probabilities = sigmoid(x @ weight + bias)
    error = probabilities - y
    grad_weight = x.T @ error / max(len(x), 1)
    grad_bias = float(np.mean(error))
    updated_weight = np.asarray(weight - learning_rate * grad_weight, dtype=np.float32)
    updated_bias = float(bias - learning_rate * grad_bias)
    epsilon = 1e-9
    loss = -float(
        np.mean(
            y * np.log(probabilities + epsilon)
            + (1.0 - y) * np.log(1.0 - probabilities + epsilon)
        )
    )
    return updated_weight, updated_bias, loss


def local_update(
    global_model: FederatedModel,
    client: ClientData,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> tuple[FederatedModel, dict[str, float]]:
    clean = client.normalized()
    if not len(clean):
        raise ValueError(f"empty local fitting segment: {client.client_id}")
    model = MeanGradientAE(AE_DIMS, weights=global_model.autoencoder.W)
    weight = np.asarray(global_model.head_weight, dtype=np.float32).copy()
    bias = float(global_model.head_bias)
    ae_losses: list[float] = []
    head_losses: list[float] = []
    for epoch in range(int(epochs)):
        nuisance = empirical_normal_nuisance(
            clean, seed=int(seed) + 1009 * (epoch + 1)
        )
        inputs = np.concatenate([clean, nuisance], axis=0)
        targets = np.concatenate([clean, clean], axis=0)
        ae_losses.append(model.mean_step(inputs, targets, learning_rate))
        pseudo = generalized_pseudo_anomalies(
            clean, seed=int(seed) + 2027 * (epoch + 1)
        )
        normal_features = head_features(model, clean)
        pseudo_features = head_features(model, pseudo)
        classifier_features = np.concatenate(
            [normal_features, pseudo_features], axis=0
        )
        labels = np.concatenate(
            [np.zeros(len(clean)), np.ones(len(pseudo))], axis=0
        )
        weight, bias, loss = _head_step(
            weight,
            bias,
            classifier_features,
            labels,
            learning_rate=learning_rate,
        )
        head_losses.append(loss)
    return (
        FederatedModel(model, weight, bias),
        {
            "ae_loss": float(np.mean(ae_losses)),
            "head_loss": float(np.mean(head_losses)),
            "local_samples": float(len(clean)),
        },
    )


def train_federated(
    clients: Sequence[ClientData],
    *,
    seed: int,
    rounds: int,
    epochs: int,
    learning_rate: float,
) -> tuple[FederatedModel, list[dict[str, float]]]:
    if not clients or len({client.client_id for client in clients}) != len(clients):
        raise ValueError("V27 needs unique nonempty source clients")
    weights = raw_v12._init_ae_weights(int(seed))
    global_model = FederatedModel(
        DenseAE(AE_DIMS, weights=weights),
        np.zeros(HEAD_DIM, dtype=np.float32),
        0.0,
    )
    trace = []
    for round_index in range(1, int(rounds) + 1):
        local_models = []
        diagnostics = []
        for client_index, client in enumerate(clients):
            local, diagnostic = local_update(
                global_model,
                client,
                seed=int(seed) + round_index * 100003 + client_index * 997,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            local_models.append(local)
            diagnostics.append(diagnostic)
        averaged_weights = [
            np.mean(
                np.stack(
                    [local.autoencoder.W[layer] for local in local_models],
                    axis=0,
                ),
                axis=0,
            ).astype(np.float32)
            for layer in range(len(AE_DIMS) - 1)
        ]
        global_model = FederatedModel(
            DenseAE(AE_DIMS, weights=averaged_weights),
            np.mean(
                np.stack([local.head_weight for local in local_models]), axis=0
            ).astype(np.float32),
            float(np.mean([local.head_bias for local in local_models])),
        )
        trace.append(
            {
                "round": float(round_index),
                "clients": float(len(clients)),
                "mean_ae_loss": float(
                    np.mean([item["ae_loss"] for item in diagnostics])
                ),
                "mean_head_loss": float(
                    np.mean([item["head_loss"] for item in diagnostics])
                ),
                "minimum_local_samples": float(
                    min(item["local_samples"] for item in diagnostics)
                ),
                "maximum_local_samples": float(
                    max(item["local_samples"] for item in diagnostics)
                ),
            }
        )
    return global_model, trace


def personalize(
    global_model: FederatedModel,
    client: ClientData,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> FederatedModel:
    model, _ = local_update(
        global_model,
        client,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    return model


class PersonalizedScorer:
    def __init__(
        self,
        model: FederatedModel,
        input_calibration: np.ndarray,
        local_fitting: np.ndarray,
        score_calibration: np.ndarray,
    ):
        self.model = model
        self.center, self.scale = robust_center_scale(input_calibration)
        fit = normalize(local_fitting, self.center, self.scale)
        self.pca_center = np.mean(fit, axis=0)
        _, _, vt = np.linalg.svd(fit - self.pca_center, full_matrices=False)
        self.pca_basis = vt[: min(8, len(vt))].astype(np.float32)
        calibration = normalize(score_calibration, self.center, self.scale)
        raw = self._raw_heads(calibration)
        self.head_calibration = [scalar_center_scale(column) for column in raw.T]
        fused = self._fuse(raw)
        self.fused_center, self.fused_scale = scalar_center_scale(fused)

    def _raw_heads(self, normalized: np.ndarray) -> np.ndarray:
        ae_features = head_features(self.model.autoencoder, normalized)
        ae = ae_features[:, 0]
        centered = normalized - self.pca_center
        rebuilt = (centered @ self.pca_basis.T) @ self.pca_basis
        pca = np.log1p(np.mean((centered - rebuilt) ** 2, axis=1))
        pseudo_probability = sigmoid(
            ae_features @ self.model.head_weight + self.model.head_bias
        )
        return np.column_stack([ae, pca, pseudo_probability])

    def _fuse(self, raw: np.ndarray) -> np.ndarray:
        standardized = []
        for index, (center, scale) in enumerate(self.head_calibration):
            standardized.append(
                np.clip((raw[:, index] - center) / scale, -12.0, 30.0)
            )
        return np.max(np.column_stack(standardized), axis=1)

    def scores(self, values: np.ndarray) -> np.ndarray:
        normalized = normalize(values, self.center, self.scale)
        return self._fuse(self._raw_heads(normalized))


def persistent_latch(scores: np.ndarray, threshold: float) -> np.ndarray:
    flags = np.asarray(scores, dtype=np.float64) > float(threshold)
    latched = np.zeros(len(flags), dtype=np.int8)
    is_latched = False
    for index in range(len(flags)):
        start = max(0, index - 4)
        if index - start + 1 >= 5 and int(np.sum(flags[start : index + 1])) >= 3:
            is_latched = True
        latched[index] = int(is_latched)
    return latched


def first_alarm(indices: np.ndarray, latched: np.ndarray) -> int:
    positions = np.flatnonzero(latched)
    return int(indices[positions[0]]) if len(positions) else -1


__all__ = [
    "AE_DIMS",
    "ClientData",
    "FederatedModel",
    "PersonalizedScorer",
    "empirical_normal_nuisance",
    "first_alarm",
    "generalized_pseudo_anomalies",
    "head_features",
    "normalize",
    "persistent_latch",
    "personalize",
    "robust_center_scale",
    "train_federated",
]
