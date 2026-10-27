"""V29 representation, calibration, and causal alarm primitives.

The module is deliberately label-free.  It accepts already-causal 100-D feature
streams and healthy allocation slices, but it has no access to degradation
onsets, failure labels, final lifetime, or target outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

import raw_online_federated_v12 as raw_v12
import raw_online_federated_v27_detector as v27d


FEATURE_DIM = 100
AE_DIMS = (100, 32, 8, 32, 100)
SVDD_DIMS = (100, 32, 8)
PREDICTOR_DIMS = (100, 32, 100)
RANDOM_SEEDS = (101, 211, 307, 401, 503)
EPS = 1e-9
SIMPLE_RULE_INDICES = np.asarray(
    # Within each 32-D channel summary, time-domain RMS, kurtosis, and
    # crest-factor are coordinates 0, 2, and 4.  V12 concatenates mean,
    # across-channel spread, and extreme summaries at offsets 0, 32, and 64.
    [0, 2, 4, 32, 34, 36, 64, 66, 68],
    dtype=np.int64,
)

SINGLE_ARMS = (
    "LOCAL_PCA16",
    "LOCAL_PCA16_FEWSHOT25",
    "FED_PCA16",
    "RANDOM_AE_MEDIAN",
    "LOCAL_AE_RECON",
    "LOCAL_AE_RECON_FEWSHOT25",
    "FED_AE_RECON",
    "FED_AE_PERSONALIZED_RECON",
    "LOCAL_AE_PSEUDO_HEAD",
    "FED_AE_PSEUDO_HEAD",
    "FED_AE_PERSONALIZED_PSEUDO_HEAD",
    "LOCAL_DEEP_SVDD",
    "FED_DEEP_SVDD",
    "FED_SYNTHETIC_SVDD",
    "LOCAL_NEXT_PREDICTOR",
    "FED_NEXT_PREDICTOR",
    "INPUT_PROFILE_RULE",
    "RMS_KURTOSIS_CREST_RULE",
)

SYMMETRIC_ARMS = {
    "FED_AE_PCA_SYMMETRIC": ("FED_PCA16", "FED_AE_RECON"),
    "FED_PERSONALIZED_AE_PCA_SYMMETRIC": (
        "LOCAL_PCA16",
        "FED_AE_PERSONALIZED_RECON",
    ),
    "FED_SVDD_PCA_SYMMETRIC": ("FED_PCA16", "FED_DEEP_SVDD"),
    "FED_SYNTHETIC_SVDD_PCA_SYMMETRIC": (
        "FED_PCA16",
        "FED_SYNTHETIC_SVDD",
    ),
    "FED_NEXT_PREDICTOR_PCA_SYMMETRIC": (
        "FED_PCA16",
        "FED_NEXT_PREDICTOR",
    ),
}

CALIBRATION_VARIANTS = (
    "ROBUST_Z",
    "EMPIRICAL_TAIL",
    "ROBUST_Z_EWMA",
    "ROBUST_Z_CUSUM",
)


def _finite_matrix(values: np.ndarray, *, columns: int = FEATURE_DIM) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if (
        array.ndim != 2
        or array.shape[1] != columns
        or len(array) < 1
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"expected finite nonempty matrix with {columns} columns")
    return array


def robust_center_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return v27d.robust_center_scale(_finite_matrix(values))


def normalize(
    values: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return v27d.normalize(values, center, scale)


@dataclass(frozen=True)
class PCAModel:
    center: np.ndarray
    basis: np.ndarray

    def score(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        centered = array - self.center
        rebuilt = (centered @ self.basis.T) @ self.basis
        return np.log1p(np.mean((centered - rebuilt) ** 2, axis=1)).astype(
            np.float64
        )


def fit_pca(values: np.ndarray, components: int = 16) -> PCAModel:
    array = _finite_matrix(values)
    center = np.mean(array, axis=0)
    _, _, vt = np.linalg.svd(array - center, full_matrices=False)
    basis = vt[: min(int(components), len(vt))].astype(np.float32)
    return PCAModel(center.astype(np.float32), basis)


def fit_federated_pca(
    client_values: Sequence[np.ndarray], components: int = 16
) -> PCAModel:
    """Equal-client PCA from local first and second moments.

    Every client contributes exactly one mean and one covariance regardless of
    its number of healthy acquisitions.
    """

    clients = [_finite_matrix(values) for values in client_values]
    if not clients:
        raise ValueError("federated PCA needs at least one client")
    means = [np.mean(values, axis=0, dtype=np.float64) for values in clients]
    global_mean = np.mean(np.stack(means), axis=0)
    second_moments = []
    for values in clients:
        x = np.asarray(values, dtype=np.float64)
        second_moments.append((x.T @ x) / len(x))
    second = np.mean(np.stack(second_moments), axis=0)
    covariance = second - np.outer(global_mean, global_mean)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][: int(components)]
    basis = eigenvectors[:, order].T.astype(np.float32)
    return PCAModel(global_mean.astype(np.float32), basis)


def _xavier_weights(dims: Sequence[int], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(int(seed))
    result = []
    for incoming, outgoing in zip(dims[:-1], dims[1:]):
        limit = np.sqrt(6.0 / (incoming + outgoing))
        result.append(
            rng.uniform(-limit, limit, size=(outgoing, incoming)).astype(
                np.float32
            )
        )
    return result


def _copy_weights(weights: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(weight, dtype=np.float32).copy() for weight in weights]


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=np.float32), 0.0)


def _leaky_relu(values: np.ndarray, slope: float = 0.1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return np.where(array >= 0.0, array, slope * array).astype(np.float32)


def _forward_dense(
    values: np.ndarray,
    weights: Sequence[np.ndarray],
    *,
    leaky: bool = False,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    current = np.asarray(values, dtype=np.float32)
    activations = [current]
    preactivations: list[np.ndarray] = []
    for index, weight in enumerate(weights):
        z = current @ np.asarray(weight, dtype=np.float32).T
        preactivations.append(z)
        if index == len(weights) - 1:
            current = z.astype(np.float32)
        else:
            current = _leaky_relu(z) if leaky else _relu(z)
        activations.append(current)
    return current, activations, preactivations


def _clip_gradients(
    gradients: Sequence[np.ndarray], maximum_norm: float = 10.0
) -> list[np.ndarray]:
    total = float(
        np.sqrt(sum(float(np.sum(gradient * gradient)) for gradient in gradients))
    )
    scale = 1.0 if total <= maximum_norm else maximum_norm / max(total, EPS)
    return [
        (np.asarray(gradient, dtype=np.float32) * scale).astype(np.float32)
        for gradient in gradients
    ]


def _mse_gradients(
    model_input: np.ndarray,
    target: np.ndarray,
    weights: Sequence[np.ndarray],
    *,
    leaky: bool = False,
    weight_decay: float = 0.0,
) -> tuple[float, list[np.ndarray]]:
    output, activations, preactivations = _forward_dense(
        model_input, weights, leaky=leaky
    )
    expected = np.asarray(target, dtype=np.float32)
    difference = output - expected
    loss = float(np.mean(difference * difference))
    delta = (2.0 / max(difference.size, 1)) * difference
    gradients = [np.zeros_like(weight) for weight in weights]
    for layer in range(len(weights) - 1, -1, -1):
        gradients[layer] = (
            delta.T @ activations[layer]
            + float(weight_decay) * np.asarray(weights[layer])
        ).astype(np.float32)
        if layer > 0:
            delta = delta @ np.asarray(weights[layer], dtype=np.float32)
            z = preactivations[layer - 1]
            derivative = (
                np.where(z >= 0.0, 1.0, 0.1)
                if leaky
                else (z > 0.0).astype(np.float32)
            )
            delta = delta * derivative
    return loss, _clip_gradients(gradients)


def _mini_batches(
    count: int, batch_size: int, rng: np.random.Generator
) -> Iterable[np.ndarray]:
    order = rng.permutation(int(count))
    for start in range(0, len(order), int(batch_size)):
        yield order[start : start + int(batch_size)]


@dataclass(frozen=True)
class AEModel:
    weights: tuple[np.ndarray, ...]
    head_weight: np.ndarray
    head_bias: float

    def reconstruct(self, values: np.ndarray) -> np.ndarray:
        output, _, _ = _forward_dense(values, self.weights)
        return output

    def head_features(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        residual = np.abs(array - self.reconstruct(array))
        return np.column_stack(
            [
                np.log1p(np.mean(residual * residual, axis=1)),
                np.mean(np.abs(array), axis=1),
                np.max(np.abs(array), axis=1),
            ]
        ).astype(np.float32)

    def reconstruction_score(self, values: np.ndarray) -> np.ndarray:
        return self.head_features(values)[:, 0].astype(np.float64)

    def pseudo_probability(self, values: np.ndarray) -> np.ndarray:
        logits = self.head_features(values) @ self.head_weight + self.head_bias
        logits = np.clip(logits, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)


def initial_ae(seed: int) -> AEModel:
    weights = tuple(
        np.asarray(weight, dtype=np.float32)
        for weight in raw_v12._init_ae_weights(int(seed))
    )
    return AEModel(weights, np.zeros(3, dtype=np.float32), 0.0)


def _empirical_nuisance(clean: np.ndarray, seed: int) -> np.ndarray:
    return v27d.empirical_normal_nuisance(clean, seed=int(seed))


def _pseudo_anomalies(clean: np.ndarray, seed: int) -> np.ndarray:
    return v27d.generalized_pseudo_anomalies(clean, seed=int(seed))


def train_ae_local(
    initial: AEModel,
    clean: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[AEModel, dict[str, float]]:
    values = _finite_matrix(clean)
    weights = _copy_weights(initial.weights)
    head_weight = np.asarray(initial.head_weight, dtype=np.float32).copy()
    head_bias = float(initial.head_bias)
    rng = np.random.default_rng(int(seed))
    reconstruction_losses: list[float] = []
    head_losses: list[float] = []
    for epoch in range(int(epochs)):
        nuisance = _empirical_nuisance(
            values, int(seed) + 1009 * (epoch + 1)
        )
        model_input = np.concatenate([values, nuisance], axis=0)
        target = np.concatenate([values, values], axis=0)
        for batch in _mini_batches(len(model_input), batch_size, rng):
            loss, gradients = _mse_gradients(
                model_input[batch],
                target[batch],
                weights,
                weight_decay=weight_decay,
            )
            weights = [
                (
                    weight - float(learning_rate) * gradient
                ).astype(np.float32)
                for weight, gradient in zip(weights, gradients)
            ]
            reconstruction_losses.append(loss)

        current = AEModel(tuple(weights), head_weight, head_bias)
        pseudo = _pseudo_anomalies(
            values, int(seed) + 2027 * (epoch + 1)
        )
        x = np.concatenate(
            [current.head_features(values), current.head_features(pseudo)],
            axis=0,
        ).astype(np.float64)
        y = np.concatenate(
            [np.zeros(len(values)), np.ones(len(pseudo))]
        ).astype(np.float64)
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(x @ head_weight + head_bias, -40.0, 40.0))
        )
        error = probability - y
        grad_weight = x.T @ error / len(x)
        grad_bias = float(np.mean(error))
        head_weight = (
            head_weight - float(learning_rate) * grad_weight
        ).astype(np.float32)
        head_bias -= float(learning_rate) * grad_bias
        head_losses.append(
            -float(
                np.mean(
                    y * np.log(probability + EPS)
                    + (1.0 - y) * np.log(1.0 - probability + EPS)
                )
            )
        )
    return (
        AEModel(tuple(weights), head_weight, head_bias),
        {
            "reconstruction_loss": float(np.mean(reconstruction_losses)),
            "pseudo_head_loss": float(np.mean(head_losses)),
            "samples": float(len(values)),
        },
    )


def _mean_ae(models: Sequence[AEModel]) -> AEModel:
    if not models:
        raise ValueError("cannot average no autoencoders")
    weights = tuple(
        np.mean(
            np.stack([model.weights[layer] for model in models]), axis=0
        ).astype(np.float32)
        for layer in range(len(models[0].weights))
    )
    return AEModel(
        weights,
        np.mean(np.stack([model.head_weight for model in models]), axis=0).astype(
            np.float32
        ),
        float(np.mean([model.head_bias for model in models])),
    )


def train_federated_ae(
    clients: Sequence[np.ndarray],
    *,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[AEModel, list[dict[str, float]]]:
    values = [_finite_matrix(client) for client in clients]
    model = initial_ae(seed)
    trace = []
    for round_index in range(int(rounds)):
        local_models = []
        local_diagnostics = []
        for client_index, client in enumerate(values):
            local, diagnostic = train_ae_local(
                model,
                client,
                epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=int(seed)
                + (round_index + 1) * 100003
                + client_index * 997,
            )
            local_models.append(local)
            local_diagnostics.append(diagnostic)
        model = _mean_ae(local_models)
        trace.append(
            {
                "round": round_index + 1,
                "clients": len(values),
                "mean_reconstruction_loss": float(
                    np.mean(
                        [
                            item["reconstruction_loss"]
                            for item in local_diagnostics
                        ]
                    )
                ),
                "mean_pseudo_head_loss": float(
                    np.mean(
                        [item["pseudo_head_loss"] for item in local_diagnostics]
                    )
                ),
            }
        )
    return model, trace


@dataclass(frozen=True)
class SVDDModel:
    weights: tuple[np.ndarray, ...]
    center: np.ndarray

    def embed(self, values: np.ndarray) -> np.ndarray:
        output, _, _ = _forward_dense(values, self.weights, leaky=True)
        return output

    def score(self, values: np.ndarray) -> np.ndarray:
        difference = self.embed(values) - self.center
        return np.mean(difference * difference, axis=1).astype(np.float64)


def _svdd_initial_center(
    weights: Sequence[np.ndarray], clients: Sequence[np.ndarray]
) -> np.ndarray:
    per_client = []
    for client in clients:
        output, _, _ = _forward_dense(client, weights, leaky=True)
        per_client.append(np.mean(output, axis=0))
    center = np.mean(np.stack(per_client), axis=0).astype(np.float32)
    small = np.abs(center) < 0.1
    center[small & (center < 0)] = -0.1
    center[small & (center >= 0)] = 0.1
    return center


def _svdd_gradients(
    normal: np.ndarray,
    weights: Sequence[np.ndarray],
    center: np.ndarray,
    *,
    pseudo: np.ndarray | None,
    margin: float,
    variance_floor: float,
    weight_decay: float,
) -> tuple[float, list[np.ndarray], float]:
    z, activations, preactivations = _forward_dense(
        normal, weights, leaky=True
    )
    difference = z - center
    normal_loss = float(np.mean(difference * difference))
    delta = 2.0 * difference / max(difference.size, 1)
    variance = np.var(z, axis=0)
    low = variance < float(variance_floor)
    variance_loss = float(np.sum(np.maximum(variance_floor - variance, 0.0)))
    if np.any(low):
        centered = z - np.mean(z, axis=0, keepdims=True)
        delta[:, low] += (
            -0.2
            * 2.0
            * centered[:, low]
            / max(len(z), 1)
        )
    pseudo_loss = 0.0
    if pseudo is not None and len(pseudo):
        z_pseudo, pseudo_activations, pseudo_preactivations = _forward_dense(
            pseudo, weights, leaky=True
        )
        pseudo_diff = z_pseudo - center
        pseudo_distance = np.mean(pseudo_diff * pseudo_diff, axis=1)
        active = pseudo_distance < float(margin)
        pseudo_loss = float(
            np.mean(np.maximum(float(margin) - pseudo_distance, 0.0))
        )
        pseudo_delta = np.zeros_like(z_pseudo)
        if np.any(active):
            pseudo_delta[active] = (
                -2.0
                * pseudo_diff[active]
                / max(np.sum(active) * pseudo_diff.shape[1], 1)
            )
        delta *= 0.5
        pseudo_delta *= 0.5
    else:
        z_pseudo = None
        pseudo_activations = None
        pseudo_preactivations = None
        pseudo_delta = None

    gradients = [np.zeros_like(weight) for weight in weights]

    def accumulate(
        current_delta: np.ndarray,
        current_activations: Sequence[np.ndarray],
        current_preactivations: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        local = [np.zeros_like(weight) for weight in weights]
        d = current_delta
        for layer in range(len(weights) - 1, -1, -1):
            local[layer] = (d.T @ current_activations[layer]).astype(np.float32)
            if layer > 0:
                d = d @ np.asarray(weights[layer], dtype=np.float32)
                derivative = np.where(
                    current_preactivations[layer - 1] >= 0.0, 1.0, 0.1
                )
                d = d * derivative
        return local

    normal_gradients = accumulate(delta, activations, preactivations)
    if pseudo_delta is not None:
        pseudo_gradients = accumulate(
            pseudo_delta, pseudo_activations, pseudo_preactivations
        )
    else:
        pseudo_gradients = [np.zeros_like(weight) for weight in weights]
    for layer, weight in enumerate(weights):
        gradients[layer] = (
            normal_gradients[layer]
            + pseudo_gradients[layer]
            + float(weight_decay) * np.asarray(weight)
        ).astype(np.float32)
    loss = normal_loss + 0.2 * variance_loss + 0.5 * pseudo_loss
    return loss, _clip_gradients(gradients), float(np.mean(variance))


def train_svdd_local(
    initial: SVDDModel,
    clean: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    synthetic_margin: float | None,
    variance_floor: float,
) -> tuple[SVDDModel, dict[str, float]]:
    values = _finite_matrix(clean)
    weights = _copy_weights(initial.weights)
    rng = np.random.default_rng(int(seed))
    losses = []
    variances = []
    for epoch in range(int(epochs)):
        pseudo_all = (
            _pseudo_anomalies(values, int(seed) + 3001 * (epoch + 1))
            if synthetic_margin is not None
            else None
        )
        for batch in _mini_batches(len(values), batch_size, rng):
            pseudo = None if pseudo_all is None else pseudo_all[batch]
            loss, gradients, variance = _svdd_gradients(
                values[batch],
                weights,
                initial.center,
                pseudo=pseudo,
                margin=(
                    float(synthetic_margin)
                    if synthetic_margin is not None
                    else 0.0
                ),
                variance_floor=variance_floor,
                weight_decay=weight_decay,
            )
            weights = [
                (weight - float(learning_rate) * gradient).astype(np.float32)
                for weight, gradient in zip(weights, gradients)
            ]
            losses.append(loss)
            variances.append(variance)
    model = SVDDModel(tuple(weights), initial.center.copy())
    final_variance = float(np.mean(np.var(model.embed(values), axis=0)))
    return model, {
        "loss": float(np.mean(losses)),
        "batch_latent_variance": float(np.mean(variances)),
        "final_latent_variance": final_variance,
        "samples": float(len(values)),
    }


def initial_svdd(
    clients: Sequence[np.ndarray], *, seed: int
) -> SVDDModel:
    values = [_finite_matrix(client) for client in clients]
    weights = _xavier_weights(SVDD_DIMS, seed)
    center = _svdd_initial_center(weights, values)
    return SVDDModel(tuple(weights), center)


def _mean_svdd(models: Sequence[SVDDModel], center: np.ndarray) -> SVDDModel:
    return SVDDModel(
        tuple(
            np.mean(
                np.stack([model.weights[layer] for model in models]), axis=0
            ).astype(np.float32)
            for layer in range(len(models[0].weights))
        ),
        np.asarray(center, dtype=np.float32).copy(),
    )


def train_federated_svdd(
    clients: Sequence[np.ndarray],
    *,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    synthetic_margin: float | None,
    variance_floor: float,
) -> tuple[SVDDModel, list[dict[str, float]]]:
    values = [_finite_matrix(client) for client in clients]
    model = initial_svdd(values, seed=seed)
    trace = []
    for round_index in range(int(rounds)):
        local_models = []
        diagnostics = []
        for client_index, client in enumerate(values):
            local, diagnostic = train_svdd_local(
                model,
                client,
                epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=int(seed)
                + (round_index + 1) * 200003
                + client_index * 991,
                synthetic_margin=synthetic_margin,
                variance_floor=variance_floor,
            )
            local_models.append(local)
            diagnostics.append(diagnostic)
        model = _mean_svdd(local_models, model.center)
        equal_client_variance = float(
            np.mean(
                [
                    np.mean(np.var(model.embed(client), axis=0))
                    for client in values
                ]
            )
        )
        trace.append(
            {
                "round": round_index + 1,
                "clients": len(values),
                "mean_loss": float(
                    np.mean([item["loss"] for item in diagnostics])
                ),
                "equal_client_latent_variance": equal_client_variance,
            }
        )
    return model, trace


@dataclass(frozen=True)
class PredictorModel:
    weights: tuple[np.ndarray, ...]

    def predict(self, values: np.ndarray) -> np.ndarray:
        output, _, _ = _forward_dense(values, self.weights)
        return output

    def score_pairs(
        self, previous: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        difference = self.predict(previous) - np.asarray(
            current, dtype=np.float32
        )
        return np.log1p(np.mean(difference * difference, axis=1)).astype(
            np.float64
        )


def initial_predictor(seed: int) -> PredictorModel:
    return PredictorModel(tuple(_xavier_weights(PREDICTOR_DIMS, seed)))


def train_predictor_local(
    initial: PredictorModel,
    clean: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[PredictorModel, dict[str, float]]:
    values = _finite_matrix(clean)
    if len(values) < 2:
        raise ValueError("next-acquisition predictor needs at least two samples")
    model_input = values[:-1]
    target = values[1:]
    weights = _copy_weights(initial.weights)
    rng = np.random.default_rng(int(seed))
    losses = []
    for _ in range(int(epochs)):
        for batch in _mini_batches(len(model_input), batch_size, rng):
            loss, gradients = _mse_gradients(
                model_input[batch],
                target[batch],
                weights,
                weight_decay=weight_decay,
            )
            weights = [
                (weight - float(learning_rate) * gradient).astype(np.float32)
                for weight, gradient in zip(weights, gradients)
            ]
            losses.append(loss)
    return PredictorModel(tuple(weights)), {
        "loss": float(np.mean(losses)),
        "pairs": float(len(model_input)),
    }


def _mean_predictor(models: Sequence[PredictorModel]) -> PredictorModel:
    return PredictorModel(
        tuple(
            np.mean(
                np.stack([model.weights[layer] for model in models]), axis=0
            ).astype(np.float32)
            for layer in range(len(models[0].weights))
        )
    )


def train_federated_predictor(
    clients: Sequence[np.ndarray],
    *,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[PredictorModel, list[dict[str, float]]]:
    values = [_finite_matrix(client) for client in clients]
    model = initial_predictor(seed)
    trace = []
    for round_index in range(int(rounds)):
        local_models = []
        diagnostics = []
        for client_index, client in enumerate(values):
            local, diagnostic = train_predictor_local(
                model,
                client,
                epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                seed=int(seed)
                + (round_index + 1) * 300007
                + client_index * 983,
            )
            local_models.append(local)
            diagnostics.append(diagnostic)
        model = _mean_predictor(local_models)
        trace.append(
            {
                "round": round_index + 1,
                "clients": len(values),
                "mean_loss": float(
                    np.mean([item["loss"] for item in diagnostics])
                ),
            }
        )
    return model, trace


@dataclass(frozen=True)
class GlobalBundle:
    federated_pca: PCAModel
    federated_ae: AEModel
    federated_svdd: SVDDModel
    synthetic_svdd: SVDDModel
    federated_predictor: PredictorModel


def scalar_center_scale(values: np.ndarray) -> tuple[float, float]:
    return v27d.scalar_center_scale(np.asarray(values, dtype=np.float64))


def robust_z(
    calibration: np.ndarray, values: np.ndarray
) -> np.ndarray:
    center, scale = scalar_center_scale(calibration)
    return np.clip(
        (np.asarray(values, dtype=np.float64) - center) / scale,
        -12.0,
        30.0,
    )


def empirical_tail(
    calibration: np.ndarray, values: np.ndarray
) -> np.ndarray:
    reference = np.sort(np.asarray(calibration, dtype=np.float64).reshape(-1))
    current = np.asarray(values, dtype=np.float64).reshape(-1)
    greater_equal = len(reference) - np.searchsorted(
        reference, current, side="left"
    )
    pvalue = (1.0 + greater_equal) / (len(reference) + 1.0)
    return -np.log10(np.maximum(pvalue, 1.0 / (len(reference) + 1.0)))


def ewma(values: np.ndarray, smoothing: float = 0.1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.empty_like(array)
    state = 0.0
    for index, value in enumerate(array):
        state = float(smoothing) * float(value) + (
            1.0 - float(smoothing)
        ) * state
        output[index] = state
    return output


def positive_cusum(values: np.ndarray, allowance: float = 0.5) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.empty_like(array)
    state = 0.0
    for index, value in enumerate(array):
        state = max(0.0, state + float(value) - float(allowance))
        output[index] = state
    return output


def calibration_variants(
    raw_calibration: np.ndarray, raw_values: np.ndarray
) -> dict[str, np.ndarray]:
    z = robust_z(raw_calibration, raw_values)
    return {
        "ROBUST_Z": z,
        "EMPIRICAL_TAIL": empirical_tail(raw_calibration, raw_values),
        "ROBUST_Z_EWMA": ewma(z, 0.1),
        "ROBUST_Z_CUSUM": positive_cusum(z, 0.5),
    }


def k_of_m_evidence(
    scores: np.ndarray, threshold: float, k: int, m: int
) -> np.ndarray:
    array = np.asarray(scores, dtype=np.float64)
    flags = array > float(threshold)
    result = np.zeros(len(flags), dtype=np.int8)
    for end in range(int(m) - 1, len(flags)):
        result[end] = int(
            np.sum(flags[end - int(m) + 1 : end + 1]) >= int(k)
        )
    return result


def first_alarm(
    evidence: np.ndarray, indices: np.ndarray
) -> int:
    positions = np.flatnonzero(np.asarray(evidence, dtype=np.int8))
    return int(indices[positions[0]]) if len(positions) else -1


def symmetric_flags(
    first: np.ndarray,
    second: np.ndarray,
    *,
    moderate: float,
    strong: float,
) -> np.ndarray:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("symmetric branches must have matching shapes")
    return (
        (a > float(strong))
        | (b > float(strong))
        | ((a > float(moderate)) & (b > float(moderate)))
    ).astype(np.int8)


def k_of_m_flags(flags: np.ndarray, k: int, m: int) -> np.ndarray:
    array = np.asarray(flags, dtype=np.int8)
    result = np.zeros(len(array), dtype=np.int8)
    for end in range(int(m) - 1, len(array)):
        result[end] = int(
            np.sum(array[end - int(m) + 1 : end + 1]) >= int(k)
        )
    return result


class DeviceScorer:
    """Fit permitted local models and emit label-free calibrated score streams."""

    def __init__(
        self,
        bundle: GlobalBundle,
        input_calibration: np.ndarray,
        local_fitting: np.ndarray,
        score_calibration: np.ndarray,
        *,
        training: dict,
        seed: int,
    ):
        self.bundle = bundle
        self.input_center, self.input_scale = robust_center_scale(
            input_calibration
        )
        self.fit = normalize(
            local_fitting, self.input_center, self.input_scale
        )
        self.postcal = normalize(
            score_calibration, self.input_center, self.input_scale
        )
        self.local_pca = fit_pca(self.fit, 16)
        few_shot_count = max(
            2,
            int(
                np.ceil(
                    float(training["few_shot_local_fitting_fraction"])
                    * len(self.fit)
                )
            ),
        )
        self.few_shot_fit = self.fit[: min(few_shot_count, len(self.fit))]
        self.local_pca_fewshot = fit_pca(self.few_shot_fit, 16)
        self.local_ae, _ = train_ae_local(
            initial_ae(int(training["initialization_seed"])),
            self.fit,
            epochs=int(training["local_standalone_epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            seed=int(seed) + 11,
        )
        self.local_ae_fewshot, _ = train_ae_local(
            initial_ae(int(training["initialization_seed"])),
            self.few_shot_fit,
            epochs=int(training["local_standalone_epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            seed=int(seed) + 13,
        )
        self.personalized_ae, _ = train_ae_local(
            bundle.federated_ae,
            self.fit,
            epochs=int(training["local_epochs_per_round"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            seed=int(seed) + 17,
        )
        local_svdd_initial = initial_svdd([self.fit], seed=int(seed) + 23)
        self.local_svdd, _ = train_svdd_local(
            local_svdd_initial,
            self.fit,
            epochs=int(training["local_standalone_epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            seed=int(seed) + 29,
            synthetic_margin=None,
            variance_floor=float(training["deep_svdd_minimum_latent_variance"]),
        )
        self.local_predictor, _ = train_predictor_local(
            initial_predictor(int(seed) + 31),
            self.fit,
            epochs=int(training["local_standalone_epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            seed=int(seed) + 37,
        )
        self.random_models = [
            initial_ae(random_seed) for random_seed in RANDOM_SEEDS
        ]
        self.raw_postcal = self._raw(
            self.postcal,
            previous=np.vstack([self.fit[-1:], self.postcal[:-1]]),
        )

    def _raw(
        self, normalized: np.ndarray, *, previous: np.ndarray
    ) -> dict[str, np.ndarray]:
        values = np.asarray(normalized, dtype=np.float32)
        prior = np.asarray(previous, dtype=np.float32)
        random_scores = np.column_stack(
            [model.reconstruction_score(values) for model in self.random_models]
        )
        input_profile = np.max(np.abs(values), axis=1)
        simple = np.max(
            np.abs(values[:, SIMPLE_RULE_INDICES]), axis=1
        )
        return {
            "LOCAL_PCA16": self.local_pca.score(values),
            "LOCAL_PCA16_FEWSHOT25": self.local_pca_fewshot.score(values),
            "FED_PCA16": self.bundle.federated_pca.score(values),
            "RANDOM_AE_MEDIAN": np.median(random_scores, axis=1),
            "LOCAL_AE_RECON": self.local_ae.reconstruction_score(values),
            "LOCAL_AE_RECON_FEWSHOT25": (
                self.local_ae_fewshot.reconstruction_score(values)
            ),
            "FED_AE_RECON": self.bundle.federated_ae.reconstruction_score(values),
            "FED_AE_PERSONALIZED_RECON": (
                self.personalized_ae.reconstruction_score(values)
            ),
            "LOCAL_AE_PSEUDO_HEAD": self.local_ae.pseudo_probability(values),
            "FED_AE_PSEUDO_HEAD": self.bundle.federated_ae.pseudo_probability(
                values
            ),
            "FED_AE_PERSONALIZED_PSEUDO_HEAD": (
                self.personalized_ae.pseudo_probability(values)
            ),
            "LOCAL_DEEP_SVDD": self.local_svdd.score(values),
            "FED_DEEP_SVDD": self.bundle.federated_svdd.score(values),
            "FED_SYNTHETIC_SVDD": self.bundle.synthetic_svdd.score(values),
            "LOCAL_NEXT_PREDICTOR": self.local_predictor.score_pairs(
                prior, values
            ),
            "FED_NEXT_PREDICTOR": self.bundle.federated_predictor.score_pairs(
                prior, values
            ),
            "INPUT_PROFILE_RULE": input_profile.astype(np.float64),
            "RMS_KURTOSIS_CREST_RULE": simple.astype(np.float64),
        }

    def score(
        self, values: np.ndarray, *, previous: np.ndarray
    ) -> dict[str, dict[str, np.ndarray]]:
        normalized = normalize(values, self.input_center, self.input_scale)
        normalized_previous = normalize(
            previous, self.input_center, self.input_scale
        )
        raw = self._raw(normalized, previous=normalized_previous)
        return {
            arm: calibration_variants(self.raw_postcal[arm], raw[arm])
            for arm in SINGLE_ARMS
        }


def empirical_feature_noise(
    values: np.ndarray, fitting: np.ndarray, *, scale: float, seed: int
) -> np.ndarray:
    """Feature-domain stress using measured local increments, not raw-signal SNR."""

    current = _finite_matrix(values)
    reference = _finite_matrix(fitting)
    rng = np.random.default_rng(int(seed))
    differences = (
        np.diff(reference, axis=0)
        if len(reference) >= 2
        else np.zeros((1, FEATURE_DIM), dtype=np.float32)
    )
    selected = differences[
        rng.integers(0, len(differences), size=len(current))
    ]
    return (current + float(scale) * selected).astype(np.float32)


def accepted_reference_union_score(
    anchor_score: np.ndarray,
    accepted_score: np.ndarray,
    *,
    accepted_anchor_center: float,
    accepted_anchor_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return novelty-to-all-normal-modes and incremental anchor progression.

    ``min`` is intentional: a point is novel only when it is far from every
    accepted normal reference.  The immutable anchor remains available through
    the incremental progression channel but cannot independently retrigger just
    because the accepted regime differs from commissioning.
    """

    anchor = np.asarray(anchor_score, dtype=np.float64)
    accepted = np.asarray(accepted_score, dtype=np.float64)
    if anchor.shape != accepted.shape:
        raise ValueError("dual-reference streams must match")
    novelty = np.minimum(anchor, accepted)
    incremental = (
        anchor - float(accepted_anchor_center)
    ) / max(float(accepted_anchor_scale), 1.0)
    return novelty, incremental


def model_parameter_count(bundle: GlobalBundle) -> dict[str, int]:
    return {
        "FED_PCA16": int(bundle.federated_pca.basis.size + bundle.federated_pca.center.size),
        "FED_AE_RECON": int(
            sum(weight.size for weight in bundle.federated_ae.weights)
            + bundle.federated_ae.head_weight.size
            + 1
        ),
        "FED_DEEP_SVDD": int(
            sum(weight.size for weight in bundle.federated_svdd.weights)
            + bundle.federated_svdd.center.size
        ),
        "FED_SYNTHETIC_SVDD": int(
            sum(weight.size for weight in bundle.synthetic_svdd.weights)
            + bundle.synthetic_svdd.center.size
        ),
        "FED_NEXT_PREDICTOR": int(
            sum(weight.size for weight in bundle.federated_predictor.weights)
        ),
    }


__all__ = [
    "AEModel",
    "CALIBRATION_VARIANTS",
    "DeviceScorer",
    "GlobalBundle",
    "PCAModel",
    "PredictorModel",
    "RANDOM_SEEDS",
    "SIMPLE_RULE_INDICES",
    "SINGLE_ARMS",
    "SVDDModel",
    "SYMMETRIC_ARMS",
    "accepted_reference_union_score",
    "calibration_variants",
    "empirical_feature_noise",
    "first_alarm",
    "fit_federated_pca",
    "fit_pca",
    "initial_ae",
    "initial_predictor",
    "initial_svdd",
    "k_of_m_evidence",
    "k_of_m_flags",
    "model_parameter_count",
    "normalize",
    "robust_center_scale",
    "scalar_center_scale",
    "symmetric_flags",
    "train_ae_local",
    "train_federated_ae",
    "train_federated_predictor",
    "train_federated_svdd",
    "train_predictor_local",
    "train_svdd_local",
]
