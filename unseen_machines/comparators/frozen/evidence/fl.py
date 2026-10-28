"""Generic tanh autoencoder and three federated optimisers (NumPy only).

For layers [22, 12, 22] this reproduces research/healthy_only_causal_20260908/
wider_models.py exactly (same initialisation order, same gradient, same
FedAdam step), which the replication gate in run_study.py verifies.
"""
from __future__ import annotations
import numpy as np

LR, L2, EPOCHS = 0.03, 1e-4, 2
ETA, B1, B2, TAU = 0.01, 0.9, 0.99, 0.001


def init(layers, seed):
    """Weights drawn sequentially from one generator; biases zero."""
    r = np.random.default_rng(int(seed))
    params = []
    for fan_in, fan_out in zip(layers[:-1], layers[1:]):
        s = np.sqrt(2.0 / (fan_in + fan_out))
        params.append(r.normal(0, s, (fan_in, fan_out)))
        params.append(np.zeros(fan_out))
    return tuple(params)


def forward(x, p):
    """Returns output and the list of hidden activations."""
    hs = []
    a = x
    n_layers = len(p) // 2
    for i in range(n_layers):
        z = a @ p[2 * i] + p[2 * i + 1]
        if i < n_layers - 1:
            a = np.tanh(z)
            hs.append(a)
        else:
            a = z
    return a, hs


def grads(x, p, prox=None, mu=0.0):
    """Gradient of mean-squared reconstruction + 0.5*L2*sum(W^2) [+ prox]."""
    y, hs = forward(x, p)
    n = max(x.size, 1)
    delta = 2 * (y - x) / n                      # d loss / d output (linear)
    n_layers = len(p) // 2
    acts = [x] + hs                              # input to each layer
    g = [None] * len(p)
    for i in reversed(range(n_layers)):
        w = p[2 * i]
        g[2 * i] = acts[i].T @ delta + L2 * w
        g[2 * i + 1] = np.sum(delta, 0)
        if i > 0:
            dh = delta @ w.T
            delta = dh * (1 - acts[i] * acts[i])
    if prox is not None and mu > 0:
        g = [gi + mu * (pi - qi) for gi, pi, qi in zip(g, p, prox)]
    return tuple(g)


def local_train(x, p, mu=0.0):
    start = p
    for _ in range(EPOCHS):
        g = grads(x, p, prox=start if mu > 0 else None, mu=mu)
        p = tuple(v - LR * q for v, q in zip(p, g))
    return p


class Server:
    def __init__(self, algorithm, params, mu=0.0):
        if algorithm not in ("fedavg", "fedadam", "fedprox"):
            raise ValueError(algorithm)
        self.algorithm, self.p, self.mu, self.t = algorithm, params, float(mu), 0
        self.m = tuple(np.zeros_like(v) for v in params)
        self.v = tuple(np.zeros_like(v) for v in params)

    def round(self, clients):
        old = self.p
        mu = self.mu if self.algorithm == "fedprox" else 0.0
        local = [local_train(x, old, mu) for x in clients.values()]
        # Same expression as wider_models._adam, so h12/fedadam is bit-identical.
        d = tuple(np.mean(np.stack([q[i] - old[i] for q in local]), 0) for i in range(len(old)))
        self.t += 1
        if self.algorithm == "fedadam":
            t = self.t
            self.m = tuple(B1 * self.m[i] + (1 - B1) * d[i] for i in range(len(old)))
            self.v = tuple(B2 * self.v[i] + (1 - B2) * d[i] * d[i] for i in range(len(old)))
            self.p = tuple(old[i] + ETA * (self.m[i] / (1 - B1 ** t)) /
                           (np.sqrt(self.v[i] / (1 - B2 ** t)) + TAU) for i in range(len(old)))
        else:
            self.p = tuple(old[i] + d[i] for i in range(len(old)))
        return self.p


def score(p, x):
    y, _ = forward(np.asarray(x, float), p)
    return np.mean((y - x) ** 2, axis=1)


def n_params(layers):
    return sum(a * b + b for a, b in zip(layers[:-1], layers[1:]))


def fragments(layers, body=200, header=16):
    return int(np.ceil((4 * n_params(layers) + header) / body))


def stop_round(losses, epsilon, patience, max_rounds):
    """First t >= P+1 with (B_{t-P} - B_t)/B_{t-P} < epsilon, else max_rounds.

    losses[0] is round 1. Uses only losses up to t.
    """
    best = np.minimum.accumulate(np.asarray(losses, float))
    for t in range(patience + 1, len(best) + 1):
        prev, cur = best[t - 1 - patience], best[t - 1]
        if (prev - cur) / prev < epsilon:
            return t, "rule"
    return max_rounds, "cap"
