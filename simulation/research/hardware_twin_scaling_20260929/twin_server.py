"""Hardware-twin FedAdam server: float64, exactly fl.Server's arithmetic
(research/fl_algorithm_comparison_20260929/fl.py), but taking a list of
ALREADY-DEQUANTISED (delta, weight) pairs rather than raw client data.

This module has no dependency on twin.py, fl.py, or numpy beyond the
standard import, precisely so the future hardware server (which never sees
simulation-only code) can import it unchanged: it receives whatever the
radio/gateway layer already decoded to float64 tensors, and a weight (1 for
a PER_MACHINE machine delta, the reporting machine count for a PER_BOARD
board delta), and does nothing but the FedAdam step.

protocol.json "server": "float64 FedAdam exactly as fl.Server (ETA 0.01,
B1 0.9, B2 0.99, TAU 0.001), equal machine weighting, 12 rounds."

Equal-weight aggregation (all pairs weight 1, as PER_MACHINE with no loss
uses) is computed with the identical expression fl.Server.round uses
(np.mean over a stacked axis-0 array), so it is bit-identical to fl.Server
-- see test_twin.py's server bit-identity test.
"""
from __future__ import annotations

import numpy as np

ETA, B1, B2, TAU = 0.01, 0.9, 0.99, 0.001


def _weighted_mean_tensor(arrays, weights):
    """Weighted mean of a list of same-shape float64 arrays. When every
    weight is equal this reduces to exactly np.mean(np.stack(arrays), 0),
    the same expression fl.Server.round uses, so equal-weight aggregation
    (PER_MACHINE, no loss) is bit-identical to fl.Server."""
    w = np.asarray(weights, dtype=np.float64)
    if np.all(w == w[0]):
        return np.mean(np.stack(arrays), axis=0)
    stack = np.stack(arrays)
    shape = (len(w),) + (1,) * (stack.ndim - 1)
    return np.sum(stack * w.reshape(shape), axis=0) / w.sum()


class Server:
    """FedAdam server over dequantised (delta, weight) pairs.

    params: the initial model, a tuple of arrays (e.g. fl.init(layers,
    seed)); kept and updated in float64 regardless of the dtype passed in.
    """

    def __init__(self, params):
        self.p = tuple(np.asarray(v, dtype=np.float64) for v in params)
        self.m = tuple(np.zeros_like(v) for v in self.p)
        self.v = tuple(np.zeros_like(v) for v in self.p)
        self.t = 0

    def round(self, deq_delta_weight_pairs):
        """deq_delta_weight_pairs: list of (delta_tuple, weight). delta_tuple
        has the same tensor shapes as self.p, already dequantised to float64
        (or any dtype -- cast to float64 here). weight: 1 for a per-machine
        delta, machine-count for a per-board mean delta.

        If the list is empty (nothing arrived this round, per dropout.json's
        empty_round semantics), the server is left unchanged and the FedAdam
        step counter does NOT advance.
        """
        if not deq_delta_weight_pairs:
            return self.p

        old = self.p
        n_tensors = len(old)
        deltas = [tuple(np.asarray(d[i], dtype=np.float64) for i in range(n_tensors))
                  for d, _w in deq_delta_weight_pairs]
        weights = [w for _d, w in deq_delta_weight_pairs]

        d = tuple(_weighted_mean_tensor([deltas[k][i] for k in range(len(deltas))], weights)
                  for i in range(n_tensors))

        self.t += 1
        t = self.t
        self.m = tuple(B1 * self.m[i] + (1 - B1) * d[i] for i in range(n_tensors))
        self.v = tuple(B2 * self.v[i] + (1 - B2) * d[i] * d[i] for i in range(n_tensors))
        self.p = tuple(old[i] + ETA * (self.m[i] / (1 - B1 ** t)) /
                        (np.sqrt(self.v[i] / (1 - B2 ** t)) + TAU) for i in range(n_tensors))
        return self.p
