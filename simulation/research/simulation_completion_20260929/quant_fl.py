"""int8 quantisation of every FL exchange (downlink + uplink), on top of
fl.py's tanh autoencoder and FedAvg/FedAdam optimisers. See int8.json.

Reuses fl.py's init/local_train/Server-step expressions exactly (arm "F32"
must be bit-identical to fl.Server -- see exchange_semantics.aggregation in
int8.json). Does not edit fl.py or any other existing file.

Quantisation follows int8.json's exchange_semantics:
  - server keeps float64 master weights + FedAdam m/v; quantisation is only
    ever applied "on the air" (downlink and uplink messages).
  - downlink each round: the server's CURRENT absolute global weights are
    quantised and every client dequantises that as its round-start model
    (round 0/the initial model is quantised too, before round 1 training).
  - uplink: delta = local_trained - dequantised_global_received, quantised
    (and top-K sparsified where the arm says so, no error feedback -- an
    untransmitted entry contributes 0 to that weight's mean, not its true
    value).
  - aggregation: server delta = unweighted mean of the DEQUANTISED client
    deltas; the FedAdam/FedAvg step is applied to that mean exactly as
    fl.Server.round does.
  - scoring: every device scores with the dequantised FINAL global it holds,
    i.e. one more downlink-quantisation of the server's post-round-12
    params (no further local training).

Rounding is C roundf semantics: half away from zero, then clip to the arm's
range (matches tinyfl_protocol.quantize_weight and tinyfl_client.ino's
quantize()).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

import fl

# ---------------------------------------------------------------------------
# Fixed constants (int8.json).
# ---------------------------------------------------------------------------
LAYERS = [22, 12, 22]
N_PARAMS = fl.n_params(LAYERS)          # 562
assert N_PARAMS == 562

NUM_WEIGHTS_DEPLOYED = 464               # tinyfl_protocol.NUM_WEIGHTS
SPARSITY_K_DEPLOYED = 128                # tinyfl_protocol.SPARSITY_K
QUANT_DENOMINATOR = 127.0 * 0.1          # tinyfl_protocol.QUANT_DENOMINATOR == 12.7
DEPLOYED_STEP = 1.0 / QUANT_DENOMINATOR  # fixed per-weight quantisation step
DEPLOYED_LO, DEPLOYED_HI = -128, 127     # tinyfl_protocol.quantize_weight clip

DYN_LO, DYN_HI = -127, 127               # tinyfl_client.ino quantize() clip
DYN_FLOOR = 1e-7                         # tinyfl_client.ino chooseQuantScale floor

# K generalised to 562 weights, exactly as int8.json states.
K_TOPK = round(N_PARAMS * SPARSITY_K_DEPLOYED / NUM_WEIGHTS_DEPLOYED)
assert K_TOPK == 155

ARMS = ("F32", "Q_DEPLOYED", "Q_DYN_MSG", "Q_DYN_TENSOR", "Q_DYN_SPARSE")

# ---------------------------------------------------------------------------
# Flatten / unflatten a params tuple (W1, b1, W2, b2) <-> one length-562 vector.
# ---------------------------------------------------------------------------
def _shapes(layers=LAYERS):
    shapes = []
    for fan_in, fan_out in zip(layers[:-1], layers[1:]):
        shapes.append((fan_in, fan_out))
        shapes.append((fan_out,))
    return shapes


SHAPES = _shapes()
SIZES = [int(np.prod(s)) for s in SHAPES]
OFFSETS = np.cumsum([0] + SIZES)
TENSOR_SLICES = [slice(int(OFFSETS[i]), int(OFFSETS[i + 1])) for i in range(len(SHAPES))]


def flatten(params) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=float).ravel() for p in params])


def unflatten(vec: np.ndarray) -> tuple:
    vec = np.asarray(vec, dtype=float)
    return tuple(vec[TENSOR_SLICES[i]].reshape(SHAPES[i]) for i in range(len(SHAPES)))


# ---------------------------------------------------------------------------
# roundf semantics: half away from zero.
# ---------------------------------------------------------------------------
def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


# ---------------------------------------------------------------------------
# Core scalar/vector quantise-dequantise primitive.
# ---------------------------------------------------------------------------
def quantize_dequantize(vec: np.ndarray, scale: float, lo: int, hi: int):
    """Return (dequantised float vector, integer codes, scale used)."""
    vec = np.asarray(vec, dtype=float)
    codes = round_half_away_from_zero(vec / scale)
    codes = np.clip(codes, lo, hi)
    return codes * scale, codes, scale


def dynamic_scale(vec: np.ndarray) -> float:
    """max|x|/127, floored at 1e-7 (tinyfl_client.ino chooseQuantScale)."""
    vec = np.asarray(vec, dtype=float)
    max_abs = float(np.max(np.abs(vec))) if vec.size else 0.0
    return max(max_abs / 127.0, DYN_FLOOR)


def quantize_weight_deployed(value: float) -> int:
    """Single-weight version, matches tinyfl_protocol.quantize_weight exactly."""
    scaled = float(value) * QUANT_DENOMINATOR
    rounded = np.floor(scaled + 0.5) if scaled >= 0 else np.ceil(scaled - 0.5)
    return int(max(DEPLOYED_LO, min(DEPLOYED_HI, rounded)))


# ---------------------------------------------------------------------------
# Top-K selection by |delta|, deterministic tie-break by index.
# ---------------------------------------------------------------------------
def topk_indices(delta: np.ndarray, k: int) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    order = np.lexsort((np.arange(delta.size), -np.abs(delta)))
    return np.sort(order[:k])


# ---------------------------------------------------------------------------
# Per-arm downlink / uplink quantisation. Each returns:
#   deq_vec: dequantised float vector (562,) actually held/used
#   stats:   dict with at least {"zero_frac": ..., "n_transmitted": ...}
# Downlink is always dense (all arms send the full model every round).
# ---------------------------------------------------------------------------
def downlink_quantize(w: np.ndarray, arm: str):
    w = np.asarray(w, dtype=float)
    if arm == "F32":
        return w.copy(), {"zero_frac": np.nan, "n_transmitted": 0, "n_zero": 0}
    if arm == "Q_DEPLOYED":
        deq, codes, _ = quantize_dequantize(w, DEPLOYED_STEP, DEPLOYED_LO, DEPLOYED_HI)
        n_zero = int(np.sum(codes == 0))
        return deq, {"zero_frac": n_zero / w.size, "n_transmitted": w.size, "n_zero": n_zero}
    if arm in ("Q_DYN_MSG", "Q_DYN_SPARSE"):
        scale = dynamic_scale(w)
        deq, codes, _ = quantize_dequantize(w, scale, DYN_LO, DYN_HI)
        n_zero = int(np.sum(codes == 0))
        return deq, {"zero_frac": n_zero / w.size, "n_transmitted": w.size, "n_zero": n_zero}
    if arm == "Q_DYN_TENSOR":
        out = np.empty_like(w)
        n_zero = 0
        for sl in TENSOR_SLICES:
            scale = dynamic_scale(w[sl])
            deq, codes, _ = quantize_dequantize(w[sl], scale, DYN_LO, DYN_HI)
            out[sl] = deq
            n_zero += int(np.sum(codes == 0))
        return out, {"zero_frac": n_zero / w.size, "n_transmitted": w.size, "n_zero": n_zero}
    raise ValueError(arm)


def uplink_quantize(delta: np.ndarray, arm: str):
    """Returns (deq_full_562_vector_with_untransmitted_as_zero, stats)."""
    delta = np.asarray(delta, dtype=float)
    n = delta.size
    if arm == "F32":
        return delta.copy(), {"zero_frac": np.nan, "n_transmitted": 0, "n_zero": 0}

    if arm == "Q_DEPLOYED":
        idx = topk_indices(delta, K_TOPK)
        deq_k, codes, _ = quantize_dequantize(delta[idx], DEPLOYED_STEP, DEPLOYED_LO, DEPLOYED_HI)
        out = np.zeros(n, dtype=float)
        out[idx] = deq_k
        n_zero = int(np.sum(codes == 0))
        return out, {"zero_frac": n_zero / K_TOPK, "n_transmitted": K_TOPK, "n_zero": n_zero}

    if arm == "Q_DYN_MSG":
        scale = dynamic_scale(delta)
        deq, codes, _ = quantize_dequantize(delta, scale, DYN_LO, DYN_HI)
        n_zero = int(np.sum(codes == 0))
        return deq, {"zero_frac": n_zero / n, "n_transmitted": n, "n_zero": n_zero}

    if arm == "Q_DYN_TENSOR":
        out = np.empty_like(delta)
        n_zero = 0
        for sl in TENSOR_SLICES:
            scale = dynamic_scale(delta[sl])
            deq, codes, _ = quantize_dequantize(delta[sl], scale, DYN_LO, DYN_HI)
            out[sl] = deq
            n_zero += int(np.sum(codes == 0))
        return out, {"zero_frac": n_zero / n, "n_transmitted": n, "n_zero": n_zero}

    if arm == "Q_DYN_SPARSE":
        idx = topk_indices(delta, K_TOPK)
        scale = dynamic_scale(delta[idx])
        deq_k, codes, _ = quantize_dequantize(delta[idx], scale, DYN_LO, DYN_HI)
        out = np.zeros(n, dtype=float)
        out[idx] = deq_k
        n_zero = int(np.sum(codes == 0))
        return out, {"zero_frac": n_zero / K_TOPK, "n_transmitted": K_TOPK, "n_zero": n_zero}

    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Payload accounting (int8.json "payload_accounting"): 16-byte header,
# 200-byte fragment bodies, fl.fragments convention.
# ---------------------------------------------------------------------------
HEADER, BODY = 16, 200


def _fragments(bytes_: int) -> int:
    return int(np.ceil((bytes_ + HEADER) / BODY))


def payload_bytes_fragments(arm: str):
    """Returns {"downlink": {"bytes":..,"fragments":..}, "uplink": {...}}."""
    if arm == "F32":
        b = 4 * N_PARAMS  # 2248 -- float32 weights, dense, both directions
        return {"downlink": {"bytes": b, "fragments": _fragments(b)},
                "uplink": {"bytes": b, "fragments": _fragments(b)}}
    if arm == "Q_DEPLOYED":
        down_b = N_PARAMS + 4 * 0  # fixed scale, not transmitted
        up_b = K_TOPK * (2 + 1) + 4
        return {"downlink": {"bytes": down_b, "fragments": _fragments(down_b)},
                "uplink": {"bytes": up_b, "fragments": _fragments(up_b)}}
    if arm == "Q_DYN_MSG":
        b = N_PARAMS + 4 * 1  # one dynamic scale per message
        return {"downlink": {"bytes": b, "fragments": _fragments(b)},
                "uplink": {"bytes": b, "fragments": _fragments(b)}}
    if arm == "Q_DYN_TENSOR":
        b = N_PARAMS + 4 * 4  # four dynamic scales (one per tensor)
        return {"downlink": {"bytes": b, "fragments": _fragments(b)},
                "uplink": {"bytes": b, "fragments": _fragments(b)}}
    if arm == "Q_DYN_SPARSE":
        down_b = N_PARAMS + 4 * 1  # Q_DYN_MSG-style downlink
        up_b = K_TOPK * (2 + 1) + 4  # top-K uplink + one dynamic scale
        return {"downlink": {"bytes": down_b, "fragments": _fragments(down_b)},
                "uplink": {"bytes": up_b, "fragments": _fragments(up_b)}}
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------
def train_quantised(clients: dict, layers: Sequence[int], seed, algorithm: str, rounds: int, arm: str):
    """Run `rounds` rounds of `algorithm` ("fedavg"|"fedadam") with every
    exchange quantised per `arm`.

    clients: dict {machine_id: x_rows}, iterated in dict order (matches
        fl.Server/dropout_fl.train_with_loss client ordering).

    Returns (server_params, final_global, round_diagnostics):
      server_params:  the server's float64 master weights after `rounds`
                       rounds (== fl.Server.p for arm F32).
      final_global:    the dequantised global every device holds for
                       scoring (one extra downlink-quantisation of
                       server_params; == server_params for arm F32).
      round_diagnostics: list of length `rounds` of dicts with
                       mean_relative_downlink_weight_error,
                       uplink_zero_fraction, server_change_norm.
    """
    if algorithm not in ("fedavg", "fedadam"):
        raise ValueError(algorithm)
    if arm not in ARMS:
        raise ValueError(arm)

    client_ids = list(clients.keys())

    server_params = fl.init(layers, seed)
    m = tuple(np.zeros_like(v) for v in server_params)
    v = tuple(np.zeros_like(vv) for vv in server_params)
    t = 0

    # Round-0 downlink: the initial model, quantised, is what every client
    # starts round 1 from.
    w0 = flatten(server_params)
    deq0, _ = downlink_quantize(w0, arm)
    deployed_global = unflatten(deq0)

    diagnostics = []
    for r in range(rounds):
        old = server_params  # float64 master weights at round start

        local = [fl.local_train(clients[cid], deployed_global, mu=0.0) for cid in client_ids]

        # Uplink: delta = local_trained - DEQUANTISED global the client
        # actually trained from (deployed_global), then quantised (+
        # sparsified per arm), no error feedback.
        client_deq_deltas = []
        zero_total, transmitted_total = 0, 0
        for p in local:
            delta_vec = flatten(p) - flatten(deployed_global)
            deq_delta_vec, up_stats = uplink_quantize(delta_vec, arm)
            client_deq_deltas.append(unflatten(deq_delta_vec))
            if up_stats["n_transmitted"]:
                zero_total += up_stats["n_zero"]
                transmitted_total += up_stats["n_transmitted"]

        # Same expression fl.Server.round / dropout_fl use, so arm "F32" is
        # bit-identical (deployed_global == old exactly when arm == "F32",
        # since downlink_quantize is then a no-op).
        d = tuple(np.mean(np.stack([q[i] for q in client_deq_deltas]), 0) for i in range(len(old)))

        t += 1
        if algorithm == "fedadam":
            m = tuple(fl.B1 * m[i] + (1 - fl.B1) * d[i] for i in range(len(old)))
            v = tuple(fl.B2 * v[i] + (1 - fl.B2) * d[i] * d[i] for i in range(len(old)))
            server_params = tuple(
                old[i] + fl.ETA * (m[i] / (1 - fl.B1 ** t)) / (np.sqrt(v[i] / (1 - fl.B2 ** t)) + fl.TAU)
                for i in range(len(old))
            )
        else:
            server_params = tuple(old[i] + d[i] for i in range(len(old)))

        change_norm = float(np.linalg.norm(flatten(server_params) - flatten(old)))

        # Next round's (or, on the last round, the final scoring) downlink.
        w = flatten(server_params)
        deq, dl_stats = downlink_quantize(w, arm)
        deployed_global = unflatten(deq)
        w_norm = np.linalg.norm(w)
        downlink_err = float(np.linalg.norm(deq - w) / w_norm) if w_norm > 0 else 0.0

        diagnostics.append({
            "round": r + 1,
            "mean_relative_downlink_weight_error": downlink_err,
            "uplink_zero_fraction": (zero_total / transmitted_total) if transmitted_total else float("nan"),
            "server_change_norm": change_norm,
        })

    final_global = deployed_global
    return server_params, final_global, diagnostics
