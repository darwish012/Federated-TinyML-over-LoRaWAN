"""Device (hardware-twin) side of the LilyGO/ESP32 firmware, written the way
the C firmware will compute it: float32 arithmetic everywhere, an explicit
median/MAD standardisation from raw per-stream features, a float32 tanh
autoencoder trained with fl.py's exact algorithm (H12 [22,12,22]), and int8
encode/decode using the Q_DYN_TENSOR convention (per-tensor dynamic scale,
C `roundf` half-away-from-zero, clip [-127,127]).

Reference code this module mirrors (READ, not imported, so the twin has no
hidden float64 dependency on the reference at runtime):
  - research/fl_algorithm_comparison_20260929/fl.py           (AE algorithm)
  - research/simulation_completion_20260929/quant_fl.py       (Q_DYN_TENSOR)
  - research/healthy_only_suite_20260907/data_cache.py        (standardisation
    formula: physical_matrix)

See protocol.json (research/hardware_twin_scaling_20260929/protocol.json)
for the exact specification; this file implements it and must not edit it.

The server side lives in twin_server.py, not here, so a future hardware
server can import twin_server unchanged, with no twin.py/fl.py dependency.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Model shape (protocol.json "model": "H12 [22,12,22] tanh AE").
# ---------------------------------------------------------------------------
LAYERS = (22, 12, 22)
LR = np.float32(0.03)
L2 = np.float32(1e-4)
EPOCHS = 2

# int8 quantisation constants (Q_DYN_TENSOR, quant_fl.py / int8.json).
DYN_LO, DYN_HI = -127, 127
DYN_FLOOR = 1e-7

MAD_K = 1.4826
STD_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# Tensor shapes / flatten / unflatten for LAYERS, mirroring quant_fl._shapes.
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
N_PARAMS = int(OFFSETS[-1])
N_TENSORS = len(SHAPES)


def flatten(params) -> np.ndarray:
    return np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in params])


def unflatten(vec: np.ndarray) -> tuple:
    vec = np.asarray(vec, dtype=np.float32)
    return tuple(vec[TENSOR_SLICES[i]].reshape(SHAPES[i]) for i in range(N_TENSORS))


def zeros_like_params():
    return tuple(np.zeros(s, dtype=np.float32) for s in SHAPES)


def subtract_params(a, b):
    """a - b, tensor-wise (both length-4 tuples)."""
    return tuple(np.asarray(a[i], dtype=np.float32) - np.asarray(b[i], dtype=np.float32) for i in range(N_TENSORS))


def add_params(a, b):
    return tuple(np.asarray(a[i], dtype=np.float32) + np.asarray(b[i], dtype=np.float32) for i in range(N_TENSORS))


def scale_params(a, k):
    k = np.float32(k)
    return tuple(np.asarray(a[i], dtype=np.float32) * k for i in range(N_TENSORS))


def mean_params(list_of_params):
    """Equal-weight float32 mean of a list of param tuples (PER_BOARD's
    board-side average of its active machines' float32 deltas)."""
    n = np.float32(len(list_of_params))
    acc = zeros_like_params()
    for p in list_of_params:
        acc = add_params(acc, p)
    return scale_params(acc, np.float32(1.0) / n)


def to_float32(params):
    return tuple(np.asarray(p, dtype=np.float32) for p in params)


# ---------------------------------------------------------------------------
# On-device standardisation from raw per-stream features (protocol.json
# on_device_pipeline.standardisation):
#   centre = median (even count = mean of the two middle values)
#   scale  = max(1.4826*MAD, population std); 1.0 if that is <= 1e-8
#   x      = |asinh((v - centre) / scale)|
#   elementwise max over streams
# Each stream is assumed already sliced to the 22 raw features (that slicing
# happens once, off-device, when the SD card is written -- see sd_export.py
# / data_cache.FEATURE_INDICES); the device never sees the other columns.
# ---------------------------------------------------------------------------
def median_axis0(a: np.ndarray) -> np.ndarray:
    """Median along axis 0, explicit even-count "mean of the two middle
    values" rule, in `a`'s own dtype throughout (no upcast)."""
    a = np.asarray(a)
    s = np.sort(a, axis=0)
    n = s.shape[0]
    if n == 0:
        raise ValueError("median of empty array")
    mid = n // 2
    if n % 2 == 1:
        return s[mid].copy()
    half = a.dtype.type(0.5) if np.issubdtype(a.dtype, np.floating) else 0.5
    return (s[mid - 1] + s[mid]) * half


def population_std_axis0(a: np.ndarray) -> np.ndarray:
    """Population standard deviation (ddof=0) along axis 0, in `a`'s dtype."""
    a = np.asarray(a)
    mean = np.mean(a, axis=0, dtype=a.dtype)
    dev = a - mean
    var = np.mean(dev * dev, axis=0, dtype=a.dtype)
    return np.sqrt(var)


def stream_standardisation_params(train_rows: np.ndarray, dtype=np.float32):
    """centre, scale for one stream's train rows (rows x 22), in `dtype`."""
    fit = np.asarray(train_rows, dtype=dtype)
    centre = median_axis0(fit)
    mad = dtype(MAD_K) * median_axis0(np.abs(fit - centre))
    std = population_std_axis0(fit)
    scale = np.maximum(mad, std)
    floor = np.asarray(STD_FLOOR, dtype=dtype)
    one = np.asarray(1.0, dtype=dtype)
    scale = np.where(scale > floor, scale, one)
    return centre.astype(dtype), scale.astype(dtype)


def apply_standardisation(stream: np.ndarray, centre: np.ndarray, scale: np.ndarray, dtype=np.float32) -> np.ndarray:
    v = np.asarray(stream, dtype=dtype)
    return np.abs(np.arcsinh((v - centre) / scale)).astype(dtype)


def standardize(streams: Sequence[np.ndarray], train_idx: np.ndarray, dtype=np.float32):
    """Full on-device standardisation: per-stream centre/scale from the
    machine's train rows, transform every row of every stream, elementwise
    max over streams. `streams`: list of (n_rows, 22) raw arrays, already
    sliced to the 22 features (as stored on the SD card). Returns (x,
    per_stream_params) where per_stream_params is a list of (centre, scale).
    """
    ti = np.asarray(train_idx, dtype=np.int64)
    transformed = []
    params = []
    for s in streams:
        v = np.asarray(s, dtype=dtype)
        centre, scale = stream_standardisation_params(v[ti], dtype=dtype)
        transformed.append(apply_standardisation(v, centre, scale, dtype=dtype))
        params.append((centre, scale))
    x = np.max(np.stack(transformed), axis=0).astype(dtype)
    return x, params


# ---------------------------------------------------------------------------
# Float32 tanh autoencoder: fl.py's forward/grads/local_train, unchanged
# algorithm, float32 arithmetic (mirrors fl.forward/fl.grads/fl.local_train).
# ---------------------------------------------------------------------------
def forward(x, p):
    x = np.asarray(x, dtype=np.float32)
    hs = []
    a = x
    n_layers = len(p) // 2
    for i in range(n_layers):
        z = a @ np.asarray(p[2 * i], dtype=np.float32) + np.asarray(p[2 * i + 1], dtype=np.float32)
        if i < n_layers - 1:
            a = np.tanh(z)
            hs.append(a)
        else:
            a = z
    return a, hs


def grads(x, p):
    x = np.asarray(x, dtype=np.float32)
    y, hs = forward(x, p)
    n = np.float32(max(x.size, 1))
    delta = np.float32(2.0) * (y - x) / n
    n_layers = len(p) // 2
    acts = [x] + hs
    g = [None] * len(p)
    for i in reversed(range(n_layers)):
        w = np.asarray(p[2 * i], dtype=np.float32)
        g[2 * i] = acts[i].T @ delta + L2 * w
        g[2 * i + 1] = np.sum(delta, axis=0, dtype=np.float32)
        if i > 0:
            dh = delta @ w.T
            delta = (dh * (np.float32(1.0) - acts[i] * acts[i])).astype(np.float32)
    return tuple(g)


def local_train(x, p, epochs: int = EPOCHS):
    p = to_float32(p)
    for _ in range(epochs):
        g = grads(x, p)
        p = tuple((np.asarray(v, dtype=np.float32) - LR * np.asarray(q, dtype=np.float32)) for v, q in zip(p, g))
    return p


def score(p, x):
    y, _ = forward(np.asarray(x, dtype=np.float32), p)
    x = np.asarray(x, dtype=np.float32)
    return np.mean((y - x) ** 2, axis=1, dtype=np.float32)


# ---------------------------------------------------------------------------
# Q_DYN_TENSOR int8 codec (quant_fl.py convention): per tensor, scale =
# max|x|/127 floored at 1e-7; C roundf semantics (half away from zero);
# clip to [-127, 127].
# ---------------------------------------------------------------------------
def round_half_away_from_zero(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def dynamic_scale(vec: np.ndarray) -> float:
    vec = np.asarray(vec, dtype=np.float64)
    max_abs = float(np.max(np.abs(vec))) if vec.size else 0.0
    return max(max_abs / 127.0, DYN_FLOOR)


def encode_tensor(vec: np.ndarray):
    """vec (float32/float64) -> (codes int8 array, scale float64)."""
    vec64 = np.asarray(vec, dtype=np.float64)
    scale = dynamic_scale(vec64)
    codes = round_half_away_from_zero(vec64 / scale)
    codes = np.clip(codes, DYN_LO, DYN_HI).astype(np.int8)
    return codes, scale


def decode_tensor(codes: np.ndarray, scale: float, dtype=np.float32) -> np.ndarray:
    return (codes.astype(np.float64) * scale).astype(dtype)


def encode_params(params):
    """Per-tensor Q_DYN_TENSOR encode of a 4-tensor params tuple.
    Returns (codes_list[4 int8 arrays], scales_list[4 floats])."""
    codes_list, scales_list = [], []
    for t in params:
        codes, scale = encode_tensor(np.asarray(t, dtype=np.float64).ravel())
        codes_list.append(codes.reshape(np.asarray(t).shape))
        scales_list.append(scale)
    return codes_list, scales_list


def decode_params(codes_list, scales_list, dtype=np.float32):
    return tuple(decode_tensor(codes_list[i], scales_list[i], dtype=dtype) for i in range(len(codes_list)))


def all_codes_flat(codes_list) -> np.ndarray:
    """Flatten the 4 tensors' int8 codes into one length-562 int array, in
    the fixed W1,b1,W2,b2 order -- used for the golden-reference export and
    the hardware-vs-twin global_model criterion (>=99% of 562 codes)."""
    return np.concatenate([np.asarray(c, dtype=np.int64).ravel() for c in codes_list])


# ---------------------------------------------------------------------------
# Uplink modes (protocol.json "uplink_modes").
# ---------------------------------------------------------------------------
def per_machine_uplink_message(local_trained_params, received_global_params):
    """PER_MACHINE: this machine's own quantised delta vs the dequantised
    global it trained from. Returns (codes_list, scales_list, delta_f32)."""
    delta = subtract_params(local_trained_params, received_global_params)
    codes_list, scales_list = encode_params(delta)
    return codes_list, scales_list, delta


def per_board_uplink_message(board_local_trained, board_received_global):
    """PER_BOARD: average this board's active machines' float32 deltas
    (equal weight, float32 arithmetic), quantise that mean ONCE.
    board_local_trained / board_received_global: lists, one entry per
    active machine on the board, same order.
    Returns (codes_list, scales_list, mean_delta_f32, machine_count)."""
    n = len(board_local_trained)
    deltas = [subtract_params(board_local_trained[i], board_received_global[i]) for i in range(n)]
    mean_delta = mean_params(deltas)
    codes_list, scales_list = encode_params(mean_delta)
    return codes_list, scales_list, mean_delta, n


def downlink_message(server_params_f64):
    """Server-side encode of its current absolute global weights
    (Q_DYN_TENSOR), for one Class C multicast per round to all boards."""
    params_f32 = to_float32(server_params_f64)
    return encode_params(params_f32)


def receive_downlink(codes_list, scales_list, dtype=np.float32):
    """Device-side decode of the broadcast global -> this device's new
    local copy (float32)."""
    return decode_params(codes_list, scales_list, dtype=dtype)


# ---------------------------------------------------------------------------
# Delivery masks (protocol.json "delivery_mask"; semantics as
# fl_algorithm_comparison_20260929/dropout.json): default all delivered.
# ---------------------------------------------------------------------------
def default_mask(rounds: int, n: int) -> np.ndarray:
    return np.ones((rounds, n), dtype=bool)
