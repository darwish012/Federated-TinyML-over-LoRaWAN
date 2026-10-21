"""Dataset-free update codecs and LoRaWAN accounting for V39."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class CodecResult:
    decoded: np.ndarray
    residual: np.ndarray
    body_bytes: int
    retained: int
    relative_l2_error: float


def _finite_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if vector.size < 1 or not np.all(np.isfinite(vector)):
        raise ValueError("codec expects a nonempty finite vector")
    return vector


def codec_body_bytes(codec: str, parameters: int, topk: int = 64) -> int:
    n = int(parameters)
    k = min(int(topk), n)
    if n < 1 or k < 1:
        raise ValueError("invalid parameter or top-k count")
    if codec == "dense_float32":
        return 4 * n
    if codec == "int8_all":
        return 4 + n
    if codec == "topk_float":
        return 4 + 6 * k
    if codec == "topk_int8":
        return 4 + 3 * k
    raise ValueError(f"unknown codec: {codec}")


def encode_decode(
    values: np.ndarray,
    codec: str,
    *,
    topk: int = 64,
    residual: np.ndarray | None = None,
    error_feedback: bool = False,
) -> CodecResult:
    """Simulate the exact decoded vector and byte body seen by the receiver."""
    vector = _finite_vector(values)
    carry = np.zeros_like(vector)
    if error_feedback:
        if residual is None:
            residual = np.zeros_like(vector)
        carry = _finite_vector(residual)
        if carry.shape != vector.shape:
            raise ValueError("error-feedback residual shape mismatch")
    effective = vector + carry
    decoded = np.zeros_like(effective)
    retained = len(effective)
    if codec == "dense_float32":
        decoded = effective.copy()
    elif codec == "int8_all":
        maximum = float(np.max(np.abs(effective)))
        scale = maximum / 127.0 if maximum > 0.0 else 1.0
        quantized = np.clip(np.rint(effective / scale), -127, 127).astype(np.int8)
        decoded = quantized.astype(np.float32) * np.float32(scale)
    elif codec in {"topk_float", "topk_int8"}:
        retained = min(int(topk), len(effective))
        indices = np.argsort(-np.abs(effective), kind="stable")[:retained]
        if codec == "topk_float":
            decoded[indices] = effective[indices]
        else:
            selected = effective[indices]
            maximum = float(np.max(np.abs(selected))) if retained else 0.0
            scale = maximum / 127.0 if maximum > 0.0 else 1.0
            quantized = np.clip(np.rint(selected / scale), -127, 127).astype(np.int8)
            decoded[indices] = quantized.astype(np.float32) * np.float32(scale)
    else:
        raise ValueError(f"unknown codec: {codec}")
    next_residual = effective - decoded if error_feedback else np.zeros_like(vector)
    denominator = max(float(np.linalg.norm(effective)), EPS)
    return CodecResult(
        decoded=decoded.astype(np.float32),
        residual=next_residual.astype(np.float32),
        body_bytes=codec_body_bytes(codec, len(vector), topk),
        retained=int(retained),
        relative_l2_error=float(np.linalg.norm(effective - decoded) / denominator),
    )


def fragment_lengths(body_bytes: int, max_application_payload: int, header_bytes: int = 8) -> list[int]:
    """Return application-payload lengths, including the per-fragment header."""
    body = int(body_bytes)
    maximum = int(max_application_payload)
    header = int(header_bytes)
    capacity = maximum - header
    if body < 0 or capacity < 1:
        raise ValueError("invalid fragmentation parameters")
    if body == 0:
        return []
    fragments = []
    remaining = body
    while remaining:
        carried = min(capacity, remaining)
        fragments.append(header + carried)
        remaining -= carried
    return fragments


def lora_time_on_air_seconds(
    physical_payload_bytes: int,
    *,
    spreading_factor: int,
    bandwidth_hz: int = 125_000,
    coding_rate_denominator: int = 5,
    preamble_symbols: int = 8,
) -> float:
    """LoRa explicit-header packet airtime with payload CRC enabled.

    ``physical_payload_bytes`` includes the LoRaWAN MHDR/FHDR/FPort/MIC bytes.
    The equation is the standard Semtech LoRa modulation airtime equation.
    """
    payload = int(physical_payload_bytes)
    sf = int(spreading_factor)
    bw = int(bandwidth_hz)
    if payload < 0 or sf not in range(7, 13) or bw <= 0:
        raise ValueError("invalid LoRa airtime parameters")
    cr = int(coding_rate_denominator) - 4
    if cr not in range(1, 5):
        raise ValueError("coding rate denominator must be between 5 and 8")
    low_data_rate = 1 if sf >= 11 and bw == 125_000 else 0
    explicit_header = 0
    crc_enabled = 1
    symbol = (2**sf) / float(bw)
    preamble = (int(preamble_symbols) + 4.25) * symbol
    numerator = 8 * payload - 4 * sf + 28 + 16 * crc_enabled - 20 * explicit_header
    denominator = 4 * (sf - 2 * low_data_rate)
    payload_symbols = 8 + max(math.ceil(numerator / denominator) * (cr + 4), 0)
    return float(preamble + payload_symbols * symbol)


def message_airtime_seconds(
    body_bytes: int,
    *,
    max_application_payload: int,
    spreading_factor: int,
    fragment_header_bytes: int = 8,
    lorawan_phy_overhead_bytes: int = 13,
) -> dict[str, float | int]:
    fragments = fragment_lengths(body_bytes, max_application_payload, fragment_header_bytes)
    airtimes = [
        lora_time_on_air_seconds(
            length + int(lorawan_phy_overhead_bytes), spreading_factor=spreading_factor
        )
        for length in fragments
    ]
    return {
        "body_bytes": int(body_bytes),
        "fragments": len(fragments),
        "application_bytes_with_fragment_headers": int(sum(fragments)),
        "physical_bytes_minimum": int(
            sum(length + int(lorawan_phy_overhead_bytes) for length in fragments)
        ),
        "airtime_seconds": float(sum(airtimes)),
    }


def rounds_per_day(
    *,
    clients: int,
    uplink_fragments_per_client_round: int,
    uplink_airtime_per_client_round: float,
    downlink_fragments_per_client_round: int,
    downlink_airtime_per_client_round: float,
    client_uplink_airtime_budget: float,
    gateway_downlink_airtime_budget: float,
    max_downlink_messages_per_client: int | None = None,
) -> int:
    """Bound synchronous all-client rounds under client and gateway limits."""
    if min(clients, uplink_fragments_per_client_round, downlink_fragments_per_client_round) < 1:
        return 0
    limits = [
        math.floor(float(client_uplink_airtime_budget) / max(uplink_airtime_per_client_round, EPS)),
        math.floor(
            float(gateway_downlink_airtime_budget)
            / max(float(clients) * downlink_airtime_per_client_round, EPS)
        ),
    ]
    if max_downlink_messages_per_client is not None:
        limits.append(
            math.floor(int(max_downlink_messages_per_client) / downlink_fragments_per_client_round)
        )
    return max(0, int(min(limits)))


def retry_multiplier(loss_probability: float) -> float:
    probability = float(loss_probability)
    if probability < 0.0 or probability >= 1.0:
        raise ValueError("loss probability must be in [0,1)")
    return float(1.0 / (1.0 - probability))


__all__ = [
    "CodecResult",
    "codec_body_bytes",
    "encode_decode",
    "fragment_lengths",
    "lora_time_on_air_seconds",
    "message_airtime_seconds",
    "retry_multiplier",
    "rounds_per_day",
]
