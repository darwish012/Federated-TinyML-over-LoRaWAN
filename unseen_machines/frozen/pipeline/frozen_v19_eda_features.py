"""Exact raw-acquisition EDA feature definitions consumed by V19 onset.

Transcribed from the locked ``bearing_eda_deep_analysis.ipynb`` feature cell.
Only the five V19 inputs are returned.  This path is independent of detector
features and must not consume detector outputs.
"""

from __future__ import annotations

import math

import numpy as np


EPS = 1e-12
BAND_EDGES = np.asarray([0.00, 0.05, 0.10, 0.20, 0.40, 1.00])
MAX_FFT_SAMPLES = 16_384


def acquisition_features(channels: np.ndarray) -> dict[str, float]:
    values = np.asarray(channels, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError(f"Expected channels x samples, got {values.shape}")
    rows = [_channel_features(row) for row in values]
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in (
            "rms_mean",
            "kurtosis_mean",
            "crest_factor_mean",
            "spectral_entropy_mean",
            "band_4_relative_power_mean",
        )
    }


def _channel_features(signal: np.ndarray) -> dict[str, float]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if not np.all(finite):
        x = x[finite]
    if len(x) < 2:
        raise ValueError("Channel has fewer than two finite samples")

    mean = float(np.mean(x))
    centered = x - mean
    std = float(np.std(centered))
    absolute = np.abs(x)
    rms = float(np.sqrt(np.mean(x**2)))
    peak_abs = float(np.max(absolute))
    kurtosis = float(np.mean(centered**4) / std**4) if std > EPS else 0.0

    if len(centered) > MAX_FFT_SAMPLES:
        stride = math.ceil(len(centered) / MAX_FFT_SAMPLES)
        spectrum_input = centered[::stride][:MAX_FFT_SAMPLES]
    else:
        spectrum_input = centered
    power = np.abs(np.fft.rfft(spectrum_input * np.hanning(len(spectrum_input)))) ** 2
    power = power[1:]
    if not len(power) or float(np.sum(power)) <= EPS:
        probabilities = np.ones(max(len(power), 1)) / max(len(power), 1)
        frequencies = np.linspace(0, 1, len(probabilities))
    else:
        probabilities = power / np.sum(power)
        frequencies = np.linspace(0, 1, len(probabilities), endpoint=True)
    spectral_entropy = float(
        -np.sum(probabilities * np.log(probabilities + EPS))
        / np.log(max(len(probabilities), 2))
    )
    mask = (frequencies >= BAND_EDGES[4]) & (frequencies <= BAND_EDGES[5])
    return {
        "rms_mean": rms,
        "kurtosis_mean": kurtosis,
        "crest_factor_mean": peak_abs / max(rms, EPS),
        "spectral_entropy_mean": spectral_entropy,
        "band_4_relative_power_mean": float(np.sum(probabilities[mask])),
    }


__all__ = ["acquisition_features"]
