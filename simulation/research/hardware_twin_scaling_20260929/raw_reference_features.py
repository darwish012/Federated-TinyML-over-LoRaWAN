"""The float64 100-D per-acquisition feature extractor, transcribed verbatim
from research/external_dtu_fe8_20260922/kaggle_notebook.py CELL 2 (the
"source of truth" named in this study's task brief). That cell is itself
"Transcribed from raw_online_federated_v12.py" and carries its own
bit-for-bit self-check against a local reference-case bundle.

This module is intentionally NOT an import of kaggle_notebook.py (that file
is a Kaggle %-cell script; its cell 1 opens a /kaggle/input bundle that does
not exist on this machine, so importing it top-to-bottom would fail before
reaching cell 2) and NOT an import of raw_online_federated_v12.py either
(this module must stay correct even if v12.py is later touched by someone
else's work). Instead the four function bodies below are copied character-
for-character from kaggle_notebook.py's cell 2, and `_selfcheck()` proves at
import time that they agree with raw_online_federated_v12.py's
_time_features / _spectral_features / local_preprocess_current_acquisition
on random synthetic signals -- the same kind of transcription-proof the
kaggle notebook itself performs against its bundle.

FEATURE_INDICES is the fixed 22-of-100 selection the on-device model uses
(data_cache.FEATURE_INDICES / results/healthy_only_suite_20260907/cache/
manifest.json config.feature_indices): per channel, the 8 signed-log time
features [rms, peak-to-peak, kurtosis, skewness, crest, clearance, shape,
impulse] (indices 0-7) and 3 spectral features [entropy, centroid, spread]
(indices 28-30), each taken twice -- once from the channel-mean block
(indices 0-31) and once from the "largest |value| channel" block
(indices 64-95, i.e. +64 offset) -- omitting the spectral "dominant"
feature and the raw mean/stddev/maximum/minimum time features (indices
8-11 / 39-42/103 etc.) entirely.
"""
from __future__ import annotations

import math

import numpy as np

FEATURE_DIM, CHANNEL_FEATURE_DIM, FFT_BANDS, MAX_FFT_SAMPLES = 100, 32, 16, 16384

FEATURE_INDICES = np.asarray(
    [0, 1, 2, 3, 4, 5, 6, 7, 28, 29, 30, 64, 65, 66, 67, 68, 69, 70, 71, 92, 93, 94],
    dtype=np.int64,
)


# ---------------------------------------------------------------------------
# Verbatim transcription of kaggle_notebook.py CELL 2 (float64).
# ---------------------------------------------------------------------------
def _signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


def _time_features(signal):
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    mean = float(np.mean(x)); centered = x - mean; absolute = np.abs(x)
    maximum, minimum = float(np.max(x)), float(np.min(x))
    rms = float(np.sqrt(np.mean(x * x)))
    stddev = float(np.sqrt(np.mean(centered * centered)))
    abs_mean = float(np.mean(absolute)); max_abs = float(np.max(absolute))
    if stddev > 1e-12:
        skewness = float(np.mean(centered ** 3) / stddev ** 3)
        kurtosis = float(np.mean(centered ** 4) / stddev ** 4)
    else:
        skewness = kurtosis = 0.0
    sqrt_mean_sq = float(np.mean(np.sqrt(absolute)) ** 2)
    raw = np.asarray([rms, maximum - minimum, kurtosis, skewness,
                      max_abs / max(rms, 1e-12), max_abs / max(sqrt_mean_sq, 1e-12),
                      rms / max(abs_mean, 1e-12), max_abs / max(abs_mean, 1e-12),
                      mean, stddev, maximum, minimum], dtype=np.float64)
    return _signed_log(raw)


def _spectral_features(signal):
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(x) > MAX_FFT_SAMPLES:
        stride = int(math.ceil(len(x) / MAX_FFT_SAMPLES))
        x = x[::stride][:MAX_FFT_SAMPLES]
    centered = x - float(np.mean(x))
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))[1:] ** 2
    power = np.maximum(power, 1e-20)
    bands = np.array_split(power, FFT_BANDS)
    log_power = np.asarray([0.5 * math.log(float(np.mean(b)) + 1e-20) for b in bands])
    probabilities = power / float(np.sum(power))
    frequencies = np.linspace(0.0, 1.0, len(power), endpoint=True)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-20))) / math.log(len(power))
    centroid = float(np.sum(probabilities * frequencies))
    spread = float(np.sqrt(np.sum(probabilities * (frequencies - centroid) ** 2)))
    dominant = float(frequencies[int(np.argmax(power))])
    return np.concatenate([log_power, [entropy, centroid, spread, dominant]])


def local_preprocess_current_acquisition(channels):
    values = np.asarray(channels, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 512:
        raise ValueError(f'invalid current acquisition shape {values.shape}')
    if not np.all(np.isfinite(values)):
        raise ValueError('non-finite raw acquisition')
    per_channel = np.asarray([np.concatenate([_time_features(r), _spectral_features(r)])
                              for r in values], dtype=np.float64)
    mean = np.mean(per_channel, axis=0)
    spread = np.std(per_channel, axis=0)
    extreme = per_channel[np.argmax(np.abs(per_channel), axis=0), np.arange(CHANNEL_FEATURE_DIM)]
    rms = np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2, axis=1))
    if len(values) > 1:
        limit = min(values.shape[1], 4096)
        correlation = np.corrcoef(values[:, :limit])
        offdiag = np.abs(correlation[np.triu_indices(len(values), 1)])
        offdiag = offdiag[np.isfinite(offdiag)]
        corr_mean = float(np.mean(offdiag)) if len(offdiag) else 0.0
        corr_max = float(np.max(offdiag)) if len(offdiag) else 0.0
    else:
        corr_mean = corr_max = 0.0
    cross = np.asarray([corr_mean, corr_max,
                        math.log((float(np.max(rms)) + 1e-12) / (float(np.min(rms)) + 1e-12)),
                        math.log1p(float(np.sqrt(np.sum(rms * rms))))])
    result = np.concatenate([mean, spread, extreme, cross]).astype(np.float32)
    if result.shape != (FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError('invalid local feature vector')
    return result


def select_22(features100: np.ndarray) -> np.ndarray:
    """The fixed 22-of-100 selection (FEATURE_INDICES), float64."""
    return np.asarray(features100, dtype=np.float64)[..., FEATURE_INDICES]


# ---------------------------------------------------------------------------
# Self-check against raw_online_federated_v12.py at import time.
# ---------------------------------------------------------------------------
def _selfcheck() -> None:
    import raw_online_federated_v12 as v12

    rng = np.random.default_rng(20260929)
    for n_channels, n_samples in ((1, 600), (2, 4097), (1, 20001), (3, 1024)):
        signal = rng.normal(size=(n_channels, n_samples)).astype(np.float32)
        mine = local_preprocess_current_acquisition(signal)
        theirs = v12.local_preprocess_current_acquisition(signal)
        if not np.array_equal(mine, theirs):
            raise RuntimeError(
                "raw_reference_features transcription diverges from "
                "raw_online_federated_v12.local_preprocess_current_acquisition "
                f"(shape {signal.shape}); STOP, do not proceed with a guess"
            )


_selfcheck()
