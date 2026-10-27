"""Fresh DLR Tests 2/3/5 adapter for V41.

Listing functions are metadata-only. Sensor values are opened only by
``save_feature_stream`` after the source release gate and model lock exist.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import math
from pathlib import Path
import re
from typing import Iterator

import numpy as np
from scipy.io import loadmat, whosmat

import raw_online_federated_v12 as v12


HERE = Path(__file__).resolve().parent
RAW_ROOT = HERE / "v41" / "extracted" / "dlr"
FEATURE_DIM = 100
MAX_RAW_SAMPLES = 65_536


@dataclass(frozen=True, order=True)
class TargetDevice:
    test: int
    bearing: int

    @property
    def dataset(self) -> str:
        return "dlr"

    @property
    def unit(self) -> str:
        return f"Test_{self.test}_B{self.bearing}"

    @property
    def client_id(self) -> str:
        return f"dlr::{self.unit}"


def target_devices() -> list[TargetDevice]:
    result = [TargetDevice(test, bearing) for test in (2, 3, 5) for bearing in (1, 2)]
    if len(result) != 6 or len({item.client_id for item in result}) != 6:
        raise AssertionError("V41 needs six unique physical-bearing clients")
    return result


def _numbers(value: str | Path) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", str(value)))


def test_files(device: TargetDevice) -> list[Path]:
    root = RAW_ROOT / f"Test_{device.test}"
    paths = list(root.rglob(f"*Acc_Sensor_{device.bearing}.mat"))

    def key(path: Path) -> tuple[int, ...]:
        found = re.search(r"Data_(\d+)_to_(\d+)", path.name)
        return (int(found.group(1)),) if found else _numbers(path.name)

    return sorted(paths, key=key)


def raw_tokens(device: TargetDevice) -> list[str]:
    """Return paths only; never deserialize a MAT sensor matrix."""
    return [str(path.relative_to(RAW_ROOT)) for path in test_files(device)]


def split_acquisitions(matrix: np.ndarray, gap_seconds: float = 1.0) -> list[slice]:
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < 512:
        raise ValueError("invalid DLR time/value matrix")
    boundaries = np.flatnonzero(np.diff(values[:, 0]) > float(gap_seconds)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(values)]])
    return [slice(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= 512]


def _feature(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32).reshape(1, -1)
    values = values[:, np.all(np.isfinite(values), axis=0)]
    if values.shape[1] < 512:
        raise ValueError("DLR acquisition has fewer than 512 finite samples")
    if values.shape[1] > MAX_RAW_SAMPLES:
        stride = int(math.ceil(values.shape[1] / MAX_RAW_SAMPLES))
        values = values[:, ::stride][:, :MAX_RAW_SAMPLES]
    result = v12.local_preprocess_current_acquisition(values)
    if result.shape != (FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError("invalid V41 100-D causal feature")
    return np.asarray(result, dtype=np.float32)


def iter_records(device: TargetDevice) -> Iterator[tuple[int, np.ndarray, np.ndarray, str]]:
    index = 0
    files = test_files(device)
    if not files:
        raise FileNotFoundError(f"no extracted files for {device.client_id} below {RAW_ROOT}")
    for path in files:
        candidates = [
            name for name, shape, _ in whosmat(path)
            if len(shape) == 2 and shape[1] >= 2
        ]
        if len(candidates) != 1:
            raise ValueError(f"unique DLR sensor matrix not found: {path}")
        matrix = np.asarray(loadmat(path, variable_names=candidates)[candidates[0]])
        for segment in split_acquisitions(matrix):
            time = matrix[segment, 0]
            signal = matrix[segment, 1]
            context = np.asarray(
                [float(time[0]), float(time[-1]), float(np.median(np.diff(time)))],
                dtype=np.float64,
            )
            token = f"{path.relative_to(RAW_ROOT)}#segment={index}"
            yield index, _feature(signal), context, token
            index += 1
        del matrix
        gc.collect()


def save_feature_stream(path: Path, device: TargetDevice) -> dict[str, object]:
    features: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    tokens: list[str] = []
    for index, feature, context, token in iter_records(device):
        if index != len(features):
            raise RuntimeError(f"noncausal DLR acquisition order: {device.client_id}")
        features.append(feature)
        contexts.append(context)
        tokens.append(token)
    if len(features) <= 32:
        raise RuntimeError(f"{device.client_id} has too few acquisitions for V41")
    temporary = path.with_suffix(".tmp.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        temporary,
        features=np.asarray(features, dtype=np.float32),
        acquisition_index=np.arange(len(features), dtype=np.int32),
        context=np.asarray(contexts, dtype=np.float64),
        tokens=np.asarray(tokens),
        client_id=np.asarray(device.client_id),
        target_sensor_values_opened=np.asarray(True),
    )
    temporary.replace(path)
    return {
        "client_id": device.client_id,
        "acquisitions": len(features),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
    }


def load_feature_stream(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["features"], dtype=np.float32)
        indices = np.asarray(archive["acquisition_index"], dtype=np.int32)
    if (
        values.ndim != 2 or values.shape[1] != FEATURE_DIM
        or not np.all(np.isfinite(values))
        or not np.array_equal(indices, np.arange(len(indices), dtype=np.int32))
    ):
        raise ValueError(f"invalid V41 stream: {path}")
    return values, indices


__all__ = [
    "RAW_ROOT", "TargetDevice", "iter_records", "load_feature_stream",
    "raw_tokens", "save_feature_stream", "split_acquisitions", "target_devices",
]
