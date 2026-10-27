"""Raw, causal target-data adapters for V30.

Every iterator yields exactly one current acquisition at a time.  It never
receives a degradation label or onset.  The returned 100-D feature is the
unchanged V12 device-local transform used by the historical V29 source model.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

import raw_online_federated_v12 as v12


HERE = Path(__file__).resolve().parent
RAW_ROOT = HERE / "v30" / "extracted"
FEATURE_DIM = 100
MAX_RAW_SAMPLES = 65_536


@dataclass(frozen=True, order=True)
class TargetDevice:
    dataset: str
    unit: str
    lane: str

    @property
    def client_id(self) -> str:
        return f"{self.dataset}::{self.unit}"


@dataclass(frozen=True)
class FeatureRecord:
    index: int
    feature: np.ndarray
    context: np.ndarray
    token: str


def target_devices() -> list[TargetDevice]:
    devices = [
        *(TargetDevice("unsw", f"Test_{i}", "timeline") for i in range(1, 5)),
        TargetDevice("kaist", "Bearing_1", "timeline"),
        *(
            TargetDevice("dlr", f"Test_{test}_B{bearing}", "timeline")
            for test in (1, 4, 6, 7, 8)
            for bearing in (1, 2)
        ),
        *(
            TargetDevice("ottawa", f"Bearing_{i:02d}", "staged_state")
            for i in range(1, 21)
        ),
    ]
    if len(devices) != 35 or len({d.client_id for d in devices}) != 35:
        raise AssertionError("V30 target roster must contain 35 unique clients")
    return devices


def _numbers(value: str | Path) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", str(value)))


def _decimate(channels: np.ndarray, limit: int = MAX_RAW_SAMPLES) -> np.ndarray:
    values = np.asarray(channels, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(f"invalid raw acquisition {values.shape}")
    # Missing sensor samples are removed using only the current acquisition;
    # no future fill, backward fill, or cross-acquisition imputation is allowed.
    values = values[:, np.all(np.isfinite(values), axis=0)]
    if values.shape[1] < 512:
        raise ValueError(f"invalid raw acquisition {values.shape}")
    if values.shape[1] > int(limit):
        stride = int(math.ceil(values.shape[1] / int(limit)))
        values = values[:, ::stride][:, : int(limit)]
    return np.ascontiguousarray(values, dtype=np.float32)


def _feature(channels: np.ndarray) -> np.ndarray:
    result = v12.local_preprocess_current_acquisition(_decimate(channels))
    if result.shape != (FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError("invalid V30 feature")
    return result


def _unsw_files(device: TargetDevice) -> list[Path]:
    root = RAW_ROOT / "unsw" / "raw" / device.unit
    paths = list(root.rglob("6Hz/*.mat"))
    return sorted(paths, key=lambda path: _numbers(path.stem))


def _kaist_files() -> list[Path]:
    return sorted((RAW_ROOT / "kaist" / "raw").glob("*.csv"), key=lambda p: p.name)


def _ottawa_csv_root() -> Path:
    matches = [
        path for path in (RAW_ROOT / "ottawa").rglob("1_CSV_Raw_Data_Files*")
        if path.is_dir()
    ]
    if len(matches) != 1:
        raise FileNotFoundError("unique Ottawa raw-CSV directory not found")
    return matches[0]


def _ottawa_files(device: TargetDevice) -> list[Path]:
    number = int(device.unit.split("_")[-1])
    root = _ottawa_csv_root()
    healthy = list(root.rglob(f"H_{number}_0.csv"))
    prefix = "I" if number <= 5 else "O" if number <= 10 else "B" if number <= 15 else "C"
    later = [
        path
        for stage in (1, 2)
        for path in root.rglob(f"{prefix}_{number}_{stage}.csv")
    ]
    result = healthy + later
    if len(result) != 3:
        raise FileNotFoundError(f"Ottawa triplet missing for {device.client_id}")
    return result


def _dlr_files(device: TargetDevice) -> list[Path]:
    match = re.fullmatch(r"Test_(\d+)_B([12])", device.unit)
    if match is None:
        raise ValueError(device.unit)
    test, bearing = (int(value) for value in match.groups())
    root = RAW_ROOT / "dlr" / f"Test_{test}"
    paths = list(root.glob(f"*Acc_Sensor_{bearing}.mat"))

    def data_start(path: Path) -> int:
        found = re.search(r"Data_(\d+)_to_(\d+)", path.name)
        if found is None:
            raise ValueError(f"DLR data range missing: {path.name}")
        return int(found.group(1))

    return sorted(paths, key=data_start)


def raw_tokens(device: TargetDevice) -> list[str]:
    """Metadata-only inventory used by audit; it does not open sensor values."""
    if device.dataset == "unsw":
        paths = _unsw_files(device)
    elif device.dataset == "kaist":
        paths = _kaist_files()
    elif device.dataset == "ottawa":
        paths = _ottawa_files(device)
    elif device.dataset == "dlr":
        paths = _dlr_files(device)
    else:
        raise ValueError(device.dataset)
    return [str(path.relative_to(RAW_ROOT)) for path in paths]


def _unsw_records(device: TargetDevice) -> Iterator[FeatureRecord]:
    for index, path in enumerate(_unsw_files(device)):
        payload = loadmat(
            path,
            variable_names=["Fs", "accH", "accV", "loadCell", "tacho"],
            squeeze_me=True,
        )
        channels = np.vstack([payload["accH"], payload["accV"]])
        context = np.asarray(
            [
                float(np.asarray(payload.get("Fs", np.nan)).reshape(-1)[0]),
                float(np.mean(payload.get("loadCell", np.asarray([np.nan])))),
                float(np.mean(payload.get("tacho", np.asarray([np.nan])))),
                float(_numbers(path.stem)[0]),
            ],
            dtype=np.float32,
        )
        yield FeatureRecord(index, _feature(channels), context, str(path.relative_to(RAW_ROOT)))
        del payload, channels


def _kaist_records(device: TargetDevice) -> Iterator[FeatureRecord]:
    del device
    for index, path in enumerate(_kaist_files()):
        frame = pd.read_csv(path, header=None, dtype=np.float32)
        if frame.shape[1] < 2:
            raise ValueError(f"KAIST file has too few columns: {path}")
        channels = frame.iloc[:, :2].to_numpy(dtype=np.float32).T
        context_values = frame.iloc[:, 2:].to_numpy(dtype=np.float32)
        context = np.asarray(
            [
                float(np.nanmedian(context_values[:, column]))
                if context_values.ndim == 2 and column < context_values.shape[1]
                else np.nan
                for column in range(2)
            ],
            dtype=np.float32,
        )
        yield FeatureRecord(index, _feature(channels), context, str(path.relative_to(RAW_ROOT)))
        del frame, channels, context_values
        gc.collect()


def _ottawa_records(device: TargetDevice) -> Iterator[FeatureRecord]:
    index = 0
    for path in _ottawa_files(device):
        frame = pd.read_csv(path, dtype=np.float32)
        if frame.shape[1] < 2:
            raise ValueError(f"Ottawa file has too few columns: {path}")
        rows = len(frame)
        window = 42_000 if rows >= 420_000 else max(512, rows // 10)
        for start in range(0, rows - window + 1, window):
            block = frame.iloc[start : start + window]
            channels = block.iloc[:, :2].to_numpy(dtype=np.float32).T
            context = np.asarray(
                [float(np.nanmedian(block.iloc[:, col])) for col in range(2, min(6, frame.shape[1]))],
                dtype=np.float32,
            )
            yield FeatureRecord(
                index,
                _feature(channels),
                context,
                f"{path.relative_to(RAW_ROOT)}#window={start // window}",
            )
            index += 1
        del frame
        gc.collect()


def split_dlr_acquisitions(matrix: np.ndarray, gap_seconds: float = 1.0) -> list[slice]:
    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < 512:
        raise ValueError("invalid DLR time/value matrix")
    boundaries = np.flatnonzero(np.diff(values[:, 0]) > float(gap_seconds)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(values)]])
    return [slice(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= 512]


def _dlr_records(device: TargetDevice) -> Iterator[FeatureRecord]:
    index = 0
    for path in _dlr_files(device):
        variables = whosmat(path)
        candidates = [name for name, shape, _ in variables if len(shape) == 2 and shape[1] >= 2]
        if len(candidates) != 1:
            raise ValueError(f"unique DLR sensor matrix not found: {path}")
        matrix = np.asarray(loadmat(path, variable_names=candidates, squeeze_me=False)[candidates[0]])
        for segment in split_dlr_acquisitions(matrix):
            time = matrix[segment, 0]
            signal = matrix[segment, 1]
            context = np.asarray(
                [float(time[0]), float(time[-1]), float(np.median(np.diff(time)))],
                dtype=np.float32,
            )
            yield FeatureRecord(
                index,
                _feature(signal[None, :]),
                context,
                f"{path.relative_to(RAW_ROOT)}#segment={index}",
            )
            index += 1
        del matrix
        gc.collect()


def iter_feature_records(device: TargetDevice) -> Iterator[FeatureRecord]:
    if device.dataset == "unsw":
        yield from _unsw_records(device)
    elif device.dataset == "kaist":
        yield from _kaist_records(device)
    elif device.dataset == "ottawa":
        yield from _ottawa_records(device)
    elif device.dataset == "dlr":
        yield from _dlr_records(device)
    else:
        raise ValueError(device.dataset)


def save_feature_stream(path: Path, device: TargetDevice) -> dict[str, object]:
    features: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    tokens: list[str] = []
    for record in iter_feature_records(device):
        if record.index != len(features):
            raise RuntimeError(f"non-causal acquisition order for {device.client_id}")
        features.append(record.feature)
        contexts.append(record.context)
        tokens.append(record.token)
    if not features:
        raise RuntimeError(f"no acquisitions for {device.client_id}")
    width = max(len(value) for value in contexts)
    context_matrix = np.full((len(contexts), width), np.nan, dtype=np.float32)
    for row, value in enumerate(contexts):
        context_matrix[row, : len(value)] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        features=np.asarray(features, dtype=np.float32),
        context=context_matrix,
        tokens=np.asarray(tokens),
        dataset=np.asarray(device.dataset),
        unit=np.asarray(device.unit),
        lane=np.asarray(device.lane),
    )
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return {"client_id": device.client_id, "acquisitions": len(features), "sha256": digest}


def load_feature_stream(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        context = np.asarray(archive["context"], dtype=np.float32)
        tokens = np.asarray(archive["tokens"]).astype(str)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM or len(features) != len(tokens):
        raise ValueError(f"invalid V30 stream: {path}")
    return features, context, tokens


def inventory_manifest() -> dict[str, object]:
    rows = []
    for device in target_devices():
        tokens = raw_tokens(device)
        rows.append(
            {
                "client_id": device.client_id,
                "dataset": device.dataset,
                "unit": device.unit,
                "lane": device.lane,
                "raw_files": len(tokens),
                "listing_sha256": hashlib.sha256(
                    json.dumps(tokens, sort_keys=True).encode("utf-8")
                ).hexdigest().upper(),
            }
        )
    return {"clients": rows, "target_sensor_values_opened": False}


__all__ = [
    "FeatureRecord",
    "TargetDevice",
    "inventory_manifest",
    "iter_feature_records",
    "load_feature_stream",
    "raw_tokens",
    "save_feature_stream",
    "split_dlr_acquisitions",
    "target_devices",
]
