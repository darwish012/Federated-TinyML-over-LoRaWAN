"""V12: raw, causal, one-device-per-bearing federated simulation.

This is a new experiment.  It deliberately does not import or consume any
V7/V8/V10/V11 feature cache.  Every simulated device opens its own raw sensor
source in chronological order, converts only the current acquisition to a
feature vector, and retains only past local state.  The server receives model
parameters and aggregate metrics, never raw signals or feature matrices.

The frozen split is 40 source devices and 17 held-out devices across Ferrara,
XJTU-SY, PRONOSTIA, IMS, HIT, and LDM-UPB.  The experiment is retrospective:
several dataset outcomes were inspected in earlier studies.  HIT contains
fixed-fault recordings and is therefore reported as fault-state transfer,
not as run-to-failure early warning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import average_precision_score

from learning_failure_experiments import DenseAE


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "raw_online_federated_v12_protocol.json"
OUT = HERE / "results" / "raw_online_federated_v12"
SOURCE_JOURNALS = OUT / "source_node_journals"
TARGET_JOURNALS = OUT / "heldout_node_journals"
DATA_ROOT = Path(r"E:\masters\datasets")

FEATURE_DIM = 100
CHANNEL_FEATURE_DIM = 32
HEAD_DIM = 203
COMMISSIONING = 20
FFT_BANDS = 16
MAX_FFT_SAMPLES = 16_384
AE_DIMS = (FEATURE_DIM, 32, 8, 32, FEATURE_DIM)
SCHEMA_VERSION = 1
SEED = 20_260_722


@dataclass(frozen=True)
class ClientSpec:
    dataset: str
    unit: str
    role: str
    task: str
    status: str

    @property
    def client_id(self) -> str:
        return f"{self.dataset}::{self.unit}"


@dataclass
class RawAcquisition:
    index: int
    channels: np.ndarray
    source_token: str


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_digest() -> str:
    return _sha256(PROTOCOL_PATH)


def _natural_number(path: Path | str) -> int:
    matches = re.findall(r"(\d+)", Path(path).stem)
    if not matches:
        raise ValueError(f"no numeric chronology in {path}")
    return int(matches[-1])


def _all_client_specs() -> list[ClientSpec]:
    protocol = _read_json(PROTOCOL_PATH)
    train = dict(protocol["train_clients"])
    test = dict(protocol["test_clients"])
    ims_failed = set(dict(protocol["known_source_status"])["ims_failed"])
    specs: list[ClientSpec] = []
    for role, mapping in (("source_train", train), ("heldout_test", test)):
        for dataset, units_raw in mapping.items():
            for unit_raw in units_raw:
                unit = str(unit_raw)
                if dataset == "hit":
                    task = "fixed_fault_transfer"
                    status = "healthy" if unit in {"data1", "data2"} else "fixed_fault"
                elif dataset == "ims":
                    task = "run_to_failure"
                    status = "failed" if unit in ims_failed else "censored"
                else:
                    task = "run_to_failure"
                    status = "failed"
                specs.append(ClientSpec(dataset, unit, role, task, status))
    if len(specs) != 57:
        raise ValueError(f"protocol enumerates {len(specs)} clients, expected 57")
    if sum(spec.role == "source_train" for spec in specs) != 40:
        raise ValueError("protocol must contain 40 source clients")
    if sum(spec.role == "heldout_test" for spec in specs) != 17:
        raise ValueError("protocol must contain 17 held-out clients")
    if len({spec.client_id for spec in specs}) != 57:
        raise ValueError("duplicate client identity in frozen split")
    return specs


def _xjtu_directory(unit: str) -> Path:
    matches = []
    for directory in (DATA_ROOT / "xjtu").rglob(unit):
        if directory.is_dir() and list(directory.glob("*.csv")):
            matches.append(directory)
    if len(matches) != 1:
        raise ValueError(f"expected one raw XJTU directory for {unit}, found {matches}")
    return matches[0]


def _pronostia_directory(unit: str) -> Path:
    base = DATA_ROOT / "pronostia" / "phm-ieee-2012-data-challenge-dataset-master"
    matches = [base / "Learning_set" / unit, base / "Full_Test_Set" / unit]
    found = [path for path in matches if path.is_dir()]
    if len(found) != 1:
        raise ValueError(f"expected one official PRONOSTIA directory for {unit}")
    return found[0]


def _ims_mapping(unit: str) -> tuple[Path, tuple[int, ...]]:
    match = re.fullmatch(r"R([123])_B([1-4])", unit)
    if not match:
        raise ValueError(f"invalid IMS unit {unit}")
    run, bearing = int(match.group(1)), int(match.group(2))
    if run == 1:
        directory = DATA_ROOT / "ims" / "ims_raw" / "1st_test"
        columns = (2 * (bearing - 1), 2 * (bearing - 1) + 1)
    elif run == 2:
        directory = DATA_ROOT / "ims" / "ims_raw" / "2nd_test"
        columns = (bearing - 1,)
    else:
        directory = DATA_ROOT / "ims" / "ims_raw" / "4th_test" / "txt"
        columns = (bearing - 1,)
    return directory, columns


def _raw_listing(spec: ClientSpec) -> tuple[list[str], int]:
    """Return names/size evidence without opening sensor values."""
    if spec.dataset == "ferrara":
        files = sorted((DATA_ROOT / "ferrara" / spec.unit).rglob("*.mat"), key=_natural_number)
    elif spec.dataset == "xjtu":
        files = sorted(_xjtu_directory(spec.unit).glob("*.csv"), key=_natural_number)
    elif spec.dataset == "pronostia":
        files = sorted(_pronostia_directory(spec.unit).glob("acc_*.csv"), key=_natural_number)
    elif spec.dataset == "ims":
        directory, _ = _ims_mapping(spec.unit)
        files = sorted(path for path in directory.iterdir() if path.is_file())
    elif spec.dataset == "hit":
        files = [DATA_ROOT / "hit" / "HIT-dataset" / f"{spec.unit}.npy"]
    elif spec.dataset == "ldm_upb":
        archive_path = DATA_ROOT / "ldm_upb" / f"{spec.unit}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            names = sorted(
                name for name in archive.namelist()
                if re.search(rf"/vibrationData/data_{spec.unit}_M\d+\.mat$", name, re.I)
            )
            return [f"{archive_path.name}!{name}" for name in names], archive_path.stat().st_size
    else:
        raise ValueError(spec.dataset)
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"missing raw chronology for {spec.client_id}")
    return [str(path.relative_to(DATA_ROOT)) for path in files], sum(path.stat().st_size for path in files)


def _listing_digest(names: Sequence[str], total_bytes: int) -> str:
    payload = json.dumps({"names": list(names), "bytes": int(total_bytes)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def audit_raw_split() -> dict[str, object]:
    rows = []
    for spec in _all_client_specs():
        names, total_bytes = _raw_listing(spec)
        rows.append(
            {
                "client_id": spec.client_id,
                "dataset": spec.dataset,
                "unit": spec.unit,
                "role": spec.role,
                "task": spec.task,
                "status": spec.status,
                "raw_acquisitions": len(names),
                "raw_bytes_or_archive_bytes": total_bytes,
                "raw_listing_sha256": _listing_digest(names, total_bytes),
                "first_raw_token": names[0],
                "last_raw_token": names[-1],
            }
        )
    _write_csv(OUT / "raw_client_manifest.csv", rows)
    payload = {
        "protocol_sha256": _protocol_digest(),
        "clients": len(rows),
        "source_clients": sum(row["role"] == "source_train" for row in rows),
        "heldout_clients": sum(row["role"] == "heldout_test" for row in rows),
        "datasets": sorted({row["dataset"] for row in rows}),
        "all_raw_listing_digest": hashlib.sha256(
            "\n".join(str(row["raw_listing_sha256"]) for row in rows).encode()
        ).hexdigest(),
        "server_received_raw": False,
        "server_preprocessed_features": False,
    }
    _write_json(OUT / "raw_audit.json", payload)
    return payload


def _ferrara_signal(path: Path) -> np.ndarray:
    payload = loadmat(path, variable_names=["y"], squeeze_me=True)
    if "y" not in payload:
        payload = loadmat(path, squeeze_me=True)
        arrays = [
            np.asarray(value) for key, value in payload.items()
            if not key.startswith("__") and np.asarray(value).size >= 512
        ]
        if not arrays:
            raise ValueError(f"no vibration array in {path}")
        signal = max(arrays, key=lambda values: values.size)
    else:
        signal = payload["y"]
    return np.asarray(signal, dtype=np.float32).reshape(1, -1)


def _xjtu_signal(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"invalid XJTU acquisition {path}")
    return values[:, :2].T


def _pronostia_signal(path: Path) -> np.ndarray:
    text = path.read_bytes().decode("ascii")
    values = np.fromstring(text.replace(",", " ").replace(";", " "), sep=" ")
    if values.size % 6:
        raise ValueError(f"malformed PRONOSTIA acquisition {path}")
    rows = values.reshape(-1, 6)
    return np.asarray(rows[:, 4:6].T, dtype=np.float32)


def _ims_signal(path: Path, columns: tuple[int, ...]) -> np.ndarray:
    values = np.loadtxt(path, delimiter="\t", usecols=columns, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    return values.T


def _ldm_signal(payload: bytes) -> np.ndarray:
    values = loadmat(
        io.BytesIO(payload),
        variable_names=["accHorizFrontal_C", "accHorizRear_A"],
        squeeze_me=True,
    )
    rows = []
    for key in ("accHorizFrontal_C", "accHorizRear_A"):
        if key in values:
            rows.append(np.asarray(values[key], dtype=np.float32).reshape(-1))
    if not rows:
        raise ValueError("LDM acquisition lacks vibration channels")
    size = min(len(row) for row in rows)
    return np.stack([row[:size] for row in rows])


def iter_raw_acquisitions(spec: ClientSpec, *, start: int = 0) -> Iterator[RawAcquisition]:
    """Yield current raw acquisition only, in strict chronological order."""
    expected = int(start)
    if spec.dataset == "ferrara":
        files = sorted((DATA_ROOT / "ferrara" / spec.unit).rglob("*.mat"), key=_natural_number)
        reader = _ferrara_signal
    elif spec.dataset == "xjtu":
        files = sorted(_xjtu_directory(spec.unit).glob("*.csv"), key=_natural_number)
        reader = _xjtu_signal
    elif spec.dataset == "pronostia":
        files = sorted(_pronostia_directory(spec.unit).glob("acc_*.csv"), key=_natural_number)
        reader = _pronostia_signal
    elif spec.dataset == "ims":
        directory, columns = _ims_mapping(spec.unit)
        files = sorted(path for path in directory.iterdir() if path.is_file())
        reader = lambda path: _ims_signal(path, columns)
    elif spec.dataset == "hit":
        path = DATA_ROOT / "hit" / "HIT-dataset" / f"{spec.unit}.npy"
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        for index in range(start, len(values)):
            if index != expected:
                raise RuntimeError("non-causal HIT read")
            channels = np.asarray(values[index, :6, :], dtype=np.float32)
            yield RawAcquisition(index, channels, f"{path.name}#{index}")
            expected += 1
        return
    elif spec.dataset == "ldm_upb":
        path = DATA_ROOT / "ldm_upb" / f"{spec.unit}.zip"
        with zipfile.ZipFile(path) as archive:
            names = sorted(
                (name for name in archive.namelist() if re.search(
                    rf"/vibrationData/data_{spec.unit}_M\d+\.mat$", name, re.I
                )),
                key=_natural_number,
            )
            for index, name in enumerate(names[start:], start=start):
                if index != expected:
                    raise RuntimeError("non-causal LDM read")
                yield RawAcquisition(index, _ldm_signal(archive.read(name)), f"{path.name}!{name}")
                expected += 1
        return
    else:
        raise ValueError(spec.dataset)
    for index, path in enumerate(files[start:], start=start):
        if index != expected:
            raise RuntimeError(f"non-causal read for {spec.client_id}")
        yield RawAcquisition(index, reader(path), str(path.relative_to(DATA_ROOT)))
        expected += 1


def _signed_log(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _time_features(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    mean = float(np.mean(x))
    centered = x - mean
    absolute = np.abs(x)
    maximum, minimum = float(np.max(x)), float(np.min(x))
    rms = float(np.sqrt(np.mean(x * x)))
    stddev = float(np.sqrt(np.mean(centered * centered)))
    abs_mean = float(np.mean(absolute))
    max_abs = float(np.max(absolute))
    if stddev > 1e-12:
        skewness = float(np.mean(centered**3) / stddev**3)
        kurtosis = float(np.mean(centered**4) / stddev**4)
    else:
        skewness = kurtosis = 0.0
    sqrt_mean_sq = float(np.mean(np.sqrt(absolute)) ** 2)
    raw = np.asarray(
        [
            rms, maximum - minimum, kurtosis, skewness,
            max_abs / max(rms, 1e-12), max_abs / max(sqrt_mean_sq, 1e-12),
            rms / max(abs_mean, 1e-12), max_abs / max(abs_mean, 1e-12),
            mean, stddev, maximum, minimum,
        ],
        dtype=np.float64,
    )
    return _signed_log(raw)


def _spectral_features(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(x) > MAX_FFT_SAMPLES:
        stride = int(math.ceil(len(x) / MAX_FFT_SAMPLES))
        x = x[::stride][:MAX_FFT_SAMPLES]
    centered = x - float(np.mean(x))
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered))))[1:] ** 2
    power = np.maximum(power, 1e-20)
    bands = np.array_split(power, FFT_BANDS)
    log_power = np.asarray([0.5 * math.log(float(np.mean(band)) + 1e-20) for band in bands])
    probabilities = power / float(np.sum(power))
    frequencies = np.linspace(0.0, 1.0, len(power), endpoint=True)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-20))) / math.log(len(power))
    centroid = float(np.sum(probabilities * frequencies))
    spread = float(np.sqrt(np.sum(probabilities * (frequencies - centroid) ** 2)))
    dominant = float(frequencies[int(np.argmax(power))])
    return np.concatenate([log_power, [entropy, centroid, spread, dominant]])


def local_preprocess_current_acquisition(channels: np.ndarray) -> np.ndarray:
    """Device-only transform; accepts exactly one current raw acquisition."""
    values = np.asarray(channels, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 512:
        raise ValueError(f"invalid current acquisition shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite raw acquisition")
    per_channel = np.asarray(
        [np.concatenate([_time_features(row), _spectral_features(row)]) for row in values],
        dtype=np.float64,
    )
    mean = np.mean(per_channel, axis=0)
    spread = np.std(per_channel, axis=0)
    extreme = per_channel[np.argmax(np.abs(per_channel), axis=0), np.arange(CHANNEL_FEATURE_DIM)]
    rms = np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2, axis=1))
    if len(values) > 1:
        limit = min(values.shape[1], 4096)
        sample = values[:, :limit]
        correlation = np.corrcoef(sample)
        offdiag = np.abs(correlation[np.triu_indices(len(values), 1)])
        offdiag = offdiag[np.isfinite(offdiag)]
        corr_mean = float(np.mean(offdiag)) if len(offdiag) else 0.0
        corr_max = float(np.max(offdiag)) if len(offdiag) else 0.0
    else:
        corr_mean = corr_max = 0.0
    cross = np.asarray(
        [
            corr_mean,
            corr_max,
            math.log((float(np.max(rms)) + 1e-12) / (float(np.min(rms)) + 1e-12)),
            math.log1p(float(np.sqrt(np.sum(rms * rms)))),
        ]
    )
    result = np.concatenate([mean, spread, extreme, cross]).astype(np.float32)
    if result.shape != (FEATURE_DIM,) or not np.all(np.isfinite(result)):
        raise ValueError("invalid local feature vector")
    return result


def _journal_path(spec: ClientSpec) -> Path:
    root = SOURCE_JOURNALS if spec.role == "source_train" else TARGET_JOURNALS
    safe = spec.client_id.replace("::", "__")
    return root / f"{safe}_local_history_v{SCHEMA_VERSION}.npz"


def _journal_signature(spec: ClientSpec) -> str:
    names, total_bytes = _raw_listing(spec)
    payload = {
        "protocol_sha256": _protocol_digest(),
        "schema_version": SCHEMA_VERSION,
        "client_id": spec.client_id,
        "listing": _listing_digest(names, total_bytes),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _fit_local_normalizer(commissioning: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(commissioning, dtype=np.float64)
    if values.shape != (COMMISSIONING, FEATURE_DIM):
        raise ValueError(f"commissioning shape {values.shape} is not {(COMMISSIONING, FEATURE_DIM)}")
    location = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - location), axis=0)
    fallback = np.std(values, axis=0, ddof=1)
    scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
    return location.astype(np.float32), scale.astype(np.float32)


def _normalize(features: np.ndarray, location: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = (np.asarray(features, dtype=np.float32) - location) / scale
    return np.clip(values, -12.0, 12.0).astype(np.float32)


class FederatedNode:
    """All feature history and labels remain private to one simulated device."""

    def __init__(self, spec: ClientSpec, features: np.ndarray, source_tokens: Sequence[str]):
        self.spec = spec
        self._features = np.asarray(features, dtype=np.float32)
        self._source_tokens = tuple(str(value) for value in source_tokens)
        if len(self._features) <= COMMISSIONING:
            raise ValueError(f"{spec.client_id} is too short")
        if self._features.shape[1:] != (FEATURE_DIM,):
            raise ValueError(f"bad private feature shape for {spec.client_id}")
        self.location, self.scale = _fit_local_normalizer(self._features[:COMMISSIONING])
        self._normalized = _normalize(self._features, self.location, self.scale)
        self._pca_center: np.ndarray | None = None
        self._pca_basis: np.ndarray | None = None
        self._descriptor_cache: np.ndarray | None = None

    @property
    def observed_count(self) -> int:
        return len(self._features)

    def commissioning_count(self) -> int:
        return COMMISSIONING

    def local_ae_update(
        self,
        global_flat: np.ndarray,
        base_weights: list[np.ndarray],
        *,
        epochs: int,
        learning_rate: float,
    ) -> tuple[np.ndarray, float]:
        """Train only on locally stored past commissioning observations."""
        model = DenseAE(AE_DIMS, weights=base_weights)
        model.set_trainable(global_flat, reset_optimizer=True)
        samples = self._normalized[:COMMISSIONING]
        losses = model.train(samples, epochs=epochs, lr=learning_rate)
        return model.get_trainable(), float(losses[-1])

    def private_labels(self) -> np.ndarray:
        n = self.observed_count
        labels = np.zeros(n, dtype=np.float32)
        if self.spec.task == "fixed_fault_transfer":
            if self.spec.status == "fixed_fault":
                labels[COMMISSIONING:] = 1.0
        elif self.spec.status == "failed":
            labels[int(math.ceil(0.60 * n)) :] = 1.0
        return labels

    def _fit_private_pca(self) -> None:
        commissioning = np.asarray(self._normalized[:COMMISSIONING], dtype=np.float64)
        center = np.mean(commissioning, axis=0)
        _, _, vt = np.linalg.svd(commissioning - center, full_matrices=False)
        self._pca_center = center.astype(np.float32)
        self._pca_basis = vt[: min(8, len(vt))].astype(np.float32)

    @staticmethod
    def _robust_score(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        ref = np.asarray(reference, dtype=np.float64)
        location = float(np.median(ref))
        scale = 1.4826 * float(np.median(np.abs(ref - location)))
        if scale < 1e-6:
            scale = max(float(np.std(ref, ddof=1)), 1e-6)
        return np.clip((np.asarray(values, dtype=np.float64) - location) / scale, -12.0, 30.0)

    def private_descriptors(self, model: DenseAE) -> np.ndarray:
        """Return local model input for private FL gradient computation only."""
        if self._descriptor_cache is not None:
            return self._descriptor_cache
        z = np.asarray(self._normalized, dtype=np.float32)
        if self._pca_basis is None:
            self._fit_private_pca()
        assert self._pca_center is not None and self._pca_basis is not None
        ae_loss = np.asarray(model.losses(z), dtype=np.float64)
        centered = z - self._pca_center
        reconstruction = (centered @ self._pca_basis.T) @ self._pca_basis
        pca_loss = np.mean((centered - reconstruction) ** 2, axis=1)
        distance = np.mean(np.abs(z), axis=1)
        ae_z = self._robust_score(np.log1p(ae_loss), np.log1p(ae_loss[:COMMISSIONING]))
        pca_z = self._robust_score(np.log1p(pca_loss), np.log1p(pca_loss[:COMMISSIONING]))
        dist_z = self._robust_score(distance, distance[:COMMISSIONING])
        delta = np.zeros_like(z)
        for index in range(COMMISSIONING, len(z)):
            start = max(0, index - 5)
            delta[index] = z[index] - np.mean(z[start:index], axis=0)
        self._descriptor_cache = np.column_stack([z, delta, ae_z, pca_z, dist_z]).astype(np.float32)
        return self._descriptor_cache

    def label_free_score(self, model: DenseAE) -> np.ndarray:
        descriptors = self.private_descriptors(model)
        components = descriptors[:, -3:]
        return np.max(components, axis=1).astype(np.float32)

    def private_head_gradient(
        self,
        model: DenseAE,
        weights: np.ndarray,
        bias: float,
        l2: float,
    ) -> tuple[np.ndarray, float, float]:
        x = self.private_descriptors(model)[COMMISSIONING:]
        y = self.private_labels()[COMMISSIONING:]
        logits = np.clip(x @ weights + bias, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        positive = y > 0.5
        negative = ~positive
        sample_weight = np.zeros(len(y), dtype=np.float64)
        if np.any(positive) and np.any(negative):
            sample_weight[positive] = 0.5 / np.sum(positive)
            sample_weight[negative] = 0.5 / np.sum(negative)
        else:
            sample_weight[:] = 1.0 / max(len(y), 1)
        residual = sample_weight * (probability - y)
        grad_w = x.T @ residual + l2 * weights
        grad_b = float(np.sum(residual))
        eps = 1e-8
        loss = -float(np.sum(sample_weight * (y * np.log(probability + eps) + (1-y) * np.log(1-probability + eps))))
        return grad_w.astype(np.float32), grad_b, loss

    def private_head_probability(self, model: DenseAE, weights: np.ndarray, bias: float) -> np.ndarray:
        x = self.private_descriptors(model)
        logits = np.clip(x @ weights + bias, -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def build_local_history(spec: ClientSpec, *, rebuild: bool = False) -> Path:
    """Run the physical-node loop; no full-life raw array is ever loaded."""
    path = _journal_path(spec)
    signature = _journal_signature(spec)
    if path.is_file() and not rebuild:
        with np.load(path, allow_pickle=False) as payload:
            stored = bytes(payload["signature_utf8"]).decode("utf-8")
        if stored == signature:
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    features: list[np.ndarray] = []
    tokens: list[str] = []
    expected = 0
    started = time.perf_counter()
    for acquisition in iter_raw_acquisitions(spec):
        if acquisition.index != expected:
            raise RuntimeError(f"future access detected at {spec.client_id}: {acquisition.index} != {expected}")
        features.append(local_preprocess_current_acquisition(acquisition.channels))
        tokens.append(acquisition.source_token)
        expected += 1
        if expected == 1 or expected % 250 == 0:
            print(f"  {spec.client_id}: observed {expected} raw acquisitions", flush=True)
        del acquisition
    array = np.asarray(features, dtype=np.float32)
    if len(array) <= COMMISSIONING:
        raise ValueError(f"{spec.client_id} has only {len(array)} observations")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        signature_utf8=np.frombuffer(signature.encode(), dtype=np.uint8),
        features=array,
        source_tokens_utf8=np.frombuffer("\n".join(tokens).encode(), dtype=np.uint8),
        observed_indices=np.arange(len(array), dtype=np.int32),
        local_preprocessing=np.asarray(True),
        server_preprocessing=np.asarray(False),
    )
    temporary.replace(path)
    print(
        f"{spec.client_id}: local raw stream complete, {len(array)} observations in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return path


def load_local_node(spec: ClientSpec) -> FederatedNode:
    path = _journal_path(spec)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = _journal_signature(spec)
    with np.load(path, allow_pickle=False) as payload:
        stored = bytes(payload["signature_utf8"]).decode("utf-8")
        if stored != expected:
            raise ValueError(f"stale local journal for {spec.client_id}")
        features = np.asarray(payload["features"], dtype=np.float32)
        tokens = bytes(payload["source_tokens_utf8"]).decode("utf-8").splitlines()
        indices = np.asarray(payload["observed_indices"])
        if not bool(payload["local_preprocessing"]) or bool(payload["server_preprocessing"]):
            raise ValueError("local/server preprocessing contract violation")
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError(f"non-chronological journal for {spec.client_id}")
    return FederatedNode(spec, features, tokens)


def _init_ae_weights(seed: int = SEED) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = []
    for input_dim, output_dim in zip(AE_DIMS[:-1], AE_DIMS[1:]):
        limit = math.sqrt(6.0 / (input_dim + output_dim))
        weights.append(rng.uniform(-limit, limit, size=(output_dim, input_dim)).astype(np.float32))
    return weights


def _equal_client_mean(updates: Sequence[np.ndarray]) -> np.ndarray:
    if not updates:
        raise ValueError("cannot aggregate zero clients")
    shape = np.asarray(updates[0]).shape
    if any(np.asarray(update).shape != shape for update in updates):
        raise ValueError("client update shape mismatch")
    return np.mean(np.stack(updates), axis=0, dtype=np.float64).astype(np.float32)


def train_equal_device_autoencoder(nodes: Sequence[FederatedNode]) -> tuple[DenseAE, list[dict[str, object]]]:
    if len(nodes) != 40 or len({node.spec.client_id for node in nodes}) != 40:
        raise ValueError("AE federation requires exactly 40 unique source devices")
    protocol = _read_json(PROTOCOL_PATH)
    config = dict(protocol["federated_training"])
    rounds = int(config["rounds"])
    epochs = int(config["local_epochs"])
    learning_rate = float(config["learning_rate"])
    base_weights = _init_ae_weights()
    prototype = DenseAE(AE_DIMS, weights=base_weights)
    global_flat = prototype.get_trainable()
    trace = []
    for round_index in range(rounds):
        updates = []
        losses = []
        for node in nodes:
            local_flat, loss = node.local_ae_update(
                global_flat,
                base_weights,
                epochs=epochs,
                learning_rate=learning_rate,
            )
            updates.append(local_flat)
            losses.append(loss)
        # Equal device vote: sample counts never enter this operation.
        global_flat = _equal_client_mean(updates)
        trace.append(
            {
                "round": round_index + 1,
                "clients": 40,
                "weight_per_client": 1.0 / 40.0,
                "mean_local_loss": float(np.mean(losses)),
                "min_local_loss": float(np.min(losses)),
                "max_local_loss": float(np.max(losses)),
            }
        )
        print(f"AE round {round_index+1:02d}/{rounds}: loss={np.mean(losses):.6f}", flush=True)
    model = DenseAE(AE_DIMS, weights=base_weights)
    model.set_trainable(global_flat, reset_optimizer=True)
    return model, trace


def train_equal_device_head(
    nodes: Sequence[FederatedNode],
    model: DenseAE,
    *,
    rounds: int = 160,
    learning_rate: float = 0.02,
    l2: float = 0.03,
) -> tuple[np.ndarray, float, list[dict[str, object]]]:
    if not nodes:
        raise ValueError("head federation needs clients")
    weights = np.zeros(HEAD_DIM, dtype=np.float32)
    bias = 0.0
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = v_b = 0.0
    trace = []
    beta1, beta2 = 0.9, 0.999
    for round_index in range(1, rounds + 1):
        gradients = [node.private_head_gradient(model, weights, bias, l2) for node in nodes]
        grad_w = np.mean(np.stack([row[0] for row in gradients]), axis=0)
        grad_b = float(np.mean([row[1] for row in gradients]))
        m_w = beta1 * m_w + (1-beta1) * grad_w
        v_w = beta2 * v_w + (1-beta2) * (grad_w * grad_w)
        m_b = beta1 * m_b + (1-beta1) * grad_b
        v_b = beta2 * v_b + (1-beta2) * (grad_b * grad_b)
        mwh = m_w / (1-beta1**round_index)
        vwh = v_w / (1-beta2**round_index)
        mbh = m_b / (1-beta1**round_index)
        vbh = v_b / (1-beta2**round_index)
        weights -= learning_rate * mwh / (np.sqrt(vwh) + 1e-8)
        bias -= learning_rate * mbh / (math.sqrt(vbh) + 1e-8)
        if round_index == 1 or round_index % 10 == 0 or round_index == rounds:
            row = {
                "round": round_index,
                "clients": len(nodes),
                "weight_per_client": 1.0 / len(nodes),
                "mean_local_loss": float(np.mean([item[2] for item in gradients])),
                "gradient_norm": float(np.linalg.norm(grad_w)),
            }
            trace.append(row)
            print(
                f"Head round {round_index:03d}/{rounds}: "
                f"loss={row['mean_local_loss']:.5f}",
                flush=True,
            )
    return weights, float(bias), trace


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(predictions, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tn = int(np.sum((y == 0) & (p == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def _persistent_latch(scores: np.ndarray, threshold: float, k: int, window: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    output = np.zeros(len(values), dtype=np.int8)
    hits: list[int] = []
    latched = False
    for index, value in enumerate(values):
        if index < COMMISSIONING:
            continue
        hits.append(int(value >= threshold))
        if len(hits) > window:
            hits.pop(0)
        if len(hits) == window and sum(hits) >= k:
            latched = True
        output[index] = int(latched)
    return output


def _event_class(spec: ClientSpec, predictions: np.ndarray) -> tuple[str, int | None]:
    alarms = np.flatnonzero(np.asarray(predictions)[COMMISSIONING:] > 0)
    first = int(alarms[0] + COMMISSIONING) if len(alarms) else None
    if spec.task == "fixed_fault_transfer":
        return ("fixed_fault_detected" if first is not None else "fixed_fault_missed", first)
    if spec.status != "failed":
        return ("censored_clean" if first is None else "censored_false_alarm", first)
    n = len(predictions)
    premature_end = int(math.ceil(0.60 * n))
    terminal_start = int(math.ceil(0.90 * n))
    if first is None:
        return "missed", None
    if first < premature_end:
        return "premature", first
    if first < terminal_start:
        return "valid_early", first
    return "late_terminal", first


def _evaluate_score_map(
    specs: Sequence[ClientSpec],
    score_map: dict[str, np.ndarray],
    *,
    threshold: float,
    k: int,
    window: int,
) -> dict[str, object]:
    labels_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    event_counts: dict[str, int] = {}
    rows = []
    for spec in specs:
        scores = np.asarray(score_map[spec.client_id], dtype=np.float32)
        node = load_local_node(spec)
        labels = node.private_labels()
        predictions = _persistent_latch(scores, threshold, k, window)
        use = np.arange(len(scores)) >= COMMISSIONING
        labels_all.append(labels[use])
        predictions_all.append(predictions[use])
        scores_all.append(scores[use])
        klass, first = _event_class(spec, predictions)
        event_counts[klass] = event_counts.get(klass, 0) + 1
        rows.append({"client_id": spec.client_id, "event_class": klass, "first_alarm": first})
    labels = np.concatenate(labels_all)
    predictions = np.concatenate(predictions_all)
    scores = np.concatenate(scores_all)
    metrics = _confusion(labels, predictions)
    metrics["auprc"] = float(average_precision_score(labels, scores)) if np.any(labels) else 0.0
    run_failures = sum(spec.task == "run_to_failure" and spec.status == "failed" for spec in specs)
    valid = event_counts.get("valid_early", 0)
    premature = event_counts.get("premature", 0)
    late = event_counts.get("late_terminal", 0)
    missed = event_counts.get("missed", 0)
    metrics.update(
        {
            "threshold": threshold,
            "persistence_k": k,
            "persistence_window": window,
            "event_counts": event_counts,
            "run_failure_devices": run_failures,
            "valid_early_rate": valid / run_failures if run_failures else 0.0,
            "premature_rate": premature / run_failures if run_failures else 0.0,
            "terminal_recall": (valid + premature + late) / run_failures if run_failures else 0.0,
            "late_rate": late / run_failures if run_failures else 0.0,
            "missed_rate": missed / run_failures if run_failures else 0.0,
            "rows": rows,
        }
    )
    return metrics


def _fold_index(spec: ClientSpec) -> int:
    digest = hashlib.sha256(("v12-fold::" + spec.client_id).encode()).digest()
    return int.from_bytes(digest[:4], "little") % 5


def source_oof_probabilities(
    nodes: Sequence[FederatedNode], model: DenseAE
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    probabilities: dict[str, np.ndarray] = {}
    trace = []
    for fold in range(5):
        training = [node for node in nodes if _fold_index(node.spec) != fold]
        validation = [node for node in nodes if _fold_index(node.spec) == fold]
        if not training or not validation:
            raise ValueError(f"empty source fold {fold}")
        print(f"Source OOF fold {fold+1}/5: {len(training)} train, {len(validation)} validation", flush=True)
        weights, bias, fold_trace = train_equal_device_head(
            training, model, rounds=90, learning_rate=0.025, l2=0.03
        )
        for node in validation:
            probabilities[node.spec.client_id] = node.private_head_probability(model, weights, bias)
        trace.append(
            {
                "fold": fold,
                "training_clients": [node.spec.client_id for node in training],
                "validation_clients": [node.spec.client_id for node in validation],
                "last_trace": fold_trace[-1],
            }
        )
    if set(probabilities) != {node.spec.client_id for node in nodes}:
        raise RuntimeError("OOF prediction coverage failure")
    return probabilities, trace


def select_source_policy(
    specs: Sequence[ClientSpec], probabilities: dict[str, np.ndarray]
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidates = []
    for threshold in np.linspace(0.30, 0.90, 25):
        for k, window in ((1, 1), (2, 3), (3, 5), (5, 8), (8, 10)):
            result = _evaluate_score_map(
                specs, probabilities, threshold=float(threshold), k=k, window=window
            )
            objective = (
                float(result["f1"])
                + 0.45 * float(result["valid_early_rate"])
                - 0.80 * float(result["premature_rate"])
                - 0.25 * float(result["late_rate"])
                - 0.40 * float(result["missed_rate"])
            )
            candidates.append(
                {
                    "threshold": float(threshold),
                    "k": k,
                    "window": window,
                    "objective": objective,
                    "f1": float(result["f1"]),
                    "precision": float(result["precision"]),
                    "recall": float(result["recall"]),
                    "valid_early_rate": float(result["valid_early_rate"]),
                    "premature_rate": float(result["premature_rate"]),
                    "terminal_recall": float(result["terminal_recall"]),
                }
            )
    candidates.sort(
        key=lambda row: (
            row["objective"], row["f1"], row["valid_early_rate"],
            -row["premature_rate"], row["threshold"]
        ),
        reverse=True,
    )
    best = candidates[0]
    policy = {
        "selected_on": "five-fold source-device out-of-fold probabilities only",
        "heldout_labels_used": False,
        "threshold": best["threshold"],
        "persistence_k": best["k"],
        "persistence_window": best["window"],
        "source_oof_metrics": best,
    }
    return policy, candidates


def _save_checkpoint(
    model: DenseAE, head_weights: np.ndarray, head_bias: float
) -> Path:
    path = OUT / "v12_source_model.npz"
    temporary = path.with_suffix(".tmp.npz")
    payload: dict[str, np.ndarray] = {
        f"ae_W{index}": np.asarray(weight, dtype=np.float32)
        for index, weight in enumerate(model.W)
    }
    payload["head_weights"] = np.asarray(head_weights, dtype=np.float32)
    payload["head_bias"] = np.asarray([head_bias], dtype=np.float32)
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    return path


def _load_checkpoint() -> tuple[DenseAE, np.ndarray, float]:
    path = OUT / "v12_source_model.npz"
    with np.load(path, allow_pickle=False) as payload:
        weights = [np.asarray(payload[f"ae_W{index}"], dtype=np.float32) for index in range(len(AE_DIMS)-1)]
        head = np.asarray(payload["head_weights"], dtype=np.float32)
        bias = float(payload["head_bias"][0])
    return DenseAE(AE_DIMS, weights=weights), head, bias


def _build_worker(args: tuple[ClientSpec, bool]) -> str:
    spec, rebuild = args
    return str(build_local_history(spec, rebuild=rebuild))


def _build_histories(specs: Sequence[ClientSpec], workers: int, rebuild: bool) -> None:
    if workers <= 1:
        for spec in specs:
            build_local_history(spec, rebuild=rebuild)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_build_worker, (spec, rebuild)): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            future.result()
            print(f"Completed local journal: {spec.client_id}", flush=True)


def train_source_experiment(*, workers: int, rebuild: bool) -> dict[str, object]:
    specs = [spec for spec in _all_client_specs() if spec.role == "source_train"]
    if TARGET_JOURNALS.exists() and any(TARGET_JOURNALS.glob("*.npz")):
        raise RuntimeError(
            "held-out journals already exist; move/remove the V12 output directory before claiming a fresh locked test"
        )
    _build_histories(specs, workers, rebuild)
    nodes = [load_local_node(spec) for spec in specs]
    model, ae_trace = train_equal_device_autoencoder(nodes)
    oof_probability, oof_trace = source_oof_probabilities(nodes, model)
    policy, candidates = select_source_policy(specs, oof_probability)
    final_head, final_bias, head_trace = train_equal_device_head(
        nodes, model, rounds=180, learning_rate=0.025, l2=0.03
    )
    checkpoint = _save_checkpoint(model, final_head, final_bias)
    _write_json(OUT / "source_selected_policy.json", policy)
    _write_csv(OUT / "source_policy_candidates.csv", candidates)
    _write_json(OUT / "source_ae_trace.json", ae_trace)
    _write_json(OUT / "source_head_trace.json", head_trace)
    _write_json(OUT / "source_oof_trace.json", oof_trace)
    oof_eval = _evaluate_score_map(
        specs,
        oof_probability,
        threshold=float(policy["threshold"]),
        k=int(policy["persistence_k"]),
        window=int(policy["persistence_window"]),
    )
    oof_rows = oof_eval.pop("rows")
    _write_csv(OUT / "source_oof_per_device.csv", oof_rows)
    _write_json(OUT / "source_oof_summary.json", oof_eval)
    lock = {
        "status": "source_model_and_policy_locked_before_any_v12_heldout_raw_stream_is_opened",
        "retrospective": True,
        "protocol_sha256": _protocol_digest(),
        "source_code_sha256": _sha256(Path(__file__)),
        "checkpoint_sha256": _sha256(checkpoint),
        "policy_sha256": _sha256(OUT / "source_selected_policy.json"),
        "source_clients": [spec.client_id for spec in specs],
        "source_journals": {spec.client_id: _journal_signature(spec) for spec in specs},
        "heldout_journals_present_at_lock": False,
        "aggregation_weight_per_device": 1.0 / 40.0,
        "server_received_raw": False,
        "server_computed_features": False,
    }
    _write_json(OUT / "v12_source_lock.json", lock)
    payload = {"policy": policy, "source_oof": oof_eval, "lock": lock}
    _write_json(OUT / "source_training_summary.json", payload)
    return payload


class OnlineScorer:
    def __init__(
        self,
        commissioning_features: np.ndarray,
        model: DenseAE,
        head_weights: np.ndarray,
        head_bias: float,
    ):
        self.model = model
        self.head_weights = head_weights
        self.head_bias = head_bias
        self.location, self.scale = _fit_local_normalizer(commissioning_features)
        self.z_history = list(_normalize(commissioning_features, self.location, self.scale))
        commissioning_z = np.asarray(self.z_history, dtype=np.float32)
        self.pca_center = np.mean(commissioning_z, axis=0)
        _, _, vt = np.linalg.svd(commissioning_z - self.pca_center, full_matrices=False)
        self.pca_basis = vt[: min(8, len(vt))].astype(np.float32)
        self.ae_ref = np.log1p(model.losses(commissioning_z))
        centered = commissioning_z - self.pca_center
        reconstruction = (centered @ self.pca_basis.T) @ self.pca_basis
        self.pca_ref = np.log1p(np.mean((centered - reconstruction) ** 2, axis=1))
        self.dist_ref = np.mean(np.abs(commissioning_z), axis=1)

    @staticmethod
    def _one_robust(value: float, reference: np.ndarray) -> float:
        location = float(np.median(reference))
        scale = 1.4826 * float(np.median(np.abs(reference - location)))
        if scale < 1e-6:
            scale = max(float(np.std(reference, ddof=1)), 1e-6)
        return float(np.clip((value - location) / scale, -12.0, 30.0))

    def score_current(self, feature: np.ndarray) -> tuple[float, float]:
        z = _normalize(np.asarray(feature)[None, :], self.location, self.scale)[0]
        ae = float(self.model.losses(z[None, :])[0])
        centered = z - self.pca_center
        reconstructed = (centered @ self.pca_basis.T) @ self.pca_basis
        pca = float(np.mean((centered - reconstructed) ** 2))
        distance = float(np.mean(np.abs(z)))
        components = np.asarray(
            [
                self._one_robust(math.log1p(ae), self.ae_ref),
                self._one_robust(math.log1p(pca), self.pca_ref),
                self._one_robust(distance, self.dist_ref),
            ],
            dtype=np.float32,
        )
        delta = z - np.mean(np.asarray(self.z_history[-5:]), axis=0)
        descriptor = np.concatenate([z, delta, components])
        logit = float(np.clip(descriptor @ self.head_weights + self.head_bias, -30.0, 30.0))
        probability = 1.0 / (1.0 + math.exp(-logit))
        self.z_history.append(z)
        return float(np.max(components)), probability


def _lock_verified() -> dict[str, object]:
    lock_path = OUT / "v12_source_lock.json"
    lock = _read_json(lock_path)
    if lock.get("status") != "source_model_and_policy_locked_before_any_v12_heldout_raw_stream_is_opened":
        raise ValueError("invalid source lock")
    if lock["protocol_sha256"] != _protocol_digest():
        raise ValueError("protocol changed after source lock")
    if lock["source_code_sha256"] != _sha256(Path(__file__)):
        raise ValueError("V12 source code changed after source lock")
    if lock["checkpoint_sha256"] != _sha256(OUT / "v12_source_model.npz"):
        raise ValueError("checkpoint changed after source lock")
    if lock["policy_sha256"] != _sha256(OUT / "source_selected_policy.json"):
        raise ValueError("policy changed after source lock")
    return lock


class _OnlineLatch:
    def __init__(self, threshold: float, k: int, window: int):
        self.threshold = float(threshold)
        self.k = int(k)
        self.window = int(window)
        self.hits: list[int] = []
        self.latched = False

    def step(self, score: float) -> int:
        self.hits.append(int(score >= self.threshold))
        if len(self.hits) > self.window:
            self.hits.pop(0)
        if len(self.hits) == self.window and sum(self.hits) >= self.k:
            self.latched = True
        return int(self.latched)


def score_heldout_node_once(
    spec: ClientSpec,
    model: DenseAE,
    head_weights: np.ndarray,
    head_bias: float,
    policy: dict[str, object],
    *,
    rebuild: bool,
) -> dict[str, object]:
    path = _journal_path(spec)
    if path.exists() and not rebuild:
        raise RuntimeError(
            f"{path} already exists; --rebuild is required to demonstrate a new raw one-pass replay"
        )
    signature = _journal_signature(spec)
    features: list[np.ndarray] = []
    tokens: list[str] = []
    supervised_scores: list[float] = []
    label_free_scores: list[float] = []
    supervised_predictions: list[int] = []
    label_free_predictions: list[int] = []
    selected_latch = _OnlineLatch(
        float(policy["threshold"]),
        int(policy["persistence_k"]),
        int(policy["persistence_window"]),
    )
    fixed_latch = _OnlineLatch(3.0, 3, 5)
    scorer: OnlineScorer | None = None
    expected = 0
    started = time.perf_counter()
    for acquisition in iter_raw_acquisitions(spec):
        if acquisition.index != expected:
            raise RuntimeError(f"future acquisition exposed to {spec.client_id}")
        feature = local_preprocess_current_acquisition(acquisition.channels)
        features.append(feature)
        tokens.append(acquisition.source_token)
        if expected + 1 == COMMISSIONING:
            scorer = OnlineScorer(np.asarray(features), model, head_weights, head_bias)
        if expected < COMMISSIONING:
            supervised_scores.append(0.0)
            label_free_scores.append(0.0)
            supervised_predictions.append(0)
            label_free_predictions.append(0)
        else:
            assert scorer is not None
            label_free, probability = scorer.score_current(feature)
            label_free_scores.append(label_free)
            supervised_scores.append(probability)
            label_free_predictions.append(fixed_latch.step(label_free))
            supervised_predictions.append(selected_latch.step(probability))
        expected += 1
        if expected == 1 or expected % 250 == 0:
            print(f"  TEST {spec.client_id}: causally scored {expected}", flush=True)
        del acquisition
    array = np.asarray(features, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        signature_utf8=np.frombuffer(signature.encode(), dtype=np.uint8),
        features=array,
        source_tokens_utf8=np.frombuffer("\n".join(tokens).encode(), dtype=np.uint8),
        observed_indices=np.arange(len(array), dtype=np.int32),
        local_preprocessing=np.asarray(True),
        server_preprocessing=np.asarray(False),
        score_generated_before_next_raw=np.asarray(True),
        supervised_score=np.asarray(supervised_scores, dtype=np.float32),
        label_free_score=np.asarray(label_free_scores, dtype=np.float32),
        supervised_prediction=np.asarray(supervised_predictions, dtype=np.int8),
        label_free_prediction=np.asarray(label_free_predictions, dtype=np.int8),
    )
    temporary.replace(path)
    print(
        f"TEST {spec.client_id}: {len(array)} raw acquisitions scored online in "
        f"{time.perf_counter()-started:.1f}s",
        flush=True,
    )
    return {
        "client_id": spec.client_id,
        "observations": len(array),
        "journal_sha256": _sha256(path),
        "signature": signature,
    }


def _scores_from_target_journals(
    specs: Sequence[ClientSpec], key: str
) -> dict[str, np.ndarray]:
    output = {}
    for spec in specs:
        with np.load(_journal_path(spec), allow_pickle=False) as payload:
            if not bool(payload["score_generated_before_next_raw"]):
                raise ValueError("target journal lacks online-scoring guarantee")
            output[spec.client_id] = np.asarray(payload[key], dtype=np.float32)
    return output


def _subset_evaluation(
    specs: Sequence[ClientSpec], score_map: dict[str, np.ndarray], policy: dict[str, object]
) -> dict[str, object]:
    return _evaluate_score_map(
        specs,
        score_map,
        threshold=float(policy["threshold"]),
        k=int(policy["persistence_k"]),
        window=int(policy["persistence_window"]),
    )


def _metrics_without_rows(payload: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    copied = dict(payload)
    rows = list(copied.pop("rows"))
    return copied, rows


def _report_markdown(summary: dict[str, object]) -> str:
    sup = dict(summary["supervised_source_trained"])
    lf = dict(summary["label_free_fixed"])
    genuine = dict(sup["genuine_run_to_failure_14"])
    all17 = dict(sup["all_requested_17"])
    hit = dict(sup["hit_fixed_fault_3"])
    lines = [
        "# V12 raw-online 40-client -> 17-client retrospective report",
        "",
        "## Claim boundary",
        "",
        "This is a **retrospective multi-dataset held-out-bearing experiment**, not untouched prospective confirmation. "
        "All V12 features were rebuilt from raw signals inside the simulated node.  A node observed one chronological "
        "acquisition at a time; the server received equal-weight model updates and never raw data or feature matrices.",
        "",
        "HIT is a fixed-fault benchmark, not a run-to-failure dataset. Its three held-out files are therefore reported "
        "as fault-state transfer and excluded from the genuine early-warning event rate.",
        "",
        "## Frozen split",
        "",
        "- 40 source clients: Ferrara 3, XJTU 12, PRONOSTIA 14, IMS 9, HIT healthy 2, LDM-UPB 0.",
        "- 17 held-out clients: Ferrara 3, XJTU 3, PRONOSTIA 3, IMS 3, HIT fault 3, LDM-UPB 2.",
        "- Each client had exactly equal server weight; source histories with more samples did not receive more weight.",
        "",
        "## Held-out results",
        "",
        "| Track | Evaluation | F1 | Precision | Recall | AUPRC | Valid early | Premature | Terminal recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Source-supervised FL | 14 genuine run-to-failure | {100*genuine['f1']:.2f}% | {100*genuine['precision']:.2f}% | {100*genuine['recall']:.2f}% | {100*genuine['auprc']:.2f}% | {100*genuine['valid_early_rate']:.2f}% | {100*genuine['premature_rate']:.2f}% | {100*genuine['terminal_recall']:.2f}% |",
        f"| Source-supervised FL | all requested 17 | {100*all17['f1']:.2f}% | {100*all17['precision']:.2f}% | {100*all17['recall']:.2f}% | {100*all17['auprc']:.2f}% | task-mixed | task-mixed | task-mixed |",
        f"| Source-supervised FL | 3 HIT fixed-fault | {100*hit['f1']:.2f}% | {100*hit['precision']:.2f}% | {100*hit['recall']:.2f}% | {100*hit['auprc']:.2f}% | n/a | n/a | n/a |",
        f"| Label-free AE/PCA/distance | 14 genuine run-to-failure | {100*lf['genuine_run_to_failure_14']['f1']:.2f}% | {100*lf['genuine_run_to_failure_14']['precision']:.2f}% | {100*lf['genuine_run_to_failure_14']['recall']:.2f}% | {100*lf['genuine_run_to_failure_14']['auprc']:.2f}% | {100*lf['genuine_run_to_failure_14']['valid_early_rate']:.2f}% | {100*lf['genuine_run_to_failure_14']['premature_rate']:.2f}% | {100*lf['genuine_run_to_failure_14']['terminal_recall']:.2f}% |",
        "",
        "## Interpretation",
        "",
        "The 90% F1 and perfect event-detection values in the frozen protocol are success goals, not assumptions. "
        "Any unmet goal is preserved in the tables rather than hidden or retuned on the held-out labels. Normalized-life "
        "warning zones are proxy labels and are not claimed to be verified physical degradation onset.",
        "",
    ]
    return "\n".join(lines)


def test_heldout_experiment(*, rebuild: bool) -> dict[str, object]:
    lock = _lock_verified()
    specs = [spec for spec in _all_client_specs() if spec.role == "heldout_test"]
    model, head_weights, head_bias = _load_checkpoint()
    policy = _read_json(OUT / "source_selected_policy.json")
    audit_rows = []
    for spec in specs:
        audit_rows.append(
            score_heldout_node_once(
                spec, model, head_weights, head_bias, policy, rebuild=rebuild
            )
        )
    supervised_scores = _scores_from_target_journals(specs, "supervised_score")
    label_free_scores = _scores_from_target_journals(specs, "label_free_score")
    genuine = [spec for spec in specs if spec.task == "run_to_failure"]
    hit = [spec for spec in specs if spec.task == "fixed_fault_transfer"]
    supervised_policy = {
        "threshold": policy["threshold"],
        "persistence_k": policy["persistence_k"],
        "persistence_window": policy["persistence_window"],
    }
    label_free_policy = {"threshold": 3.0, "persistence_k": 3, "persistence_window": 5}
    groups = {
        "genuine_run_to_failure_14": genuine,
        "hit_fixed_fault_3": hit,
        "all_requested_17": specs,
    }
    supervised_result = {}
    label_free_result = {}
    per_device_rows = []
    for name, group in groups.items():
        value, rows = _metrics_without_rows(_subset_evaluation(group, supervised_scores, supervised_policy))
        supervised_result[name] = value
        for row in rows:
            row.update({"track": "source_supervised", "evaluation_group": name})
            per_device_rows.append(row)
        value, rows = _metrics_without_rows(_subset_evaluation(group, label_free_scores, label_free_policy))
        label_free_result[name] = value
        for row in rows:
            row.update({"track": "label_free", "evaluation_group": name})
            per_device_rows.append(row)
    by_dataset = {}
    for dataset in sorted({spec.dataset for spec in specs}):
        group = [spec for spec in specs if spec.dataset == dataset]
        metrics, _ = _metrics_without_rows(_subset_evaluation(group, supervised_scores, supervised_policy))
        by_dataset[dataset] = metrics
    summary = {
        "provenance": "retrospective multi-dataset held-out-bearing experiment; not untouched prospective confirmation",
        "source_lock_verified": lock,
        "supervised_source_trained": supervised_result,
        "label_free_fixed": label_free_result,
        "supervised_by_dataset": by_dataset,
        "source_selected_policy": policy,
        "heldout_raw_online_audit": audit_rows,
        "heldout_labels_used_for_model_or_policy": False,
        "server_received_raw": False,
        "server_preprocessed_features": False,
    }
    _write_json(OUT / "heldout_summary.json", summary)
    _write_csv(OUT / "heldout_per_device.csv", per_device_rows)
    (OUT / "V12_REPORT.md").write_text(_report_markdown(summary), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "train", "test", "all"), default="all")
    parser.add_argument("--workers", type=int, default=max(1, min(3, os.cpu_count() or 1)))
    parser.add_argument("--rebuild", action="store_true", help="rebuild V12 local journals directly from raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase in {"audit", "all"}:
        print(json.dumps(audit_raw_split(), indent=2), flush=True)
    if args.phase in {"train", "all"}:
        print(json.dumps(train_source_experiment(workers=args.workers, rebuild=args.rebuild), indent=2), flush=True)
    if args.phase in {"test", "all"}:
        result = test_heldout_experiment(rebuild=args.rebuild)
        genuine = result["supervised_source_trained"]["genuine_run_to_failure_14"]
        print(
            f"V12 held-out genuine RTF: F1={100*genuine['f1']:.2f}% "
            f"valid={100*genuine['valid_early_rate']:.2f}% "
            f"premature={100*genuine['premature_rate']:.2f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
