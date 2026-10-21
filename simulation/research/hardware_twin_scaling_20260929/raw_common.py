"""Shared raw-waveform access layer for the RAW SD-card hardware experiment
(LilyGO T3 v1.6.1, computing features on-device from raw vibration signals).

This module does not reimplement any dataset's raw-parsing logic. It
dispatches, per machine, to the exact functions that already produced the
committed feature cache (results/healthy_only_suite_20260907/cache), found
by reading each cached machine's `source_hashes` in
results/healthy_only_suite_20260907/cache/manifest.json and classifying
those paths by which study produced them:

  - raw_online_federated_v19/feature_streams/{lane}/{dataset}__{unit}.npz
        -> raw_online_federated_v12.py (ferrara, xjtu, pronostia, ims, hit):
           iter_raw_acquisitions() + local_preprocess_current_acquisition(),
           NO extra decimation before the 100-D transform.
  - raw_online_federated_v30_prereg1/target_streams/{dataset}/{unit}.npz
        -> raw_online_federated_v30_data.py (dlr Tests 1/4/6/7/8, kaist,
           unsw): file lists from _dlr_files/_kaist_files/_unsw_files,
           signals decimated to <=65536 samples (_decimate) THEN
           local_preprocess_current_acquisition().
  - raw_online_federated_v44_frozen4/target_streams/{dataset}__{unit}.npz
        -> raw_online_federated_v41_data.py (dlr Tests 2/5, the V41R1
           subset): file list from test_files(), same decimate-then-feature
           rule as V30 (independently re-derived constant, same value).
  - raw_online_federated_v63_frozen1/paderborn_predictions/paderborn__{
    bearing}.npz
        -> raw_online_federated_v63.py (paderborn K003-K006): one archive
           entry (MAT file) yields TWO acquisitions (first half / second
           half of its vibration_1 channel), no decimation before the
           100-D transform.

Every raw file is opened through the ORIGINAL module's own file-discovery
function (e.g. `raw_online_federated_v30_data._dlr_files`) so the file set
this module reads is provably the same file set that produced the cached
features -- Gate R (raw_gate_r.py) is what proves the *values* reproduce.

The 100-D feature math itself always goes through raw_reference_features.py
(the kaggle_notebook.py cell-2 transcription), never through the v12/v30/v41
modules' own feature calls, so a change to those modules cannot silently
change what this study checks against.
"""
from __future__ import annotations

import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

HERE = Path(__file__).resolve().parent            # research/hardware_twin_scaling_20260929
ROOT = HERE.parents[1]                             # full_pipeline_simulation

for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import raw_online_federated_v12 as v12              # noqa: E402
import raw_online_federated_v30_data as v30data      # noqa: E402
import raw_online_federated_v41_data as v41data      # noqa: E402
import raw_online_federated_v63 as v63               # noqa: E402
import machine_order as mo                           # noqa: E402
import raw_reference_features as rf                  # noqa: E402

CACHE_DIR = ROOT / "results" / "healthy_only_suite_20260907" / "cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

V19_ROOT = ROOT / "results" / "raw_online_federated_v19" / "feature_streams"
V30_ROOT = ROOT / "results" / "raw_online_federated_v30_prereg1" / "target_streams"
V44_ROOT = ROOT / "results" / "raw_online_federated_v44_frozen4" / "target_streams"
V63_PADERBORN_ROOT = ROOT / "results" / "raw_online_federated_v63_frozen1" / "paderborn_predictions"

DECIMATE_LIMIT = 65_536  # v30data.MAX_RAW_SAMPLES == v41data.MAX_RAW_SAMPLES
assert v30data.MAX_RAW_SAMPLES == DECIMATE_LIMIT == v41data.MAX_RAW_SAMPLES


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def devices_by_id(manifest: dict | None = None) -> dict[str, dict]:
    manifest = manifest if manifest is not None else load_manifest()
    return {d["machine_id"]: d for d in manifest["devices"]}


# ---------------------------------------------------------------------------
# Classify a cached machine's source_hashes path -> which raw loader to use.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StreamRef:
    kind: str        # 'v19' | 'v30' | 'v41r1' | 'paderborn'
    dataset: str
    unit: str
    lane: str | None = None  # only for 'v19'

    @property
    def stream_label(self) -> str:
        return f"{self.kind}:{self.dataset}::{self.unit}"


def classify_source_path(path: str | Path) -> StreamRef:
    path = Path(path)
    parts = path.parts
    if "raw_online_federated_v19" in parts:
        lane = path.parent.name
        dataset, _, unit = path.stem.partition("__")
        return StreamRef("v19", dataset, unit, lane)
    if "raw_online_federated_v30_prereg1" in parts:
        dataset = path.parent.name
        unit = path.stem
        return StreamRef("v30", dataset, unit)
    if "raw_online_federated_v44_frozen4" in parts:
        dataset, _, unit = path.stem.partition("__")
        return StreamRef("v41r1", dataset, unit)
    if "raw_online_federated_v63_frozen1" in parts and "paderborn_predictions" in parts:
        dataset, _, unit = path.stem.partition("__")
        return StreamRef("paderborn", dataset, unit)
    raise ValueError(f"unrecognised source feature path (no known raw loader): {path}")


def machine_streams(machine_id: str, manifest: dict | None = None) -> list[StreamRef]:
    """The 1 or 2 StreamRef's this cached machine's `x` was built from,
    in the same order as manifest source_hashes (dict order == insertion
    order == the order data_cache.py wrote them)."""
    manifest = manifest if manifest is not None else load_manifest()
    dev = devices_by_id(manifest)[machine_id]
    return [classify_source_path(p) for p in dev["source_hashes"]]


def source_feature_path(ref: StreamRef) -> Path:
    if ref.kind == "v19":
        return V19_ROOT / ref.lane / f"{ref.dataset}__{ref.unit}.npz"
    if ref.kind == "v30":
        return V30_ROOT / ref.dataset / f"{ref.unit}.npz"
    if ref.kind == "v41r1":
        return V44_ROOT / f"{ref.dataset}__{ref.unit}.npz"
    if ref.kind == "paderborn":
        return V63_PADERBORN_ROOT / f"{ref.dataset}__{ref.unit}.npz"
    raise ValueError(ref.kind)


def load_source_features(ref: StreamRef) -> np.ndarray:
    """The precomputed 100-D features/row this stream's cache entry was
    built from (float32, (n_acquisitions, 100))."""
    path = source_feature_path(ref)
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["features"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-kind raw file discovery (reuses each study's own file-listing
# functions; never guesses a new glob).
# ---------------------------------------------------------------------------
def _v19_spec(ref: StreamRef) -> v12.ClientSpec:
    for spec in v12._all_client_specs():
        if spec.dataset == ref.dataset and spec.unit == ref.unit:
            return spec
    raise KeyError(f"no v12 ClientSpec for {ref.stream_label}")


def _v30_device(ref: StreamRef) -> v30data.TargetDevice:
    lane = "staged_state" if ref.dataset == "ottawa" else "timeline"
    return v30data.TargetDevice(ref.dataset, ref.unit, lane)


def _v41r1_device(ref: StreamRef) -> v41data.TargetDevice:
    import re
    m = re.fullmatch(r"Test_(\d+)_B([12])", ref.unit)
    if not m:
        raise ValueError(ref.unit)
    return v41data.TargetDevice(int(m.group(1)), int(m.group(2)))


def raw_file_paths(ref: StreamRef) -> list[Path]:
    """Every raw source file this stream is built from, in read order."""
    if ref.kind == "v19":
        spec = _v19_spec(ref)
        names, _total = v12._raw_listing(spec)
        return [v12.DATA_ROOT / n for n in names]
    if ref.kind == "v30":
        device = _v30_device(ref)
        if ref.dataset == "unsw":
            return v30data._unsw_files(device)
        if ref.dataset == "kaist":
            return v30data._kaist_files()
        if ref.dataset == "dlr":
            return v30data._dlr_files(device)
        raise ValueError(ref.dataset)
    if ref.kind == "v41r1":
        return v41data.test_files(_v41r1_device(ref))
    if ref.kind == "paderborn":
        archive_path = v63.PADERBORN
        with zipfile.ZipFile(archive_path) as archive:
            entries = v63._paderborn_entries(archive, ref.unit)
        # Represented as pseudo-paths "archive.zip!entry" (never extracted).
        return [Path(f"{archive_path}!{name}") for name in entries]
    raise ValueError(ref.kind)


def raw_file_dtype(ref: StreamRef, path: Path) -> str:
    """Best-effort *source* dtype label for one raw file (header-only
    inspection where the format allows it; never loads full array data)."""
    try:
        if ref.kind in ("v30", "v41r1") and ref.dataset == "dlr":
            variables = whosmat(str(path))
            candidates = [(nm, dt) for nm, shape, dt in variables if len(shape) == 2 and shape[1] >= 2]
            return candidates[0][1] if candidates else "unknown(mat)"
        if ref.kind == "v19" and ref.dataset == "ferrara":
            variables = whosmat(str(path))
            for nm, _shape, dt in variables:
                if nm == "y":
                    return dt
            return variables[0][2] if variables else "unknown(mat)"
        if ref.kind == "v19" and ref.dataset == "hit":
            with open(path, "rb") as fh:
                version = np.lib.format.read_magic(fh)
                _shape, _fortran, dtype = np.lib.format._read_array_header(fh, version)
            return str(dtype)
        if ref.kind == "v30" and ref.dataset == "unsw":
            variables = whosmat(str(path))
            for nm, _shape, dt in variables:
                if nm == "accH":
                    return dt
            return "unknown(mat)"
        if ref.kind == "v30" and ref.dataset == "kaist":
            return "text/ascii (parsed float32)"
        if ref.kind == "v19" and ref.dataset == "xjtu":
            return "text/ascii (parsed float32)"
        if ref.kind == "v19" and ref.dataset == "pronostia":
            return "text/ascii (parsed float64 by loader)"
        if ref.kind == "v19" and ref.dataset == "ims":
            return "text/ascii (parsed float32 by loader)"
        if ref.kind == "paderborn":
            return "double (MATLAB struct field Y.vibration_1.Data, nested)"
    except Exception as exc:  # pragma: no cover - best effort only
        return f"unknown ({exc.__class__.__name__})"
    return "unknown"


# ---------------------------------------------------------------------------
# Per-kind channel-level acquisition iteration (index, channels[C,N], token).
# Channels are returned exactly as the producing study's loader built them,
# BEFORE any decimation, so callers can apply v30/v41's decimate-then-feature
# rule explicitly (decimate_if_v30_style below) or skip it (v19/paderborn).
# ---------------------------------------------------------------------------
def decimates(ref: StreamRef) -> bool:
    """Whether this stream's producing study decimated to <=65536 samples
    before the 100-D transform (protocol: v30/v41r1 do; v19/paderborn
    don't)."""
    return ref.kind in ("v30", "v41r1") and ref.dataset in ("dlr", "kaist", "unsw")


def decimate_v30_style(channels: np.ndarray, limit: int = DECIMATE_LIMIT) -> np.ndarray:
    """v30data._decimate / v41data._feature's stride rule: drop columns with
    any non-finite value (current-acquisition-only), then stride down to
    <= limit samples if longer."""
    values = np.asarray(channels, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    values = values[:, np.all(np.isfinite(values), axis=0)]
    if values.shape[1] < 512:
        raise ValueError(f"invalid raw acquisition {values.shape}")
    if values.shape[1] > int(limit):
        stride = int(math.ceil(values.shape[1] / int(limit)))
        values = values[:, ::stride][:, : int(limit)]
    return np.ascontiguousarray(values, dtype=np.float32)


def iter_stream_channels(ref: StreamRef) -> Iterator[tuple[int, np.ndarray, str]]:
    """Yield (index, channels[C,N] float32, token) in causal order, for the
    RAW (not-yet-decimated) signal, exactly as the producing study read it."""
    if ref.kind == "v19":
        spec = _v19_spec(ref)
        for acq in v12.iter_raw_acquisitions(spec):
            yield acq.index, np.asarray(acq.channels, dtype=np.float32), acq.source_token
        return

    if ref.kind == "v30":
        device = _v30_device(ref)
        if ref.dataset == "unsw":
            for index, path in enumerate(v30data._unsw_files(device)):
                payload = loadmat(path, variable_names=["accH", "accV"], squeeze_me=True)
                channels = np.vstack([payload["accH"], payload["accV"]]).astype(np.float32)
                yield index, channels, str(path.relative_to(v30data.RAW_ROOT))
                del payload, channels
            return
        if ref.dataset == "kaist":
            for index, path in enumerate(v30data._kaist_files()):
                frame = pd.read_csv(path, header=None, dtype=np.float32)
                channels = frame.iloc[:, :2].to_numpy(dtype=np.float32).T
                yield index, channels, str(path.relative_to(v30data.RAW_ROOT))
                del frame, channels
            return
        if ref.dataset == "dlr":
            index = 0
            for path in v30data._dlr_files(device):
                variables = whosmat(path)
                candidates = [name for name, shape, _ in variables if len(shape) == 2 and shape[1] >= 2]
                if len(candidates) != 1:
                    raise ValueError(f"unique DLR sensor matrix not found: {path}")
                matrix = np.asarray(loadmat(path, variable_names=candidates, squeeze_me=False)[candidates[0]])
                for segment in v30data.split_dlr_acquisitions(matrix):
                    signal = matrix[segment, 1].astype(np.float32)
                    yield index, signal[None, :], f"{path.relative_to(v30data.RAW_ROOT)}#segment={index}"
                    index += 1
                del matrix
            return
        raise ValueError(ref.dataset)

    if ref.kind == "v41r1":
        device = _v41r1_device(ref)
        index = 0
        for path in v41data.test_files(device):
            variables = whosmat(path)
            candidates = [name for name, shape, _ in variables if len(shape) == 2 and shape[1] >= 2]
            if len(candidates) != 1:
                raise ValueError(f"unique DLR sensor matrix not found: {path}")
            matrix = np.asarray(loadmat(path, variable_names=candidates)[candidates[0]])
            for segment in v41data.split_acquisitions(matrix):
                signal = matrix[segment, 1].astype(np.float32)
                yield index, signal[None, :], f"{path.relative_to(v41data.RAW_ROOT)}#segment={index}"
                index += 1
            del matrix
        return

    if ref.kind == "paderborn":
        with zipfile.ZipFile(v63.PADERBORN) as archive:
            entries = v63._paderborn_entries(archive, ref.unit)
            index = 0
            for entry in entries:
                signal = v63._paderborn_signal(archive.read(entry))
                midpoint = len(signal) // 2
                for half in (signal[:midpoint], signal[midpoint:]):
                    yield index, half[None, :].astype(np.float32), f"{entry}#half={index % 2}"
                    index += 1
        return

    raise ValueError(ref.kind)


def stream_features(ref: StreamRef) -> tuple[np.ndarray, list[str]]:
    """Recompute the 100-D feature for every acquisition of one stream from
    raw, via raw_reference_features (the kaggle_notebook cell-2 source of
    truth), applying the producing study's decimate-or-not rule. Returns
    (features[n,100] float64, tokens)."""
    rows: list[np.ndarray] = []
    tokens: list[str] = []
    decim = decimates(ref)
    for index, channels, token in iter_stream_channels(ref):
        if index != len(rows):
            raise RuntimeError(f"non-causal acquisition order for {ref.stream_label}: {index} != {len(rows)}")
        x = decimate_v30_style(channels) if decim else channels
        feat = rf.local_preprocess_current_acquisition(x)
        rows.append(np.asarray(feat, dtype=np.float64))
        tokens.append(token)
    return np.asarray(rows, dtype=np.float64), tokens


# ---------------------------------------------------------------------------
# Machine roster (positions 0-53) with per-machine stream classification.
# ---------------------------------------------------------------------------
def hardware_machines(levels_hw=(4, 8, 12, 16, 20, 24, 28)):
    """(order, assignment): order = the 54-machine list (machine_order.py),
    assignment = machine_id -> board. Positions 0-27 are hardware machines
    (amendment_hardware_scope.json)."""
    info = mo.build_all()
    order = info["order"]
    assignment = mo.build_board_assignment(order)
    return order, assignment


HARDWARE_POSITIONS = 28  # positions 0..27 (amendment_hardware_scope.json)
