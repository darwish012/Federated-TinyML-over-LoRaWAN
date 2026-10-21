"""Amendment-2 raw representation: int16 (and, for the step-1 comparison,
int8) quantisation of the fed raw samples, and the one-raw-pass-per-machine
computation of the float32 22-D raw features / standardised x from them.

Quantisation (amendment2_int16_raw_order.json "sd_content", generalised to
int8 for step 1 only):
    per (acquisition, channel):
        max_abs = max |x|               (x = the FED float32 samples, i.e.
                                         after the producing study's own
                                         decimation, exactly what the
                                         extractor computes on)
        scale   = float32(max_abs) / float32(QMAX)        float32 division
                  (1.0 if max_abs == 0)
        q       = roundf(x / scale)      float32 division, C roundf
                  (half away from zero), then clip to [-QMAX, QMAX]
        QMAX = 32767 (int16)  |  127 (int8)
    on the device: x_hat = float32(q) * scale   (float32 multiply; q is an
    exact float32 so this is one correctly-rounded product)

All arithmetic is float32 so a C firmware with `float` reproduces it.

This module also holds the per-machine CHECKPOINTED pass. For every machine
one raw pass is made (the slow part: parsing the source files). Its result
is written atomically to
    results/hardware_twin_scaling_20260929/raw_v2/cache/<machine>.npz
and, for hardware machines, the int16 sample data section of each stream
(the on-card body, see sd_format_v2.py) is written next to it under
    .../raw_v2/cache/bodies/<machine>__s<k>.body
The npz is written LAST, so a machine is "done" iff its npz exists; a
killed run loses at most the machine it was on.
"""
from __future__ import annotations

import json
import os
import sys
import time
import zlib
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "healthy_only_suite_20260907"))
sys.path.insert(0, str(HERE.parent / "fl_algorithm_comparison_20260929"))
sys.path.insert(0, str(HERE.parent / "healthy_only_causal_20260908"))
sys.path.insert(0, str(ROOT))

F32 = np.float32
QMAX = {"int16": 32767, "int8": 127}
QDTYPE = {"int16": np.dtype("<i2"), "int8": np.dtype("i1")}

RESULTS = ROOT / "results" / "hardware_twin_scaling_20260929"
RAW_V2 = RESULTS / "raw_v2"
CACHE_V2 = RAW_V2 / "cache"
BODY_DIR = CACHE_V2 / "bodies"
COMMITTED_CACHE = ROOT / "results" / "healthy_only_suite_20260907" / "cache"


# ---------------------------------------------------------------------------
# Quantiser (pure numpy, no dataset code -- unit tested in test_v2.py).
# ---------------------------------------------------------------------------
def roundf(y: np.ndarray) -> np.ndarray:
    """C roundf: nearest integer, ties away from zero, EXACT in float32
    (no x+0.5 trick, which mis-rounds e.g. 0.49999997f)."""
    y = np.asarray(y, dtype=F32)
    t = np.trunc(y)
    frac = np.abs(y - t)                      # exact in float32
    return (t + np.where(frac >= F32(0.5), np.sign(y), F32(0.0))).astype(F32)


def quantise_channel(x: np.ndarray, rep: str = "int16"):
    """x: 1-D float32 samples of one (acquisition, channel).
    Returns (q, scale): q = int16/int8 array, scale = np.float32."""
    qmax = QMAX[rep]
    x = np.asarray(x, dtype=F32).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite samples cannot be quantised")
    max_abs = F32(np.max(np.abs(x))) if x.size else F32(0.0)
    if max_abs == 0:
        scale = F32(1.0)
    else:
        scale = F32(max_abs / F32(qmax))
        if scale == 0:                        # float32 underflow (denormal max_abs)
            scale = F32(1.0)
    y = x / scale                             # float32 / float32 scalar -> float32
    q = np.clip(roundf(y), F32(-qmax), F32(qmax))
    return q.astype(QDTYPE[rep].newbyteorder("=")), scale


def dequantise_channel(q: np.ndarray, scale) -> np.ndarray:
    """Device-side reconstruction: float32(q) * scale, float32."""
    return (np.asarray(q).astype(F32) * F32(scale)).astype(F32)


def roundtrip(channels: np.ndarray, rep: str = "int16") -> np.ndarray:
    """channels[C, N] float32 -> float32 array the device computes on."""
    out = np.empty(channels.shape, dtype=F32)
    for c in range(channels.shape[0]):
        q, s = quantise_channel(channels[c], rep)
        out[c] = dequantise_channel(q, s)
    return out


# ---------------------------------------------------------------------------
# Per-machine checkpointed pass.
# ---------------------------------------------------------------------------
def cache_path(machine_id: str) -> Path:
    return CACHE_V2 / (machine_id.replace("::", "__") + ".npz")


def body_path(machine_id: str, stream: int) -> Path:
    return BODY_DIR / f"{machine_id.replace('::', '__')}__s{stream}.body"


def load_machine_cache(machine_id: str) -> dict:
    with np.load(cache_path(machine_id), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def machine_done(machine_id: str, need_int8: bool, need_body: bool) -> bool:
    p = cache_path(machine_id)
    if not p.exists():
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            files = set(z.files)
            if "complete" not in files:
                return False
            if need_int8 and "raw22_int8" not in files:
                return False
            if need_body:
                if "body_crc32" not in files:
                    return False
                n_streams = int(z["raw22_int16"].shape[0])
                for s in range(n_streams):
                    if not body_path(machine_id, s).exists():
                        return False
    except Exception:
        return False
    return True


def compute_machine(machine_id: str, position: int, need_int8: bool, need_body: bool,
                    log=print) -> Path:
    """One raw pass for `machine_id`. Writes cache npz (+ int16 body files)."""
    import raw_common as rc
    import raw_features_f32 as rff
    import twin

    t0 = time.time()
    manifest = rc.load_manifest()
    refs = rc.machine_streams(machine_id, manifest)
    reps = ["int16"] + (["int8"] if need_int8 else [])

    with np.load(COMMITTED_CACHE / (machine_id.replace("::", "__") + ".npz"), allow_pickle=False) as z:
        train_idx = np.asarray(z["train_indices"], dtype=np.int64)
        cached_rows = int(np.asarray(z["x"]).shape[0])

    if need_body:
        BODY_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_V2.mkdir(parents=True, exist_ok=True)

    raw22 = {r: [] for r in reps}          # rep -> list per stream of (n,22)
    lengths, n_channels_list, body_crc, body_bytes = [], [], [], []
    for si, ref in enumerate(refs):
        decim = rc.decimates(ref)
        rows = {r: [] for r in reps}
        lens = []
        n_ch = None
        fh = None
        crc = 0
        nbytes = 0
        tmp = None
        if need_body:
            tmp = body_path(machine_id, si).with_suffix(".body.part")
            fh = tmp.open("wb")
        try:
            for index, channels, _token in rc.iter_stream_channels(ref):
                if index != len(lens):
                    raise RuntimeError(f"non-causal acquisition order for {ref.stream_label}")
                fed = rc.decimate_v30_style(channels) if decim else np.asarray(channels, dtype=F32)
                if n_ch is None:
                    n_ch = int(fed.shape[0])
                elif fed.shape[0] != n_ch:
                    raise RuntimeError(f"channel count changed in {ref.stream_label}")
                lens.append(int(fed.shape[1]))
                for rep in reps:
                    rt = np.empty(fed.shape, dtype=F32)
                    for c in range(fed.shape[0]):
                        q, s = quantise_channel(fed[c], rep)
                        rt[c] = dequantise_channel(q, s)
                        if rep == "int16" and fh is not None:
                            blob = np.asarray(s, dtype="<f4").tobytes() + q.astype("<i2").tobytes()
                            fh.write(blob)
                            crc = zlib.crc32(blob, crc)
                            nbytes += len(blob)
                    rows[rep].append(rff.acquisition_22_f32(rt))
        finally:
            if fh is not None:
                fh.close()
        if need_body:
            os.replace(tmp, body_path(machine_id, si))
        for rep in reps:
            raw22[rep].append(np.asarray(rows[rep], dtype=F32))
        lengths.append(np.asarray(lens, dtype=np.int64))
        n_channels_list.append(n_ch)
        body_crc.append(crc & 0xFFFFFFFF)
        body_bytes.append(nbytes)

    n_streams = [len(lens) for lens in lengths]
    n_min = min(n_streams)
    if n_min != cached_rows:
        raise RuntimeError(f"{machine_id}: raw stream rows {n_streams} vs committed cache rows {cached_rows}")
    if len(set(n_streams)) != 1 and need_body:
        raise RuntimeError(f"{machine_id}: streams have different acquisition counts {n_streams}; "
                           f"body files hold all acquisitions -- handle explicitly")

    out = {"complete": np.asarray(1, dtype=np.int8),
           "n_rows": np.asarray(n_min, dtype=np.int64),
           "n_channels": np.asarray(n_channels_list, dtype=np.int64),
           "stream_acquisitions": np.asarray(n_streams, dtype=np.int64),
           "lengths": np.stack([l[:n_min] for l in lengths])}
    for rep in reps:
        streams = [s[:n_min] for s in raw22[rep]]
        out[f"raw22_{rep}"] = np.stack(streams).astype(F32)          # (S, n, 22)
        x, _params = twin.standardize(streams, train_idx, dtype=np.float32)
        out[f"x_{rep}"] = x.astype(F32)
    if need_body:
        out["body_crc32"] = np.asarray(body_crc, dtype=np.uint32)
        out["body_bytes"] = np.asarray(body_bytes, dtype=np.int64)
    out["elapsed_s"] = np.asarray(time.time() - t0)

    final = cache_path(machine_id)
    tmp_npz = final.with_name(final.stem + ".tmp.npz")
    np.savez(tmp_npz, **out)
    os.replace(tmp_npz, final)
    log(f"[pos {position:2d}] {machine_id:22s} done in {time.time() - t0:7.1f}s "
        f"rows={n_min} streams={len(refs)} reps={reps} body={need_body}")
    return final


def main_one(argv=None):
    """CLI: python raw_int16_v2.py --machine ID --position P [--int8] [--body]"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True)
    ap.add_argument("--position", type=int, required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--body", action="store_true")
    a = ap.parse_args(argv)
    if machine_done(a.machine, a.int8, a.body):
        print(f"[pos {a.position:2d}] {a.machine} already done, skipping", flush=True)
        return 0
    compute_machine(a.machine, a.position, a.int8, a.body, log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main_one())
