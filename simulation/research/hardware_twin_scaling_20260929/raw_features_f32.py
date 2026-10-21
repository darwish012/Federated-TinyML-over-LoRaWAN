"""Deliverable 3: the float32 feature extractor, written the way the C
firmware will compute it -- STREAMED from SD in fixed-size chunks, ONE
pass over each channel's samples, raw-moment (not centred two-pass)
formulas for skewness/kurtosis, and only the already-strided <=16384-
sample subsequence held in RAM for the FFT.

This supersedes an earlier two-pass, whole-buffer-in-RAM version of this
file (its results are kept at float32_deviation_twopass_centered_SUPERSEDED
.json/.log for comparison only). The coordinator's review of the RAM
finding in raw_inventory.py determined the heap flag is not a blocker
*if* the firmware streams: this module now matches that design exactly.

Algorithm (streaming, single pass per channel):

  time_descriptors_f32_streaming(chunk_source[N], chunk_size):
      accumulators (float64 -- "double accumulators on the ESP32 are
      fine": ESP32 has no hardware double FPU, so double ops are software-
      emulated, but this is 8 scalars updated N times, not a bottleneck):
          S1=S2=S3=S4=Sabs=Ssqrt = 0.0;  xmin=+inf; xmax=-inf
      for each chunk (chunk_size samples, e.g. read from SD into a small
                       float32 buffer, then discarded):
          for v in chunk:                          # explicit loop order
              v32 = (float32) v
              S1    += v32                          # Sum x
              S2    += v32*v32                      # Sum x^2
              S3    += v32*v32*v32                  # Sum x^3
              S4    += v32*v32*v32*v32               # Sum x^4
              Sabs  += |v32|                        # Sum |x|
              Ssqrt += sqrt(|v32|)                  # Sum sqrt(|x|)
              xmin = min(xmin, v32); xmax = max(xmax, v32)
      # ---- raw-moment central-moment reconstruction (single pass) ----
      n = N
      mean = S1/n
      m2 = S2/n - mean^2                                    # population variance
      m3 = S3/n - 3*mean*(S2/n) + 2*mean^3                  # 3rd central moment
      m4 = S4/n - 4*mean*(S3/n) + 6*mean^2*(S2/n) - 3*mean^4  # 4th central moment
      stddev = sqrt(max(m2, 0))
      skewness = m3/stddev^3 if stddev > 1e-6 else 0        # RAW-MOMENT form
      kurtosis = m4/stddev^4 if stddev > 1e-6 else 0        # RAW-MOMENT form
      abs_mean = Sabs/n
      max_abs  = max(|xmin|, |xmax|)     # identity: max_i|x_i| == max(|xmin|,|xmax|)
                                          # (every x_i in [xmin,xmax], and xmin/xmax
                                          #  are themselves achieved samples) --
                                          # avoids a second |.| max pass.
      sqrt_mean_sq = (Ssqrt/n)^2
      rms = sqrt(S2/n)
      8 raw descriptors -> signed_log, exactly as the two-pass version:
          [rms, xmax-xmin, kurtosis, skewness,
           max_abs/max(rms,eps), max_abs/max(sqrt_mean_sq,eps),
           rms/max(abs_mean,eps), max_abs/max(abs_mean,eps)]

  Numerical note (why this is measured separately from the two-pass form):
  rms/ptp/abs_mean/sqrt_mean_sq/max_abs are IDENTICAL formulas to the
  two-pass version (both are already single-pass sums). Only skewness and
  kurtosis change: the two-pass form centres first (x - mean) and then
  sums powers of the centred values (numerically stable, no cancellation);
  this raw-moment form sums powers of the UNCENTRED values and subtracts
  large near-equal terms (S3/n vs 3*mean*S2/n, etc.), which loses relative
  precision when |mean| is large compared to the spread of x -- a real,
  measured deviation, reported per machine below, not hidden.

  spectral_descriptors_f32_streaming(chunk_source[N], chunk_size):
      stride = ceil(N/16384) if N>16384 else 1               # known from N up front
      fft_buffer = float32[min(N,16384)]                     # <=64 KB, the ONLY
                                                               # full-length buffer
      global_i = 0
      for each chunk (chunk_size samples):
          for v in chunk:
              if global_i % stride == 0 and (global_i//stride) < len(fft_buffer):
                  fft_buffer[global_i // stride] = (float32) v
              global_i += 1
      # from here on, everything operates on the <=16384-sample fft_buffer only:
      mean = sum(fft_buffer)/M                                # single pass, tiny
      centred = fft_buffer - mean
      windowed = centred * hann(M)          # hann(M)[k] computed on the fly,
                                             # no separate 64 KB window array needed
      X = rfft(windowed)                    # float32 in -> complex64 (kiss_fftr
                                             # stand-in, same length M on the C side)
      power = |X[1:]|^2, floored at 1e-20f  # M/2 complex bins -> <=64 KB
      entropy/centroid/spread as before (probabilities = power/sum(power), etc.)

  per_channel_22 / acquisition_22_f32: unchanged from the earlier version
  (mean-over-channels block + per-descriptor argmax-|value| extreme block,
  22 values in FEATURE_INDICES order). Channels are processed one at a
  time and their buffers discarded before the next channel starts, so
  peak RAM does NOT scale with channel count (only the 11-descriptor row
  per channel -- 44 bytes -- needs to be kept once each channel finishes).

Then: per-stream standardisation and elementwise max over streams is
twin.standardize (research/hardware_twin_scaling_20260929/twin.py,
read-only import here, never edited or duplicated) -- the same formula
protocol.json's on_device_pipeline specifies, already float32-exact.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "healthy_only_suite_20260907"))
sys.path.insert(0, str(ROOT))
import twin  # noqa: E402  -- read-only import; standardize() is used unmodified

MAX_FFT_SAMPLES = 16_384
CHUNK_SAMPLES = 1024          # representative SD-read chunk size (documented, not tuned)
F32 = np.float32
F64 = np.float64
EPS32 = np.float32(1e-12)

TIME_DESCRIPTOR_NAMES = ("rms", "ptp", "kurtosis", "skewness", "crest", "clearance", "shape", "impulse")
SPECTRAL_DESCRIPTOR_NAMES = ("entropy", "centroid", "spread")
DESCRIPTOR_NAMES_11 = TIME_DESCRIPTOR_NAMES + SPECTRAL_DESCRIPTOR_NAMES

# ---------------------------------------------------------------------------
# Per-acquisition RAM budget under streaming (see module docstring), stated
# here as constants so raw_inventory.py / raw_sd_plan.md can quote them.
# ---------------------------------------------------------------------------
RAM_CHUNK_BUFFER_BYTES = CHUNK_SAMPLES * 4                       # one float32 chunk
RAM_FFT_REAL_BUFFER_BYTES = MAX_FFT_SAMPLES * 4                  # strided real samples, <=64 KiB
RAM_FFT_COMPLEX_OUTPUT_BYTES = (MAX_FFT_SAMPLES // 2 + 1) * 8     # rfft output, complex64
RAM_ACCUMULATORS_BYTES = 8 * 8 + 2 * 4                            # 8 float64 sums + xmin/xmax float32
RAM_PER_CHANNEL_PEAK_BYTES_NOT_REUSED = (
    RAM_CHUNK_BUFFER_BYTES + RAM_FFT_REAL_BUFFER_BYTES + RAM_FFT_COMPLEX_OUTPUT_BYTES + RAM_ACCUMULATORS_BYTES
)
# If the complex spectrum is computed in place over (part of) the real
# buffer's memory (a standard real-FFT technique; kiss_fftr's packed
# format overlaps input/output), the complex-output allocation is not
# additional:
RAM_PER_CHANNEL_PEAK_BYTES_BUFFER_REUSE = (
    RAM_CHUNK_BUFFER_BYTES + RAM_FFT_REAL_BUFFER_BYTES + RAM_ACCUMULATORS_BYTES
)


def _signed_log_f32(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=F32)
    return (np.sign(values) * np.log1p(np.abs(values))).astype(F32)


def stride_indices(n: int, limit: int = MAX_FFT_SAMPLES) -> np.ndarray:
    """The stride rule (protocol: 'strided down to <=16384 samples'):
    stride = ceil(n/limit); every `stride`-th global sample index."""
    if n <= limit:
        return np.arange(n)
    stride = int(math.ceil(n / limit))
    return np.arange(0, n, stride)[:limit]


def _chunks(signal: np.ndarray, chunk_size: int = CHUNK_SAMPLES):
    """Yields (start, chunk) -- simulates sequential SD reads. `signal` is
    already in memory here (this is a Python simulation of firmware
    behaviour, not real hardware), but the accumulation logic below never
    indexes outside the current chunk, so it is a faithful single-pass,
    bounded-memory algorithm regardless of where the bytes actually come
    from."""
    n = signal.shape[0]
    for start in range(0, n, chunk_size):
        yield start, signal[start:start + chunk_size]


def time_descriptors_f32_streaming(signal: np.ndarray, chunk_size: int = CHUNK_SAMPLES) -> np.ndarray:
    """8 signed-log time descriptors via ONE streaming pass, float64
    (double) accumulators, raw-moment skewness/kurtosis (see module
    docstring for the exact formulas and their numerical trade-off vs a
    two-pass centred form)."""
    x = np.asarray(signal, dtype=F32).reshape(-1)
    n = x.shape[0]
    if n == 0:
        raise ValueError("empty signal")

    s1 = s2 = s3 = s4 = sabs = ssqrt = F64(0.0)
    xmin = F32(np.inf)
    xmax = F32(-np.inf)
    for _start, chunk in _chunks(x, chunk_size):
        c = chunk.astype(F32)
        # Per-chunk partial sums, accumulated into float64 totals -- this
        # is mathematically the single running-sum loop of the docstring,
        # computed chunk-at-a-time (numpy vectorised within a chunk,
        # sequential across chunks) rather than one Python-level scalar
        # loop over every sample, for tractable runtime on real data.
        s1 += F64(np.sum(c, dtype=F64))
        s2 += F64(np.sum(c.astype(F64) ** 2))
        s3 += F64(np.sum(c.astype(F64) ** 3))
        s4 += F64(np.sum(c.astype(F64) ** 4))
        sabs += F64(np.sum(np.abs(c), dtype=F64))
        ssqrt += F64(np.sum(np.sqrt(np.abs(c)), dtype=F64))
        cmax = F32(np.max(c))
        cmin = F32(np.min(c))
        if cmax > xmax:
            xmax = cmax
        if cmin < xmin:
            xmin = cmin

    mean = F64(s1 / n)
    m2 = F64(s2 / n - mean * mean)
    m3 = F64(s3 / n - 3.0 * mean * (s2 / n) + 2.0 * mean ** 3)
    m4 = F64(s4 / n - 4.0 * mean * (s3 / n) + 6.0 * mean * mean * (s2 / n) - 3.0 * mean ** 4)
    stddev = F32(np.sqrt(max(m2, 0.0)))
    if stddev > F32(1e-6):
        skewness = F32(m3 / (stddev ** 3))
        kurtosis = F32(m4 / (stddev ** 4))
    else:
        skewness = F32(0.0)
        kurtosis = F32(0.0)

    abs_mean = F32(sabs / n)
    max_abs = F32(max(abs(float(xmin)), abs(float(xmax))))
    sqrt_mean_sq = F32((ssqrt / n) ** 2)
    rms = F32(np.sqrt(max(s2 / n, 0.0)))

    raw = np.asarray([
        rms, xmax - xmin, kurtosis, skewness,
        max_abs / max(rms, EPS32), max_abs / max(sqrt_mean_sq, EPS32),
        rms / max(abs_mean, EPS32), max_abs / max(abs_mean, EPS32),
    ], dtype=F32)
    return _signed_log_f32(raw)


def hann_window_f32(n: int) -> np.ndarray:
    """Symmetric Hann window, float32, matching np.hanning(n) exactly."""
    if n <= 1:
        return np.ones(max(n, 0), dtype=F32)
    k = np.arange(n, dtype=np.float64)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * k / (n - 1))
    return w.astype(F32)


def build_fft_buffer_streaming(signal: np.ndarray, chunk_size: int = CHUNK_SAMPLES) -> np.ndarray:
    """The <=16384-sample strided subsequence, built during the SAME
    single streaming pass a real device would use (only the selected
    samples are ever kept; the rest of each chunk is discarded)."""
    x = np.asarray(signal, dtype=F32).reshape(-1)
    n = x.shape[0]
    idx = stride_indices(n)
    buf = np.empty(idx.shape[0], dtype=F32)
    # idx is an arithmetic sequence (0, stride, 2*stride, ...); this is
    # the vectorised form of "for each chunk, copy the samples whose
    # global index is a multiple of stride into fft_buffer" -- exact same
    # selection, computed without a second full-length pass.
    for start, chunk in _chunks(x, chunk_size):
        end = start + chunk.shape[0]
        sel = idx[(idx >= start) & (idx < end)]
        if sel.size:
            buf[np.searchsorted(idx, sel)] = chunk[sel - start].astype(F32)
    return buf


def spectral_descriptors_f32_streaming(signal: np.ndarray, chunk_size: int = CHUNK_SAMPLES) -> np.ndarray:
    """3 spectral descriptors (entropy, centroid, spread), float32, built
    from the streamed <=16384-sample FFT buffer only."""
    strided = build_fft_buffer_streaming(signal, chunk_size)
    m = strided.shape[0]

    mean = F32(np.sum(strided, dtype=F64) / m)
    centered = (strided - mean).astype(F32)
    window = hann_window_f32(m)
    windowed = (centered * window).astype(F32)

    spectrum = np.fft.rfft(windowed)  # float32 in -> complex64 (numpy), stand-in for kiss_fftr
    power = (np.abs(spectrum[1:]).astype(F32)) ** F32(2.0)
    power = np.maximum(power, F32(1e-20))

    total = F32(np.sum(power, dtype=F64))
    probabilities = (power / total).astype(F32)
    frequencies = np.linspace(0.0, 1.0, power.shape[0], endpoint=True).astype(F32)

    log_len = F32(math.log(power.shape[0]))
    entropy = F32(-np.sum(probabilities * np.log(probabilities + F32(1e-20)), dtype=F64) / log_len)
    centroid = F32(np.sum(probabilities * frequencies, dtype=F64))
    spread = F32(np.sqrt(F32(np.sum(probabilities * (frequencies - centroid) ** 2, dtype=F64))))
    return np.asarray([entropy, centroid, spread], dtype=F32)


def channel_descriptors_11_f32(signal: np.ndarray, chunk_size: int = CHUNK_SAMPLES) -> np.ndarray:
    """The 11 descriptors (8 time + 3 spectral) for one channel, float32,
    streaming/single-pass."""
    td = time_descriptors_f32_streaming(signal, chunk_size)
    sd = spectral_descriptors_f32_streaming(signal, chunk_size)
    return np.concatenate([td, sd]).astype(F32)


def acquisition_22_f32(channels: np.ndarray, chunk_size: int = CHUNK_SAMPLES) -> np.ndarray:
    """channels[C,N] (any C>=1, N>=512) -> the 22 FEATURE_INDICES-order
    values: mean-over-channels (11) then per-descriptor argmax-|value|
    channel selection (11). Channels are processed strictly one at a time
    (streaming each fully before starting the next), so peak RAM does not
    scale with channel count."""
    values = np.asarray(channels, dtype=F32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 512:
        raise ValueError(f"invalid current acquisition shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite raw acquisition")
    rows = np.stack([channel_descriptors_11_f32(values[c], chunk_size) for c in range(values.shape[0])]).astype(F32)
    mean_block = np.mean(rows, axis=0, dtype=np.float64).astype(F32)
    extreme_idx = np.argmax(np.abs(rows), axis=0)
    extreme_block = rows[extreme_idx, np.arange(rows.shape[1])]
    result = np.concatenate([mean_block, extreme_block]).astype(F32)
    if result.shape != (22,) or not np.all(np.isfinite(result)):
        raise ValueError("invalid float32 22-D feature vector")
    return result


# ---------------------------------------------------------------------------
# Per-machine float32-vs-float64 deviation measurement (deliverable 3's
# "measure the float32-vs-float64 deviation of the final standardized 22-D
# x per machine"). Reference (float64) = the already Gate-R-verified
# committed cache x. Candidate (float32) = this module's streaming raw-
# moment raw-waveform pipeline, end to end, through
# twin.standardize(..., dtype=float32).
# ---------------------------------------------------------------------------
def compute_machine_x32(mid: str, manifest: dict, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (x32, cached_x64) for one machine, memory-bounded (one
    machine's streams at a time)."""
    import raw_common as rc

    refs = rc.machine_streams(mid, manifest)
    stream_22 = []
    for ref in refs:
        rows = []
        decim = rc.decimates(ref)
        for index, channels, _token in rc.iter_stream_channels(ref):
            if index != len(rows):
                raise RuntimeError(f"non-causal acquisition order for {ref.stream_label}")
            x = rc.decimate_v30_style(channels) if decim else channels
            rows.append(acquisition_22_f32(x))
        stream_22.append(np.asarray(rows, dtype=np.float32))

    n_min = min(len(s) for s in stream_22)
    stream_22 = [s[:n_min] for s in stream_22]

    cache_path = cache_dir / (mid.replace("::", "__") + ".npz")
    with np.load(cache_path, allow_pickle=False) as z:
        cached_x = np.asarray(z["x"], dtype=np.float64)
        train_idx = np.asarray(z["train_indices"], dtype=np.int64)

    x32, _params = twin.standardize(stream_22, train_idx, dtype=np.float32)
    return x32, cached_x


def main() -> dict:
    import json
    import time
    import raw_common as rc

    OUT_DIR = ROOT / "results" / "hardware_twin_scaling_20260929" / "raw"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR = ROOT / "results" / "healthy_only_suite_20260907" / "cache"

    manifest = rc.load_manifest()
    order, _assignment = rc.hardware_machines()
    hw_order = order[: rc.HARDWARE_POSITIONS]

    rows = []
    for position, dmeta in enumerate(hw_order):
        mid = dmeta["machine_id"]
        t0 = time.time()
        try:
            x32, x64 = compute_machine_x32(mid, manifest, CACHE_DIR)
            if x32.shape != x64.shape:
                rows.append({"position": position, "machine_id": mid, "dataset": dmeta["dataset"],
                             "error": f"shape mismatch {x32.shape} vs {x64.shape}"})
                print(f"[{position:2d}] {mid}: SHAPE MISMATCH {x32.shape} vs {x64.shape}", flush=True)
                continue
            diff = np.abs(x32.astype(np.float64) - x64)
            row = {
                "position": position, "machine_id": mid, "dataset": dmeta["dataset"],
                "n_rows": int(x32.shape[0]),
                "max_abs_diff": float(np.max(diff)),
                "mean_abs_diff": float(np.mean(diff)),
                "p99_abs_diff": float(np.quantile(diff, 0.99)),
                "elapsed_s": time.time() - t0, "error": None,
            }
            rows.append(row)
            print(f"[{position:2d}] {mid:22s} ({dmeta['dataset']:10s}) max_abs_diff={row['max_abs_diff']:.3e} "
                  f"mean_abs_diff={row['mean_abs_diff']:.3e} ({row['elapsed_s']:.1f}s)", flush=True)
        except Exception as exc:
            rows.append({"position": position, "machine_id": mid, "dataset": dmeta["dataset"],
                         "error": f"{exc.__class__.__name__}: {exc}"})
            print(f"[{position:2d}] {mid}: ERROR {exc}", flush=True)

    ok_rows = [r for r in rows if r.get("error") is None]
    summary = {
        "extractor": "streaming single-pass, raw-moment skewness/kurtosis (raw_features_f32.py, current version)",
        "chunk_samples": CHUNK_SAMPLES,
        "ram_per_channel_peak_bytes_not_reused": RAM_PER_CHANNEL_PEAK_BYTES_NOT_REUSED,
        "ram_per_channel_peak_bytes_buffer_reuse": RAM_PER_CHANNEL_PEAK_BYTES_BUFFER_REUSE,
        "n_machines": len(rows), "n_ok": len(ok_rows),
        "overall_max_abs_diff": max((r["max_abs_diff"] for r in ok_rows), default=None),
        "overall_mean_abs_diff": float(np.mean([r["mean_abs_diff"] for r in ok_rows])) if ok_rows else None,
        "per_machine": rows,
    }
    (OUT_DIR / "float32_deviation.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\noverall max_abs_diff (streaming float32 vs float64, level-28 hw machines) = {summary['overall_max_abs_diff']}")
    print(f"RAM/channel peak (buffers not reused): {RAM_PER_CHANNEL_PEAK_BYTES_NOT_REUSED} bytes "
          f"({RAM_PER_CHANNEL_PEAK_BYTES_NOT_REUSED/1024:.1f} KiB)")
    print(f"RAM/channel peak (real/complex buffer reuse): {RAM_PER_CHANNEL_PEAK_BYTES_BUFFER_REUSE} bytes "
          f"({RAM_PER_CHANNEL_PEAK_BYTES_BUFFER_REUSE/1024:.1f} KiB)")
    return summary


if __name__ == "__main__":
    main()
