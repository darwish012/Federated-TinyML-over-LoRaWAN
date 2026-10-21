// Streaming float32 feature extractor: a C++ port of
// research/hardware_twin_scaling_20260929/raw_features_f32.py.
//
// One pass over each channel's int16 samples (dequantised as float32
// (float)q * scale), double accumulators for the raw moments, only the strided
// <= 16384-sample subsequence kept for the FFT.  Channels are processed one at a
// time; only their 11 descriptors survive, then collapse into 22 values:
//   [0..10]  mean over channels  (rms, ptp, kurtosis, skewness, crest,
//                                 clearance, shape, impulse, entropy,
//                                 centroid, spread)
//   [11..21] the channel with the largest |value| per descriptor
// (time descriptors are signed-log values).
#pragma once

#include "tf_fft.h"
#include "tf_source.h"

namespace tinyfl {

class TfExtractor {
 public:
  TfExtractor() {}
  ~TfExtractor() { release(); }

  // Reads ONE acquisition from `src` (all channels) and writes the 22 raw
  // features.  false on any error (see tf_last_error()).
  bool extract(TfAcquisitionSource& src, float out22[kFeat]);

  // Frees the FFT buffers and plan.  Call between phases so the heap is whole.
  void release();

  // Bytes held by the extractor right now (FFT buffer + spectrum).
  size_t heapBytes() const { return (buf_ ? cap_ * 4u : 0u) + (pw_ ? (cap_ / 2 + 1) * 4u : 0u); }

  // Descriptors of the CURRENT channel of `src` (beginChannel already called,
  // `scale` is what it returned) holding n samples.  out11 = 8 time + 3
  // spectral.  Exposed for the host tests.
  bool channelDescriptors(TfAcquisitionSource& src, uint32_t n, float scale, float out11[11]);

 private:
  bool ensure(uint32_t M);
  bool spectral(uint32_t M, float out3[3]);

  float* buf_ = nullptr;   // strided samples, then windowed signal
  float* pw_ = nullptr;    // power spectrum, K = M/2 bins
  size_t cap_ = 0;         // capacity of buf_ in floats
  int16_t chunk_[kChunkSamples];
};

}  // namespace tinyfl
