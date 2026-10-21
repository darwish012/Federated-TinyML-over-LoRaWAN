#include "tf_features.h"

namespace tinyfl {

static const double kPiFeat = 3.14159265358979323846;
static const float kEps32 = 1e-12f;  // raw_features_f32.EPS32

void TfExtractor::release() {
  if (buf_) tf_free(buf_);
  if (pw_) tf_free(pw_);
  buf_ = nullptr;
  pw_ = nullptr;
  cap_ = 0;
  tf_fft_release();
}

bool TfExtractor::ensure(uint32_t M) {
  bool direct = tf_fft_method((int)M,nullptr,nullptr)==1;
  if (buf_ && cap_ >= M) {
    if (!direct && pw_) { tf_free(pw_); pw_=nullptr; }
    if (direct && !pw_) pw_=static_cast<float*>(tf_alloc(sizeof(float)*(cap_/2+1)));
    return !direct || pw_;
  }
  if (buf_) tf_free(buf_);
  if (pw_) tf_free(pw_);
  buf_ = nullptr;
  pw_ = nullptr;
  cap_ = 0;
  buf_ = static_cast<float*>(tf_alloc(sizeof(float) * (size_t)M));
  if (direct) pw_ = static_cast<float*>(tf_alloc(sizeof(float) * ((size_t)M / 2 + 1)));
  if (!buf_ || (direct && !pw_)) {
    if (buf_) tf_free(buf_);
    if (pw_) tf_free(pw_);
    buf_ = pw_ = nullptr;
    tf_set_error("no heap for the %u-sample FFT buffer", (unsigned)M);
    return false;
  }
  cap_ = M;
  return true;
}

static inline float signed_log(float v) {
  const float s = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f);
  return s * log1pf(fabsf(v));
}

// Spectral descriptors of the strided subsequence held in buf_[0..M):
// mean-centre, Hann window (symmetric, n-1), power spectrum of bins 1..M/2,
// entropy / centroid / spread.  Every rounding step follows the numpy float32
// code in raw_features_f32.spectral_descriptors_f32_streaming.
bool TfExtractor::spectral(uint32_t Mu, float out3[3]) {
  const int M = (int)Mu;
  float* a = buf_;
  double sum = 0.0;
  for (int i = 0; i < M; ++i) sum += (double)a[i];
  const float mean = (float)(sum / (double)M);

  // Hann: w_k = 0.5 - 0.5*cos(2*pi*k/(M-1)) in double, via the Chebyshev
  // recurrence (error ~1e-12), then rounded to float32 like hann_window_f32.
  {
    const double delta = 2.0 * kPiFeat / (double)(M - 1);
    const double c1 = cos(delta);
    const double twoC = 2.0 * c1;
    double cprev = 1.0, ccur = c1;
    for (int k = 0; k < M; ++k) {
      double c;
      if (k == 0) {
        c = 1.0;
      } else if (k == 1) {
        c = c1;
      } else {
        c = twoC * ccur - cprev;
        cprev = ccur;
        ccur = c;
      }
      const float w = (float)(0.5 - 0.5 * c);
      const float centred = a[k] - mean;
      a[k] = centred * w;
      if ((k & 4095) == 4095) tf_yield();
    }
  }

  const int K = M / 2;
  if (!tf_power_spectrum(a, M, pw_)) {
    tf_set_error("FFT plan allocation failed (M=%d)", M);
    return false;
  }
  float* pw = pw_;
  auto power = [&](int i)->float& { return pw ? pw[i] : *tf_packed_power_bin(a,M,i+1); };
  // power = max(|X|^2, 1e-20f); total = float32(sum in double)
  double tot = 0.0;
  for (int i = 0; i < K; ++i) {
    if (!(power(i) >= 1e-20f)) power(i) = 1e-20f;  // also catches NaN
    tot += (double)power(i);
  }
  const float total = (float)tot;
  for (int i = 0; i < K; ++i) power(i) = power(i) / total;  // probabilities, float32
  tf_yield();

  const float logLen = (float)log((double)K);
  const double step = 1.0 / (double)(K - 1);
  double sumE = 0.0, sumC = 0.0;
  for (int i = 0; i < K; ++i) {
    const float p = power(i);
    const float lg = logf(p + 1e-20f);
    sumE += (double)(p * lg);
    const float f = (i == K - 1) ? 1.0f : (float)((double)i * step);
    sumC += (double)(p * f);
    if ((i & 2047) == 2047) tf_yield();
  }
  const float entropy = (float)(-sumE / (double)logLen);
  const float centroid = (float)sumC;
  double sumS = 0.0;
  for (int i = 0; i < K; ++i) {
    const float f = (i == K - 1) ? 1.0f : (float)((double)i * step);
    const float dlt = f - centroid;
    const float d2 = dlt * dlt;
    sumS += (double)(power(i) * d2);
  }
  const float spread = sqrtf((float)sumS);
  out3[0] = entropy;
  out3[1] = centroid;
  out3[2] = spread;
  return true;
}

bool TfExtractor::channelDescriptors(TfAcquisitionSource& src, uint32_t n, float scale,
                                     float out11[11]) {
  // Stride rule: stride = ceil(n/16384) if n > 16384; M = ceil(n/stride).
  uint32_t stride = 1, M = n;
  if (n > (uint32_t)kMaxFftSamples) {
    stride = (n - 1) / kMaxFftSamples + 1;
    M = (n - 1) / stride + 1;
    if (M > (uint32_t)kMaxFftSamples) M = kMaxFftSamples;
  }
  if (!ensure(M)) return false;
  float* fb = buf_;

  // ---- the single streaming pass -------------------------------------
  double s1 = 0, s2 = 0, s3 = 0, s4 = 0, sabs = 0, ssqrt = 0;
  float xmin = INFINITY, xmax = -INFINITY;
  uint32_t gi = 0, sel = 0, nextCap = 0;
  for (;;) {
    tf_progress("sd_read",gi,n);
    const int got = src.readChunk(chunk_, kChunkSamples);
    tf_progress("features",gi,n);
    if (got < 0) {
      tf_set_error("sample read failed");
      return false;
    }
    if (got == 0) break;
    if (gi + (uint32_t)got > n) {
      tf_set_error("channel longer than its %u samples", (unsigned)n);
      return false;
    }
    for (int i = 0; i < got; ++i) {
      const float v = (float)chunk_[i] * scale;  // float32 multiply
      const double dv = (double)v;
      const double v2 = dv * dv;
      s1 += dv;
      s2 += v2;
      s3 += v2 * dv;
      s4 += v2 * v2;
      sabs += fabs(dv);
      ssqrt += (double)sqrtf(fabsf(v));
      if (v < xmin) xmin = v;
      if (v > xmax) xmax = v;
      if (gi == nextCap) {
        fb[sel++] = v;
        nextCap += stride;
      }
      ++gi;
    }
    tf_yield();
  }
  if (gi != n) {
    tf_set_error("channel has %u samples, expected %u", (unsigned)gi, (unsigned)n);
    return false;
  }
  if (sel != M) {
    tf_set_error("internal: strided %u != M %u", (unsigned)sel, (unsigned)M);
    return false;
  }

  // ---- raw-moment reconstruction (double), then float32 rounding ----------
  const double nn = (double)n;
  const double mean = s1 / nn;
  const double m2 = s2 / nn - mean * mean;
  const double m3 = s3 / nn - 3.0 * mean * (s2 / nn) + 2.0 * pow(mean, 3.0);
  const double m4 =
      s4 / nn - 4.0 * mean * (s3 / nn) + 6.0 * mean * mean * (s2 / nn) - 3.0 * pow(mean, 4.0);
  const float stddev = (float)sqrt(m2 > 0.0 ? m2 : 0.0);
  float skewness = 0.0f, kurtosis = 0.0f;
  if (stddev > 1e-6f) {
    // numpy 1.x: np.float32 ** int -> float64
    skewness = (float)(m3 / pow((double)stddev, 3.0));
    kurtosis = (float)(m4 / pow((double)stddev, 4.0));
  }
  const float absMean = (float)(sabs / nn);
  const float maxAbs = (float)fmax(fabs((double)xmin), fabs((double)xmax));
  const float sqrtMeanSq = (float)pow(ssqrt / nn, 2.0);
  const float rms = (float)sqrt((s2 / nn) > 0.0 ? (s2 / nn) : 0.0);

  float raw[8];
  raw[0] = rms;
  raw[1] = xmax - xmin;
  raw[2] = kurtosis;
  raw[3] = skewness;
  raw[4] = maxAbs / (rms > kEps32 ? rms : kEps32);
  raw[5] = maxAbs / (sqrtMeanSq > kEps32 ? sqrtMeanSq : kEps32);
  raw[6] = rms / (absMean > kEps32 ? absMean : kEps32);
  raw[7] = maxAbs / (absMean > kEps32 ? absMean : kEps32);
  for (int i = 0; i < 8; ++i) out11[i] = signed_log(raw[i]);

  return spectral(M, out11 + 8);
}

bool TfExtractor::extract(TfAcquisitionSource& src, float out22[kFeat]) {
  TfAcqInfo info;
  if (!src.nextAcquisition(&info)) {
    tf_set_error("no further acquisition");
    return false;
  }
  const int C = info.channels;
  if (C < 1 || C > kMaxChannels) {
    tf_set_error("bad channel count %d", C);
    return false;
  }
  if (info.samples < 512) {
    tf_set_error("acquisition %u has only %u samples", (unsigned)info.index, (unsigned)info.samples);
    return false;
  }
  float rows[kMaxChannels][11];
  for (int c = 0; c < C; ++c) {
    float scale = 0.0f;
    if (!src.beginChannel((uint16_t)c, &scale)) {
      tf_set_error("beginChannel(%d) failed", c);
      return false;
    }
    if (!isfinite(scale) || scale<=0) {
      tf_set_error("non-finite scale in acquisition %u", (unsigned)info.index);
      return false;
    }
    if (!channelDescriptors(src, info.samples, scale, rows[c])) return false;
  }
  // Channel collapse: mean block (double accumulation) + argmax-|value| block.
  for (int j = 0; j < 11; ++j) {
    double acc = 0.0;
    int best = 0;
    float bestAbs = fabsf(rows[0][j]);
    for (int c = 0; c < C; ++c) {
      acc += (double)rows[c][j];
      const float av = fabsf(rows[c][j]);
      if (av > bestAbs) {
        bestAbs = av;
        best = c;
      }
    }
    out22[j] = (float)(acc / (double)C);
    out22[11 + j] = rows[best][j];
  }
  for (int j = 0; j < kFeat; ++j) {
    if (!isfinite(out22[j])) {
      tf_set_error("non-finite feature %d in acquisition %u", j, (unsigned)info.index);
      return false;
    }
  }
  return true;
}

}  // namespace tinyfl
