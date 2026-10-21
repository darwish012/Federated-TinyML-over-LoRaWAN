#include "tf_model.h"

#include <algorithm>
#include <stdlib.h>

namespace tinyfl {

static const int F = kFeat;
static const int H = kHidden;
static const int OFF_W1 = 0, OFF_B1 = 264, OFF_W2 = 276, OFF_B2 = 540;

// ------------------------------------------------------------------ forward
void tf_forward(const float* p, const float* x, float* h, float* y) {
  const float* W1 = p + OFF_W1;
  const float* b1 = p + OFF_B1;
  const float* W2 = p + OFF_W2;
  const float* b2 = p + OFF_B2;
  float acc[H];
  for (int j = 0; j < H; ++j) acc[j] = 0.0f;
  for (int i = 0; i < F; ++i) {
    const float xi = x[i];
    const float* w = W1 + i * H;
    for (int j = 0; j < H; ++j) acc[j] += xi * w[j];
  }
  for (int j = 0; j < H; ++j) h[j] = tanhf(acc[j] + b1[j]);
  float ya[F];
  for (int k = 0; k < F; ++k) ya[k] = 0.0f;
  for (int j = 0; j < H; ++j) {
    const float hj = h[j];
    const float* w = W2 + j * F;
    for (int k = 0; k < F; ++k) ya[k] += hj * w[k];
  }
  for (int k = 0; k < F; ++k) y[k] = ya[k] + b2[k];
}

// numpy float32 pairwise sum of 22 elements (n >= 8: eight partial sums)
static inline float pairwise22(const float* a) {
  float r0 = a[0], r1 = a[1], r2 = a[2], r3 = a[3], r4 = a[4], r5 = a[5], r6 = a[6], r7 = a[7];
  r0 += a[8]; r1 += a[9]; r2 += a[10]; r3 += a[11];
  r4 += a[12]; r5 += a[13]; r6 += a[14]; r7 += a[15];
  float res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
  res += a[16]; res += a[17]; res += a[18]; res += a[19]; res += a[20]; res += a[21];
  return res;
}

float tf_score_row(const float* p, const float* x) {
  float h[H], y[F], e[F];
  tf_forward(p, x, h, y);
  for (int k = 0; k < F; ++k) {
    const float d = y[k] - x[k];
    e[k] = d * d;
  }
  return pairwise22(e) / 22.0f;
}

float tf_mean_loss(const float* p, const float* x, int n) {
  double acc = 0.0;
  float h[H], y[F];
  for (int r = 0; r < n; ++r) {
    const float* xr = x + (size_t)r * F;
    tf_forward(p, xr, h, y);
    for (int k = 0; k < F; ++k) {
      const float d = y[k] - xr[k];
      acc += (double)(d * d);
    }
  }
  return (float)(acc / ((double)n * F));
}

// ------------------------------------------------------------------ training
void tf_local_train(const float* x, int n, float* p) {
  float gW1[F * H], gb1[H], gW2[H * F], gb2[F];
  const float nf = (float)((n * F) > 0 ? (n * F) : 1);
  for (int epoch = 0; epoch < kLocalEpochs; ++epoch) {
    memset(gW1, 0, sizeof(gW1));
    memset(gb1, 0, sizeof(gb1));
    memset(gW2, 0, sizeof(gW2));
    memset(gb2, 0, sizeof(gb2));
    const float* W2 = p + OFF_W2;
    float h[H], y[F], d[F], d1[H];
    for (int r = 0; r < n; ++r) {
      if ((r & 31) == 0) tf_yield();
      const float* xr = x + (size_t)r * F;
      tf_forward(p, xr, h, y);
      // delta = 2.0 * (y - x) / n   (float32, n = number of elements)
      for (int k = 0; k < F; ++k) d[k] = (2.0f * (y[k] - xr[k])) / nf;
      for (int k = 0; k < F; ++k) gb2[k] += d[k];
      for (int j = 0; j < H; ++j) {
        const float hj = h[j];
        float* g = gW2 + j * F;
        for (int k = 0; k < F; ++k) g[k] += hj * d[k];
        // dh = delta @ W2.T ; delta1 = dh * (1 - h*h)
        const float* w = W2 + j * F;
        float acc = 0.0f;
        for (int k = 0; k < F; ++k) acc += d[k] * w[k];
        d1[j] = acc * (1.0f - hj * hj);
        gb1[j] += d1[j];
      }
      for (int i = 0; i < F; ++i) {
        const float xi = xr[i];
        float* g = gW1 + i * H;
        for (int j = 0; j < H; ++j) g[j] += xi * d1[j];
      }
    }
    // p = p - lr * (grad + L2 * w)  (L2 on the weight matrices only)
    for (int i = 0; i < F * H; ++i) p[OFF_W1 + i] = p[OFF_W1 + i] - kLearningRate * (gW1[i] + kL2 * p[OFF_W1 + i]);
    for (int j = 0; j < H; ++j) p[OFF_B1 + j] = p[OFF_B1 + j] - kLearningRate * gb1[j];
    for (int i = 0; i < H * F; ++i) p[OFF_W2 + i] = p[OFF_W2 + i] - kLearningRate * (gW2[i] + kL2 * p[OFF_W2 + i]);
    for (int k = 0; k < F; ++k) p[OFF_B2 + k] = p[OFF_B2 + k] - kLearningRate * gb2[k];
    tf_yield();
  }
}

// --------------------------------------------------------- standardisation
static float median_sorted(const float* s, int n) {
  const int mid = n / 2;
  if (n & 1) return s[mid];
  return (s[mid - 1] + s[mid]) * 0.5f;  // even count: mean of the two middle values
}

bool tf_std_params(const float* rows, int n, float centre[kFeat], float scale[kFeat]) {
  if (n < 1) {
    tf_set_error("no train rows to standardise");
    return false;
  }
  float* col = static_cast<float*>(tf_alloc(sizeof(float) * (size_t)n));
  float* dev = static_cast<float*>(tf_alloc(sizeof(float) * (size_t)n));
  if (!col || !dev) {
    if (col) tf_free(col);
    if (dev) tf_free(dev);
    tf_set_error("no heap for standardisation");
    return false;
  }
  const float fn = (float)n;
  for (int j = 0; j < F; ++j) {
    for (int i = 0; i < n; ++i) {
      col[i] = rows[(size_t)i * F + j];
      if (!isfinite(col[i])) { tf_free(col); tf_free(dev); tf_set_error("nonfinite training feature"); return false; }
    }
    std::sort(col, col + n);
    const float c = median_sorted(col, n);
    for (int i = 0; i < n; ++i) dev[i] = fabsf(rows[(size_t)i * F + j] - c);
    std::sort(dev, dev + n);
    const float mad = 1.4826f * median_sorted(dev, n);
    // population std, float32, sequential
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += rows[(size_t)i * F + j];
    const float mean = sum / fn;
    float sq = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float dv = rows[(size_t)i * F + j] - mean;
      sq += dv * dv;
    }
    const float sd = sqrtf(sq / fn);
    float s = (mad >= sd) ? mad : sd;  // np.maximum
    if (!(s > 1e-8f)) s = 1.0f;
    centre[j] = c;
    scale[j] = s;
    tf_yield();
  }
  tf_free(col);
  tf_free(dev);
  return true;
}

void tf_std_apply(const float* raw, const float centre[kFeat], const float scale[kFeat],
                  float out[kFeat]) {
  for (int j = 0; j < F; ++j) out[j] = fabsf(asinhf((raw[j] - centre[j]) / scale[j]));
}

// -------------------------------------------------------------------- int8
static inline double round_half_away(double y) { return (y >= 0.0) ? floor(y + 0.5) : ceil(y - 0.5); }

void tf_quantise(const float* v, int8_t codes[kNumParams], float scale32[kNumTensors]) {
  for (int t = 0; t < kNumTensors; ++t) {
    const int off = kTensorOffset[t], len = kTensorSize[t];
    double maxAbs = 0.0;
    for (int i = 0; i < len; ++i) {
      const double a = fabs((double)v[off + i]);
      if (a > maxAbs) maxAbs = a;
    }
    double scale = maxAbs / 127.0;
    if (scale < 1e-7) scale = 1e-7;
    for (int i = 0; i < len; ++i) {
      double q = round_half_away((double)v[off + i] / scale);
      if (q > 127.0) q = 127.0;
      if (q < -127.0) q = -127.0;
      codes[off + i] = (int8_t)q;
    }
    scale32[t] = (float)scale;
  }
}

void tf_dequantise(const int8_t codes[kNumParams], const float scale32[kNumTensors],
                   float out[kNumParams]) {
  for (int t = 0; t < kNumTensors; ++t) {
    const int off = kTensorOffset[t], len = kTensorSize[t];
    const double s = (double)scale32[t];
    for (int i = 0; i < len; ++i) out[off + i] = (float)((double)codes[off + i] * s);
  }
}

}  // namespace tinyfl
