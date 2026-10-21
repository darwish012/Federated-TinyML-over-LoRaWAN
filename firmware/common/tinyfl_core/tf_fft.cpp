#include "tf_fft.h"

#include <stdlib.h>

namespace tinyfl {

static const double kPiFft = 3.14159265358979323846;

// ============================================================ TfSmallFft
// Recursive mixed-radix decimation in time, after the classic kissfft
// structure (BSD-3, Mark Borgerding).  Implemented from scratch here; only the
// well known algorithm is shared.

static inline TfCpx cmul(TfCpx a, TfCpx b) {
  TfCpx r;
  r.re = a.re * b.re - a.im * b.im;
  r.im = a.re * b.im + a.im * b.re;
  return r;
}

bool TfSmallFft::init(int n) {
  release();
  if (n < 1) return false;
  n_ = n;
  tw_ = static_cast<TfCpx*>(tf_alloc(sizeof(TfCpx) * (size_t)n));
  if (!tw_) {
    n_ = 0;
    return false;
  }
  for (int j = 0; j < n; ++j) {
    const double ph = -2.0 * kPiFft * (double)j / (double)n;
    tw_[j].re = (float)cos(ph);
    tw_[j].im = (float)sin(ph);
  }
  // Factorisation: 4s first, then 2, 3, 5, 7, ...
  nfac_ = 0;
  int m = n;
  int p = 4;
  const int floorSqrt = (int)floor(sqrt((double)n));
  int maxRadix = 1;
  while (m > 1) {
    while (m % p) {
      switch (p) {
        case 4: p = 2; break;
        case 2: p = 3; break;
        default: p += 2; break;
      }
      if (p > floorSqrt) p = m;
    }
    m /= p;
    factors_[nfac_++] = p;
    factors_[nfac_++] = m;
    if (p > maxRadix) maxRadix = p;
    if (nfac_ >= 62) break;  // cannot happen for n <= kMaxSub
  }
  scratch_ = static_cast<TfCpx*>(tf_alloc(sizeof(TfCpx) * (size_t)maxRadix));
  if (!scratch_) {
    release();
    return false;
  }
  return true;
}

void TfSmallFft::release() {
  if (tw_) tf_free(tw_);
  if (scratch_) tf_free(scratch_);
  tw_ = nullptr;
  scratch_ = nullptr;
  n_ = 0;
  nfac_ = 0;
}

void TfSmallFft::transform(const TfCpx* in, TfCpx* out) {
  if (n_ == 1) {
    out[0] = in[0];
    return;
  }
  work(out, in, 1, factors_);
}

void TfSmallFft::work(TfCpx* Fout, const TfCpx* f, int fstride, const int* factors) {
  const int p = *factors++;
  const int m = *factors++;
  TfCpx* const Fout_beg = Fout;
  TfCpx* const Fout_end = Fout + p * m;
  if (m == 1) {
    do {
      *Fout = *f;
      f += fstride;
    } while (++Fout != Fout_end);
  } else {
    do {
      work(Fout, f, fstride * p, factors);
      f += fstride;
    } while ((Fout += m) != Fout_end);
  }
  Fout = Fout_beg;

  if (p == 2) {
    TfCpx* F2 = Fout + m;
    for (int k = 0; k < m; ++k) {
      const TfCpx t = cmul(F2[k], tw_[k * fstride]);
      F2[k].re = Fout[k].re - t.re;
      F2[k].im = Fout[k].im - t.im;
      Fout[k].re += t.re;
      Fout[k].im += t.im;
    }
  } else if (p == 4) {
    for (int k = 0; k < m; ++k) {
      const TfCpx s0 = cmul(Fout[k + m], tw_[k * fstride]);
      const TfCpx s1 = cmul(Fout[k + 2 * m], tw_[2 * k * fstride]);
      const TfCpx s2 = cmul(Fout[k + 3 * m], tw_[3 * k * fstride]);
      TfCpx s5;
      s5.re = Fout[k].re - s1.re;
      s5.im = Fout[k].im - s1.im;
      Fout[k].re += s1.re;
      Fout[k].im += s1.im;
      TfCpx s3, s4;
      s3.re = s0.re + s2.re;
      s3.im = s0.im + s2.im;
      s4.re = s0.re - s2.re;
      s4.im = s0.im - s2.im;
      Fout[k + 2 * m].re = Fout[k].re - s3.re;
      Fout[k + 2 * m].im = Fout[k].im - s3.im;
      Fout[k].re += s3.re;
      Fout[k].im += s3.im;
      Fout[k + m].re = s5.re + s4.im;
      Fout[k + m].im = s5.im - s4.re;
      Fout[k + 3 * m].re = s5.re - s4.im;
      Fout[k + 3 * m].im = s5.im + s4.re;
    }
  } else {
    // Generic radix p (3, 5, 53, 151, ...).
    const int N = n_;
    for (int u = 0; u < m; ++u) {
      int k = u;
      for (int q1 = 0; q1 < p; ++q1) {
        scratch_[q1] = Fout[k];
        k += m;
      }
      k = u;
      for (int q1 = 0; q1 < p; ++q1) {
        int twidx = 0;
        TfCpx acc = scratch_[0];
        for (int q = 1; q < p; ++q) {
          twidx += fstride * k;
          if (twidx >= N) twidx -= N;
          const TfCpx t = cmul(scratch_[q], tw_[twidx]);
          acc.re += t.re;
          acc.im += t.im;
        }
        Fout[k] = acc;
        k += m;
      }
    }
  }
}

// ============================================================== the plan
namespace {

struct Plan {
  int M = 0;
  bool direct = false;
  int P = 0, Q = 0;
  TfSmallFft fp, fq;
  TfCpx* cin = nullptr;
  TfCpx* cout_ = nullptr;
  TfCpx* twLo = nullptr;
  TfCpx* twHi = nullptr;
  int B = 0;
  int nHi = 0;
};

Plan g_plan;

void plan_release(Plan& pl) {
  pl.fp.release();
  pl.fq.release();
  if (pl.cin) tf_free(pl.cin);
  if (pl.cout_) tf_free(pl.cout_);
  if (pl.twLo) tf_free(pl.twLo);
  if (pl.twHi) tf_free(pl.twHi);
  pl.cin = pl.cout_ = pl.twLo = pl.twHi = nullptr;
  pl.M = 0;
  pl.P = pl.Q = 0;
  pl.B = pl.nHi = 0;
  pl.direct = false;
}

// Split M = P*Q with P the divisor closest below sqrt(M).
bool split(int M, int* P, int* Q) {
  for (int d = (int)floor(sqrt((double)M)); d >= 2; --d) {
    if (M % d == 0) {
      const int e = M / d;
      if (d <= kMaxSub && e <= kMaxSub) {
        *P = d;
        *Q = e;
        return true;
      }
      return false;
    }
  }
  return false;
}

bool plan_make(Plan& pl, int M) {
  if (pl.M == M) return true;
  plan_release(pl);
  pl.M = M;
  int P = 0, Q = 0;
  if (!split(M, &P, &Q)) {
    pl.direct = true;
    return true;
  }
  pl.P = P;
  pl.Q = Q;
  const int L = P > Q ? P : Q;
  bool ok = pl.fp.init(P) && pl.fq.init(Q);
  pl.cin = static_cast<TfCpx*>(tf_alloc(sizeof(TfCpx) * (size_t)L));
  pl.cout_ = static_cast<TfCpx*>(tf_alloc(sizeof(TfCpx) * (size_t)L));
  int B = (int)ceil(sqrt((double)M));
  int nHi = M / B + 1;
  pl.B = B;
  pl.nHi = nHi;
  pl.twLo = static_cast<TfCpx*>(tf_alloc(sizeof(TfCpx) * (size_t)B));
  pl.twHi = static_cast<TfCpx*>(tf_alloc(sizeof(TfCpx) * (size_t)nHi));
  if (!ok || !pl.cin || !pl.cout_ || !pl.twLo || !pl.twHi) {
    plan_release(pl);
    return false;
  }
  for (int j = 0; j < B; ++j) {
    const double ph = -2.0 * kPiFft * (double)j / (double)M;
    pl.twLo[j].re = (float)cos(ph);
    pl.twLo[j].im = (float)sin(ph);
  }
  for (int j = 0; j < nHi; ++j) {
    const double ph = -2.0 * kPiFft * (double)((long long)B * j) / (double)M;
    pl.twHi[j].re = (float)cos(ph);
    pl.twHi[j].im = (float)sin(ph);
  }
  return true;
}

inline int re_slot(int k1, int P) {
  if (k1 == 0) return 0;
  if ((P & 1) == 0) return (k1 == P / 2) ? 1 : 2 * k1;
  return 2 * k1 - 1;
}
inline int im_slot(int k1, int P) {
  if (k1 == 0) return -1;
  if ((P & 1) == 0) return (k1 == P / 2) ? -1 : 2 * k1 + 1;
  return 2 * k1;
}

inline float power_of(float re, float im) {
  const float h = hypotf(re, im);  // numpy: np.abs(complex64) -> float32 hypot
  return h * h;                    // then ** 2 in float32
}

void four_step(Plan& pl, float* a, int M, float* pw) {
  const int P = pl.P, Q = pl.Q, K = M / 2;
  const int halfP = P / 2;  // last k1 row: P/2 (even P: Nyquist row, odd P: (P-1)/2)

  // ---- stage 1: column transforms, in place, packed real output ----
  for (int n2 = 0; n2 < Q; ++n2) {
    for (int n1 = 0; n1 < P; ++n1) {
      pl.cin[n1].re = a[(size_t)Q * n1 + n2];
      pl.cin[n1].im = 0.0f;
    }
    pl.fp.transform(pl.cin, pl.cout_);
    a[n2] = pl.cout_[0].re;
    for (int k1 = 1; k1 <= halfP; ++k1) {
      a[(size_t)Q * re_slot(k1, P) + n2] = pl.cout_[k1].re;
      const int is = im_slot(k1, P);
      if (is >= 0) a[(size_t)Q * is + n2] = pl.cout_[k1].im;
    }
    if ((n2 & 7) == 7) tf_yield();
  }

  // ---- stage 2: row transforms with twiddle ----
  const bool evenP = (P & 1) == 0;
  for (int k1 = 0; k1 <= halfP; ++k1) {
    const int rs = re_slot(k1, P), is = im_slot(k1, P);
    for (int n2 = 0; n2 < Q; ++n2) {
      float zr = a[(size_t)Q * rs + n2];
      float zi = (is >= 0) ? a[(size_t)Q * is + n2] : 0.0f;
      if (k1 > 0) {
        const int idx = n2 * k1;
        const int hi = idx / pl.B;
        const int lo = idx - hi * pl.B;
        const TfCpx w = cmul(pl.twLo[lo], pl.twHi[hi]);
        const float tr = zr * w.re - zi * w.im;
        const float ti = zr * w.im + zi * w.re;
        zr = tr;
        zi = ti;
      }
      pl.cin[n2].re = zr;
      pl.cin[n2].im = zi;
    }
    pl.fq.transform(pl.cin, pl.cout_);

    const bool selfMirror = (k1 == 0) || (evenP && k1 == P / 2);
    for (int k2 = 0; k2 < Q; ++k2) {
      const long long k = (long long)k1 + (long long)P * k2;
      int b;
      if (selfMirror) {
        if (k < 1 || k > K) continue;
        b = (int)k;
      } else {
        b = (k <= K) ? (int)k : (int)(M - k);
      }
      float power = power_of(pl.cout_[k2].re, pl.cout_[k2].im);
      if (pw) pw[b - 1] = power;
      else a[(size_t)Q * rs + k2] = power;
    }
    tf_yield();
  }
}

// Goertzel bank, 4 bins per pass over the signal, difference forms that stay
// accurate near omega = 0 (A) and omega = pi (B).
void goertzel_bank(const float* a, int M, float* pw) {
  const int K = M / 2;
  for (int k0 = 1; k0 <= K; k0 += 4) {
    const int nb = (K - k0 + 1) < 4 ? (K - k0 + 1) : 4;
    const double w0 = 2.0 * kPiFft * (double)k0 / (double)M;
    const bool variantA = cos(w0) >= 0.0;
    float par[4];
    for (int j = 0; j < 4; ++j) {
      const int kk = k0 + (j < nb ? j : nb - 1);
      const double half = kPiFft * (double)kk / (double)M;  // omega/2
      par[j] = variantA ? (float)(4.0 * sin(half) * sin(half)) : (float)(4.0 * cos(half) * cos(half));
    }
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    const float p0 = par[0], p1 = par[1], p2 = par[2], p3 = par[3];
    if (variantA) {
      for (int n = 0; n < M; ++n) {
        const float x = a[n];
        d0 += x - p0 * s0; s0 += d0;
        d1 += x - p1 * s1; s1 += d1;
        d2 += x - p2 * s2; s2 += d2;
        d3 += x - p3 * s3; s3 += d3;
      }
    } else {
      for (int n = 0; n < M; ++n) {
        const float x = a[n];
        d0 = x + p0 * s0 - d0; s0 = d0 - s0;
        d1 = x + p1 * s1 - d1; s1 = d1 - s1;
        d2 = x + p2 * s2 - d2; s2 = d2 - s2;
        d3 = x + p3 * s3 - d3; s3 = d3 - s3;
      }
    }
    const float ss[4] = {s0, s1, s2, s3};
    const float dd[4] = {d0, d1, d2, d3};
    for (int j = 0; j < nb; ++j) {
      float power;
      if (variantA)
        power = dd[j] * dd[j] + par[j] * ss[j] * (ss[j] - dd[j]);
      else
        power = dd[j] * dd[j] - par[j] * ss[j] * (dd[j] - ss[j]);
      pw[k0 + j - 1] = power > 0.0f ? power : 0.0f;
    }
    tf_yield();
  }
}

}  // namespace

int tf_fft_method(int M, int* P, int* Q) {
  int p = 0, q = 0;
  const bool ok = split(M, &p, &q);
  if (P) *P = p;
  if (Q) *Q = q;
  return ok ? 0 : 1;
}

bool tf_power_spectrum(float* a, int M, float* pw) {
  if (M < 4 || M > kMaxFftSamples || !a) return false;
  if (!plan_make(g_plan, M)) return false;
  if (g_plan.direct)
    { if (!pw) return false; goertzel_bank(a, M, pw); }
  else
    four_step(g_plan, a, M, pw);
  return true;
}

float* tf_packed_power_bin(float* a, int M, int bin) {
  const int P=g_plan.P, Q=g_plan.Q;
  if (!P || bin<1 || bin>M/2 || g_plan.M!=M) return nullptr;
  int k1=bin%P, k2=bin/P;
  if (k1>P/2) { int mirror=M-bin; k1=mirror%P; k2=mirror/P; }
  return a+(size_t)Q*re_slot(k1,P)+k2;
}

void tf_fft_release() { plan_release(g_plan); }

}  // namespace tinyfl
