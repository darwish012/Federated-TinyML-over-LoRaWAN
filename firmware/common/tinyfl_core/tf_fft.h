// Real-input power spectrum for ANY length M (odd, prime, mixed radix).
//
// Why not kiss_fftr: it needs an even length, and three FFT lengths in the
// hardware cohort are not (kaist::Bearing_1 M=16129 = 127*127, paderborn::K004
// M=16001 prime, K006 M=16006 = 2*53*151).  It also needs a separate output and
// scratch of the same order as the input (~290 KB for M=16384), which the ESP32
// does not have. Four-step powers reuse the input buffer (~M floats + plan).
// The prime/large-factor fallback uses 1.5*M floats.
//
// Method ("four step", exact, not an approximation):
//   M = P*Q, n = Q*n1 + n2, k = k1 + P*k2
//   stage 1: Q column transforms of length P over the strided input, kept
//            IN PLACE in packed real form (a real column has a Hermitian
//            spectrum, so P reals hold all of it);
//   stage 2: for k1 = 0..P/2 one row transform of length Q after the twiddle
//            W_M^(n2*k1); the other half follows from X[M-k] = conj(X[k]).
// The sub-transforms are a small mixed-radix (2, 4, generic) complex FFT.
// When M has no factorisation into two factors <= kMaxSub (M prime, or
// 2*prime), a bank of 4-way interleaved Goertzel filters in the numerically
// stable difference forms is used instead: O(M*K), needs no tables. ESP32
// execution time is NOT measured; benchmark prime-length streams before runs.
//
// Memory: the input buffer (M floats) is transformed in place; the spectrum is
// K = M/2 floats.  The plan (twiddle tables) is a few KB.
#pragma once

#include "tf_common.h"

namespace tinyfl {

struct TfCpx {
  float re, im;
};

constexpr int kMaxSub = 512;  // longest sub-transform of the four-step plan

// Small complex mixed-radix FFT (kiss-style recursive decimation in time).
class TfSmallFft {
 public:
  TfSmallFft() {}
  ~TfSmallFft() { release(); }
  bool init(int n);
  void release();
  int size() const { return n_; }
  // Out of place; `in` and `out` must not overlap.
  void transform(const TfCpx* in, TfCpx* out);

 private:
  void work(TfCpx* out, const TfCpx* in, int fstride, const int* factors);
  int n_ = 0;
  int nfac_ = 0;
  int factors_[64];
  TfCpx* tw_ = nullptr;
  TfCpx* scratch_ = nullptr;
};

// Power spectrum of the real signal a[0..M): pw[b-1] = float32(|X_b|)^2 for
// b = 1..M/2.  `a` is destroyed.  pw must hold M/2 floats.  Returns false only
// if a temporary allocation fails.
bool tf_power_spectrum(float* a, int M, float* pw);
// Four-step only: pw=nullptr retains powers in the transformed input's rows,
// saving the separate M/2-float output allocation. Bins remain independently
// addressable for float32 probability normalization; no approximation.
float* tf_packed_power_bin(float* a, int M, int bin);

// Frees the cached plan (twiddle tables).  Safe to call at any time.
void tf_fft_release();

// Which method M will use: 0 = four step, 1 = Goertzel bank.  For tests/docs.
int tf_fft_method(int M, int* P, int* Q);

}  // namespace tinyfl
