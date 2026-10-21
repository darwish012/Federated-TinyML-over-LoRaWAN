// H12 autoencoder [22,12,22] (tanh), float32: forward, full-batch local
// training, scoring, per-stream standardisation and the int8 codec.
// Port of research/hardware_twin_scaling_20260929/twin.py (which mirrors
// fl_algorithm_comparison_20260929/fl.py).
//
// Flat parameter layout (562 floats), tensor order as on the wire:
//   W1 [22][12] row-major @0, b1 [12] @264, W2 [12][22] row-major @276, b2 [22] @540
#pragma once

#include "tf_common.h"

namespace tinyfl {

// ------------------------------------------------------------------ model
// h = tanh(x W1 + b1); y = h W2 + b2.  h[12], y[22] are outputs.
void tf_forward(const float* p, const float* x, float* h, float* y);

// Reconstruction score of one standardised row: mean((y-x)^2) with numpy's
// float32 pairwise summation order for 22 elements.
float tf_score_row(const float* p, const float* x);

// Mean squared error over ALL elements of n rows (the UPDATE train_loss).
float tf_mean_loss(const float* p, const float* x, int n);

// fl.local_train: kLocalEpochs full-batch epochs, lr 0.03, L2 1e-4 on weight
// matrices only, loss = mean squared error over all elements.  x is n x 22.
// p is updated in place.
void tf_local_train(const float* x, int n, float* p);

// ------------------------------------------------------- standardisation
// Per-feature centre / scale of one stream's train rows (n x 22, row-major):
//   centre = median; scale = max(1.4826*MAD, population std); 1.0 if <= 1e-8.
// All arithmetic float32, sequential in row order, like numpy.
bool tf_std_params(const float* rows, int n, float centre[kFeat], float scale[kFeat]);

// out = |asinh((raw - centre)/scale)| (float32)
void tf_std_apply(const float* raw, const float centre[kFeat], const float scale[kFeat],
                  float out[kFeat]);

// ----------------------------------------------------------------- int8
// Q_DYN_TENSOR: per tensor scale = max(max|x|/127, 1e-7) computed in double on
// the float32 values; q = round-half-away(x/scale) clipped to [-127,127].
// scale32 is what goes on the wire (float32 of the double scale).
void tf_quantise(const float* v, int8_t codes[kNumParams], float scale32[kNumTensors]);

// (float)((double)code * (double)scale32)
void tf_dequantise(const int8_t codes[kNumParams], const float scale32[kNumTensors],
                   float out[kNumParams]);

}  // namespace tinyfl
