// TinyFL core -- shared constants, hooks and small helpers.
//
// The whole of common/tinyfl_core is portable C++ with NO Arduino dependency,
// so that it can be compiled on a host (host_tests/) and checked number for
// number against the Python twin before anything is flashed.  The ESP32 sketch
// pulls the same sources in through tinyfl_core_all.cpp.
#pragma once

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace tinyfl {

// ---------------------------------------------------------------- model shape
constexpr int kFeat = 22;    // input features == output size
constexpr int kHidden = 12;  // H12
constexpr int kNumTensors = 4;
constexpr int kTensorSize[kNumTensors] = {kFeat * kHidden, kHidden, kHidden * kFeat, kFeat};
constexpr int kTensorOffset[kNumTensors] = {0, 264, 276, 540};
constexpr int kNumParams = 562;  // 264 + 12 + 264 + 22

// ------------------------------------------------------------- data limits
constexpr int kMaxStreams = 4;      // ims machines have four streams
constexpr int kMaxChannels = 6;     // hit has six channels
constexpr int kMaxTrainRows = 997;  // fixed hardware-card RAM budget
constexpr int kMaxFftSamples = 16384;
constexpr int kChunkSamples = 2048; // int16 samples per SD read
constexpr int kMaxLevels = 14;
constexpr int kMaxModelsPerMachine = 7;  // hardware levels 4..28

// Alarm layer: dense_flags(scores, train, cal, window=21, k=13, multiplier=2.0)
constexpr int kAlarmWindow = 21;
constexpr int kAlarmK = 13;
constexpr double kAlarmMultiplier = 2.0;

// Local training: fl.local_train semantics, float32.
constexpr int kLocalEpochs = 2;
constexpr float kLearningRate = 0.03f;
constexpr float kL2 = 1e-4f;

// ------------------------------------------------------------------ hooks
// Long computations call tf_yield() often.  The ESP32 sketch points it at the
// watchdog feed and (during replay) at os_runloop_once() so LMIC keeps running;
// the host leaves it empty.
typedef void (*TfYieldFn)(void* ctx);
void tf_set_yield(TfYieldFn fn, void* ctx);
void tf_yield();

// Progress reporting for the OLED: stage is a short static string.
typedef void (*TfProgressFn)(void* ctx, const char* stage, uint32_t done, uint32_t total);
void tf_set_progress(TfProgressFn fn, void* ctx);
void tf_progress(const char* stage, uint32_t done, uint32_t total);

// Heap.  malloc/free on both targets; kept behind two names so the sketch
// could redirect them without touching the algorithms.
void* tf_alloc(size_t bytes);
void tf_free(void* p);
size_t tf_heap_current();
size_t tf_heap_peak();
void tf_heap_reset_peak();

// Error text of the most recent failure (static buffer, single-threaded use).
void tf_set_error(const char* fmt, ...);
const char* tf_last_error();

// ------------------------------------------------------------------ CRC-32
// zlib crc32(): pass 0 to start, feed the previous return value to continue.
uint32_t tf_crc32(uint32_t crc, const void* data, size_t length);

// ------------------------------------------------------- little-endian io
inline void put_u16(uint8_t* p, uint16_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
}
inline void put_u32(uint8_t* p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}
inline void put_f32(uint8_t* p, float v) {
  uint32_t u;
  memcpy(&u, &v, 4);
  put_u32(p, u);
}
inline uint16_t get_u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline uint32_t get_u32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline float get_f32(const uint8_t* p) {
  uint32_t u = get_u32(p);
  float v;
  memcpy(&v, &u, 4);
  return v;
}

}  // namespace tinyfl
