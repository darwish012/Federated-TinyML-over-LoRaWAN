#include "tf_common.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

namespace tinyfl {

// ------------------------------------------------------------------ hooks
static TfYieldFn g_yield = nullptr;
static void* g_yield_ctx = nullptr;
static TfProgressFn g_progress = nullptr;
static void* g_progress_ctx = nullptr;

void tf_set_yield(TfYieldFn fn, void* ctx) {
  g_yield = fn;
  g_yield_ctx = ctx;
}
void tf_yield() {
  if (g_yield) g_yield(g_yield_ctx);
}
void tf_set_progress(TfProgressFn fn, void* ctx) {
  g_progress = fn;
  g_progress_ctx = ctx;
}
void tf_progress(const char* stage, uint32_t done, uint32_t total) {
  if (g_progress) g_progress(g_progress_ctx, stage, done, total);
}

// Aligned bookkeeping records ONLY core allocations (not LMIC/FS/OLED).
union AllocHead { size_t bytes; long double align; void* ptr; };
static size_t g_heap = 0, g_peak = 0;
void* tf_alloc(size_t bytes) {
  if (bytes > SIZE_MAX - sizeof(AllocHead)) return nullptr;
  AllocHead* h = static_cast<AllocHead*>(malloc(bytes + sizeof(AllocHead)));
  if (!h) return nullptr;
  h->bytes = bytes;
  g_heap += bytes;
  if (g_heap > g_peak) g_peak = g_heap;
  return h + 1;
}
void tf_free(void* p) {
  if (!p) return;
  AllocHead* h = static_cast<AllocHead*>(p) - 1;
  g_heap -= h->bytes;
  free(h);
}
size_t tf_heap_current() { return g_heap; }
size_t tf_heap_peak() { return g_peak; }
void tf_heap_reset_peak() { g_peak = g_heap; }

// ------------------------------------------------------------------ errors
static char g_error[96] = "";

void tf_set_error(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_error, sizeof(g_error), fmt, ap);
  va_end(ap);
}
const char* tf_last_error() { return g_error; }

// ------------------------------------------------------------------ CRC-32
// Standard reflected CRC-32 (poly 0xEDB88320), the value zlib.crc32 returns.
static uint32_t g_crc_table[256];
static bool g_crc_ready = false;

static void crc_init() {
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    g_crc_table[i] = c;
  }
  g_crc_ready = true;
}

uint32_t tf_crc32(uint32_t crc, const void* data, size_t length) {
  if (!g_crc_ready) crc_init();
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint32_t c = ~crc;
  for (size_t i = 0; i < length; ++i) c = g_crc_table[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
  return ~c;
}

}  // namespace tinyfl
