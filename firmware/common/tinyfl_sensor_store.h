#pragma once
// Feature store of the sensor-only build (TINYFL_SENSOR): the live counterpart of the SD
// card's raw files, kept in LittleFS through the same TfFileSystem.
//
//   m<PPPP>.feat   append-only log, one row per acquisition: 22 raw float32 features
//                  (little-endian) + CRC32 of those 88 bytes = 92 bytes. Rows 0..nTrain-1
//                  are the train history, the next nCal rows the calibration history (the
//                  same partition the SD cards encode), later rows are live monitoring.
//   m<PPPP>.x      written by sensorPrepare() in exactly the TFX1 format of
//                  tf_prepare_machine() (one stream), so tf_load_prepared() and training
//                  are unchanged.
//
// Raw samples are never stored: an acquisition is reduced to its 22 features as soon as it
// is captured, as a deployed sensor would. Standardisation and the training matrix follow
// tf_prepare_machine() line by line for one stream.
#include "tinyfl_core/tf_machine.h"

namespace tinyfl {

static constexpr uint32_t kFeatRowBytes = 4 * kFeat + 4;

inline void sensorLogName(char* buf, uint16_t position) { snprintf(buf, 16, "m%04x.feat", (unsigned)position); }

inline bool sensorLogAppend(TfFileSystem* fs, uint16_t position, const float raw[kFeat]) {
  char name[16]; sensorLogName(name, position);
  uint8_t b[kFeatRowBytes];
  for (int j = 0; j < kFeat; ++j) {
    if (!isfinite(raw[j])) { tf_set_error("nonfinite live feature"); return false; }
    put_f32(b + 4 * j, raw[j]);
  }
  put_u32(b + 4 * kFeat, tf_crc32(0, b, 4 * kFeat));
  TfFile* f = fs->open(name, kOpenAppend);
  if (!f) { tf_set_error("cannot append %s", name); return false; }
  bool ok = f->write(b, sizeof(b)) == sizeof(b);
  f->flush(); delete f;
  if (!ok) tf_set_error("%s: append failed", name);
  return ok;
}

// Complete rows in the log (a torn last row from a reset mid-write is not counted).
inline uint32_t sensorLogRows(TfFileSystem* fs, uint16_t position) {
  char name[16]; sensorLogName(name, position);
  if (!fs->exists(name)) return 0;
  TfFile* f = fs->open(name, kOpenRead);
  if (!f) return 0;
  uint64_t n = f->size() / kFeatRowBytes; delete f;
  return (uint32_t)n;
}

inline bool sensorLogReadRow(TfFile* f, float raw[kFeat], uint32_t* crcAcc = nullptr) {
  uint8_t b[kFeatRowBytes];
  if (f->read(b, sizeof(b)) != (int64_t)sizeof(b)) { tf_set_error("feature log: short read"); return false; }
  if (get_u32(b + 4 * kFeat) != tf_crc32(0, b, 4 * kFeat)) { tf_set_error("feature log: row CRC mismatch"); return false; }
  for (int j = 0; j < kFeat; ++j) raw[j] = get_f32(b + 4 * j);
  if (crcAcc) *crcAcc = tf_crc32(*crcAcc, b, sizeof(b));
  return true;
}

// Identity of the history (train + calibration rows), the sensor build's inputCrc.
inline bool sensorHistoryCrc(TfFileSystem* fs, uint16_t position, uint32_t rows, uint32_t* crc) {
  char name[16]; sensorLogName(name, position);
  TfFile* f = fs->open(name, kOpenRead);
  if (!f) { tf_set_error("no %s", name); return false; }
  *crc = 0; float raw[kFeat]; bool ok = true;
  for (uint32_t r = 0; r < rows && ok; ++r) { ok = sensorLogReadRow(f, raw, crc); if ((r & 15) == 15) tf_yield(); }
  delete f; return ok;
}

// History rows 0..nTrain+nCal-1 must already be in the log. Writes m<PPPP>.x.
inline bool sensorPrepare(TfFileSystem* fs, uint16_t position, uint16_t nTrain, uint16_t nCal, TfMachineInfo* out) {
  *out = TfMachineInfo();
  out->position = position; out->nStreams = 1; out->nTrain = nTrain; out->nAcq = (uint32_t)nTrain + nCal;
  out->part = {0, nTrain, nTrain, nCal};
  out->check.rows = (uint16_t)(nTrain + nCal);   // check.compared stays false: no reference file
  if (nTrain < 1 || nTrain > (uint16_t)kMaxTrainRows || nCal < 1) { tf_set_error("sensor history %u/%u", nTrain, nCal); return false; }
  if (sensorLogRows(fs, position) < out->nAcq) { tf_set_error("sensor history incomplete"); return false; }
  if (!sensorHistoryCrc(fs, position, out->nAcq, &out->inputCrc)) return false;

  float* R = static_cast<float*>(tf_alloc(sizeof(float) * (size_t)nTrain * kFeat));
  if (!R) { tf_set_error("no heap for the train features"); return false; }
  char name[16]; sensorLogName(name, position);
  TfFile* f = fs->open(name, kOpenRead);
  bool ok = f != nullptr;
  for (uint32_t r = 0; r < nTrain && ok; ++r) ok = sensorLogReadRow(f, R + (size_t)r * kFeat);
  delete f;
  ok = ok && tf_std_params(R, (int)nTrain, out->std.centre[0], out->std.scale[0]);
  if (!ok) { tf_free(R); return false; }
  for (uint32_t r = 0; r < nTrain; ++r) {           // in place: x = |asinh((f - c) / s)|
    float v[kFeat]; tf_std_apply(R + (size_t)r * kFeat, out->std.centre[0], out->std.scale[0], v);
    memcpy(R + (size_t)r * kFeat, v, sizeof(v));
  }

  tf_x_name(name, position);
  TfFile* xf = fs->open(name, kOpenWrite);
  if (!xf) { tf_free(R); tf_set_error("cannot create %s", name); return false; }
  uint8_t head[48];
  memset(head, 0, sizeof(head));
  memcpy(head, "TFX1", 4);
  put_u16(head + 4, 1);
  put_u16(head + 6, position);
  put_u16(head + 8, 1);
  put_u16(head + 10, nTrain);
  put_u32(head + 12, out->nAcq);
  put_u32(head + 16, out->part.trainStart);
  put_u32(head + 20, out->part.trainCount);
  put_u32(head + 24, out->part.calStart);
  put_u32(head + 28, out->part.calCount);
  put_u16(head + 32, out->check.rows);
  put_f32(head + 36, out->check.maxAbs);
  put_f32(head + 40, out->check.medAbs);
  put_u32(head + 44, 0u);
  uint32_t crc = tf_crc32(0, head, sizeof(head));
  ok = xf->write(head, sizeof(head)) == sizeof(head);
  if (ok) {
    uint8_t b[2 * 4 * kFeat];
    for (int j = 0; j < kFeat; ++j) {
      put_f32(b + 4 * j, out->std.centre[0][j]);
      put_f32(b + 4 * kFeat + 4 * j, out->std.scale[0][j]);
    }
    crc = tf_crc32(crc, b, sizeof(b));
    ok = xf->write(b, sizeof(b)) == sizeof(b);
  }
  for (uint32_t r = 0; r < nTrain && ok; ++r) {
    uint8_t b[4 * kFeat];
    for (int j = 0; j < kFeat; ++j) put_f32(b + 4 * j, R[(size_t)r * kFeat + j]);
    crc = tf_crc32(crc, b, sizeof(b));
    ok = xf->write(b, sizeof(b)) == sizeof(b);
  }
  if (ok) { uint8_t t[4]; put_u32(t, crc); ok = xf->write(t, 4) == 4; }
  xf->flush(); delete xf; tf_free(R);
  if (!ok) tf_set_error("write of %s failed", name);
  return ok;
}

// Replay over the stored feature rows: same interface as v4::FeatureReplay, no raw data.
class SensorReplay {
 public:
  ~SensorReplay() { delete f_; }
  bool begin(TfFileSystem* fs, const TfMachineInfo& info, uint32_t next) {
    info_ = info; next_ = next;
    char name[16]; sensorLogName(name, info.position);
    delete f_; f_ = fs->open(name, kOpenRead);
    if (!f_) { tf_set_error("no %s", name); return false; }
    if (f_->size() / kFeatRowBytes < info.nAcq || next > info.nAcq) { tf_set_error("feature log shorter than replay"); return false; }
    return f_->seek((uint64_t)next * kFeatRowBytes);
  }
  bool step(float* x) {
    if (next_ >= info_.nAcq) return false;
    float raw[kFeat];
    if (!sensorLogReadRow(f_, raw)) return false;
    tf_std_apply(raw, info_.std.centre[0], info_.std.scale[0], x);
    next_++; return true;
  }
  bool finish() { return true; }   // every row was CRC-checked as it was read
  void releaseFFT() {}
 private:
  TfMachineInfo info_{};
  uint32_t next_ = 0;
  TfFile* f_ = nullptr;
};

}  // namespace tinyfl
