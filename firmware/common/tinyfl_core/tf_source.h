// Acquisition sources and the file abstraction the core reads through.
//
// Two implementations of TfAcquisitionSource exist:
//   * SdReplaySource (tf_rawfile.h): reads m<PP>s<S>.raw, PROTOCOL_V3 section 5.
//   * ImuSource (common/tinyfl_imu_source.h): a stub for a live 16-bit IMU.
// The extractor only ever sees this interface, so swapping SD replay for a real
// sensor changes nothing downstream.
#pragma once

#include "tf_common.h"

namespace tinyfl {

// ---------------------------------------------------------------- file layer
enum TfOpenMode { kOpenRead = 0, kOpenWrite = 1, kOpenAppend = 2 };

class TfFile {
 public:
  virtual ~TfFile() {}
  virtual int64_t read(void* dst, size_t n) = 0;           // bytes read, <0 on error
  virtual size_t write(const void* src, size_t n) = 0;     // bytes written
  virtual bool seek(uint64_t pos) = 0;
  virtual uint64_t size() = 0;
  virtual uint64_t position() = 0;
  virtual void flush() {}
};

class TfFileSystem {
 public:
  virtual ~TfFileSystem() {}
  // Root-relative 8.3 name without leading slash, e.g. "m00s0.raw".
  virtual TfFile* open(const char* name, TfOpenMode mode) = 0;  // caller deletes
  virtual bool exists(const char* name) = 0;
  virtual bool remove(const char* name) = 0;
};

// ------------------------------------------------------------ acquisition API
struct TfAcqInfo {
  uint32_t index;     // acquisition index within the machine
  uint32_t samples;   // samples per channel in THIS acquisition
  uint16_t channels;
};

class TfAcquisitionSource {
 public:
  virtual ~TfAcquisitionSource() {}
  virtual uint16_t channels() const = 0;
  // Total acquisitions available, or 0 for an unbounded live source.
  virtual uint32_t acquisitionCount() const = 0;
  // Advance to the next acquisition.  false = end of data or error.
  virtual bool nextAcquisition(TfAcqInfo* info) = 0;
  // Start channel c (channels are visited 0..C-1 in order); returns the float32
  // scale: sample = (float)int16 * scale.
  virtual bool beginChannel(uint16_t c, float* scale) = 0;
  // Up to maxSamples int16 samples of the current channel; 0 at end of channel,
  // negative on error.
  virtual int readChunk(int16_t* dst, int maxSamples) = 0;
};

}  // namespace tinyfl
