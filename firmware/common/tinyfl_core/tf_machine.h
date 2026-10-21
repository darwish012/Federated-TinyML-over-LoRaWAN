// One emulated machine: card files -> features -> standardisation -> training
// matrix -> (later) a full history replay.  Portable; the file system is
// reached through TfFileSystem.
//
//   m<PP>s<S>.raw   raw int16 streams (read only)
//   m<PP>.chk       reference features for the self-check (read only)
//   m<PP>.x         DERIVED, written by the board: standardisation parameters
//                   and the standardised train matrix (see prepare)
//   m<PP>r<S>.tmp   scratch (raw train features of one stream), deleted
#pragma once

#include "tf_alarm.h"
#include "tf_features.h"
#include "tf_model.h"
#include "tf_rawfile.h"

namespace tinyfl {

struct TfMachineStd {
  float centre[kMaxStreams][kFeat];
  float scale[kMaxStreams][kFeat];
};

struct TfSelfCheck {
  uint16_t rows = 0;      // rows per stream compared (train + calibration)
  float maxAbs = 0.0f;    // max |device - reference| over every value
  float medAbs = 0.0f;    // exact float32 median |device - reference|
  bool compared = false;  // false if there was no usable .chk
};

struct TfMachineInfo {
  uint16_t position = 0;
  uint8_t nStreams = 0;
  uint16_t nTrain = 0;
  uint32_t nAcq = 0;
  TfPartition part = {0, 0, 0, 0};
  TfMachineStd std;
  TfSelfCheck check;
  uint32_t inputCrc = 0; // combined raw header/data CRCs, filled by prepare
};

// File names (buf >= 16 bytes).
void tf_raw_name(char* buf, uint16_t position, uint8_t stream);
void tf_chk_name(char* buf, uint16_t position);
void tf_x_name(char* buf, uint16_t position);

// Computes everything for one machine and writes m<PP>.x:
//   * extracts the 22 raw features of every train and calibration row of every
//     stream (streaming, FFT buffers freed at the end via ex.release()),
//   * compares them with m<PP>.chk (SELF_CHECK),
//   * per-stream median/MAD standardisation from the train rows,
//   * builds x = max over streams of |asinh((f-c)/s)| for the train rows.
// Progress goes through tf_progress("SELF-CHECK", done, total).
bool tf_prepare_machine(TfFileSystem* fs, uint16_t position, TfExtractor& ex, TfMachineInfo* out);

// Loads m<PP>.x (verifying its CRC).  If xRows != nullptr it receives a heap
// block of nTrain*22 floats that the caller frees with tf_free().
bool tf_load_prepared(TfFileSystem* fs, uint16_t position, TfMachineInfo* info, float** xRows);

// One federated round on one machine: trains a copy of `global` (562 float32,
// the dequantised GLOBAL) on the standardised train matrix, quantises the
// delta and writes the 594-byte UPDATE (rev 3.2).  round = training round
// 0..11; baseSlot = round_id slot of the GLOBAL trained from.  Returns the
// train loss.
float tf_build_update(const float* x, int nTrain, const float* global, uint8_t position,
                      uint8_t levelIndex, uint8_t round, uint16_t runId, uint8_t baseSlot,
                      uint8_t* update594);

// ------------------------------------------------------------------ replay
struct TfReplayModel {
  uint8_t levelIndex;
  float w[kNumParams];  // dequantised level model
};

struct TfReplayResult {
  uint32_t index;
  float score[kMaxModelsPerMachine];
  TfAlarmEvent ev[kMaxModelsPerMachine];
  bool badScore;  // a score was not finite and was clamped
};

class TfReplay {
 public:
  ~TfReplay() { close(); }
  // Opens every stream file; models/states are owned by the caller and must
  // outlive the replay.  states[i] belongs to models[i].
  bool begin(TfFileSystem* fs, const TfMachineInfo& info, const TfReplayModel* models, int nModels,
             TfAlarmState* states);
  // Resume support: position at acquisition a (disables the data-CRC check).
  bool seekTo(uint32_t a);
  uint32_t nextIndex() const { return next_; }
  uint32_t total() const { return info_.nAcq; }
  // Scores acquisition nextIndex() with every model and advances.
  bool step(TfReplayResult* out);
  // After the last step: data-CRC verdict of every stream.  *checked is the
  // number of streams whose CRC could be evaluated, *bad the number that failed.
  void finish(int* checked, int* bad);
  void close();
  size_t heapBytes() const { return ex_.heapBytes(); }

 private:
  TfFileSystem* fs_ = nullptr;
  TfMachineInfo info_;
  const TfReplayModel* models_ = nullptr;
  int nModels_ = 0;
  TfAlarmState* states_ = nullptr;
  TfSdReplaySource src_[kMaxStreams];
  TfExtractor ex_;
  uint32_t next_ = 0;
};

}  // namespace tinyfl
