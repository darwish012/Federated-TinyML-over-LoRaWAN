// PROTOCOL_V3 (rev 3.2) message codec, portable.  The server side lives in
// server/protocol_v3.py; host_tests/test_core.py checks the two byte for byte.
#pragma once

#include "tf_common.h"

namespace tinyfl {

// ---------------------------------------------------------------- framing
constexpr uint8_t kMagic = 0xF1;
constexpr uint8_t kVersion = 0x03;
constexpr uint8_t kFPort = 10;
constexpr size_t kHeaderBytes = 7;
constexpr size_t kFragmentBody = 214;
constexpr size_t kMaxFrame = kHeaderBytes + kFragmentBody;  // 221

enum TfMsgType : uint8_t {
  kMsgUpdateFragment = 0x01,
  kMsgClassCReady = 0x02,
  kMsgGlobalFragment = 0x03,
  kMsgGlobalStatus = 0x04,
  kMsgUpdateRepair = 0x05,
  kMsgBoardHello = 0x06,
  kMsgAlarm = 0x07,
  kMsgSelfCheck = 0x08,
  kMsgReplayDone = 0x09,
};

// rev 3.2 sizes
constexpr size_t kUpdateBytes = 594;
constexpr size_t kGlobalBytes = 586;
constexpr size_t kUpdateFragments = 3;
constexpr size_t kGlobalFragments = 3;
// Every UPDATE fragment body: 1 byte position + <= 213 bytes of the UPDATE.
constexpr size_t kUpdateChunk = 213;
constexpr uint8_t kRoundIdNone = 0xFF;  // HELLO / ALARM / SELF_CHECK / REPLAY_DONE

// UPDATE offsets (rev 3.2 section 3.1)
constexpr size_t kUpPosition = 0, kUpLevel = 1, kUpRound = 2, kUpRunId = 3, kUpBaseSlot = 5,
                 kUpRows = 6, kUpScales = 8, kUpDelta = 24, kUpLoss = 586, kUpCrc = 590;
// GLOBAL offsets (section 3.2)
constexpr size_t kGlLevel = 0, kGlRound = 1, kGlRunId = 2, kGlScales = 4, kGlWeights = 20,
                 kGlCrc = 582;

// round_id = level_index * 13 + g, g = 0..12 (section 2).
inline uint8_t tf_slot(uint8_t levelIndex, uint8_t g) { return (uint8_t)(levelIndex * 13u + g); }

struct TfFrameHeader {
  uint8_t type, roundId, index, count, bodyLen;
};

// Writes the 7 header bytes; returns 7.
size_t tf_write_header(uint8_t* out, uint8_t type, uint8_t roundId, uint8_t index, uint8_t count,
                       uint8_t bodyLen);
// Validates magic/version/length; body points into `frame`.
bool tf_parse_frame(const uint8_t* frame, size_t len, TfFrameHeader* h, const uint8_t** body);

// ------------------------------------------------------------------ UPDATE
// 594 bytes.  `round` = training round 0..11 (header slot g = round + 1);
// base_slot = slot of the GLOBAL trained from; rows saturate at 65535.
size_t tf_encode_update(uint8_t* out, uint8_t position, uint8_t levelIndex, uint8_t round,
                        uint16_t runId, uint8_t baseSlot, uint32_t nTrainRows,
                        const float scale[kNumTensors], const int8_t codes[kNumParams],
                        float trainLoss);
// Fragment `index` (0..2) of a finished UPDATE as a complete radio frame.
size_t tf_update_fragment(uint8_t* frame, const uint8_t update[kUpdateBytes], uint8_t roundId,
                          uint8_t index);

struct TfUpdateView {
  uint8_t position, levelIndex, round, baseSlot;
  uint16_t runId, nTrainRows;
  float scale[kNumTensors];
  int8_t codes[kNumParams];
  float trainLoss;
};
bool tf_decode_update(const uint8_t* in, TfUpdateView* v);  // CRC, -128, scales

// ------------------------------------------------------------------ GLOBAL
// 586 bytes: level_index, round(0..12), u16 run_id, f32 scale[4], i8 w[562], crc
size_t tf_encode_global(uint8_t* out, uint8_t levelIndex, uint8_t round, uint16_t runId,
                        const float scale[kNumTensors], const int8_t codes[kNumParams]);

struct TfGlobalView {
  uint8_t levelIndex, round;
  uint16_t runId;
  float scale[kNumTensors];
  int8_t codes[kNumParams];
  uint32_t crc;
};
bool tf_decode_global(const uint8_t* in, TfGlobalView* v);  // CRC, -128, scales finite > 0

// Collects the three GLOBAL_FRAGMENT frames of one burst.
class TfGlobalReassembler {
 public:
  enum Result { kIgnored, kPartial, kComplete, kDuplicate };
  void reset() {
    mask_ = 0;
    roundId_ = 0xFE;
  }
  // frame = whole application payload.  A fragment whose round_id differs from
  // the burst in progress starts a fresh burst.
  Result add(const uint8_t* frame, size_t len);
  uint8_t mask() const { return mask_; }
  uint8_t roundId() const { return roundId_; }
  const uint8_t* payload() const { return buf_; }  // 586 bytes once complete
  uint16_t missingMask() const { return (uint16_t)(7u & ~mask_); }

 private:
  uint8_t buf_[kGlobalBytes];
  uint8_t mask_ = 0;
  uint8_t roundId_ = 0xFE;
};

// ---------------------------------------------------------- single frames
// Each returns the complete frame length.  No logical CRC (section 3.3).
size_t tf_encode_hello(uint8_t* frame, uint8_t board, uint8_t nMachines, uint8_t fw, uint8_t flags,
                       uint16_t runId);
size_t tf_encode_alarm(uint8_t* frame, uint8_t position, uint8_t levelIndex, uint32_t acq,
                       float score, float threshold);
size_t tf_encode_self_check(uint8_t* frame, uint8_t position, uint16_t rows, float maxAbs,
                            float medAbs);
size_t tf_encode_replay_done(uint8_t* frame, uint8_t position, uint8_t levelMask, uint32_t acqScored,
                             uint8_t alarmsLatched);
size_t tf_encode_ready(uint8_t* frame, uint8_t roundId, uint8_t uplinkDr, uint8_t rx2Dr);
// 2-byte body: bit i set = GLOBAL fragment i NOT held (v2 semantics; 0 = all held).
size_t tf_encode_status(uint8_t* frame, uint8_t roundId, uint16_t missingMask);
// UPDATE_REPAIR: body = u16 mask (bits 0..2) + u8 position.
bool tf_parse_repair(const uint8_t* frame, size_t len, uint8_t* roundId, uint16_t* mask,
                     uint8_t* position);
size_t tf_encode_repair(uint8_t* frame, uint8_t roundId, uint8_t position, uint16_t missingMask);
size_t tf_global_fragment(uint8_t* frame, const uint8_t* global, uint8_t roundId, uint8_t index);

constexpr uint8_t kHelloSdMounted = 0x01;
constexpr uint8_t kHelloSelfCheckOk = 0x02;
constexpr uint8_t kHelloImuSource = 0x04;

}  // namespace tinyfl
