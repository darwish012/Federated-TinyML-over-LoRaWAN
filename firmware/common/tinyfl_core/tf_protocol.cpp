#include "tf_protocol.h"

namespace tinyfl {

size_t tf_write_header(uint8_t* out, uint8_t type, uint8_t roundId, uint8_t index, uint8_t count,
                       uint8_t bodyLen) {
  out[0] = kMagic;
  out[1] = kVersion;
  out[2] = type;
  out[3] = roundId;
  out[4] = index;
  out[5] = count;
  out[6] = bodyLen;
  return kHeaderBytes;
}

bool tf_parse_frame(const uint8_t* frame, size_t len, TfFrameHeader* h, const uint8_t** body) {
  if (len < kHeaderBytes || len > kMaxFrame) return false;
  if (frame[0] != kMagic || frame[1] != kVersion) return false;
  if (frame[6] > kFragmentBody || len != kHeaderBytes + frame[6]) return false;
  if (frame[2] < 1 || frame[2] > 9 || frame[4] >= frame[5]) return false;
  if ((frame[2] == 1 || frame[2] == 3) && frame[5] != 3) return false;
  if (frame[2] >= 6 && frame[5] != 1) return false;
  h->type = frame[2];
  h->roundId = frame[3];
  h->index = frame[4];
  h->count = frame[5];
  h->bodyLen = frame[6];
  *body = frame + kHeaderBytes;
  return true;
}

static bool scales_ok(const float* s) {
  for (int t = 0; t < kNumTensors; ++t)
    if (!(isfinite(s[t]) && s[t] > 0.0f)) return false;
  return true;
}

// ------------------------------------------------------------------ UPDATE
size_t tf_encode_update(uint8_t* out, uint8_t position, uint8_t levelIndex, uint8_t round,
                        uint16_t runId, uint8_t baseSlot, uint32_t nTrainRows,
                        const float scale[kNumTensors], const int8_t codes[kNumParams],
                        float trainLoss) {
  out[kUpPosition] = position;
  out[kUpLevel] = levelIndex;
  out[kUpRound] = round;
  put_u16(out + kUpRunId, runId);
  out[kUpBaseSlot] = baseSlot;
  put_u16(out + kUpRows, nTrainRows > 0xFFFFu ? (uint16_t)0xFFFF : (uint16_t)nTrainRows);
  for (int t = 0; t < kNumTensors; ++t) put_f32(out + kUpScales + 4 * t, scale[t]);
  memcpy(out + kUpDelta, codes, kNumParams);
  put_f32(out + kUpLoss, trainLoss);
  put_u32(out + kUpCrc, tf_crc32(0, out, kUpCrc));
  return kUpdateBytes;
}

size_t tf_update_fragment(uint8_t* frame, const uint8_t update[kUpdateBytes], uint8_t roundId,
                          uint8_t index) {
  if (index >= kUpdateFragments) return 0;
  const size_t off = (size_t)index * kUpdateChunk;
  size_t n = kUpdateBytes - off;
  if (n > kUpdateChunk) n = kUpdateChunk;
  tf_write_header(frame, kMsgUpdateFragment, roundId, index, (uint8_t)kUpdateFragments,
                  (uint8_t)(1 + n));
  frame[kHeaderBytes] = update[kUpPosition];  // every fragment is attributable on its own
  memcpy(frame + kHeaderBytes + 1, update + off, n);
  return kHeaderBytes + 1 + n;
}

bool tf_decode_update(const uint8_t* in, TfUpdateView* v) {
  if (get_u32(in + kUpCrc) != tf_crc32(0, in, kUpCrc)) return false;
  v->position = in[kUpPosition];
  v->levelIndex = in[kUpLevel];
  v->round = in[kUpRound];
  v->runId = get_u16(in + kUpRunId);
  v->baseSlot = in[kUpBaseSlot];
  v->nTrainRows = get_u16(in + kUpRows);
  for (int t = 0; t < kNumTensors; ++t) v->scale[t] = get_f32(in + kUpScales + 4 * t);
  memcpy(v->codes, in + kUpDelta, kNumParams);
  v->trainLoss = get_f32(in + kUpLoss);
  if (v->position > 53 || v->levelIndex > 6 || v->round > 11) return false;
  if (!scales_ok(v->scale)) return false;
  for (int i = 0; i < kNumParams; ++i)
    if (v->codes[i] == -128) return false;
  return true;
}

// ------------------------------------------------------------------ GLOBAL
size_t tf_encode_global(uint8_t* out, uint8_t levelIndex, uint8_t round, uint16_t runId,
                        const float scale[kNumTensors], const int8_t codes[kNumParams]) {
  out[kGlLevel] = levelIndex;
  out[kGlRound] = round;
  put_u16(out + kGlRunId, runId);
  for (int t = 0; t < kNumTensors; ++t) put_f32(out + kGlScales + 4 * t, scale[t]);
  memcpy(out + kGlWeights, codes, kNumParams);
  put_u32(out + kGlCrc, tf_crc32(0, out, kGlCrc));
  return kGlobalBytes;
}

bool tf_decode_global(const uint8_t* in, TfGlobalView* v) {
  const uint32_t crc = tf_crc32(0, in, kGlCrc);
  if (get_u32(in + kGlCrc) != crc) return false;
  v->levelIndex = in[kGlLevel];
  v->round = in[kGlRound];
  v->runId = get_u16(in + kGlRunId);
  for (int t = 0; t < kNumTensors; ++t) v->scale[t] = get_f32(in + kGlScales + 4 * t);
  memcpy(v->codes, in + kGlWeights, kNumParams);
  v->crc = crc;
  if (v->levelIndex > 6 || v->round > 12) return false;
  if (!scales_ok(v->scale)) return false;
  for (int i = 0; i < kNumParams; ++i)
    if (v->codes[i] == -128) return false;
  return true;
}

TfGlobalReassembler::Result TfGlobalReassembler::add(const uint8_t* frame, size_t len) {
  TfFrameHeader h;
  const uint8_t* body;
  if (!tf_parse_frame(frame, len, &h, &body)) return kIgnored;
  if (h.type != kMsgGlobalFragment || h.count != kGlobalFragments || h.index >= kGlobalFragments)
    return kIgnored;
  const size_t off = (size_t)h.index * kFragmentBody;
  size_t want = kGlobalBytes - off;
  if (want > kFragmentBody) want = kFragmentBody;
  if (h.bodyLen != want) return kIgnored;
  if (h.roundId != roundId_) {  // a different burst: start over
    mask_ = 0;
    roundId_ = h.roundId;
  }
  const uint8_t bit = (uint8_t)(1u << h.index);
  const bool dup = (mask_ & bit) != 0;
  if (dup && memcmp(buf_ + off, body, want) != 0) return kIgnored;
  if (!dup) memcpy(buf_ + off, body, want);  // a duplicate never overwrites
  mask_ |= bit;
  if (mask_ == 7) return dup ? kDuplicate : kComplete;
  return dup ? kDuplicate : kPartial;
}

// ---------------------------------------------------------- single frames
size_t tf_encode_hello(uint8_t* frame, uint8_t board, uint8_t nMachines, uint8_t fw, uint8_t flags,
                       uint16_t runId) {
  tf_write_header(frame, kMsgBoardHello, kRoundIdNone, 0, 1, 6);
  frame[7] = board;
  frame[8] = nMachines;
  frame[9] = fw;
  frame[10] = flags;
  put_u16(frame + 11, runId);
  return kHeaderBytes + 6;
}

size_t tf_encode_alarm(uint8_t* frame, uint8_t position, uint8_t levelIndex, uint32_t acq,
                       float score, float threshold) {
  tf_write_header(frame, kMsgAlarm, kRoundIdNone, 0, 1, 14);
  frame[7] = position;
  frame[8] = levelIndex;
  put_u32(frame + 9, acq);
  put_f32(frame + 13, score);
  put_f32(frame + 17, threshold);
  return kHeaderBytes + 14;
}

size_t tf_encode_self_check(uint8_t* frame, uint8_t position, uint16_t rows, float maxAbs,
                            float medAbs) {
  tf_write_header(frame, kMsgSelfCheck, kRoundIdNone, 0, 1, 11);
  frame[7] = position;
  put_u16(frame + 8, rows);
  put_f32(frame + 10, maxAbs);
  put_f32(frame + 14, medAbs);
  return kHeaderBytes + 11;
}

size_t tf_encode_replay_done(uint8_t* frame, uint8_t position, uint8_t levelMask, uint32_t acqScored,
                             uint8_t alarmsLatched) {
  tf_write_header(frame, kMsgReplayDone, kRoundIdNone, 0, 1, 7);
  frame[7] = position;
  frame[8] = levelMask;
  put_u32(frame + 9, acqScored);
  frame[13] = alarmsLatched;
  return kHeaderBytes + 7;
}

size_t tf_encode_ready(uint8_t* frame, uint8_t roundId, uint8_t uplinkDr, uint8_t rx2Dr) {
  tf_write_header(frame, kMsgClassCReady, roundId, 0, (uint8_t)kGlobalFragments, 2);
  frame[7] = uplinkDr;
  frame[8] = rx2Dr;
  return kHeaderBytes + 2;
}

size_t tf_encode_status(uint8_t* frame, uint8_t roundId, uint16_t missingMask) {
  if (missingMask & ~7u) return 0;
  tf_write_header(frame, kMsgGlobalStatus, roundId, 0, (uint8_t)kGlobalFragments, 2);
  put_u16(frame + 7, (uint16_t)(missingMask & 7u));
  return kHeaderBytes + 2;
}

bool tf_parse_repair(const uint8_t* frame, size_t len, uint8_t* roundId, uint16_t* mask,
                     uint8_t* position) {
  TfFrameHeader h;
  const uint8_t* body;
  if (!tf_parse_frame(frame, len, &h, &body)) return false;
  if (h.type != kMsgUpdateRepair || h.bodyLen != 3) return false;
  const uint16_t m = get_u16(body);
  if (!m || (m & ~7u) || body[2] > 53 || h.index != 0 || h.count != 3) return false;
  *roundId = h.roundId;
  *mask = m;
  *position = body[2];
  return true;
}

size_t tf_encode_repair(uint8_t* frame,uint8_t roundId,uint8_t position,uint16_t mask) {
  if(!mask || (mask&~7u) || position>53) return 0;
  tf_write_header(frame,kMsgUpdateRepair,roundId,0,3,3);
  put_u16(frame+7,mask); frame[9]=position; return 10;
}
size_t tf_global_fragment(uint8_t* frame,const uint8_t* global,uint8_t roundId,uint8_t index) {
  if(index>=3) return 0;
  size_t off=index*kFragmentBody, n=kGlobalBytes-off;
  if(n>kFragmentBody) n=kFragmentBody;
  tf_write_header(frame,kMsgGlobalFragment,roundId,index,3,(uint8_t)n);
  memcpy(frame+7,global+off,n); return 7+n;
}

}  // namespace tinyfl
