#include "tf_machine.h"

#include <stdio.h>

#include "tf_protocol.h"

namespace tinyfl {

#ifdef TINYFL_SD_V4
#define TF_POSITION_FMT "%04x"
#else
#define TF_POSITION_FMT "%02u"
#endif
void tf_raw_name(char* buf, uint16_t position, uint8_t stream) {
  snprintf(buf, 16, "m" TF_POSITION_FMT "s%u.raw", (unsigned)position, (unsigned)stream);
}
void tf_chk_name(char* buf, uint16_t position) { snprintf(buf, 16, "m" TF_POSITION_FMT ".chk", (unsigned)position); }
void tf_x_name(char* buf, uint16_t position) { snprintf(buf, 16, "m" TF_POSITION_FMT ".x", (unsigned)position); }
static void tf_tmp_name(char* buf, uint16_t position, uint8_t stream) {
  snprintf(buf, 16, "m" TF_POSITION_FMT "r%u.tmp", (unsigned)position, (unsigned)stream);
}

// ---------------------------------------------------- self-check statistics
// Exact median without keeping all differences in RAM. Nonnegative finite
// float32 bit patterns have the same order as their unsigned integer values.
// Four radix-selection passes over a derived SD scratch file find both middle
// ranks, requiring only 2 KiB of histograms, independent of stream/row count.
struct DiffHist {
  uint32_t total;
  float maxAbs;
  TfFile* file;
  void init() { memset(this, 0, sizeof(*this)); }
  bool add(float d) {
    if(!isfinite(d) || d<0) return false;
    uint8_t b[4]; put_f32(b,d);
    if(!file || file->write(b,4)!=4) return false;
    ++total;
    if (d > maxAbs) maxAbs = d;
    return true;
  }
  bool median(float* out) {
    if(!file) { tf_set_error("self-check median: differences file did not reopen"); return false; }
    if(!total || file->size()!=4ull*total) {
      tf_set_error("self-check median: file size %lu, expected %lu", (unsigned long)file->size(), (unsigned long)(4ul*total));
      return false;
    }
    uint32_t ranks[2]={(total-1)/2,total/2}, prefixes[2]={}, mask=0;
    for(int byte=3;byte>=0;byte--) {
      uint32_t bins[2][256]={}; uint8_t block[512];
      if(!file->seek(0)) { tf_set_error("self-check median: seek failed"); return false; }
      uint32_t done=0;
      while(done<total) {
        size_t n=(total-done)>128?128:total-done;
        if(file->read(block,4*n)!=(int64_t)(4*n)) {
          tf_set_error("self-check median: short read at value %lu of %lu", (unsigned long)done, (unsigned long)total);
          return false;
        }
        for(size_t j=0;j<n;j++) {
          uint32_t bits=get_u32(block+4*j);
          for(int k=0;k<2;k++) if((bits&mask)==prefixes[k]) bins[k][(bits>>(8*byte))&255]++;
        }
        done+=n; tf_yield();
      }
      for(int k=0;k<2;k++) {
        int digit=0;
        while(digit<256 && ranks[k]>=bins[k][digit]) { ranks[k]-=bins[k][digit]; digit++; }
        if(digit==256) { tf_set_error("self-check median: rank not found"); return false; }
        prefixes[k]|=(uint32_t)digit<<(8*byte);
      }
      mask|=0xffu<<(8*byte);
    }
    float a,b; memcpy(&a,prefixes,4); memcpy(&b,prefixes+1,4);
    *out=(a+b)*0.5f; return true;
  }
};

// ------------------------------------------------------------------ prepare
static const uint32_t kXHeader = 48;

bool tf_prepare_machine(TfFileSystem* fs, uint16_t position, TfExtractor& ex, TfMachineInfo* out) {
  TfSdReplaySource src[kMaxStreams];
  char name[16];
  *out = TfMachineInfo();
  out->position = position;

  // ---- open the streams ------------------------------------------------
  tf_raw_name(name, position, 0);
  if (!src[0].open(fs, name)) return false;
  const TfRawHeader h0 = src[0].header();
  if (h0.position != position || h0.streamIndex != 0) {
    tf_set_error("%s: position %u in header", name, (unsigned)h0.position);
    return false;
  }
  const int S = h0.nStreams;
  if (S < 1 || S > kMaxStreams) {
    tf_set_error("%s: %d streams", name, S);
    return false;
  }
  for (int s = 1; s < S; ++s) {
    tf_raw_name(name, position, (uint8_t)s);
    if (!src[s].open(fs, name)) return false;
    const TfRawHeader& h = src[s].header();
    bool same = h.position == position && h.nAcq == h0.nAcq && h.nStreams == h0.nStreams && h.streamIndex == s;
    for (int i = 0; i < 10; ++i) same = same && h.part[i] == h0.part[i];
    if (!same) {
      tf_set_error("%s: header differs from stream 0", name);
      return false;
    }
  }
  const uint32_t nTrain = h0.trainCount();
  const uint32_t nCal = h0.calCount();
  if (nTrain < 1 || nTrain > (uint32_t)kMaxTrainRows || nCal < 1) {
    tf_set_error("m%02u: %u train / %u cal rows", (unsigned)position, (unsigned)nTrain, (unsigned)nCal);
    return false;
  }
  if (h0.trainStart() + nTrain > h0.calStart() || h0.calStart() + nCal > h0.nAcq) {
    tf_set_error("m%02u: train/calibration ranges overlap or overrun", (unsigned)position);
    return false;
  }
  out->nStreams = (uint8_t)S;
  out->nTrain = (uint16_t)nTrain;
  out->nAcq = h0.nAcq;
  out->part = {h0.trainStart(), nTrain, h0.calStart(), nCal};
  for(int s=0;s<S;s++) { uint8_t b[4]; put_u32(b,src[s].identityCrc()); out->inputCrc=tf_crc32(out->inputCrc,b,4); }

  // ---- self-check reference (optional) -----------------------------------
  TfChkFile chk;
  bool haveChk = false;
  tf_chk_name(name, position);
  if (fs->exists(name)) {
    if (chk.open(fs, name)) {
      if (chk.position() == position && chk.nStreams() == S && chk.nRows() == nTrain + nCal) {
        haveChk = true;
      } else {
        tf_set_error("%s: shape does not match the raw files", name);
      }
    }
  }
  if (!haveChk) { tf_set_error("%s: missing or invalid mandatory self-check", name); return false; }
  DiffHist* histp = static_cast<DiffHist*>(tf_alloc(sizeof(DiffHist)));
  if (!histp) {
    tf_set_error("no heap for the self-check histogram");
    return false;
  }
  histp->init();
  char diffName[16]; snprintf(diffName,sizeof(diffName),"m" TF_POSITION_FMT ".dif",(unsigned)position);
  histp->file=fs->open(diffName,kOpenWrite);
  if(!histp->file) { tf_free(histp); tf_set_error("cannot create self-check differences"); return false; }
  DiffHist& hist = *histp;
  struct HistGuard {
    DiffHist* p;
    ~HistGuard() { if(p->file) delete p->file; tf_free(p); }
  } histGuard = {histp};

  // ---- extraction pass: raw features of train + calibration rows ----------
  const uint32_t rowsPerStream = nTrain + nCal;
  const uint32_t totalRows = rowsPerStream * (uint32_t)S;
  uint32_t doneRows = 0;
  for (int s = 0; s < S; ++s) {
    char tmp[16];
    tf_tmp_name(tmp, position, (uint8_t)s);
    TfFile* scratch = fs->open(tmp, kOpenWrite);
    if (!scratch) {
      tf_set_error("cannot create %s", tmp);
      return false;
    }
    for (uint32_t r = 0; r < rowsPerStream; ++r) {
      const uint32_t idx = (r < nTrain) ? h0.trainStart() + r : h0.calStart() + (r - nTrain);
      if (src[s].nextIndex() != idx && !src[s].seekAcquisition(idx)) {
        delete scratch;
        tf_set_error("seek to acquisition %u failed", (unsigned)idx);
        return false;
      }
      float row[kFeat];
      if (!ex.extract(src[s], row)) {
        delete scratch;
        return false;
      }
      if (r < nTrain) {
        uint8_t b[4 * kFeat];
        for (int j = 0; j < kFeat; ++j) put_f32(b + 4 * j, row[j]);
        if (scratch->write(b, sizeof(b)) != sizeof(b)) {
          delete scratch;
          tf_set_error("scratch write failed");
          return false;
        }
      }
      if (haveChk) {
        float ref[kFeat];
        if (!chk.readRow((uint16_t)s, r, ref)) {
          delete scratch;
          tf_set_error("check-file read failed");
          return false;
        }
        for (int j = 0; j < kFeat; ++j) if(!hist.add(fabsf(row[j] - ref[j]))) {
          delete scratch; tf_set_error("self-check difference write failed"); return false;
        }
      }
      tf_progress("SELF-CHECK", ++doneRows, totalRows);
    }
    delete scratch;
  }
  ex.release();
  for (int s = 0; s < S; ++s) src[s].close();
  chk.close();
  if (haveChk) {
    out->check.compared = true;
    out->check.rows = (uint16_t)(rowsPerStream > 65535u ? 65535u : rowsPerStream);
    out->check.maxAbs = hist.maxAbs;
    hist.file->flush(); delete hist.file; hist.file=fs->open(diffName,kOpenRead);
    if(!hist.median(&out->check.medAbs)) return false;  // median() sets the detailed error
    delete hist.file; hist.file=nullptr; fs->remove(diffName);
  }

  // ---- standardisation parameters, one stream at a time ------------------
  float* R = static_cast<float*>(tf_alloc(sizeof(float) * (size_t)nTrain * kFeat));
  if (!R) {
    tf_set_error("no heap for the train features");
    return false;
  }
  for (int s = 0; s < S; ++s) {
    char tmp[16];
    tf_tmp_name(tmp, position, (uint8_t)s);
    TfFile* f = fs->open(tmp, kOpenRead);
    if (!f) {
      tf_free(R);
      tf_set_error("cannot reopen %s", tmp);
      return false;
    }
    const size_t bytes = sizeof(float) * (size_t)nTrain * kFeat;
    // reads are little-endian floats: same layout as memory on both targets
    if (f->read(R, bytes) != (int64_t)bytes) {
      delete f;
      tf_free(R);
      tf_set_error("%s: short read", tmp);
      return false;
    }
    delete f;
    if (!tf_std_params(R, (int)nTrain, out->std.centre[s], out->std.scale[s])) {
      tf_free(R);
      return false;
    }
  }
  tf_free(R);

  // ---- training matrix: x = max over streams of the standardised rows ----
  float* X = static_cast<float*>(tf_alloc(sizeof(float) * (size_t)nTrain * kFeat));
  if (!X) {
    tf_set_error("no heap for the training matrix");
    return false;
  }
  for (int s = 0; s < S; ++s) {
    char tmp[16];
    tf_tmp_name(tmp, position, (uint8_t)s);
    TfFile* f = fs->open(tmp, kOpenRead);
    if (!f) {
      tf_free(X);
      tf_set_error("cannot reopen %s", tmp);
      return false;
    }
    for (uint32_t r = 0; r < nTrain; ++r) {
      uint8_t b[4 * kFeat];
      if (f->read(b, sizeof(b)) != (int64_t)sizeof(b)) {
        delete f;
        tf_free(X);
        tf_set_error("%s: short read", tmp);
        return false;
      }
      float raw[kFeat], v[kFeat];
      for (int j = 0; j < kFeat; ++j) raw[j] = get_f32(b + 4 * j);
      tf_std_apply(raw, out->std.centre[s], out->std.scale[s], v);
      float* xr = X + (size_t)r * kFeat;
      if (s == 0) {
        for (int j = 0; j < kFeat; ++j) xr[j] = v[j];
      } else {
        for (int j = 0; j < kFeat; ++j)
          if (v[j] > xr[j]) xr[j] = v[j];
      }
    }
    delete f;
    tf_yield();
  }
  for (int s = 0; s < S; ++s) {
    char tmp[16];
    tf_tmp_name(tmp, position, (uint8_t)s);
    fs->remove(tmp);
  }

  // ---- write m<PP>.x -----------------------------------------------------
  tf_x_name(name, position);
  TfFile* xf = fs->open(name, kOpenWrite);
  if (!xf) {
    tf_free(X);
    tf_set_error("cannot create %s", name);
    return false;
  }
  uint8_t head[kXHeader];
  memset(head, 0, sizeof(head));
  memcpy(head, "TFX1", 4);
  put_u16(head + 4, 1);
  put_u16(head + 6, position);
  put_u16(head + 8, (uint16_t)S);
  put_u16(head + 10, (uint16_t)nTrain);
  put_u32(head + 12, out->nAcq);
  put_u32(head + 16, out->part.trainStart);
  put_u32(head + 20, out->part.trainCount);
  put_u32(head + 24, out->part.calStart);
  put_u32(head + 28, out->part.calCount);
  put_u16(head + 32, out->check.rows);
  put_f32(head + 36, out->check.maxAbs);
  put_f32(head + 40, out->check.medAbs);
  put_u32(head + 44, out->check.compared ? 1u : 0u);
  uint32_t crc = tf_crc32(0, head, sizeof(head));
  bool ok = xf->write(head, sizeof(head)) == sizeof(head);
  for (int s = 0; s < S && ok; ++s) {
    uint8_t b[2 * 4 * kFeat];
    for (int j = 0; j < kFeat; ++j) {
      put_f32(b + 4 * j, out->std.centre[s][j]);
      put_f32(b + 4 * kFeat + 4 * j, out->std.scale[s][j]);
    }
    crc = tf_crc32(crc, b, sizeof(b));
    ok = xf->write(b, sizeof(b)) == sizeof(b);
  }
  for (uint32_t r = 0; r < nTrain && ok; ++r) {
    uint8_t b[4 * kFeat];
    for (int j = 0; j < kFeat; ++j) put_f32(b + 4 * j, X[(size_t)r * kFeat + j]);
    crc = tf_crc32(crc, b, sizeof(b));
    ok = xf->write(b, sizeof(b)) == sizeof(b);
  }
  if (ok) {
    uint8_t t[4];
    put_u32(t, crc);
    ok = xf->write(t, 4) == 4;
  }
  xf->flush();
  delete xf;
  tf_free(X);
  if (!ok) {
    tf_set_error("write of %s failed", name);
    return false;
  }
  return true;
}

bool tf_load_prepared(TfFileSystem* fs, uint16_t position, TfMachineInfo* info, float** xRows) {
  char name[16];
  tf_x_name(name, position);
  TfFile* f = fs->open(name, kOpenRead);
  if (!f) {
    tf_set_error("no %s", name);
    return false;
  }
  uint8_t head[kXHeader];
  if (f->read(head, sizeof(head)) != (int64_t)sizeof(head) || memcmp(head, "TFX1", 4) != 0 ||
      get_u16(head + 4) != 1 || get_u16(head + 6) != position) {
    delete f;
    tf_set_error("%s: bad header", name);
    return false;
  }
  *info = TfMachineInfo();
  info->position = position;
  const int S = get_u16(head + 8);
  const uint32_t nTrain = get_u16(head + 10);
  if (S < 1 || S > kMaxStreams || nTrain < 1 || nTrain > (uint32_t)kMaxTrainRows) {
    delete f;
    tf_set_error("%s: implausible header", name);
    return false;
  }
  info->nStreams = (uint8_t)S;
  info->nTrain = (uint16_t)nTrain;
  info->nAcq = get_u32(head + 12);
  info->part = {get_u32(head + 16), get_u32(head + 20), get_u32(head + 24), get_u32(head + 28)};
  info->check.rows = get_u16(head + 32);
  info->check.maxAbs = get_f32(head + 36);
  info->check.medAbs = get_f32(head + 40);
  info->check.compared = get_u32(head + 44) != 0;
  const uint64_t want = kXHeader + (uint64_t)S * 2 * 4 * kFeat + (uint64_t)nTrain * 4 * kFeat + 4;
  if (f->size() != want) {
    delete f;
    tf_set_error("%s: size mismatch", name);
    return false;
  }
  // Validate the derived file CRC and all floats before allocating a matrix.
  uint32_t preCrc = tf_crc32(0, head, sizeof(head));
  uint64_t preLeft = want - sizeof(head) - 4;
  uint8_t preBlock[512];
  while (preLeft) {
    size_t n = preLeft > sizeof(preBlock) ? sizeof(preBlock) : (size_t)preLeft;
    if (f->read(preBlock, n) != (int64_t)n) { delete f; return false; }
    preCrc = tf_crc32(preCrc, preBlock, n);
    for (size_t i = 0; i < n; i += 4)
      if (!isfinite(get_f32(preBlock + i))) { delete f; tf_set_error("nonfinite prepared feature"); return false; }
    preLeft -= n;
    tf_yield();
  }
  uint8_t preTail[4];
  if (f->read(preTail, 4) != 4 || get_u32(preTail) != preCrc || !f->seek(sizeof(head))) {
    delete f; tf_set_error("%s: prepared CRC mismatch", name); return false;
  }
  uint32_t crc = tf_crc32(0, head, sizeof(head));
  for (int s = 0; s < S; ++s) {
    uint8_t b[2 * 4 * kFeat];
    if (f->read(b, sizeof(b)) != (int64_t)sizeof(b)) {
      delete f;
      tf_set_error("%s: short read", name);
      return false;
    }
    crc = tf_crc32(crc, b, sizeof(b));
    for (int j = 0; j < kFeat; ++j) {
      info->std.centre[s][j] = get_f32(b + 4 * j);
      info->std.scale[s][j] = get_f32(b + 4 * kFeat + 4 * j);
      if (info->std.scale[s][j] <= 0) { delete f; tf_set_error("invalid prepared scale"); return false; }
    }
  }
  float* X = nullptr;
  if (xRows) {
    X = static_cast<float*>(tf_alloc(sizeof(float) * (size_t)nTrain * kFeat));
    if (!X) {
      delete f;
      tf_set_error("no heap for the training matrix");
      return false;
    }
  }
  for (uint32_t r = 0; r < nTrain; ++r) {
    uint8_t b[4 * kFeat];
    if (f->read(b, sizeof(b)) != (int64_t)sizeof(b)) {
      delete f;
      if (X) tf_free(X);
      tf_set_error("%s: short read", name);
      return false;
    }
    crc = tf_crc32(crc, b, sizeof(b));
    if (X)
      for (int j = 0; j < kFeat; ++j) X[(size_t)r * kFeat + j] = get_f32(b + 4 * j);
  }
  uint8_t t[4];
  const bool crcOk = f->read(t, 4) == 4 && get_u32(t) == crc;
  delete f;
  if (!crcOk) {
    if (X) tf_free(X);
    tf_set_error("%s: CRC mismatch", name);
    return false;
  }
  if (xRows) *xRows = X;
  return true;
}

// ------------------------------------------------------------------- update
float tf_build_update(const float* x, int nTrain, const float* global, uint8_t position,
                      uint8_t levelIndex, uint8_t round, uint16_t runId, uint8_t baseSlot,
                      uint8_t* update594) {
  float local[kNumParams];
  memcpy(local, global, sizeof(local));
  tf_local_train(x, nTrain, local);
  const float loss = tf_mean_loss(local, x, nTrain);
  for (int i = 0; i < kNumParams; ++i) local[i] -= global[i];
  int8_t codes[kNumParams];
  float scale[kNumTensors];
  tf_quantise(local, codes, scale);
  tf_encode_update(update594, position, levelIndex, round, runId, baseSlot, (uint32_t)nTrain, scale,
                   codes, loss);
  return loss;
}

// ------------------------------------------------------------------- replay
bool TfReplay::begin(TfFileSystem* fs, const TfMachineInfo& info, const TfReplayModel* models,
                     int nModels, TfAlarmState* states) {
  close();
  fs_ = fs;
  info_ = info;
  models_ = models;
  nModels_ = nModels;
  states_ = states;
  if (nModels < 1 || nModels > kMaxModelsPerMachine) {
    tf_set_error("replay needs 1..%d models", kMaxModelsPerMachine);
    return false;
  }
  for (int s = 0; s < info.nStreams; ++s) {
    char name[16];
    tf_raw_name(name, info.position, (uint8_t)s);
    if (!src_[s].open(fs, name)) return false;
    const auto& h=src_[s].header();
    if (h.position!=info.position || h.streamIndex!=s || h.nStreams!=info.nStreams ||
        h.nAcq != info.nAcq || h.trainStart()!=info.part.trainStart || h.trainCount()!=info.part.trainCount ||
        h.calStart()!=info.part.calStart || h.calCount()!=info.part.calCount) {
      tf_set_error("%s: acquisition count changed since prepare", name);
      return false;
    }
  }
  next_ = 0;
  return true;
}

bool TfReplay::seekTo(uint32_t a) {
  for (int s = 0; s < info_.nStreams; ++s)
    if (!src_[s].seekAcquisition(a)) return false;
  next_ = a;
  return true;
}

void TfReplay::close() {
  for (int s = 0; s < kMaxStreams; ++s) src_[s].close();
  ex_.release();
}

bool TfReplay::step(TfReplayResult* out) {
  if (next_ >= info_.nAcq) {
    tf_set_error("replay finished");
    return false;
  }
  float x[kFeat];
  for (int s = 0; s < info_.nStreams; ++s) {
    float raw[kFeat], v[kFeat];
    if (!ex_.extract(src_[s], raw)) return false;
    tf_std_apply(raw, info_.std.centre[s], info_.std.scale[s], v);
    if (s == 0) {
      for (int j = 0; j < kFeat; ++j) x[j] = v[j];
    } else {
      for (int j = 0; j < kFeat; ++j)
        if (v[j] > x[j]) x[j] = v[j];
    }
  }
  out->index = next_;
  tf_progress("replay",next_,info_.nAcq);
  out->badScore = false;
  for (int m = 0; m < nModels_; ++m) {
    float sc = tf_score_row(models_[m].w, x);
    if (!isfinite(sc) || sc < 0.0f) {
      sc = 3.4028234663852886e38f;  // FLT_MAX: a broken row counts as anomalous
      out->badScore = true;
    }
    out->score[m] = sc;
    out->ev[m] = tf_alarm_step(&states_[m], info_.part, next_, sc);
  }
  ++next_;
  return true;
}

void TfReplay::finish(int* checked, int* bad) {
  *checked = 0;
  *bad = 0;
  for (int s = 0; s < info_.nStreams; ++s) {
    bool ok = false;
    if (src_[s].finishDataCrc(&ok)) {
      ++*checked;
      if (!ok) ++*bad;
    }
  }
}

}  // namespace tinyfl
