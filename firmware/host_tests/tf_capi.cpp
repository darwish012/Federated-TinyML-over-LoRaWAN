// C API over the portable core, for ctypes tests (host_tests/test_core.py).
// Built as a Windows DLL with zig c++ (build_host.ps1); nothing here is used by
// the firmware.
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "../common/tinyfl_core/tf_machine.h"
#include "../common/tinyfl_core/tf_protocol.h"
#include "../common/tinyfl_core/tf_replay_v4.h"

using namespace tinyfl;

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif

// ------------------------------------------------------------ stdio file system
namespace {

class StdioFile : public TfFile {
 public:
  explicit StdioFile(FILE* f) : f_(f) {}
  ~StdioFile() override { if (f_) fclose(f_); }
  int64_t read(void* dst, size_t n) override { return (int64_t)fread(dst, 1, n, f_); }
  size_t write(const void* src, size_t n) override { return fwrite(src, 1, n, f_); }
  bool seek(uint64_t pos) override { return _fseeki64(f_, (long long)pos, SEEK_SET) == 0; }
  uint64_t size() override {
    long long cur = _ftelli64(f_);
    _fseeki64(f_, 0, SEEK_END);
    long long e = _ftelli64(f_);
    _fseeki64(f_, cur, SEEK_SET);
    return (uint64_t)e;
  }
  uint64_t position() override { return (uint64_t)_ftelli64(f_); }
  void flush() override { fflush(f_); }

 private:
  FILE* f_;
};

class StdioFs : public TfFileSystem {
 public:
  explicit StdioFs(const char* dir) : dir_(dir) {
    if (!dir_.empty() && dir_.back() != '/' && dir_.back() != '\\') dir_ += "/";
  }
  TfFile* open(const char* name, TfOpenMode mode) override {
    FILE* f = fopen((dir_ + name).c_str(), mode == kOpenRead ? "rb" : (mode == kOpenWrite ? "wb" : "ab"));
    return f ? new StdioFile(f) : nullptr;
  }
  bool exists(const char* name) override {
    FILE* f = fopen((dir_ + name).c_str(), "rb");
    if (f) fclose(f);
    return f != nullptr;
  }
  bool remove(const char* name) override { return ::remove((dir_ + name).c_str()) == 0; }

 private:
  std::string dir_;
};

// One acquisition held in memory.
class MemSource : public TfAcquisitionSource {
 public:
  MemSource(const int16_t* q, const float* scales, int C, int n) : q_(q), s_(scales), C_(C), n_(n) {}
  uint16_t channels() const override { return (uint16_t)C_; }
  uint32_t acquisitionCount() const override { return 1; }
  bool nextAcquisition(TfAcqInfo* info) override {
    if (given_) return false;
    given_ = true;
    info->index = 0;
    info->samples = (uint32_t)n_;
    info->channels = (uint16_t)C_;
    return true;
  }
  bool beginChannel(uint16_t c, float* scale) override {
    cur_ = c;
    pos_ = 0;
    *scale = s_[c];
    return true;
  }
  int readChunk(int16_t* dst, int maxSamples) override {
    int left = n_ - pos_;
    int k = left < maxSamples ? left : maxSamples;
    if (k <= 0) return 0;
    memcpy(dst, q_ + (size_t)cur_ * n_ + pos_, 2u * (size_t)k);
    pos_ += k;
    return k;
  }

 private:
  const int16_t* q_;
  const float* s_;
  int C_, n_, cur_ = 0, pos_ = 0;
  bool given_ = false;
};

TfExtractor* g_ex = nullptr;
TfExtractor& ex() {
  if (!g_ex) g_ex = new TfExtractor();
  return *g_ex;
}
long g_yields = 0;
void countYield(void*) { ++g_yields; }

}  // namespace

// ----------------------------------------------------------------- general
API const char* tfc_last_error() { return tf_last_error(); }
API long tfc_yield_count() { return g_yields; }
API int tfc_is_little_endian() {
  uint32_t x = 1;
  return *reinterpret_cast<uint8_t*>(&x) == 1;
}
API uint32_t tfc_crc32(uint32_t crc, const uint8_t* d, size_t n) { return tf_crc32(crc, d, n); }
API void tfc_init() { tf_set_yield(countYield, nullptr); }
API size_t tfc_heap_current() { return tf_heap_current(); }
API size_t tfc_heap_peak() { return tf_heap_peak(); }
API void tfc_heap_reset_peak() { tf_heap_reset_peak(); }
API int tfc_raw_validate(const char* dir, const char* name) {
  StdioFs fs(dir); TfSdReplaySource src; return src.open(&fs, name);
}
API int tfc_parse_frame(const uint8_t* f, size_t len, uint8_t* header, uint8_t* data) {
  TfFrameHeader h; const uint8_t* body;
  if (!tf_parse_frame(f, len, &h, &body)) return 0;
  header[0]=h.type; header[1]=h.roundId; header[2]=h.index; header[3]=h.count; header[4]=h.bodyLen;
  memcpy(data, body, h.bodyLen); return 1;
}

// ------------------------------------------------------------------ features
API int tfc_channel_descriptors(const int16_t* q, int n, float scale, float* out11) {
  MemSource src(q, &scale, 1, n);
  TfAcqInfo info;
  src.nextAcquisition(&info);
  float s;
  src.beginChannel(0, &s);
  return ex().channelDescriptors(src, (uint32_t)n, s, out11) ? 1 : 0;
}
API int tfc_extract(const int16_t* q, const float* scales, int C, int n, float* out22) {
  MemSource src(q, scales, C, n);
  return ex().extract(src, out22) ? 1 : 0;
}
API size_t tfc_extractor_heap() { return ex().heapBytes(); }
API void tfc_extractor_release() { ex().release(); }

// power spectrum of a real signal (input copied)
API int tfc_fft_power(const float* a, int M, float* pw) {
  float* tmp = (float*)malloc(sizeof(float) * (size_t)M);
  memcpy(tmp, a, sizeof(float) * (size_t)M);
  bool ok = tf_power_spectrum(tmp, M, pw);
  free(tmp);
  return ok ? 1 : 0;
}
API int tfc_fft_method(int M) { return tf_fft_method(M, nullptr, nullptr); }

// -------------------------------------------------------------------- model
API int tfc_std_params(const float* rows, int n, float* centre, float* scale) {
  return tf_std_params(rows, n, centre, scale) ? 1 : 0;
}
API void tfc_std_apply(const float* raw, const float* c, const float* s, float* out) {
  tf_std_apply(raw, c, s, out);
}
API void tfc_local_train(const float* x, int n, float* p) { tf_local_train(x, n, p); }
API float tfc_mean_loss(const float* p, const float* x, int n) { return tf_mean_loss(p, x, n); }
API void tfc_score(const float* p, const float* x, int n, float* out) {
  for (int i = 0; i < n; ++i) out[i] = tf_score_row(p, x + (size_t)i * kFeat);
}
API void tfc_quantise(const float* v, int8_t* codes, float* scale32) { tf_quantise(v, codes, scale32); }
API void tfc_dequantise(const int8_t* codes, const float* scale32, float* out) {
  tf_dequantise(codes, scale32, out);
}

// -------------------------------------------------------------------- alarm
// flags[i], stat[i]; thr[0] = threshold (NaN if never ready); latched[0] = first alarm or -1
API int tfc_alarm(const float* scores, int n, uint32_t trainStart, uint32_t trainCount,
                  uint32_t calStart, uint32_t calCount, uint8_t* flags, float* stat, double* thr,
                  int64_t* firstAlarm, int* latchedEvents) {
  TfPartition part = {trainStart, trainCount, calStart, calCount};
  TfAlarmState st;
  tf_alarm_init(&st);
  *firstAlarm = -1;
  *latchedEvents = 0;
  for (int i = 0; i < n; ++i) {
    TfAlarmEvent ev = tf_alarm_step(&st, part, (uint32_t)i, scores[i]);
    flags[i] = ev.flag ? 1 : 0;
    stat[i] = ev.stat;
    if (ev.newlyLatched) {
      ++*latchedEvents;
      *firstAlarm = i;
    }
  }
  thr[0] = st.thresholdReady ? st.threshold : NAN;
  return 1;
}

// ----------------------------------------------------------------- protocol
API size_t tfc_encode_update(uint8_t* out, int pos, int level, int round, int run, int base,
                             uint32_t rows, const float* scale, const int8_t* codes, float loss) {
  return tf_encode_update(out, (uint8_t)pos, (uint8_t)level, (uint8_t)round, (uint16_t)run,
                          (uint8_t)base, rows, scale, codes, loss);
}
API size_t tfc_update_fragment(uint8_t* frame, const uint8_t* update, int roundId, int index) {
  return tf_update_fragment(frame, update, (uint8_t)roundId, (uint8_t)index);
}
API size_t tfc_encode_global(uint8_t* out, int level, int round, int run, const float* scale,
                             const int8_t* codes) {
  return tf_encode_global(out, (uint8_t)level, (uint8_t)round, (uint16_t)run, scale, codes);
}
API int tfc_decode_global(const uint8_t* in, int* level, int* round, int* run, float* scale,
                          int8_t* codes) {
  TfGlobalView v;
  if (!tf_decode_global(in, &v)) return 0;
  *level = v.levelIndex;
  *round = v.round;
  *run = v.runId;
  memcpy(scale, v.scale, sizeof(v.scale));
  memcpy(codes, v.codes, kNumParams);
  return 1;
}
API int tfc_decode_update(const uint8_t* in, int* pos, int* level, int* round, int* run, int* base,
                          int* rows, float* scale, int8_t* codes, float* loss) {
  TfUpdateView v;
  if (!tf_decode_update(in, &v)) return 0;
  *pos = v.position; *level = v.levelIndex; *round = v.round; *run = v.runId; *base = v.baseSlot;
  *rows = v.nTrainRows;
  memcpy(scale, v.scale, sizeof(v.scale));
  memcpy(codes, v.codes, kNumParams);
  *loss = v.trainLoss;
  return 1;
}
API size_t tfc_encode_hello(uint8_t* f, int board, int n, int fw, int flags, int run) {
  return tf_encode_hello(f, (uint8_t)board, (uint8_t)n, (uint8_t)fw, (uint8_t)flags, (uint16_t)run);
}
API size_t tfc_encode_alarm(uint8_t* f, int pos, int level, uint32_t acq, float score, float thr) {
  return tf_encode_alarm(f, (uint8_t)pos, (uint8_t)level, acq, score, thr);
}
API size_t tfc_encode_self_check(uint8_t* f, int pos, int rows, float mx, float md) {
  return tf_encode_self_check(f, (uint8_t)pos, (uint16_t)rows, mx, md);
}
API size_t tfc_encode_replay_done(uint8_t* f, int pos, int mask, uint32_t acq, int alarms) {
  return tf_encode_replay_done(f, (uint8_t)pos, (uint8_t)mask, acq, (uint8_t)alarms);
}
API size_t tfc_encode_ready(uint8_t* f, int rid, int dr, int rx2) {
  return tf_encode_ready(f, (uint8_t)rid, (uint8_t)dr, (uint8_t)rx2);
}
API size_t tfc_encode_status(uint8_t* f, int rid, int missing) {
  return tf_encode_status(f, (uint8_t)rid, (uint16_t)missing);
}
API size_t tfc_encode_repair(uint8_t* f,int rid,int pos,int missing) {
  return tf_encode_repair(f,(uint8_t)rid,(uint8_t)pos,(uint16_t)missing);
}
API size_t tfc_global_fragment(uint8_t* f,const uint8_t* g,int rid,int index) {
  return tf_global_fragment(f,g,(uint8_t)rid,(uint8_t)index);
}
API int tfc_parse_repair(const uint8_t* f, size_t len, int* rid, int* mask, int* pos) {
  uint8_t r, p;
  uint16_t m;
  if (!tf_parse_repair(f, len, &r, &m, &p)) return 0;
  *rid = r; *mask = m; *pos = p;
  return 1;
}
// reassembler: feed frames, returns Result; payload copied out when complete
API int tfc_reassemble(const uint8_t* frames, const int* lens, int nFrames, uint8_t* payload,
                       int* results) {
  static TfGlobalReassembler ra;
  ra.reset();
  int off = 0, last = -1;
  for (int i = 0; i < nFrames; ++i) {
    last = (int)ra.add(frames + off, (size_t)lens[i]);
    results[i] = last;
    off += lens[i];
  }
  if (ra.mask() == 7) memcpy(payload, ra.payload(), kGlobalBytes);
  return ra.mask();
}

// ------------------------------------------------------------ machine pipeline
API int tfc_prepare(const char* dir, int pos, int* nStreams, int* nTrain, float* check3,
                    float* centre, float* scale) {
  StdioFs fs(dir);
  TfMachineInfo info;
  if (!tf_prepare_machine(&fs, (uint16_t)pos, ex(), &info)) return 0;
  *nStreams = info.nStreams;
  *nTrain = info.nTrain;
  check3[0] = (float)info.check.rows;
  check3[1] = info.check.maxAbs;
  check3[2] = info.check.medAbs;
  for (int s = 0; s < info.nStreams; ++s)
    for (int j = 0; j < kFeat; ++j) {
      centre[s * kFeat + j] = info.std.centre[s][j];
      scale[s * kFeat + j] = info.std.scale[s][j];
    }
  return 1;
}
API int tfc_load_x(const char* dir, int pos, float* x, int capRows, int* nTrain) {
  StdioFs fs(dir);
  TfMachineInfo info;
  float* X = nullptr;
  if (!tf_load_prepared(&fs, (uint16_t)pos, &info, &X)) return 0;
  *nTrain = info.nTrain;
  if (info.nTrain <= capRows) memcpy(x, X, sizeof(float) * (size_t)info.nTrain * kFeat);
  tf_free(X);
  return info.nTrain <= capRows ? 1 : 0;
}
API int tfc_build_update(const float* x, int n, const float* global, int pos, int level, int round,
                         int run, int base, uint8_t* out594, float* loss) {
  *loss = tf_build_update(x, n, global, (uint8_t)pos, (uint8_t)level, (uint8_t)round, (uint16_t)run,
                          (uint8_t)base, out594);
  return 1;
}

// Replays a whole machine.  scores/flags/stats: nAcq*nModels (model-major inside an
// acquisition).  thr: nModels doubles, firstAlarm: nModels int64.
API int tfc_replay(const char* dir, int pos, int nModels, const float* models, const int* levels,
                   int nAcqCap, float* scores, uint8_t* flags, float* stats, double* thr,
                   int64_t* firstAlarm, int* crcChecked, int* crcBad, int* nAcqOut) {
  StdioFs fs(dir);
  TfMachineInfo info;
  if (!tf_load_prepared(&fs, (uint16_t)pos, &info, nullptr)) return 0;
  TfReplayModel* rm = new TfReplayModel[nModels];
  TfAlarmState* st = new TfAlarmState[nModels];
  for (int m = 0; m < nModels; ++m) {
    rm[m].levelIndex = (uint8_t)levels[m];
    memcpy(rm[m].w, models + (size_t)m * kNumParams, sizeof(float) * kNumParams);
    tf_alarm_init(&st[m]);
    firstAlarm[m] = -1;
  }
  TfReplay* rp = new TfReplay();
  int ok = 1;
  if (!rp->begin(&fs, info, rm, nModels, st) || (int)info.nAcq > nAcqCap) {
    ok = 0;
  } else {
    TfReplayResult r;
    while (rp->nextIndex() < info.nAcq) {
      if (!rp->step(&r)) { ok = 0; break; }
      for (int m = 0; m < nModels; ++m) {
        const size_t k = (size_t)r.index * nModels + m;
        scores[k] = r.score[m];
        flags[k] = r.ev[m].flag ? 1 : 0;
        stats[k] = r.ev[m].stat;
        if (r.ev[m].newlyLatched) firstAlarm[m] = r.index;
      }
    }
    if (ok) rp->finish(crcChecked, crcBad);
    for (int m = 0; m < nModels; ++m) thr[m] = st[m].thresholdReady ? st[m].threshold : NAN;
  }
  *nAcqOut = (int)info.nAcq;
  delete rp;
  delete[] rm;
  delete[] st;
  return ok;
}

// Same one-pass feature replay used by the production v4 sketch. Model/state
// streaming is exercised without the Arduino FS/radio wrappers.
API int tfv4_replay_features(const char* dir,int pos,int nModels,const float* models,
                            int start,int nAcqCap,float* scores,uint8_t* flags,int* nAcqOut) {
  StdioFs fs(dir); TfMachineInfo info;
  if(!tf_load_prepared(&fs,uint16_t(pos),&info,nullptr) || nModels<1 || nModels>255 ||
     int(info.nAcq)>nAcqCap || start<0 || uint32_t(start)>=info.nAcq) return 0;
  tinyfl::v4::FeatureReplay replay;
  if(!replay.begin(&fs,info,0)) return 0;
  TfAlarmState* states=new TfAlarmState[nModels];
  for(int m=0;m<nModels;m++) tf_alarm_init(states+m);
  int ok=1;
  for(uint32_t a=0;a<info.nAcq;a++) {
    float x[kFeat]; if(!replay.step(x)) { ok=0; break; }
    for(int m=0;m<nModels;m++) {
      float score=tf_score_row(models+size_t(m)*kNumParams,x);
      auto ev=tf_alarm_step(states+m,info.part,a,score);
      scores[size_t(a)*nModels+m]=score; flags[size_t(a)*nModels+m]=ev.flag;
    }
  }
  if(ok && !replay.finish()) ok=0;
  // Exercise the resume-seek path and compare its suffix extraction/scoring.
  if(ok && start) {
    tinyfl::v4::FeatureReplay resumed;
    if(!resumed.begin(&fs,info,start)) ok=0;
    for(uint32_t a=start;ok && a<info.nAcq;a++) {
      float x[kFeat]; if(!resumed.step(x)) { ok=0; break; }
      for(int m=0;m<nModels;m++)
        if(tf_score_row(models+size_t(m)*kNumParams,x)!=scores[size_t(a)*nModels+m]) ok=0;
    }
    if(ok && !resumed.finish()) ok=0;
  }
  *nAcqOut=info.nAcq; delete[] states; return ok;
}
