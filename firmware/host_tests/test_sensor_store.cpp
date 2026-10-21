// Host test of the sensor-only build's feature store (common/tinyfl_sensor_store.h):
//   log append/rows/CRC, sensorPrepare -> m<PPPP>.x read back by the unchanged
//   tf_load_prepared, and SensorReplay == tf_std_apply on every stored row.
// Build + run: see run_sensor_store_test.ps1.
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "../common/tinyfl_sensor_store.h"

using namespace tinyfl;

class MemFile : public TfFile {
 public:
  MemFile(std::vector<uint8_t>* d, TfOpenMode mode) : d_(d) {
    if (mode == kOpenWrite) d_->clear();
    pos_ = mode == kOpenAppend ? d_->size() : 0;
  }
  int64_t read(void* dst, size_t n) override {
    size_t k = pos_ >= d_->size() ? 0 : std::min(n, d_->size() - pos_);
    memcpy(dst, d_->data() + pos_, k); pos_ += k; return (int64_t)k;
  }
  size_t write(const void* src, size_t n) override {
    if (d_->size() < pos_ + n) d_->resize(pos_ + n);
    memcpy(d_->data() + pos_, src, n); pos_ += n; return n;
  }
  bool seek(uint64_t p) override { if (p > d_->size()) return false; pos_ = (size_t)p; return true; }
  uint64_t size() override { return d_->size(); }
  uint64_t position() override { return pos_; }
 private:
  std::vector<uint8_t>* d_;
  size_t pos_;
};

class MemFs : public TfFileSystem {
 public:
  TfFile* open(const char* name, TfOpenMode mode) override {
    auto it = files.find(name);
    if (it == files.end()) { if (mode == kOpenRead) return nullptr; it = files.emplace(name, std::vector<uint8_t>()).first; }
    return new MemFile(&it->second, mode);
  }
  bool exists(const char* name) override { return files.count(name) != 0; }
  bool remove(const char* name) override { return files.erase(name) != 0; }
  std::map<std::string, std::vector<uint8_t>> files;
};

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s:%d %s (%s)\n", __FILE__, __LINE__, #cond, tf_last_error()); ++failures; } } while (0)

int main() {
  const uint16_t pos = 0x0007, nTrain = 60, nCal = 30, nLive = 25;
  const uint32_t nAll = nTrain + nCal + nLive;
  MemFs fs;
  std::mt19937 rng(17070);
  std::normal_distribution<float> g(0.0f, 1.0f);
  std::vector<float> rows(nAll * kFeat);
  for (auto& v : rows) v = 3.0f + g(rng);

  for (uint32_t r = 0; r < nAll; ++r) CHECK(sensorLogAppend(&fs, pos, &rows[r * kFeat]));
  CHECK(sensorLogRows(&fs, pos) == nAll);

  TfMachineInfo info;
  CHECK(sensorPrepare(&fs, pos, nTrain, nCal, &info));
  CHECK(info.nStreams == 1 && info.nTrain == nTrain && info.nAcq == (uint32_t)nTrain + nCal);
  CHECK(info.part.trainStart == 0 && info.part.trainCount == nTrain && info.part.calStart == nTrain && info.part.calCount == nCal);
  CHECK(!info.check.compared && info.inputCrc != 0);

  // The prepared file reads back through the unchanged loader, bit-identical to a direct computation.
  float centre[kFeat], scale[kFeat];
  CHECK(tf_std_params(rows.data(), nTrain, centre, scale));
  TfMachineInfo loaded; float* x = nullptr;
  CHECK(tf_load_prepared(&fs, pos, &loaded, &x));
  CHECK(x != nullptr && loaded.nTrain == nTrain && loaded.part.calCount == nCal);
  for (int j = 0; j < kFeat; ++j) CHECK(loaded.std.centre[0][j] == centre[j] && loaded.std.scale[0][j] == scale[j]);
  for (uint32_t r = 0; r < nTrain && x; ++r) {
    float v[kFeat]; tf_std_apply(&rows[r * kFeat], centre, scale, v);
    for (int j = 0; j < kFeat; ++j) CHECK(x[r * kFeat + j] == v[j]);
  }
  if (x) tf_free(x);

  // History identity is stable and covers only the history rows (live rows do not change it).
  uint32_t crc = 0;
  CHECK(sensorHistoryCrc(&fs, pos, nTrain + nCal, &crc) && crc == info.inputCrc);

  // Replay over all rows (history + live), resumed half-way, matches tf_std_apply.
  loaded.nAcq = sensorLogRows(&fs, pos);
  for (uint32_t start : {0u, 50u}) {
    SensorReplay rep;
    CHECK(rep.begin(&fs, loaded, start));
    for (uint32_t r = start; r < nAll; ++r) {
      float got[kFeat], want[kFeat];
      CHECK(rep.step(got));
      tf_std_apply(&rows[r * kFeat], centre, scale, want);
      for (int j = 0; j < kFeat; ++j) CHECK(got[j] == want[j]);
    }
    float extra[kFeat];
    CHECK(!rep.step(extra));
    CHECK(rep.finish());
  }

  // A torn last row (reset mid-append) is not counted; a corrupted row is detected.
  char name[16]; sensorLogName(name, pos);
  fs.files[name].push_back(0xAB);
  CHECK(sensorLogRows(&fs, pos) == nAll);
  fs.files[name][5 * kFeatRowBytes + 3] ^= 0x40;
  CHECK(!sensorHistoryCrc(&fs, pos, nTrain + nCal, &crc));
  float nan[kFeat]; for (auto& v : nan) v = NAN;
  CHECK(!sensorLogAppend(&fs, pos, nan));

  std::printf(failures ? "sensor store: %d FAILED\n" : "sensor store: all checks passed\n", failures);
  return failures ? 1 : 0;
}
