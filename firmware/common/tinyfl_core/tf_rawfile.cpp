#include "tf_rawfile.h"

namespace tinyfl {
#ifdef TINYFL_SD_V4
static constexpr uint16_t kMaxRawPosition=65534;
#else
static constexpr uint16_t kMaxRawPosition=53;
#endif

// ============================================================ SdReplaySource
static const uint32_t kFixedHeader = 66;  // bytes before the optional length table

bool TfSdReplaySource::readExact(void* dst, size_t n) {
  int64_t got = f_->read(dst, n);
  if (got < 0 || (size_t)got != n) return false;
  filePos_ += n;
  return true;
}

void TfSdReplaySource::close() {
  if (f_) delete f_;
  f_ = nullptr;
  if (lengths_) tf_free(lengths_);
  lengths_ = nullptr;
  inAcq_ = false;
  curChannel_ = -1;
}

bool TfSdReplaySource::open(TfFileSystem* fs, const char* name, bool verifyData) {
  tf_progress("sd_read",0,1);
  close();
  f_ = fs->open(name, kOpenRead);
  if (!f_) {
    tf_set_error("cannot open %s", name);
    return false;
  }
  uint8_t fx[kFixedHeader];
  filePos_ = 0;
  if (!readExact(fx, sizeof(fx))) {
    tf_set_error("%s: short header", name);
    close();
    return false;
  }
  if (memcmp(fx, "TFR1", 4) != 0) {
    tf_set_error("%s: bad magic", name);
    close();
    return false;
  }
  h_.version = get_u16(fx + 4);
  h_.headerBytes = get_u32(fx + 6);
  h_.position = get_u16(fx + 10);
  h_.streamIndex = get_u16(fx + 12);
  h_.nStreams = get_u16(fx + 14);
  h_.nChannels = get_u16(fx + 16);
  h_.nAcq = get_u32(fx + 18);
  h_.samplesFixed = get_u32(fx + 22);
  for (int i = 0; i < 10; ++i) h_.part[i] = get_u32(fx + 26 + 4 * i);
  if (h_.version != 1 || h_.nChannels < 1 || h_.nChannels > kMaxChannels || h_.nStreams < 1 ||
      h_.nStreams > kMaxStreams || h_.nAcq == 0 || h_.position > kMaxRawPosition || h_.streamIndex >= h_.nStreams) {
    tf_set_error("%s: implausible header", name);
    close();
    return false;
  }
  // Rev 3.2 bounds, all checked BEFORE anything is allocated from the header.
  if (h_.nAcq > 100000u) {
    tf_set_error("%s: %u acquisitions is implausible", name, (unsigned)h_.nAcq);
    close();
    return false;
  }
  const uint32_t expect = kFixedHeader + 4 + (h_.samplesFixed == 0 ? 4u * h_.nAcq : 0u);
  if (h_.headerBytes != expect || f_->size() < h_.headerBytes + 4u) {
    tf_set_error("%s: header_bytes %u != %u", name, (unsigned)h_.headerBytes, (unsigned)expect);
    close();
    return false;
  }
  if (h_.samplesFixed != 0 && h_.samplesFixed < 512) {
    tf_set_error("%s: samples_fixed %u", name, (unsigned)h_.samplesFixed);
    close();
    return false;
  }
  {
    // partitions: contiguous, in order, covering 0..n-1; empty is (0,0)
    uint64_t cursor = 0;
    bool okp = true;
    for (int i = 0; i < 5; ++i) {
      const uint32_t st = h_.part[2 * i], ct = h_.part[2 * i + 1];
      if (ct == 0) {
        if (st != 0) okp = false;
      } else {
        if (st != cursor) okp = false;
        cursor += ct;
      }
    }
    if (!okp || cursor != h_.nAcq) {
      tf_set_error("%s: partitions are not contiguous/complete", name);
      close();
      return false;
    }
  }
  uint32_t crc = tf_crc32(0, fx, sizeof(fx));
  uint64_t total = h_.headerBytes;
  if (h_.samplesFixed)
    total += (uint64_t)h_.nAcq * h_.nChannels * (4 + 2ull * h_.samplesFixed);
  if (h_.samplesFixed == 0) {
    // Keep the table in RAM when it is small (<= 4096 acquisitions = 16 KB): lengthOf() then
    // never seeks back to the header. Those per-acquisition back-and-forth seeks made FatFs
    // re-walk the cluster chain each time and were the only difference between the failing
    // preflight of card 02 and a clean straight-through read of the same file. Larger tables
    // are still read from SD on demand, so a malicious header cannot exhaust RAM.
    if (h_.nAcq <= 4096) lengths_ = static_cast<uint32_t*>(tf_alloc(4u * h_.nAcq));
    uint8_t tmp[256];
    uint32_t done = 0;
    while (done < h_.nAcq) {
      uint32_t n = h_.nAcq - done;
      if (n > 64) n = 64;
      if (!readExact(tmp, 4 * n)) {
        tf_set_error("%s: short length table", name);
        close();
        return false;
      }
      crc = tf_crc32(crc, tmp, 4 * n);
      for (uint32_t i = 0; i < n; ++i) {
        uint32_t length = get_u32(tmp + 4 * i);
        if (length < 512) { tf_set_error("%s: short variable acquisition", name); close(); return false; }
        if (lengths_) lengths_[done + i] = length;
        total += (uint64_t)h_.nChannels * (4 + 2ull * length);
      }
      done += n;
      tf_yield();
    }
  }
  uint8_t tail[4];
  if (!readExact(tail, 4)) {
    tf_set_error("%s: short header CRC", name);
    close();
    return false;
  }
  if (get_u32(tail) != crc) {
    tf_set_error("%s: header CRC mismatch", name);
    close();
    return false;
  }
  uint32_t headerCrc = crc;
  // File size must be exactly header + records + 4-byte data CRC.
  dataEnd_ = total;
  total += 4;
  if (f_->size() != total) {
    tf_set_error("%s: size %llu != expected %llu", name, (unsigned long long)f_->size(),
                 (unsigned long long)total);
    close();
    return false;
  }
  // Full preflight before any feature/FFT/train allocation: CRC every data byte
  // and check EVERY scale (not just the training prefix). Payload stays on SD.
  uint8_t block[1024];
  uint32_t dataCrc = 0;
  if (!verifyData) {
    if (!f_->seek(dataEnd_) || !readExact(tail, 4) || !f_->seek(h_.headerBytes)) {
      tf_set_error("%s: cannot read data CRC trailer", name); close(); return false;
    }
    dataCrc = get_u32(tail);
  } else
  for (uint32_t a = 0; a < h_.nAcq; ++a) {
    uint32_t length = lengthOf(a);
    if (!length) { close(); return false; }
    for (uint16_t c = 0; c < h_.nChannels; ++c) {
      if (!readExact(tail, 4) || !isfinite(get_f32(tail)) || get_f32(tail) <= 0) {
        tf_set_error("%s: invalid channel scale", name); close(); return false;
      }
      dataCrc = tf_crc32(dataCrc, tail, 4);
      uint64_t left = 2ull * length;
      while (left) {
        size_t n = left > sizeof(block) ? sizeof(block) : (size_t)left;
        if (!readExact(block, n)) {
          tf_set_error("%s: short data at acquisition %lu of %lu (byte %llu)", name, (unsigned long)a,
                       (unsigned long)h_.nAcq, (unsigned long long)filePos_);
          close(); return false;
        }
        dataCrc = tf_crc32(dataCrc, block, n);
        left -= n;
        tf_yield();
      }
    }
  }
  if (verifyData && (!readExact(tail, 4) || get_u32(tail) != dataCrc || !f_->seek(h_.headerBytes))) {
    tf_set_error("%s: data CRC mismatch", name); close(); return false;
  }
  uint8_t identity[8]; put_u32(identity,headerCrc); put_u32(identity+4,dataCrc);
  identityCrc_=tf_crc32(0,identity,sizeof(identity));
  cur_ = 0;
  inAcq_ = false;
  crc_ = 0;
  crcActive_ = true;
  filePos_ = h_.headerBytes;  // header fully consumed
  return true;
}

uint32_t TfSdReplaySource::lengthOf(uint32_t a) const {
  if (!f_ || a >= h_.nAcq) return 0;
  if (h_.samplesFixed) return h_.samplesFixed;
  if (lengths_) return lengths_[a];
  uint8_t b[4];
  uint64_t saved = f_->position();
  bool ok = f_->seek(kFixedHeader + 4ull * a) && f_->read(b, 4) == 4;
  ok = f_->seek(saved) && ok;
  if (!ok) { tf_set_error("length table read failed"); return 0; }
  return get_u32(b);
}

uint64_t TfSdReplaySource::recordOffset(uint32_t a) const {
  if (h_.samplesFixed)
    return h_.headerBytes + (uint64_t)a * h_.nChannels * (4 + 2ull * h_.samplesFixed);
  uint64_t off = h_.headerBytes;
  for (uint32_t i = 0; i < a; ++i) off += (uint64_t)h_.nChannels * (4 + 2ull * lengthOf(i));
  return off;
}

bool TfSdReplaySource::seekAcquisition(uint32_t a) {
  if (!f_ || a > h_.nAcq) return false;
  if (a != cur_ || inAcq_) crcActive_ = false;  // not a straight sequential read any more
  if (!f_->seek(recordOffset(a))) return false;
  filePos_ = recordOffset(a);
  cur_ = a;
  inAcq_ = false;
  curChannel_ = -1;
  return true;
}

bool TfSdReplaySource::nextAcquisition(TfAcqInfo* info) {
  if (!f_ || cur_ >= h_.nAcq) return false;
  if (inAcq_) {
    // The caller abandoned the previous acquisition part-way: re-synchronise.
    if (!seekAcquisition(cur_)) return false;
  }
  info->index = cur_;
  info->samples = lengthOf(cur_);
  info->channels = h_.nChannels;
  curLen_ = info->samples;
  curChannel_ = -1;
  inAcq_ = true;
  return true;
}

bool TfSdReplaySource::beginChannel(uint16_t c, float* scale) {
  if (!f_ || !inAcq_ || (int32_t)c != curChannel_ + 1 || c >= h_.nChannels) return false;
  // Previous channel must have been consumed completely.
  if (curChannel_ >= 0 && remaining_ != 0) return false;
  uint8_t b[4];
  if (!readExact(b, 4)) return false;
  if (crcActive_) crc_ = tf_crc32(crc_, b, 4);
  *scale = get_f32(b);
  if (!(isfinite(*scale) && *scale > 0.0f)) {
    tf_set_error("acquisition %u channel %u: scale %g is not finite and positive", (unsigned)cur_,
                 (unsigned)c, (double)*scale);
    return false;
  }
  curChannel_ = c;
  remaining_ = curLen_;
  return true;
}

int TfSdReplaySource::readChunk(int16_t* dst, int maxSamples) {
  if (!f_ || !inAcq_ || curChannel_ < 0 || maxSamples <= 0 || !dst) return -1;
  if (remaining_ == 0) {
    if ((uint32_t)curChannel_ + 1 == h_.nChannels) {  // last channel of the record done
      inAcq_ = false;
      ++cur_;
    }
    return 0;
  }
  int n = maxSamples;
  if ((uint32_t)n > remaining_) n = (int)remaining_;
  if (!readExact(dst, 2u * (size_t)n)) return -1;
  if (crcActive_) crc_ = tf_crc32(crc_, dst, 2u * (size_t)n);
  remaining_ -= (uint32_t)n;
  return n;
}

bool TfSdReplaySource::finishDataCrc(bool* ok) {
  if (!f_ || !crcActive_ || cur_ != h_.nAcq || inAcq_) return false;
  // The last readChunk that returned 0 may not have happened if the caller
  // stopped exactly at the channel end; the position tells us either way.
  if (filePos_ != dataEnd_) return false;
  uint8_t b[4];
  if (!readExact(b, 4)) return false;
  *ok = (get_u32(b) == crc_);
  return true;
}

// ================================================================== TfChkFile
void TfChkFile::close() {
  if (f_) delete f_;
  f_ = nullptr;
}

bool TfChkFile::open(TfFileSystem* fs, const char* name) {
  close();
  f_ = fs->open(name, kOpenRead);
  if (!f_) {
    tf_set_error("cannot open %s", name);
    return false;
  }
  uint8_t h[16];
  if (f_->read(h, 16) != 16 || memcmp(h, "TFC1", 4) != 0 || get_u16(h + 4) != 1) {
    tf_set_error("%s: bad check-file header", name);
    close();
    return false;
  }
  position_ = get_u16(h + 6);
  nRows_ = get_u32(h + 8);
  nStreams_ = get_u16(h + 12);
  if (position_ > kMaxRawPosition || !nRows_ || nRows_ > 100000 || !nStreams_ ||
      nStreams_ > kMaxStreams || get_u16(h + 14) != 0) {
    tf_set_error("%s: invalid check dimensions/reserved", name); close(); return false;
  }
  const uint64_t want = 16ull + 4ull * kFeat * nRows_ * nStreams_ + 4ull;
  if (f_->size() != want) {
    tf_set_error("%s: size %llu != %llu", name, (unsigned long long)f_->size(),
                 (unsigned long long)want);
    close();
    return false;
  }
  // whole-file CRC in one sequential pass
  uint32_t crc = tf_crc32(0, h, 16);
  uint64_t left = want - 16 - 4;
  uint8_t buf[512];
  while (left) {
    size_t n = left > sizeof(buf) ? sizeof(buf) : (size_t)left;
    if (f_->read(buf, n) != (int64_t)n) {
      tf_set_error("%s: short read", name);
      close();
      return false;
    }
    crc = tf_crc32(crc, buf, n);
    left -= n;
    tf_yield();
  }
  uint8_t t[4];
  if (f_->read(t, 4) != 4 || get_u32(t) != crc) {
    tf_set_error("%s: CRC mismatch", name);
    close();
    return false;
  }
  return true;
}

bool TfChkFile::readRow(uint16_t stream, uint32_t row, float out22[kFeat]) {
  if (!f_ || stream >= nStreams_ || row >= nRows_) return false;
  const uint64_t off = 16ull + 4ull * kFeat * ((uint64_t)stream * nRows_ + row);
  if (!f_->seek(off)) return false;
  uint8_t b[4 * kFeat];
  if (f_->read(b, sizeof(b)) != (int64_t)sizeof(b)) return false;
  for (int i = 0; i < kFeat; ++i) {
    out22[i] = get_f32(b + 4 * i);
    if (!isfinite(out22[i])) { tf_set_error("nonfinite check feature"); return false; }
  }
  return true;
}

}  // namespace tinyfl
