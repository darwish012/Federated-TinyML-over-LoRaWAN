// SdReplaySource: reads m<PP>s<S>.raw (PROTOCOL_V3 section 5.1) as an
// acquisition source. Open preflights BOTH CRCs, all scales and exact length
// without data-dependent allocations, then rewinds for feature extraction.
// Sequential replay also checks the data CRC again. Also the reader
// for the self-check file m<PP>.chk (section 5.2).
#pragma once

#include "tf_source.h"

namespace tinyfl {

struct TfRawHeader {
  uint16_t version = 0;
  uint32_t headerBytes = 0;
  uint16_t position = 0;
  uint16_t streamIndex = 0;
  uint16_t nStreams = 0;
  uint16_t nChannels = 0;
  uint32_t nAcq = 0;
  uint32_t samplesFixed = 0;   // 0 = the length table applies
  uint32_t part[10] = {0};     // (start,count) x {train, cal, healthy, unhealthy, unscored}

  uint32_t trainStart() const { return part[0]; }
  uint32_t trainCount() const { return part[1]; }
  uint32_t calStart() const { return part[2]; }
  uint32_t calCount() const { return part[3]; }
  uint32_t calEnd() const { return part[2] + part[3] - 1u; }  // last calibration index
};

class TfSdReplaySource : public TfAcquisitionSource {
 public:
  TfSdReplaySource() {}
  ~TfSdReplaySource() override { close(); }

  // Opens and validates: magic, version, header CRC, sizes against the file
  // length.  false on any problem (tf_last_error()).  verifyData=false skips the
  // full data-CRC pass and takes the identity from the stored data CRC trailer
  // (fast boot when the card was already verified on an earlier boot).
  bool open(TfFileSystem* fs, const char* name, bool verifyData = true);
  void close();

  const TfRawHeader& header() const { return h_; }
  bool isOpen() const { return f_ != nullptr; }
  uint32_t identityCrc() const { return identityCrc_; }

  // Samples per channel of acquisition a.
  uint32_t lengthOf(uint32_t a) const;
  // Position so that nextAcquisition() returns acquisition a.  Seeking anywhere
  // but 0 (or the next acquisition) disables the running data-CRC.
  bool seekAcquisition(uint32_t a);
  uint32_t nextIndex() const { return cur_; }

  // After the last acquisition has been consumed strictly in order from 0:
  // reads the trailer and reports the data CRC verdict.  Returns false if the
  // CRC could not be evaluated (seeks happened) -- *ok is then untouched.
  bool finishDataCrc(bool* ok);
  bool dataCrcTracked() const { return crcActive_; }

  // TfAcquisitionSource
  uint16_t channels() const override { return h_.nChannels; }
  uint32_t acquisitionCount() const override { return h_.nAcq; }
  bool nextAcquisition(TfAcqInfo* info) override;
  bool beginChannel(uint16_t c, float* scale) override;
  int readChunk(int16_t* dst, int maxSamples) override;

 private:
  uint64_t recordOffset(uint32_t a) const;
  bool readExact(void* dst, size_t n);

  TfFile* f_ = nullptr;
  TfRawHeader h_;
  uint32_t* lengths_ = nullptr;
  uint64_t dataEnd_ = 0;     // offset of the data CRC trailer
  uint32_t cur_ = 0;         // next acquisition to hand out
  uint64_t filePos_ = 0;     // where the underlying file is positioned
  // current acquisition state
  int32_t curChannel_ = -1;
  uint32_t curLen_ = 0;
  uint32_t remaining_ = 0;   // samples left in the current channel
  bool inAcq_ = false;
  // running data CRC
  uint32_t crc_ = 0;
  bool crcActive_ = true;
  uint32_t identityCrc_ = 0;
};

// ---------------------------------------------------------------- .chk file
class TfChkFile {
 public:
  ~TfChkFile() { close(); }
  // Opens, checks sizes and the whole-file CRC (one sequential pass).
  bool open(TfFileSystem* fs, const char* name);
  void close();
  uint16_t position() const { return position_; }
  uint32_t nRows() const { return nRows_; }      // per stream
  uint16_t nStreams() const { return nStreams_; }
  // 22 floats of (stream, row).
  bool readRow(uint16_t stream, uint32_t row, float out22[kFeat]);

 private:
  TfFile* f_ = nullptr;
  uint16_t position_ = 0;
  uint16_t nStreams_ = 0;
  uint32_t nRows_ = 0;
};

}  // namespace tinyfl
