#pragma once
#include "tinyfl_core/tf_source.h"

namespace tinyfl {
// Deliberately non-producing stub: no fabricated readings. A concrete driver
// must configure sample rate, signed-int16 range/scales and acquisition size,
// then supply the same channel-major sequential API as the SD reader.
// Wire is shared with SSD1306: SDA=21, SCL=22. Initialise Wire only once in
// setup; avoid address collisions and serialize sensor/OLED bus operations.
class ImuSource : public TfAcquisitionSource {
 public:
  static constexpr int kSda = 21, kScl = 22;
  uint16_t channels() const override { return 0; }
  uint32_t acquisitionCount() const override { return 0; }
  bool nextAcquisition(TfAcqInfo*) override {
    tf_set_error("ImuSource is not provisioned: implement a real sensor driver");
    return false;
  }
  bool beginChannel(uint16_t, float*) override { return false; }
  int readChunk(int16_t*, int) override { return -1; }
};
}
