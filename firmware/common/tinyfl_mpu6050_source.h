#pragma once
// Live MPU6050 acquisition source for the sensor-only power build (TINYFL_SENSOR).
//
// One acquisition = kSamples accelerometer samples at 1 kHz on 3 channels (x, y, z), read
// through the FIFO so no sample is lost while the radio and LMIC are serviced. The chip
// sleeps between acquisitions (5 uA) and only the accelerometer runs during one (gyros in
// standby, 500 uA): the power build measures a realistic low-power sensing duty cycle.
//
// capture() fills a channel-major int16 buffer; the TfAcquisitionSource API then hands it to
// the unchanged feature extractor exactly like an SD acquisition. Scale: +-2 g full range,
// 16384 LSB/g, so samples are in g.
#include <Wire.h>
#include "tinyfl_core/tf_source.h"

namespace tinyfl {

class Mpu6050Source : public TfAcquisitionSource {
 public:
  static constexpr uint8_t kAddr = 0x68;
  static constexpr uint32_t kSamples = 4096;   // 4.1 s at 1 kHz; FFT uses all of it
  static constexpr uint16_t kChannels = 3;
  static constexpr float kScale = 1.0f / 16384.0f;

  // Wire must already be running (SDA 21 / SCL 22, shared with the OLED).
  bool begin(TwoWire& wire) {
    w_ = &wire;
    if (!data_) data_ = static_cast<int16_t*>(malloc(sizeof(int16_t) * kSamples * kChannels));
    if (!data_) { tf_set_error("MPU6050: no heap for the sample buffer"); return false; }
    // A reset in the middle of an I2C transfer can leave the MPU6050 holding SDA low (seen once
    // on the bench right after a reflash): retry, clocking the bus free between attempts.
    uint8_t who = 0;
    bool answered = false;
    for (int attempt = 0; attempt < 5 && !answered; ++attempt) {
      if (attempt) { busRecover(); delay(100); }
      answered = readReg(0x75, &who, 1);
    }
    if (!answered) { tf_set_error("MPU6050: no answer at 0x68 (check wiring)"); return false; }
    whoAmI_ = who;   // 0x68 genuine; MPU6500-family clones answer 0x70/0x72/0x98 with the same map
    if (who != 0x68 && who != 0x70 && who != 0x72 && who != 0x98) {
      tf_set_error("MPU6050: unexpected WHO_AM_I 0x%02x", who); return false;
    }
    bool ok = writeReg(0x6B, 0x80);             // device reset
    delay(100);
    ok = ok && writeReg(0x6B, 0x00)              // wake, internal 8 MHz oscillator
            && writeReg(0x6C, 0x07)              // gyros X/Y/Z standby, accelerometer on
            && writeReg(0x1A, 0x01)              // DLPF 184 Hz -> 1 kHz internal rate
            && writeReg(0x19, 0x00)              // sample rate = 1 kHz / (1 + 0)
            && writeReg(0x1C, 0x00)              // +-2 g
            && writeReg(0x23, 0x08)              // FIFO: accelerometer only (6 bytes/sample)
            && writeReg(0x6B, 0x40);             // sleep until the first acquisition
    if (!ok) { tf_set_error("MPU6050: configuration write failed"); return false; }
    return true;
  }
  uint8_t whoAmI() const { return whoAmI_; }
  // Optional: called between FIFO drains with the time the FIFO can safely fill (light-sleep
  // build). nullptr = busy-poll with tf_yield().
  void (*idleHook)(uint32_t ms) = nullptr;

  // Wakes the sensor, records one acquisition, puts it back to sleep. false on an I2C error
  // or a FIFO overflow (then nothing is handed to the extractor).
  bool capture() {
    captured_ = false;
    for (int attempt = 0; attempt < 2 && !captured_; ++attempt) captured_ = captureOnce();
    if (!captured_) writeReg(0x6B, 0x40);
    return captured_;
  }

  uint16_t channels() const override { return kChannels; }
  uint32_t acquisitionCount() const override { return 0; }   // live, unbounded
  bool nextAcquisition(TfAcqInfo* info) override {
    if (!captured_) { tf_set_error("MPU6050: no captured acquisition"); return false; }
    captured_ = false;
    info->index = index_++; info->samples = kSamples; info->channels = kChannels;
    return true;
  }
  bool beginChannel(uint16_t c, float* scale) override {
    if (c >= kChannels) return false;
    ch_ = c; pos_ = 0; *scale = kScale; return true;
  }
  int readChunk(int16_t* dst, int maxSamples) override {
    uint32_t n = kSamples - pos_;
    if (n > (uint32_t)maxSamples) n = (uint32_t)maxSamples;
    memcpy(dst, data_ + (size_t)ch_ * kSamples + pos_, n * sizeof(int16_t));
    pos_ += n; return (int)n;
  }

 private:
  bool captureOnce() {
    uint8_t status;
    if (!writeReg(0x6B, 0x00)) return false;     // wake (accelerometer start-up ~20 ms)
    delay(30);
    if (!writeReg(0x6A, 0x04) || !writeReg(0x6A, 0x40) || !readReg(0x3A, &status, 1)) return false;  // reset + enable FIFO, clear INT_STATUS
    uint32_t got = 0, idle = millis();
    uint8_t buf[120];                            // 20 samples per transfer (Wire buffer is 128 B)
    while (got < kSamples) {
      uint8_t cnt[2];
      if (!readReg(0x72, cnt, 2)) return false;
      uint32_t bytes = ((uint32_t)cnt[0] << 8) | cnt[1];
      if (bytes >= 1024) { tf_set_error("MPU6050: FIFO overflow"); return false; }
      uint32_t samples = bytes / 6;
      if (!samples) {
        if (millis() - idle > 200) { tf_set_error("MPU6050: no samples"); return false; }
        tf_yield(); continue;
      }
      idle = millis();
      while (samples && got < kSamples) {
        uint32_t take = samples > 20 ? 20 : samples;
        if (take > kSamples - got) take = kSamples - got;
        if (!readReg(0x74, buf, take * 6)) return false;
        for (uint32_t i = 0; i < take; ++i, ++got)
          for (int c = 0; c < kChannels; ++c)
            data_[(size_t)c * kSamples + got] = (int16_t)((buf[6 * i + 2 * c] << 8) | buf[6 * i + 2 * c + 1]);
        samples -= take;
      }
      // FIFO holds 170 samples (170 ms): let it fill for 100 ms (~600 B) before the next drain.
      if (idleHook && kSamples - got > 100) idleHook(100); else tf_yield();   // services LMIC/watchdog
    }
    if (!readReg(0x3A, &status, 1) || (status & 0x10)) { tf_set_error("MPU6050: FIFO overflow"); return false; }
    return writeReg(0x6A, 0x00) && writeReg(0x6B, 0x40);   // FIFO off, sleep
  }
  // Standard I2C bus clear: up to 9 SCL pulses until the slave releases SDA, then a STOP.
  void busRecover() {
    w_->end();
    pinMode(kSda, INPUT_PULLUP); pinMode(kScl, OUTPUT_OPEN_DRAIN); digitalWrite(kScl, HIGH);
    for (int i = 0; i < 9 && digitalRead(kSda) == LOW; ++i) {
      digitalWrite(kScl, LOW); delayMicroseconds(5); digitalWrite(kScl, HIGH); delayMicroseconds(5);
    }
    pinMode(kSda, OUTPUT_OPEN_DRAIN); digitalWrite(kSda, LOW); delayMicroseconds(5);
    digitalWrite(kSda, HIGH); delayMicroseconds(5);              // STOP: SDA rises while SCL high
    w_->begin(kSda, kScl); w_->setClock(400000);
  }
  static constexpr int kSda = 21, kScl = 22;
  bool writeReg(uint8_t reg, uint8_t v) {
    w_->beginTransmission(kAddr); w_->write(reg); w_->write(v);
    return w_->endTransmission() == 0;
  }
  bool readReg(uint8_t reg, uint8_t* dst, size_t n) {
    w_->beginTransmission(kAddr); w_->write(reg);
    if (w_->endTransmission(false) != 0) return false;
    if (w_->requestFrom((int)kAddr, (int)n) != (int)n) return false;
    for (size_t i = 0; i < n; ++i) dst[i] = (uint8_t)w_->read();
    return true;
  }

  TwoWire* w_ = nullptr;
  int16_t* data_ = nullptr;
  uint32_t index_ = 0, pos_ = 0;
  uint16_t ch_ = 0;
  uint8_t whoAmI_ = 0;
  bool captured_ = false;
};

}  // namespace tinyfl
