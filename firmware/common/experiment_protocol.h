#pragma once

#include <Arduino.h>

namespace tinyfl_hw {

constexpr uint8_t kMagic = 0xF1;
constexpr uint8_t kVersion = 0x02;
constexpr uint8_t kUpdateFragment = 0x01;
constexpr uint8_t kClassCReady = 0x02;
constexpr uint8_t kGlobalFragment = 0x03;
constexpr uint8_t kGlobalStatus = 0x04;
// Downlink asking this node to resend the UPDATE fragments the server never
// received. Unconfirmed uplinks have no retransmission of their own, so one
// lost fragment would otherwise discard the whole 2202-byte update.
constexpr uint8_t kUpdateRepair = 0x05;

constexpr uint8_t kFPort = 10;
constexpr size_t kHeaderBytes = 7;
constexpr size_t kFragmentBodyBytes = 214;
constexpr size_t kMaxApplicationBytes = kHeaderBytes + kFragmentBodyBytes;

// These are the largest V91 messages after the online-stopping overhead:
// uplink 2202 bytes and downlink 2241 bytes.
constexpr size_t kUpdateBytes = 2202;
constexpr size_t kGlobalBytes = 2241;
constexpr size_t kCrcBytes = 4;
constexpr size_t kUpdateDataBytes = kUpdateBytes - kCrcBytes;
constexpr size_t kGlobalDataBytes = kGlobalBytes - kCrcBytes;
constexpr uint8_t kUpdateFragments =
    static_cast<uint8_t>((kUpdateBytes + kFragmentBodyBytes - 1) /
                         kFragmentBodyBytes);
constexpr uint8_t kGlobalFragments =
    static_cast<uint8_t>((kGlobalBytes + kFragmentBodyBytes - 1) /
                         kFragmentBodyBytes);
constexpr uint16_t kAllGlobalFragments =
    static_cast<uint16_t>((1u << kGlobalFragments) - 1u);
constexpr uint16_t kAllUpdateFragments =
    static_cast<uint16_t>((1u << kUpdateFragments) - 1u);

static_assert(kMaxApplicationBytes == 221, "DR4/DR5 application payload must be 221 bytes");
static_assert(kUpdateFragments == 11, "V91 maximum uplink must use 11 fragments");
static_assert(kGlobalFragments == 11, "V91 maximum downlink must use 11 fragments");

inline uint32_t crc32Update(uint32_t crc, uint8_t value) {
  crc ^= value;
  for (uint8_t bit = 0; bit < 8; ++bit) {
    const uint32_t mask = 0u - (crc & 1u);
    crc = (crc >> 1u) ^ (0xEDB88320u & mask);
  }
  return crc;
}

inline uint32_t crc32(const uint8_t* data, size_t length) {
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < length; ++i) {
    crc = crc32Update(crc, data[i]);
  }
  return crc ^ 0xFFFFFFFFu;
}

inline void writeLe16(uint8_t* output, uint16_t value) {
  output[0] = static_cast<uint8_t>(value);
  output[1] = static_cast<uint8_t>(value >> 8u);
}

inline uint16_t readLe16(const uint8_t* input) {
  return static_cast<uint16_t>(input[0]) |
         (static_cast<uint16_t>(input[1]) << 8u);
}

inline void writeLe32(uint8_t* output, uint32_t value) {
  output[0] = static_cast<uint8_t>(value);
  output[1] = static_cast<uint8_t>(value >> 8u);
  output[2] = static_cast<uint8_t>(value >> 16u);
  output[3] = static_cast<uint8_t>(value >> 24u);
}

inline uint32_t readLe32(const uint8_t* input) {
  return static_cast<uint32_t>(input[0]) |
         (static_cast<uint32_t>(input[1]) << 8u) |
         (static_cast<uint32_t>(input[2]) << 16u) |
         (static_cast<uint32_t>(input[3]) << 24u);
}

}  // namespace tinyfl_hw
