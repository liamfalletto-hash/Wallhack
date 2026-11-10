// HLK-LD2450 report frame decoder. Pure C++ (no Arduino), so it is tested on the laptop.
//
// Frame (30 bytes): AA FF 03 00 | 3 targets x 8 bytes | 55 CC
// Target: x, y (mm), speed (cm/s), resolution (mm), each uint16 little-endian.
// x, y and speed use a sign bit instead of two's complement: MSB set = positive,
// value = raw - 0x8000; MSB clear = negative, value = -raw. An all-zero target is empty.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace ld2450 {

constexpr size_t FRAME_LEN = 30;
constexpr size_t MAX_TARGETS = 3;
constexpr uint8_t HEADER[4] = {0xAA, 0xFF, 0x03, 0x00};
constexpr uint8_t FOOTER[2] = {0x55, 0xCC};

struct Target {
  bool valid;
  float x_m;    // right of boresight
  float y_m;    // along boresight
  float v_mps;  // radial speed, sign as reported by the sensor
};

inline int32_t signed_value(uint16_t raw) {
  return (raw & 0x8000) ? int32_t(raw - 0x8000) : -int32_t(raw);
}

inline uint16_t u16_le(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }

class Parser {
 public:
  // Feed one byte; returns true when a complete, well-formed frame has been decoded.
  bool feed(uint8_t b) {
    if (len_ < sizeof(HEADER)) {
      if (b == HEADER[len_]) {
        buf_[len_++] = b;
      } else {
        len_ = (b == HEADER[0]) ? 1 : 0;  // a stray byte may itself start the next header
        buf_[0] = b;
      }
      return false;
    }
    buf_[len_++] = b;
    if (len_ < FRAME_LEN) return false;
    len_ = 0;
    if (buf_[FRAME_LEN - 2] != FOOTER[0] || buf_[FRAME_LEN - 1] != FOOTER[1]) return false;
    decode();
    return true;
  }

  const Target& target(size_t i) const { return targets_[i]; }

 private:
  void decode() {
    for (size_t i = 0; i < MAX_TARGETS; i++) {
      const uint8_t* t = buf_ + sizeof(HEADER) + 8 * i;
      uint16_t rx = u16_le(t), ry = u16_le(t + 2), rv = u16_le(t + 4), rres = u16_le(t + 6);
      Target& out = targets_[i];
      out.valid = (rx | ry | rv | rres) != 0;
      out.x_m = signed_value(rx) / 1000.0f;   // mm -> m
      out.y_m = signed_value(ry) / 1000.0f;
      out.v_mps = signed_value(rv) / 100.0f;  // cm/s -> m/s
    }
  }

  uint8_t buf_[FRAME_LEN] = {};
  size_t len_ = 0;
  Target targets_[MAX_TARGETS] = {};
};

}  // namespace ld2450
