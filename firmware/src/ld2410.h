// HLK-LD2410 report frame decoder. Pure C++ (no Arduino), so it is tested on the laptop.
//
// Frame: F4 F3 F2 F1 | length (uint16 LE) | data | F8 F7 F6 F5
// Data (basic and engineering mode start the same way):
//   type (0x02 basic, 0x01 engineering), 0xAA, target state,
//   moving distance (cm, uint16 LE), moving energy, still distance (cm, uint16 LE),
//   still energy, detection distance (cm, uint16 LE), ... 0x55, check byte.
// Target state: 0 none, 1 moving, 2 still, 3 both.
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace ld2410 {

constexpr uint8_t HEADER[4] = {0xF4, 0xF3, 0xF2, 0xF1};
constexpr uint8_t FOOTER[4] = {0xF8, 0xF7, 0xF6, 0xF5};
constexpr size_t MAX_DATA_LEN = 64;  // engineering mode is 35 bytes
constexpr size_t MIN_DATA_LEN = 13;
constexpr uint8_t DATA_HEAD = 0xAA;

enum State : uint8_t { NONE = 0, MOVING = 1, STILL = 2, BOTH = 3 };

struct Reading {
  uint8_t state;
  float moving_m;
  float still_m;
};

class Parser {
 public:
  // Feed one byte; returns true when a complete, well-formed report has been decoded.
  bool feed(uint8_t b) {
    switch (phase_) {
      case HEAD:
        pos_ = (b == HEADER[pos_]) ? pos_ + 1 : (b == HEADER[0] ? 1 : 0);
        if (pos_ == sizeof(HEADER)) start(LEN);
        return false;
      case LEN:
        len_bytes_[pos_++] = b;
        if (pos_ == 2) {
          data_len_ = size_t(len_bytes_[0] | (len_bytes_[1] << 8));
          if (data_len_ < MIN_DATA_LEN || data_len_ > MAX_DATA_LEN) {
            start(HEAD);  // not a report frame (e.g. a command ACK) or corrupted
          } else {
            start(DATA);
          }
        }
        return false;
      case DATA:
        data_[pos_++] = b;
        if (pos_ == data_len_) start(TAIL);
        return false;
      case TAIL:
        if (b != FOOTER[pos_]) {
          start(HEAD);
          return false;
        }
        if (++pos_ < sizeof(FOOTER)) return false;
        start(HEAD);
        return decode();
    }
    return false;
  }

  const Reading& reading() const { return reading_; }

 private:
  enum Phase { HEAD, LEN, DATA, TAIL };

  void start(Phase p) {
    phase_ = p;
    pos_ = 0;
  }

  bool decode() {
    if (data_[1] != DATA_HEAD || data_[2] > BOTH) return false;
    reading_.state = data_[2];
    reading_.moving_m = (data_[3] | (data_[4] << 8)) / 100.0f;  // cm -> m
    reading_.still_m = (data_[6] | (data_[7] << 8)) / 100.0f;
    return true;
  }

  Phase phase_ = HEAD;
  size_t pos_ = 0;
  uint8_t len_bytes_[2] = {};
  size_t data_len_ = 0;
  uint8_t data_[MAX_DATA_LEN] = {};
  Reading reading_ = {};
};

}  // namespace ld2410
