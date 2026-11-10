// Test driver for the firmware's pure decoders, built and run by tests/test_firmware.py.
// One command per stdin line; hex bytes may contain spaces:
//   ld2450 <hex>                      -> one JSON line per decoded frame, then "end"
//   ld2410 <hex>                      -> one JSON line per decoded report, then "end"
//   frame <yaw|-> <flip 0|1> <ld2450 hex> [| <ld2410 hex>]  -> the frame JSON, or "overflow"
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "frame_json.h"

static std::vector<uint8_t> hex_bytes(const std::string& s) {
  std::vector<uint8_t> out;
  std::string digits;
  for (char c : s)
    if (isxdigit(static_cast<unsigned char>(c))) digits += c;
  for (size_t i = 0; i + 1 < digits.size(); i += 2) out.push_back(uint8_t(std::stoi(digits.substr(i, 2), nullptr, 16)));
  return out;
}

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    std::string cmd;
    in >> cmd;
    std::string rest;
    std::getline(in, rest);
    if (cmd == "ld2450") {
      ld2450::Parser p;
      for (uint8_t b : hex_bytes(rest)) {
        if (!p.feed(b)) continue;
        printf("[");
        for (size_t i = 0; i < ld2450::MAX_TARGETS; i++) {
          const auto& t = p.target(i);
          printf("%s{\"valid\":%s,\"x\":%.3f,\"y\":%.3f,\"v\":%.2f}", i ? "," : "", t.valid ? "true" : "false",
                 (double)t.x_m, (double)t.y_m, (double)t.v_mps);
        }
        printf("]\n");
      }
      printf("end\n");
    } else if (cmd == "ld2410") {
      ld2410::Parser p;
      for (uint8_t b : hex_bytes(rest)) {
        if (!p.feed(b)) continue;
        const auto& r = p.reading();
        printf("{\"state\":%u,\"moving_m\":%.2f,\"still_m\":%.2f}\n", r.state, (double)r.moving_m, (double)r.still_m);
      }
      printf("end\n");
    } else if (cmd == "frame") {
      std::istringstream args(rest);
      std::string yaw, flip;
      args >> yaw >> flip;
      std::string hex;
      std::getline(args, hex);
      size_t bar = hex.find('|');
      ld2450::Parser radar;
      for (uint8_t b : hex_bytes(hex.substr(0, bar))) radar.feed(b);
      frame_json::Frame f = {};
      f.seq = 7;
      f.t_dev_s = 12.5f;
      f.has_yaw = yaw != "-";
      f.yaw_rad = f.has_yaw ? std::stof(yaw) : 0.0f;
      f.radar = &radar;
      f.flip_x = flip == "1";
      if (bar != std::string::npos) {
        ld2410::Parser presence;
        for (uint8_t b : hex_bytes(hex.substr(bar + 1))) f.has_presence |= presence.feed(b);
        f.presence = presence.reading();
      }
      char buf[512];
      int n = frame_json::write(buf, sizeof(buf), f);
      if (n < 0) {
        printf("overflow\n");
      } else {
        printf("%s\n", buf);
      }
    }
    fflush(stdout);
  }
  return 0;
}
