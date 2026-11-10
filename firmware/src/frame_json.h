// Serialises one frame in the format of docs/data-formats.md. Pure C++, no heap.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ld2410.h"
#include "ld2450.h"

namespace frame_json {

constexpr int FORMAT_VERSION = 1;

struct Frame {
  uint32_t seq;
  float t_dev_s;
  bool has_yaw;
  float yaw_rad;
  const ld2450::Parser* radar;
  bool flip_x;  // radar mounted upside down: mirror left/right
  bool has_presence;
  ld2410::Reading presence;
};

// Writes JSON (no newline) into buf. Returns the length, or -1 if buf is too small.
inline int write(char* buf, size_t cap, const Frame& f) {
  size_t n = 0;
  auto put = [&](const char* fmt, auto... args) {
    if (n >= cap) return;
    int w = snprintf(buf + n, cap - n, fmt, args...);
    n = (w < 0) ? cap : n + size_t(w);
  };
  auto lit = [&](const char* text) { put("%s", text); };

  put("{\"v\":%d,\"src\":\"ld2450\",\"seq\":%lu,\"t_dev\":%.3f,", FORMAT_VERSION, (unsigned long)f.seq,
      (double)f.t_dev_s);
  if (f.has_yaw) {
    put("\"yaw\":%.4f,", (double)f.yaw_rad);
  } else {
    lit("\"yaw\":null,");
  }
  lit("\"pose\":null,\"targets\":[");
  bool first = true;
  for (size_t i = 0; i < ld2450::MAX_TARGETS; i++) {
    const ld2450::Target& t = f.radar->target(i);
    if (!t.valid) continue;
    float x = f.flip_x ? -t.x_m : t.x_m;
    put("%s{\"x\":%.3f,\"y\":%.3f,\"v\":%.2f}", first ? "" : ",", (double)x, (double)t.y_m, (double)t.v_mps);
    first = false;
  }
  lit("]");
  if (f.has_presence) {
    static const char* const NAMES[] = {"none", "moving", "still", "both"};
    const ld2410::Reading& p = f.presence;
    put(",\"presence\":{\"state\":\"%s\",", NAMES[p.state & 3]);
    if (p.state & ld2410::MOVING) {
      put("\"moving_m\":%.2f,", (double)p.moving_m);
    } else {
      lit("\"moving_m\":null,");
    }
    if (p.state & ld2410::STILL) {
      put("\"still_m\":%.2f}", (double)p.still_m);
    } else {
      lit("\"still_m\":null}");
    }
  }
  lit("}");
  return n < cap ? int(n) : -1;
}

}  // namespace frame_json
