// Radar helmet firmware: reads LD2450 + LD2410 + BNO085 and streams one JSON frame per
// LD2450 report over USB serial and Wi-Fi UDP broadcast. Tracking runs on the laptop.
// USB lines that are not frames start with '#', so the laptop can ignore them.

#include <Adafruit_BNO08x.h>
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <math.h>

#include "frame_json.h"
#include "ld2410.h"
#include "ld2450.h"

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Copy firmware/src/secrets.example.h to firmware/src/secrets.h and fill in your Wi-Fi details"
#endif

namespace {

constexpr uint32_t USB_BAUD = 115200;
constexpr uint32_t SENSOR_BAUD = 256000;  // LD2450 and LD2410 factory default
constexpr int LD2450_RX_PIN = 16, LD2450_TX_PIN = 17;
constexpr int LD2410_RX_PIN = 25, LD2410_TX_PIN = 26;
constexpr int I2C_SDA_PIN = 21, I2C_SCL_PIN = 22;
constexpr uint8_t IMU_I2C_ADDR = 0x4A;
constexpr uint32_t IMU_REPORT_US = 10000;  // 100 Hz game rotation vector
constexpr uint16_t UDP_PORT = 5005;
constexpr uint32_t RETRY_MS = 5000;           // Wi-Fi reconnect and IMU re-detect interval
constexpr uint32_t PRESENCE_STALE_MS = 1000;  // older LD2410 readings are dropped from frames

// Mounting. Check these on the first run (see firmware/README.md).
constexpr bool FLIP_X = false;        // targets appear on the wrong side: set true
constexpr float IMU_YAW_SIGN = 1.0f;  // turning left must increase yaw: else set -1

ld2450::Parser radar;
ld2410::Parser presence;
Adafruit_BNO08x imu(-1);
WiFiUDP udp;

bool imu_ok = false;
bool has_yaw = false;
float yaw_rad = 0.0f;
uint32_t presence_ms = 0;
bool has_presence_reading = false;
uint32_t seq = 0;
uint32_t last_retry_ms = 0;
char json[512];

bool start_imu() {
  if (!imu.begin_I2C(IMU_I2C_ADDR, &Wire)) return false;
  return imu.enableReport(SH2_GAME_ROTATION_VECTOR, IMU_REPORT_US);
}

void read_imu() {
  if (!imu_ok) return;
  if (imu.wasReset()) imu.enableReport(SH2_GAME_ROTATION_VECTOR, IMU_REPORT_US);
  sh2_SensorValue_t ev;
  while (imu.getSensorEvent(&ev)) {
    if (ev.sensorId != SH2_GAME_ROTATION_VECTOR) continue;
    const auto& q = ev.un.gameRotationVector;
    // Heading about the vertical axis, counter-clockwise positive (chip mounted flat).
    float yaw = atan2f(2.0f * (q.real * q.k + q.i * q.j), 1.0f - 2.0f * (q.j * q.j + q.k * q.k));
    yaw_rad = IMU_YAW_SIGN * yaw;
    has_yaw = true;
  }
}

void read_presence() {
  while (Serial1.available()) {
    if (presence.feed(uint8_t(Serial1.read()))) {
      presence_ms = millis();
      has_presence_reading = true;
    }
  }
}

void send_frame() {
  frame_json::Frame f = {};
  f.seq = ++seq;
  f.t_dev_s = millis() / 1000.0f;
  f.has_yaw = imu_ok && has_yaw;
  f.yaw_rad = yaw_rad;
  f.radar = &radar;
  f.flip_x = FLIP_X;
  f.has_presence = has_presence_reading && millis() - presence_ms < PRESENCE_STALE_MS;
  f.presence = presence.reading();
  int n = frame_json::write(json, sizeof(json), f);
  if (n < 0) {
    Serial.println("# frame too long, dropped");
    return;
  }
  Serial.write(json, n);
  Serial.write('\n');
  if (WiFi.status() == WL_CONNECTED) {
    IPAddress ip = WiFi.localIP(), mask = WiFi.subnetMask();
    IPAddress broadcast(ip[0] | ~mask[0], ip[1] | ~mask[1], ip[2] | ~mask[2], ip[3] | ~mask[3]);
    udp.beginPacket(broadcast, UDP_PORT);
    udp.write(reinterpret_cast<const uint8_t*>(json), n);
    udp.endPacket();
  }
}

void retry_links() {
  if (millis() - last_retry_ms < RETRY_MS) return;
  last_retry_ms = millis();
  if (WiFi.status() != WL_CONNECTED) WiFi.reconnect();
  if (!imu_ok) {
    imu_ok = start_imu();
    Serial.println(imu_ok ? "# imu found" : "# no imu: frames carry yaw null");
  }
}

}  // namespace

void setup() {
  Serial.begin(USB_BAUD);
  Serial2.begin(SENSOR_BAUD, SERIAL_8N1, LD2450_RX_PIN, LD2450_TX_PIN);
  Serial1.begin(SENSOR_BAUD, SERIAL_8N1, LD2410_RX_PIN, LD2410_TX_PIN);
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  imu_ok = start_imu();
  Serial.println(imu_ok ? "# imu found" : "# no imu: frames carry yaw null");
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);  // non-blocking; frames go over USB until it connects
  Serial.printf("# streaming frames, udp broadcast port %u once wifi connects\n", UDP_PORT);
}

void loop() {
  read_imu();
  read_presence();
  while (Serial2.available()) {
    if (radar.feed(uint8_t(Serial2.read()))) send_frame();
  }
  retry_links();
}
