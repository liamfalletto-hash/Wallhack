# Helmet firmware (ESP32)

Reads the LD2450 radar, the LD2410 presence sensor and the BNO085 orientation sensor, and
streams one JSON frame per radar report (10 Hz) over USB and Wi-Fi. The frame format is
in [`docs/data-formats.md`](../docs/data-formats.md). Tracking runs on the laptop.

## Wiring

| Part | Part pin | ESP32 pin |
|---|---|---|
| LD2450 | 5V / GND | 5V / GND |
| LD2450 | TX | GPIO16 |
| LD2450 | RX | GPIO17 |
| LD2410C | 5V / GND | 5V / GND |
| LD2410C | TX | GPIO25 |
| LD2410C | RX | GPIO26 |
| BNO085 | VIN / GND | 3V3 / GND |
| BNO085 | SDA | GPIO21 |
| BNO085 | SCL | GPIO22 |

Mount the LD2410 next to the LD2450, facing the same way. Mount the BNO085 flat (chip
facing up).

## Build and upload

1. Install the **PlatformIO IDE** extension in VS Code.
2. Copy `src/secrets.example.h` to `src/secrets.h` and fill in your Wi-Fi name and
   password. `secrets.h` is gitignored and never leaves your computer.
3. Plug in the ESP32 by USB and run, from the repository root:
   ```bash
   pio run -d firmware -t upload
   ```
4. Watch the output (Ctrl+C to stop):
   ```bash
   pio device monitor -d firmware
   ```
   Frames are lines starting with `{`; status messages start with `#`.

## Live view on the laptop

Stand where the map says you are (default 2.0, 2.2), face the direction given by the
last number (degrees, 0 = map +x), then:

```bash
python -m host.viewer --live --helmet 2.0,2.2,0
```

Over USB instead of Wi-Fi (needs `pip install -e ".[usb]"`; find the port with
`ls /dev/cu.*`):

```bash
python -m host.viewer --live --serial /dev/cu.usbserial-0001 --helmet 2.0,2.2,0
```

Every live session is recorded to `data/live_<date>_<time>.jsonl` and can be replayed
with `python -m host.viewer data/live_....jsonl`. Without hardware, `python -m
sim.fake_helmet` sends simulated frames the same way. On macOS, allow incoming
connections for Python if the firewall asks.

## First-run checks (untested on device)

None of this has run on real hardware yet. On the first power-up:

1. **Radar left/right:** stand to the radar's *right*. Frames should show a positive
   `x`. If it is negative, set `FLIP_X = true` in `src/main.cpp` and upload again.
2. **Heading:** with the helmet still, `yaw` should hold steady; turn your head *left*
   and `yaw` should increase. If it decreases, set `IMU_YAW_SIGN = -1.0f`. If the
   output says `# no imu`, check the SDA/SCL wires.
3. **Presence:** sit still 2 m in front of the sensors. Frames should carry
   `"presence": {"state": "still", ...}` with `still_m` near 2. If `presence` never
   appears, check the LD2410's TX/RX wires (they cross over: sensor TX goes to ESP32 RX).
