# Wallhack

A helmet-mounted radar system that tracks people on the other side of a wall
and predicts where they are going next.

A 24 GHz mmWave radar (HLK-LD2450) and an orientation sensor (BNO085) sit on a
helmet and are read by an ESP32. The ESP32 streams detections to a laptop or
Raspberry Pi, where Python code tracks each person, predicts their path for
the next 1–2 seconds, and draws a live map. A small OLED on the helmet brim can
show a simplified dot map without a laptop.

> **Status:** planning. No code yet. The design lives in [`docs/`](docs/).

> **Consent:** only track people who know they are being tracked (for example
> your roommate, for recording data). Covertly recording neighbours or anyone
> else is illegal in most of the EU (GDPR) and many other places.

## How it works

```
[LD2450 + BNO085 on ESP32] ──┐
                             ├──> frames (one shared format)
[Simulator]  ────────────────┘            │
                                          ▼
                       Ingest + logger (JSON Lines files)
                                          ▼
          Tracker: match detections to people → pluggable motion filter
                                          ▼
                    Prediction (where each person is in 1–2 s)
                                          ▼
             Live map viewer      |      Evaluation (error tables)
```

The simulator and the real radar produce the **same frame format**, so
everything after ingest can be built and measured before any hardware arrives.
See [`docs/architecture.md`](docs/architecture.md) and
[`docs/data-formats.md`](docs/data-formats.md).

## Repository layout

| Folder | Contents | Language |
|---|---|---|
| `firmware/` | ESP32 sketch: read radar + IMU, stream frames, optional OLED | Arduino C++ |
| `sim/` | Floor plans, simulated people, radar sensor model | Python |
| `tracker/` | Data association, motion filters, prediction | Python |
| `eval/` | Metrics and filter comparison reports | Python |
| `viewer/` | Live and replay 2D map | Python |
| `maps/` | Floor plan files (JSON) | data |
| `data/` | Recorded sessions (not committed, see `.gitignore`) | data |
| `docs/` | Design documents | Markdown |

## Milestones

1. **Skeleton and data formats.** This commit: docs, folder layout, formats.
2. **Simulator + straight-line Kalman baseline**, with evaluation numbers.
3. **IMM filter** (walking / turning / standing), compared to the baseline.
4. **Floor-plan particle filter**, compared to both.
5. **Real hardware:** ESP32 streaming, logging, replaying recordings.
6. **GRU prediction model** trained on simulated + real recordings.
7. *(Optional)* **TI IWR6843AOP** support and an estimated silhouette.

Milestones 1–4 need no hardware.

## Hardware

| Part | Purpose | Approx. price |
|---|---|---|
| HLK-LD2450 24 GHz radar | Detects up to 3 people, 10 Hz | €5–15 |
| ESP32 DevKitC | Reads sensors, streams over Wi-Fi/USB | €13–24 |
| BNO085 IMU breakout | Helmet heading | €20–25 |
| 0.96" SSD1306 OLED (optional) | Brim display | €4–6 |
| USB power bank, wires, mount | Power and mounting | €20–30 |
| *Later:* TI IWR6843AOP EVM | 3D point cloud, 20–30 Hz | €170–200 |

## Requirements (planned)

- Python 3.11+ (NumPy, SciPy, Matplotlib; PyTorch for milestone 6)
- Arduino IDE or PlatformIO with the ESP32 board package
