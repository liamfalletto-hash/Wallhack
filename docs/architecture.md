# Architecture

## Goals

1. Track up to 3 people through an interior wall with the HLK-LD2450 radar.
2. Predict each person's position 1 s and 2 s ahead, and measure how good that
   prediction is.
3. Show the result live: on a laptop map, and optionally on a helmet OLED.
4. Work in **any** building. A floor plan improves results when available but
   is never required.

## Design principles

- **One frame format** for simulated and real data. Everything downstream of
  ingest cannot tell them apart. See [`data-formats.md`](data-formats.md).
- **Keep the firmware dumb.** The ESP32 only reads sensors and forwards raw
  values. All coordinate transforms, tracking and prediction run in Python,
  where they are easy to test and change.
- **Everything is replayable.** Every live session is logged to a file, and
  any file can be replayed through any filter. Comparisons always run on the
  same recordings.
- **Floor plans are optional input files**, never hard-coded.
- **Pluggable filters** behind one interface, so the evaluation can run them
  side by side.

## Components

### 1. Firmware (`firmware/`, Arduino C++)

- Parses LD2450 UART frames (256000 baud, 3 target slots, ~10 Hz).
- Reads yaw from the BNO085 over I2C (address 0x4A).
- Sends one frame per radar update over UDP (Wi-Fi) or USB serial, in the
  format from `data-formats.md`. Values stay in the **sensor frame**; the host
  does the rotation into room coordinates.
- Optional: drives the SSD1306 OLED (I2C 0x3C, sharing SDA=GPIO21, SCL=GPIO22
  with the IMU) with a simple dot map, so the helmet works standalone.

### 2. Simulator (`sim/`, Python)

- Loads a floor plan, or generates a random one (rooms, doors, furniture).
- Simulated people walk between points of interest along realistic routes
  (through doorways, around furniture), with stops, turns and sitting.
- Radar model matching LD2450 limits:
  - field of view about ±60°, range up to about 6 m;
  - range noise and angle noise (a few degrees);
  - at most 3 reported targets;
  - missed detections and false detections, more frequent behind walls,
    depending on wall material.
- Simulated helmet pose (fixed at first, moving later).
- Writes frames **plus ground truth** (true position of each person).

### 3. Ingest and logging

- Reads frames from UDP, serial, or a file.
- Timestamps on arrival and writes every frame to `data/<session>.jsonl`.
- Converts sensor-frame detections to room coordinates using helmet pose.

### 4. Tracker (`tracker/`, Python)

**Data association.** Matches new detections to existing tracks with the
Hungarian algorithm and a distance gate (Mahalanobis). Handles track birth
(new person appears), confirmation (seen in N of M frames) and deletion
(not seen for a set time).

**Motion filters**, all behind one interface (`predict(dt)`, `update(z)`,
`forecast(horizon)`):

| Filter | Idea | Needs map? |
|---|---|---|
| Constant-velocity Kalman | Baseline: straight lines | No |
| IMM | Mixes walking, turning and standing-still models | No |
| Map particle filter | Particles cannot cross walls, prefer doorways | Yes |
| GRU predictor | Learned from recordings (see below) | Optional |

**GRU predictor (milestone 6).** To work in new buildings it uses relative
inputs only: recent displacements and speed, plus optional local map features
(distance and direction to the nearest walls and doorways). It is trained on
many random simulated floor plans, then fine-tuned on real recordings. Never
trained on absolute room coordinates.

### 5. Evaluation (`eval/`, Python)

Runs every filter on the same recordings and reports:

- position error (RMSE) of the current estimate;
- prediction error at 1 s and 2 s horizons;
- identity switches (track jumps to a different person);
- track survival through missed detections.

Output: a comparison table and plots. This is the "error dropped from X to Y"
result.

### 6. Viewer (`viewer/`, Python)

Live or replay 2D map: floor plan, radar field of view, one dot per tracked
person, predicted path for the next 2 s, and ground truth when replaying
simulated data.

## Open problems

- **Wearer movement.** The helmet heading comes from the IMU, but its
  *position* in the room is unknown if the wearer walks. Version 1 assumes the
  wearer stands still, or that their position is given (simulation). Later
  options: step counting with the IMU (pedestrian dead reckoning), or marking
  the standing position on the map by hand.
- **Ground truth for real recordings.** The true position of someone behind a
  wall is not directly observable. Plan: walk pre-marked paths with timed
  waypoints, or record the far room with a phone camera and label positions.
- **Wall attenuation.** 24 GHz passes drywall, wood and doors fairly well but
  is strongly absorbed by brick and concrete. Measure the real walls early and
  tune the simulator's detection model to match.
