# Data formats

All units are SI: metres, seconds, radians, metres per second.

## Coordinate frames

**Sensor frame** (what the radar reports):

- origin at the radar;
- `+y` straight ahead (boresight), `+x` to the right when looking forward;
- the LD2450 reports millimetres; the firmware converts to metres.

**Room frame** (what tracking runs in):

- origin and axes defined by the floor plan file;
- `+x` and `+y` horizontal, angles measured counter-clockwise from `+x`.

**Helmet pose** links the two: position `(px, py)` in the room frame and yaw
`ψ` (direction the radar faces, counter-clockwise from room `+x`). A sensor
point `(xs, ys)` maps to the room frame as:

```
forward = (cos ψ, sin ψ)
right   = (sin ψ, -cos ψ)
room    = (px, py) + ys · forward + xs · right
```

## Frame (one radar update)

One JSON object. Logged as **JSON Lines** (one frame per line) in
`data/<session>.jsonl`. The ESP32 sends the same object as one UDP packet
(default port `5005`) or one line over USB serial.

```json
{
  "v": 1,
  "src": "ld2450",
  "seq": 1842,
  "t_dev": 184.217,
  "t": 1730745601.532,
  "yaw": 1.5708,
  "pose": null,
  "targets": [
    {"x": -0.42, "y": 3.10, "v": 0.85},
    {"x": 1.20, "y": 4.75, "v": 0.00}
  ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `v` | int | Format version. Currently `1`. |
| `src` | string | `"ld2450"`, `"iwr6843"` or `"sim"`. |
| `seq` | int | Frame counter from the source. Gaps mean dropped frames. |
| `t_dev` | float | Device time in seconds (ESP32 `millis()/1000`, or sim clock). |
| `t` | float | Host receive time, Unix seconds. Added by ingest; firmware omits it. |
| `yaw` | float or null | Helmet yaw ψ in radians. `null` if the IMU is missing. |
| `pose` | object or null | Helmet position `{"x", "y"}` in the room frame, if known. |
| `targets` | list | 0–3 detections in the **sensor frame**. Empty slots are omitted. |
| `targets[].x`, `.y` | float | Position in metres. |
| `targets[].v` | float | Radial speed in m/s from the radar (sign as reported). |

### Presence (optional)

When an HLK-LD2410 presence sensor is fitted next to the LD2450, frames carry its
latest reading. It sees people who are sitting or standing still (breathing, small
movements), which the LD2450 often misses, but it gives distance only, no direction.

```json
"presence": {"state": "still", "moving_m": null, "still_m": 3.375}
```

| Field | Type | Meaning |
|---|---|---|
| `state` | string | `"none"`, `"moving"`, `"still"` or `"both"`. |
| `moving_m` | float or null | Distance to the nearest moving person, metres. |
| `still_m` | float or null | Distance to the nearest still person, metres. |

Distances are only known to within one 0.75 m gate; values are the gate centre.
Frames without the field (no LD2410) are valid.

Simulated frames add a `truth` field, used only by evaluation and the viewer:

```json
"truth": {
  "pose": {"x": 0.5, "y": 1.0, "yaw": 1.5708},
  "people": [
    {"id": 0, "x": 2.10, "y": 3.95, "vx": 0.6, "vy": 0.1, "mode": "walk"}
  ]
}
```

`mode` is one of `"walk"`, `"turn"`, `"stand"`, `"sit"`.

Future fields (IWR6843 point clouds, per-target point lists) will be added
under a new format version.

## Session file

`data/<session>.jsonl`. The first line is a header, every other line is a
frame:

```json
{"type": "header", "v": 1, "created": "2026-11-04T21:30:00Z", "src": "sim", "map": "maps/demo_flat.json", "notes": "seed 42, 2 people"}
```

## Floor plan file

`maps/<name>.json`. Rough is fine: about 20 cm accuracy is enough, since radar
error is larger than that.

```json
{
  "v": 1,
  "name": "demo_flat",
  "units": "m",
  "bounds": {"xmin": 0, "ymin": 0, "xmax": 8, "ymax": 5},
  "walls": [
    {"a": [0, 0], "b": [8, 0], "material": "brick"},
    {"a": [4, 0], "b": [4, 2], "material": "drywall"},
    {"a": [4, 2.9], "b": [4, 5], "material": "drywall"}
  ],
  "doors": [
    {"a": [4, 2], "b": [4, 2.9]}
  ],
  "obstacles": [
    {"name": "sofa", "polygon": [[5, 3.8], [7.2, 3.8], [7.2, 4.7], [5, 4.7]]}
  ],
  "rooms": [
    {"name": "bedroom", "polygon": [[0, 0], [4, 0], [4, 5], [0, 5]]},
    {"name": "living", "polygon": [[4, 0], [8, 0], [8, 5], [4, 5]]}
  ]
}
```

| Field | Meaning |
|---|---|
| `walls` | Line segments people cannot cross. Doorways are gaps between walls. |
| `walls[].material` | `"drywall"`, `"wood"`, `"glass"`, `"brick"`, `"concrete"`. The simulator uses it for radar attenuation. |
| `doors` | The gaps, listed explicitly so the particle filter and simulator can route through them. |
| `obstacles` | Furniture people walk around (polygons). |
| `rooms` | Optional, for labels and simulated destinations. |
