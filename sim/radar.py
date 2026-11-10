"""HLK-LD2450 sensor model: turns true positions into what the radar would report.

Every number below is an estimate from the datasheet and published tests. They are
meant to be recalibrated against the real sensor and real walls.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sim import floorplan as fpl
from sim.floorplan import FloorPlan
from sim.people import Person

FRAME_PERIOD_S = 0.1  # LD2450 reports at 10 Hz
MAX_TARGETS = 3
MAX_RANGE_M = 6.0
MIN_RANGE_M = 0.3
HALF_FOV_RAD = np.radians(60.0)

RANGE_NOISE_M = 0.08
ANGLE_NOISE_RAD = np.radians(3.0)
SPEED_NOISE_MPS = 0.10
BODY_REFLECTION_OFFSET_M = 0.10  # strongest return comes from the near side of the torso
MERGE_DISTANCE_M = 0.5  # closer targets are not resolved as separate people

BASE_DETECTION_P = 0.95
STILL_SPEED_MPS = 0.1
STILL_DETECTION_FACTOR = 0.4  # the LD2450 loses people who stop moving
FAR_RANGE_START_M = 4.0
FAR_RANGE_FACTOR = 0.6  # detection factor at MAX_RANGE_M, linear from FAR_RANGE_START_M
# Per wall crossed (radar signal goes through it twice); brick and concrete absorb most of 24 GHz.
WALL_DETECTION_FACTOR = {"drywall": 0.85, "wood": 0.8, "glass": 0.75, "brick": 0.3, "concrete": 0.1}
# Misses come in bursts, not independently: 0 = independent frames, 1 = state never changes.
DETECTION_CORRELATION = 0.6
FALSE_TARGETS_PER_FRAME = 0.02

YAW_NOISE_RAD = np.radians(0.3)
YAW_DRIFT_RAD_PER_SQRT_S = np.radians(0.05)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float  # direction the radar faces, CCW from world +x

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y])


@dataclass(frozen=True)
class Target:
    x: float  # metres, right of boresight
    y: float  # metres, along boresight
    v: float  # radial speed m/s, positive = moving away


@dataclass(frozen=True)
class RadarState:
    visible: dict[int, bool]  # last detection outcome per person id
    yaw_bias: float
    seq: int


def initial_state() -> RadarState:
    return RadarState(visible={}, yaw_bias=0.0, seq=0)


def to_sensor(helmet: Pose, p: np.ndarray) -> np.ndarray:
    d = np.asarray(p, float) - helmet.xy
    right = np.array([np.sin(helmet.yaw), -np.cos(helmet.yaw)])
    forward = np.array([np.cos(helmet.yaw), np.sin(helmet.yaw)])
    return np.array([d @ right, d @ forward])


def in_view(xs: float, ys: float) -> bool:
    r = np.hypot(xs, ys)
    return bool(ys > 0 and MIN_RANGE_M <= r <= MAX_RANGE_M and abs(np.arctan2(xs, ys)) <= HALF_FOV_RAD)


def detection_probability(fp: FloorPlan, helmet: Pose, person: Person) -> float:
    xs, ys = to_sensor(helmet, person.pos)
    if not in_view(xs, ys):
        return 0.0
    p = BASE_DETECTION_P
    for material in fpl.crossed_walls(fp, helmet.xy, person.pos):
        p *= WALL_DETECTION_FACTOR[material]
    if person.speed < STILL_SPEED_MPS:
        p *= STILL_DETECTION_FACTOR
    r = float(np.hypot(xs, ys))
    if r > FAR_RANGE_START_M:
        p *= 1 - (1 - FAR_RANGE_FACTOR) * (r - FAR_RANGE_START_M) / (MAX_RANGE_M - FAR_RANGE_START_M)
    return p


def sense(
    state: RadarState,
    fp: FloorPlan,
    helmet: Pose,
    people: Sequence[Person],
    rng: np.random.Generator,
    dt: float = FRAME_PERIOD_S,
) -> tuple[RadarState, list[Target], float]:
    """One radar frame: (new state, reported targets nearest first, IMU yaw reading)."""
    visible: dict[int, bool] = {}
    detected: list[Person] = []
    for person in people:
        p = detection_probability(fp, helmet, person)
        was_visible = state.visible[person.id] if person.id in state.visible else rng.random() < p
        p_now = p + (1 - p) * DETECTION_CORRELATION if was_visible else p * (1 - DETECTION_CORRELATION)
        visible[person.id] = p > 0 and rng.random() < p_now
        if visible[person.id]:
            detected.append(person)

    hits = [_measure(helmet, pos, vel, rng) for pos, vel in _merge(detected)]

    for _ in range(rng.poisson(FALSE_TARGETS_PER_FRAME)):
        r, az = rng.uniform(MIN_RANGE_M, MAX_RANGE_M), rng.uniform(-HALF_FOV_RAD, HALF_FOV_RAD)
        hits.append((r * np.sin(az), r * np.cos(az), rng.normal(0.0, 0.2)))

    targets = [_quantise(*h) for h in hits]
    targets = [t for t in targets if in_view(t.x, t.y)]
    targets.sort(key=lambda t: np.hypot(t.x, t.y))

    yaw_bias = state.yaw_bias + rng.normal(0.0, YAW_DRIFT_RAD_PER_SQRT_S * np.sqrt(dt))
    yaw = _wrap(helmet.yaw + yaw_bias + rng.normal(0.0, YAW_NOISE_RAD))
    return RadarState(visible, yaw_bias, state.seq + 1), targets[:MAX_TARGETS], yaw


def _measure(helmet: Pose, pos: np.ndarray, vel: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    xs, ys = to_sensor(helmet, pos)
    r_true = float(np.hypot(xs, ys))
    r = r_true - BODY_REFLECTION_OFFSET_M + rng.normal(0.0, RANGE_NOISE_M)
    az = float(np.arctan2(xs, ys)) + rng.normal(0.0, ANGLE_NOISE_RAD)
    radial = float(vel @ (pos - helmet.xy)) / r_true
    return r * np.sin(az), r * np.cos(az), radial + rng.normal(0.0, SPEED_NOISE_MPS)


def _merge(detected: list[Person]) -> list[tuple[np.ndarray, np.ndarray]]:
    # Resolution depends on true separation, so people closer than MERGE_DISTANCE_M become one blob.
    groups: list[list[Person]] = []
    for p in detected:
        group = next((g for g in groups if np.linalg.norm(g[0].pos - p.pos) < MERGE_DISTANCE_M), None)
        if group is None:
            groups.append([p])
        else:
            group.append(p)
    return [(np.mean([p.pos for p in g], axis=0), np.mean([p.velocity for p in g], axis=0)) for g in groups]


def _quantise(x: float, y: float, v: float) -> Target:
    # The sensor reports whole millimetres and cm/s.
    return Target(round(float(x), 3), round(float(y), 3), round(float(v), 2))


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)
