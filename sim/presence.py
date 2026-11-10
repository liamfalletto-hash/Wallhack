"""HLK-LD2410 presence sensor model: distance to the nearest moving and nearest still person.

It complements the LD2450, which loses people who stop moving. Numbers are estimates
to recalibrate on the real sensor; STILL_WALL_EXPONENT is the most uncertain one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sim import floorplan as fpl
from sim import radar
from sim.floorplan import FloorPlan
from sim.people import Person
from sim.radar import Pose

GATE_M = 0.75  # default distance gate size; distances are only known to within one gate
MAX_RANGE_M = 6.0
MOVING_DETECTION_P = 0.95
STILL_DETECTION_P = 0.9  # breathing and small shifts, in the open
# Breathing returns are tiny, so a wall costs still detection more than it costs the LD2450:
# per-wall factor = radar.WALL_DETECTION_FACTOR ** STILL_WALL_EXPONENT.
STILL_WALL_EXPONENT = 2.0
GATE_JITTER_P = 0.15  # chance the reading lands in a neighbouring gate
DETECTION_CORRELATION = 0.6  # misses come in bursts, as in sim.radar


@dataclass(frozen=True)
class Presence:
    moving_m: float | None
    still_m: float | None

    @property
    def state(self) -> str:
        return {(False, False): "none", (True, False): "moving", (False, True): "still", (True, True): "both"}[
            (self.moving_m is not None, self.still_m is not None)
        ]

    def to_dict(self) -> dict:
        return {"state": self.state, "moving_m": self.moving_m, "still_m": self.still_m}


@dataclass(frozen=True)
class PresenceState:
    visible: dict[int, bool]


def initial_state() -> PresenceState:
    return PresenceState(visible={})


def gate_distance(r: float) -> float:
    return (np.floor(r / GATE_M) + 0.5) * GATE_M  # centre of the gate


def detection_probability(fp: FloorPlan, helmet: Pose, person: Person) -> float:
    xs, ys = radar.to_sensor(helmet, person.pos)
    if not radar.in_view(xs, ys) or np.hypot(xs, ys) > MAX_RANGE_M:
        return 0.0
    still = person.speed < radar.STILL_SPEED_MPS
    p = STILL_DETECTION_P if still else MOVING_DETECTION_P
    for material in fpl.crossed_walls(fp, helmet.xy, person.pos):
        factor = radar.WALL_DETECTION_FACTOR[material]
        p *= factor**STILL_WALL_EXPONENT if still else factor
    return p


def sense(
    state: PresenceState, fp: FloorPlan, helmet: Pose, people: Sequence[Person], rng: np.random.Generator
) -> tuple[PresenceState, Presence]:
    visible: dict[int, bool] = {}
    moving: list[float] = []
    still: list[float] = []
    for person in people:
        p = detection_probability(fp, helmet, person)
        was_visible = state.visible[person.id] if person.id in state.visible else rng.random() < p
        p_now = p + (1 - p) * DETECTION_CORRELATION if was_visible else p * (1 - DETECTION_CORRELATION)
        visible[person.id] = p > 0 and rng.random() < p_now
        if visible[person.id]:
            r = float(np.linalg.norm(person.pos - helmet.xy))
            (still if person.speed < radar.STILL_SPEED_MPS else moving).append(r)
    return PresenceState(visible), Presence(_reading(moving, rng), _reading(still, rng))


def _reading(ranges: list[float], rng: np.random.Generator) -> float | None:
    if not ranges:
        return None
    r = gate_distance(min(ranges))
    if rng.random() < GATE_JITTER_P:
        r += GATE_M * rng.choice([-1, 1])
    return round(float(np.clip(r, GATE_M / 2, gate_distance(MAX_RANGE_M - 1e-9))), 3)
