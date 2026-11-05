"""Simulated people walking realistic routes through a floor plan (ground truth only)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from sim import floorplan as fpl
from sim.floorplan import FloorPlan

SIM_DT_S = 0.02  # 50 Hz motion step; the radar samples this at 10 Hz
CRUISE_SPEED_MPS = (1.0, 1.4)  # per-walk cruise speed range
SPEED_JITTER_MPS = 0.05  # stride-to-stride speed noise (std dev of the OU process)
SPEED_JITTER_TAU_S = 1.0
ACCEL_MPS2 = 1.0
DECEL_MPS2 = 1.5
MAX_TURN_RATE_RADPS = np.radians(120.0)
WAYPOINT_REACHED_M = 0.3
GOAL_REACHED_M = 0.05
TURN_MODE_RATE_RADPS = np.radians(30.0)  # above this a walking person counts as turning
STOPPED_MPS = 0.05
SIT_PROBABILITY = 0.3
STAND_PAUSE_S = (1.0, 6.0)
SIT_PAUSE_S = (8.0, 25.0)
SAME_ROOM_PROBABILITY = 0.3  # chance the next destination is in the current room
MAX_GOAL_TRIES = 20
MIN_WALL_GAP_M = 0.12  # hard stop distance from walls and furniture
MAX_JITTER_MPS = 0.15


@dataclass(frozen=True)
class Person:
    id: int
    pos: np.ndarray  # (2,) world frame, metres
    heading: float  # radians, CCW from +x
    speed: float  # m/s along heading
    turn_rate: float  # rad/s over the last step
    mode: str  # "walk" | "turn" | "stand" | "sit"
    route: tuple[np.ndarray, ...]  # remaining waypoints, last one is the goal
    cruise: float  # target walking speed for this route
    jitter: float  # current speed noise
    pause_left: float  # seconds until the next walk starts

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])


def spawn(fp: FloorPlan, rng: np.random.Generator, pid: int) -> Person:
    pos = fpl.sample_free_point(fp, rng)
    return Person(
        id=pid,
        pos=pos,
        heading=float(rng.uniform(-np.pi, np.pi)),
        speed=0.0,
        turn_rate=0.0,
        mode="stand",
        route=(),
        cruise=0.0,
        jitter=0.0,
        pause_left=float(rng.uniform(*STAND_PAUSE_S)),
    )


def step(p: Person, fp: FloorPlan, rng: np.random.Generator, dt: float = SIM_DT_S) -> Person:
    if not p.route:
        if p.pause_left > 0:
            return replace(p, speed=0.0, turn_rate=0.0, pause_left=p.pause_left - dt)
        route = _new_route(fp, rng, p.pos)
        if route is None:
            return replace(p, pause_left=float(rng.uniform(*STAND_PAUSE_S)))
        p = replace(p, route=route, cruise=float(rng.uniform(*CRUISE_SPEED_MPS)), mode="walk")
    return _walk(p, fp, rng, dt)


def simulate(
    fp: FloorPlan, n_people: int, duration_s: float, seed: int, dt: float = SIM_DT_S
) -> Iterator[tuple[float, tuple[Person, ...]]]:
    rng = np.random.default_rng(seed)
    people = tuple(spawn(fp, rng, i) for i in range(n_people))
    for k in range(int(round(duration_s / dt)) + 1):
        yield k * dt, people
        people = tuple(step(p, fp, rng, dt) for p in people)


def _walk(p: Person, fp: FloorPlan, rng: np.random.Generator, dt: float) -> Person:
    route = p.route
    while len(route) > 1 and np.linalg.norm(route[0] - p.pos) < WAYPOINT_REACHED_M:
        route = route[1:]
    goal = route[-1]
    to_wp = route[0] - p.pos
    dist_goal = float(np.linalg.norm(goal - p.pos)) if len(route) == 1 else np.inf

    if dist_goal < GOAL_REACHED_M or (dist_goal < WAYPOINT_REACHED_M and p.speed < STOPPED_MPS):
        sit = rng.random() < SIT_PROBABILITY
        pause = rng.uniform(*(SIT_PAUSE_S if sit else STAND_PAUSE_S))
        return replace(p, speed=0.0, turn_rate=0.0, route=(), mode="sit" if sit else "stand", pause_left=float(pause))

    err = _wrap(float(np.arctan2(to_wp[1], to_wp[0])) - p.heading)
    turn = float(np.clip(err, -MAX_TURN_RATE_RADPS * dt, MAX_TURN_RATE_RADPS * dt))
    heading = _wrap(p.heading + turn)

    # Slow down for sharp turns and for the final stop, like a person does.
    jitter = (
        p.jitter
        - p.jitter * dt / SPEED_JITTER_TAU_S
        + SPEED_JITTER_MPS * np.sqrt(2 * dt / SPEED_JITTER_TAU_S) * rng.standard_normal()
    )
    jitter = float(np.clip(jitter, -MAX_JITTER_MPS, MAX_JITTER_MPS))
    target = (p.cruise + jitter) * max(0.0, np.cos(err)) ** 2
    target = min(target, np.sqrt(2 * DECEL_MPS2 * dist_goal))
    rate = ACCEL_MPS2 if target > p.speed else DECEL_MPS2
    speed = float(np.clip(target, p.speed - rate * dt, p.speed + rate * dt))

    new_pos = p.pos + speed * dt * np.array([np.cos(heading), np.sin(heading)])
    if fpl.point_clearance(fp, new_pos) < MIN_WALL_GAP_M:
        # Turn-rate limits can overshoot a corner; stop rather than clip through a wall.
        # Steps are a few cm, far below MIN_WALL_GAP_M, so a point check cannot tunnel.
        new_pos, speed = p.pos, 0.0
    mode = "turn" if abs(turn) / dt > TURN_MODE_RATE_RADPS else ("walk" if speed > STOPPED_MPS else "stand")
    return replace(
        p, pos=new_pos, heading=heading, speed=speed, turn_rate=turn / dt, mode=mode, route=route, jitter=jitter
    )


def _new_route(fp: FloorPlan, rng: np.random.Generator, pos: np.ndarray) -> tuple[np.ndarray, ...] | None:
    here = fpl.room_of(fp, pos)
    others = [r for r in fp.rooms if r != here]
    for _ in range(MAX_GOAL_TRIES):
        room = here if (not others or rng.random() < SAME_ROOM_PROBABILITY) else others[rng.integers(len(others))]
        goal = fpl.sample_free_point(fp, rng, room)
        if np.linalg.norm(goal - pos) < 1.0:
            continue
        route = fpl.plan_route(fp, pos, goal)
        if route is not None:
            return tuple(route)
    return None


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi
