"""Multi-person tracker: constant-velocity Kalman filter per person, Hungarian association.

State per track is [x, y, vx, vy] in the world frame (metres, m/s).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from host.geometry import sensor_to_world

# LD2450 measurement noise (polar); converted to a per-detection world covariance.
RANGE_STD_M = 0.08
ANGLE_STD_RAD = np.radians(3.0)
# People change speed and direction; white-acceleration noise for the constant-velocity model.
ACCEL_STD_MPS2 = 1.0
INIT_SPEED_STD_MPS = 1.0
GATE_CHI2 = 9.21  # 99% for 2 degrees of freedom
CONFIRM_HITS = 3  # within the last CONFIRM_WINDOW frames
CONFIRM_WINDOW = 5
MAX_COAST_CONFIRMED_S = 3.0
MAX_COAST_TENTATIVE_S = 0.5
UNASSIGNABLE = 1e9

H = np.hstack([np.eye(2), np.zeros((2, 2))])


@dataclass(frozen=True)
class Detection:
    z: np.ndarray  # (2,) world position
    R: np.ndarray  # (2, 2) world covariance


@dataclass(frozen=True)
class Track:
    id: int
    x: np.ndarray  # (4,) [x, y, vx, vy]
    P: np.ndarray  # (4, 4)
    last_update_t: float
    hits: tuple[bool, ...]  # recent frames, newest last, at most CONFIRM_WINDOW long
    confirmed: bool

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2]

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:]


@dataclass(frozen=True)
class TrackerState:
    tracks: tuple[Track, ...] = ()
    next_id: int = 0
    t: float | None = None

    @property
    def confirmed(self) -> tuple[Track, ...]:
        return tuple(tr for tr in self.tracks if tr.confirmed)


def detections_from_frame(frame: dict) -> list[Detection]:
    targets = frame.get("targets", [])
    if not targets:
        return []
    pose = frame.get("pose") or {"x": 0.0, "y": 0.0}
    yaw = frame.get("yaw") or 0.0
    xs = np.array([t["x"] for t in targets], float)
    ys = np.array([t["y"] for t in targets], float)
    world = sensor_to_world(np.array([pose["x"], pose["y"]]), yaw, xs, ys)
    # Columns map head-frame (right, forward) onto world axes.
    rot = np.column_stack([[np.sin(yaw), -np.cos(yaw)], [np.cos(yaw), np.sin(yaw)]])
    return [Detection(z, rot @ _sensor_cov(x, y) @ rot.T) for z, x, y in zip(world, xs, ys, strict=True)]


def step(state: TrackerState, detections: Sequence[Detection], t: float) -> TrackerState:
    dt = 0.0 if state.t is None else t - state.t
    tracks = [_predict(tr, dt) for tr in state.tracks]

    matched: dict[int, int] = {}
    if tracks and detections:
        cost = np.array([[_mahalanobis2(tr, d) for d in detections] for tr in tracks])
        gated = np.where(cost <= GATE_CHI2, cost, UNASSIGNABLE)
        rows, cols = linear_sum_assignment(gated)
        matched = {int(r): int(c) for r, c in zip(rows, cols, strict=True) if gated[r, c] < UNASSIGNABLE}

    updated: list[Track] = []
    for i, tr in enumerate(tracks):
        hit = i in matched
        if hit:
            tr = _update(tr, detections[matched[i]], t)
        hits = (*tr.hits, hit)[-CONFIRM_WINDOW:]
        tr = replace(tr, hits=hits, confirmed=tr.confirmed or sum(hits) >= CONFIRM_HITS)
        max_coast = MAX_COAST_CONFIRMED_S if tr.confirmed else MAX_COAST_TENTATIVE_S
        if t - tr.last_update_t <= max_coast:
            updated.append(tr)

    next_id = state.next_id
    for j in sorted(set(range(len(detections))) - set(matched.values())):
        updated.append(_new_track(next_id, detections[j], t))
        next_id += 1
    return TrackerState(tuple(updated), next_id, t)


def forecast(track: Track, horizon_s: float) -> np.ndarray:
    return track.pos + track.vel * horizon_s


def _sensor_cov(x: float, y: float) -> np.ndarray:
    r, az = np.hypot(x, y), np.arctan2(x, y)  # x = r sin(az), y = r cos(az)
    J = np.array([[np.sin(az), r * np.cos(az)], [np.cos(az), -r * np.sin(az)]])
    return J @ np.diag([RANGE_STD_M**2, ANGLE_STD_RAD**2]) @ J.T


def _predict(tr: Track, dt: float) -> Track:
    if dt <= 0:
        return tr
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * ACCEL_STD_MPS2**2
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = q
    Q[np.ix_([1, 3], [1, 3])] = q
    return replace(tr, x=F @ tr.x, P=F @ tr.P @ F.T + Q)


def _mahalanobis2(tr: Track, d: Detection) -> float:
    y = d.z - H @ tr.x
    S = H @ tr.P @ H.T + d.R
    return float(y @ np.linalg.solve(S, y))


def _update(tr: Track, d: Detection, t: float) -> Track:
    S = H @ tr.P @ H.T + d.R
    K = tr.P @ H.T @ np.linalg.inv(S)
    x = tr.x + K @ (d.z - H @ tr.x)
    IKH = np.eye(4) - K @ H
    P = IKH @ tr.P @ IKH.T + K @ d.R @ K.T  # Joseph form keeps P symmetric and positive
    return replace(tr, x=x, P=P, last_update_t=t)


def _new_track(track_id: int, d: Detection, t: float) -> Track:
    P = np.zeros((4, 4))
    P[:2, :2] = d.R
    P[2:, 2:] = np.eye(2) * INIT_SPEED_STD_MPS**2
    return Track(track_id, np.array([*d.z, 0.0, 0.0]), P, t, (True,), False)
