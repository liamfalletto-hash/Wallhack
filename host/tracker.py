"""Multi-person tracker: one motion filter per person, Hungarian association.

Filters: "cv" (constant-velocity Kalman, the baseline) or "imm" (walking / manoeuvring /
still, see host/imm.py). State per track is [x, y, vx, vy] in the world frame (metres, m/s).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from host import imm, kalman
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
MODELS = ("cv", "imm")
# LD2410 fusion (imm only). A reading is a 0.75 m gate centre and can land one gate off.
PRESENCE_MATCH_M = 0.75
STILL_READING_LIKELIHOOD = np.array([0.3, 0.3, 1.0])  # walking, manoeuvring, still
MOVING_READING_LIKELIHOOD = np.array([1.0, 1.0, 0.3])
# Range-only readings cannot correct position, so cap how long they alone keep a track.
MAX_COAST_WITH_PRESENCE_S = 20.0


@dataclass(frozen=True)
class Params:
    accel_std_mps2: float = ACCEL_STD_MPS2  # cv only
    max_coast_confirmed_s: float = MAX_COAST_CONFIRMED_S
    model: str = "cv"
    use_presence: bool = False  # LD2410 readings; needs model="imm"

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown tracker model {self.model!r}, choose from {MODELS}")
        if self.use_presence and self.model != "imm":
            raise ValueError("use_presence needs model='imm'")


DEFAULT_PARAMS = Params()


@dataclass(frozen=True)
class Detection:
    z: np.ndarray  # (2,) world position
    R: np.ndarray  # (2, 2) world covariance


@dataclass(frozen=True)
class PresenceReading:
    helmet: np.ndarray  # (2,) world position the distances are measured from
    moving_m: float | None
    still_m: float | None


@dataclass(frozen=True)
class Track:
    id: int
    x: np.ndarray  # (4,) [x, y, vx, vy]; for imm the probability-weighted combination
    P: np.ndarray  # (4, 4)
    last_update_t: float
    hits: tuple[bool, ...]  # recent frames, newest last, at most CONFIRM_WINDOW long
    confirmed: bool
    imm: imm.IMMState | None = None
    last_presence_t: float = -np.inf  # last LD2410 still reading that matched this track

    @property
    def mode(self) -> str | None:
        return self.imm.mode if self.imm else None

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


def presence_from_frame(frame: dict) -> PresenceReading | None:
    pr = frame.get("presence")
    if not pr:
        return None
    pose = frame.get("pose") or {"x": 0.0, "y": 0.0}
    return PresenceReading(np.array([pose["x"], pose["y"]]), pr.get("moving_m"), pr.get("still_m"))


def step(
    state: TrackerState,
    detections: Sequence[Detection],
    t: float,
    params: Params = DEFAULT_PARAMS,
    presence: PresenceReading | None = None,
) -> TrackerState:
    dt = 0.0 if state.t is None else t - state.t
    tracks = [_predict(tr, dt, params.accel_std_mps2) for tr in state.tracks]

    matched: dict[int, int] = {}
    if tracks and detections:
        cost = np.array([[kalman.mahalanobis2(tr.x, tr.P, d.z, d.R) for d in detections] for tr in tracks])
        gated = np.where(cost <= GATE_CHI2, cost, UNASSIGNABLE)
        rows, cols = linear_sum_assignment(gated)
        matched = {int(r): int(c) for r, c in zip(rows, cols, strict=True) if gated[r, c] < UNASSIGNABLE}

    still_owner = moving_owner = None
    if params.use_presence and presence is not None:
        still_owner = _presence_owner(tracks, presence.helmet, presence.still_m)
        moving_owner = _presence_owner(tracks, presence.helmet, presence.moving_m)

    updated: list[Track] = []
    for i, tr in enumerate(tracks):
        hit = i in matched
        if hit:
            tr = _update(tr, detections[matched[i]], t)
        elif tr.imm is not None:
            tr = _with_imm(tr, imm.miss(tr.imm))
        if i in (still_owner, moving_owner):
            tr = _apply_presence(tr, still=i == still_owner, moving=i == moving_owner, t=t)
        hits = (*tr.hits, hit)[-CONFIRM_WINDOW:]
        tr = replace(tr, hits=hits, confirmed=tr.confirmed or sum(hits) >= CONFIRM_HITS)
        if _alive(tr, t, params):
            updated.append(tr)

    next_id = state.next_id
    for j in sorted(set(range(len(detections))) - set(matched.values())):
        updated.append(_new_track(next_id, detections[j], t, params.model))
        next_id += 1
    return TrackerState(tuple(updated), next_id, t)


def forecast(track: Track, horizon_s: float) -> np.ndarray:
    if track.imm is not None:
        return imm.forecast(track.imm, horizon_s)
    return track.pos + track.vel * horizon_s


def _sensor_cov(x: float, y: float) -> np.ndarray:
    r, az = np.hypot(x, y), np.arctan2(x, y)  # x = r sin(az), y = r cos(az)
    J = np.array([[np.sin(az), r * np.cos(az)], [np.cos(az), -r * np.sin(az)]])
    return J @ np.diag([RANGE_STD_M**2, ANGLE_STD_RAD**2]) @ J.T


def _predict(tr: Track, dt: float, accel_std: float) -> Track:
    if dt <= 0:
        return tr
    if tr.imm is not None:
        return _with_imm(tr, imm.predict(tr.imm, dt))
    x, P = kalman.predict(tr.x, tr.P, *kalman.constant_velocity(dt, accel_std))
    return replace(tr, x=x, P=P)


def _update(tr: Track, d: Detection, t: float) -> Track:
    if tr.imm is not None:
        return replace(_with_imm(tr, imm.update(tr.imm, d.z, d.R)), last_update_t=t)
    x, P, _ = kalman.update(tr.x, tr.P, d.z, d.R)
    return replace(tr, x=x, P=P, last_update_t=t)


def _presence_owner(tracks: Sequence[Track], helmet: np.ndarray, reading_m: float | None) -> int | None:
    # The LD2410 reports one person per kind (the nearest), so a reading supports at most one
    # track: the imm track whose range is closest to it, if within PRESENCE_MATCH_M.
    if reading_m is None:
        return None
    gaps = [(abs(float(np.linalg.norm(tr.pos - helmet)) - reading_m), i) for i, tr in enumerate(tracks) if tr.imm]
    best = min(gaps, default=None)
    return best[1] if best is not None and best[0] <= PRESENCE_MATCH_M else None


def _apply_presence(tr: Track, still: bool, moving: bool, t: float) -> Track:
    likelihood = np.ones(3)
    if still:
        likelihood *= STILL_READING_LIKELIHOOD
    if moving:
        likelihood *= MOVING_READING_LIKELIHOOD
    tr = _with_imm(tr, imm.reweight(tr.imm, likelihood))
    return replace(tr, last_presence_t=t) if still and tr.mode == "still" else tr


def _alive(tr: Track, t: float, params: Params) -> bool:
    if not tr.confirmed:
        return t - tr.last_update_t <= MAX_COAST_TENTATIVE_S
    last_seen = max(tr.last_update_t, tr.last_presence_t)
    return t - last_seen <= params.max_coast_confirmed_s and t - tr.last_update_t <= MAX_COAST_WITH_PRESENCE_S


def _with_imm(tr: Track, s: imm.IMMState) -> Track:
    x, P = imm.combine(s)
    return replace(tr, x=x, P=P, imm=s)


def _new_track(track_id: int, d: Detection, t: float, model: str) -> Track:
    P = np.zeros((4, 4))
    P[:2, :2] = d.R
    P[2:, 2:] = np.eye(2) * INIT_SPEED_STD_MPS**2
    x = np.array([*d.z, 0.0, 0.0])
    track = Track(track_id, x, P, t, (True,), False)
    return _with_imm(track, imm.init(x, P)) if model == "imm" else track
