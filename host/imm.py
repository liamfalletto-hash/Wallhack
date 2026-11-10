"""Interacting multiple model (IMM) filter: walking, manoeuvring and still, on one shared state.

Missed detections are evidence too: the LD2450 misses still people far more often than
walking ones, so each miss shifts probability towards "still" instead of letting the
track slide on at its last speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from host import kalman

MODES = ("walking", "manoeuvring", "still")
WALK_ACCEL_STD_MPS2 = 0.5
MANOEUVRE_ACCEL_STD_MPS2 = 3.0
STILL_DRIFT_STD_MPS = 0.05
STILL_VEL_STD_MPS = 0.01
# Per-frame (10 Hz) switching probabilities, row = from, column = to.
TRANSITION = np.array(
    [
        [0.96, 0.03, 0.01],
        [0.10, 0.80, 0.10],
        [0.01, 0.03, 0.96],
    ]
)
# Chance the LD2450 detects a person in each mode; measured on the simulator for now,
# to be replaced by a calibration run on the real sensor.
DETECTION_P = np.array([0.80, 0.80, 0.35])
INITIAL_MU = np.array([0.4, 0.2, 0.4])


@dataclass(frozen=True)
class IMMState:
    x: np.ndarray  # (3, 4) per-model state
    P: np.ndarray  # (3, 4, 4) per-model covariance
    mu: np.ndarray  # (3,) model probabilities, sum to 1

    @property
    def mode(self) -> str:
        return MODES[int(np.argmax(self.mu))]


def init(x: np.ndarray, P: np.ndarray) -> IMMState:
    return IMMState(np.repeat(x[None], 3, axis=0), np.repeat(P[None], 3, axis=0), INITIAL_MU.copy())


def predict(s: IMMState, dt: float) -> IMMState:
    if dt <= 0:
        return s
    c = TRANSITION.T @ s.mu  # predicted mode probabilities
    w = TRANSITION * s.mu[:, None] / c[None, :]  # w[i, j] = P(was i | now j)
    xs, Ps = np.empty_like(s.x), np.empty_like(s.P)
    for j, (F, Q) in enumerate(_models(dt)):
        x0 = w[:, j] @ s.x
        d = s.x - x0
        P0 = np.einsum("i,ijk->jk", w[:, j], s.P + d[:, :, None] * d[:, None, :])
        xs[j], Ps[j] = kalman.predict(x0, P0, F, Q)
    return IMMState(xs, Ps, c)


def update(s: IMMState, z: np.ndarray, R: np.ndarray) -> IMMState:
    xs, Ps, logw = np.empty_like(s.x), np.empty_like(s.P), np.empty(3)
    for j in range(3):
        xs[j], Ps[j], loglik = kalman.update(s.x[j], s.P[j], z, R)
        logw[j] = loglik + np.log(DETECTION_P[j]) + np.log(s.mu[j])
    return IMMState(xs, Ps, _normalise(logw))


def miss(s: IMMState) -> IMMState:
    return IMMState(s.x, s.P, _normalise(np.log(1 - DETECTION_P) + np.log(s.mu)))


def reweight(s: IMMState, likelihood: np.ndarray) -> IMMState:
    """Fold in per-mode evidence from another sensor (e.g. the LD2410)."""
    return IMMState(s.x, s.P, _normalise(np.log(likelihood) + np.log(s.mu)))


def combine(s: IMMState) -> tuple[np.ndarray, np.ndarray]:
    x = s.mu @ s.x
    d = s.x - x
    return x, np.einsum("i,ijk->jk", s.mu, s.P + d[:, :, None] * d[:, None, :])


def forecast(s: IMMState, horizon_s: float) -> np.ndarray:
    # Walking and manoeuvring extrapolate their velocity; still stays put.
    per_model = s.x[:, :2] + s.x[:, 2:] * horizon_s
    per_model[2] = s.x[2, :2]
    return s.mu @ per_model


def _models(dt: float) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        kalman.constant_velocity(dt, WALK_ACCEL_STD_MPS2),
        kalman.constant_velocity(dt, MANOEUVRE_ACCEL_STD_MPS2),
        kalman.stationary(dt, STILL_DRIFT_STD_MPS, STILL_VEL_STD_MPS),
    ]


def _normalise(logw: np.ndarray) -> np.ndarray:
    w = np.exp(logw - logw.max())
    return w / w.sum()
