"""Kalman primitives on the shared state [x, y, vx, vy] with position-only measurements."""

from __future__ import annotations

import numpy as np

H = np.hstack([np.eye(2), np.zeros((2, 2))])


def constant_velocity(dt: float, accel_std: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * accel_std**2
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = q
    Q[np.ix_([1, 3], [1, 3])] = q
    return F, Q


def stationary(dt: float, drift_std: float, vel_std: float) -> tuple[np.ndarray, np.ndarray]:
    # Velocity is reset to zero each step; position only drifts (shifting in a chair, swaying).
    F = np.diag([1.0, 1.0, 0.0, 0.0])
    Q = np.diag([drift_std**2 * dt, drift_std**2 * dt, vel_std**2, vel_std**2])
    return F, Q


def predict(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return F @ x, F @ P @ F.T + Q


def innovation(x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return z - H @ x, H @ P @ H.T + R


def mahalanobis2(x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray) -> float:
    y, S = innovation(x, P, z, R)
    return float(y @ np.linalg.solve(S, y))


def update(x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Updated (x, P) and the measurement's log-likelihood under this model."""
    y, S = innovation(x, P, z, R)
    K = P @ H.T @ np.linalg.inv(S)
    IKH = np.eye(4) - K @ H
    P_new = IKH @ P @ IKH.T + K @ R @ K.T  # Joseph form keeps P symmetric and positive
    loglik = -0.5 * float(y @ np.linalg.solve(S, y) + np.log(np.linalg.det(2 * np.pi * S)))
    return x + K @ y, P_new, loglik
