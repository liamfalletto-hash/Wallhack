"""Conversions between the radar (head) frame and the room (world) frame."""

from __future__ import annotations

import numpy as np


def sensor_to_world(pose_xy: np.ndarray, yaw: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Head-frame points (x right, y forward) to world (N, 2), radar facing `yaw` CCW from world +x."""
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([np.sin(yaw), -np.cos(yaw)])
    xs, ys = np.atleast_1d(xs).astype(float), np.atleast_1d(ys).astype(float)
    return np.asarray(pose_xy, float) + ys[:, None] * forward + xs[:, None] * right


def frame_targets_world(frame: dict) -> np.ndarray:
    """World positions of a frame's targets; missing pose or yaw fall back to the origin / 0 rad."""
    pose = frame.get("pose") or {"x": 0.0, "y": 0.0}
    targets = frame.get("targets", [])
    if not targets:
        return np.empty((0, 2))
    xs = np.array([t["x"] for t in targets])
    ys = np.array([t["y"] for t in targets])
    return sensor_to_world(np.array([pose["x"], pose["y"]]), frame.get("yaw") or 0.0, xs, ys)
