import numpy as np
import pytest

from host import benchmark
from sim import radar

POSE = radar.Pose(0.0, 0.0, 0.0)  # radar faces world +x
DT = 0.1


def frame(k: int, people: list[tuple[float, float, float, float]], ghosts=()) -> dict:
    """people: (x, y, vx, vy) in world; each gets a perfect detection, ghosts are extra detections."""
    targets = []
    for x, y in [(p[0], p[1]) for p in people] + list(ghosts):
        xs, ys = radar.to_sensor(POSE, (x, y))
        targets.append({"x": float(xs), "y": float(ys), "v": 0.0})
    return {
        "t_dev": k * DT,
        "yaw": POSE.yaw,
        "pose": {"x": POSE.x, "y": POSE.y},
        "targets": targets,
        "truth": {
            "pose": {"x": POSE.x, "y": POSE.y, "yaw": POSE.yaw},
            "people": [{"id": i, "x": x, "y": y, "vx": vx, "vy": vy} for i, (x, y, vx, vy) in enumerate(people)],
        },
    }


def test_constant_walk_favours_kalman_and_is_fully_covered():
    frames = [frame(k, [(1.0 + 0.1 * k, 0.5, 1.0, 0.0)]) for k in range(40)]
    r = benchmark.evaluate(frames)
    assert r.kalman_err.mean() < 0.1 < r.static_err.mean()
    assert r.walking.all()
    assert r.false_track_frames == 0
    assert r.coverage == pytest.approx(38 / 40)  # confirmation takes 3 frames


def test_still_person_and_ghost_count_as_still_and_false_track():
    frames = [frame(k, [(3.0, 0.0, 0.0, 0.0)], ghosts=[(3.0, 2.5)]) for k in range(40)]
    r = benchmark.evaluate(frames)
    assert not r.walking.any()
    assert r.static_err.max() == pytest.approx(0.0, abs=1e-6)
    assert r.false_track_frames == 38  # the ghost's track, never used as a forecast
    assert len(r.kalman_err) == 28  # confirmed from frame 2; frames 2-29 have a frame 1 s later


def test_combine_adds_counts_and_concatenates_errors():
    a = benchmark.evaluate([frame(k, [(3.0, 0.0, 0.0, 0.0)]) for k in range(20)])
    both = benchmark.combine([a, a])
    assert len(both.kalman_err) == 2 * len(a.kalman_err)
    assert (both.covered, both.visible) == (2 * a.covered, 2 * a.visible)
    np.testing.assert_array_equal(both.walking, np.concatenate([a.walking, a.walking]))
