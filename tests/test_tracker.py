import numpy as np
import pytest

from host import tracker
from host.tracker import Detection, TrackerState

DT = 0.1
NOISE_M = 0.1
R = np.eye(2) * NOISE_M**2


def run(positions_per_frame, seed=0, noise=NOISE_M):
    """positions_per_frame: list of lists of true (x, y); None entries are missed detections."""
    rng, state, history = np.random.default_rng(seed), TrackerState(), []
    for k, positions in enumerate(positions_per_frame):
        dets = [Detection(np.asarray(p) + rng.normal(0, noise, 2), R) for p in positions if p is not None]
        state = tracker.step(state, dets, k * DT)
        history.append(state)
    return history


def walk(start, velocity, n):
    return [np.asarray(start, float) + np.asarray(velocity, float) * k * DT for k in range(n)]


def nearest(state, p):
    return min(state.confirmed, key=lambda tr: np.linalg.norm(tr.pos - p))


def test_straight_walk_converges_to_position_and_velocity():
    path = walk((4.5, 1.0), (1.0, 0.3), 50)
    history = run([[p] for p in path])
    final = history[-1]
    assert len(final.confirmed) == 1
    tr = final.confirmed[0]
    assert np.linalg.norm(tr.pos - path[-1]) < 0.15
    late_vel = np.mean([s.confirmed[0].vel for s in history[-20:]], axis=0)  # single frames are noisy
    assert np.linalg.norm(late_vel - [1.0, 0.3]) < 0.1
    np.testing.assert_allclose(tracker.forecast(tr, 1.0), tr.pos + tr.vel)


def test_confirmation_needs_three_hits():
    history = run([[(5.0, 2.0)]] * 3)
    assert [len(s.confirmed) for s in history] == [0, 0, 1]


def test_crossing_people_keep_their_ids():
    a = walk((4.5, 1.0), (1.0, 1.0), 30)  # diagonal paths that cross at about 1.5 s
    b = walk((4.5, 4.0), (1.0, -1.0), 30)
    history = run([[pa, pb] for pa, pb in zip(a, b, strict=True)], seed=1)
    start, end = history[5], history[-1]
    assert len(end.confirmed) == 2
    assert nearest(start, a[5]).id == nearest(end, a[-1]).id
    assert nearest(start, b[5]).id == nearest(end, b[-1]).id


def test_isolated_false_targets_never_confirm():
    rng = np.random.default_rng(3)
    # 10% of frames carry a ghost, 5x the simulator's rate, scattered over the living room.
    frames = [[tuple(rng.uniform((4, 0), (8, 5)))] if rng.random() < 0.1 else [] for _ in range(1000)]
    assert all(not s.confirmed for s in run(frames, noise=0.0))


def test_track_coasts_through_gap_then_is_deleted():
    path = walk((4.5, 2.0), (0.8, 0.0), 80)
    gap = [[p] for p in path[:20]] + [[None]] * 20 + [[p] for p in path[40:]]
    history = run(gap)
    track_id = history[19].confirmed[0].id
    assert [tr.id for tr in history[39].confirmed] == [track_id]  # 2 s without detections
    assert [tr.id for tr in history[-1].confirmed] == [track_id]  # picked up again, same id

    lost = run([[p] for p in path[:20]] + [[]] * 40)
    assert not lost[-1].tracks  # gone after MAX_COAST_CONFIRMED_S


def test_detection_covariance_from_polar_noise():
    frame = {"pose": {"x": 1.0, "y": 1.0}, "yaw": 0.0, "targets": [{"x": 0.0, "y": 3.0, "v": 0.0}]}
    (d,) = tracker.detections_from_frame(frame)
    np.testing.assert_allclose(d.z, [4.0, 1.0])
    # Radar faces world +x: range noise lies along x, angle noise across it (3 m lever arm).
    np.testing.assert_allclose(np.diag(d.R), [tracker.RANGE_STD_M**2, (3 * tracker.ANGLE_STD_RAD) ** 2], rtol=1e-9)
    assert d.R[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_detection_covariance_rotates_with_yaw():
    frame = {"pose": {"x": 0.0, "y": 0.0}, "yaw": np.pi / 2, "targets": [{"x": 0.0, "y": 3.0, "v": 0.0}]}
    (d,) = tracker.detections_from_frame(frame)
    np.testing.assert_allclose(d.z, [0.0, 3.0], atol=1e-12)
    assert d.R[1, 1] == pytest.approx(tracker.RANGE_STD_M**2)  # now along world y
    assert tracker.detections_from_frame({"targets": []}) == []


def test_covariance_stays_symmetric_positive_definite():
    final = run([[p] for p in walk((4.5, 1.0), (1.0, 0.0), 100)])[-1]
    P = final.tracks[0].P
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(P) > 0)
