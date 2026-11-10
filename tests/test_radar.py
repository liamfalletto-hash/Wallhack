from pathlib import Path

import numpy as np
import pytest

from sim import floorplan as fpl
from sim import radar
from sim.people import Person
from sim.radar import Pose

DEMO = Path(__file__).parent.parent / "maps" / "demo_flat.json"
HELMET = Pose(2.0, 2.2, 0.0)  # bedroom, facing +x towards the drywall at x=4
N_FRAMES = 20000


@pytest.fixture(scope="module")
def fp() -> fpl.FloorPlan:
    return fpl.load(DEMO)


def person(pid: int, x: float, y: float, heading: float = np.pi / 2, speed: float = 1.0) -> Person:
    return Person(pid, np.array([x, y]), heading, speed, 0.0, "walk" if speed else "stand", (), 1.0, 0.0, 0.0)


def run_frames(fp, people, helmet=HELMET, n=N_FRAMES, seed=0):
    rng, state = np.random.default_rng(seed), radar.initial_state()
    for _ in range(n):
        state, targets, yaw = radar.sense(state, fp, helmet, people, rng)
        yield targets, yaw


def near(targets, helmet, p, radius=1.0):
    xs, ys = radar.to_sensor(helmet, p.pos)
    return [t for t in targets if np.hypot(t.x - xs, t.y - ys) < radius]


def test_to_sensor_axes():
    np.testing.assert_allclose(radar.to_sensor(Pose(0, 0, 0), [2, 0]), [0, 2], atol=1e-12)
    np.testing.assert_allclose(radar.to_sensor(Pose(0, 0, 0), [2, -1]), [1, 2], atol=1e-12)  # world -y is right
    np.testing.assert_allclose(radar.to_sensor(Pose(1, 1, np.pi / 2), [1, 4]), [0, 3], atol=1e-12)


@pytest.mark.parametrize(
    "xs, ys, expected",
    [(0, 3, True), (0, 6.5, False), (0, 0.1, False), (0, -2, False), (2.5, 2, True), (3.5, 2, False)],
)
def test_field_of_view(xs, ys, expected):
    assert radar.in_view(xs, ys) is expected


def test_detection_probability_factors(fp):
    base = radar.BASE_DETECTION_P
    assert radar.detection_probability(fp, HELMET, person(0, 3.0, 2.2)) == pytest.approx(base)
    behind_drywall = radar.detection_probability(fp, HELMET, person(0, 5.0, 1.0))
    assert behind_drywall == pytest.approx(base * radar.WALL_DETECTION_FACTOR["drywall"])
    still = radar.detection_probability(fp, HELMET, person(0, 5.0, 1.0, speed=0.0))
    assert still == pytest.approx(behind_drywall * radar.STILL_DETECTION_FACTOR)
    assert radar.detection_probability(fp, HELMET, person(0, 1.0, 2.2)) == 0.0  # behind the wearer


def test_detection_rate_behind_wall_matches_model_and_misses_are_bursty(fp):
    p = person(0, 5.0, 1.0)
    detected = np.array([bool(near(t, HELMET, p)) for t, _ in run_frames(fp, [p])])
    expected = radar.detection_probability(fp, HELMET, p)
    assert detected.mean() == pytest.approx(expected, abs=0.02)
    after_miss = detected[1:][~detected[:-1]]
    assert after_miss.mean() < expected - 0.2  # a miss makes the next miss more likely


def test_measurement_noise_matches_model(fp):
    p = person(0, 5.0, 2.2)  # straight ahead through the doorway, 3 m away
    hits = [near(t, HELMET, p)[0] for t, _ in run_frames(fp, [p], n=5000) if near(t, HELMET, p)]
    r = np.array([np.hypot(h.x, h.y) for h in hits])
    lateral = np.array([h.x for h in hits])
    assert r.mean() - 3.0 == pytest.approx(-radar.BODY_REFLECTION_OFFSET_M, abs=0.01)
    assert r.std() == pytest.approx(radar.RANGE_NOISE_M, rel=0.1)
    assert lateral.std() == pytest.approx(3.0 * np.sin(radar.ANGLE_NOISE_RAD), rel=0.1)


def test_radial_speed_sign(fp):
    away = person(0, 3.0, 2.2, heading=0.0, speed=1.2)
    towards = person(0, 3.0, 2.2, heading=np.pi, speed=1.2)
    for p, sign in ((away, 1), (towards, -1)):
        v = [near(t, HELMET, p)[0].v for t, _ in run_frames(fp, [p], n=2000) if near(t, HELMET, p)]
        assert np.mean(v) == pytest.approx(sign * 1.2, abs=0.02)


def test_close_people_merge_into_one_target(fp, monkeypatch):
    monkeypatch.setattr(radar, "FALSE_TARGETS_PER_FRAME", 0.0)
    a, b = person(0, 3.0, 2.0), person(1, 3.0, 2.3)
    counts = [len(near(t, HELMET, a)) for t, _ in run_frames(fp, [a, b], n=2000)]
    assert max(counts) == 1


def test_at_most_three_targets_nearest_first_and_quantised(fp):
    crowd = [person(i, 2.8 + 0.6 * i, 2.45) for i in range(5)]
    for targets, _ in run_frames(fp, crowd, n=2000):
        assert len(targets) <= radar.MAX_TARGETS
        ranges = [np.hypot(t.x, t.y) for t in targets]
        assert ranges == sorted(ranges)
        for t in targets:
            assert radar.in_view(t.x, t.y)
            assert round(t.x * 1000) == pytest.approx(t.x * 1000, abs=1e-6)
            assert round(t.v * 100) == pytest.approx(t.v * 100, abs=1e-6)


def test_yaw_reading_stays_close_to_truth(fp):
    yaws = np.array([yaw for _, yaw in run_frames(fp, [], n=6000)])  # 10 minutes
    assert np.degrees(np.abs(yaws - HELMET.yaw)).mean() < 3.0
