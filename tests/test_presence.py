from pathlib import Path

import numpy as np
import pytest

from sim import floorplan as fpl
from sim import presence, radar
from sim.people import Person
from sim.radar import Pose

DEMO = Path(__file__).parent.parent / "maps" / "demo_flat.json"
HELMET = Pose(2.0, 2.2, 0.0)  # bedroom, facing +x towards the drywall at x=4


@pytest.fixture(scope="module")
def fp() -> fpl.FloorPlan:
    return fpl.load(DEMO)


def person(pid: int, x: float, y: float, speed: float = 0.0) -> Person:
    return Person(pid, np.array([x, y]), np.pi / 2, speed, 0.0, "walk" if speed else "sit", (), 1.0, 0.0, 0.0)


def readings(fp, people, n=10000, seed=0) -> list[presence.Presence]:
    rng, state, out = np.random.default_rng(seed), presence.initial_state(), []
    for _ in range(n):
        state, reading = presence.sense(state, fp, HELMET, people, rng)
        out.append(reading)
    return out


@pytest.mark.parametrize("r, gate", [(0.1, 0.375), (0.75, 1.125), (3.2, 3.375), (5.99, 5.625)])
def test_gate_distance_is_gate_centre(r, gate):
    assert presence.gate_distance(r) == pytest.approx(gate)


def test_still_detection_through_wall_is_penalised_more_than_moving(fp):
    open_still = presence.detection_probability(fp, HELMET, person(0, 3.0, 2.2))
    wall_still = presence.detection_probability(fp, HELMET, person(0, 5.0, 1.0))
    wall_moving = presence.detection_probability(fp, HELMET, person(0, 5.0, 1.0, speed=1.0))
    drywall = radar.WALL_DETECTION_FACTOR["drywall"]
    assert open_still == pytest.approx(presence.STILL_DETECTION_P)
    assert wall_still == pytest.approx(presence.STILL_DETECTION_P * drywall**presence.STILL_WALL_EXPONENT)
    assert wall_moving == pytest.approx(presence.MOVING_DETECTION_P * drywall)
    assert presence.detection_probability(fp, HELMET, person(0, 1.0, 2.2)) == 0.0  # behind the wearer


def test_still_person_rate_and_distance(fp):
    p = person(0, 5.0, 1.0)  # behind drywall, 3.23 m away
    rs = readings(fp, [p])
    hits = [r.still_m for r in rs if r.still_m is not None]
    assert all(r.moving_m is None for r in rs)
    assert len(hits) / len(rs) == pytest.approx(presence.detection_probability(fp, HELMET, p), abs=0.03)
    true_gate = presence.gate_distance(np.hypot(3.0, 1.2))
    assert np.mean(np.isclose(hits, true_gate)) == pytest.approx(1 - presence.GATE_JITTER_P, abs=0.03)
    assert all(abs(h - true_gate) <= presence.GATE_M + 1e-9 for h in hits)


def test_reports_nearest_of_each_kind(fp):
    near_still, far_still, walker = person(0, 3.0, 2.2), person(1, 3.0, 3.5), person(2, 3.5, 1.5, speed=1.0)
    rs = [r for r in readings(fp, [near_still, far_still, walker], n=3000) if r.state == "both"]
    assert rs
    assert np.median([r.still_m for r in rs]) == pytest.approx(presence.gate_distance(1.0))
    assert np.median([r.moving_m for r in rs]) == pytest.approx(presence.gate_distance(np.hypot(1.5, 0.7)))


def test_state_names():
    assert presence.Presence(None, None).state == "none"
    assert presence.Presence(1.0, None).state == "moving"
    assert presence.Presence(None, 2.0).to_dict() == {"state": "still", "moving_m": None, "still_m": 2.0}
    assert presence.Presence(1.0, 2.0).state == "both"
