from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from sim import floorplan as fpl
from sim import people

DEMO = Path(__file__).parent.parent / "maps" / "demo_flat.json"
DURATION_S = 300.0


@pytest.fixture(scope="module")
def fp() -> fpl.FloorPlan:
    return fpl.load(DEMO)


@pytest.fixture(scope="module")
def run(fp) -> list[tuple[float, tuple[people.Person, ...]]]:
    return list(people.simulate(fp, n_people=3, duration_s=DURATION_S, seed=7))


def test_nobody_walks_through_walls_or_furniture(fp, run):
    for (_, before), (_, after) in pairwise(run):
        for a, b in zip(before, after, strict=True):
            assert not fpl.crossed_walls(fp, a.pos, b.pos)
            assert not fpl.inside_obstacle(fp, b.pos)
            assert fpl.point_clearance(fp, b.pos) >= people.MIN_WALL_GAP_M


def test_motion_stays_within_human_limits(run):
    max_speed = people.CRUISE_SPEED_MPS[1] + people.MAX_JITTER_MPS
    for _, ps in run:
        for p in ps:
            assert 0.0 <= p.speed <= max_speed + 1e-9
            assert abs(p.turn_rate) <= people.MAX_TURN_RATE_RADPS + 1e-9


def test_position_changes_match_reported_velocity(run):
    dt = people.SIM_DT_S
    for (_, before), (_, after) in pairwise(run):
        for a, b in zip(before, after, strict=True):
            np.testing.assert_allclose(b.pos - a.pos, b.velocity * dt, atol=1e-9)


def test_every_behaviour_occurs_and_people_move_between_rooms(fp, run):
    modes = Counter(p.mode for _, ps in run for p in ps)
    assert set(modes) == {"walk", "turn", "stand", "sit"}
    for pid in range(3):
        rooms = {fpl.room_of(fp, ps[pid].pos) for _, ps in run[::50]}
        assert rooms == {"bedroom", "living"}


def test_same_seed_is_reproducible(fp):
    a = [p.pos for _, ps in people.simulate(fp, 2, 30.0, seed=3) for p in ps]
    b = [p.pos for _, ps in people.simulate(fp, 2, 30.0, seed=3) for p in ps]
    c = [p.pos for _, ps in people.simulate(fp, 2, 30.0, seed=4) for p in ps]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_timestamps_are_uniform(run):
    t = np.array([t for t, _ in run])
    np.testing.assert_allclose(np.diff(t), people.SIM_DT_S)
    assert t[-1] == pytest.approx(DURATION_S)
