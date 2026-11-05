import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sim import floorplan as fpl

DEMO = Path(__file__).parent.parent / "maps" / "demo_flat.json"
RAW = json.loads(DEMO.read_text())


@pytest.fixture(scope="module")
def fp() -> fpl.FloorPlan:
    return fpl.parse(RAW)


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda d: d.update(v=2), "version"),
        (lambda d: d.update(units="ft"), "metres"),
        (lambda d: d["walls"][0].update(material="steel"), "materials"),
        (lambda d: d["walls"][0].update(b=d["walls"][0]["a"]), "zero-length"),
        (lambda d: d.update(walls=[]), "no walls"),
        (lambda d: d["obstacles"][0].update(polygon=[[0, 0], [1, 1]]), "at least 3"),
    ],
)
def test_parse_rejects_bad_config(mutate, msg):
    d = copy.deepcopy(RAW)
    mutate(d)
    with pytest.raises(ValueError, match=msg):
        fpl.parse(d)


def test_crossed_walls(fp):
    assert fpl.crossed_walls(fp, [2, 1], [6, 1]) == ["drywall"]
    assert fpl.crossed_walls(fp, [2, 2.45], [6, 2.45]) == []  # straight through the doorway
    assert sorted(fpl.crossed_walls(fp, [2, 1], [9, 1])) == ["brick", "drywall"]
    assert fpl.crossed_walls(fp, [1, 1], [3, 2]) == []


def test_crossed_walls_counts_grazing_contact(fp):
    assert fpl.crossed_walls(fp, [3, 1], [4, 1]) == ["drywall"]


def test_room_and_obstacle_lookup(fp):
    assert fpl.room_of(fp, np.array([1.0, 1.0])) == "bedroom"
    assert fpl.room_of(fp, np.array([6.0, 1.0])) == "living"
    assert fpl.inside_obstacle(fp, np.array([6.1, 1.9]))  # table
    assert not fpl.is_free_point(fp, np.array([6.1, 1.9]))
    assert not fpl.is_free_point(fp, np.array([0.1, 1.0]))  # too close to the outer wall


def test_route_goes_through_door_and_stays_clear(fp):
    start, goal = np.array([1.0, 1.0]), np.array([7.0, 1.0])
    route = fpl.plan_route(fp, start, goal)
    assert route is not None
    np.testing.assert_allclose(route[-1], goal)
    legs = list(zip([start, *route[:-1]], route, strict=True))
    assert all(fpl.clearance(fp, a, b) >= fpl.BODY_RADIUS_M for a, b in legs)
    crossings = [a[1] + (b[1] - a[1]) * (4 - a[0]) / (b[0] - a[0]) for a, b in legs if (a[0] - 4) * (b[0] - 4) < 0]
    assert len(crossings) == 1 and 2.0 < crossings[0] < 2.9  # interior wall at x=4, door spans y 2.0-2.9


def test_route_avoids_free_standing_furniture(fp):
    start, goal = np.array([6.1, 0.8]), np.array([6.1, 3.0])  # table sits in between
    route = fpl.plan_route(fp, start, goal)
    assert route is not None and len(route) >= 2
    assert all(not fpl.crossed_walls(fp, a, b) for a, b in zip([start, *route[:-1]], route, strict=True))


def test_unreachable_goal_returns_none(fp):
    sealed = copy.deepcopy(RAW)
    sealed["walls"].append({"a": [4, 2], "b": [4, 2.9], "material": "drywall"})
    sealed_fp = fpl.parse(sealed)
    assert fpl.plan_route(sealed_fp, np.array([1.0, 1.0]), np.array([7.0, 1.0])) is None


def test_sample_free_point_respects_room(fp):
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = fpl.sample_free_point(fp, rng, "living")
        assert fpl.room_of(fp, p) == "living" and fpl.is_free_point(fp, p)
