import json
from pathlib import Path

import numpy as np
import pytest

from sim import floorplan as fpl
from sim import radar, run

DEMO = Path(__file__).parent.parent / "maps" / "demo_flat.json"
FRAME_KEYS = {"v", "src", "seq", "t_dev", "t", "yaw", "pose", "targets", "presence", "truth"}


@pytest.fixture(scope="module")
def frames() -> list[dict]:
    fp = fpl.load(DEMO)
    return list(run.frames(fp, run.parse_helmet(run.DEFAULT_HELMET), n_people=2, duration_s=60.0, seed=5))


def test_frames_follow_the_documented_format(frames):
    assert len(frames) == 601
    for k, f in enumerate(frames):
        assert set(f) == FRAME_KEYS
        assert f["src"] == "sim" and f["seq"] == k + 1
        assert f["t_dev"] == pytest.approx(k * radar.FRAME_PERIOD_S)
        assert len(f["targets"]) <= radar.MAX_TARGETS
        assert all(set(t) == {"x", "y", "v"} for t in f["targets"])
        assert f["presence"]["state"] in {"none", "moving", "still", "both"}
        assert [p["id"] for p in f["truth"]["people"]] == [0, 1]
        json.dumps(f)


def test_some_frames_detect_people_and_some_miss(frames):
    counts = [len(f["targets"]) for f in frames]
    assert 0 in counts and max(counts) >= 1


def test_same_seed_gives_identical_recording(frames):
    fp = fpl.load(DEMO)
    again = list(run.frames(fp, run.parse_helmet(run.DEFAULT_HELMET), n_people=2, duration_s=60.0, seed=5))
    assert again == frames


def test_main_writes_header_and_frames(tmp_path):
    out = tmp_path / "s.jsonl"
    run.main(["--map", str(DEMO), "--people", "1", "--duration", "5", "--seed", "2", "--out", str(out)])
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[0]["type"] == "header" and lines[0]["src"] == "sim"
    assert len(lines) == 1 + 51


def test_bad_helmet_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="x,y,yaw_deg"):
        run.parse_helmet("2,2")
    with pytest.raises(SystemExit):
        run.main(["--map", str(DEMO), "--helmet", "6.1,1.9,0", "--out", str(tmp_path / "x.jsonl")])  # table


def test_helmet_parsing_converts_degrees():
    assert run.parse_helmet("1,2,90").yaw == pytest.approx(np.pi / 2)
