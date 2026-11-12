import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from host import ingest, viewer
from host.geometry import frame_targets_world, sensor_to_world
from sim import fake_helmet, radar, run
from sim import floorplan as fpl

DEMO = Path(__file__).parent.parent / "maps" / "demo_flat.json"


@pytest.mark.parametrize("yaw", [0.0, 0.7, -2.5])
def test_sensor_to_world_inverts_the_radar_transform(yaw):
    helmet = radar.Pose(2.0, 1.5, yaw)
    world = np.array([[3.0, 2.0], [5.5, 4.0], [1.0, 0.5]])
    sensor = np.array([radar.to_sensor(helmet, p) for p in world])
    np.testing.assert_allclose(sensor_to_world(helmet.xy, yaw, sensor[:, 0], sensor[:, 1]), world, atol=1e-12)


def test_frame_targets_world_handles_missing_imu_and_pose():
    frame = {"yaw": None, "pose": None, "targets": [{"x": 0.0, "y": 2.0, "v": 0.0}]}
    np.testing.assert_allclose(frame_targets_world(frame), [[2.0, 0.0]])  # yaw 0 faces world +x
    assert frame_targets_world({"targets": []}).shape == (0, 2)


def test_load_session_rejects_files_without_header(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"v": 1, "targets": []}) + "\n")
    with pytest.raises(ValueError, match="header"):
        viewer.load_session(bad)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    session = tmp_path_factory.mktemp("s") / "s.jsonl"
    run.main(["--map", str(DEMO), "--people", "2", "--duration", "3", "--seed", "1", "--out", str(session)])
    srv = viewer.serve(session, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.read()


def test_serves_page_map_and_session(server):
    assert b"<canvas" in get(server + "/")
    assert json.loads(get(server + "/map"))["name"] == "demo_flat"
    assert json.loads(get(server + "/session"))["n"] == 31


def test_event_stream_sends_frames_in_order_with_world_targets(server):
    body = get(server + "/events?speed=64&start=5").decode()
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith('data: {"')]
    assert [e["i"] for e in events] == list(range(5, 31))
    assert all(len(e["world"]) == len(e["targets"]) for e in events)
    assert any(e["tracks"] for e in events)
    assert all(len(t["path"]) == len(viewer.FORECAST_STEPS_S) for e in events for t in e["tracks"])
    assert {t["mode"] for e in events for t in e["tracks"]} <= {"walking", "manoeuvring", "still"}
    assert body.rstrip().endswith("event: end\ndata: {}")


def test_bad_query_is_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        get(server + "/events?speed=fast")
    assert err.value.code == 400


def firmware_like_frames(stop):
    """Simulated frames as the firmware sends them: no pose, raw IMU heading, no ground truth."""
    fp = fpl.load(DEMO)
    for f in run.frames(fp, run.parse_helmet(run.DEFAULT_HELMET), n_people=2, duration_s=30, seed=1):
        if stop.is_set():
            return
        time.sleep(0.005)
        yield fake_helmet.as_firmware(f)


def test_live_viewer_streams_tracks_and_records(tmp_path):
    helmet = ingest.HelmetPose(2.0, 2.2, 0.0)
    srv, stop, log_path = viewer.serve_live(firmware_like_frames, "test", helmet, DEMO, port=0, log_dir=tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        info = json.loads(get(base + "/session"))
        assert info["live"] and info["log"] == str(log_path) and info["header"]["src"] == "test"
        events = []
        with urllib.request.urlopen(base + "/events", timeout=5) as r:
            while len(events) < 60:
                line = r.readline().decode()
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        idx = [e["i"] for e in events]
        assert idx == sorted(idx) and len(set(idx)) == len(idx)
        assert all(e["pose"] == {"x": 2.0, "y": 2.2} and "t" in e for e in events)
        assert any(e["tracks"] for e in events)
    finally:
        stop.set()
        srv.shutdown()
    lines = log_path.read_text().splitlines()
    assert json.loads(lines[0])["type"] == "header" and len(lines) > 60


def test_demo_source_is_firmware_like_and_stops_promptly():
    stop = threading.Event()
    gen = fake_helmet.realtime_frames(stop, duration_s=5.0)
    first = next(gen)
    assert first["pose"] is None and "truth" not in first and first["src"] == "ld2450"
    stop.set()
    t0 = time.monotonic()
    assert list(gen) == []
    assert time.monotonic() - t0 < 0.5
