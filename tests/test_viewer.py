import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from host import viewer
from host.geometry import frame_targets_world, sensor_to_world
from sim import radar, run

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
