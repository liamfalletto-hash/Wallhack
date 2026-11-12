import json
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from host import ingest
from host.ingest import Aligner, HelmetPose

HELMET = HelmetPose(2.0, 2.2, np.radians(90))
FRAME = {"v": 1, "src": "ld2450", "seq": 1, "t_dev": 1.0, "yaw": 0.3, "pose": None, "targets": []}


@pytest.mark.parametrize(
    "line, ok",
    [
        (json.dumps(FRAME).encode() + b"\r\n", True),
        (json.dumps(FRAME), True),
        (b"# imu found\n", False),
        (b'{"v":1,"targ', False),  # truncated USB line
        (b'{"hello": 1}', False),  # JSON but not a frame
        (b"\xff\xfe garbage", False),
    ],
)
def test_parse_line(line, ok):
    assert (ingest.parse_line(line) is not None) is ok


def test_align_fills_pose_and_rotates_heading_onto_the_map():
    state, first = ingest.align(Aligner(HELMET), FRAME, host_t=100.0)
    assert first["pose"] == {"x": 2.0, "y": 2.2} and first["t"] == 100.0
    assert first["yaw"] == pytest.approx(np.radians(90), abs=1e-4)  # first reading = --helmet yaw
    _, later = ingest.align(state, {**FRAME, "yaw": 0.3 + 0.5}, host_t=100.1)
    assert later["yaw"] == pytest.approx(np.radians(90) + 0.5, abs=1e-4)
    _, wrapped = ingest.align(state, {**FRAME, "yaw": 0.3 + 3.0}, host_t=100.2)
    assert -np.pi <= wrapped["yaw"] < np.pi


def test_align_keeps_missing_imu_and_known_pose():
    state, out = ingest.align(Aligner(HELMET), {**FRAME, "yaw": None, "pose": {"x": 5, "y": 1}}, 0.0)
    assert out["yaw"] is None and out["pose"] == {"x": 5, "y": 1}
    assert state.imu_yaw0 is None


def test_parse_helmet():
    assert ingest.parse_helmet("1,2,180").yaw == pytest.approx(np.pi)
    with pytest.raises(ValueError, match="x,y,yaw_deg"):
        ingest.parse_helmet("1,2")


def collect(gen, n, stop):
    out = []
    for frame in gen:
        out.append(frame)
        if len(out) == n:
            break
    stop.set()
    return out


def free_port(kind=socket.SOCK_DGRAM) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_udp_frames_receive_datagrams_and_skip_status():
    port, stop, got = free_port(), threading.Event(), []
    t = threading.Thread(target=lambda: got.extend(collect(ingest.udp_frames(port, stop), 2, stop)))
    t.start()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for _ in range(20):  # resend until the receiver has bound
            for payload in (b"# status", json.dumps(FRAME).encode(), json.dumps({**FRAME, "seq": 2}).encode()):
                s.sendto(payload, ("127.0.0.1", port))
            t.join(0.1)
            if not t.is_alive():
                break
    stop.set()
    t.join(2)
    assert [f["seq"] for f in got][:2] in ([1, 2], [2, 1], [1, 1], [2, 2])


def test_serial_frames_read_lines(tmp_path):
    # pyserial's socket:// URL stands in for a USB port: same readline() path, no hardware.
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def feed():
        conn, _ = server.accept()
        lines = [b"# boot", json.dumps(FRAME).encode(), json.dumps({**FRAME, "seq": 2}).encode()]
        conn.sendall(b"\n".join(lines) + b"\n")
        conn.close()

    threading.Thread(target=feed, daemon=True).start()
    stop = threading.Event()
    got = collect(ingest.serial_frames(f"socket://127.0.0.1:{server.getsockname()[1]}", stop), 2, stop)
    server.close()
    assert [f["seq"] for f in got] == [1, 2]


def test_open_log_writes_header(tmp_path: Path):
    path, f = ingest.open_log(tmp_path / "data", ingest.header("udp:5005", "maps/demo_flat.json", HELMET))
    f.write(json.dumps(FRAME) + "\n")
    f.close()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["type"] == "header" and lines[0]["src"] == "udp:5005"
    assert path.name.startswith("live_") and len(lines) == 2
