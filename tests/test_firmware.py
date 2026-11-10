"""Host-side tests for the firmware's pure C++ decoders and JSON writer (built with the system C++ compiler)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
CXX = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")

pytestmark = pytest.mark.skipif(CXX is None, reason="no C++ compiler on this machine")

# Example report from the HLK-LD2450 user manual: one target at x = -782 mm, y = 1713 mm,
# speed -16 cm/s, resolution 320 mm; the other two slots empty.
LD2450_MANUAL = "AA FF 03 00 0E 03 B1 86 10 00 40 01" + " 00" * 16 + " 55 CC"
# Two targets: x = +500 mm, y = 2000 mm, +25 cm/s; x = -1200 mm, y = 3500 mm, 0 cm/s.
LD2450_TWO = "AA FF 03 00 F4 81 D0 87 19 80 68 01 B0 04 AC 8D 00 00 68 01" + " 00" * 8 + " 55 CC"


def ld2410_report(state: int, moving_cm: int, still_cm: int) -> str:
    """Basic-mode report built from the protocol's documented layout."""
    data = [0x02, 0xAA, state, moving_cm & 0xFF, moving_cm >> 8, 50, still_cm & 0xFF, still_cm >> 8, 40, 0, 0, 0x55, 0]
    frame = [0xF4, 0xF3, 0xF2, 0xF1, len(data), 0x00, *data, 0xF8, 0xF7, 0xF6, 0xF5]
    return " ".join(f"{b:02X}" for b in frame)


@pytest.fixture(scope="module")
def driver(tmp_path_factory) -> Path:
    exe = tmp_path_factory.mktemp("fw") / "driver"
    src = ROOT / "tests" / "firmware" / "driver.cpp"
    include = str(ROOT / "firmware" / "src")
    cmd = [CXX, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", include, str(src), "-o", str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    return exe


def run(driver: Path, command: str) -> list[str]:
    out = subprocess.run([str(driver)], input=command + "\n", capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line != "end"]


def test_ld2450_manual_example(driver):
    (frame,) = run(driver, "ld2450 " + LD2450_MANUAL)
    targets = json.loads(frame)
    assert targets[0] == {"valid": True, "x": -0.782, "y": 1.713, "v": -0.16}
    assert not targets[1]["valid"] and not targets[2]["valid"]


def test_ld2450_sign_bit_both_ways(driver):
    (frame,) = run(driver, "ld2450 " + LD2450_TWO)
    a, b, c = json.loads(frame)
    assert (a["x"], a["y"], a["v"]) == (0.5, 2.0, 0.25)
    assert (b["x"], b["y"], b["v"]) == (-1.2, 3.5, 0.0)
    assert not c["valid"]


def test_ld2450_resyncs_after_garbage_and_split_frames(driver):
    stream = "AA 00 13 AA AA FF 03 " + LD2450_MANUAL + " FF " + LD2450_TWO
    frames = run(driver, "ld2450 " + stream)
    assert len(frames) == 2
    assert json.loads(frames[1])[0]["x"] == 0.5


def test_ld2450_rejects_bad_footer(driver):
    assert run(driver, "ld2450 " + LD2450_MANUAL[:-5] + "55 CD") == []


@pytest.mark.parametrize("state, moving, still", [(0, None, None), (1, 1.2, None), (2, None, 3.4), (3, 0.81, 2.5)])
def test_ld2410_report(driver, state, moving, still):
    (line,) = run(driver, "ld2410 " + ld2410_report(state, round((moving or 0) * 100), round((still or 0) * 100)))
    r = json.loads(line)
    assert r["state"] == state
    if moving:
        assert r["moving_m"] == pytest.approx(moving)
    if still:
        assert r["still_m"] == pytest.approx(still)


def test_ld2410_ignores_ack_frames_and_resyncs(driver):
    ack = "FD FC FB FA 04 00 FF 01 00 00 04 03 02 01"  # command ACK, different header
    stream = "F4 F3 " + ack + " " + ld2410_report(2, 0, 150)
    (line,) = run(driver, "ld2410 " + stream)
    assert json.loads(line)["still_m"] == pytest.approx(1.5)


def test_frame_json_matches_documented_format(driver):
    (line,) = run(driver, f"frame 0.5 0 {LD2450_TWO} | {ld2410_report(2, 0, 340)}")
    f = json.loads(line)
    assert f == {
        "v": 1,
        "src": "ld2450",
        "seq": 7,
        "t_dev": 12.5,
        "yaw": 0.5,
        "pose": None,
        "targets": [{"x": 0.5, "y": 2.0, "v": 0.25}, {"x": -1.2, "y": 3.5, "v": 0.0}],
        "presence": {"state": "still", "moving_m": None, "still_m": 3.4},
    }


def test_frame_json_without_imu_or_presence_and_flipped(driver):
    (line,) = run(driver, f"frame - 1 {LD2450_MANUAL}")
    f = json.loads(line)
    assert f["yaw"] is None and "presence" not in f
    assert f["targets"] == [{"x": 0.782, "y": 1.713, "v": -0.16}]


def test_frame_json_is_readable_by_the_host_tracker(driver):
    from host import tracker

    (line,) = run(driver, f"frame 0 0 {LD2450_TWO} | {ld2410_report(3, 120, 340)}")
    frame = {**json.loads(line), "pose": {"x": 1.0, "y": 1.0}}
    dets = tracker.detections_from_frame(frame)
    assert [d.z.round(3).tolist() for d in dets] == [[3.0, 0.5], [4.5, 2.2]]
    assert tracker.presence_from_frame(frame).still_m == pytest.approx(3.4)
