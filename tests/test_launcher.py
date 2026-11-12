import json
import time
import urllib.request

import pytest

from host import launcher


def wait_for_frames(url: str, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    with urllib.request.urlopen(url + "events", timeout=timeout_s) as r:
        while time.monotonic() < deadline:
            line = r.readline().decode()
            if line.startswith("data: "):
                return json.loads(line[6:])
    raise AssertionError("no frame arrived")


def test_start_demo_then_stop_and_restart(tmp_path):
    ctl = launcher.Controller(log_dir=tmp_path)
    run = ctl.start("Demo (simulated)", "2.0, 2.2, 0")
    assert ctl.start("Demo (simulated)", "2.0, 2.2, 0") is run  # second press while running is a no-op
    frame = wait_for_frames(run.url)
    assert frame["pose"] == {"x": 2.0, "y": 2.2}
    ctl.stop()
    assert ctl.running is None
    with pytest.raises(OSError):
        urllib.request.urlopen(run.url, timeout=1)
    time.sleep(0.2)
    assert run.log_path.read_text().count("\n") >= 2  # header + frames, file closed and flushed

    again = ctl.start("Demo (simulated)", "2.0, 2.2, 0")
    assert wait_for_frames(again.url)["i"] >= 0
    ctl.stop()


def test_bad_inputs_fail_with_a_clear_message(tmp_path):
    ctl = launcher.Controller(log_dir=tmp_path)
    with pytest.raises(ValueError, match="x,y,yaw_deg"):
        ctl.start("Demo (simulated)", "two, two")
    with pytest.raises(ValueError, match="USB port"):
        ctl.start("Helmet (USB)", "2, 2.2, 0", None)
    assert ctl.running is None


def test_usb_ports_returns_a_list():
    assert isinstance(launcher.usb_ports(), list)
