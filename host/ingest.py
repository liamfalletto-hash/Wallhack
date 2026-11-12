"""Receive helmet frames over Wi-Fi (UDP broadcast) or USB serial, align them to the map, log them.

The firmware sends raw sensor values: no helmet position, and a heading relative to where
the IMU was switched on. `Aligner` adds the position from --helmet and rotates the heading
so the first reading equals the --helmet yaw (start by facing that direction).
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import numpy as np

UDP_PORT = 5005  # matches firmware/src/main.cpp
SERIAL_BAUD = 115200
RECV_TIMEOUT_S = 0.5  # lets sources notice a stop request
MAX_DATAGRAM = 2048
FORMAT_VERSION = 1


@dataclass(frozen=True)
class HelmetPose:
    x: float
    y: float
    yaw: float  # radians, CCW from map +x


@dataclass(frozen=True)
class Aligner:
    helmet: HelmetPose
    imu_yaw0: float | None = None  # first IMU heading seen


def parse_helmet(text: str) -> HelmetPose:
    try:
        x, y, yaw_deg = (float(v) for v in text.split(","))
    except ValueError as e:
        raise ValueError(f"--helmet must be 'x,y,yaw_deg', got {text!r}") from e
    return HelmetPose(x, y, float(np.radians(yaw_deg)))


def parse_line(line: bytes | str) -> dict | None:
    """A frame from one USB line or UDP datagram; None for status lines ('#') and garbage."""
    text = line.decode("utf-8", "replace") if isinstance(line, bytes) else line
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        frame = json.loads(text)
    except json.JSONDecodeError:
        return None
    return frame if isinstance(frame, dict) and "targets" in frame else None


def align(state: Aligner, frame: dict, host_t: float) -> tuple[Aligner, dict]:
    out = {**frame, "t": round(host_t, 3)}
    if not frame.get("pose"):
        out["pose"] = {"x": state.helmet.x, "y": state.helmet.y}
    yaw = frame.get("yaw")
    if yaw is not None:
        if state.imu_yaw0 is None:
            state = replace(state, imu_yaw0=float(yaw))
        out["yaw"] = round(_wrap(state.helmet.yaw + float(yaw) - state.imu_yaw0), 4)
    return state, out


def header(src: str, map_path: str, helmet: HelmetPose) -> dict:
    return {
        "type": "header",
        "v": FORMAT_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "src": src,
        "map": map_path,
        "notes": f"live, helmet {helmet.x:g},{helmet.y:g},{np.degrees(helmet.yaw):g}",
    }


def open_log(directory: Path, head: dict) -> tuple[Path, TextIO]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    f = path.open("w")
    f.write(json.dumps(head) + "\n")
    f.flush()
    return path, f


def udp_frames(port: int, stop: threading.Event) -> Iterator[dict]:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))  # all interfaces, so subnet broadcasts arrive too
        sock.settimeout(RECV_TIMEOUT_S)
        while not stop.is_set():
            try:
                data, _ = sock.recvfrom(MAX_DATAGRAM)
            except TimeoutError:
                continue
            frame = parse_line(data)
            if frame is not None:
                yield frame


def serial_frames(url: str, stop: threading.Event, baud: int = SERIAL_BAUD) -> Iterator[dict]:
    import serial  # optional dependency, only needed for USB

    with serial.serial_for_url(url, baudrate=baud, timeout=RECV_TIMEOUT_S) as port:
        while not stop.is_set():
            frame = parse_line(port.readline())
            if frame is not None:
                yield frame


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)
