"""Pretend to be the helmet: send simulated frames over UDP in real time, like the firmware.

python -m sim.fake_helmet                     # to this computer (pair with: python -m host.viewer --live)
python -m sim.fake_helmet --host 192.168.1.255  # broadcast on the network, like the real ESP32
"""

from __future__ import annotations

import argparse
import json
import socket
import time

from sim import floorplan as fpl
from sim import run

DEFAULT_HOST = "127.0.0.1"
UDP_PORT = 5005  # matches firmware/src/main.cpp and host/ingest.py


def as_firmware(frame: dict) -> dict:
    # The ESP32 knows neither where the helmet stands nor the truth; heading is relative to power-on.
    return {k: v for k, v in frame.items() if k not in ("truth", "t")} | {"src": "ld2450", "pose": None}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--people", type=int, default=2)
    ap.add_argument("--duration", type=float, default=600.0, help="seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--helmet", default=run.DEFAULT_HELMET, help="x,y,yaw_deg; use the same value for the viewer")
    args = ap.parse_args(argv)

    fp = fpl.load(run.DEFAULT_MAP)
    helmet = run.parse_helmet(args.helmet)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        print(f"sending simulated frames to {args.host}:{args.port} (Ctrl+C to stop)")
        start = time.monotonic()
        try:
            for frame in run.frames(fp, helmet, args.people, args.duration, args.seed):
                delay = frame["t_dev"] - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)
                sock.sendto(json.dumps(as_firmware(frame)).encode(), (args.host, args.port))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
