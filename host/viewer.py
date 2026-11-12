"""Show radar data on the floor plan in the browser.

python -m host.viewer data/<session>.jsonl            # replay a recording
python -m host.viewer --live --helmet 2.0,2.2,0       # live from the helmet over Wi-Fi
python -m host.viewer --live --serial /dev/cu.usbserial-0001 --helmet 2.0,2.2,0   # over USB
python -m host.viewer --demo                          # live view of simulated people, no hardware
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TextIO
from urllib.parse import parse_qs, urlparse

from host import ingest, tracker
from host.geometry import frame_targets_world

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 8000
MAX_SPEED = 64.0
FORECAST_STEPS_S = (0.5, 1.0, 1.5, 2.0)  # points on the drawn predicted path
COASTING_S = 0.3  # a track not updated for longer is drawn as predicted-only
VIEWER_PARAMS = tracker.Params(model="imm", use_presence=True)  # best tracker in docs/benchmarks.md
DEFAULT_MAP = Path("maps/demo_flat.json")
DEFAULT_HELMET = "2.0,2.2,0"
LOG_DIR = Path("data")
LIVE_BUFFER = 50  # recent frames kept for browsers that connect or fall behind
PING_S = 1.0  # keep-alive comment on a quiet live stream, so dead connections are noticed


def load_session(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not lines or lines[0].get("type") != "header":
        raise ValueError(f"{path} is not a session file: first line must be a header")
    frames = lines[1:]
    if not frames:
        raise ValueError(f"{path} has no frames")
    return lines[0], frames


def overlay(state: tracker.TrackerState, f: dict) -> tuple[tracker.TrackerState, list[dict]]:
    if state.t is not None and f["t_dev"] < state.t:
        state = tracker.TrackerState()  # device clock went backwards: the helmet rebooted
    detections, presence = tracker.detections_from_frame(f), tracker.presence_from_frame(f)
    state = tracker.step(state, detections, f["t_dev"], VIEWER_PARAMS, presence)
    tracks = [
        {
            "id": tr.id,
            "x": round(float(tr.pos[0]), 3),
            "y": round(float(tr.pos[1]), 3),
            "coasting": f["t_dev"] - tr.last_update_t > COASTING_S,
            "mode": tr.mode,
            "path": [tracker.forecast(tr, h).round(3).tolist() for h in FORECAST_STEPS_S],
        }
        for tr in state.confirmed
    ]
    return state, tracks


def track_overlays(frames: list[dict]) -> list[list[dict]]:
    # Tracking is cheap, so replays run it once up front; seeking stays consistent.
    state, out = tracker.TrackerState(), []
    for f in frames:
        state, tracks = overlay(state, f)
        out.append(tracks)
    return out


class LiveHub:
    """Consumes a frame source on its own thread: aligns, logs, tracks, and fans out to browsers."""

    def __init__(self, frames: Iterator[dict], aligner: ingest.Aligner, log: TextIO) -> None:
        self._frames, self._aligner, self._log = frames, aligner, log
        self._recent: list[dict] = []
        self._cond = threading.Condition()
        self.received = 0

    def run(self) -> None:
        state = tracker.TrackerState()
        try:
            for raw in self._frames:
                self._aligner, f = ingest.align(self._aligner, raw, time.time())
                self._log.write(json.dumps(f) + "\n")
                self._log.flush()
                state, tracks = overlay(state, f)
                with self._cond:
                    self._recent = [*self._recent, with_world_targets(f, self.received, tracks)][-LIVE_BUFFER:]
                    self.received += 1
                    self._cond.notify_all()
        finally:
            self._log.close()  # the source stopped (Ctrl+C, STOP button) or failed

    def after(self, index: int, timeout: float) -> list[dict]:
        """Frames newer than `index`, waiting up to `timeout` for the next one."""
        with self._cond:
            self._cond.wait_for(lambda: self.received - 1 > index, timeout)
            return [m for m in self._recent if m["i"] > index]


def with_world_targets(frame: dict, index: int, tracks: list[dict]) -> dict:
    return {**frame, "i": index, "world": frame_targets_world(frame).round(3).tolist(), "tracks": tracks}


def make_handler(
    map_json: bytes,
    header: dict,
    frames: list[dict],
    overlays: list[list[dict]],
    hub: LiveHub | None = None,
    log_path: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    info = {"header": header, "live": hub is not None, "log": str(log_path) if log_path else None}
    info |= {"n": len(frames), "t0": frames[0]["t_dev"]} if frames else {"n": 0, "t0": None}
    session = json.dumps(info).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                self._send(STATIC_DIR.joinpath("index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/map":
                self._send(map_json, "application/json")
            elif url.path == "/session":
                self._send(session, "application/json")
            elif url.path == "/events" and hub is not None:
                self._stream_live(hub)
            elif url.path == "/events":
                self._stream(parse_qs(url.query))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _event_headers(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def _stream_live(self, hub: LiveHub) -> None:
            self._event_headers()
            seen = hub.received - 2  # start with the latest frame, not the whole backlog
            try:
                while True:
                    messages = hub.after(seen, PING_S)
                    for m in messages:
                        self.wfile.write(f"data: {json.dumps(m)}\n\n".encode())
                        seen = m["i"]
                    if not messages:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser closed the tab

        def _stream(self, query: dict[str, list[str]]) -> None:
            try:
                speed = min(max(float(query.get("speed", ["1"])[0]), 0.1), MAX_SPEED)
                start = min(max(int(query.get("start", ["0"])[0]), 0), len(frames) - 1)
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "speed and start must be numbers")
                return
            self._event_headers()
            wall0, t0 = time.monotonic(), frames[start]["t_dev"]
            try:
                for i in range(start, len(frames)):
                    delay = (frames[i]["t_dev"] - t0) / speed - (time.monotonic() - wall0)
                    if delay > 0:
                        time.sleep(delay)
                    self.wfile.write(f"data: {json.dumps(with_world_targets(frames[i], i, overlays[i]))}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"event: end\ndata: {}\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser paused, seeked or closed the tab

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # one line per frame request would flood the terminal

    return Handler


def serve(session_path: Path, port: int = DEFAULT_PORT, map_path: Path | None = None) -> ThreadingHTTPServer:
    header, frames = load_session(session_path)
    map_path = map_path or Path(header.get("map") or "")
    if not map_path.is_file():
        raise ValueError(f"floor plan {str(map_path)!r} not found; pass --map")
    return _server(port, make_handler(map_path.read_bytes(), header, frames, track_overlays(frames)))


def serve_live(
    source: Callable[[threading.Event], Iterator[dict]],
    src_name: str,
    helmet: ingest.HelmetPose,
    map_path: Path = DEFAULT_MAP,
    port: int = DEFAULT_PORT,
    log_dir: Path = LOG_DIR,
) -> tuple[ThreadingHTTPServer, threading.Event, Path]:
    """Start the live viewer; returns the server, a stop event for the source, and the log path."""
    if not map_path.is_file():
        raise ValueError(f"floor plan {str(map_path)!r} not found; pass --map")
    head = ingest.header(src_name, str(map_path), helmet)
    log_path, log = ingest.open_log(log_dir, head)
    stop = threading.Event()
    hub = LiveHub(source(stop), ingest.Aligner(helmet), log)
    threading.Thread(target=hub.run, daemon=True).start()
    server = _server(port, make_handler(map_path.read_bytes(), head, [], [], hub, log_path))
    return server, stop, log_path


def _server(port: int, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, nargs="?", help="recording to replay")
    ap.add_argument("--live", action="store_true", help="show live data from the helmet")
    ap.add_argument("--demo", action="store_true", help="live view of simulated people, no hardware needed")
    ap.add_argument("--serial", metavar="PORT", help="live over USB serial instead of Wi-Fi")
    ap.add_argument("--udp-port", type=int, default=ingest.UDP_PORT)
    ap.add_argument("--helmet", default=DEFAULT_HELMET, help="live: x,y,yaw_deg where you stand on the map")
    ap.add_argument("--map", type=Path, help="floor plan; default: the session's, or maps/demo_flat.json live")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    args.live = args.live or args.demo
    if args.live == (args.session is not None):
        ap.error("give either a recording to replay or --live / --demo")
    stop = None
    try:
        if args.live:
            helmet = ingest.parse_helmet(args.helmet)
            if args.demo:
                from sim.fake_helmet import realtime_frames

                source, name = (lambda ev: realtime_frames(ev, helmet=args.helmet)), "demo (simulated)"
            elif args.serial:
                source, name = (lambda ev: ingest.serial_frames(args.serial, ev)), f"serial:{args.serial}"
            else:
                source, name = (lambda ev: ingest.udp_frames(args.udp_port, ev)), f"udp:{args.udp_port}"
            server, stop, log_path = serve_live(source, name, helmet, args.map or DEFAULT_MAP, args.port)
            print(f"listening on {name}, recording to {log_path}")
        else:
            server = serve(args.session, args.port, args.map)
    except (OSError, ValueError) as e:
        ap.error(str(e))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"viewer running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if stop is not None:
            stop.set()
        server.shutdown()


if __name__ == "__main__":
    main()
