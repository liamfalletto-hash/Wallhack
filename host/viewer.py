"""Replay a recording on the floor plan in the browser: python -m host.viewer data/<session>.jsonl"""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from host import tracker
from host.geometry import frame_targets_world

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_PORT = 8000
MAX_SPEED = 64.0
FORECAST_STEPS_S = (0.5, 1.0, 1.5, 2.0)  # points on the drawn predicted path
COASTING_S = 0.3  # a track not updated for longer is drawn as predicted-only


def load_session(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not lines or lines[0].get("type") != "header":
        raise ValueError(f"{path} is not a session file: first line must be a header")
    frames = lines[1:]
    if not frames:
        raise ValueError(f"{path} has no frames")
    return lines[0], frames


def track_overlays(frames: list[dict]) -> list[list[dict]]:
    # Tracking is cheap (about 1 s per 10 minutes), so run it once up front; seeking stays consistent.
    state, out = tracker.TrackerState(), []
    for f in frames:
        state = tracker.step(state, tracker.detections_from_frame(f), f["t_dev"])
        out.append(
            [
                {
                    "id": tr.id,
                    "x": round(float(tr.pos[0]), 3),
                    "y": round(float(tr.pos[1]), 3),
                    "coasting": f["t_dev"] - tr.last_update_t > COASTING_S,
                    "path": [tracker.forecast(tr, h).round(3).tolist() for h in FORECAST_STEPS_S],
                }
                for tr in state.confirmed
            ]
        )
    return out


def with_world_targets(frame: dict, index: int, tracks: list[dict]) -> dict:
    return {**frame, "i": index, "world": frame_targets_world(frame).round(3).tolist(), "tracks": tracks}


def make_handler(
    map_json: bytes, header: dict, frames: list[dict], overlays: list[list[dict]]
) -> type[BaseHTTPRequestHandler]:
    session = json.dumps({"header": header, "n": len(frames), "t0": frames[0]["t_dev"]}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/":
                self._send(STATIC_DIR.joinpath("index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/map":
                self._send(map_json, "application/json")
            elif url.path == "/session":
                self._send(session, "application/json")
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

        def _stream(self, query: dict[str, list[str]]) -> None:
            try:
                speed = min(max(float(query.get("speed", ["1"])[0]), 0.1), MAX_SPEED)
                start = min(max(int(query.get("start", ["0"])[0]), 0), len(frames) - 1)
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "speed and start must be numbers")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
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
    handler = make_handler(map_path.read_bytes(), header, frames, track_overlays(frames))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session", type=Path)
    ap.add_argument("--map", type=Path, help="floor plan; default: the one named in the session header")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    try:
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
        server.shutdown()


if __name__ == "__main__":
    main()
