"""Wallhack launcher window with a big START button: python -m host.launcher (or open Wallhack.app)."""

from __future__ import annotations

import math
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path

from host import ingest, viewer

SOURCES = ("Helmet (Wi-Fi)", "Helmet (USB)", "Demo (simulated)")
SHUTDOWN_WAIT_S = 2.0


@dataclass
class Running:
    server: ThreadingHTTPServer
    stop: threading.Event
    url: str
    source_name: str
    log_path: Path


class Controller:
    """Starts and stops the live viewer; no GUI code, so it is testable headless."""

    def __init__(self, port: int = 0, log_dir: Path = viewer.LOG_DIR, map_path: Path = viewer.DEFAULT_MAP) -> None:
        self.port, self.log_dir, self.map_path = port, log_dir, map_path
        self.running: Running | None = None

    def start(self, source: str, helmet_text: str, serial_port: str | None = None) -> Running:
        if self.running:
            return self.running
        helmet = ingest.parse_helmet(helmet_text.replace(" ", ""))
        if source == SOURCES[0]:
            feed, name = (lambda ev: ingest.udp_frames(ingest.UDP_PORT, ev)), f"udp:{ingest.UDP_PORT}"
        elif source == SOURCES[1]:
            if not serial_port:
                raise ValueError("choose the helmet's USB port first")
            feed, name = (lambda ev: ingest.serial_frames(serial_port, ev)), f"serial:{serial_port}"
        elif source == SOURCES[2]:
            from sim.fake_helmet import realtime_frames

            feed, name = (lambda ev: realtime_frames(ev, helmet=helmet_text.replace(" ", ""))), "demo (simulated)"
        else:
            raise ValueError(f"unknown source {source!r}")
        server, stop, log_path = viewer.serve_live(feed, name, helmet, self.map_path, self.port, self.log_dir)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        self.running = Running(server, stop, url, name, log_path)
        return self.running

    def stop(self) -> None:
        if not self.running:
            return
        self.running.stop.set()
        self.running.server.shutdown()
        self.running.server.server_close()
        self.running = None


def usb_ports() -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    # macOS also lists built-in Bluetooth and debug-console ports, which are never the helmet.
    skip = ("Bluetooth", "debug-console")
    return sorted(p.device for p in list_ports.comports() if not any(k in p.device for k in skip))


W = 400  # window width, px
DARK, PAGE, CARD, TEXT, TEXT2 = "#1e2023", "#f3f5f4", "#ffffff", "#1e2023", "#7c838b"
MINT, TEAL, MINT_SOFT, TEAL_TEXT = "#62f0b4", "#1db9a7", "#e2f7f0", "#139e8f"
RED_LIGHT, RED, ERROR, SHADOW = "#ff6b6b", "#d63a3a", "#d9534f", "#e1e5e3"
FONT = "Helvetica"
TILES = (("Wi-Fi", "wifi"), ("USB", "usb"), ("Demo", "demo"))


def _mix(c1: str, c2: str, t: float) -> str:
    a, b = (tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in (c1, c2))
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b, strict=True))


def _round_rect(cv: object, x0: float, y0: float, x1: float, y1: float, r: float, **kw: object) -> None:
    pts = [
        x0 + r,
        y0,
        x1 - r,
        y0,
        x1,
        y0,
        x1,
        y0 + r,
        x1,
        y1 - r,
        x1,
        y1,
        x1 - r,
        y1,
        x0 + r,
        y1,
        x0,
        y1,
        x0,
        y1 - r,
        x0,
        y0 + r,
        x0,
        y0,
    ]
    cv.create_polygon(pts, smooth=True, **kw)


def _gradient_round_rect(
    cv: object, x0: float, y0: float, x1: float, y1: float, r: float, c1: str, c2: str, tags: str
) -> None:
    # Tk has no gradients: draw one horizontal line per pixel row, inset at the rounded corners.
    for y in range(int(y0), int(y1)):
        d = min(y - y0, y1 - 1 - y)
        inset = r - math.sqrt(max(r * r - (r - d) ** 2, 0)) if d < r else 0
        cv.create_line(x0 + inset, y, x1 - inset, y, fill=_mix(c1, c2, (y - y0) / (y1 - y0)), tags=tags)


def _gradient_circle(cv: object, cx: float, cy: float, r: float, c1: str, c2: str, tags: str) -> None:
    for dy in range(-int(r), int(r) + 1):
        half = math.sqrt(max(r * r - dy * dy, 0))
        cv.create_line(cx - half, cy + dy, cx + half, cy + dy, fill=_mix(c1, c2, (dy + r) / (2 * r)), tags=tags)
    cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline=_mix(c1, c2, 0.5), tags=tags)


def _glyph(cv: object, kind: str, cx: float, cy: float, color: str, tags: str) -> None:
    if kind == "wifi":
        for r in (6, 12, 18):
            cv.create_arc(
                cx - r,
                cy - r + 8,
                cx + r,
                cy + r + 8,
                start=45,
                extent=90,
                style="arc",
                outline=color,
                width=2.5,
                tags=tags,
            )
        cv.create_oval(cx - 2.5, cy + 5.5, cx + 2.5, cy + 10.5, fill=color, outline="", tags=tags)
    elif kind == "usb":
        cv.create_rectangle(cx - 7, cy - 4, cx + 7, cy + 10, outline=color, width=2.5, tags=tags)
        for dx in (-3, 3):
            cv.create_line(cx + dx, cy - 11, cx + dx, cy - 4, fill=color, width=2.5, tags=tags)
    else:
        cv.create_polygon(cx - 6, cy - 9, cx - 6, cy + 9, cx + 9, cy, fill=color, outline=color, tags=tags)


def main() -> None:
    import tkinter as tk
    import tkinter.font as tkfont

    ctl = Controller()
    root = tk.Tk()
    root.title("Wallhack")
    root.configure(bg=PAGE)
    root.resizable(False, False)
    cv = tk.Canvas(root, width=W, bg=PAGE, highlightthickness=0)
    cv.pack()

    source = tk.StringVar(value=SOURCES[0])
    port = tk.StringVar()
    helmet = tk.StringVar(value=viewer.DEFAULT_HELMET.replace(",", ", "))
    ui = {
        "title": "Ready",
        "subtitle": "Choose a source and press start",
        "detail": "Not running",
        "note": "",
        "error": False,
    }
    pill_font = tkfont.Font(family=FONT, size=12, weight="bold")

    def entry(var: tk.StringVar) -> tk.Entry:
        return tk.Entry(
            root,
            textvariable=var,
            bg=CARD,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=(FONT, 15),
        )

    port_entry, pos_entry = entry(port), entry(helmet)

    def pill(x1: float, cy: float, label: str, tag: str, dark: bool = True) -> None:
        w = pill_font.measure(label) + 28
        _round_rect(cv, x1 - w, cy - 15, x1, cy + 15, 15, fill=DARK if dark else MINT_SOFT, tags=tag)
        cv.create_text(x1 - w / 2, cy, text=label, fill=CARD if dark else TEAL_TEXT, font=pill_font, tags=tag)
        cv.tag_bind(tag, "<Enter>", lambda _: cv.configure(cursor="hand2"))
        cv.tag_bind(tag, "<Leave>", lambda _: cv.configure(cursor=""))

    def card(y: float, h: float, label: str) -> None:
        _round_rect(cv, 24, y + 4, W - 24, y + h + 4, 18, fill=SHADOW)
        _round_rect(cv, 24, y, W - 24, y + h, 18, fill=CARD)
        cv.create_text(42, y + 18, text=label, anchor="w", fill=TEXT2, font=(FONT, 11))

    def render() -> None:
        cv.delete("all")
        running = ctl.running is not None
        # Dark header with the big status word, like the balance panels in the reference.
        _round_rect(cv, 0, -60, W, 236, 34, fill=DARK)
        cv.create_oval(26, 22, 42, 38, fill=MINT, outline="")
        cv.create_text(50, 30, text="Wallhack", anchor="w", fill=CARD, font=(FONT, 14, "bold"))
        badge, badge_bg, badge_fg = ("LIVE", MINT, DARK) if running else ("READY", "#3a3d42", "#c8ccd0")
        _round_rect(cv, W - 92, 17, W - 24, 43, 13, fill=badge_bg)
        cv.create_text(W - 58, 30, text=badge, fill=badge_fg, font=(FONT, 11, "bold"))
        cv.create_text(W / 2, 80, text="Status", fill="#8f959c", font=(FONT, 12))
        cv.create_text(W / 2, 112, text=ui["title"], fill=CARD, font=(FONT, 32, "bold"))
        cv.create_text(W / 2, 148, text=ui["subtitle"], fill="#8f959c", font=(FONT, 12))

        # Big round button straddling the header edge.
        cx, cy, r = W / 2, 236, 60
        cv.create_oval(cx - r - 3, cy - r + 3, cx + r + 3, cy + r + 9, fill=_mix(PAGE, DARK, 0.10), outline="")
        if running:
            cv.create_oval(cx - r, cy - r, cx + r, cy + r, fill=CARD, outline=RED, width=4, tags="btn")
            cv.create_text(cx, cy, text="STOP", fill=RED, font=(FONT, 22, "bold"), tags="btn")
        else:
            _gradient_circle(cv, cx, cy, r, RED_LIGHT, RED, "btn")
            cv.create_text(cx, cy, text="START", fill=CARD, font=(FONT, 22, "bold"), tags="btn")
        cv.tag_bind("btn", "<Button-1>", toggle)
        cv.tag_bind("btn", "<Enter>", lambda _: cv.configure(cursor="hand2"))
        cv.tag_bind("btn", "<Leave>", lambda _: cv.configure(cursor=""))

        # Source tiles, like the reference's action tiles.
        y = 322
        cv.create_text(26, y, text="Source", anchor="w", fill=TEXT, font=(FONT, 14, "bold"))
        tile_w, gap, top = 104, 18, y + 18
        for i, ((label, kind), value) in enumerate(zip(TILES, SOURCES, strict=True)):
            x0 = 24 + i * (tile_w + gap)
            tag, active = f"tile{i}", source.get() == value
            _round_rect(cv, x0, top + 4, x0 + tile_w, top + 92, 20, fill=SHADOW, tags=tag)
            if active:
                _gradient_round_rect(cv, x0, top, x0 + tile_w, top + 88, 20, MINT, TEAL, tag)
            else:
                _round_rect(cv, x0, top, x0 + tile_w, top + 88, 20, fill=CARD, tags=tag)
                _round_rect(cv, x0 + 32, top + 14, x0 + 72, top + 54, 12, fill=MINT_SOFT, tags=tag)
            _glyph(cv, kind, x0 + 52, top + 34, CARD if active else TEAL_TEXT, tag)
            cv.create_text(
                x0 + 52, top + 70, text=label, fill=CARD if active else TEXT, font=(FONT, 12, "bold"), tags=tag
            )
            cv.tag_bind(tag, "<Button-1>", lambda _, v=value: source.set(v))
            cv.tag_bind(tag, "<Enter>", lambda _: cv.configure(cursor="hand2"))
            cv.tag_bind(tag, "<Leave>", lambda _: cv.configure(cursor=""))

        y = top + 112
        if source.get() == SOURCES[1]:
            card(y, 72, "USB port")
            cv.create_window(42, y + 48, window=port_entry, anchor="w", width=220)
            pill(W - 40, y + 44, "Detect", "detect", dark=False)
            cv.tag_bind("detect", "<Button-1>", lambda _: detect())
            y += 86
        card(y, 72, "Your position on the map  ·  x m, y m, facing °")
        cv.create_window(42, y + 48, window=pos_entry, anchor="w", width=W - 100)
        y += 86

        card(y, 84, "Session")
        dot = ERROR if ui["error"] else (MINT if running else "#c4c9ce")
        cv.create_oval(42, y + 41, 52, y + 51, fill=dot, outline="")
        cv.create_text(
            62,
            y + 46,
            text=ui["detail"],
            anchor="w",
            width=W - 210,
            fill=ERROR if ui["error"] else TEXT,
            font=(FONT, 13, "bold"),
        )
        cv.create_text(62, y + 66, text=ui["note"], anchor="w", width=W - 210, fill=TEXT2, font=(FONT, 11))
        pill(W - 40, y + 50, "Open view", "open", dark=running)
        cv.tag_bind("open", "<Button-1>", lambda _: ctl.running and webbrowser.open(ctl.running.url))
        cv.configure(height=y + 108)

    def set_ui(title: str, subtitle: str, detail: str, note: str = "", error: bool = False) -> None:
        ui.update(title=title, subtitle=subtitle, detail=detail, note=note, error=error)
        render()

    def detect() -> None:
        ports = usb_ports()
        if ports and port.get() not in ports:
            port.set(ports[0])
        if not ports:
            set_ui(
                "Ready",
                "Choose a source and press start",
                "No USB port found",
                "Plug in the helmet, then Detect",
                error=True,
            )

    def on_source(*_: object) -> None:
        if source.get() == SOURCES[1]:
            detect()
        render()

    def toggle(_: object = None) -> None:
        if ctl.running:
            ctl.stop()
            set_ui("Stopped", "Press start to run again", "Not running")
            return
        try:
            r = ctl.start(source.get(), helmet.get(), port.get() or None)
        except (OSError, ValueError) as e:
            set_ui("Error", "Could not start", "Not running", str(e), error=True)
            return
        set_ui("Live", r.source_name, "Recording", f"Since {time.strftime('%H:%M:%S')}")
        webbrowser.open(r.url)

    def on_close() -> None:
        ctl.stop()
        root.destroy()

    source.trace_add("write", on_source)
    root.protocol("WM_DELETE_WINDOW", on_close)
    render()
    root.lift()
    root.attributes("-topmost", True)  # appear in front when started from Finder
    root.after(500, lambda: root.attributes("-topmost", False))
    root.mainloop()


if __name__ == "__main__":
    main()
