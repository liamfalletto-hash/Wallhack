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


W, PAD = 400, 32  # window width and side margin, px
ONYX, PANEL, HAIRLINE, DIVIDER = "#0b0b0d", "#141416", "#2b2a2d", "#232225"
IVORY, MUTED, FAINT = "#f3efe7", "#8d8981", "#6f6b64"
GOLD, GOLD_LIGHT, GOLD_DEEP, GOLD_DIM = "#d4b483", "#e6cf9f", "#a8874f", "#4a3f2c"
OXBLOOD_LIGHT, OXBLOOD, OXBLOOD_EDGE, ERROR = "#a8291f", "#560d09", "#c4473b", "#d9534f"
LABELS = ("Wi-Fi", "USB", "Demo")  # same order as SOURCES


def _spaced(text: str) -> str:
    # Tk has no letter-spacing; thin spaces give the tracked small-caps look.
    return "\u2009".join(text).replace("\u2009 \u2009", "\u2009\u2009\u2009")


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


def main() -> None:
    import tkinter as tk
    import tkinter.font as tkfont

    ctl = Controller()
    root = tk.Tk()
    root.title("Wallhack")
    root.configure(bg=ONYX)
    root.resizable(False, False)

    families = set(tkfont.families(root))
    serif = next((f for f in ("Didot", "Bodoni 72", "Playfair Display", "DejaVu Serif") if f in families), "Times")
    sans = next((f for f in ("Avenir Next", "Helvetica Neue", "Inter", "DejaVu Sans") if f in families), "Helvetica")

    cv = tk.Canvas(root, width=W, bg=ONYX, highlightthickness=0)
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

    def entry(var: tk.StringVar) -> tk.Entry:
        return tk.Entry(
            root,
            textvariable=var,
            bg=ONYX,
            fg=IVORY,
            insertbackground=GOLD,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=(sans, 15),
        )

    port_entry, pos_entry = entry(port), entry(helmet)

    def link(tag: str, action: object) -> None:
        cv.tag_bind(tag, "<Button-1>", lambda _: action())
        cv.tag_bind(tag, "<Enter>", lambda _: cv.configure(cursor="hand2"))
        cv.tag_bind(tag, "<Leave>", lambda _: cv.configure(cursor=""))

    def caption(x: float, y: float, text: str, anchor: str = "w", color: str = FAINT, tag: str = "") -> None:
        cv.create_text(x, y, text=_spaced(text.upper()), anchor=anchor, fill=color, font=(sans, 9), tags=tag)

    def field(y: float, label: str, hint: str, widget: tk.Entry) -> None:
        caption(PAD, y, label)
        cv.create_text(W - PAD, y, text=hint, anchor="e", fill=FAINT, font=(sans, 10))
        cv.create_window(PAD, y + 26, window=widget, anchor="w", width=W - 2 * PAD - 70)
        cv.create_line(PAD, y + 42, W - PAD, y + 42, fill=HAIRLINE)

    def render() -> None:
        cv.delete("all")
        running = ctl.running is not None
        # A faint warm glow behind the header, fading into onyx.
        for y in range(0, 240, 2):
            cv.create_rectangle(0, y, W, y + 2, fill=_mix("#1b1813", ONYX, y / 240), outline="")

        cv.create_text(PAD, 36, text=_spaced("WALLHACK"), anchor="w", fill=GOLD, font=(serif, 15))
        badge = "LIVE" if running else "READY"
        bw = 74
        if running:
            _round_rect(cv, W - PAD - bw, 24, W - PAD, 48, 12, fill=GOLD)
        else:
            _round_rect(cv, W - PAD - bw, 24, W - PAD, 48, 12, fill="", outline=GOLD_DEEP)
        cv.create_text(W - PAD - bw / 2, 36, text=_spaced(badge), fill=ONYX if running else GOLD, font=(sans, 9))

        caption(W / 2, 96, "Status", anchor="center")
        cv.create_text(W / 2, 134, text=ui["title"], fill=IVORY, font=(sans, 40))
        cv.create_text(W / 2, 180, text=ui["subtitle"], fill=MUTED, font=(sans, 12))

        # The button: oxblood disc in a fine gold ring; outlined in gold while running.
        cx, cy, r = W / 2, 292, 72
        cv.create_oval(cx - r - 14, cy - r - 14, cx + r + 14, cy + r + 14, outline=GOLD_DIM, width=1)
        if running:
            cv.create_oval(cx - r, cy - r, cx + r, cy + r, fill=PANEL, outline=GOLD, width=1.5, tags="btn")
            cv.create_text(cx, cy, text=_spaced("STOP"), fill=GOLD, font=(sans, 15, "bold"), tags="btn")
        else:
            _gradient_circle(cv, cx, cy, r, OXBLOOD_LIGHT, OXBLOOD, "btn")
            cv.create_oval(cx - r + 5, cy - r + 5, cx + r - 5, cy + r - 5, outline=OXBLOOD_EDGE, tags="btn")
            cv.create_text(cx, cy, text=_spaced("START"), fill=IVORY, font=(sans, 15, "bold"), tags="btn")
        link("btn", toggle)

        # Source: one slim segmented control, the selected segment in champagne.
        y = 410
        caption(PAD, y, "Source")
        top, h = y + 18, 44
        _round_rect(cv, PAD, top, W - PAD, top + h, 22, fill=PANEL, outline=HAIRLINE)
        seg = (W - 2 * PAD) / 3
        for i, (value, label) in enumerate(zip(SOURCES, LABELS, strict=True)):
            x0, tag, active = PAD + i * seg, f"seg{i}", source.get() == value
            cv.create_rectangle(x0, top, x0 + seg, top + h, fill="", outline="", tags=tag)
            if active:
                _gradient_round_rect(cv, x0 + 4, top + 4, x0 + seg - 4, top + h - 4, 18, GOLD_LIGHT, GOLD_DEEP, tag)
            cv.create_text(
                x0 + seg / 2,
                top + h / 2,
                text=label,
                fill=ONYX if active else MUTED,
                font=(sans, 12, "bold" if active else "normal"),
                tags=tag,
            )
            link(tag, lambda v=value: source.set(v))

        y = top + h + 34
        if source.get() == SOURCES[1]:
            field(y, "USB port", "", port_entry)
            caption(W - PAD, y + 26, "Detect", anchor="e", color=GOLD, tag="detect")
            link("detect", detect)
            y += 68
        field(y, "Position", "x  ·  y  ·  facing °", pos_entry)
        y += 72

        cv.create_line(PAD, y, W - PAD, y, fill=DIVIDER)
        dot = ERROR if ui["error"] else (GOLD if running else FAINT)
        cv.create_oval(PAD, y + 31, PAD + 8, y + 39, fill=dot, outline="")
        cv.create_text(
            PAD + 18,
            y + 35,
            text=ui["detail"],
            anchor="w",
            fill=ERROR if ui["error"] else IVORY,
            font=(sans, 13),
            width=W - 2 * PAD - 120,
        )
        cv.create_text(
            PAD + 18, y + 56, text=ui["note"], anchor="w", fill=MUTED, font=(sans, 11), width=W - 2 * PAD - 120
        )
        cv.create_text(
            W - PAD,
            y + 35,
            text="Open view  \u2192",
            anchor="e",
            fill=GOLD if running else FAINT,
            font=(sans, 12),
            tags="open",
        )
        link("open", lambda: ctl.running and webbrowser.open(ctl.running.url))
        cv.configure(height=y + 86)

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

    def toggle() -> None:
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
