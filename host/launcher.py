"""Wallhack launcher window with a big START button: python -m host.launcher (or open Wallhack.app)."""

from __future__ import annotations

import threading
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


def main() -> None:
    import tkinter as tk

    # Same palette and type as host/static/index.html, so the launcher and the console match.
    bg, surface, surface2, line, line_strong = "#0f1114", "#15181c", "#1b1f24", "#262b32", "#353b44"
    text, text2, text3 = "#d9dde3", "#8b939e", "#5d6570"
    red, red_hover, ok, bad = "#d63a3a", "#e04848", "#4caf7a", "#d9534f"
    sans, mono = "Helvetica", "Menlo"
    labels = {SOURCES[0]: "Helmet Wi-Fi", SOURCES[1]: "Helmet USB", SOURCES[2]: "Demo"}

    ctl = Controller()
    root = tk.Tk()
    root.title("Wallhack")
    root.configure(bg=bg)
    root.resizable(False, False)

    def hline(parent: tk.Misc) -> None:
        tk.Frame(parent, bg=line, height=1).pack(fill="x")

    def clickable(widget: tk.Widget, action: object) -> None:
        widget.bind("<Button-1>", lambda _: action())
        widget.configure(cursor="hand2")

    header = tk.Frame(root, bg=surface, padx=16, pady=10)
    header.pack(fill="x")
    tk.Label(header, text="Wallhack", bg=surface, fg=text, font=(sans, 14, "bold")).pack(side="left")
    badge = tk.Label(
        header,
        text="READY",
        font=(mono, 10, "bold"),
        padx=6,
        bg=surface2,
        fg=text2,
        highlightthickness=1,
        highlightbackground=line_strong,
    )
    badge.pack(side="left", padx=10)
    tk.Label(header, text="Launcher", bg=surface, fg=text3, font=(sans, 12)).pack(side="right")
    hline(root)

    body = tk.Frame(root, bg=bg, padx=24, pady=20)
    body.pack(fill="x")
    size = 180
    button = tk.Canvas(body, width=size, height=size, bg=bg, highlightthickness=0, cursor="hand2")
    button.pack(pady=(4, 20))

    def field(label: str) -> tk.Frame:
        row = tk.Frame(body, bg=bg)
        row.pack(fill="x", pady=5)
        tk.Label(row, text=label, bg=bg, fg=text2, font=(sans, 12), width=10, anchor="w").pack(side="left")
        return row

    def entry(parent: tk.Misc, var: tk.StringVar, width: int) -> tk.Entry:
        e = tk.Entry(
            parent,
            textvariable=var,
            width=width,
            bg=surface2,
            fg=text,
            insertbackground=text,
            relief="flat",
            font=(mono, 12),
            highlightthickness=1,
            highlightbackground=line_strong,
            highlightcolor=text3,
        )
        e.pack(side="left", ipady=4)
        return e

    source = tk.StringVar(value=SOURCES[0])
    seg = tk.Frame(field("Source"), bg=line_strong, padx=1, pady=1)
    seg.pack(side="left")
    seg_buttons: dict[str, tk.Label] = {}
    for value in SOURCES:
        b = tk.Label(seg, text=labels[value], font=(sans, 12), padx=10, pady=4)
        b.pack(side="left", padx=(0 if value == SOURCES[0] else 1, 0))
        clickable(b, lambda v=value: source.set(v))
        seg_buttons[value] = b

    port = tk.StringVar()
    usb_row = field("USB port")
    entry(usb_row, port, 22)
    refresh = tk.Label(usb_row, text="Detect", bg=bg, fg=text2, font=(sans, 12), padx=8)
    refresh.pack(side="left")

    helmet = tk.StringVar(value=viewer.DEFAULT_HELMET.replace(",", ", "))
    pos_row = field("Position")
    entry(pos_row, helmet, 14)
    tk.Label(pos_row, text="x m, y m, facing °", bg=bg, fg=text3, font=(sans, 11), padx=8).pack(side="left")

    hline(root)
    footer = tk.Frame(root, bg=surface, padx=16, pady=10)
    footer.pack(fill="x")
    dot = tk.Canvas(footer, width=10, height=10, bg=surface, highlightthickness=0)
    dot.pack(side="left", anchor="n", pady=4)
    status = tk.Label(
        footer, text="Ready", bg=surface, fg=text2, font=(mono, 11), justify="left", anchor="w", wraplength=250
    )
    status.pack(side="left", padx=8, fill="x", expand=True)
    open_btn = tk.Label(
        footer,
        text="Open view",
        font=(sans, 12),
        padx=10,
        pady=3,
        bg=surface2,
        fg=text3,
        highlightthickness=1,
        highlightbackground=line_strong,
    )
    open_btn.pack(side="right", anchor="n")

    def set_status(msg: str, color: str) -> None:
        status.configure(text=msg, fg=text if color == ok else (bad if color == bad else text2))
        dot.delete("all")
        dot.create_oval(1, 1, 9, 9, fill=color, outline="")

    def draw(hover: bool = False) -> None:
        button.delete("all")
        if ctl.running:
            button.create_oval(8, 8, size - 8, size - 8, fill=surface, outline=red, width=3)
            button.create_text(size / 2, size / 2, text="STOP", fill=red, font=(sans, 28, "bold"))
        else:
            button.create_oval(8, 8, size - 8, size - 8, fill=red_hover if hover else red, outline="")
            button.create_text(size / 2, size / 2, text="START", fill="white", font=(sans, 28, "bold"))

    def on_source(*_: object) -> None:
        for value, b in seg_buttons.items():
            active = value == source.get()
            b.configure(bg=surface2 if active else bg, fg=text if active else text2)
        if source.get() == SOURCES[1]:
            usb_row.pack(fill="x", pady=5, before=pos_row)
            detect()
        else:
            usb_row.pack_forget()

    def detect() -> None:
        ports = usb_ports()
        if ports and port.get() not in ports:
            port.set(ports[0])
        if not ports:
            set_status("No USB serial port found. Plug in the helmet and press Detect.", bad)

    def open_view() -> None:
        if ctl.running:
            webbrowser.open(ctl.running.url)

    def toggle(_: object = None) -> None:
        if ctl.running:
            ctl.stop()
            set_status("Stopped", text3)
            badge.configure(text="READY", bg=surface2, fg=text2)
            open_btn.configure(fg=text3)
        else:
            try:
                r = ctl.start(source.get(), helmet.get(), port.get() or None)
            except (OSError, ValueError) as e:
                set_status(f"Could not start: {e}", bad)
                draw()
                return
            set_status(f"{r.source_name}\nRecording {r.log_path}", ok)
            badge.configure(text="LIVE", bg=ok, fg=bg)
            open_btn.configure(fg=text)
            webbrowser.open(r.url)
        draw()

    def on_close() -> None:
        ctl.stop()
        root.destroy()

    clickable(refresh, detect)
    clickable(open_btn, open_view)
    source.trace_add("write", on_source)
    button.bind("<Button-1>", toggle)
    button.bind("<Enter>", lambda _: draw(hover=True))
    button.bind("<Leave>", lambda _: draw())
    root.protocol("WM_DELETE_WINDOW", on_close)
    on_source()
    set_status("Ready", text3)
    draw()
    root.lift()
    root.attributes("-topmost", True)  # appear in front when started from Finder
    root.after(500, lambda: root.attributes("-topmost", False))
    root.mainloop()


if __name__ == "__main__":
    main()
