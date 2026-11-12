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
    return sorted(p.device for p in list_ports.comports() if "Bluetooth" not in p.device)


def main() -> None:
    import tkinter as tk
    from tkinter import ttk

    bg, surface, text, muted = "#0f1114", "#15181c", "#d9dde3", "#8b939e"
    red, red_hover, red_dark, ok, bad = "#d63a3a", "#e85050", "#7a1f1f", "#4caf7a", "#d9534f"
    font = "Helvetica"

    ctl = Controller()
    root = tk.Tk()
    root.title("Wallhack")
    root.configure(bg=bg, padx=28, pady=24)
    root.resizable(False, False)

    tk.Label(root, text="Wallhack", bg=bg, fg=text, font=(font, 22, "bold")).pack()
    tk.Label(root, text="Through-wall radar console", bg=bg, fg=muted, font=(font, 12)).pack(pady=(0, 16))

    size = 200
    button = tk.Canvas(root, width=size, height=size, bg=bg, highlightthickness=0, cursor="hand2")
    button.pack(pady=8)

    form = tk.Frame(root, bg=bg)
    form.pack(fill="x", pady=(14, 0))
    source = tk.StringVar(value=SOURCES[0])
    port = tk.StringVar()
    helmet = tk.StringVar(value=viewer.DEFAULT_HELMET.replace(",", ", "))

    def row(label: str) -> tk.Frame:
        f = tk.Frame(form, bg=bg)
        f.pack(fill="x", pady=3)
        tk.Label(f, text=label, bg=bg, fg=muted, font=(font, 12), width=14, anchor="w").pack(side="left")
        return f

    ttk.Combobox(row("Source"), textvariable=source, values=SOURCES, state="readonly", width=22).pack(side="left")
    usb_row = row("USB port")
    usb_box = ttk.Combobox(usb_row, textvariable=port, width=22)
    usb_box.pack(side="left")
    tk.Entry(
        row("Position x, y, °"),
        textvariable=helmet,
        width=24,
        bg=surface,
        fg=text,
        insertbackground=text,
        relief="flat",
        highlightthickness=1,
        highlightbackground="#353b44",
    ).pack(side="left")

    status = tk.Label(root, text="Ready", bg=bg, fg=muted, font=(font, 12), wraplength=340, justify="center")
    status.pack(pady=(16, 6))
    open_btn = tk.Button(
        root,
        text="Open view in browser",
        state="disabled",
        font=(font, 12),
        command=lambda: ctl.running and webbrowser.open(ctl.running.url),
    )
    open_btn.pack()

    def draw(hover: bool = False) -> None:
        button.delete("all")
        running = ctl.running is not None
        fill = surface if running else (red_hover if hover else red)
        button.create_oval(6, 6, size - 6, size - 6, fill=red_dark if not running else bg, outline="")
        button.create_oval(16, 16, size - 16, size - 16, fill=fill, outline=red if running else "", width=4)
        button.create_text(
            size / 2,
            size / 2,
            text="STOP" if running else "START",
            fill=red if running else "white",
            font=(font, 30, "bold"),
        )

    def on_source(*_: object) -> None:
        usb = source.get() == SOURCES[1]
        if usb:
            usb_box.configure(values=usb_ports())
            if not port.get() and usb_box["values"]:
                port.set(usb_box["values"][0])
        for child in usb_row.winfo_children():
            child.configure(state="normal" if usb else "disabled")

    def toggle(_: object = None) -> None:
        if ctl.running:
            ctl.stop()
            status.configure(text="Stopped", fg=muted)
            open_btn.configure(state="disabled")
        else:
            try:
                r = ctl.start(source.get(), helmet.get(), port.get() or None)
            except (OSError, ValueError) as e:
                status.configure(text=f"Could not start: {e}", fg=bad)
                draw()
                return
            status.configure(text=f"Running · {r.source_name}\nRecording {r.log_path}", fg=ok)
            open_btn.configure(state="normal")
            webbrowser.open(r.url)
        draw()

    def on_close() -> None:
        ctl.stop()
        root.destroy()

    source.trace_add("write", on_source)
    button.bind("<Button-1>", toggle)
    button.bind("<Enter>", lambda _: draw(hover=True))
    button.bind("<Leave>", lambda _: draw())
    root.protocol("WM_DELETE_WINDOW", on_close)
    on_source()
    draw()
    root.lift()
    root.attributes("-topmost", True)  # appear in front when started from Finder
    root.after(500, lambda: root.attributes("-topmost", False))
    root.mainloop()


if __name__ == "__main__":
    main()
