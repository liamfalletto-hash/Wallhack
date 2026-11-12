#!/bin/bash
# Double-click in Finder to start the live view from the helmet.
# Close this window (or press Ctrl+C) to stop.
cd "$(dirname "$0")" || exit 1
if [ ! -f .venv/bin/activate ]; then
  echo "No .venv found. First run once in this folder:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[dev,usb]'"
  read -r -p "Press Enter to close."
  exit 1
fi
source .venv/bin/activate
python -m host.viewer --live --port 8050 "$@"
