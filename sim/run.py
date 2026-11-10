"""Generate a simulated recording: python -m sim.run --people 2 --duration 300 --seed 1"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sim import floorplan as fpl
from sim import people, radar
from sim.floorplan import FloorPlan
from sim.radar import Pose

FORMAT_VERSION = 1
DEFAULT_MAP = "maps/demo_flat.json"
DEFAULT_HELMET = "2.0,2.2,0"  # x m, y m, yaw deg: bedroom, facing the living room wall


def frames(fp: FloorPlan, helmet: Pose, n_people: int, duration_s: float, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng([seed, 1])  # separate stream so radar noise never changes people's paths
    every = round(radar.FRAME_PERIOD_S / people.SIM_DT_S)
    state = radar.initial_state()
    for k, (t, ps) in enumerate(people.simulate(fp, n_people, duration_s, seed)):
        if k % every:
            continue
        state, targets, yaw = radar.sense(state, fp, helmet, ps, rng)
        yield {
            "v": FORMAT_VERSION,
            "src": "sim",
            "seq": state.seq,
            "t_dev": round(t, 3),
            "t": round(t, 3),
            "yaw": round(yaw, 4),
            "pose": {"x": helmet.x, "y": helmet.y},
            "targets": [{"x": tg.x, "y": tg.y, "v": tg.v} for tg in targets],
            "truth": {
                "pose": {"x": helmet.x, "y": helmet.y, "yaw": helmet.yaw},
                "people": [_truth(p) for p in ps],
            },
        }


def parse_helmet(text: str) -> Pose:
    try:
        x, y, yaw_deg = (float(v) for v in text.split(","))
    except ValueError as e:
        raise ValueError(f"--helmet must be 'x,y,yaw_deg', got {text!r}") from e
    return Pose(x, y, float(np.radians(yaw_deg)))


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--people", type=int, default=2)
    ap.add_argument("--duration", type=float, default=300.0, help="seconds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--helmet", default=DEFAULT_HELMET, help="x,y,yaw_deg in the floor plan frame")
    ap.add_argument("--out", type=Path, help="default: data/sim_<map>_s<seed>.jsonl")
    args = ap.parse_args(argv)

    fp = fpl.load(args.map)
    helmet = parse_helmet(args.helmet)
    if not fpl.is_free_point(fp, helmet.xy):
        ap.error(f"helmet position {helmet.x},{helmet.y} is inside a wall or furniture")
    out = args.out or Path("data") / f"sim_{fp.name}_s{args.seed}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "type": "header",
        "v": FORMAT_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "src": "sim",
        "map": args.map,
        "notes": f"seed {args.seed}, {args.people} people, {args.duration:g} s, helmet {args.helmet}",
    }
    with out.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for frame in frames(fp, helmet, args.people, args.duration, args.seed):
            f.write(json.dumps(frame) + "\n")
    print(f"wrote {out}")
    return out


def _truth(p: people.Person) -> dict:
    x, y, vx, vy = (round(float(v), 3) + 0.0 for v in (*p.pos, *p.velocity))  # + 0.0 drops "-0.0"
    return {"id": p.id, "x": x, "y": y, "vx": vx, "vy": vy, "mode": p.mode}


if __name__ == "__main__":
    main()
