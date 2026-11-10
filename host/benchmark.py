"""1-second forecast-error benchmark on simulated recordings.

    python -m host.benchmark              # evaluate every tracker in TRACKERS on the fixed test seeds
    python -m host.benchmark --tune       # grid-search cv tracker params on separate tuning seeds
    python -m host.benchmark data/a.jsonl # evaluate on recordings that contain ground truth

Rules, kept fixed so results stay comparable:
- Each confirmed track is matched to the nearest true person within MATCH_RADIUS_M at
  forecast time; tracks with nobody that close count as false tracks, not as errors.
- Forecast error = distance between the track's forecast and that same person's true
  position HORIZON_S later.
- Coverage = share of (frame, person in the radar's view) pairs with a confirmed track
  within MATCH_RADIUS_M. A tracker cannot improve its error by dropping hard tracks
  without coverage showing it.
- Errors are also split by whether the person is walking at forecast time
  (true speed above WALKING_MPS), since a model can win one case and lose the other.
- "no motion" forecasts the track's current position; it uses the same tracks, so the
  difference to "forecast" isolates the prediction itself.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from host import tracker
from host.viewer import load_session
from sim import floorplan as fpl
from sim import radar, run

HORIZON_S = 1.0
MATCH_RADIUS_M = 1.0
TEST_SEEDS = range(0, 5)
TUNE_SEEDS = range(100, 105)  # disjoint from TEST_SEEDS so tuning cannot overfit the reported number
N_PEOPLE = 2
DURATION_S = 300.0
TUNE_ACCEL_STD_MPS2 = (0.5, 1.0, 2.0, 4.0)
TUNE_MAX_COAST_S = (1.0, 2.0, 3.0)
TRACKERS = {
    "cv": tracker.Params(model="cv"),
    "imm": tracker.Params(model="imm"),
    "imm + LD2410": tracker.Params(model="imm", use_presence=True),
}
COVERAGE_SLACK = 0.02  # tuning may not buy lower error with more than this coverage loss
WALKING_MPS = 0.1


@dataclass(frozen=True)
class Result:
    kalman_err: np.ndarray
    static_err: np.ndarray
    walking: np.ndarray  # bool per forecast
    covered: int
    visible: int
    false_track_frames: int
    track_frames: int

    @property
    def coverage(self) -> float:
        return self.covered / max(self.visible, 1)

    @property
    def false_track_share(self) -> float:
        return self.false_track_frames / max(self.track_frames, 1)


def evaluate(frames: list[dict], params: tracker.Params = tracker.DEFAULT_PARAMS) -> Result:
    period = frames[1]["t_dev"] - frames[0]["t_dev"]
    ahead = round(HORIZON_S / period)
    truth = [{p["id"]: np.array([p["x"], p["y"]]) for p in f["truth"]["people"]} for f in frames]
    speed = [{p["id"]: np.hypot(p["vx"], p["vy"]) for p in f["truth"]["people"]} for f in frames]

    state = tracker.TrackerState()
    kalman, static, walking = [], [], []
    covered = visible = false_tracks = track_frames = 0
    for k, f in enumerate(frames):
        state = tracker.step(
            state, tracker.detections_from_frame(f), f["t_dev"], params, tracker.presence_from_frame(f)
        )
        tracks = state.confirmed
        pose = radar.Pose(f["truth"]["pose"]["x"], f["truth"]["pose"]["y"], f["truth"]["pose"]["yaw"])
        for pos in truth[k].values():
            if radar.in_view(*radar.to_sensor(pose, pos)):
                visible += 1
                covered += any(np.linalg.norm(tr.pos - pos) <= MATCH_RADIUS_M for tr in tracks)
        for tr in tracks:
            track_frames += 1
            pid, dist = min(((i, np.linalg.norm(tr.pos - p)) for i, p in truth[k].items()), key=lambda x: x[1])
            if dist > MATCH_RADIUS_M:
                false_tracks += 1
            elif k + ahead < len(frames):
                future = truth[k + ahead][pid]
                kalman.append(np.linalg.norm(tracker.forecast(tr, HORIZON_S) - future))
                static.append(np.linalg.norm(tr.pos - future))
                walking.append(speed[k][pid] > WALKING_MPS)
    return Result(
        np.array(kalman), np.array(static), np.array(walking, bool), covered, visible, false_tracks, track_frames
    )


def combine(results: list[Result]) -> Result:
    return Result(
        np.concatenate([r.kalman_err for r in results]),
        np.concatenate([r.static_err for r in results]),
        np.concatenate([r.walking for r in results]),
        sum(r.covered for r in results),
        sum(r.visible for r in results),
        sum(r.false_track_frames for r in results),
        sum(r.track_frames for r in results),
    )


def simulated_sessions(seeds: range) -> list[list[dict]]:
    fp = fpl.load(run.DEFAULT_MAP)
    helmet = run.parse_helmet(run.DEFAULT_HELMET)
    return [list(run.frames(fp, helmet, N_PEOPLE, DURATION_S, seed)) for seed in seeds]


def tune(sessions: list[list[dict]]) -> tuple[tracker.Params, list[tuple[tracker.Params, Result]]]:
    rows = []
    for accel, coast in itertools.product(TUNE_ACCEL_STD_MPS2, TUNE_MAX_COAST_S):
        params = tracker.Params(accel, coast)
        rows.append((params, combine([evaluate(s, params) for s in sessions])))
    best_coverage = max(r.coverage for _, r in rows)
    eligible = [(p, r) for p, r in rows if r.coverage >= best_coverage - COVERAGE_SLACK]
    return min(eligible, key=lambda pr: pr[1].kalman_err.mean())[0], rows


def report(r: Result) -> str:
    lines = [f"{'predictor':<10} {'people':<8} {'mean':>7} {'p90':>7} {'n':>8}"]
    for name, err in (("forecast", r.kalman_err), ("no motion", r.static_err)):
        for subset, mask in (("all", np.ones_like(r.walking)), ("walking", r.walking), ("still", ~r.walking)):
            e = err[mask]
            if len(e):
                lines.append(f"{name:<10} {subset:<8} {e.mean():>6.3f}m {np.percentile(e, 90):>6.3f}m {len(e):>8}")
    lines.append(f"coverage {r.coverage:.1%}   false tracks {r.false_track_share:.1%} of track-frames")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", type=Path, help="recordings with ground truth (default: simulate)")
    ap.add_argument("--tune", action="store_true", help="grid-search tracker params on the tuning seeds")
    args = ap.parse_args(argv)

    if args.tune:
        best, rows = tune(simulated_sessions(TUNE_SEEDS))
        print(f"{'accel':>6} {'coast':>6} {'mean':>7} {'p90':>7} {'coverage':>9}")
        for p, r in rows:
            mark = "  <- best" if p == best else ""
            print(
                f"{p.accel_std_mps2:>6} {p.max_coast_confirmed_s:>6} {r.kalman_err.mean():>6.3f}m "
                f"{np.percentile(r.kalman_err, 90):>6.3f}m {r.coverage:>8.1%}{mark}"
            )
        return

    if args.sessions:
        sessions = [load_session(p)[1] for p in args.sessions]
        if any("truth" not in f for s in sessions for f in s):
            ap.error("benchmark needs recordings with ground truth (simulated)")
        label = f"{len(sessions)} recording(s)"
    else:
        sessions = simulated_sessions(TEST_SEEDS)
        label = f"test seeds {TEST_SEEDS.start}-{TEST_SEEDS.stop - 1}, {N_PEOPLE} people, {DURATION_S:g} s each"
    print(f"1-second forecast error, {label}")
    for name, params in TRACKERS.items():
        print(f"\n== tracker: {name}")
        print(report(combine([evaluate(s, params) for s in sessions])))


if __name__ == "__main__":
    main()
