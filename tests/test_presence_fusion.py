import numpy as np
import pytest

from host import tracker
from host.tracker import Detection, Params, PresenceReading, TrackerState

DT = 0.1
R = np.eye(2) * 0.05**2
HELMET = np.array([0.0, 0.0])
FUSED = Params(model="imm", use_presence=True)


def run(frames, params=FUSED):
    """frames: per frame (detected positions, still_m reading or None)."""
    state, history = TrackerState(), []
    for k, (positions, still_m) in enumerate(frames):
        dets = [Detection(np.asarray(p, float), R) for p in positions]
        presence = PresenceReading(HELMET, None, still_m)
        state = tracker.step(state, dets, k * DT, params, presence)
        history.append(state)
    return history


def sit_down(seconds_quiet: float, still_m: float | None, at=(3.0, 0.0)):
    """Seen still for 2 s, then the LD2450 loses them; the LD2410 keeps reporting still_m."""
    return [([at], still_m)] * 20 + [([], still_m)] * round(seconds_quiet / DT)


def test_presence_from_frame():
    assert tracker.presence_from_frame({"targets": []}) is None
    p = tracker.presence_from_frame({"pose": {"x": 1, "y": 2}, "presence": {"state": "still", "still_m": 3.375}})
    np.testing.assert_allclose(p.helmet, [1, 2])
    assert (p.still_m, p.moving_m) == (3.375, None)


def test_presence_needs_imm():
    with pytest.raises(ValueError, match="model='imm'"):
        Params(model="cv", use_presence=True)


def test_still_readings_keep_a_quiet_track_alive():
    assert run(sit_down(8.0, still_m=3.375))[-1].confirmed  # LD2410 sees them: kept
    assert not run(sit_down(8.0, still_m=None))[-1].tracks  # nothing: gone after 3 s
    assert not run(sit_down(8.0, still_m=5.625))[-1].tracks  # reading is someone farther away


def test_range_only_evidence_has_a_cap():
    history = run(sit_down(25.0, still_m=3.375))
    alive_s = sum(bool(s.confirmed) for s in history[20:]) * DT
    assert alive_s == pytest.approx(tracker.MAX_COAST_WITH_PRESENCE_S, abs=0.15)


def test_still_reading_shifts_belief_to_still():
    # Seen every frame while standing: imm alone reads "walking at ~0 m/s".
    frames = [([(3.0, 0.0)], 3.375)] * 30
    fused = run(frames)[-1].confirmed[0]
    plain = run(frames, Params(model="imm"))[-1].confirmed[0]
    assert fused.mode == "still"
    assert fused.imm.mu[2] > plain.imm.mu[2]


def test_reading_goes_to_one_track_only():
    # Two people standing at 2 m and 4 m; the LD2410 reports the nearer one.
    frames = [([(2.0, 0.0), (4.0, 0.0)], 2.125)] * 20
    near, far = sorted(run(frames)[-1].confirmed, key=lambda tr: tr.pos[0])
    assert near.last_presence_t > 0
    assert far.last_presence_t == -np.inf
