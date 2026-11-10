import numpy as np
import pytest

from host import imm, tracker
from host.tracker import Detection, Params, TrackerState

DT = 0.1
R = np.eye(2) * 0.1**2
IMM = Params(model="imm")


def run(positions, params=IMM, seed=0):
    """positions: per frame a true (x, y), or None for a missed detection."""
    rng, state, history = np.random.default_rng(seed), TrackerState(), []
    for k, p in enumerate(positions):
        dets = [] if p is None else [Detection(np.asarray(p) + rng.normal(0, 0.1, 2), R)]
        state = tracker.step(state, dets, k * DT, params)
        history.append(state)
    return history


def walk_then_stop(n_walk=30, n_stop=30):
    walk = [np.array([4.5 + 1.0 * k * DT, 2.0]) for k in range(n_walk)]
    return walk + [walk[-1]] * n_stop


def test_probabilities_stay_normalised_through_every_operation():
    s = imm.init(np.array([1.0, 2.0, 0.5, 0.0]), np.eye(4) * 0.1)
    for op in (lambda s: imm.predict(s, DT), lambda s: imm.update(s, np.array([1.1, 2.0]), R), imm.miss):
        s = op(s)
        assert s.mu.sum() == pytest.approx(1.0)
        assert np.all(s.mu > 0)


def test_misses_shift_belief_towards_still():
    s = imm.init(np.zeros(4), np.eye(4))
    before = s.mu[2]
    for _ in range(5):
        s = imm.miss(s)
    assert s.mu[2] > before and s.mode == "still"


def test_steady_walk_is_walking_and_forecasts_ahead():
    path = [np.array([4.5 + 1.0 * k * DT, 2.0]) for k in range(50)]
    tr = run(path)[-1].confirmed[0]
    assert tr.mode in ("walking", "manoeuvring")
    assert np.linalg.norm(tracker.forecast(tr, 1.0) - (path[-1] + [1.0, 0.0])) < 0.3


@pytest.mark.parametrize("seed", range(5))
def test_switches_to_still_within_a_second_of_stopping(seed):
    # After stopping, the LD2450 sees a still person only about a third of the time.
    rng = np.random.default_rng(seed)
    positions = walk_then_stop()
    positions = positions[:30] + [p if rng.random() < 0.35 else None for p in positions[30:]]
    tr = run(positions, seed=seed)[39].confirmed[0]  # 1 s after stopping
    assert tr.mode == "still"
    assert np.linalg.norm(tracker.forecast(tr, 1.0) - positions[29]) < 0.25


def test_stopped_person_seen_every_frame_is_no_worse_than_cv():
    # Known limitation: constant detections favour "walking at ~0 m/s" over "still", so velocity
    # noise leaks into the forecast as with cv. LD2410 still readings are meant to fix this.
    positions = walk_then_stop()

    def mean_err(params):
        errs = [
            np.linalg.norm(tracker.forecast(s.confirmed[0], 1.0) - positions[-1])
            for seed in range(5)
            for s in run(positions, params, seed)[45:60]
        ]
        return np.mean(errs)

    assert mean_err(IMM) <= mean_err(Params(model="cv"))


def test_stopping_out_of_sight_does_not_slide_like_cv():
    positions = walk_then_stop()
    positions = positions[:30] + [None] * 20  # stops and the radar loses them
    imm_tr = run(positions)[-1].confirmed[0]
    cv_tr = run(positions, Params(model="cv"))[-1].confirmed[0]
    stop = positions[29]
    assert np.linalg.norm(imm_tr.pos - stop) < 0.5 * np.linalg.norm(cv_tr.pos - stop)
    assert imm_tr.mode == "still"


def test_unknown_model_fails_loudly():
    with pytest.raises(ValueError, match="unknown tracker model"):
        Params(model="lstm")
