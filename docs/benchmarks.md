# Benchmark results

Produced with `python -m host.benchmark`. Rules (matching, coverage, the "no motion"
reference) are documented at the top of `host/benchmark.py` and must not change between
entries; if they do, re-run every row.

Test set: simulated, seeds 0–4, 2 people, 300 s each, `maps/demo_flat.json`, helmet at
(2.0, 2.2) facing +x. Tracker parameters tuned on seeds 100–104 (`--tune`).

## 1-second forecast error

| Date | Tracker | All: mean / p90 | Walking: mean / p90 | Still: mean / p90 | Coverage | False tracks |
|---|---|---|---|---|---|---|
| 2026-11-10 | No motion (reference) | 0.529 / 1.228 m | 0.998 / 1.412 m | 0.258 / 0.564 m | 94.8% | 16.9% |
| 2026-11-10 | Constant-velocity Kalman (σa = 1.0, coast 3 s) | **0.599 / 1.278 m** | 0.781 / 1.422 m | 0.494 / 1.126 m | 94.8% | 16.9% |

**Reading:** the Kalman forecast beats "no motion" for walking people (0.78 vs 1.00 m)
but loses for still people (0.49 vs 0.26 m): noise in its velocity estimate is
extrapolated for a full second. People are still about 63% of the time, so overall it
is worse than not predicting at all. Most false tracks are coasting tracks that slid on
after their person stopped. Both point at the same fix: a motion model that can switch
to "standing still" (IMM).
