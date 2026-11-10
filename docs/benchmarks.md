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
| 2026-11-10 | IMM: walking / manoeuvring / still, misses as evidence (untuned) | **0.393 / 0.981 m** | 0.752 / 1.398 m | 0.206 / 0.447 m | 97.0% | 10.6% |
| 2026-11-10 | IMM + LD2410 still/moving readings (untuned) | **0.380 / 0.958 m** | 0.747 / 1.372 m | 0.201 / 0.450 m | 98.1% | 11.6% |

**Reading:** the Kalman forecast beats "no motion" for walking people (0.78 vs 1.00 m)
but loses for still people (0.49 vs 0.26 m): noise in its velocity estimate is
extrapolated for a full second. People are still about 63% of the time, so overall it
is worse than not predicting at all. Most false tracks are coasting tracks that slid on
after their person stopped. Both point at the same fix: a motion model that can switch
to "standing still" (IMM).

**IMM (2026-11-10):** 34% lower mean error than the constant-velocity baseline (p90 −23%),
and now better than "no motion" overall (0.39 vs 0.47 m on its own tracks). The gain is
almost all on still people (0.49 → 0.21 m): each missed detection is treated as evidence
that the person stopped, so tracks no longer slide on. Coverage rose and false tracks fell
for the same reason. Parameters are starting values, not tuned; tuning waits for real
sensor data. Known limitation: a stopped person who is still detected every frame is
read as "walking at ~0 m/s", so velocity noise still leaks into the forecast; LD2410
still readings are the planned fix.

**IMM + LD2410 (2026-11-10):** a modest further gain, 0.393 → 0.380 m mean and coverage
97.0 → 98.1%, with false tracks slightly up (10.6 → 11.6%) because range-only readings
keep quiet tracks alive longer. Each reading supports at most one track (the LD2410
reports only the nearest still and nearest moving person); an earlier version that let a
reading support every track at that distance raised false tracks to 12.8%. The forecast
counts rise because more seated people stay tracked. Gains on real hardware depend on how
well the LD2410 sees still people through walls, which is unmeasured.
