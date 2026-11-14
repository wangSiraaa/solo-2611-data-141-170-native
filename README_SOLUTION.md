# `passes` — visible communication windows (synthetic fixture)

A Python 3.11 solver for one synthetic LEO TLE and one fixed WGS84 ground station over a
24-hour UTC plan. It publishes only the visible sub-windows supported by the SGP4 orbit
and the supplied periodic horizon mask. No antenna scheduling, link budget, orbit
fitting, GUI, or real-space-weather / EOP data are involved.

## Quick start

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .          # provides the `passes` console script

.venv/bin/passes --out out          # exit 0 if all independent checks pass
.venv/bin/passes --out out --quiet  # artifacts only
```

Equivalent: `.venv/bin/python -m passes --out out`.

Artifacts in `--out` (default `out/`):

| File | Contents |
| --- | --- |
| `passes.json` | Model description, search diagnostics, pass tree, window list, boundary summary, precision limits |
| `boundary_evidence.json` | Per-boundary bracket, refined root, TEME/ECEF/ENU coordinates, transition diagnosis, scalar re-solve |
| `observations.csv` | Reviewable observation table (boundaries + regular grid + peaks) with all intermediate coordinates |
| `verification.json` | ENU axial checks, scalar-vs-vector comparison, fine-audit matches, occlusion-gap audits |

Exit codes: `0` success and all checks green, `1` solver ran but a verification check failed,
`2` malformed input, `3` SGP4 propagation failure (located timestamp written to
`propagation_error.json`).

## Numeric model (exactly as the task README specifies)

1. **Propagation.** `sgp4==2.23` `SatrecArray.sgp4(jd, fr)` returns TEME position in **km**.
   The Julian date is kept as a two-part value (`jd` integer part, `fr` day fraction) so that
   seconds-of-day are not added to a ~2.46e6-day number. SGP4 error codes are reported, never
   treated as “invisible”.
2. **Earth rotation.** `D = (JD - 2451545.0)`, `T = D/36525`,
   `GMST = (280.46061837 + 360.98564736629·D + 0.000387933·T² − T³/38710000) mod 360`.
   UT1 = UTC, polar motion = 0.
3. **TEME → ECEF.** `x = cos θ·x_teme + sin θ·y_teme`, `y = −sin θ·x_teme + cos θ·y_teme`,
   `z = z_teme`.
4. **Station.** WGS84 `a = 6378.137 km`, `f = 1/298.257223563`. The input
   `ellipsoid_height_m` is **ellipsoidal (geodetic) height in metres**; it is converted to
   kilometres exactly once in `passes/io.py`. The site vector and the orthonormal
   ECEF→ENU basis are built from the geodetic latitude/longitude.
5. **ENU / angles.** `ENU = R·(r_ecef − r_site)`, azimuth `atan2(east, north)` in
   `[0,360)` degrees **clockwise from true north**, elevation `atan2(up, |horiz|)` in degrees.
   No refraction.
6. **Horizon.** Periodic piecewise-linear interpolation of `horizon.csv`; the last node is
   joined to the first through 360°. The azimuth seam is only a wrap in the *interpolation
   domain*: azimuth is unwrapped (`np.unwrap`) for trajectory reasoning, so 359°→1° is a
   +2° step, never a −358° jump.
7. **Visibility.** `elevation ≥ horizon(azimuth)`. Plans are continuous UTC, half-open
   `[start, end)`. Zero-duration tangencies are not windows.

## Window search and boundary refinement

| Stage | Step | Purpose |
| --- | --- | --- |
| Coarse scan | **0.25 s** (configurable `--search-step`) | Generate candidate sign-change brackets only. Never used to join samples into a window. |
| Verification scan | **0.02 s** (`--verify-step`) | Rescan the whole day; every sign change is added as a bracket, so a short occlusion hidden between coarse samples is still found. Also records azimuth for node crossings. |
| Mask-node probes | **0.0025 s** (`--node-probe-step`) | Between successive azimuth-node crossings, segments whose threshold exceeds the base 8° horizon are sampled densely. This catches a narrow obstacle crossing even when the pass is fast. |
| Audit scan | **0.005 s** (`--audit-step`) | Fully independent full-day rescan; its refined roots are matched to the published roots within the plan tolerance. |
| Root refinement | Brent `xtol` **1e-4 s** | Applied to the continuous margin `gap(t) = elevation(t) − threshold(azimuth(t))`. The margin is continuous even across mask nodes, though its derivative is not. |
| Boundary check | ±0.2 s, ±0.05 s, ±0.001 s | Independent scalar model re-solves each crossing and re-evaluates visibility on both sides. |

Brackets from all stages are refined, then **merged only within a few `xtol`** (≈4e-4 s) to
remove grid-phase duplicates. A genuine short occlusion is seconds long, so it stays distinct.
Every resulting pair of roots is classified at its midpoint: positive-length visible pieces
become separate windows; anything ≤ 1 µs is discarded as a tangency and listed in
`search.discarded_tangencies`.

Each boundary record carries:

- UTC, `seconds_from_start_s`, JD, GMST;
- azimuth, elevation, horizon threshold, signed gap;
- full TEME, ECEF and ENU coordinates at the boundary, and samples 0.2 s before/after;
- the refinement bracket, residual, iteration count and contributing sources;
- a transition diagnosis (mask-feature occlusion/clearing vs. orbital elevation rise/set);
- an independent scalar re-solve and visibility flips at the plan tolerance;
- `within_plan_tolerance`, computed against the plan's 0.2 s requirement.

Reasons used: `enter_visibility` / `exit_visibility`, with detail `elevation_rise_above_horizon`,
`elevation_set_below_horizon`, `occluded_by_horizon_feature`, `horizon_feature_cleared`; plan-edge
boundaries use `start_of_plan_truncation` / `end_of_plan_truncation`.

## Verification (always run)

1. **ENU axial-vector check** (`verification.enu_axis_checks`): displacements of 1000 km along
   each axis must map to pure E/N/U; the east×north cross product must equal up; the ellipsoid
   surface normal must equal up; the basis must be orthonormal to 1e-12.
2. **Independent scalar model** (`passes/verify.py`): a second code path with its own `Satrec`,
   scalar `sgp4`, explicit (non-matrix) site/ENU algebra and hand-written horizon bracketing. It
   agrees with the vectorised chain to ~1e-12 km in ENU and ~1e-14° in angles; the maxima are
   recorded in `verification.scalar_vector_comparison`.
3. **Boundary sides.** Each crossing is re-solved from the scalar model and sampled on both sides
   at several offsets; visibility must flip and the two models must agree.
4. **Audit root matching.** An independent 0.005 s scan must find exactly the same crossings.
   `unmatched_audit_roots` and `unmatched_solver_roots` are both zero on success.
5. **Short-gap audit.** The interior of every occlusion gap between adjacent windows of the
   same pass is re-sampled at 0.005 s (excluding a 1 ms endpoint guard for root residual) to
   prove there is no hidden visible sub-window.
6. **Boundary neighbourhood audit.** ±1 s around every root contains exactly one sign change.

`verification.ok` is the conjunction of all of the above, and the process exits non-zero if
any check fails.

## What the fixture demonstrates

The `horizon.csv` 36.087° block spanning azimuth **270.687°–273.087°** turns the ~17:13 UTC
pass into **two** windows separated by a **2.684 s** full occlusion, even though the satellite
is still ~28° above the *base* 8° horizon. All six other passes are single windows:

```
pass 1  00:30:42.677556Z – 00:38:01.223944Z   peak 87.475°   1 window
pass 2  15:35:36.533134Z – 15:40:36.047337Z   peak 15.028°   1 window
pass 3  17:11:03.775120Z – 17:13:19.117724Z   peak 27.779°   2 windows  ← split by 36° obstacle
        17:13:21.801907Z – 17:18:05.222061Z   peak 40.999°
pass 4  18:51:22.706668Z – 18:52:42.366383Z   peak  8.339°   1 window
pass 5  22:07:26.233163Z – 22:09:43.023127Z   peak  9.038°   1 window
pass 6  23:42:24.931040Z – 23:49:32.957879Z   peak 47.502°   1 window
```

A 60 s sampling grid would step straight across the 2.7 s occlusion (both neighbouring
one-minute samples are visible); the verification/node-probe/audit layers are what make the
split reliable. This is exercised by a unit test.

## Precision limits (stated in `passes.json.precision_limits`)

- The TLE is synthetic; SGP4 remains the orbital model; no external ephemeris/EOP/leap seconds.
- UT1=UTC and polar motion=0 per the task model. Real UT1−UTC near 0.9 s would rotate the Earth
  by ~0.004° and shift boundaries by ≪0.2 s; that is outside the agreed model.
- No refraction; geometric elevations. Real refraction near the horizon is ~0.5°.
- The piecewise-linear horizon is treated as exact; geometry between nodes is not modelled.
- Time strings are µs-rounded; `seconds_from_start_s` is authoritative.
- Detection floor: a complete occlusion shorter than ~2× the local sampling step (≈2×0.0025 s
  on raised node segments) cannot be guaranteed. The fixture's occlusion is ~1000× that floor.

## Tests

```sh
.venv/bin/python -m pytest tests/ -q
```

Covers the periodic horizon join, azimuth-wrap unwrapping, node crossings, km/m conversion,
ENU axial vectors, GMST at J2000, scalar vs. vector agreement, located propagation failures, the
1-minute-grid blind spot, the split pass, per-boundary tolerance and independent refinement,
tangency discarding, duplicate-root merging, endpoint truncation, and plan validation.
