# Periodic molecular-trajectory unwrapping (`unwrap`)

Recover un-folded finite molecules and their continuous centre-of-mass (COM)
trajectories from coordinates that were folded back into a fixed periodic
(triclinic) cell. The implementation deliberately **refuses to guess**: a step
is only continued when exactly one lattice image satisfies the plan's maximum
COM displacement.

## Install

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.11, NumPy 1.26, SciPy 1.13 (SciPy is not required by the current code
path but is available).

## Run

```sh
.venv/bin/python unwrap \
  --trajectory fixtures/trajectory.npz \
  --topology   fixtures/topology.json \
  --plan       fixtures/plan.json \
  --output     out/report.json \
  --npz-output out/unwrapped.npz
```

Main-trajectory and negative (cycle) runs:

```sh
# main: bond bound comes from plan.json (1.2 nm)
.venv/bin/python unwrap --trajectory fixtures/trajectory.npz \
  --topology fixtures/topology.json --plan fixtures/plan.json \
  --output out/report.json --npz-output out/unwrapped.npz

# cycle-case: bond bound comes from cycle-topology.json (3.6 nm)
.venv/bin/python unwrap --trajectory fixtures/cycle-case.npz \
  --topology fixtures/cycle-topology.json --plan fixtures/plan.json \
  --output out/cycle-report.json
```

An explicit `--max-bond-nm` overrides the bond bound for either case.

## Input conventions

* `positions_nm` is `frame × atom × xyz`, nm; atom order follows `topology.atoms`.
* `cell_columns_nm` is 3×3 with the three lattice vectors as **columns**.
* `time_ps` is strictly increasing, ps.
* `topology.bonds_zero_based` are zero-based global atom indices.
* `topology.atoms[i].mass` is relative atomic mass.
* The cell is fixed; only integer lattice translations are considered.

## Method

1. **Within-frame reconstruction (per molecule, per frame).**
   For each bond, all integer lattice images `m` with
   `‖Δr + H·m‖ ≤ max_bond_nm` are enumerated. The bound is a rigorous finite
   box: with `c = −H⁻¹Δr`, every feasible `m` satisfies
   `|m_k − c_k| ≤ ‖row_k(H⁻¹)‖ · max_bond_nm`; the box is then filtered by the
   exact Euclidean distance. The root atom is fixed to its folded image (this
   is the arbitrary gauge), and a spanning tree proposes offsets, **but every
   bond is checked, including cycle-closing non-tree bonds.** If no assignment
   closes all bonds, the frame is reported as `no_consistent_lattice_image`
   with a concrete closure-conflict witness. Multiple internal assignments are
   reported as `ambiguous_within_frame`.

2. **Continuity (per molecule).**
   The mass-weighted COM is computed from the reconstructed atoms. For each
   adjacent frame with `dt ≤ max_gap_ps`, all integer translations with
   `‖ΔCOM + H·m‖ ≤ max_com_speed_nm_per_ps · dt` are enumerated.
   * Exactly one → continue, add it to the cumulative translation.
   * Zero or more than one → **break**. The nearest image is never chosen.
   * `dt > max_gap_ps` → break (time under-sampling is not hidden).
   Cumulative translation resets at each break, so displacement is never
   accumulated across segments.

3. **MSD.**
   For each requested lag, only frame pairs with `|Δt − lag| ≤ lag_tol`
   **inside one continuous segment** are counted. Lags with no such pair are
   marked `covered=false`, `msd_nm2=null` — never zero. Per-molecule and pooled
   pair counts are reported.

## Output

`--output` is a JSON report containing:

* `checks` — validated masses, bond references, time diffs, cell determinant /
  condition number, folded-coordinate range, provenance SHA-256 hashes.
* `summary` — frame/atom/molecule counts, break list, per-lag pooled MSD, and
  the origin-gauge note.
* `molecules.<id>` — per-frame reconstruction status, lattice offsets (absolute
  and per-step), unwrapped atoms, bond lengths, COM, segment id, cumulative
  translation, and break reasons; per-molecule MSD; boundary crossings.
* `conventions` — units, cell layout, lattice-image definition, gauge note.

`--npz-output` stores the reconstructed arrays with the original `frame_index`,
`atom_index`, and `atom_ids`, plus `positions_unwrapped_nm`,
`atom_lattice_offsets_total`, `com_unwrapped_nm`, `segment_id`,
`step_lattice_translation`, and `cumulative_lattice_translation`. Invalid
entries (e.g. a frame whose molecule could not be reconstructed) are marked via
the `*_valid` masks and left as NaN; they are never silently zeroed.

## Gauge / traceability

The absolute lattice image of the first frame in each continuous segment is
**arbitrary** — the folded input does not determine it. All absolute
coordinates are reported in that per-segment gauge. Molecular shape, bond
lengths, and within-segment displacements are gauge-invariant: shifting a whole
segment by one lattice vector changes neither. Every unwrapped coordinate folds
back exactly to the original folded coordinate (`unwrapped_frac − offset ==
folded_frac`), which the report verifies per frame.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Covers: candidate search vs brute force, triclinic component-rounding
counterexample, analytic bond lengths / mass-weighted COM / displacements,
gauge invariance, unique/ambiguous/none/gap continuity decisions, no
cross-break accumulation, MSD pair counts and uncovered lags, fixture
boundary-crossing detection, and the cycle-closure negative case.
