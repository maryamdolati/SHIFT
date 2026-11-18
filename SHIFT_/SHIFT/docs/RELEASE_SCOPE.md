# Public source-release scope

This GitHub package is intended for **code inspection**, not to assert that benchmark evaluations have been rerun or to reproduce the manuscript's numerical tables from the archive alone.

## Included

- SHIFT temporal/spatial architecture and associated utilities.
- External published KRNO interface, with the upstream KRNO code kept outside this repository.
- Source-level benchmark loaders, data contract checks, metric implementations and evaluation routines.
- Explicit experimental controls and selected algorithmic/protocol regression tests.
- Research configuration registry; known unknowns and discrepancies remain visible.
- SHA-256 source-file manifest for release integrity.

## Deliberately excluded

- Raw, processed, or access-controlled datasets; record identifiers; medical records.
- Trained weights, checkpoint archives, logs, empirical output tables, and generated plots.
- The previously included legacy `experiments/scripts/shift_review/prepare_mujoco.py` script. It independently sampled masks and generated a manifest incompatible with the provenance contract in `load_irregular`; it is **not** an official MuJoCo data exporter. Consult the upstream's native preprocessing instead.
- Any copied/vendored code from the official KRNO repository.

## Release-specific changes

1. `shift-run` now defaults to the `shift` architecture; other architectures remain selectable explicitly.
2. The incompatible legacy MuJoCo exporter was excluded rather than presented as a native preprocessing path.
3. The main README prioritizes source inspection, the exact code map and honest provenance limitations.
4. Continuous integration performs **static source checks only**; it does not run training, benchmark tests, or claim numerical reproducibility.
5. The source manifest is regenerated for this archive.

These changes do not edit the core SHIFT neural-operator architecture, its numerical training objectives, the original KRNO source, the benchmark result tables, or any model weights.

## Author pre-publication checklist

- [ ] Confirm public release is compatible with any active double-blind review policy.
- [ ] Confirm author ownership/authorization and select a LICENSE for SHIFT-specific code.
- [ ] Check repository history for embedded credentials, private data, patient IDs or large artifacts.
- [ ] Use a new clean repository or verify its existing history; `.gitignore` does not erase already tracked files.
- [ ] Distinguish the revised source from any historical code/configuration that generated published or submitted tables.
- [ ] Pin the upstream KRNO commit before claiming any fully specified experimental comparison.
