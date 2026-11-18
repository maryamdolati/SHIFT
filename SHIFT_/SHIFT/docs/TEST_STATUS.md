# Verification scope for this source-only GitHub release

The source-release packaging step checked Python syntax and the SHA-256 file inventory. These checks do **not** import KRNO, train a model, evaluate a benchmark, or demonstrate alignment with historical SHIFT results.

**Checked for this release:**

- `python -m compileall -q krno scripts experiments run_paper_protocols.py train_script.py`
- `python scripts/verify_source_manifest.py`
- No checkpoint or data artifacts are included in the archive.

**Not performed for this release:**

- Full `pytest` test suite or GPU/model execution.
- Native upstream KRNO integration validation.
- Benchmark evaluation or result reproduction.
- End-to-end proof that reviewer controls improve forecasting performance.

Regression tests are provided as inspectable source under `tests/`. The existence of these tests must not be presented as a claim that they were executed as part of this release. See [reproduction status](REPRODUCTION_STATUS.md) for scientific evidence and protocol limitations.
