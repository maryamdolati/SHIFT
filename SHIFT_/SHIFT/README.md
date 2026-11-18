# SHIFT — Input-Conditioned Stochastic Temporal Reparameterization for Neural Operator Forecasting

**Source-code release for the SHIFT research manuscript.** SHIFT learns a history-conditioned distribution over *chronology-preserving* temporal reparameterizations, resamples an encoded history onto a uniform latent-time grid, and uses multi-scale spectral–temporal processing together with a Khatri–Rao neural operator (KRNO) to forecast future observations.

> **Scope of this release.** This archive exposes model source, dataset interfaces, benchmark-protocol descriptions, controlled variants, and source-level tests. It deliberately includes **no** raw datasets, sensitive records, pretrained weights, training logs, or final benchmark scores. The supplied implementation was assembled and protocol-audited for source inspection; it is **not independently certified as the exact historical code/configuration used for each manuscript result**. See [reproduction status](docs/REPRODUCTION_STATUS.md). Source availability alone does not validate scientific claims or establish fair numerical comparisons.

## Start here: read the implementation

| What to inspect | Source | Main functions / classes |
| --- | --- | --- |
| Input-dependent stochastic monotone warp | [`krno/shift_krno/model.py`](krno/shift_krno/model.py) | `TemporalEncoder`, `ConditionalWarp.forward` |
| Warped-to-uniform latent-grid interpolation | [`model.py`](krno/shift_krno/model.py) | `latent_resample`, `transfer_donor_map` |
| Multi-scale temporal branches, adaptive gates, FFT | [`model.py`](krno/shift_krno/model.py) | `MultiscaleSpectral` |
| Full temporal architecture and uncertainty decomposition | [`model.py`](krno/shift_krno/model.py) | `SHIFT`, `ShiftOutput` |
| Spatio-temporal extension | [`model.py`](krno/shift_krno/model.py) | `SpatialSHIFT` |
| Connection to the **external**, published KRNO library | [`model.py`](krno/shift_krno/model.py) | `OriginalKRNOBridge`, `OriginalSpatialKRNOBridge` |
| Data contracts (observations, masks, timestamps, targets) | [`data.py`](krno/shift_krno/data.py) | Dataset loaders and provenance checks |
| Loss functions and metrics | [`losses.py`](krno/shift_krno/losses.py), [`evaluate.py`](krno/shift_krno/evaluate.py) | Forecast losses, benchmark evaluation, rollouts |
| Source-level tests | [`tests/`](tests/) | Warp, mask, protocol, evaluation, control checks |

**A reviewer-oriented map of method claims, controls, and protocol concerns is in [`docs/REVIEWER_CODE_MAP.md`](docs/REVIEWER_CODE_MAP.md).** Each entry points to an inspectable implementation, and distinguishes implemented controls from unverified experimental evidence.

## Method at a glance

Given observed values, observation masks, time stamps, and time gaps, SHIFT constructs an input-conditioned distribution over interval rates. For each sample, positive rates and positive physical-time gaps generate normalized increments:

\[
 r_i=\operatorname{softplus}(\mu_i+\sigma_i\epsilon_i),\qquad
 m_i=\frac{r_i\Delta t_i}{\sum_j r_j\Delta t_j},\qquad
 \phi(t_{i+1})=\phi(t_i)+m_i.
\]

The cumulative map is ordered and maps the observed time span to \([0,1]\). Encoded observations are interpolated once onto a uniform latent-time grid. The resulting sequence is processed by three temporal convolutional branches (kernel sizes 3, 5, 9), input-dependent scale gating, spectral refinement, and the KRNO-backed forecasting operator. Repeating this over sampled maps produces a finite forecast mixture. The reported `u_value` and `u_temporal` objects respectively represent average conditional variance and between-warp variance of conditional means. These are **model-defined** quantities; the between-warp term is not by itself proof of an identifiable physical clock or calibrated uncertainty.

## What this repository includes

```text
SHIFT/
├── README.md                        Overview and source entry points
├── krno/shift_krno/                 SHIFT models, data contracts, losses and evaluation
├── scripts/                         Research controls, dataset exporters and diagnostics
├── experiments/scripts/             Thin benchmark wrappers and auxiliary audit helpers
├── experiments/dataset/README.md    Data inputs and provenance expectations
├── protocols/                       Per-benchmark implementation contracts
├── protocol_registry.json           Declared benchmark settings and known unknowns
├── docs/                             Code map, reviewer studies, release caveats
├── tests/                            Source-level regression tests
├── run_paper_protocols.py           Explicit native-KRNO vs SHIFT planning
├── pyproject.toml                    Python package metadata
└── SOURCE_MANIFEST.json              SHA-256 inventory of released files
```

Benchmarks covered by the source interfaces: **MIMIC-III, USHCN, Human Activity, MuJoCo, Crypto, Player Trajectory, M4 Quarterly/Weekly/Daily, and Shallow Water**. Dataset definitions and metric conventions are documented in [`protocols/`](protocols/) and [`experiments/dataset/README.md`](experiments/dataset/README.md).

## KRNO baseline and provenance

SHIFT is built upon the **published original KRNO** framework:

- Srinath Dama, Kevin Course and Prasanth B. Nair, *Shifting Time: Time-series Forecasting with Khatri-Rao Neural Operators*, ICML 2025, [paper](https://proceedings.mlr.press/v267/dama25a.html) and [official implementation](https://github.com/srinathdama/ShiftingTime).
- The original `khatriraonop` library is an **external dependency**. This archive does **not** copy or modify its upstream source. Its implementation choices are preserved as the reference baseline, including those for which additional verification might be useful.
- `krno_reference` and `krno_warp` in this repository are **local controlled-architecture variants**, not a claim to have replaced the original task-specific KRNO runners. `run_paper_protocols.py` distinguishes the upstream native baseline from the local variants.

The repository does not assert that an architecture-size comparison alone isolates the effect of warping. For that question inspect the fixed-coordinate, deterministic-warp, random/unconditional-warp, input-noise, dropout and donor-warp code paths in the [reviewer map](docs/REVIEWER_CODE_MAP.md).

## Installation (optional; not required to inspect the source)

The package is designed for Python 3.10 with PyTorch; install the upstream KRNO library from its official repository and record the exact upstream commit if using the original operator. The upstream library may need additional dependencies specified by its maintainers. This repository's Python package can be installed with:

```bash
pip install -e '.[test]'
# For original operator experiments, separately install:
# https://github.com/srinathdama/ShiftingTime/tree/main/krno
```

The command-line entry point `shift-run` selects **`--architecture shift`** by default. `--architecture krno_reference` and `--architecture krno_warp` select explicitly labelled local control paths. The source can be reviewed without executing these commands.

Optional source-oriented checks (not benchmark reproduction):

```bash
python scripts/verify_source_manifest.py
python -m compileall -q krno scripts experiments run_paper_protocols.py
```

The included tests and protocol launchers are supplied for transparency, not as evidence that results have been reproduced. No benchmark training or evaluation has been performed as part of preparing this GitHub archive.

## Research controls, limits and attribution

The controlled study launcher is [`scripts/reviewer_studies.py`](scripts/reviewer_studies.py). Model intervention modes are implemented in `ConditionalWarp` and linked through [`krno/shift_krno/run.py`](krno/shift_krno/run.py). Auxiliary scripts cover independent-donor mappings, a synthetic informative-region task, parameter-count matching, finite-mixture scoring, and an ensemble comparator. **Having the source for a control is not the same as reporting a completed control experiment.** The [reviewer study notes](docs/REVIEWER_STUDIES.md) document what is implemented and what would still be needed for empirical conclusions.

Some entries in [`protocol_registry.json`](protocol_registry.json) are deliberately marked as reconstructed or unresolved where exact historical settings were unavailable in the supplied source. In particular, **do not** interpret the launchers as a certified reproduction of all paper tables. See the [source discrepancies](docs/REPRODUCTION_STATUS.md).

**Data/privacy:** No patient-level records, record identifiers, credentials, datasets or checkpoints should be committed. MIMIC-III access terms apply. See [`NOTICE.md`](NOTICE.md).

**License:** A license for the authors' SHIFT-specific code has **not** been selected in this release. The upstream KRNO library has its own license. Choose and add an appropriate license before offering reuse rights; do not copy upstream source into this repository without checking license obligations.
