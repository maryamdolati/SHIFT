# Reviewer-oriented source map

This guide maps the SHIFT manuscript's *algorithmic statements* and the concrete code-related reviewer concerns to files in this repository. It is a **source inspection guide**, not a claim that unexecuted experiments validate the paper or that every baseline was fairly retuned. The reference point for KRNO is the separately published [official KRNO implementation](https://github.com/srinathdama/ShiftingTime).

## 1. Core algorithm

| Manuscript component | Code entry | What is visible in the source |
| --- | --- | --- |
| Input, mask, and timestamp encoding | `krno/shift_krno/model.py` → `validate_input`, `TemporalEncoder` | Masked observations, masks, time gaps and relative times; strictly increasing processed times |
| Learned input-conditioned rates | `ConditionalWarp.forward` | Two networks compute interval-level `mu` and `lv` from adjacent encoded history and masks |
| Stochastic and deterministic regimes | `ConditionalWarp.forward` | Gaussian reparameterization and `softplus` positive rates; modes `conditional`, `deterministic`, `fixed`, `random`, `unconditional`, `shuffled` |
| Chronology-preserving time coordinates | `ConditionalWarp.forward` | Normalized positive rate × positive elapsed-time increments; cumulative map has endpoints 0 and 1 |
| Single resampling of encoded history | `latent_resample` | Piecewise-linear interpolation from sampled warped coordinates to a uniform latent grid |
| Multi-scale and gating | `MultiscaleSpectral.forward` | Parallel convolutions with kernels 3, 5 and 9, softmax gating, FFT transfer and residual fusion |
| Operator-backed forecast | `OriginalKRNOBridge` | Uses external published `khatriraonop.models.KhatriRaoNO_v2` and its operator interface |
| Complete temporal model | `SHIFT.forward` | Per-warp latent resampling → multi-scale module → KRNO → prediction heads → aggregation |
| Spatial model | `SpatialSHIFT.forward`, `OriginalSpatialKRNOBridge` | Spatial field encoding, shared temporal warp and original spatial KRNO interface |
| Forecast mixture and variance split | `ShiftOutput`, `SHIFT.forward`, `SpatialSHIFT.forward` | `mean(mus)`, `mean(vars_)`, `mean((mus - mean(mus))**2)` |

**Interpretation limit:** `u_temporal` is forecast variability under *sampled learned time mappings*, not epistemic uncertainty about a uniquely identifiable time axis. The uncertainty decomposition follows the variance identity of the represented mixture; calibration must be evaluated separately.

## 2. Source-level reviewer concerns

| Concern from reviews | Inspect | Evidence in code / remaining limit |
| --- | --- | --- |
| Larger architecture may explain gains | `scripts/capacity_match.py`, `scripts/reviewer_studies.py` | Parameter-count matcher and fixed-coordinate control present. Native KRNO is **not** automatically parameter-matched. No measured attribution inferred from these files alone. |
| Is stochasticity merely regularization? | `model.py` → `ConditionalWarp.forward`; `scripts/reviewer_studies.py` → `ARMS` | Controls `random`, `unconditional`, input noise and dropout are declared. They require completed matched evaluations for a scientific answer. |
| Use another history's warp rather than deleting warp | `model.py` → `transfer_donor_map`; `krno/shift_krno/reviewer.py`; `run.py` (`--donor-audit`) | Warped maps from donor observations can be transferred without donor targets. Donor selection and its variability are described in `docs/REVIEWER_STUDIES.md`. |
| Are informative regions actually expanded? | `scripts/synthetic_information.py` | Synthetic informative-region mechanism diagnostic supplied; no outcome is claimed. |
| Crypto last feature and WRMSE weights | `data.py` → `load_crypto`; `losses.py`; `evaluate.py`; `scripts/prepare_crypto_weights.py` | Asset weights, last-feature targeting, horizon and metric conventions are explicit; see `protocols/crypto.md`. |
| MuJoCo missingness: masking vs interpolation | `data.py` → `load_irregular`; `scripts/export_irregular.py`; `protocols/mujoco.md` | Loader requires observed-value/mask/timestamp provenance. No independent random-mask substitute is presented as official preprocessing. Original MuJoCo preprocessing provenance remains necessary. |
| Shallow Water 5-step vs rollout to 0.6 s | `evaluate.py` → `eval_shallow`; `protocols/shallow.md`; `tests/test_protocols.py` | 5-step blocks and iterative forecast interface are distinguishable; source includes a length/rollout regression test. |
| Fair seeds and horizon definitions | `protocol_registry.json`; `protocols/`; `run_paper_protocols.py` | Source-specific settings and unresolved historical ambiguities are recorded, rather than guessed. |
| Fixed-coordinate and deterministic warp controls | `scripts/reviewer_studies.py` → `ARMS`; `model.py` | Controls change time-reparameterization mode inside the SHIFT architecture. |
| Uncertainty/calibration baselines | `probabilistic.py`, `uncertainty_audit.py`, `scripts/evaluate_ensemble.py` | Mixture and ensemble auditing code exists. External probabilistic methods (including ProFITi) are **not** included as reproduced native baselines. |

## 3. Which code path is the primary method?

- **Full SHIFT:** `krno/shift_krno/model.py` → `SHIFT` (temporal) and `SpatialSHIFT` (spatio-temporal), selected with `--architecture shift` in `krno/shift_krno/run.py`.
- **Reference upstream KRNO:** official external repository and its native benchmark scripts. It is not modified or vendored here.
- **Local controls:** `KRNOReference`, `KRNOWarpOnly`, `KRNOControl`, spatial counterparts, `mode=fixed`, `mode=deterministic` and other experimental modes. These should never be mislabeled as the *native* published KRNO benchmark code.
- **Protocol planning:** `run_paper_protocols.py` provides separate labelled paths. Reconstruction guards signal unresolved manuscript settings.

## 4. Status of the code package

**Included:** model definitions, data and mask contracts, losses, metrics, reviewer control entry points, static tests, research scripts and protocol documentation.

**Not included:** raw data, access-controlled records, historical checkpoints, benchmark logs, empirical proof of gains, parameter-matched native reruns or independently verified numerical reproduction.

For known discrepancies see [`REPRODUCTION_STATUS.md`](REPRODUCTION_STATUS.md); for controlled experiment requirements see [`REVIEWER_STUDIES.md`](REVIEWER_STUDIES.md). Those qualifications are intentionally retained for transparent review.
