# Reviewer experiments

The following are experimental questions, not guaranteed outcomes. These experiments are not evidence until run on verified inputs with frozen settings. Retain all seed-level artifacts and report variability.

## Geometry versus stochastic regularization

Use `scripts/reviewer_studies.py` with a predeclared JSON configuration. All arms use a common objective (`mix_nll` or positive-weight `mse_plus_nll`), identical data, hyperparameters, seed list and evaluation metric, with warp regularization disabled to isolate the intervention. The Full SHIFT controls retain the surrounding model and vary only the designated intervention. Choose input-noise amplitude and dropout probability using validation, not test. Use an equal validation-search budget for every arm if tuning is performed.

Example plan (the JSON must contain actual chosen settings, not copied scores):

```bash
python scripts/reviewer_studies.py --dataset crypto --data-root data --upstream-root external/ShiftingTime --config configs/crypto.json --out runs/reviewer
```

Required config fields: `epochs`, `patience`, `batch_size`, `grad_accum`, `lr`, `weight_decay`, `width`, `kernel_hidden`, `train_samples`, `eval_samples`, `input_noise`, `dropout`, `objective`, `capacity_widths`. For regular cases, explicitly provide `lookback`, `block`, `train_stride` and the selected validation convention. For `mse_plus_nll`, provide a positive `nll_weight`. K must be at least two for stochastic mixture training and evaluation. Use `--execute` for validation-only runs. After finalizing the configurations, `--test --execute` uses the frozen checkpoints and writes test artifacts separately. It does not train on test. Do not repeatedly modify settings after inspecting test results.

For a trained conditional model, the donor audit selects an independent validation history. It transfers its realized monotone map by normalized physical-time interpolation, without using donor targets or replacing recipient observations. This handles unequal sequence lengths and batch size one. The current implementation uses one fixed donor per run; a stronger study should repeat over predeclared independent donors and report their variability.

## Capacity and tuning

Use `scripts/capacity_match.py` with a predeclared list of widths and kernel widths. It counts actual trainable parameters, picks the closest candidate without inspecting prediction errors, and fails if the relative gap exceeds the declared tolerance. It matches the local KRNO control (not the unmodified native baseline). Report exact parameter counts for native KRNO, SHIFT and matched control separately. Apply the selected candidate to a separate control run using the same likelihood, split, horizon and tuning budget. The shared reviewer launcher does not itself certify that a parameter match was achieved.

## Known informative regions

`python scripts/synthetic_information.py --out runs/synthetic.json --seeds 1 2 3 4 5 --epochs 20 --lr 0.001 --samples 4` defines a separate simulated task with either broad or moving localized signal support. The target depends on a latent amplitude. It records the fraction of warp mass assigned to informative intervals versus their fraction of physical-time length, and prediction error across all selected modes. The signal-region marker is part of the synthetic input, not a future target. This is a mechanism diagnostic, not an original KRNO benchmark or a proof of an identifiable physical clock.

## Uncertainty and calibration

Train every stochastic mixture arm with K>=2 and a proper likelihood. Record finite Gaussian mixture marginal NLL, CRPS, PIT, empirical interval coverage, interval scores and calibration error. Compute between-warp variance only for the geometry model and compare it with forecast errors, including example-level Spearman association and high/low error quartiles. For recursive forecasts the current audit treats each local block as an example; full-trajectory covariance and trajectory-level rank association require a distinct propagated-uncertainty analysis.

Evaluate the 0%, 30%, 50%, and 70% native MuJoCo regimes with matched seeds and independently verified masks. A rise in temporal variance is a hypothesis, not a required result. The local probabilistic KRNO control is included; external probabilistic neural operators and ProFITi require their own native, implementation-matched reruns. Do not substitute citation values for calibrated paired results.

## Reviewer issue matrix

| Concern | Code path | Status |
|---|---|---|
| Crypto wrong feature or missing weights | `data.py`, `losses.py`, `prepare_crypto_weights.py` | Patched; direct native formula test |
| Crypto wrong total horizon and windows | `data.py`, `evaluate.py`, `losses.py` | Patched; native fixture parity |
| Player metric horizon or window subsampling | `data.py`, `evaluate.py` | Full native 30-step horizon and all five origins; actual historical run pending |
| MuJoCo interpolation or absent masks | `load_irregular` | Reject non-native interpolation; original dependency/data still needed |
| Shallow 5-step-only test | `eval_shallow` | 11 recursive blocks, forecast frames only |
| Seed cherry-picking | `protocol_registry.json` | Source-documented seeds locked; Quarterly provenance qualified |
| Capacity mismatch | `capacity_match.py` | Actual-count candidate matcher; training/evaluation not run |
| Random and unconditional warp | `reviewer_studies.py`, `model.py` | Control paths implemented |
| Noise and dropout | `run.py`, `reviewer_studies.py` | Shared-budget controls implemented |
| Donor warp | `reviewer.py`, `model.py` | Independent donor implementation tested |
| Informative region hypothesis | `synthetic_information.py` | Task implemented, no reported outcome |
| Trained variance and K=1 collapse | `run.py`, `losses.py` | Likelihood and sampling guards; no manufactured uncertainty |
| Error/uncertainty association | `uncertainty_audit.py` | Example/block-level statistics implemented |
| Equal losses and tuning | `reviewer_studies.py` | Shared objective and fixed budget; equal full search requires executing declared candidates |
| External continuous-time or probabilistic baselines | Native references | Reruns outstanding |

## Deep ensemble comparison

`scripts/evaluate_ensemble.py` combines independent fixed-coordinate, likelihood-trained checkpoints using the exact same evaluation functions. The script checks distinct training seeds, frozen protocol settings, operator source commit and data hashes. It scores the finite predictive mixture and labels the between-member variance as ensemble disagreement, not temporal-warp sensitivity. For MuJoCo, an ensemble requires separating the training seed from the frozen data-generation seed; the script refuses this case until that experiment is explicitly configured. No ensemble or external probabilistic baseline scores are included.
