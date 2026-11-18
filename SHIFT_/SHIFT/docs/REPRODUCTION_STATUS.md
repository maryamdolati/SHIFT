# Reproduction status and source discrepancies

| Item | Native reference | Reconstructed package status |
|---|---|---|
| Irregular record split | Seed 42; split original records before chunking | Native collate exporter and ragged loader; actual data not provided |
| MIMIC/USHCN/Activity seeds | 1–5 in official run_all scripts | Locked |
| USHCN horizon | Paper 24 months; native code `pred_window=1` | Export native 1-month labels; historical SHIFT claim unresolved |
| Irregular metrics | Equal weights across observed variables | Macro MSE/MAE implemented for the three T-PatchGNN tasks |
| MuJoCo | Original `get_data_krno` in native runner | Missing external dataset implementation; no interpolation or independent random masks substituted; seed/rate and provenance required |
| Crypto | Last of 8 features, 14 asset weights, total all 15 horizons | Direct upstream function parity test passed on fixtures; source weight exporter supplied |
| Crypto long block | Native code uses step 11, paper describes 11–15 | Both reported separately |
| Crypto origins | Start 100, stride 15; train stride 100 | Native loader parity test passed on fixtures |
| Player Trajectory | Train 30→15; test 30→30 across origins 50,100,150,200,250; full per-trajectory normalization | Loader parity test passed; all test windows scored; manuscript's 15-step test assertion must be checked against original runs |
| Player target | KNF data generation uses two velocity components | Do not silently relabel as positions without checking actual conversion |
| Player normalization | Native test trajectory mean/std use entire test trajectory | Reproduced explicitly for native comparison; a leakage-safe variant needs a separately declared causal preprocessing regime and reruns of all methods |
| M4 | Official competition data and complete horizons 8/13/14 | Loader alignment tested on fixture; actual dataset identity still requires verification |
| M4 Quarterly validation | Paper 10%; `run_exp.py` 5% | Native 5% default, explicit 10% alternative; never combine numbers |
| M4 Quarterly seeds | Default 2021; selected multi-seed configuration absent | Native selected rerun blocked; no invented optimum |
| M4 short series | Native slicing may return short history | Fail rather than silently exclude or pad; need explicitly verified native handling before testing long SHIFT lookbacks |
| Shallow Water | LOCA 1000/1000, window stride 1, seed 42, 5→5 training, 11 block rollout | Window generation and relative L2 corrected; full data/runtime not supplied |
| Native baseline model | Task-specific original CNN, KRNO, RevIN, scheduler and quadrature | Native runner separate from local controls |
| SHIFT architecture | Manuscript-specific temporal and spatial models | Reconstructed core and spatial depth/width; exact original source and unpublished settings needed |
| SHIFT uncertainty | Some manuscript objectives do not train predictive variance | Probabilistic evaluation blocked for untrained heads; shared mixture-likelihood study provided separately |
| Activity | Reported settings are final fine-tuning stage | Original initialization/checkpoint and stage schedule not supplied |
| All benchmark results | Need historical logs or independent real-data reruns | Not reproduced or asserted |

## Audit scope

Native comparison tests import the source definitions of M4, Crypto and Player Trajectory loaders and the native WRMSE evaluator, then compare synthetic inputs. They establish function-level parity for those fixtures, not dataset identity. Other tests check monotonicity, losses, masks, rollout, donor controls, and locked settings. Native operator end-to-end execution and every real benchmark remain unverified.

## Required before claiming exact reproduction

1. Pin and install the original code and external dependencies, including the precise native MuJoCo `datasets.mujoco.get_data_krno` implementation.
2. Re-export each dataset and compare source hashes, record identities, normalization, forecast origins and masks with the exact native runner.
3. Supply original SHIFT settings marked missing in the launcher, including the original objective details, initial checkpoints where applicable, and all training schedules.
4. Correct manuscript statements to match the actual selected protocols; never relabel numbers from different horizons or scales.
5. Train and evaluate each seed with a frozen configuration; retain checkpoints, data/source hashes, and validation-selection records.
6. Run reviewer controls independently and report favorable, null and unfavorable cases without post-hoc seed or task selection.
