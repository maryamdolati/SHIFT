# Dataset preparation

Only external benchmark data are used. See the [KRNO dataset instructions](https://github.com/srinathdama/ShiftingTime/tree/main/experiments/dataset) and its README. Do not include data in the Git repository.

```text
data/
├── irregular/
│   ├── mimic/{train.npz,val.npz,test.npz,manifest.json}
│   ├── ushcn/{train.npz,val.npz,test.npz,manifest.json}
│   └── activity/{train.npz,val.npz,test.npz,manifest.json}
├── irregular_missing_0.3/seed_0/mujoco/{train.npz,val.npz,test.npz,manifest.json}
├── crypto/{train.npy,test.npy,weights.npy,assets.json}
├── trajectory/{train.npy,test.npy}
├── m4/Train/{Quarterly,Weekly,Daily}-train.csv
├── m4/Test/{Quarterly,Weekly,Daily}-test.csv
└── shallow/{shallow_water.npz,manifest.json}
```

## Irregular time series

Obtain MIMIC-III, USHCN and Activity via the exact native T-PatchGNN-based preprocessing pipeline. With a clean native checkout and its real raw data installed at the original expected paths:

```bash
python scripts/export_irregular.py --dataset mimic --upstream-root external/ShiftingTime --out data/irregular
```

Repeat for `ushcn` and `activity`. The exporter derives the split using the native constructors and collate functions, preserving packed variable-length observations, timestamps, masks, target arrays, and record IDs. Verify output against the actual native run. Clinical identities remain private; never commit the manifests or data.

## MuJoCo

The native KRNO runner calls `datasets.mujoco.get_data_krno`; the external dependency is not bundled in the supplied source. Do not generate new random missingness or substitute spline-interpolated values for removed observations. Export the exact native per-seed, per-rate observations, masks, times, targets and split IDs. Manifest must include `source_repository`, `source_commit`, `source_split_hashes`, `preprocessing_description`, `protocol: stable_neural_sde_krno`, `seed`, `missing_rate`, `missingness_source`, `preprocessing_source_sha256`, and `representation: observed_values_masks_timestamps`. The runner rejects missing or mismatched provenance. A verified native exporter is outstanding until the original dependency is supplied.

## Crypto

Use official KNF `train.npy` and `test.npy` files in asset order 0–13. After independently confirming feature and asset ordering, export the official metric weights:

```bash
python scripts/prepare_crypto_weights.py --upstream-root external/ShiftingTime --out data/crypto --asset-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13
```

The evaluator requires the exact 14-weight vector and scores the final feature across all 15 horizons. It refuses a missing or wrong vector. `assets.json` declares identity/order but cannot prove content identity by itself.

## Player Trajectory

Use the exact KNF training and test arrays; do not regenerate random trajectory selection without the native seed and source identities. Verify the two velocity features and the original whole-trajectory standardization convention. Preserve all evaluation origins, not a capped subset.

## M4

Download the official competition CSVs and preserve series IDs in original order. Confirm equality with the original KNF `.npy` preprocessing for the chosen subsets. Short histories or invalid standard deviations fail explicitly rather than silently excluding test series.

## Shallow Water

Use the original LOCA train and test `.npz` archives. Export without prepartitioning the native training trajectories:

```bash
python scripts/export_shallow.py --train path/to/train_SW.npz --test path/to/test_SW.npz --source-commit VERIFIED_SOURCE_COMMIT --out data/shallow
```

Supply the real LOCA version or commit, not the placeholder. The loader constructs stride-one 5→5 training windows and a seed-42 80/20 window split. Test uses 1000 independent simulations, initial five observed frames and 55 predicted frames.
