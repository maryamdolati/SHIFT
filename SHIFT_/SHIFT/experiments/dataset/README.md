# Dataset preparation

See the [KRNO dataset instructions](https://github.com/srinathdama/ShiftingTime/tree/main/experiments/dataset) and its README. Do not include data in the Git repository.



## Irregular time series

Obtain MIMIC-III, USHCN and Activity via the exact native T-PatchGNN-based preprocessing pipeline. With a clean native checkout and its real raw data installed at the original expected paths:

```bash
python scripts/export_irregular.py --dataset mimic --upstream-root external/ShiftingTime --out data/irregular
```

Repeat for `ushcn` and `activity`. The exporter derives the split using the native constructors and collate functions, preserving packed variable-length observations, timestamps, masks, target arrays, and record IDs. Verify output against the actual native run. Clinical identities remain private; never commit the manifests or data.

