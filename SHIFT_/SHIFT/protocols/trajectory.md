# trajectory — KRNO benchmark protocol

| Setting | Reference |
|---|---|
| Source | KRNO Appendix H.4; official run_traj_KRNO.sh |
| Context | 30 steps |
| Test horizon | 30 test steps, 15 training |
| Data split | Native chronological eligible window split; full per-trajectory standardization, including test; denormalize for RMSE |
| Metric | RMSE over all 30 future steps; additionally 1-10, 11-20, 21-30 |
| Seed(s) | 2021, 2024, 2025, 2027 |
| Seed provenance | Official run_traj_KRNO.sh |

Original KNF two velocity components; train stride 2, origins 50/100/150/200/250; recursive 15-step blocks; all test windows, native trajectory-wise normalization (including full test trajectory). Manuscript 15-step test claim unresolved.

The registered SHIFT configuration is a manuscript-derived reconstruction, not a verified historical run. The local KRNO control is not the official native task model. Inspect the locked plan and resolve its blockers before executing comparisons. No data or scores are provided.

```bash
python run_paper_protocols.py --dataset trajectory --data-root data --upstream-root external/ShiftingTime --plan-only
```

References: [KRNO paper](https://proceedings.mlr.press/v267/dama25a.html), [official source](https://github.com/srinathdama/ShiftingTime), [reproduction status](../docs/REPRODUCTION_STATUS.md).
