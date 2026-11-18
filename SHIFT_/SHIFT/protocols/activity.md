# activity — KRNO benchmark protocol

| Setting | Reference |
|---|---|
| Source | KRNO Appendix H.0.1 |
| Context | 3000 ms |
| Test horizon | 1000 ms |
| Data split | 60/20/20 original T-PatchGNN-processed trajectories; preserve masks and timestamps |
| Metric | Variable-wise masked MSE and MAE |
| Seed(s) | 1, 2, 3, 4, 5 |
| Seed provenance | Official run_all1.sh, run_all2.sh, run_all3.sh |

12 variables; native record split before chunking. The SHIFT manuscript gives final fine-tuning settings; initialization and stage schedule required.

The registered SHIFT configuration is a manuscript-derived reconstruction, not a verified historical run. The local KRNO control is not the official native task model. Inspect the locked plan and resolve its blockers before executing comparisons. No data or scores are provided.

```bash
python run_paper_protocols.py --dataset activity --data-root data --upstream-root external/ShiftingTime --plan-only
```

References: [KRNO paper](https://proceedings.mlr.press/v267/dama25a.html), [official source](https://github.com/srinathdama/ShiftingTime), [reproduction status](../docs/REPRODUCTION_STATUS.md).
