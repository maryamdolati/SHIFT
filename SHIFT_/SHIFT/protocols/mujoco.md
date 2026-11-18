# mujoco — KRNO benchmark protocol

| Setting | Reference |
|---|---|
| Source | KRNO §3.2.1 and ShiftingTime/experiments/scripts/mujoco/run_all{1,2,3,4}.sh |
| Context | 50 steps |
| Test horizon | 10 steps |
| Data split | Identical dataset, missingness and original Stable Neural SDE preprocessing for both arms |
| Metric | MSE at 0%, 30%, 50%, and 70% dropped observations, evaluated separately |
| Seed(s) | 0, 1, 2, 3, 4 |
| Seed provenance | Official run_all1.sh through run_all4.sh loop over 0..4 |

14 dimensions, 100 Hopper trajectories of 100 points in the paper. Conditions 0/30/50/70%; use original get_data_krno, not independent masks or interpolated substitutes. Native preprocessing dependency and data are outstanding.

The registered SHIFT configuration is a manuscript-derived reconstruction, not a verified historical run. The local KRNO control is not the official native task model. Inspect the locked plan and resolve its blockers before executing comparisons. No data or scores are provided.

```bash
python run_paper_protocols.py --dataset mujoco --data-root data --upstream-root external/ShiftingTime --plan-only --missing-rate 0.3
```

References: [KRNO paper](https://proceedings.mlr.press/v267/dama25a.html), [official source](https://github.com/srinathdama/ShiftingTime), [reproduction status](../docs/REPRODUCTION_STATUS.md).
