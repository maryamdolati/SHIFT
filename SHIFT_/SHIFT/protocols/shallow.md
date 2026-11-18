# shallow — KRNO benchmark protocol

| Setting | Reference |
|---|---|
| Source | KRNO §3.1 and official shallow_water_krno/train_script.py |
| Context | 5 frames |
| Test horizon | 55 forecast frames (60 including context) |
| Data split | LOCA shallow water train/test archives; upstream training windows random-split 80/20 with seed 42 |
| Metric | Relative L2 for rho,u,v over recursively predicted frames only |
| Seed(s) | 42 |
| Seed provenance | Official shallow_water_krno/train_script.py sets seed=42 |

LOCA 1000 train/1000 independent test trajectories, 32×32, height/u/v, dt=.01s, stride-one training windows, seed-42 80/20 random window split. 11 recursive five-frame calls, no ground-truth feedback, initial context excluded from relative L2.

The registered SHIFT configuration is a manuscript-derived reconstruction, not a verified historical run. The local KRNO control is not the official native task model. Inspect the locked plan and resolve its blockers before executing comparisons. No data or scores are provided.

```bash
python run_paper_protocols.py --dataset shallow --data-root data --upstream-root external/ShiftingTime --plan-only
```

References: [KRNO paper](https://proceedings.mlr.press/v267/dama25a.html), [official source](https://github.com/srinathdama/ShiftingTime), [reproduction status](../docs/REPRODUCTION_STATUS.md).
