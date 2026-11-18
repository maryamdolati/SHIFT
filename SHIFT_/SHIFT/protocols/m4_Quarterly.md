# m4/Quarterly — KRNO benchmark protocol

| Setting | Reference |
|---|---|
| Source | KRNO Appendix H.3; official run_exp.py default |
| Context | Native selected value not documented; SHIFT 64 |
| Test horizon | 8 steps |
| Data split | Native chronological window split; Quarterly 5%, others 10%; paper says 10% |
| Metric | sMAPE across full official 8-step forecast for each series |
| Seed(s) | 2021 |
| Seed provenance | Official run_exp.py default seed=2021, not a verified selected multi-seed list |

Official train/test IDs. Appendix H.3 validation 10%; executable 5%. Native default seed 2021 is not a selected multi-seed list.

The registered SHIFT configuration is a manuscript-derived reconstruction, not a verified historical run. The local KRNO control is not the official native task model. Inspect the locked plan and resolve its blockers before executing comparisons. No data or scores are provided.

```bash
python run_paper_protocols.py --dataset m4/Quarterly --data-root data --upstream-root external/ShiftingTime --plan-only
```

References: [KRNO paper](https://proceedings.mlr.press/v267/dama25a.html), [official source](https://github.com/srinathdama/ShiftingTime), [reproduction status](../docs/REPRODUCTION_STATUS.md).
