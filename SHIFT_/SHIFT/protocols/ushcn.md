# ushcn — KRNO benchmark protocol

| Setting | Reference |
|---|---|
| Source | KRNO Appendix H.0.1 |
| Context | 24 months |
| Test horizon | Native 1 month; paper 24 months |
| Data split | Original record split before native 48-month chunking |
| Metric | Variable-wise masked MSE and MAE |
| Seed(s) | 1, 2, 3, 4, 5 |
| Seed provenance | Official run_all1.sh, run_all2.sh, run_all3.sh |

5 weather variables; native 48-month chunking after original record split. Paper/code discrepancy must be resolved in the manuscript.

The registered SHIFT configuration is a manuscript-derived reconstruction, not a verified historical run. The local KRNO control is not the official native task model. Inspect the locked plan and resolve its blockers before executing comparisons. No data or scores are provided.

```bash
python run_paper_protocols.py --dataset ushcn --data-root data --upstream-root external/ShiftingTime --plan-only
```

References: [KRNO paper](https://proceedings.mlr.press/v267/dama25a.html), [official source](https://github.com/srinathdama/ShiftingTime), [reproduction status](../docs/REPRODUCTION_STATUS.md).
