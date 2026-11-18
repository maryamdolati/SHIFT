# crypto — KRNO benchmark protocol

| Setting | Reference |
|---|---|
| Source | KRNO Appendix H.4; official run_crypto_KRNO.sh; KNF_eval_metrics.py |
| Context | 80 steps |
| Test horizon | 15 steps |
| Data split | Native chronological eligible window split, full train per-asset standardization |
| Metric | WRMSE on last feature over all 15 steps; official long uses step 11 |
| Seed(s) | 2021, 2024, 2025, 2027 |
| Seed provenance | Official run_crypto_KRNO.sh |

14 assets × 8 features; stride 100 train/validation; first test origin 100, stride 15. Native full-train per-asset standardization; chronological eligible-window split. Original squared weights, last feature, total all 15 horizons; official long index is step 11, separate from full steps 11–15.

The registered SHIFT configuration is a manuscript-derived reconstruction, not a verified historical run. The local KRNO control is not the official native task model. Inspect the locked plan and resolve its blockers before executing comparisons. No data or scores are provided.

```bash
python run_paper_protocols.py --dataset crypto --data-root data --upstream-root external/ShiftingTime --plan-only
```

References: [KRNO paper](https://proceedings.mlr.press/v267/dama25a.html), [official source](https://github.com/srinathdama/ShiftingTime), [reproduction status](../docs/REPRODUCTION_STATUS.md).
