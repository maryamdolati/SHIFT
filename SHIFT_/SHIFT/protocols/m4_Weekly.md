# m4/Weekly — KRNO benchmark protocol

| Setting | Reference |
|---|---|
| Source | KRNO Appendix H.3; official run_m4_KRNO.sh |
| Context | Native 60; SHIFT 104 |
| Test horizon | 13 steps |
| Data split | Native chronological window split; Quarterly 5%, others 10%; paper says 10% |
| Metric | sMAPE for official 13-step forecast |
| Seed(s) | 2021, 2023, 2025, 2030 |
| Seed provenance | Official run_m4_KRNO.sh annotated result seeds |

Native train stride 3; chronological eligible-window validation 10%; original per-series train standardization.

The registered SHIFT configuration is a manuscript-derived reconstruction, not a verified historical run. The local KRNO control is not the official native task model. Inspect the locked plan and resolve its blockers before executing comparisons. No data or scores are provided.

```bash
python run_paper_protocols.py --dataset m4/Weekly --data-root data --upstream-root external/ShiftingTime --plan-only
```

References: [KRNO paper](https://proceedings.mlr.press/v267/dama25a.html), [official source](https://github.com/srinathdama/ShiftingTime), [reproduction status](../docs/REPRODUCTION_STATUS.md).
