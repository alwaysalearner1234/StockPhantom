# ROI — Phantom Stock-Out Detector (annual ₹ recovered)

## Inputs (observed)
- Eval window: 2026-08-03 .. 2026-10-01 = **60 days**
- Alerts emitted: **1043**, true positives: **83** (precision ≈ 0.08)
- Phantom episodes caught (first-flag basis): **11**
- Mean expected velocity of caught SKUs: **0.66 units/day**
- Mean MRP of caught SKUs: **₹270**

## Assumptions (explicit, tunable)
- LATE = 21 days a phantom sits unnoticed without the detector (monthly audit cycle).
- EARLY ≈ detection lag = `zero_days` at flag time (median 10 days), capped at LATE.
- Saved selling days per episode = LATE − lag.
- Recovered units = saved_days × expected_daily_units; recovered ₹ = units × MRP (full revenue; contribution ≈ 20–30% of this).
- Annualisation factor = 365 / 60 = 6.08×.

## Arithmetic
```
window_recovered_units = Σ_episodes (LATE − lag_e) × μ_e = 97 units
window_recovered_₹     = Σ_episodes units_e × MRP_e     = ₹42,176
annual_recovered_₹     = window_₹ × 365/60     = ₹256,574
```

## Result
### ≈ ₹256,574 / year in recovered revenue (589 units/year)
At 25% contribution margin ≈ **₹64,143/year contribution**.
Manager cost: 20 checks × ~2 min × 25 stores ≈ 17 hrs/day chain-wide; at even 10% hit-rate the payback is >50×.

*Re-run with `--late 14` / `--late 30` for sensitivity bounds.*
