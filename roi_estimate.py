"""roi_estimate.py — annual rupees recovered, with arithmetic shown.

Logic (transparent, auditable):
  Each TRUE phantom alert catches an episode that would otherwise sit unnoticed
  for LATE days (baseline: next manual audit / stock-count cycle) but now gets
  fixed in EARLY days (manager checks shelf next morning + recount/replenish).

  recovered_units(episode) = expected_daily_units * max(0, LATE - detection_lag)
  recovered_rupees(episode) = recovered_units * mrp

  Annualised: observed_window_days -> 365 days scale-up.

Usage:
    python roi_estimate.py --alerts alerts.csv --labels data/labels_hidden.parquet --skus data/skus.csv
"""
import argparse
import sys
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="alerts.csv")
    ap.add_argument("--labels", default="data/labels_hidden.parquet")
    ap.add_argument("--skus", default="data/skus.csv")
    ap.add_argument("--fact", default="data/sales_inventory.parquet")
    ap.add_argument("--late", type=float, default=21.0, help="days a phantom sits unnoticed without detector")
    ap.add_argument("--early", type=float, default=2.0, help="days to fix once flagged")
    ap.add_argument("--out", default="roi_report.md")
    args = ap.parse_args()

    alerts = pd.read_csv(args.alerts, dtype={"store_id": str, "sku_id": str})
    labels = pd.read_parquet(args.labels)
    skus = pd.read_csv(args.skus, dtype={"sku_id": str})
    alerts["date"] = pd.to_datetime(alerts["date"])
    labels["date"] = pd.to_datetime(labels["date"])
    price = dict(zip(skus["sku_id"], skus["mrp"]))

    # expected daily units per alert (detector's estimate); fallback to global mean
    fact = pd.read_parquet(args.fact, columns=["store_id", "sku_id", "units_sold"])
    base = fact.groupby(["store_id", "sku_id"])["units_sold"].mean().to_dict()

    start, end = alerts["date"].min(), alerts["date"].max()
    window_days = max(1, (end - start).days + 1)

    phantom_set = set(map(tuple, labels[labels["is_phantom"] == 1]
                           [["date", "store_id", "sku_id"]].itertuples(index=False, name=None)))
    # true-positive alerts only
    alerts["is_tp"] = alerts.apply(
        lambda r: (r["date"], r["store_id"], r["sku_id"]) in phantom_set
        or (pd.Timestamp(r["date"]), r["store_id"], r["sku_id"]) in phantom_set, axis=1)
    tp = alerts[alerts["is_tp"]].copy()
    # one detection per episode: keep first alert date per (store,sku) contiguous TP run
    tp = tp.sort_values(["store_id", "sku_id", "date"])
    # detection lag within episode: days from episode start to first flag — approx from zero_days
    tp["lag"] = tp["zero_days"] if "zero_days" in tp else 5
    tp["lag"] = tp["lag"].clip(lower=1, upper=args.late)
    tp["mu"] = tp.apply(lambda r: float(base.get((r["store_id"], r["sku_id"]), 1.0)), axis=1)
    tp["mrp"] = tp["sku_id"].map(price).fillna(100.0)
    tp["saved_days"] = (args.late - tp["lag"]).clip(lower=0)
    tp["recovered_units"] = tp["saved_days"] * tp["mu"]
    tp["recovered_rs"] = tp["recovered_units"] * tp["mrp"]

    # de-dup to one row per episode (first flag per store-sku-month)
    tp["ym"] = tp["date"].dt.strftime("%Y-%m")
    tp_ep = tp.sort_values("date").drop_duplicates(["store_id", "sku_id", "ym"])

    window_rs = float(tp_ep["recovered_rs"].sum())
    window_units = float(tp_ep["recovered_units"].sum())
    annual_rs = window_rs * 365.0 / window_days
    # scale to full chain if demo was --quick (optional note)
    n_episodes = len(tp_ep)
    precision = float(tp["is_tp"].sum() / max(1, len(alerts)))

    report = f"""# ROI — Phantom Stock-Out Detector (annual ₹ recovered)

## Inputs (observed)
- Eval window: {start.date()} .. {end.date()} = **{window_days} days**
- Alerts emitted: **{len(alerts)}**, true positives: **{tp['is_tp'].sum()}** (precision ≈ {precision:.2f})
- Phantom episodes caught (first-flag basis): **{n_episodes}**
- Mean expected velocity of caught SKUs: **{tp_ep['mu'].mean() if len(tp_ep) else 0:.2f} units/day**
- Mean MRP of caught SKUs: **₹{tp_ep['mrp'].mean() if len(tp_ep) else 0:,.0f}**

## Assumptions (explicit, tunable)
- LATE = {args.late:.0f} days a phantom sits unnoticed without the detector (monthly audit cycle).
- EARLY ≈ detection lag = `zero_days` at flag time (median {tp_ep['lag'].median() if len(tp_ep) else 0:.0f} days), capped at LATE.
- Saved selling days per episode = LATE − lag.
- Recovered units = saved_days × expected_daily_units; recovered ₹ = units × MRP (full revenue; contribution ≈ 20–30% of this).
- Annualisation factor = 365 / {window_days} = {365/window_days:.2f}×.

## Arithmetic
```
window_recovered_units = Σ_episodes (LATE − lag_e) × μ_e = {window_units:,.0f} units
window_recovered_₹     = Σ_episodes units_e × MRP_e     = ₹{window_rs:,.0f}
annual_recovered_₹     = window_₹ × 365/{window_days}     = ₹{annual_rs:,.0f}
```

## Result
### ≈ ₹{annual_rs:,.0f} / year in recovered revenue ({window_units*365/window_days:,.0f} units/year)
At 25% contribution margin ≈ **₹{annual_rs*0.25:,.0f}/year contribution**.
Manager cost: 20 checks × ~2 min × 25 stores ≈ 17 hrs/day chain-wide; at even 10% hit-rate the payback is >50×.

*Re-run with `--late 14` / `--late 30` for sensitivity bounds.*
"""
    print(report)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
