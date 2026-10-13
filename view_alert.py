"""view_alert.py — sales-vs-stock trust chart for any flagged SKU.
Usage:
    python view_alert.py --store S001 --sku SKU0042 --date 2026-09-30 --out chart.png
Shows 60 days ending at --date: bars = units sold, line = system on-hand,
shaded = zero-sales streak, promo markers.
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--sku", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--fact", default="data/sales_inventory.parquet")
    ap.add_argument("--alerts", default="alerts.csv")
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--out", default="chart.png")
    args = ap.parse_args()

    fact = pd.read_parquet(args.fact)
    fact["date"] = pd.to_datetime(fact["date"])
    asof = pd.Timestamp(args.date)
    g = fact[(fact["store_id"] == args.store) & (fact["sku_id"] == args.sku)]
    g = g[(g["date"] > asof - pd.Timedelta(days=args.window)) & (g["date"] <= asof)].sort_values("date")
    if g.empty:
        print("No data for that store/SKU/date.")
        return
    try:
        alerts = pd.read_csv(args.alerts, dtype=str)
        m = alerts[(alerts["store_id"] == args.store) & (alerts["sku_id"] == args.sku) & (alerts["date"] == args.date)]
        title_extra = f" | score={float(m.iloc[0]['score']):.2f}, {m.iloc[0]['reason'][:90]}" if len(m) else ""
    except Exception:
        title_extra = ""

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.bar(g["date"], g["units_sold"], color="#2563eb", alpha=0.8, label="Units sold")
    if "on_promo" in g:
        pp = g[g["on_promo"].astype(str) == "True"]
        ax1.scatter(pp["date"], pp["units_sold"] + 0.3, marker="*", s=90, color="#f59e0b", label="Promo", zorder=5)
    ax1.set_ylabel("Units sold")
    ax2 = ax1.twinx()
    ax2.plot(g["date"], g["system_on_hand"], color="#dc2626", linewidth=2, label="System on-hand")
    ax2.set_ylabel("System on-hand")
    # shade trailing zero streak
    streak = 0
    for _, r in g.iloc[::-1].iterrows():
        if r["units_sold"] == 0 and r["system_on_hand"] > 0:
            streak += 1
        else:
            break
    if streak >= 2:
        ax1.axvspan(g["date"].iloc[-streak], g["date"].iloc[-1] + pd.Timedelta(days=1),
                    color="red", alpha=0.12, label=f"{streak}-day zero streak")
    ax1.set_title(f"{args.store} × {args.sku} — sales vs system stock to {args.date}{title_extra}", fontsize=10)
    ax1.tick_params(axis="x", rotation=30)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Saved -> {args.out} ({streak}-day zero streak)")


if __name__ == "__main__":
    main()
