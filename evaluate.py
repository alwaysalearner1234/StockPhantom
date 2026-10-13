"""evaluate.py — precision@20 per store-day + recall on hidden labels.

Metric that matters to a manager: of the 20 SKUs we ask them to check each
morning in each store, how many were truly phantom? Averaged over store-days.

Usage:
    python evaluate.py --alerts alerts.csv --labels data/labels_hidden.parquet
    python evaluate.py --alerts alerts.csv --labels data/labels_hidden.parquet --dates 2026-08-01:2026-10-01
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="alerts.csv")
    ap.add_argument("--labels", default="data/labels_hidden.parquet")
    ap.add_argument("--dates", default=None, help="eval window START:END, default = alerts date range")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    alerts = pd.read_csv(args.alerts, dtype={"store_id": str, "sku_id": str})
    labels = pd.read_parquet(args.labels)
    alerts["date"] = pd.to_datetime(alerts["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    if args.dates:
        s, e = args.dates.split(":")
        start, end = pd.Timestamp(s), pd.Timestamp(e)
    else:
        start, end = alerts["date"].min(), alerts["date"].max()
    labels_w = labels[(labels["date"] >= start) & (labels["date"] <= end)].copy()
    alerts_w = alerts[(alerts["date"] >= start) & (alerts["date"] <= end)].copy()
    alerts_w = alerts_w.sort_values(["store_id", "date", "rank"])
    alerts_w = alerts_w.groupby(["store_id", "date"], as_index=False).head(args.topk)

    phantom = labels_w[labels_w["is_phantom"] == 1][["date", "store_id", "sku_id"]].drop_duplicates()
    phantom_set = set(map(tuple, phantom[["date", "store_id", "sku_id"]].itertuples(index=False, name=None)))

    # per store-day precision@K over ALL store-days in window (stores x dates),
    # not just days we emitted alerts — checking nothing when shelf is fine counts as 1.0
    # only if no phantom that day, else 0.0. Standard retail version averages over
    # store-days that HAVE at least one phantom (the days a manager could act).
    stores = sorted(labels_w["store_id"].unique())
    days = pd.date_range(start, end, freq="D")
    precisions_all, precisions_actionable = [], []
    n_actionable = 0
    for st in stores:
        pal = alerts_w[alerts_w["store_id"] == st].groupby("date")["sku_id"].apply(set).to_dict()
        phl = phantom[phantom["store_id"] == st].groupby("date")["sku_id"].apply(set).to_dict()
        for d in days:
            pred = pal.get(d, set())
            true = phl.get(d, set())
            if len(true) == 0:
                precisions_all.append(1.0 if len(pred) == 0 else 1.0 - len(pred - true) / args.topk * 0 + len(pred & true) / args.topk)
                # simpler: no phantom => precision defined as 1 if we flagged nothing, else penalised
                precisions_all[-1] = len(pred & true) / args.topk if len(pred) else 1.0
            else:
                n_actionable += 1
                p = len(pred & true) / min(args.topk, max(1, len(pred))) if len(pred) else 0.0
                precisions_all.append(len(pred & true) / args.topk)
                precisions_actionable.append(p)

    import numpy as np
    mean_p_all = float(np.mean(precisions_all)) if precisions_all else 0.0
    mean_p_act = float(np.mean(precisions_actionable)) if precisions_actionable else 0.0

    # recall (row-level): phantom SKU-days caught in top-K that day / all phantom SKU-days
    pred_tuples = set(map(tuple, alerts_w[["date", "store_id", "sku_id"]].itertuples(index=False, name=None)))
    # normalise Timestamp vs datetime for set compare
    hits = len(phantom_set & pred_tuples)
    recall = hits / max(1, len(phantom_set))

    # episode recall: group consecutive phantom days per (store,sku) into episodes
    ph = labels_w.sort_values(["store_id", "sku_id", "date"])
    ph = ph[ph["is_phantom"] == 1].copy()
    ep_total, ep_hit = 0, 0
    for _, g in ph.groupby(["store_id", "sku_id"]):
        dates = sorted(g["date"].unique())
        # split on gaps >1 day
        eps = [[dates[0]]]
        for d in dates[1:]:
            if (d - eps[-1][-1]).days > 1:
                eps.append([d])
            else:
                eps[-1].append(d)
        for ep in eps:
            ep_total += 1
            if any((d, g["store_id"].iloc[0], g["sku_id"].iloc[0]) in pred_tuples or
                   (pd.Timestamp(d), g["store_id"].iloc[0], g["sku_id"].iloc[0]) in pred_tuples for d in ep):
                ep_hit += 1
    ep_recall = ep_hit / max(1, ep_total)

    print("=" * 60)
    print(f"Eval window : {start.date()} .. {end.date()} ({len(days)} days x {len(stores)} stores)")
    print(f"Phantom SKU-days in window : {len(phantom_set)}  | episodes: {ep_total}")
    print(f"Alerts scored (top-{args.topk}/store-day): {len(alerts_w)}")
    print(f"precision@{args.topk} (all store-days)        : {mean_p_all:.4f}")
    print(f"precision@{args.topk} (phantom-days only)     : {mean_p_act:.4f}  <- manager metric")
    print(f"recall (phantom SKU-days caught) : {recall:.4f} ({hits}/{len(phantom_set)})")
    print(f"episode recall (>=1 hit/episode)  : {ep_recall:.4f} ({ep_hit}/{ep_total})")
    print(f"actionable store-days (had phantom): {n_actionable}")
    print("=" * 60)


if __name__ == "__main__":
    main()
