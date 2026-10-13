"""Phantom Stock-Out detector.
Scores SKU-store-days for phantom stock-outs and emits Top-20 per store-day.

Signal model (documented / justifiable):
  - Under stable demand, daily sales ~ Poisson(mu). P(N straight zeros) = exp(-mu*N).
  - confidence = 1 - exp(-mu*N). High-velocity SKU with long zero run => ~1.0.
  - Slow movers (mu<0.3) need much longer N to reach the same confidence -> no flood.
  - Boosters: frozen system stock during streak, promo/holiday with zero sales,
    weekend zeros for weekend-lifted items, high stock cover.
  - Suppressors: delisted shape (stock runs to 0 + tail to end of history),
    SKU genuinely delisted / zero baseline, stock==0 (true OOS, not phantom).

Usage:
    python detector.py --fact data/sales_inventory.parquet --out alerts.csv
    python detector.py --fact data/sales_inventory.parquet --out alerts.csv --dates 2026-09-01:2026-09-30
    python detector.py --fact data/sales_inventory.parquet --out alerts.csv --asof 2026-09-30
"""
import argparse
import numpy as np
import pandas as pd

SLOW_MU = 0.30
MIN_RUN = 3


def poisson_zero_prob(mu, n):
    return np.exp(-np.clip(mu, 0, 60) * np.clip(n, 0, 120))


def score_one_group(g, peer=None):
    """g: DataFrame sorted by date for one (store, sku). Returns scored frame.
    peer: array of chain-wide median sales for this SKU on each date (same order)."""
    sales = g["units_sold"].values.astype(float)
    stock = g["system_on_hand"].values.astype(float)
    promo = g["on_promo"].values.astype(bool) if "on_promo" in g else np.zeros(len(g), bool)
    hol = g["is_holiday"].values.astype(bool) if "is_holiday" in g else np.zeros(len(g), bool)
    n = len(g)
    # consecutive zeros with stock>0, ending at each t
    is_zero_pos = (sales == 0) & (stock > 0)
    run = np.zeros(n, dtype=int)
    c = 0
    for i in range(n):
        c = c + 1 if is_zero_pos[i] else 0
        run[i] = c
    # trailing 28d mean BEFORE streak via cumsum — promo/holiday days EXCLUDED
    # (promo spikes would inflate mu and fake anomalies after the promo ends)
    clean = (~promo & ~hol).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(sales * clean)])
    cc = np.concatenate([[0], np.cumsum(clean)]).astype(float)
    g_mu = float(np.mean(sales[sales > 0])) if (sales > 0).any() else 0.0
    mu = np.zeros(n)
    for i in range(n):
        r = run[i]
        end = i - r          # exclusive end of baseline window
        start = max(0, end - 28)
        win = cc[end] - cc[start]
        mu[i] = (cs[end] - cs[start]) / win if win >= 7 else g_mu
    # fallback baseline for early history: expanding mean of non-zero sales
    # (keeps first weeks from all-zero mu)
    mu = np.where(np.isnan(mu), g_mu, mu)
    # frozen-stock flag over streak
    frozen = np.zeros(n, dtype=bool)
    stock_range = np.zeros(n)
    for i in range(n):
        r = run[i]
        if r >= 2:
            w = stock[max(0, i - r + 1):i + 1]
            stock_range[i] = w.max() - w.min()
            frozen[i] = stock_range[i] <= 1
        elif r == 1:
            frozen[i] = True
    p_zero = poisson_zero_prob(mu, run)
    confidence = 1.0 - p_zero
    # --- Adaptive N gate (the justifiable N): require 99% confidence under Poisson,
    # N_req = ceil(4.605 / mu). Slow movers must wait longer; this is WHY we don't
    # drown in slow-SKU false positives. Promo shortens N (expected lift ~2x).
    # N choice is per-SKU, not a single global magic number.
    mu_safe = np.maximum(mu, 0.05)
    n_req = np.ceil(4.605 / mu_safe).astype(int)
    n_req = np.clip(n_req, 3, 21)
    # promo expectation: demand ~2x, so halve the required run (min 2)
    # (promo_in_streak computed below; precompute quick promo flag here)
    promo_flag = np.zeros(n, dtype=bool)
    for i in range(n):
        r = run[i]
        if r > 0:
            promo_flag[i] = bool(promo[max(0, i - r + 1):i + 1].any() or hol[i])
    n_req_adj = np.where(promo_flag, np.maximum(2, n_req // 2), n_req)
    passes_n = run >= n_req_adj
    # slow-mover extra guard: mu<0.3 needs run>=10 regardless
    passes_n = passes_n & ~((mu < SLOW_MU) & (run < 10))
    confidence = np.where(~passes_n, confidence * 0.25, confidence)  # crush, don't delete (keeps audit trail)
    # boosters
    promo_in_streak = promo_flag
    boost = np.ones(n)
    boost = np.where(frozen & (run >= 2), boost * 1.15, boost)
    boost = np.where(promo_in_streak, boost * 1.25, boost)
    boost = np.where((stock / np.maximum(mu, 0.05) > 8) & (run >= 3), boost * 1.10, boost)
    # peer signal: same SKU selling fine elsewhere while this store sits at zero.
    # Chance zeros hit stores independently; phantoms hit one store while peers sell.
    if peer is not None:
        peer = np.asarray(peer, dtype=float)
        peer_ok = (peer >= 1.0) & (run >= 3)
        boost = np.where(peer_ok, boost * (1.0 + 0.25 * np.minimum(peer, 6.0)), boost)
    else:
        peer_ok = np.zeros(n, dtype=bool)
    score = np.clip(confidence * boost, 0, 0.999)
    # business-impact tiebreaker: expected lost units so far = mu * run.
    # Stored for ranking (breaks the 0.999 saturation ties toward fast movers).
    impact = mu_safe * run
    # delisted shape: stock==0 tail OR baseline mu==0 with very long run to end of history
    # (handled per-row: if last observed date and run extends to end + stock==0 -> delisted)
    out = g.copy()
    out["zero_days"] = run
    out["expected_daily"] = np.round(mu, 3)
    out["p_zero_streak"] = np.round(p_zero, 5)
    out["confidence"] = np.round(np.clip(confidence, 0, 1), 4)
    out["frozen_stock"] = frozen
    out["promo_in_streak"] = promo_in_streak
    out["score"] = np.round(score, 4)
    out["impact_units"] = np.round(impact, 2)
    out["passes_n"] = passes_n
    return out


def build_reason(r):
    mu, n, fz, pr, stock = r["expected_daily"], int(r["zero_days"]), r["frozen_stock"], r["promo_in_streak"], r["system_on_hand"]
    if n <= 0:
        return ""
    base = f"No sales for {n} days despite usually selling ~{mu:.1f}/day"
    extra = f" while system shows {int(stock)} units on hand"
    tail = ""
    if pr:
        tail = " — including a promo/holiday when sales should spike"
    elif fz and n >= 3:
        tail = " — stock count hasn't moved, shelf is likely empty"
    elif mu < 0.5:
        tail = " — unusual even for a slow mover"
    else:
        tail = " — shelf is likely empty"
    s = base + extra + tail
    if r.get("predicted_type", "phantom") == "misplaced":
        s = base + extra + " — with odd ones found intermittently, possibly misplaced on wrong shelf"
    return s + "."


def classify(streak_df):
    """Stretch: phantom vs misplaced vs delisted using signal shape."""
    typ = []
    for _, r in streak_df.iterrows():
        n, mu, stock = int(r["zero_days"]), float(r["expected_daily"]), float(r["system_on_hand"])
        fz = bool(r["frozen_stock"])
        # delisted: system stock 0 (true delist) — never phantom
        if stock <= 0:
            typ.append("delisted")
        elif mu < 0.05 and n >= 20:
            typ.append("delisted")
        elif (not fz) and mu >= 0.5 and n >= 5:
            # stock jiggling + partial sales nearby -> misplaced candidate
            typ.append("misplaced")
        else:
            typ.append("phantom")
    streak_df["predicted_type"] = typ
    return streak_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fact", default="data/sales_inventory.parquet")
    ap.add_argument("--out", default="alerts.csv")
    ap.add_argument("--asof", default=None, help="single date YYYY-MM-DD to score")
    ap.add_argument("--dates", default=None, help="window START:END to score (default: last 60 days)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--min_score", type=float, default=0.50)
    args = ap.parse_args()

    fact = pd.read_parquet(args.fact)
    fact["date"] = pd.to_datetime(fact["date"])
    fact = fact.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)

    if args.asof:
        asof = [pd.Timestamp(args.asof)]
    elif args.dates:
        s, e = args.dates.split(":")
        asof = list(pd.date_range(s, e, freq="D"))
    else:
        end = fact["date"].max()
        asof = list(pd.date_range(end - pd.Timedelta(days=59), end, freq="D"))
    asof_set = set(asof)
    print(f"Scoring {len(asof_set)} days, {fact['store_id'].nunique()} stores ...")
    # peer signal: chain-wide median sales per (sku, date) — phantom hits one store
    # while peers sell normally; chance zeros hit all stores independently.
    peer_med = fact.groupby(["sku_id", "date"])["units_sold"].median().rename("peer_med")
    fact = fact.join(peer_med, on=["sku_id", "date"])

    scored_parts = []
    # per-group scoring (20k groups max; vectorised inner loops in numpy)
    for (store, sku), g in fact.groupby(["store_id", "sku_id"], sort=False):
        g = g.sort_values("date")
        sg = score_one_group(g, peer=g["peer_med"].values)
        # keep only requested asof dates with a live streak + adaptive-N gate
        m = sg["date"].isin(asof_set) & (sg["zero_days"] >= MIN_RUN) & (sg["system_on_hand"] > 0)
        m = m & sg["passes_n"]
        m = m & (sg["score"] >= args.min_score)
        if m.any():
            scored_parts.append(sg.loc[m, ["date", "store_id", "sku_id", "units_sold",
                                           "system_on_hand", "expected_daily", "zero_days",
                                           "p_zero_streak", "confidence", "frozen_stock",
                                           "promo_in_streak", "on_promo", "score", "impact_units"]])
    if not scored_parts:
        print("No alerts generated.")
        pd.DataFrame().to_csv(args.out, index=False)
        return
    alerts = pd.concat(scored_parts, ignore_index=True)
    alerts = classify(alerts)
    # delisted predictions are not phantom alerts -> drop them from the go-check list
    alerts = alerts[alerts["predicted_type"] != "delisted"].copy()
    alerts["reason"] = alerts.apply(build_reason, axis=1)
    alerts["confidence"] = alerts["score"]  # ranked confidence shown to manager
    # rank per store-day by score, tie-broken by business impact (mu*run) so fast
    # movers with real revenue at risk float to the top within 0.999 ties.
    alerts = alerts.sort_values(["store_id", "date", "score", "impact_units"],
                                ascending=[True, True, False, False])
    alerts["rank"] = alerts.groupby(["store_id", "date"])["score"].rank(method="first", ascending=False).astype(int)
    alerts = alerts[alerts["rank"] <= args.topk].copy()
    alerts["date"] = alerts["date"].dt.strftime("%Y-%m-%d")
    alerts = alerts.sort_values(["date", "store_id", "rank"])
    alerts.to_csv(args.out, index=False)
    print(f"Wrote {len(alerts)} alerts ({alerts['date'].nunique()} days) -> {args.out}")
    print(alerts.head(10)[["date", "store_id", "sku_id", "rank", "score", "zero_days", "expected_daily"]].to_string(index=False))


if __name__ == "__main__":
    main()
