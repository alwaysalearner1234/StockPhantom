"""Synthetic data generator for Phantom Stock-Out Detector.
Generates ~800 SKUs x 25 stores x 18 months of daily sales + system inventory.

Usage:
    python generate_data.py --quick            # small demo: 120 SKUs, 8 stores, 270 days
    python generate_data.py                    # full: 800 SKUs, 25 stores, 540 days
    python generate_data.py --n_skus 800 --n_stores 25 --days 540 --seed 42

Outputs (data/):
    stores.csv, skus.csv, promotions.csv, sales_inventory.csv, labels_hidden.csv
"""
import argparse
import os
import numpy as np
import pandas as pd

CATEGORIES = {
    "Staples": 0.22, "Snacks": 0.18, "Beverages": 0.15,
    "Personal Care": 0.15, "Household": 0.12, "Dairy": 0.10, "Baby Care": 0.08,
}

FORMATS = ["Hypermarket", "Supermarket", "Express"]
CITY_TIERS = [1, 2, 3]


def gen_masters(n_skus, n_stores, rng):
    # --- stores ---
    stores = []
    for s in range(n_stores):
        fmt = rng.choice(FORMATS, p=[0.25, 0.5, 0.25])
        tier = int(rng.choice(CITY_TIERS, p=[0.3, 0.4, 0.3]))
        size = {"Hypermarket": rng.integers(20000, 60000),
                "Supermarket": rng.integers(5000, 15000),
                "Express": rng.integers(800, 2500)}[fmt]
        fmt_mult = {"Hypermarket": 1.6, "Supermarket": 1.0, "Express": 0.55}[fmt]
        tier_mult = {1: 1.3, 2: 1.0, 3: 0.75}[tier]
        stores.append((f"S{s:03d}", fmt, tier, size, fmt_mult * tier_mult))
    stores_df = pd.DataFrame(stores, columns=["store_id", "format", "city_tier", "size_sqft", "demand_mult"])
    # --- skus ---
    cats = list(CATEGORIES.keys())
    cat_p = np.array(list(CATEGORIES.values()))
    cat_p /= cat_p.sum()
    skus = []
    for i in range(n_skus):
        cat = rng.choice(cats, p=cat_p)
        # velocity mix: 15% fast, 35% medium, 50% slow
        u = rng.random()
        if u < 0.15:
            base = rng.uniform(4.0, 18.0); tier_v = "fast"
        elif u < 0.50:
            base = rng.uniform(0.8, 4.0); tier_v = "medium"
        else:
            base = rng.uniform(0.08, 0.8); tier_v = "slow"
        mrp = float(np.clip(rng.lognormal(np.log(120), 0.9), 9, 1999).round(0))
        # round mrp to 9-ending
        mrp = float(int(mrp // 10 * 10 + 9))
        skus.append((f"SKU{i:04d}", cat, tier_v, round(float(base), 3), mrp))
    skus_df = pd.DataFrame(skus, columns=["sku_id", "category", "velocity_tier", "base_rate", "mrp"])
    return stores_df, skus_df


def build_calendar(days, end_date="2026-10-01"):
    dates = pd.date_range(end=end_date, periods=days, freq="D")
    cal = pd.DataFrame({"date": dates})
    cal["dow"] = cal["date"].dt.dayofweek  # 0=Mon
    # weekend lift Sat/Sun
    cal["weekday_mult"] = np.where(cal["dow"] >= 5, 1.35, 1.0)
    cal["weekday_mult"] = np.where(cal["dow"] == 0, 0.9, cal["weekday_mult"])  # Monday dip
    # annual seasonality: sinusoid + Diwali spike (late Oct/early Nov) + summer beverages
    doy = cal["date"].dt.dayofyear.values
    cal["season_mult"] = 1.0 + 0.12 * np.sin(2 * np.pi * (doy - 80) / 365.0)
    # Diwali window: Oct 20 - Nov 10 each year
    md = cal["date"].dt.strftime("%m-%d")
    diwali = md.between("10-20", "11-10")
    cal.loc[diwali, "season_mult"] *= 1.45
    cal.loc[md.between("12-20", "12-31"), "season_mult"] *= 1.25
    # holidays flag
    cal["is_holiday"] = False
    cal.loc[diwali, "is_holiday"] = True
    cal.loc[md.isin(["01-01", "08-15", "12-25", "03-08", "05-01"]), "is_holiday"] = True
    return cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_skus", type=int, default=800)
    ap.add_argument("--n_stores", type=int, default=25)
    ap.add_argument("--days", type=int, default=540)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="small demo dataset")
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--phantom_combo_rate", type=float, default=0.03,
                    help="fraction of SKU-store combos getting a phantom episode")
    args = ap.parse_args()
    if args.quick:
        args.n_skus, args.n_stores, args.days = 120, 8, 270

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    stores_df, skus_df = gen_masters(args.n_skus, args.n_stores, rng)
    cal = build_calendar(args.days)
    dates = cal["date"].dt.strftime("%Y-%m-%d").values
    n_days = len(dates)
    weekday_mult = cal["weekday_mult"].values
    season_mult = cal["season_mult"].values
    is_holiday = cal["is_holiday"].values.astype(bool)

    # ---- promotion calendar: ~6% of SKU-store-days on promo, in 7-day flights ----
    # To keep it tractable, promos are per SKU-week nationally + store participation.
    n_promo_flights = max(10, args.n_skus // 8)
    promo_days = np.zeros((args.n_stores * args.n_skus, n_days), dtype=bool)
    promo_records = []
    combos = [(s, k) for s in range(args.n_stores) for k in range(args.n_skus)]
    # choose random flights
    for _ in range(n_promo_flights):
        sku = rng.integers(0, args.n_skus)
        start = rng.integers(0, n_days - 8)
        part_stores = rng.choice(args.n_stores, size=rng.integers(2, args.n_stores + 1), replace=False)
        disc = float(rng.choice([0.10, 0.15, 0.20, 0.25, 0.30]))
        for s in part_stores:
            idx = s * args.n_skus + sku
            promo_days[idx, start:start + 7] = True
        promo_records.append((dates[start], f"SKU{sku:04d}", f"{len(part_stores)} stores", disc))
    promo_df = pd.DataFrame(promo_records, columns=["start_date", "sku_id", "store_scope", "discount_pct"])
    promo_df.to_csv(os.path.join(args.out, "promotions.csv"), index=False)

    store_mult = stores_df["demand_mult"].values  # (n_stores,)
    base_rate = skus_df["base_rate"].values       # (n_skus,)
    mrp = skus_df["mrp"].values

    # Expected demand surface per combo-day: outer product + time mults + promo lift + noise
    # Build in chunks per store to bound memory: (n_skus, n_days) per store.
    print(f"Generating {args.n_stores} stores x {args.n_skus} SKUs x {n_days} days "
          f"= {args.n_stores*args.n_skus*n_days/1e6:.2f}M rows ...")

    # ---- choose anomaly combos ----
    n_combos = args.n_stores * args.n_skus
    n_phantom_combos = int(n_combos * args.phantom_combo_rate)
    phantom_combo_idx = set(rng.choice(n_combos, size=n_phantom_combos, replace=False).tolist())
    # extra combos for misplaced (1.5%) and delisted (1%)
    remaining = list(set(range(n_combos)) - phantom_combo_idx)
    n_mis = int(n_combos * 0.015)
    n_del = int(n_combos * 0.010)
    mis_combo_idx = set(rng.choice(remaining, size=min(n_mis, len(remaining)), replace=False).tolist())
    remaining2 = list(set(remaining) - mis_combo_idx)
    del_combo_idx = set(rng.choice(remaining2, size=min(n_del, len(remaining2)), replace=False).tolist())

    fact_chunks = []
    label_chunks = []
    promo_lift = 2.2  # avg promo multiplier (category-varying noise added later)

    for s in range(args.n_stores):
        sm = store_mult[s]
        # (n_skus, n_days) expected rate
        lam = base_rate[:, None] * sm  # (n_skus,1)
        time_mult = (weekday_mult * season_mult)[None, :]  # (1, n_days)
        lam = lam * time_mult
        # promo lift
        pl = promo_days[s * args.n_skus:(s + 1) * args.n_skus, :]
        lam = lam * np.where(pl, rng.uniform(1.8, 3.0, size=(args.n_skus, 1)), 1.0)
        # small per-combo persistent random effect
        combo_effect = rng.lognormal(0, 0.25, size=(args.n_skus, 1))
        lam = lam * combo_effect
        sales = rng.poisson(np.clip(lam, 0, 60)).astype(np.int16)

        # ---- system inventory: sawtooth replenishment, frozen during phantom ----
        # target cover ~ 10-20 days of base demand, min 4 units
        target = np.clip((base_rate * sm * rng.uniform(10, 20, size=args.n_skus))[:, None]
                         + rng.integers(2, 6, size=(args.n_skus, 1)), 4, 120).astype(float)
        inv = np.zeros_like(sales, dtype=np.int16)
        # vectorised sawtooth: walk forward in blocks of 30 days with python loop over time (540/30=18 iters)
        cur = (target[:, 0] * rng.uniform(0.6, 1.0, size=args.n_skus)).astype(float)
        for t in range(n_days):
            # replenish every 7-14d per sku: use periodic mask
            # cheap approximation: if cur < 0.35*target -> refill to target with noise
            need = cur < (0.35 * target[:, 0])
            # stagger: only refill on pseudo-random weekday per sku to avoid sync
            cur = cur - sales[:, t]
            cur[need] = target[need, 0] * rng.uniform(0.9, 1.1, size=need.sum())
            cur = np.clip(cur, 0, 200)
            inv[:, t] = cur.astype(np.int16)
        # ensure normal-days inventory positive most of the time for fast movers
        # ---- inject anomalies ----
        labels = np.zeros((args.n_skus, n_days), dtype=np.int8)
        types = np.full((args.n_skus, n_days), "", dtype=object)
        for k in range(args.n_skus):
            gidx = s * args.n_skus + k
            if gidx in phantom_combo_idx:
                n_ep = 1 if rng.random() < 0.7 else 2
                for _ in range(n_ep):
                    ep_len = int(rng.integers(7, 29))
                    # bias episodes to last 180 days so eval window is rich
                    start = int(rng.integers(max(0, n_days - 200), n_days - ep_len))
                    frozen = int(max(3, inv[k, max(0, start - 1)])) if start > 0 else int(rng.integers(4, 10))
                    frozen = int(np.clip(frozen, 3, 15))
                    sales[k, start:start + ep_len] = 0
                    inv[k, start:start + ep_len] = frozen
                    labels[k, start:start + ep_len] = 1
                    types[k, start:start + ep_len] = "phantom"
            elif gidx in mis_combo_idx:
                ep_len = int(rng.integers(7, 21))
                start = int(rng.integers(max(0, n_days - 200), n_days - ep_len))
                # misplaced: sales drop ~80% but not exactly zero (occasional unit found)
                mask = rng.random(ep_len) < 0.85
                sales[k, start:start + ep_len][mask] = 0
                labels[k, start:start + ep_len] = 0  # NOT phantom
                types[k, start:start + ep_len] = "misplaced"
            elif gidx in del_combo_idx:
                # delisted: permanent zero tail + system stock runs to 0
                start = int(rng.integers(max(0, n_days - 120), n_days - 30))
                sales[k, start:] = 0
                inv[k, start:] = 0
                types[k, start:] = "delisted"

        store_ids = np.repeat(stores_df["store_id"].values[s], args.n_skus * n_days)
        sku_ids = np.tile(np.repeat(skus_df["sku_id"].values, n_days), 1)
        # build per-store frame (n_skus*n_days rows)
        df = pd.DataFrame({
            "date": np.tile(dates, args.n_skus),
            "store_id": store_ids,
            "sku_id": sku_ids,
            "units_sold": sales.reshape(-1),
            "system_on_hand": inv.reshape(-1),
            "on_promo": promo_days[s * args.n_skus:(s + 1) * args.n_skus, :].reshape(-1),
            "is_holiday": np.tile(is_holiday, args.n_skus),
        })
        lab = pd.DataFrame({
            "date": np.tile(dates, args.n_skus),
            "store_id": store_ids,
            "sku_id": sku_ids,
            "is_phantom": labels.reshape(-1),
            "anomaly_type": types.reshape(-1),
        })
        # keep only anomaly rows + a flag? No — keep all for evaluate, hidden file stays hidden
        fact_chunks.append(df)
        label_chunks.append(lab)
        print(f"  store {s+1}/{args.n_stores} done ({stores_df['store_id'].values[s]})", flush=True)

    fact = pd.concat(fact_chunks, ignore_index=True)
    labs = pd.concat(label_chunks, ignore_index=True)
    fact.to_parquet(os.path.join(args.out, "sales_inventory.parquet"), index=False)
    # also small CSV sample for inspection
    fact.head(5000).to_csv(os.path.join(args.out, "sales_inventory_sample.csv"), index=False)
    labs.to_parquet(os.path.join(args.out, "labels_hidden.parquet"), index=False)
    stores_df.drop(columns=["demand_mult"]).to_csv(os.path.join(args.out, "stores.csv"), index=False)
    skus_df.to_csv(os.path.join(args.out, "skus.csv"), index=False)
    cal_df = pd.DataFrame({"date": dates, "is_holiday": is_holiday})
    cal_df.to_csv(os.path.join(args.out, "calendar.csv"), index=False)
    print(f"Wrote {len(fact)} rows to {args.out}/sales_inventory.parquet")
    print(f"Phantom rows: {int(labs['is_phantom'].sum())} "
          f"({100*labs['is_phantom'].mean():.3f}% of rows), "
          f"combos with phantom: {n_phantom_combos}")


if __name__ == "__main__":
    main()
