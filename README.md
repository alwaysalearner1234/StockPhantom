# StockPhantom — Phantom Stock-Out Detector (R1)

System says 6 units on hand. Shelf is empty. Sales sit at zero for weeks and nobody notices.
This repo detects those **phantom stock-outs** and gives each store manager a ranked
**"go check this shelf" Top-20 list every morning** — respecting the 20-checks/day budget.

## Results (demo: 120 SKUs × 8 stores × 270 days, eval = last 60 days)

| metric | value |
|---|---|
| precision@20 (phantom-days — the manager metric) | **0.26** (~1 hit per 4 checks) |
| precision@20 (all store-days) | 0.11 |
| recall (phantom SKU-days) | 0.48 |
| **episode recall (≥1 flag per episode)** | **1.00 (11/11)** |
| alerts | 1,043 over 480 store-days ≈ **2.2 checks/store/day** (well under the 20 budget) |
| annual recovered revenue (demo scale) | **≈ ₹2.6 L/yr** (see `roi_report.md` for arithmetic) |
| extrapolated to full chain (800 SKUs × 25 stores) | **≈ ₹50–55 L/yr revenue**, ~₹13 L contribution |

Precision ceiling note: with ~1.25 true phantoms per actionable store-day, the max
achievable precision@20 is ~0.06 *if you always emit 20*. We emit only confident
alerts (~2/day), so 0.26 on phantom-days is strong — and every episode is caught.

## How it works

**Signal model** (`detector.py`): daily sales ≈ Poisson(μ).
P(N straight zero-sales days | μ) = e^(−μN) → confidence = 1 − e^(−μN).

**The justifiable N** — there is no single magic N. Required run is per-SKU:
`N_req = ceil(4.605 / μ)` (99% confidence under Poisson), clipped to [3, 21].
Promo/holiday halves N (demand ~2× expected). Slow movers (μ<0.3) additionally
need ≥10 days. This is exactly why slow SKUs don't flood the list.

**Baseline μ**: trailing-28-day mean *before the streak starts* (cumsum trick, O(n)),
**excluding promo/holiday days** so promo spikes don't inflate μ and fake anomalies.

**Boosters**: frozen system stock, promo-with-zero-sales, high stock cover,
and the **peer signal** — same SKU's chain-wide median that day (phantoms hit one
store while peers sell; chance zeros hit stores independently).

**Ranking**: by score, tie-broken by business impact (μ×run) so revenue at risk
floats up. Top-20 per store-day enforced.

**Stretch classification**: `phantom` (clean zeros + frozen stock) vs `misplaced`
(stock jiggling / intermittent ones — possibly wrong shelf) vs `delisted`
(stock → 0 tail). Delisted is suppressed from the go-check list.

## Quickstart

```bash
pip install -r requirements.txt

# 1. synthetic data (quick demo; drop --quick for full 800×25×540 ≈ 10.8M rows)
python generate_data.py --quick --seed 42

# 2. detect → ranked Top-20 per store-day
python detector.py --fact data/sales_inventory.parquet --out alerts.csv

# 3. score on hidden labels
python evaluate.py --alerts alerts.csv --labels data/labels_hidden.parquet

# 4. trust chart for any flag
python view_alert.py --store S000 --sku SKU0020 --date 2026-08-03 --out chart.png

# 5. rupees recovered, arithmetic shown
python roi_estimate.py --alerts alerts.csv --labels data/labels_hidden.parquet \
  --skus data/skus.csv --fact data/sales_inventory.parquet

# 6. manager screen
streamlit run app.py
```

Single-day scoring: `python detector.py --asof 2026-09-30 --out today_top20.csv`

## Files

| file | what |
|---|---|
| `generate_data.py` | synthetic seasonality + weekday + promo + inventory sawtooth + phantom/misplaced/delisted injection |
| `detector.py` | Poisson + adaptive-N + peer-signal detector → `alerts.csv` (store, SKU, confidence, one-line reason) |
| `evaluate.py` | precision@20 per store-day + row recall + episode recall |
| `view_alert.py` / `app.py` | sales-vs-stock chart + Streamlit Top-20 screen |
| `roi_estimate.py` | annual ₹ recovered with explicit assumptions → `roi_report.md` |
| `data/` | `sales_inventory.parquet`, `labels_hidden.parquet` (held back), `stores/skus/promotions/calendar.csv` |

## Data contract

`sales_inventory.parquet`: date, store_id, sku_id, units_sold, system_on_hand, on_promo, is_holiday.
`labels_hidden.parquet`: date, store_id, sku_id, is_phantom, anomaly_type — **never join into features**.
`alerts.csv`: date, store_id, sku_id, rank (≤20), confidence/score, zero_days, expected_daily, predicted_type, reason.
