"""app.py — Manager trust screen (Streamlit).
Run:  streamlit run app.py
Shows ranked Top-20 go-check list per store-day + sales-vs-stock chart per alert.
"""
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Phantom Stock-Out — Go Check This Shelf", layout="wide")
st.title("🏪 Phantom Stock-Out — Go Check This Shelf")

@st.cache_data
def load():
    alerts = pd.read_csv("alerts.csv", dtype={"store_id": str, "sku_id": str})
    fact = pd.read_parquet("data/sales_inventory.parquet")
    alerts["date"] = pd.to_datetime(alerts["date"]).dt.strftime("%Y-%m-%d")
    fact["date"] = pd.to_datetime(fact["date"])
    return alerts, fact

try:
    alerts, fact = load()
except Exception as e:
    st.error(f"Run the pipeline first: `python generate_data.py --quick` then `python detector.py`. ({e})")
    st.stop()

dates = sorted(alerts["date"].unique())
stores = sorted(alerts["store_id"].unique())
c1, c2 = st.columns(2)
day = c1.selectbox("Date", dates, index=len(dates) - 1)
store = c2.selectbox("Store", stores)
top = alerts[(alerts["date"] == day) & (alerts["store_id"] == store)].sort_values("rank")
st.subheader(f"{store} — {day} — {len(top)} alerts (budget: 20)")
st.dataframe(top[["rank", "sku_id", "confidence", "zero_days", "expected_daily",
                  "system_on_hand", "predicted_type", "reason"]],
             use_container_width=True, hide_index=True)
if top.empty:
    st.info("No flags today — shelves look healthy.")
    st.stop()
sel = st.selectbox("Inspect SKU", top["sku_id"].tolist())
row = top[top["sku_id"] == sel].iloc[0]
st.markdown(f"**Why flagged:** {row['reason']}  \nConfidence **{float(row['confidence']):.2f}** · "
            f"zero-days **{int(row['zero_days'])}** · usually **{float(row['expected_daily']):.1f}/day** · "
            f"system shows **{int(row['system_on_hand'])}** · type **{row.get('predicted_type','phantom')}**")
g = fact[(fact["store_id"] == store) & (fact["sku_id"] == sel)]
asof = pd.Timestamp(day)
g = g[(g["date"] > asof - pd.Timedelta(days=60)) & (g["date"] <= asof)].sort_values("date")
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(g["date"], g["units_sold"], color="#2563eb", alpha=0.85)
ax1.set_ylabel("Units sold")
ax2 = ax1.twinx()
ax2.plot(g["date"], g["system_on_hand"], color="#dc2626", linewidth=2)
ax2.set_ylabel("System on-hand")
ax1.set_title(f"{store} × {sel} — sales vs system stock to {day}")
ax1.tick_params(axis="x", rotation=30)
fig.tight_layout()
st.pyplot(fig)
