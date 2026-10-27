"""Week 4 / Day 17: adds the 30-day demand forecast chart and model-accuracy table.

Run with:  streamlit run streamlit_app/app.py
"""


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import src...` works when run standalone

import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config

st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
@st.cache_resource
def get_conn():
    return duckdb.connect(config.WAREHOUSE_PATH, read_only=True)


@st.cache_data(ttl=300)
def load_summary() -> pd.DataFrame:
    return get_conn().execute("SELECT * FROM mart_item_store_summary").df()


@st.cache_data(ttl=300)
def load_daily(store_id: str, item_id: str) -> pd.DataFrame:
    return get_conn().execute(
        "SELECT calendar_date, units_sold, sell_price FROM mart_sales_daily "
        "WHERE store_id = ? AND item_id = ? ORDER BY calendar_date", [store_id, item_id]).df()


@st.cache_data(ttl=300)
def load_forecast(store_id: str, item_id: str) -> pd.DataFrame:
    return get_conn().execute(
        "SELECT calendar_date, predicted_units, actual_units, model FROM forecast_results "
        "WHERE store_id = ? AND item_id = ? ORDER BY calendar_date", [store_id, item_id]).df()


@st.cache_data(ttl=300)
def load_model_accuracy() -> pd.DataFrame:
    return get_conn().execute("SELECT * FROM forecast_model_accuracy").df()


@st.cache_resource
def load_lightgbm_bundle():
    return joblib.load(config.MODELS_DIR / "lightgbm_model.joblib")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Retail Demand Forecasting")
summary = load_summary()
if summary.empty:
    st.error("No data found in the warehouse. Run the Week 1-3 pipeline first "
            "(`python -m src.generate_m5_style_data && python -m src.extract_load && "
            "python -m src.data_quality && (cd dbt_project && dbt run) && "
            "python -m src.forecast_prophet && python -m src.forecast_lightgbm && "
            "python -m src.forecast_evaluate_and_store`).")
    st.stop()

categories = sorted(summary["cat_id"].unique())
category = st.sidebar.selectbox("Category", categories)
stores = sorted(summary.loc[summary.cat_id == category, "store_id"].unique())
store_id = st.sidebar.selectbox("Store", stores)
items = sorted(summary.loc[(summary.cat_id == category) & (summary.store_id == store_id), "item_id"].unique())
item_id = st.sidebar.selectbox("Item", items)

st.title(f"{item_id} @ {store_id}")

daily = load_daily(store_id, item_id)
daily["calendar_date"] = pd.to_datetime(daily["calendar_date"])
fc = load_forecast(store_id, item_id)
fc["calendar_date"] = pd.to_datetime(fc["calendar_date"])

row = summary[(summary.store_id == store_id) & (summary.item_id == item_id)].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg daily units", f"{row.avg_daily_units:.1f}")
c2.metric("Avg sell price", f"${row.avg_sell_price:.2f}")
c3.metric("Total units (history)", f"{int(row.total_units):,}")
model_used = fc["model"].iloc[0] if len(fc) else "n/a"
c4.metric("Champion model for this item", model_used.title())

st.subheader("30-day demand forecast vs. history")
fig = go.Figure()
recent = daily.tail(90)
fig.add_trace(go.Scatter(x=recent.calendar_date, y=recent.units_sold, name="Actual (last 90d)",
                         mode="lines", line=dict(color="#4c72b0")))
if len(fc):
    fig.add_trace(go.Scatter(x=fc.calendar_date, y=fc.predicted_units, name=f"Forecast ({model_used})",
                             mode="lines", line=dict(color="#dd8452", dash="dash")))
    if fc["actual_units"].notna().any():
        fig.add_trace(go.Scatter(x=fc.calendar_date, y=fc.actual_units, name="Actual (holdout)",
                                 mode="lines", line=dict(color="#55a868")))
fig.update_layout(height=420, xaxis_title="Date", yaxis_title="Units", legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, width='stretch')

with st.expander("Model accuracy (WMAPE on 30-day holdout, lower is better)"):
    st.dataframe(load_model_accuracy(), width='stretch')

with st.expander("Data lineage & documentation"):
    st.markdown("""
    - **Source:** `data/raw/*.csv` (M5-schema calendar / prices / sales)
    - **Warehouse:** DuckDB (`warehouse/warehouse.duckdb`) - a local stand-in for BigQuery/Snowflake
    - **Transform:** dbt staging -> intermediate -> marts (`dbt_project/`, run `dbt docs generate && dbt docs serve` for the full lineage graph)
    - **Forecast:** Facebook Prophet (high-volume items) + LightGBM (long-tail, multi-variable), champion picked per series by holdout WMAPE
    """)
