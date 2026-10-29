"""Week 4 (Day 1-6): Streamlit web application for inventory managers.

- Day 1-4: select store/category and view the 30-day demand forecast.
- Day 5-6: "what-if" scenario input (e.g. simulate a price change) and see
  the impact on predicted volume, using the price-elasticity relationship
  learned by the LightGBM model.
- Day 7: basic access control (see check_access() below), final polish.

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


def check_access() -> bool:
    """Week 4 (Day 7): access control. If [auth] enabled = true in
    .streamlit/secrets.toml, require a login before showing any data;
    otherwise (e.g. local dev, this sandbox) skip the gate entirely."""
    secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    auth_cfg = st.secrets.get("auth", {}) if secrets_path.exists() else {}
    if not auth_cfg.get("enabled", False):
        return True
    if st.session_state.get("authed"):
        return True
    st.title("Sign in")
    user = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if auth_cfg.get("users", {}).get(user) == pw:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")
    return False


if not check_access():
    st.stop()


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


def whatif_prediction(store_id: str, item_id: str, price_change_pct: float, horizon_days: int) -> pd.DataFrame:
    """Re-run the LightGBM recursive forecast with sell_price scaled by
    (1 + price_change_pct/100) - the model learned the price -> demand
    relationship directly from the data, so this reuses it rather than a
    hard-coded elasticity formula."""
    from src.forecast_features import add_time_features, add_lag_and_rolling_features
    from src.forecast_lightgbm import forecast_forward

    bundle = load_lightgbm_bundle()
    conn = get_conn()
    hist = conn.execute(
        "SELECT store_id, item_id, dept_id, cat_id, state_id, calendar_date, units_sold, sell_price, event_name "
        "FROM mart_sales_daily WHERE store_id = ? AND item_id = ? ORDER BY calendar_date",
        [store_id, item_id]).df()
    hist["calendar_date"] = pd.to_datetime(hist["calendar_date"])
    hist = add_time_features(hist)
    hist["sell_price"] = hist["sell_price"] * (1 + price_change_pct / 100.0)
    hist = add_lag_and_rolling_features(hist)
    fc = forecast_forward(bundle["model"], hist, horizon_days)
    return fc.rename(columns={"yhat": "predicted_units"})


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
st.sidebar.divider()
price_change = st.sidebar.slider("What-if: change sell price by (%)", -30, 30, 0, step=5,
                                 help="Simulates a price change, e.g. -10 = a 10% price drop, "
                                      "and re-forecasts demand using the learned price effect.")

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

if price_change != 0:
    st.subheader(f"What-if: {price_change:+d}% price change")
    wi = whatif_prediction(store_id, item_id, price_change, config.FORECAST_HORIZON_DAYS)
    baseline_total = fc["predicted_units"].sum() if len(fc) else np.nan
    whatif_total = wi["predicted_units"].sum()
    delta = ((whatif_total / baseline_total) - 1) * 100 if baseline_total else np.nan
    wc1, wc2, wc3 = st.columns(3)
    wc1.metric("Baseline 30-day volume", f"{baseline_total:,.0f} units" if baseline_total == baseline_total else "n/a")
    wc2.metric("What-if 30-day volume", f"{whatif_total:,.0f} units")
    wc3.metric("Projected change", f"{delta:+.1f}%" if delta == delta else "n/a")

    fig2 = go.Figure()
    if len(fc):
        fig2.add_trace(go.Scatter(x=fc.calendar_date, y=fc.predicted_units, name="Baseline forecast",
                                  mode="lines", line=dict(color="#dd8452", dash="dash")))
    fig2.add_trace(go.Scatter(x=wi.calendar_date, y=wi.predicted_units,
                              name=f"What-if ({price_change:+d}% price)", mode="lines",
                              line=dict(color="#c44e52")))
    fig2.update_layout(height=380, xaxis_title="Date", yaxis_title="Predicted units")
    st.plotly_chart(fig2, width='stretch')

with st.expander("Model accuracy (WMAPE on 30-day holdout, lower is better)"):
    st.dataframe(load_model_accuracy(), width='stretch')

with st.expander("Data lineage & documentation"):
    st.markdown("""
    - **Source:** `data/raw/*.csv` (M5-schema calendar / prices / sales)
    - **Warehouse:** DuckDB (`warehouse/warehouse.duckdb`) - a local stand-in for BigQuery/Snowflake
    - **Transform:** dbt staging -> intermediate -> marts (`dbt_project/`, run `dbt docs generate && dbt docs serve` for the full lineage graph)
    - **Forecast:** Facebook Prophet (high-volume items) + LightGBM (long-tail, multi-variable), champion picked per series by holdout WMAPE
    """)

# ---------------------------------------------------------------- Week 4 Day 7: access control
# Simple username/password gate (reads .streamlit/secrets.toml -> [auth]).
# In this container the app runs with auth disabled by default (no secrets.toml present)
# so it's runnable out of the box; flip `enabled = true` in secrets.toml for real deployments.
