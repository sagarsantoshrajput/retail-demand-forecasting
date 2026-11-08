"""Day 23: Enhanced Streamlit dashboard for Retail Demand Forecasting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.20);
        padding: 16px;
        border-radius: 12px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    .dashboard-subtitle {
        color: gray;
        font-size: 0.95rem;
        margin-top: -12px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Access control
# ---------------------------------------------------------
def check_access() -> bool:
    """Require login only when authentication is enabled."""
    secrets_path = Path(__file__).resolve().parents[1] / ".streamlit" / "secrets.toml"
    auth_cfg = st.secrets.get("auth", {}) if secrets_path.exists() else {}

    if not auth_cfg.get("enabled", False):
        return True

    if st.session_state.get("authed"):
        return True

    st.title("Sign in")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign in"):
        if auth_cfg.get("users", {}).get(user) == password:
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    return False


if not check_access():
    st.stop()


# ---------------------------------------------------------
# Data access
# ---------------------------------------------------------
@st.cache_resource
def get_conn():
    return duckdb.connect(config.WAREHOUSE_PATH, read_only=True)


@st.cache_data(ttl=300)
def load_summary() -> pd.DataFrame:
    return get_conn().execute(
        "SELECT * FROM mart_item_store_summary"
    ).df()


@st.cache_data(ttl=300)
def load_daily(store_id: str, item_id: str) -> pd.DataFrame:
    return get_conn().execute(
        """
        SELECT calendar_date, units_sold, sell_price
        FROM mart_sales_daily
        WHERE store_id = ? AND item_id = ?
        ORDER BY calendar_date
        """,
        [store_id, item_id],
    ).df()


@st.cache_data(ttl=300)
def load_forecast(store_id: str, item_id: str) -> pd.DataFrame:
    return get_conn().execute(
        """
        SELECT calendar_date, predicted_units, actual_units, model
        FROM forecast_results
        WHERE store_id = ? AND item_id = ?
        ORDER BY calendar_date
        """,
        [store_id, item_id],
    ).df()


@st.cache_data(ttl=300)
def load_model_accuracy() -> pd.DataFrame:
    return get_conn().execute(
        "SELECT * FROM forecast_model_accuracy"
    ).df()


@st.cache_resource
def load_lightgbm_bundle():
    return joblib.load(config.MODELS_DIR / "lightgbm_model.joblib")


# ---------------------------------------------------------
# What-if forecast
# ---------------------------------------------------------
def whatif_prediction(
    store_id: str,
    item_id: str,
    price_change_pct: float,
    horizon_days: int,
) -> pd.DataFrame:
    """Reforecast demand after changing the historical selling price."""
    from src.forecast_features import (
        add_time_features,
        add_lag_and_rolling_features,
    )
    from src.forecast_lightgbm import forecast_forward

    bundle = load_lightgbm_bundle()
    conn = get_conn()

    hist = conn.execute(
        """
        SELECT store_id, item_id, dept_id, cat_id, state_id,
               calendar_date, units_sold, sell_price, event_name
        FROM mart_sales_daily
        WHERE store_id = ? AND item_id = ?
        ORDER BY calendar_date
        """,
        [store_id, item_id],
    ).df()

    hist["calendar_date"] = pd.to_datetime(hist["calendar_date"])
    hist = add_time_features(hist)
    hist["sell_price"] *= 1 + price_change_pct / 100.0
    hist = add_lag_and_rolling_features(hist)

    forecast = forecast_forward(
        bundle["model"], hist, horizon_days
    )

    return forecast.rename(columns={"yhat": "predicted_units"})


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.title("📊 Retail Demand Forecasting")
st.sidebar.caption("Demand planning and inventory insights")

summary = load_summary()

if summary.empty:
    st.error(
        "No data found in the warehouse. Run the project pipeline "
        "before opening the dashboard."
    )
    st.stop()

categories = sorted(summary["cat_id"].dropna().unique())
category = st.sidebar.selectbox("Product Category", categories)

stores = sorted(
    summary.loc[summary.cat_id == category, "store_id"].dropna().unique()
)
store_id = st.sidebar.selectbox("Store", stores)

items = sorted(
    summary.loc[
        (summary.cat_id == category) &
        (summary.store_id == store_id),
        "item_id",
    ].dropna().unique()
)
item_id = st.sidebar.selectbox("Product Item", items)

st.sidebar.divider()
st.sidebar.subheader("What-if Scenario")

price_change = st.sidebar.slider(
    "Change selling price (%)",
    min_value=-30,
    max_value=30,
    value=0,
    step=5,
    help="For example, -10 represents a 10% price reduction.",
)

st.sidebar.caption(
    "Use the tabs in the main dashboard to explore forecasts, "
    "scenarios and model performance."
)


# ---------------------------------------------------------
# Load selected product data
# ---------------------------------------------------------
daily = load_daily(store_id, item_id)
fc = load_forecast(store_id, item_id)

daily["calendar_date"] = pd.to_datetime(daily["calendar_date"])
fc["calendar_date"] = pd.to_datetime(fc["calendar_date"])

row = summary[
    (summary.store_id == store_id) &
    (summary.item_id == item_id)
].iloc[0]

model_used = (
    fc["model"].iloc[0]
    if not fc.empty
    else "n/a"
)

forecast_total = (
    float(fc["predicted_units"].sum())
    if not fc.empty
    else 0.0
)


# ---------------------------------------------------------
# Dashboard heading
# ---------------------------------------------------------
st.title("Retail Demand Forecasting Dashboard")
st.markdown(
    f'<p class="dashboard-subtitle">'
    f'Product: <b>{item_id}</b> &nbsp; | &nbsp; '
    f'Store: <b>{store_id}</b> &nbsp; | &nbsp; '
    f'Category: <b>{category}</b>'
    f'</p>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# KPI cards
# ---------------------------------------------------------
st.subheader("Business Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Average Daily Sales",
    f"{row.avg_daily_units:,.1f} units",
)

k2.metric(
    "Average Selling Price",
    f"${row.avg_sell_price:,.2f}",
)

k3.metric(
    "Historical Units Sold",
    f"{int(row.total_units):,}",
)

k4.metric(
    "Forecast Demand (30 Days)",
    f"{forecast_total:,.0f} units",
)

st.caption(f"Forecast model selected for this item: **{str(model_used).title()}**")


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
overview_tab, whatif_tab, accuracy_tab, documentation_tab = st.tabs(
    [
        "📈 Overview",
        "🧪 What-if Analysis",
        "🎯 Model Accuracy",
        "📚 Documentation",
    ]
)


# ---------------------------------------------------------
# Overview tab
# ---------------------------------------------------------
with overview_tab:
    st.subheader("Historical Sales Trend")

    if daily.empty:
        st.info("No historical sales data is available for this selection.")
    else:
        min_date = daily["calendar_date"].min().date()
        max_date = daily["calendar_date"].max().date()

        date_range = st.date_input(
            "Select historical date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="history_date_range",
        )

        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_daily = daily[
                daily["calendar_date"].dt.date.between(start_date, end_date)
            ]
        else:
            filtered_daily = daily

        history_fig = go.Figure()

        history_fig.add_trace(
            go.Scatter(
                x=filtered_daily["calendar_date"],
                y=filtered_daily["units_sold"],
                name="Units Sold",
                mode="lines",
                line=dict(width=2),
                fill="tozeroy",
            )
        )

        history_fig.update_layout(
            height=380,
            xaxis_title="Date",
            yaxis_title="Units Sold",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
        )

        st.plotly_chart(history_fig, width="stretch")

    st.divider()
    st.subheader("Forecast vs Actual Demand")

    if fc.empty:
        st.info("No forecast results are available for this product.")
    else:
        forecast_fig = go.Figure()

        forecast_fig.add_trace(
            go.Scatter(
                x=fc["calendar_date"],
                y=fc["predicted_units"],
                name="Predicted Demand",
                mode="lines+markers",
                line=dict(width=2, dash="dash"),
            )
        )

        if fc["actual_units"].notna().any():
            forecast_fig.add_trace(
                go.Scatter(
                    x=fc["calendar_date"],
                    y=fc["actual_units"],
                    name="Actual Demand",
                    mode="lines+markers",
                    line=dict(width=2),
                )
            )

        forecast_fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Units",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=20, r=20, t=30, b=50),
        )

        st.plotly_chart(forecast_fig, width="stretch")

        st.caption(
            "Compare predicted demand with actual holdout demand "
            "where actual observations are available."
        )


# ---------------------------------------------------------
# What-if analysis tab
# ---------------------------------------------------------
with whatif_tab:
    st.subheader("Price Change Impact on Demand")

    if price_change == 0:
        st.info(
            "Change the selling price using the slider in the sidebar "
            "to run a what-if scenario."
        )
    else:
        st.write(
            f"Scenario: selling price changes by **{price_change:+d}%**."
        )

        with st.spinner("Generating scenario forecast..."):
            wi = whatif_prediction(
                store_id,
                item_id,
                price_change,
                config.FORECAST_HORIZON_DAYS,
            )

        whatif_total = float(wi["predicted_units"].sum())

        delta = (
            ((whatif_total / forecast_total) - 1) * 100
            if forecast_total != 0
            else np.nan
        )

        w1, w2, w3 = st.columns(3)

        w1.metric(
            "Baseline Forecast",
            f"{forecast_total:,.0f} units",
        )

        w2.metric(
            "Scenario Forecast",
            f"{whatif_total:,.0f} units",
        )

        w3.metric(
            "Demand Change",
            f"{delta:+.1f}%" if np.isfinite(delta) else "n/a",
        )

        scenario_fig = go.Figure()

        if not fc.empty:
            scenario_fig.add_trace(
                go.Scatter(
                    x=fc["calendar_date"],
                    y=fc["predicted_units"],
                    name="Baseline Forecast",
                    mode="lines+markers",
                )
            )

        scenario_fig.add_trace(
            go.Scatter(
                x=wi["calendar_date"],
                y=wi["predicted_units"],
                name="What-if Forecast",
                mode="lines+markers",
                line=dict(dash="dash"),
            )
        )

        scenario_fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Predicted Units",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
        )

        st.plotly_chart(scenario_fig, width="stretch")

        st.caption(
            "Scenario forecasts use the LightGBM model and adjusted "
            "selling-price features."
        )


# ---------------------------------------------------------
# Model accuracy tab
# ---------------------------------------------------------
with accuracy_tab:
    st.subheader("Forecast Model Performance")
    st.caption("WMAPE on the 30-day holdout period; lower values indicate better accuracy.")

    accuracy = load_model_accuracy()

    if accuracy.empty:
        st.info("Model accuracy results are not available.")
    else:
        st.dataframe(
            accuracy,
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------
# Documentation tab
# ---------------------------------------------------------
with documentation_tab:
    st.subheader("Project Data Lineage")

    st.markdown("""
    **Data Sources**
    - Raw calendar, selling-price and sales data from `data/raw/`.

    **Data Warehouse**
    - DuckDB database: `warehouse/warehouse.duckdb`.

    **Data Transformation**
    - dbt staging, intermediate and mart models in `dbt_project/`.

    **Forecasting Models**
    - Prophet for selected demand series.
    - LightGBM for multi-variable demand forecasting.
    - Champion model selection based on holdout WMAPE.

    **Dashboard**
    - Streamlit provides interactive product selection,
      historical demand visualization, forecast comparison
      and price-change scenario analysis.
    """)

    st.divider()
    st.subheader("Selected Product Details")

    details = pd.DataFrame({
        "Metric": [
            "Category",
            "Store",
            "Item",
            "Forecast Model",
            "Historical Units",
            "Average Daily Units",
        ],
        "Value": [
            category,
            store_id,
            item_id,
            str(model_used).title(),
            f"{int(row.total_units):,}",
            f"{row.avg_daily_units:.2f}",
        ],
    })

    st.dataframe(details, hide_index=True, width="stretch")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "Retail Demand Forecasting & Inventory Optimization | "
    "Day 23 – Dashboard UI Enhancement"
)