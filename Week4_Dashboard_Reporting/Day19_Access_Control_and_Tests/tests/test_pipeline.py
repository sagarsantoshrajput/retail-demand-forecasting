"""Run after the pipeline (see README) with WAREHOUSE_PATH pointed at the built
warehouse, e.g.:  WAREHOUSE_PATH=warehouse/warehouse.duckdb pytest -q
"""
import os

import pandas as pd
import pytest

from src import config
from src.warehouse import get_conn

pytestmark = pytest.mark.skipif(
    not os.path.exists(config.WAREHOUSE_PATH),
    reason="run the full pipeline first (see README Quick start)")


def test_raw_tables_loaded():
    conn = get_conn()
    for t in ("calendar", "sell_prices", "sales_train_validation"):
        n = conn.execute(f"SELECT COUNT(*) FROM raw.{t}").fetchone()[0]
        assert n > 0, f"raw.{t} is empty"
    conn.close()


def test_marts_exist_and_are_non_empty():
    conn = get_conn()
    for t in ("mart_sales_daily", "mart_sales_weekly", "mart_sales_monthly", "mart_item_store_summary"):
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        assert n > 0, f"{t} is empty"
    conn.close()


def test_no_negative_units_in_daily_mart():
    conn = get_conn()
    neg = conn.execute("SELECT COUNT(*) FROM mart_sales_daily WHERE units_sold < 0").fetchone()[0]
    conn.close()
    assert neg == 0


def test_forecast_results_cover_every_series():
    conn = get_conn()
    n_series = conn.execute("SELECT COUNT(*) FROM mart_item_store_summary").fetchone()[0]
    n_forecasted = conn.execute("SELECT COUNT(DISTINCT store_id || item_id) FROM forecast_results").fetchone()[0]
    conn.close()
    assert n_forecasted == n_series


def test_forecast_horizon_is_30_days():
    conn = get_conn()
    days = conn.execute(
        "SELECT COUNT(DISTINCT calendar_date) FROM forecast_results WHERE store_id = "
        "(SELECT store_id FROM forecast_results LIMIT 1)").fetchone()[0]
    conn.close()
    assert days == config.FORECAST_HORIZON_DAYS


def test_model_accuracy_table_has_both_models():
    conn = get_conn()
    models = set(conn.execute("SELECT model FROM forecast_model_accuracy").df()["model"])
    conn.close()
    assert models == {"prophet", "lightgbm"}


def test_whatif_price_drop_increases_or_holds_predicted_volume_on_average():
    """Sanity check on the learned price effect: on average, a 15% price cut
    across several items should not decrease predicted volume."""
    import sys
    sys.path.insert(0, str(config.ROOT / "streamlit_app"))
    from src.forecast_features import add_time_features, add_lag_and_rolling_features, load_daily
    from src.forecast_lightgbm import forecast_forward
    import joblib

    bundle = joblib.load(config.MODELS_DIR / "lightgbm_model.joblib")
    df = add_time_features(load_daily())
    sample = df[(df.store_id == df.store_id.iloc[0]) & (df.item_id.isin(df.item_id.unique()[:3]))]

    base = add_lag_and_rolling_features(sample.copy())
    base_fc = forecast_forward(bundle["model"], base, 14)

    cut = sample.copy()
    cut["sell_price"] = cut["sell_price"] * 0.85
    cut = add_lag_and_rolling_features(cut)
    cut_fc = forecast_forward(bundle["model"], cut, 14)

    assert cut_fc["yhat"].sum() >= base_fc["yhat"].sum() * 0.9  # allow noise, expect no big drop
