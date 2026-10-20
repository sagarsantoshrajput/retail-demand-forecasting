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


