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


