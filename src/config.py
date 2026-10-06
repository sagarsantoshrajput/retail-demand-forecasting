"""Central configuration.

WAREHOUSE: this project uses DuckDB as a local, file-based stand-in for
Google BigQuery / Snowflake. DuckDB speaks the same SQL dialect used
throughout the dbt models, so swapping the real warehouse in later is a
one-line change: point profiles.yml's target at a dbt-bigquery or
dbt-snowflake adapter and rerun `dbt run` - no model SQL needs to change.
This mirrors how Project 1 used PostgreSQL-or-SQLite.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
WAREHOUSE_PATH = os.getenv("WAREHOUSE_PATH", str(ROOT / "warehouse" / "warehouse.duckdb"))
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models_store"

RANDOM_STATE = 42
FORECAST_HORIZON_DAYS = 30          # "30-day demand forecasts" per the spec
HIGH_VOLUME_QUANTILE = 0.75         # top 25% of items by volume -> Prophet
N_STORES = 3
N_CATEGORIES = 3
N_DEPTS_PER_CAT = 2
N_ITEMS_PER_DEPT = 8
N_DAYS_HISTORY = 3 * 365            # ~3 years, same order of magnitude as M5

for d in (DATA_RAW, REPORTS_DIR, MODELS_DIR, Path(WAREHOUSE_PATH).parent):
    d.mkdir(parents=True, exist_ok=True)
