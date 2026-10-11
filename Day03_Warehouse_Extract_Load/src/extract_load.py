"""Week 1 (Day 1-3): extraction scripts that load the raw M5 dataset into the
warehouse's `raw` schema, as-is, with no cleaning (that happens in dbt staging
models in Week 2). Reads calendar.csv, sell_prices.csv and
sales_train_validation.csv from data/raw/ - whether those are the generated
sample data or the real Kaggle M5 files, the loader doesn't care.
"""
import logging

from src import config
from src.warehouse import get_conn

log = logging.getLogger(__name__)

REQUIRED_FILES = ["calendar.csv", "sell_prices.csv", "sales_train_validation.csv"]


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    missing = [f for f in REQUIRED_FILES if not (config.DATA_RAW / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {config.DATA_RAW}. Run `python -m src.generate_m5_style_data` "
            "or drop the real M5 Kaggle CSVs into that folder.")

    conn = get_conn()
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute(f"""
        CREATE OR REPLACE TABLE raw.calendar AS
        SELECT * FROM read_csv_auto('{config.DATA_RAW / "calendar.csv"}', header=True)
    """)
    conn.execute(f"""
        CREATE OR REPLACE TABLE raw.sell_prices AS
        SELECT * FROM read_csv_auto('{config.DATA_RAW / "sell_prices.csv"}', header=True)
    """)
    conn.execute(f"""
        CREATE OR REPLACE TABLE raw.sales_train_validation AS
        SELECT * FROM read_csv_auto('{config.DATA_RAW / "sales_train_validation.csv"}', header=True)
    """)
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM raw.{t}").fetchone()[0]
             for t in ("calendar", "sell_prices", "sales_train_validation")}
    log.info("Loaded raw tables: %s", counts)
    conn.close()
    return counts


if __name__ == "__main__":
    run()
