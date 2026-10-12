"""Week 1 (Day 4-7): data quality checks on the raw layer, and a baseline
analytics report before any dbt transformation happens. Run after extract_load.
"""
import json
import logging

from src import config
from src.warehouse import get_conn

log = logging.getLogger(__name__)

CHECKS = {
    "calendar_has_no_duplicate_dates":
        "SELECT COUNT(*) - COUNT(DISTINCT date) FROM raw.calendar",
    "calendar_date_range_is_continuous":
        "SELECT COUNT(*) FROM (SELECT date::DATE - LAG(date::DATE) OVER (ORDER BY date) AS gap "
        "FROM raw.calendar) WHERE gap IS NOT NULL AND gap <> 1",
    "sell_prices_no_negative_or_null_price":
        "SELECT COUNT(*) FROM raw.sell_prices WHERE sell_price IS NULL OR sell_price <= 0",
    "sales_no_duplicate_ids":
        "SELECT COUNT(*) - COUNT(DISTINCT id) FROM raw.sales_train_validation",
}


def _negative_units_check(conn) -> int:
    """DuckDB's UNPIVOT needs a static column list, so build it from the day columns present."""
    day_cols = [r[0] for r in conn.execute(
        "SELECT column_name FROM (DESCRIBE raw.sales_train_validation) WHERE column_name LIKE 'd\\_%' ESCAPE '\\'"
    ).fetchall()]
    cols = ", ".join(f'"{c}"' for c in day_cols)
    sql = f"SELECT COUNT(*) FROM (SELECT UNNEST([{cols}]) AS units FROM raw.sales_train_validation) WHERE units < 0"
    return conn.execute(sql).fetchone()[0]


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = get_conn()
    results = {}
    for name, sql in CHECKS.items():
        try:
            val = conn.execute(sql).fetchone()[0]
        except Exception as e:  # pragma: no cover - defensive, DuckDB version differences
            val, name = f"ERROR: {e}", name
        results[name] = val
        status = "PASS" if val == 0 else "CHECK"
        log.info("[%s] %s = %s", status, name, val)

    neg = _negative_units_check(conn)
    results["sales_no_negative_units"] = neg
    log.info("[%s] %s = %s", "PASS" if neg == 0 else "CHECK", "sales_no_negative_units", neg)

    n_stores = conn.execute("SELECT COUNT(DISTINCT store_id) FROM raw.sales_train_validation").fetchone()[0]
    n_items = conn.execute("SELECT COUNT(DISTINCT item_id) FROM raw.sales_train_validation").fetchone()[0]
    date_min, date_max = conn.execute("SELECT MIN(date), MAX(date) FROM raw.calendar").fetchone()
    summary = {"checks": results, "n_stores": n_stores, "n_items": n_items,
              "date_range": [str(date_min), str(date_max)]}

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "data_quality_report.json").write_text(json.dumps(summary, indent=2, default=str))
    md = [f"# Week 1 Baseline Report\n", f"- Stores: **{n_stores}**  Items: **{n_items}**  "
          f"Date range: **{date_min} to {date_max}**\n", "## Data quality checks (0 = pass)\n"]
    md += [f"- `{k}`: {v}" for k, v in results.items()]
    (config.REPORTS_DIR / "week1_baseline_report.md").write_text("\n".join(md))
    conn.close()
    log.info("Baseline report written to reports/week1_baseline_report.md")
    return summary


if __name__ == "__main__":
    run()
