
"""Day 21: Enhanced validation for retail forecasting raw datasets."""

import json
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def add_check(checks, name, issue_count):
    """Record a validation check."""
    checks[name] = {
        "issues": int(issue_count),
        "status": "PASS" if issue_count == 0 else "FAIL",
    }


def validate_calendar(df):
    checks = {}

    if "date" not in df.columns:
        add_check(checks, "date_column_present", 1)
        return checks

    add_check(checks, "missing_dates", df["date"].isna().sum())

    parsed = pd.to_datetime(df["date"], errors="coerce")
    add_check(checks, "invalid_dates", parsed.isna().sum())
    add_check(checks, "duplicate_dates", parsed.duplicated().sum())

    required = ["wm_yr_wk", "weekday", "wday", "month", "year", "d"]
    missing = sum(
        int(df[col].isna().sum())
        for col in required if col in df.columns
    )
    add_check(checks, "missing_calendar_fields", missing)

    return checks


def validate_prices(df):
    checks = {}

    required = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    missing_columns = [c for c in required if c not in df.columns]
    add_check(checks, "required_columns_present", len(missing_columns))

    if missing_columns:
        return checks

    identifiers = ["store_id", "item_id", "wm_yr_wk"]
    missing_ids = sum(int(df[c].isna().sum()) for c in identifiers)
    add_check(checks, "missing_identifiers", missing_ids)

    prices = pd.to_numeric(df["sell_price"], errors="coerce")
    add_check(checks, "invalid_or_missing_prices", prices.isna().sum())
    add_check(checks, "non_positive_prices", (prices <= 0).sum())

    duplicate_keys = df.duplicated(identifiers).sum()
    add_check(checks, "duplicate_store_item_week", duplicate_keys)

    return checks


def validate_sales(df):
    checks = {}

    required = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    missing_columns = [c for c in required if c not in df.columns]
    add_check(checks, "required_columns_present", len(missing_columns))

    if missing_columns:
        return checks

    missing_ids = sum(int(df[c].isna().sum()) for c in required)
    add_check(checks, "missing_identifiers", missing_ids)
    add_check(checks, "duplicate_product_ids", df["id"].duplicated().sum())

    day_columns = [c for c in df.columns if c.startswith("d_")]
    add_check(checks, "daily_sales_columns_present", 0 if day_columns else 1)

    if day_columns:
        numeric = df[day_columns].apply(pd.to_numeric, errors="coerce")
        add_check(checks, "invalid_or_missing_sales", numeric.isna().sum().sum())
        add_check(checks, "negative_sales", (numeric < 0).sum().sum())

    return checks


def validate_holdout(df):
    checks = {}

    if "d" not in df.columns:
        add_check(checks, "date_column_present", 1)
        return checks

    add_check(checks, "missing_dates", df["d"].isna().sum())
    add_check(checks, "duplicate_dates", df["d"].duplicated().sum())

    product_columns = [c for c in df.columns if c != "d"]
    add_check(checks, "product_columns_present", 0 if product_columns else 1)

    if product_columns:
        numeric = df[product_columns].apply(pd.to_numeric, errors="coerce")
        add_check(checks, "invalid_or_missing_actuals", numeric.isna().sum().sum())
        add_check(checks, "negative_actuals", (numeric < 0).sum().sum())

    return checks


def validate_dataset(name, path, validator):
    if not path.exists():
        return {
            "file": path.name,
            "status": "ERROR",
            "row_count": 0,
            "checks": {"file_exists": {"issues": 1, "status": "FAIL"}},
        }

    df = pd.read_csv(path)
    checks = validator(df)

    failed = any(item["status"] == "FAIL" for item in checks.values())

    return {
        "file": path.name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "status": "FAIL" if failed else "PASS",
        "checks": checks,
    }


def run_validation():
    datasets = {
        "calendar": (RAW_DIR / "calendar.csv", validate_calendar),
        "sell_prices": (RAW_DIR / "sell_prices.csv", validate_prices),
        "sales_train_validation": (
            RAW_DIR / "sales_train_validation.csv", validate_sales
        ),
        "actuals_holdout": (RAW_DIR / "actuals_holdout.csv", validate_holdout),
    }

    results = {}

    for name, (path, validator) in datasets.items():
        try:
            results[name] = validate_dataset(name, path, validator)
        except Exception as exc:
            results[name] = {
                "file": path.name,
                "status": "ERROR",
                "error": str(exc),
                "checks": {},
            }

        log.info("%s: %s", name, results[name]["status"])

    overall = (
        "PASS"
        if all(result["status"] == "PASS" for result in results.values())
        else "FAIL"
    )

    report = {
        "project": "Retail Demand Forecasting",
        "validation": "Day 21 Enhanced Data Validation",
        "overall_status": overall,
        "datasets": results,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "day21_data_validation_report.json"
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    log.info("Overall validation: %s", overall)
    log.info("Report saved to %s", report_path)

    return report


if __name__ == "__main__":
    run_validation()
