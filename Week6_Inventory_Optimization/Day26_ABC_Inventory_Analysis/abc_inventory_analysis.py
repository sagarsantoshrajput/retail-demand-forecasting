
"""Day 26: ABC Inventory Analysis using M5 sales and price data."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

DATA_DIR = Path(config.DATA_RAW)
REPORT_DIR = Path(config.REPORTS_DIR)

SALES_FILE = DATA_DIR / "sales_train_validation.csv"
PRICE_FILE = DATA_DIR / "sell_prices.csv"
CALENDAR_FILE = DATA_DIR / "calendar.csv"

ABC_REPORT = REPORT_DIR / "day26_abc_inventory_analysis.csv"
SUMMARY_REPORT = REPORT_DIR / "day26_abc_category_summary.csv"

HISTORY_DAYS = 28
A_THRESHOLD = 0.80
B_THRESHOLD = 0.95


def load_data():
    """Load sales, calendar and selling price datasets."""
    for path in (SALES_FILE, PRICE_FILE, CALENDAR_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    sales = pd.read_csv(SALES_FILE)
    prices = pd.read_csv(PRICE_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)

    required_sales = {"store_id", "item_id"}
    required_prices = {"store_id", "item_id", "wm_yr_wk", "sell_price"}
    required_calendar = {"d", "wm_yr_wk"}

    if not required_sales.issubset(sales.columns):
        raise ValueError("Sales dataset is missing required columns.")

    if not required_prices.issubset(prices.columns):
        raise ValueError("Price dataset is missing required columns.")

    if not required_calendar.issubset(calendar.columns):
        raise ValueError("Calendar dataset is missing required columns.")

    return sales, prices, calendar


def prepare_sales(sales, calendar):
    """Convert recent daily sales into a store-product-date dataset."""
    daily_columns = [
        col for col in sales.columns
        if col.startswith("d_") and col[2:].isdigit()
    ]

    daily_columns.sort(key=lambda col: int(col[2:]))

    if len(daily_columns) < HISTORY_DAYS:
        raise ValueError("Insufficient sales history.")

    selected_days = daily_columns[-HISTORY_DAYS:]

    data = sales[
        ["store_id", "item_id"] + selected_days
    ].melt(
        id_vars=["store_id", "item_id"],
        var_name="d",
        value_name="units_sold",
    )

    data["units_sold"] = pd.to_numeric(
        data["units_sold"], errors="coerce"
    )

    data["units_sold"] = data["units_sold"].fillna(0)

    if (data["units_sold"] < 0).any():
        raise ValueError("Sales quantities cannot be negative.")

    calendar_subset = calendar[["d", "wm_yr_wk"]].drop_duplicates("d")

    data = data.merge(
        calendar_subset,
        on="d",
        how="left",
        validate="many_to_one",
    )

    if data["wm_yr_wk"].isna().any():
        raise ValueError("Some sales dates have no calendar week mapping.")

    return data, selected_days


def calculate_product_value(sales_data, prices):
    """Calculate sales revenue for each store-product combination."""
    prices = prices[
        ["store_id", "item_id", "wm_yr_wk", "sell_price"]
    ].copy()

    prices["sell_price"] = pd.to_numeric(
        prices["sell_price"], errors="coerce"
    )

    if (prices["sell_price"].dropna() < 0).any():
        raise ValueError("Selling prices cannot be negative.")

    merged = sales_data.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )

    if merged["sell_price"].isna().any():
        logger.warning(
            "%s sales records have no matching selling price and "
            "will be excluded from revenue calculations.",
            int(merged["sell_price"].isna().sum()),
        )

    merged["sales_value"] = (
        merged["units_sold"] * merged["sell_price"]
    )

    result = (
        merged.groupby(["store_id", "item_id"])
        .agg(
            total_units_sold=("units_sold", "sum"),
            total_sales_value=("sales_value", "sum"),
            priced_sales_records=("sell_price", "count"),
        )
        .reset_index()
    )

    result = result[result["priced_sales_records"] > 0].copy()

    if result.empty:
        raise ValueError("No valid sales records with matching prices.")

    return result


def classify_abc(product_data):
    """Assign ABC classes using cumulative revenue contribution."""
    result = product_data.copy()

    total_value = result["total_sales_value"].sum()

    if total_value <= 0:
        raise ValueError("Total sales value must be greater than zero.")

    result = result.sort_values(
        "total_sales_value", ascending=False
    ).reset_index(drop=True)

    result["sales_value_share_pct"] = (
        result["total_sales_value"] / total_value * 100
    )

    result["cumulative_sales_share_pct"] = (
        result["sales_value_share_pct"].cumsum()
    )

    previous_share = (
        result["cumulative_sales_share_pct"]
        - result["sales_value_share_pct"]
    ) / 100

    result["abc_category"] = np.select(
        [
            previous_share < A_THRESHOLD,
            previous_share < B_THRESHOLD,
        ],
        ["A", "B"],
        default="C",
    )

    result["rank"] = np.arange(1, len(result) + 1)

    return result


def build_category_summary(classified):
    """Summarize products and revenue contribution by ABC category."""
    summary = (
        classified.groupby("abc_category")
        .agg(
            product_count=("item_id", "count"),
            total_units_sold=("total_units_sold", "sum"),
            total_sales_value=("total_sales_value", "sum"),
        )
        .reset_index()
    )

    total_value = classified["total_sales_value"].sum()

    summary["revenue_contribution_pct"] = (
        summary["total_sales_value"] / total_value * 100
    )

    return summary.sort_values("abc_category")


def run():
    """Execute the Day 26 ABC inventory analysis."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sales, prices, calendar = load_data()

    sales_data, selected_days = prepare_sales(sales, calendar)

    product_value = calculate_product_value(
        sales_data, prices
    )

    classified = classify_abc(product_value)

    summary = build_category_summary(classified)

    classified.to_csv(ABC_REPORT, index=False)
    summary.to_csv(SUMMARY_REPORT, index=False)

    print("\nDAY 26 - ABC INVENTORY ANALYSIS")
    print("--------------------------------")
    print(f"Analysis period: {len(selected_days)} days")
    print(f"Store-product combinations: {len(classified)}")
    print(
        f"Total sales value: "
        f"{classified['total_sales_value'].sum():,.2f}"
    )

    print("\nABC Category Summary:")
    print(summary.to_string(index=False))

    print("\nTop 10 products by sales value:")
    print(classified.head(10).to_string(index=False))

    print("\nReports generated:")
    print(ABC_REPORT)
    print(SUMMARY_REPORT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
