
"""Day 25: Inventory Reorder Recommendations.

Uses recent M5 sales history to calculate demand, safety stock,
reorder points and recommended replenishment quantities.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

DATA_DIR = Path(config.DATA_RAW)
REPORT_DIR = Path(config.REPORTS_DIR)

SALES_FILE = DATA_DIR / "sales_train_validation.csv"
STOCK_FILE = DATA_DIR / "inventory_on_hand.csv"

RECOMMENDATION_FILE = REPORT_DIR / "day25_inventory_recommendations.csv"
SUMMARY_FILE = REPORT_DIR / "day25_inventory_summary.csv"

HISTORY_DAYS = 28
LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
SERVICE_FACTOR = 1.65


def load_sales(path=SALES_FILE):
    """Load M5 sales data and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Sales file not found: {path}")

    df = pd.read_csv(path)

    required = {"store_id", "item_id"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing sales columns: {sorted(missing)}")

    daily_columns = [
        col for col in df.columns
        if col.startswith("d_") and col[2:].isdigit()
    ]
    daily_columns.sort(key=lambda col: int(col[2:]))

    if len(daily_columns) < HISTORY_DAYS:
        raise ValueError("Not enough daily sales history available.")

    selected_days = daily_columns[-HISTORY_DAYS:]

    df[selected_days] = df[selected_days].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    if (df[selected_days] < 0).any().any():
        raise ValueError("Sales quantities cannot be negative.")

    return df, selected_days


def build_demand_metrics(df, daily_columns):
    """Calculate average demand and variability by store and product."""
    grouped = (
        df.groupby(["store_id", "item_id"])[daily_columns]
        .sum()
    )

    metrics = grouped.copy()

    metrics["avg_daily_demand"] = grouped.mean(axis=1)
    metrics["daily_demand_std"] = grouped.std(axis=1, ddof=1).fillna(0)

    metrics = metrics.reset_index()

    return metrics[
        [
            "store_id",
            "item_id",
            "avg_daily_demand",
            "daily_demand_std",
        ]
    ]


def load_inventory(metrics):
    """Load stock levels or create a template for the user to complete."""
    if not STOCK_FILE.exists():
        template = metrics[["store_id", "item_id"]].copy()
        template["on_hand_units"] = np.nan
        template.to_csv(STOCK_FILE, index=False)

        logger.info("Created inventory input template: %s", STOCK_FILE)

    stock = pd.read_csv(STOCK_FILE)

    required = {"store_id", "item_id", "on_hand_units"}
    missing = required - set(stock.columns)

    if missing:
        raise ValueError(
            f"Inventory file is missing columns: {sorted(missing)}"
        )

    stock["on_hand_units"] = pd.to_numeric(
        stock["on_hand_units"], errors="coerce"
    )

    if (stock["on_hand_units"].dropna() < 0).any():
        raise ValueError("On-hand inventory cannot be negative.")

    return stock[["store_id", "item_id", "on_hand_units"]]


def calculate_recommendations(metrics, stock):
    """Calculate safety stock, reorder point and order quantity."""
    result = metrics.merge(
        stock,
        on=["store_id", "item_id"],
        how="left",
    )

    result["lead_time_days"] = LEAD_TIME_DAYS
    result["review_period_days"] = REVIEW_PERIOD_DAYS

    result["safety_stock"] = (
        SERVICE_FACTOR
        * result["daily_demand_std"]
        * np.sqrt(LEAD_TIME_DAYS)
    ).apply(np.ceil)

    result["reorder_point"] = (
        result["avg_daily_demand"] * LEAD_TIME_DAYS
        + result["safety_stock"]
    ).apply(np.ceil)

    result["target_stock_level"] = (
        result["avg_daily_demand"]
        * (LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS)
        + result["safety_stock"]
    ).apply(np.ceil)

    result["recommended_order_qty"] = np.where(
        result["on_hand_units"].notna(),
        np.maximum(
            result["target_stock_level"] - result["on_hand_units"],
            0,
        ),
        np.nan,
    )

    result["inventory_status"] = np.select(
        [
            result["on_hand_units"].isna(),
            result["on_hand_units"] <= result["reorder_point"],
        ],
        [
            "Stock Data Required",
            "Reorder Required",
        ],
        default="Stock Sufficient",
    )

    result["stockout_risk"] = np.select(
        [
            result["on_hand_units"].isna(),
            result["on_hand_units"] <= result["avg_daily_demand"]
            * LEAD_TIME_DAYS,
            result["on_hand_units"] <= result["reorder_point"],
        ],
        [
            "Unknown",
            "High",
            "Medium",
        ],
        default="Low",
    )

    numeric_columns = [
        "avg_daily_demand",
        "daily_demand_std",
        "safety_stock",
        "reorder_point",
        "target_stock_level",
        "recommended_order_qty",
    ]

    result[numeric_columns] = result[numeric_columns].round(2)

    return result.sort_values(
        ["inventory_status", "stockout_risk", "reorder_point"],
        ascending=[True, True, False],
    )


def build_summary(recommendations):
    """Create store-level inventory summary."""
    return (
        recommendations.groupby("store_id")
        .agg(
            product_count=("item_id", "count"),
            avg_daily_demand=("avg_daily_demand", "mean"),
            reorder_required=(
                "inventory_status",
                lambda x: (x == "Reorder Required").sum(),
            ),
            stock_data_missing=(
                "inventory_status",
                lambda x: (x == "Stock Data Required").sum(),
            ),
            high_stockout_risk=(
                "stockout_risk",
                lambda x: (x == "High").sum(),
            ),
        )
        .reset_index()
        .round(2)
    )


def run():
    """Run Day 25 inventory recommendation workflow."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sales, daily_columns = load_sales()
    metrics = build_demand_metrics(sales, daily_columns)
    stock = load_inventory(metrics)

    recommendations = calculate_recommendations(metrics, stock)
    summary = build_summary(recommendations)

    recommendations.to_csv(RECOMMENDATION_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nDAY 25 - INVENTORY REORDER RECOMMENDATIONS")
    print("------------------------------------------")
    print(f"Products analysed: {len(recommendations)}")
    print(f"Stores analysed: {recommendations['store_id'].nunique()}")
    print(f"History used: {len(daily_columns)} days")
    print(f"Lead time: {LEAD_TIME_DAYS} days")
    print(f"Review period: {REVIEW_PERIOD_DAYS} days")

    print("\nInventory status:")
    print(recommendations["inventory_status"].value_counts().to_string())

    print("\nStockout risk:")
    print(recommendations["stockout_risk"].value_counts().to_string())

    print("\nSample recommendations:")
    print(recommendations.head(10).to_string(index=False))

    print("\nReports saved:")
    print(RECOMMENDATION_FILE)
    print(SUMMARY_FILE)

    if recommendations["on_hand_units"].isna().any():
        print(
            "\nIMPORTANT: Complete data/raw/inventory_on_hand.csv "
            "with actual stock quantities and rerun for order quantities."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
