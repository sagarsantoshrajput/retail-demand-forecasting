
from pathlib import Path

import pandas as pd


REPORT_DIR = Path("reports")

ABC_FILE = REPORT_DIR / "day26_abc_inventory_analysis.csv"
INVENTORY_FILE = REPORT_DIR / "day25_inventory_recommendations.csv"

OUTPUT_FILE = REPORT_DIR / "day27_inventory_priority.csv"
SUMMARY_FILE = REPORT_DIR / "day27_inventory_priority_summary.csv"


ABC_PRIORITY = {
    "A": "High",
    "B": "Medium",
    "C": "Low",
}


def build_inventory_priority(abc_df, inventory_df):
    required_abc = {
        "store_id", "item_id", "total_sales_value",
        "abc_category", "rank"
    }

    required_inventory = {
        "store_id", "item_id", "on_hand_units",
        "inventory_status", "stockout_risk",
        "recommended_order_qty"
    }

    if not required_abc.issubset(abc_df.columns):
        raise ValueError("ABC report is missing required columns")

    if not required_inventory.issubset(inventory_df.columns):
        raise ValueError("Inventory report is missing required columns")

    if abc_df.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Duplicate store-product combinations in ABC data")

    if inventory_df.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Duplicate store-product combinations in inventory data")

    result = abc_df.merge(
        inventory_df,
        on=["store_id", "item_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if (result["_merge"] != "both").any():
        raise ValueError("Some ABC products have no inventory recommendation")

    result.drop(columns=["_merge"], inplace=True)

    result["abc_priority"] = result["abc_category"].map(ABC_PRIORITY)

    if result["abc_priority"].isna().any():
        raise ValueError("Invalid ABC category found")

    result["stock_data_available"] = result["on_hand_units"].notna()

    def determine_action(row):
        if not row["stock_data_available"]:
            return "DATA_REQUIRED"

        category = row["abc_category"]
        risk = str(row["stockout_risk"]).strip().lower()

        if risk == "high":
            return "CRITICAL" if category == "A" else "HIGH"

        if risk == "medium":
            return "HIGH" if category == "A" else "MEDIUM"

        if risk == "low":
            return "MEDIUM" if category == "A" else "LOW"

        return "REVIEW_REQUIRED"

    result["action_priority"] = result.apply(
        determine_action, axis=1
    )

    result["recommended_action"] = result.apply(
        lambda row: (
            "Update on-hand inventory data"
            if not row["stock_data_available"]
            else (
                "Review and replenish inventory"
                if row["action_priority"] in ["CRITICAL", "HIGH"]
                else "Monitor inventory levels"
            )
        ),
        axis=1,
    )

    result.sort_values(
        by=["action_priority", "rank"],
        ascending=[True, True],
        inplace=True,
    )

    return result.reset_index(drop=True)


def build_priority_summary(priority_df):
    summary = (
        priority_df.groupby(
            ["action_priority", "abc_category"],
            dropna=False
        )
        .agg(
            product_count=("item_id", "count"),
            total_sales_value=("total_sales_value", "sum"),
            stock_data_available=("stock_data_available", "sum"),
        )
        .reset_index()
    )

    summary["total_sales_value"] = summary[
        "total_sales_value"
    ].round(2)

    return summary


def main():
    print("\nDAY 27 - INVENTORY PRIORITIZATION")
    print("--------------------------------")

    abc_df = pd.read_csv(ABC_FILE)
    inventory_df = pd.read_csv(INVENTORY_FILE)

    priority_df = build_inventory_priority(
        abc_df, inventory_df
    )

    summary_df = build_priority_summary(priority_df)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    priority_df.to_csv(OUTPUT_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("Products analysed:", len(priority_df))
    print(
        "Products requiring stock data:",
        (~priority_df["stock_data_available"]).sum()
    )

    print("\nPriority Summary:")
    print(summary_df.to_string(index=False))

    print("\nTop 10 inventory priorities:")
    print(
        priority_df[
            [
                "store_id",
                "item_id",
                "abc_category",
                "abc_priority",
                "action_priority",
                "recommended_action",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nReports generated:")
    print(OUTPUT_FILE.resolve())
    print(SUMMARY_FILE.resolve())


if __name__ == "__main__":
    main()
