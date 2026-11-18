
import pandas as pd
import pytest

from src.inventory_priority import (
    build_inventory_priority,
    build_priority_summary,
)


def sample_data():
    abc = pd.DataFrame({
        "store_id": ["S1", "S1", "S1"],
        "item_id": ["P1", "P2", "P3"],
        "total_sales_value": [800, 150, 50],
        "abc_category": ["A", "B", "C"],
        "rank": [1, 2, 3],
    })

    inventory = pd.DataFrame({
        "store_id": ["S1", "S1", "S1"],
        "item_id": ["P1", "P2", "P3"],
        "on_hand_units": [None, 20, 100],
        "inventory_status": [
            "Stock Data Required",
            "Reorder Required",
            "Healthy",
        ],
        "stockout_risk": ["Unknown", "Medium", "Low"],
        "recommended_order_qty": [None, 50, 0],
    })

    return abc, inventory

def test_priority_analysis():
    abc, inventory = sample_data()
    result = build_inventory_priority(abc, inventory)

    assert len(result) == 3

    priorities = dict(
        zip(result["item_id"], result["abc_priority"])
    )

    assert priorities == {
        "P1": "High",
        "P2": "Medium",
        "P3": "Low",
    }

def test_missing_stock_data_flagged():
    abc, inventory = sample_data()
    result = build_inventory_priority(abc, inventory)

    product = result[result["item_id"] == "P1"].iloc[0]

    assert product["action_priority"] == "DATA_REQUIRED"
    assert product["recommended_action"] == (
        "Update on-hand inventory data"
    )


def test_high_risk_a_category_is_critical():
    abc, inventory = sample_data()
    inventory.loc[0, "on_hand_units"] = 5
    inventory.loc[0, "stockout_risk"] = "High"

    result = build_inventory_priority(abc, inventory)
    product = result[result["item_id"] == "P1"].iloc[0]

    assert product["action_priority"] == "CRITICAL"


def test_priority_summary():
    abc, inventory = sample_data()
    result = build_inventory_priority(abc, inventory)
    summary = build_priority_summary(result)

    assert summary["product_count"].sum() == 3
    assert summary["total_sales_value"].sum() == 1000


def test_missing_inventory_match_rejected():
    abc, inventory = sample_data()
    inventory = inventory[inventory["item_id"] != "P3"]

    with pytest.raises(ValueError):
        build_inventory_priority(abc, inventory)
