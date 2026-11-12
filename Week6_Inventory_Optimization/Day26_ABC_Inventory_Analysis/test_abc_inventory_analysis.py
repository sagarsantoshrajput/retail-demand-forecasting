import pandas as pd
import pytest

from src.abc_inventory_analysis import (
    classify_abc,
    build_category_summary,
)


def sample_products():
    return pd.DataFrame({
        "store_id": ["S1", "S1", "S1", "S1"],
        "item_id": ["P1", "P2", "P3", "P4"],
        "total_units_sold": [100, 50, 25, 10],
        "total_sales_value": [800, 150, 40, 10],
        "priced_sales_records": [28, 28, 28, 28],
    })


def test_abc_classification():
    result = classify_abc(sample_products())
    assert len(result) == 4
    assert result.iloc[0]["abc_category"] == "A"
    assert result.iloc[-1]["abc_category"] == "C"


def test_sales_contribution_totals_100_percent():
    result = classify_abc(sample_products())
    assert result["sales_value_share_pct"].sum() == pytest.approx(100)


def test_cumulative_sales_share():
    result = classify_abc(sample_products())
    assert result["cumulative_sales_share_pct"].iloc[-1] == pytest.approx(100)


def test_category_summary():
    result = classify_abc(sample_products())
    summary = build_category_summary(result)

    assert set(summary["abc_category"]) == {"A", "B", "C"}
    assert summary["product_count"].sum() == 4
    assert summary["total_sales_value"].sum() == 1000


def test_zero_sales_value_rejected():
    products = sample_products()
    products["total_sales_value"] = 0

    with pytest.raises(ValueError):
        classify_abc(products)