
import pandas as pd

from src.data_validation import (
    validate_calendar,
    validate_prices,
    validate_sales,
    validate_holdout,
)


def test_calendar_detects_invalid_and_duplicate_dates():
    df = pd.DataFrame({
        "date": ["2024-01-01", "invalid-date", "2024-01-01"],
        "wm_yr_wk": [1, 1, 1],
        "weekday": ["Monday", "Tuesday", "Monday"],
        "wday": [1, 2, 1],
        "month": [1, 1, 1],
        "year": [2024, 2024, 2024],
        "d": ["d_1", "d_2", "d_3"],
    })
    result = validate_calendar(df)
    assert result["invalid_dates"]["issues"] == 1
    assert result["duplicate_dates"]["issues"] == 1


def test_prices_detect_non_positive_values():
    df = pd.DataFrame({
        "store_id": ["S1", "S1"],
        "item_id": ["I1", "I2"],
        "wm_yr_wk": [1, 1],
        "sell_price": [10, 0],
    })
    result = validate_prices(df)
    assert result["non_positive_prices"]["issues"] == 1


def test_sales_detect_negative_units():
    df = pd.DataFrame({
        "id": ["A", "B"],
        "item_id": ["I1", "I2"],
        "dept_id": ["D1", "D1"],
        "cat_id": ["C1", "C1"],
        "store_id": ["S1", "S1"],
        "state_id": ["CA", "CA"],
        "d_1": [5, -2],
        "d_2": [3, 4],
    })
    result = validate_sales(df)
    assert result["negative_sales"]["issues"] == 1


def test_holdout_detects_negative_actuals():
    df = pd.DataFrame({
        "d": ["d_1", "d_2"],
        "STORE_1_FOODS_1_001": [5, -1],
    })
    result = validate_holdout(df)
    assert result["negative_actuals"]["issues"] == 1
