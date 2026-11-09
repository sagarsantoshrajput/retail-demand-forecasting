
import pandas as pd
import pytest

from src.forecast_error_analysis import (
    classify_error,
    classify_bias,
    load_performance,
    build_diagnostics,
    build_recommendations,
)


def sample_data():
    return pd.DataFrame({
        "row_count": [30, 30, 30, 30],
        "mae": [2.0, 5.0, 3.0, 4.0],
        "rmse": [3.0, 7.0, 4.0, 6.0],
        "mape_pct": [15.0, 45.0, 25.0, 35.0],
        "wmape_pct": [15.0, 45.0, 25.0, 35.0],
        "forecast_bias": [-1.0, 2.0, -0.5, 1.0],
        "forecast_bias_pct": [-5.0, 10.0, -3.0, 4.0],
        "store_id": ["STORE_1", "STORE_1", "STORE_2", "STORE_2"],
        "item_id": ["ITEM_1", "ITEM_2", "ITEM_1", "ITEM_2"],
        "model": ["lightgbm", "prophet", "lightgbm", "prophet"],
    })


def test_classify_error():
    assert classify_error(15) == "Low"
    assert classify_error(25) == "Moderate"
    assert classify_error(45) == "High"


def test_classify_bias():
    assert classify_bias(2) == "Overforecasting"
    assert classify_bias(-2) == "Underforecasting"
    assert classify_bias(0) == "Balanced"


def test_build_diagnostics():
    result = build_diagnostics(sample_data())

    assert len(result["store"]) == 2
    assert len(result["product"]) == 4
    assert len(result["model"]) == 2

    assert result["store"].iloc[0]["avg_wmape_pct"] >= 0


def test_error_severity_counts():
    result = build_diagnostics(sample_data())

    severity = dict(zip(
        result["severity"]["error_severity"],
        result["severity"]["series_count"],
    ))

    assert severity["Low"] == 1
    assert severity["Moderate"] == 2
    assert severity["High"] == 1


def test_bias_direction_counts():
    result = build_diagnostics(sample_data())

    bias = dict(zip(
        result["bias"]["bias_direction"],
        result["bias"]["series_count"],
    ))

    assert bias["Underforecasting"] == 2
    assert bias["Overforecasting"] == 2


def test_recommendations():
    diagnostics = build_diagnostics(sample_data())
    recommendations = build_recommendations(diagnostics)

    assert len(recommendations) >= 3
    assert any("underforecasting" in r.lower() for r in recommendations)
    assert any("overforecasting" in r.lower() for r in recommendations)


def test_load_performance(tmp_path):
    path = tmp_path / "performance.csv"
    sample_data().to_csv(path, index=False)

    result = load_performance(path)

    assert len(result) == 4
    assert "wmape_pct" in result.columns


def test_load_performance_missing_column(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"store_id": ["STORE_1"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_performance(path)
