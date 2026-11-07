
import pytest

from src.forecast_performance import calculate_metrics


def test_metrics_for_perfect_forecast():
    result = calculate_metrics([10, 20, 30], [10, 20, 30])

    assert result["mae"] == 0
    assert result["rmse"] == 0
    assert result["mape_pct"] == 0
    assert result["wmape_pct"] == 0
    assert result["forecast_bias"] == 0


def test_metrics_for_overprediction():
    result = calculate_metrics([10, 20], [12, 24])

    assert result["mae"] == 3
    assert result["forecast_bias"] == 3
    assert result["forecast_bias"] > 0


def test_metrics_for_underprediction():
    result = calculate_metrics([10, 20], [8, 16])

    assert result["forecast_bias"] == -3
    assert result["forecast_bias"] < 0


def test_mape_excludes_zero_actuals():
    result = calculate_metrics([0, 10], [5, 12])

    assert result["mape_pct"] == pytest.approx(20)
    assert result["row_count"] == 2


def test_empty_valid_observations_raise_error():
    with pytest.raises(ValueError):
        calculate_metrics([float("nan")], [10])
