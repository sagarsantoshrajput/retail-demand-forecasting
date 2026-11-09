
"""Day 24: Forecast error diagnostics by store, product and model."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

REPORT_DIR = Path(config.PROJECT_ROOT) / "reports" if hasattr(
    config, "PROJECT_ROOT"
) else Path(__file__).resolve().parents[1] / "reports"

INPUT_FILE = REPORT_DIR / "day22_series_performance.csv"


def load_performance(path=INPUT_FILE):
    """Load and validate Day 22 series-level performance results."""
    df = pd.read_csv(path)

    required = {
        "row_count", "mae", "rmse", "mape_pct", "wmape_pct",
        "forecast_bias", "forecast_bias_pct",
        "store_id", "item_id", "model",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    numeric = list(required - {"store_id", "item_id", "model"})
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    df = df.dropna(
        subset=["store_id", "item_id", "model", "wmape_pct", "forecast_bias"]
    ).copy()

    if df.empty:
        raise ValueError("No valid performance records found.")

    if (df["wmape_pct"] < 0).any():
        raise ValueError("WMAPE cannot be negative.")

    return df


def classify_error(wmape):
    """Classify series using WMAPE percentage thresholds."""
    if pd.isna(wmape):
        return "Unknown"
    if wmape < 20:
        return "Low"
    if wmape < 40:
        return "Moderate"
    return "High"


def classify_bias(bias):
    """Positive bias means overforecasting; negative means underforecasting."""
    if pd.isna(bias):
        return "Unknown"
    if bias > 0:
        return "Overforecasting"
    if bias < 0:
        return "Underforecasting"
    return "Balanced"


def build_diagnostics(df):
    """Create product, store, model and direction diagnostics."""

    df = df.copy()
    df["error_severity"] = df["wmape_pct"].apply(classify_error)
    df["bias_direction"] = df["forecast_bias"].apply(classify_bias)

    store_summary = (
        df.groupby("store_id")
        .agg(
            series_count=("item_id", "count"),
            avg_wmape_pct=("wmape_pct", "mean"),
            avg_mae=("mae", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_forecast_bias=("forecast_bias", "mean"),
        )
        .reset_index()
        .sort_values("avg_wmape_pct", ascending=False)
    )

    product_summary = (
        df.groupby(["store_id", "item_id"])
        .agg(
            model_count=("model", "nunique"),
            avg_wmape_pct=("wmape_pct", "mean"),
            worst_wmape_pct=("wmape_pct", "max"),
            avg_mae=("mae", "mean"),
            avg_forecast_bias=("forecast_bias", "mean"),
        )
        .reset_index()
        .sort_values("avg_wmape_pct", ascending=False)
    )

    model_summary = (
        df.groupby("model")
        .agg(
            series_count=("item_id", "count"),
            avg_wmape_pct=("wmape_pct", "mean"),
            avg_mae=("mae", "mean"),
            avg_rmse=("rmse", "mean"),
            avg_forecast_bias=("forecast_bias", "mean"),
        )
        .reset_index()
    )

    severity_summary = (
        df.groupby("error_severity")
        .size()
        .rename("series_count")
        .reset_index()
    )

    bias_summary = (
        df.groupby("bias_direction")
        .size()
        .rename("series_count")
        .reset_index()
    )

    return {
        "series": df,
        "store": store_summary,
        "product": product_summary,
        "model": model_summary,
        "severity": severity_summary,
        "bias": bias_summary,
    }


def build_recommendations(diagnostics):
    """Generate practical recommendations from aggregated diagnostics."""
    recommendations = []

    high_count = int(
        (diagnostics["series"]["error_severity"] == "High").sum()
    )
    under_count = int(
        (diagnostics["series"]["bias_direction"] == "Underforecasting").sum()
    )
    over_count = int(
        (diagnostics["series"]["bias_direction"] == "Overforecasting").sum()
    )

    if high_count:
        recommendations.append(
            f"{high_count} model-series records have high WMAPE (40% or above). "
            "Prioritize these records for model and feature investigation."
        )

    if under_count:
        recommendations.append(
            f"{under_count} model-series records show underforecasting. "
            "Review demand spikes, promotions, seasonality and lag features."
        )

    if over_count:
        recommendations.append(
            f"{over_count} model-series records show overforecasting. "
            "Review excess forecast patterns, event effects and recent demand trends."
        )

    if not recommendations:
        recommendations.append(
            "No high-error or directional-bias issues were detected "
            "in the available series-level report."
        )

    if not diagnostics["store"].empty:
        worst = diagnostics["store"].iloc[0]
        recommendations.append(
            f"Store {worst['store_id']} has the highest average WMAPE "
            f"({worst['avg_wmape_pct']:.2f}%). Review its product-level forecasts."
        )

    return recommendations


def run():
    """Generate Day 24 CSV and JSON diagnostic reports."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    performance = load_performance()
    diagnostics = build_diagnostics(performance)

    diagnostics["store"].to_csv(
        REPORT_DIR / "day24_store_error_analysis.csv", index=False
    )
    diagnostics["product"].to_csv(
        REPORT_DIR / "day24_product_error_analysis.csv", index=False
    )
    diagnostics["model"].to_csv(
        REPORT_DIR / "day24_model_error_analysis.csv", index=False
    )
    diagnostics["severity"].to_csv(
        REPORT_DIR / "day24_error_severity.csv", index=False
    )
    diagnostics["bias"].to_csv(
        REPORT_DIR / "day24_forecast_bias_analysis.csv", index=False
    )

    recommendations = build_recommendations(diagnostics)

    summary = {
        "source": INPUT_FILE.name,
        "records_analyzed": int(len(performance)),
        "unique_stores": int(performance["store_id"].nunique()),
        "unique_products": int(
            performance[["store_id", "item_id"]].drop_duplicates().shape[0]
        ),
        "models_analyzed": sorted(performance["model"].unique().tolist()),
        "highest_error_products": (
            diagnostics["product"].head(10).replace(
                {np.nan: None}
            ).to_dict(orient="records")
        ),
        "recommendations": recommendations,
    }

    with open(
        REPORT_DIR / "day24_forecast_error_analysis.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4, allow_nan=False)

    print("\nDAY 24 – FORECAST ERROR ANALYSIS")
    print("--------------------------------")
    print(f"Records analyzed: {len(performance)}")
    print(f"Unique stores: {summary['unique_stores']}")
    print(f"Unique products: {summary['unique_products']}")

    print("\nStore performance:")
    print(diagnostics["store"].to_string(index=False))

    print("\nTop 10 products by average WMAPE:")
    print(diagnostics["product"].head(10).to_string(index=False))

    print("\nError severity:")
    print(diagnostics["severity"].to_string(index=False))

    print("\nForecast bias:")
    print(diagnostics["bias"].to_string(index=False))

    print("\nRecommendations:")
    for recommendation in recommendations:
        print(f"- {recommendation}")

    print("\nReports saved in:", REPORT_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
