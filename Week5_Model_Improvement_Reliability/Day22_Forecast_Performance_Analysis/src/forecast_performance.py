
"""Day 22: Detailed forecast performance analysis for Prophet and LightGBM."""

import json
import logging

import numpy as np
import pandas as pd

from src import config
def load_actuals_holdout():
    """Load holdout actuals and map them to forecast dates."""
    wide = pd.read_csv(config.DATA_RAW / "actuals_holdout.csv")

    from src.warehouse import get_conn

    conn = get_conn()
    try:
        dates = conn.execute(
            "SELECT calendar_date FROM stg_calendar ORDER BY calendar_date "
            f"OFFSET {config.N_DAYS_HISTORY} "
            f"LIMIT {config.FORECAST_HORIZON_DAYS}"
        ).df()
    finally:
        conn.close()

    wide = wide.drop(columns=["d"]).reset_index(drop=True)
    wide["calendar_date"] = pd.to_datetime(
        dates["calendar_date"].values[:len(wide)]
    )

    long = wide.melt(
        id_vars="calendar_date",
        var_name="series",
        value_name="actual_units",
    )

    split = long["series"].str.extract(r"^(STORE_\d+)_(.+)$")
    long["store_id"] = split[0]
    long["item_id"] = split[1]

    return long[
        ["store_id", "item_id", "calendar_date", "actual_units"]
    ]
log = logging.getLogger(__name__)


def calculate_metrics(actual, predicted):
    """Calculate regression and forecast-bias metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError("No valid actual/predicted observations.")

    errors = predicted - actual
    absolute_errors = np.abs(errors)

    nonzero_actual = actual != 0
    mape = (
        float(np.mean(absolute_errors[nonzero_actual] /
                      np.abs(actual[nonzero_actual])) * 100)
        if nonzero_actual.any() else None
    )

    actual_total = float(np.abs(actual).sum())
    wmape = (
        float(absolute_errors.sum() / actual_total * 100)
        if actual_total > 0 else None
    )

    bias = float(errors.mean())
    bias_pct = (
        float(errors.sum() / actual.sum() * 100)
        if actual.sum() != 0 else None
    )

    return {
        "row_count": int(len(actual)),
        "mae": float(absolute_errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mape_pct": mape,
        "wmape_pct": wmape,
        "forecast_bias": bias,
        "forecast_bias_pct": bias_pct,
    }


def prepare_forecasts():
    """Load both model forecasts and standardize their date columns."""
    prophet_path = config.MODELS_DIR / "prophet_forecasts.parquet"
    lightgbm_path = config.MODELS_DIR / "lightgbm_forecasts.parquet"

    prophet = pd.read_parquet(prophet_path)
    lightgbm = pd.read_parquet(lightgbm_path)

    prophet = prophet.rename(columns={"ds": "calendar_date"})

    required = {"store_id", "item_id", "calendar_date", "yhat", "model"}

    for name, frame in [("Prophet", prophet), ("LightGBM", lightgbm)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} forecast is missing columns: {sorted(missing)}")

    columns = ["store_id", "item_id", "calendar_date", "yhat", "model"]

    combined = pd.concat(
        [prophet[columns], lightgbm[columns]],
        ignore_index=True,
    )

    combined["calendar_date"] = pd.to_datetime(
        combined["calendar_date"], errors="coerce"
    )
    combined["yhat"] = pd.to_numeric(combined["yhat"], errors="coerce")

    combined = combined.dropna(
        subset=["store_id", "item_id", "calendar_date", "yhat", "model"]
    )

    return combined


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    forecasts = prepare_forecasts()
    actuals = load_actuals_holdout()

    actuals["calendar_date"] = pd.to_datetime(actuals["calendar_date"])
    actuals["actual_units"] = pd.to_numeric(
        actuals["actual_units"], errors="coerce"
    )

    evaluated = forecasts.merge(
        actuals,
        on=["store_id", "item_id", "calendar_date"],
        how="inner",
    )

    evaluated = evaluated.dropna(subset=["actual_units"])

    if evaluated.empty:
        raise ValueError("No matching forecast and holdout actual rows found.")

    model_results = []

    for model, group in evaluated.groupby("model"):
        metrics = calculate_metrics(
            group["actual_units"],
            group["yhat"],
        )

        metrics["model"] = model
        metrics["series_count"] = int(
            group[["store_id", "item_id"]].drop_duplicates().shape[0]
        )
        model_results.append(metrics)

    overall = pd.DataFrame(model_results)

    series_results = []

    for (store_id, item_id, model), group in evaluated.groupby(
        ["store_id", "item_id", "model"]
    ):
        metrics = calculate_metrics(
            group["actual_units"],
            group["yhat"],
        )

        metrics.update({
            "store_id": store_id,
            "item_id": item_id,
            "model": model,
        })
        series_results.append(metrics)

    per_series = pd.DataFrame(series_results)

    # Lower WMAPE indicates better weighted forecast accuracy.
    ranking = overall.sort_values(
        "wmape_pct", na_position="last"
    ).reset_index(drop=True)

    champion = (
        ranking.iloc[0]["model"]
        if not ranking.empty and pd.notna(ranking.iloc[0]["wmape_pct"])
        else None
    )

    report = {
        "project": "Retail Demand Forecasting",
        "analysis": "Day 22 Forecast Performance Analysis",
        "evaluation_rows": int(len(evaluated)),
        "overall_champion_by_wmape": champion,
        "metric_definitions": {
            "mae": "Mean Absolute Error; lower is better.",
            "rmse": "Root Mean Squared Error; lower is better.",
            "mape_pct": "Mean Absolute Percentage Error; zero actuals excluded.",
            "wmape_pct": "Weighted Mean Absolute Percentage Error; lower is better.",
            "forecast_bias": "Mean predicted minus actual; positive means overprediction.",
            "forecast_bias_pct": "Total prediction error divided by total actuals, as percent.",
        },
        "models": overall.replace({np.nan: None}).to_dict(orient="records"),
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = config.REPORTS_DIR / "day22_forecast_performance.json"
    csv_path = config.REPORTS_DIR / "day22_model_performance.csv"
    series_path = config.REPORTS_DIR / "day22_series_performance.csv"

    json_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    overall.to_csv(csv_path, index=False)
    per_series.to_csv(series_path, index=False)

    for _, row in overall.iterrows():
        log.info(
            "%s | MAE=%.4f RMSE=%.4f WMAPE=%s Bias=%.4f",
            row["model"],
            row["mae"],
            row["rmse"],
            row["wmape_pct"],
            row["forecast_bias"],
        )

    log.info("Performance report saved to %s", json_path)
    log.info("Model comparison saved to %s", csv_path)
    log.info("Per-series results saved to %s", series_path)

    return overall, per_series, report


if __name__ == "__main__":
    run()
