"""Week 3 (Day 7): evaluate Prophet vs. LightGBM against the true holdout
actuals (WMAPE per series), pick a per-series champion, and write predicted-
vs-actual forecast outputs back into the warehouse for the dashboard.
"""
import logging

import numpy as np
import pandas as pd

from src import config
from src.warehouse import get_conn

log = logging.getLogger(__name__)


def load_actuals_holdout() -> pd.DataFrame:
    """actuals_holdout.csv is wide (one column per store_item); reshape to long."""
    wide = pd.read_csv(config.DATA_RAW / "actuals_holdout.csv")
    conn = get_conn()
    dates = conn.execute("SELECT calendar_date FROM stg_calendar ORDER BY calendar_date "
                        f"OFFSET {config.N_DAYS_HISTORY} LIMIT {config.FORECAST_HORIZON_DAYS}").df()
    conn.close()
    wide = wide.drop(columns=["d"]).reset_index(drop=True)
    wide["calendar_date"] = pd.to_datetime(dates["calendar_date"].values[: len(wide)])
    long = wide.melt(id_vars="calendar_date", var_name="series", value_name="actual_units")
    split = long["series"].str.extract(r"^(STORE_\d+)_(.+)$")
    long["store_id"], long["item_id"] = split[0], split[1]
    return long[["store_id", "item_id", "calendar_date", "actual_units"]]


def wmape(actual: pd.Series, pred: pd.Series) -> float:
    return float((actual - pred).abs().sum() / max(actual.abs().sum(), 1e-9))


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    prophet_fc = pd.read_parquet(config.MODELS_DIR / "prophet_forecasts.parquet")
    lgbm_fc = pd.read_parquet(config.MODELS_DIR / "lightgbm_forecasts.parquet")
    actuals = load_actuals_holdout()

    prophet_fc["ds"] = pd.to_datetime(prophet_fc["ds"])
    lgbm_fc["calendar_date"] = pd.to_datetime(lgbm_fc["calendar_date"])
    combined = pd.concat([
        prophet_fc.rename(columns={"ds": "calendar_date"})[["store_id", "item_id", "calendar_date", "yhat", "model"]],
        lgbm_fc[["store_id", "item_id", "calendar_date", "yhat", "model"]],
    ], ignore_index=True)
    combined = combined.merge(actuals, on=["store_id", "item_id", "calendar_date"], how="left")
    combined = combined.dropna(subset=["actual_units"])

    per_series_model = (combined.groupby(["store_id", "item_id", "model"])
                        .apply(lambda g: wmape(g.actual_units, g.yhat), include_groups=False)
                        .reset_index(name="wmape"))
    champion = per_series_model.loc[per_series_model.groupby(["store_id", "item_id"])["wmape"].idxmin()]
    champion = champion.rename(columns={"model": "champion_model", "wmape": "champion_wmape"})

    overall = (per_series_model.groupby("model")["wmape"].mean().reset_index()
              .rename(columns={"wmape": "avg_wmape"}))
    log.info("Average WMAPE by model (lower is better): %s", overall.to_dict("records"))

    final = combined.merge(champion[["store_id", "item_id", "champion_model"]], on=["store_id", "item_id"])
    final = final[final["model"] == final["champion_model"]].drop(columns=["champion_model"])
    final = final.rename(columns={"yhat": "predicted_units"})

    conn = get_conn()
    conn.register("forecast_df", final)
    conn.execute("CREATE OR REPLACE TABLE forecast_results AS SELECT * FROM forecast_df")
    conn.register("champion_df", champion)
    conn.execute("CREATE OR REPLACE TABLE forecast_model_selection AS SELECT * FROM champion_df")
    conn.register("overall_df", overall)
    conn.execute("CREATE OR REPLACE TABLE forecast_model_accuracy AS SELECT * FROM overall_df")
    conn.close()

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    overall.to_csv(config.REPORTS_DIR / "forecast_accuracy_by_model.csv", index=False)
    champion.to_csv(config.REPORTS_DIR / "forecast_champion_by_series.csv", index=False)
    log.info("Stored forecast_results (%d rows), forecast_model_selection (%d series), "
            "forecast_model_accuracy in the warehouse.", len(final), len(champion))
    return overall, champion


if __name__ == "__main__":
    run()
