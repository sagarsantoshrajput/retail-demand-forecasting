"""Week 3 (Day 1-3): Facebook Prophet for HIGH-VOLUME store-item series -
models trend, weekly/yearly seasonality and calendar events/holidays
explicitly, which is where Prophet earns its keep over a generic regressor.
"""
import logging
import warnings

import joblib
import pandas as pd
from prophet import Prophet

from src import config
from src.forecast_features import add_time_features, load_daily, split_high_volume_vs_long_tail

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", module="prophet")
warnings.filterwarnings("ignore", module="cmdstanpy")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def _events_as_holidays(df: pd.DataFrame) -> pd.DataFrame:
    ev = df[["calendar_date", "event_name"]].dropna().drop_duplicates()
    ev = ev.rename(columns={"calendar_date": "ds", "event_name": "holiday"})
    return ev


def fit_one_series(hist: pd.DataFrame, holidays: pd.DataFrame, horizon: int):
    """hist: columns [calendar_date, units_sold] for a single store-item, sorted ascending."""
    d = hist.rename(columns={"calendar_date": "ds", "units_sold": "y"})[["ds", "y"]]
    m = Prophet(holidays=holidays, weekly_seasonality=True, yearly_seasonality=True,
               daily_seasonality=False, seasonality_mode="multiplicative",
               interval_width=0.8)
    m.fit(d)
    future = m.make_future_dataframe(periods=horizon, include_history=False)
    fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    fc["yhat"] = fc["yhat"].clip(lower=0)
    return m, fc


def run(horizon: int = config.FORECAST_HORIZON_DAYS):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = add_time_features(load_daily())
    hv, _, cutoff = split_high_volume_vs_long_tail(df)
    holidays = _events_as_holidays(df)
    log.info("Prophet: %d high-volume series (volume cutoff=%.0f units)",
             hv.groupby(["store_id", "item_id"]).ngroups, cutoff)

    forecasts, models = [], {}
    for (store, item), g in hv.groupby(["store_id", "item_id"]):
        g = g.sort_values("calendar_date")
        _, fc = fit_one_series(g[["calendar_date", "units_sold"]], holidays, horizon)
        fc.insert(0, "item_id", item)
        fc.insert(0, "store_id", store)
        fc["model"] = "prophet"
        forecasts.append(fc)
    out = pd.concat(forecasts, ignore_index=True)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.MODELS_DIR / "prophet_forecasts.parquet", index=False)
    log.info("Prophet forecasts written for %d series x %d days -> models_store/prophet_forecasts.parquet",
             out.groupby(["store_id", "item_id"]).ngroups, horizon)
    return out


if __name__ == "__main__":
    run()
