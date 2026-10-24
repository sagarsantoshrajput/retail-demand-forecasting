"""Week 3 (Day 4-6): LightGBM for complex, multi-variable forecasting at the
granular item level - trained across ALL store-item series jointly (store,
item, category, price and calendar as features), then used to iteratively
forecast the next N days for every series, including the long-tail ones
Prophet doesn't cover individually.
"""
import logging

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src import config
from src.forecast_features import LAGS, ROLLING_WINDOWS, add_time_features, add_lag_and_rolling_features, load_daily

log = logging.getLogger(__name__)

CAT_FEATURES = ["store_id", "item_id", "dept_id", "cat_id", "state_id"]
NUM_FEATURES = ["dow", "is_weekend", "month", "is_event", "sell_price"] + \
              [f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


def _prep_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CAT_FEATURES:
        df[c] = df[c].astype("category")
    return df


def train(df: pd.DataFrame) -> LGBMRegressor:
    train_df = df.dropna(subset=[f"lag_{max(LAGS)}"])  # need full lag history
    X, y = _prep_categoricals(train_df[ALL_FEATURES]), train_df["units_sold"]

    gkf = GroupKFold(n_splits=5)
    groups = train_df["store_id"].astype(str) + "_" + train_df["item_id"].astype(str)
    maes = []
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        m = LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05, min_child_samples=20,
                          subsample=0.8, colsample_bytree=0.8, random_state=config.RANDOM_STATE, verbosity=-1)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx], categorical_feature=CAT_FEATURES)
        pred = np.clip(m.predict(X.iloc[va_idx]), 0, None)
        maes.append(mean_absolute_error(y.iloc[va_idx], pred))
    log.info("LightGBM 5-fold grouped-CV MAE (units/day): %.3f (+/- %.3f)", np.mean(maes), np.std(maes))

    final = LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05, min_child_samples=20,
                          subsample=0.8, colsample_bytree=0.8, random_state=config.RANDOM_STATE, verbosity=-1)
    final.fit(X, y, categorical_feature=CAT_FEATURES)
    return final, float(np.mean(maes))


def forecast_forward(model: LGBMRegressor, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Iterative (recursive) multi-step forecast: predict day t+1, append it to the
    history, recompute lag/rolling features, predict t+2, and so on."""
    hist = df.copy()
    last_date = hist["calendar_date"].max()
    last_prices = hist.sort_values("calendar_date").groupby(["store_id", "item_id"])["sell_price"].last()
    last_event_free = hist[["store_id", "item_id"]].drop_duplicates()
    out_rows = []

    for step in range(1, horizon + 1):
        cur_date = last_date + pd.Timedelta(days=step)
        new_rows = last_event_free.copy()
        new_rows["calendar_date"] = cur_date
        new_rows["dow"] = cur_date.dayofweek
        new_rows["is_weekend"] = int(cur_date.dayofweek >= 5)
        new_rows["month"] = cur_date.month
        new_rows["is_event"] = 0  # no known future promos/holidays in this simple recursive setup
        new_rows["sell_price"] = new_rows.apply(lambda r: last_prices.get((r.store_id, r.item_id), np.nan), axis=1)
        new_rows["units_sold"] = np.nan
        hist = pd.concat([hist, new_rows], ignore_index=True)
        hist = add_lag_and_rolling_features(hist)

        mask = hist["calendar_date"] == cur_date
        X = _prep_categoricals(hist.loc[mask, ALL_FEATURES])
        pred = np.clip(model.predict(X), 0, None)
        hist.loc[mask, "units_sold"] = pred
        out_rows.append(hist.loc[mask, ["store_id", "item_id", "calendar_date", "units_sold"]])

    return pd.concat(out_rows, ignore_index=True).rename(columns={"units_sold": "yhat"})


def run(horizon: int = config.FORECAST_HORIZON_DAYS):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = add_time_features(load_daily())
    df = add_lag_and_rolling_features(df)

    model, cv_mae = train(df)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": ALL_FEATURES, "cv_mae": cv_mae}, config.MODELS_DIR / "lightgbm_model.joblib")

    fc = forecast_forward(model, df, horizon)
    fc["model"] = "lightgbm"
    fc.to_parquet(config.MODELS_DIR / "lightgbm_forecasts.parquet", index=False)
    log.info("LightGBM forecasts written for %d series x %d days -> models_store/lightgbm_forecasts.parquet",
             fc.groupby(["store_id", "item_id"]).ngroups, horizon)
    return fc, cv_mae


if __name__ == "__main__":
    run()
