"""Week 3 (Day 1-3): feature engineering shared by both forecasting models -
lag features, rolling means, calendar features, and the high-volume /
long-tail split that decides which series go to Prophet vs. LightGBM.
"""
import numpy as np
import pandas as pd

from src import config
from src.warehouse import get_conn

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]


def load_daily() -> pd.DataFrame:
    conn = get_conn()
    df = conn.execute("SELECT * FROM mart_sales_daily ORDER BY store_id, item_id, calendar_date").df()
    conn.close()
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["calendar_date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["calendar_date"].dt.month
    df["is_event"] = df["event_name"].notna().astype(int)
    df["sell_price"] = df["sell_price"].fillna(df.groupby(["store_id", "item_id"])["sell_price"]
                                               .transform(lambda s: s.ffill().bfill()))
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag / rolling stats computed per store-item series, shifted by 1 day to avoid leakage."""
    df = df.sort_values(["store_id", "item_id", "calendar_date"]).copy()
    g = df.groupby(["store_id", "item_id"])["units_sold"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    for w in ROLLING_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.groupby([df["store_id"], df["item_id"]]).transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 2)).mean())
    return df


def split_high_volume_vs_long_tail(df: pd.DataFrame):
    """Top HIGH_VOLUME_QUANTILE of items by total volume -> Prophet, rest -> LightGBM."""
    totals = df.groupby(["store_id", "item_id"])["units_sold"].sum().reset_index(name="total_units")
    cutoff = totals["total_units"].quantile(config.HIGH_VOLUME_QUANTILE)
    high_volume_keys = set(map(tuple, totals.loc[totals.total_units >= cutoff, ["store_id", "item_id"]].values))
    is_hv = df.apply(lambda r: (r.store_id, r.item_id) in high_volume_keys, axis=1)
    return df[is_hv].copy(), df[~is_hv].copy(), cutoff


def build_feature_set() -> pd.DataFrame:
    df = load_daily()
    df = add_time_features(df)
    df = add_lag_and_rolling_features(df)
    return df


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    feats = build_feature_set()
    hv, lt, cutoff = split_high_volume_vs_long_tail(feats)
    logging.info("Built %d feature rows across %d series. High-volume cutoff=%.1f units -> "
                "%d high-volume rows (Prophet), %d long-tail rows (LightGBM)",
                len(feats), feats.groupby(["store_id", "item_id"]).ngroups, cutoff, len(hv), len(lt))
