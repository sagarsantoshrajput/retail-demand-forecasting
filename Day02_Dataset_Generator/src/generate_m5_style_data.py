"""Week 1 (Day 1-3): produce an M5-schema-compatible dataset.

The real M5 Forecasting dataset (Walmart, Kaggle) requires a Kaggle login to
download programmatically. This script generates a synthetic dataset with the
IDENTICAL file layout and column names as M5
(calendar.csv, sell_prices.csv, sales_train_validation.csv), scaled down to
3 stores x 3 categories x 2 departments x 8 items, ~3 years of daily history,
with trend, weekly seasonality, yearly seasonality, price effects and calendar
events baked in.

To use the REAL M5 data instead: download `calendar.csv`, `sell_prices.csv`
and `sales_train_validation.csv` from Kaggle
(https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and drop
them into data/raw/ with those exact names - every downstream script reads
those three filenames, so nothing else needs to change.
"""
import logging

import numpy as np
import pandas as pd

from src import config

log = logging.getLogger(__name__)
rng = np.random.default_rng(config.RANDOM_STATE)

STORES = [f"STORE_{i+1}" for i in range(config.N_STORES)]
STATES = ["CA", "TX", "WI"][: config.N_STORES] or ["CA"] * config.N_STORES
CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"][: config.N_CATEGORIES]
EVENTS = [("SuperBowl", "Sporting"), ("ValentinesDay", "Cultural"), ("Easter", "Religious"),
          ("Mother's day", "Cultural"), ("IndependenceDay", "National"), ("LaborDay", "National"),
          ("Halloween", "Cultural"), ("Thanksgiving", "National"), ("Christmas", "Religious")]


def build_calendar(start="2022-01-01", n_days=config.N_DAYS_HISTORY + config.FORECAST_HORIZON_DAYS):
    dates = pd.date_range(start, periods=n_days, freq="D")
    df = pd.DataFrame({"date": dates})
    df["wm_yr_wk"] = df["date"].dt.strftime("%y").astype(int) * 100 + df["date"].dt.isocalendar().week.astype(int)
    df["weekday"] = df["date"].dt.day_name()
    df["wday"] = df["date"].dt.dayofweek.map(lambda d: (d + 2) % 7 + 1)  # M5's 1=Sat convention
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["d"] = [f"d_{i+1}" for i in range(len(df))]
    df["event_name_1"], df["event_type_1"] = None, None
    for name, typ in EVENTS:
        anchor = rng.choice(df.index[365:-30])  # sprinkle each event once per ~year, simplistic
        for yr_offset in range(0, (n_days // 365) + 1):
            idx = anchor - 365 * yr_offset
            if idx >= 0:
                df.loc[idx, ["event_name_1", "event_type_1"]] = [name, typ]
    for s, st in zip(STORES, STATES):
        df[f"snap_{st}"] = rng.integers(0, 2, len(df))
    return df


def build_items():
    rows = []
    for cat in CATEGORIES:
        for dept_i in range(config.N_DEPTS_PER_CAT):
            dept = f"{cat}_{dept_i + 1}"
            for item_i in range(config.N_ITEMS_PER_DEPT):
                item_id = f"{dept}_{item_i + 1:03d}"
                base_price = round(rng.uniform(2, 60), 2)
                base_demand = rng.lognormal(mean=2.0, sigma=1.1)  # heavy-tailed like real retail
                rows.append(dict(item_id=item_id, dept_id=dept, cat_id=cat, base_price=base_price,
                                 base_demand=base_demand))
    return pd.DataFrame(rows)


def build_prices(items: pd.DataFrame, calendar: pd.DataFrame):
    weeks = calendar[["wm_yr_wk"]].drop_duplicates().reset_index(drop=True)
    rows = []
    for store in STORES:
        for _, it in items.iterrows():
            price = it.base_price
            for _, wk in weeks.iterrows():
                if rng.random() < 0.04:  # occasional price change / promo
                    price = round(it.base_price * rng.uniform(0.75, 1.15), 2)
                rows.append((store, it.item_id, wk.wm_yr_wk, price))
    return pd.DataFrame(rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])


def simulate_sales(items: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
    n = len(calendar)
    t = np.arange(n)
    dow = calendar["date"].dt.dayofweek.to_numpy()
    doy = calendar["date"].dt.dayofyear.to_numpy()
    weekly = 1 + 0.25 * np.sin(2 * np.pi * (dow - 3) / 7) + 0.15 * (dow >= 5)
    yearly = 1 + 0.20 * np.sin(2 * np.pi * doy / 365.25 - 1.2)
    trend = 1 + 0.00025 * t
    event_bump = np.where(calendar["event_name_1"].notna(), rng.uniform(1.1, 1.6, n), 1.0)
    price_wk = prices.set_index(["store_id", "item_id", "wm_yr_wk"]).sell_price
    cols = []
    for store in STORES:
        for _, it in items.iterrows():
            noise = rng.normal(1, 0.25, n).clip(0.1)
            p = calendar["wm_yr_wk"].map(lambda w: price_wk.get((store, it.item_id, w), it.base_price))
            price_effect = (it.base_price / p.to_numpy()) ** 1.3  # simple elasticity: cheaper -> more sales
            demand = it.base_demand * trend * weekly * yearly * event_bump * noise * price_effect
            units = rng.poisson(np.clip(demand, 0, None))
            cols.append(pd.Series(units, name=f"{store}_{it.item_id}"))
    wide = pd.concat(cols, axis=1)
    wide.insert(0, "d", calendar["d"].values)
    return wide


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    calendar = build_calendar()
    items = build_items()
    prices = build_prices(items, calendar)
    sales_wide = simulate_sales(items, calendar, prices)

    # reshape wide daily sales into M5's sales_train_validation layout: one row per store-item, d_1..d_N columns
    hist = sales_wide.iloc[: config.N_DAYS_HISTORY]
    long_rows = []
    for store in STORES:
        for _, it in items.iterrows():
            col = f"{store}_{it.item_id}"
            row = {"id": f"{it.item_id}_{store}_validation", "item_id": it.item_id, "dept_id": it.dept_id,
                  "cat_id": it.cat_id, "store_id": store, "state_id": dict(zip(STORES, STATES))[store]}
            row.update(dict(zip(hist["d"], hist[col])))
            long_rows.append(row)
    sales_train = pd.DataFrame(long_rows)

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    calendar.to_csv(config.DATA_RAW / "calendar.csv", index=False)
    prices.to_csv(config.DATA_RAW / "sell_prices.csv", index=False)
    sales_train.to_csv(config.DATA_RAW / "sales_train_validation.csv", index=False)
    # keep the "future" tail (actuals beyond training) to score forecast accuracy against, out-of-band
    future = sales_wide.iloc[config.N_DAYS_HISTORY:]
    future.to_csv(config.DATA_RAW / "actuals_holdout.csv", index=False)

    log.info("Generated %d store-item series over %d history days (+%d holdout) -> data/raw/",
             len(sales_train), config.N_DAYS_HISTORY, len(future))
    return sales_train, calendar, prices


if __name__ == "__main__":
    run()
