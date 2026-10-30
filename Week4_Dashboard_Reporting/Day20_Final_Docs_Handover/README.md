# Retail Demand Forecasting & Inventory Optimization

Automated analytics platform for retail and supply chain teams. It analyzes historical sales,
promotions and seasonality to **predict future demand at the store and department level**, and
generates a **30-day forecast** inventory managers can use for restocking, with a **what-if price
simulator** to see how a price change would move volume.

**Stack:** Python · SQL · DuckDB (BigQuery/Snowflake-compatible SQL) · dbt · Facebook Prophet · LightGBM · Streamlit · Plotly

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export WAREHOUSE_PATH=warehouse/warehouse.duckdb     # Windows (PowerShell): $env:WAREHOUSE_PATH="warehouse/warehouse.duckdb"
mkdir -p ~/.dbt && cp dbt_profile.yml ~/.dbt/profiles.yml   # one-time dbt setup, see note below

python -m src.pipeline          # generate data -> load -> quality -> dbt -> Prophet -> LightGBM -> evaluate
pytest -q                       # 7 tests, all green
streamlit run streamlit_app/app.py     # http://localhost:8501
```
Run single stages: `python -m src.pipeline --steps prophet lightgbm`.

**dbt profile:** dbt needs `~/.dbt/profiles.yml`. This repo ships `dbt_profile.yml` at the root -
copy it to `~/.dbt/profiles.yml` (Windows: `%USERPROFILE%\.dbt\profiles.yml`) once. It reads the
`WAREHOUSE_PATH` env var, so no paths need editing.

## Dataset
The real M5 Forecasting dataset (Walmart, Kaggle) needs a Kaggle login to download
programmatically, so `src/generate_m5_style_data.py` generates a **synthetic dataset with the
identical M5 file layout** (`calendar.csv`, `sell_prices.csv`, `sales_train_validation.csv`):
3 stores x 3 categories x 2 departments x 8 items, ~3 years of daily history, with trend, weekly/
yearly seasonality, price elasticity and calendar events baked in, plus a held-out 30-day tail used
to score forecast accuracy.

**To use the real M5 data instead:** download the three CSVs from
[Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and drop them into
`data/raw/` with those exact names - every script reads those filenames, so nothing else changes.

## What's inside (mapped to the 4-week plan)
| Week | Deliverable | Where |
|---|---|---|
| 1 | BigQuery/Snowflake-style warehouse setup, extraction, data quality, date/price formatting | `src/generate_m5_style_data.py`, `src/extract_load.py`, `src/data_quality.py`, `reports/week1_baseline_report.md` |
| 2 | dbt configured and connected, staging/intermediate/marts, lineage & docs | `dbt_project/` (`dbt docs generate && dbt docs serve` for the full lineage graph) |
| 3 | Prophet (high-volume items), LightGBM (granular, multi-variable), forecasts stored back | `src/forecast_prophet.py`, `src/forecast_lightgbm.py`, `src/forecast_evaluate_and_store.py` |
| 4 | Streamlit app: store/category selection, 30-day forecast, what-if pricing, access control | `streamlit_app/app.py`, `.streamlit/secrets.toml.example` |

## Results (this repo's synthetic dataset)
| Model | Avg WMAPE (30-day holdout, lower is better) | Used for |
|---|---|---|
| **Facebook Prophet** | **27.8%** | High-volume items (top 25% by volume) - explicit seasonality + holidays |
| LightGBM | 42.0% | All items, especially the long tail - multi-variable, recursive forecast |

A per-series **champion model** is picked by holdout WMAPE and is what the Streamlit app and
`forecast_results` table show (see `reports/forecast_champion_by_series.csv`).
LightGBM's 5-fold grouped cross-validation MAE is about 5.0 units/day on one-step-ahead prediction;
its 30-day *recursive* (multi-step) forecast error is naturally higher, which is why Prophet wins on
the smooth, high-volume series while LightGBM still adds value on the noisier long tail, where its
price/calendar features matter more than pure seasonality.

## Streamlit app
- **Selectors:** category -> store -> item (populated from `mart_item_store_summary`).
- **30-day forecast chart:** last 90 days of actuals + the champion model's forecast + real holdout actuals overlaid.
- **What-if pricing:** a slider re-runs the LightGBM recursive forecast with `sell_price` scaled by
  the chosen %, using the price -> demand relationship the model learned from data (no hard-coded elasticity formula).
- **Access control:** disabled by default (runs out of the box). To enable, copy
  `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, set `enabled = true` and real
  passwords - a login gate then blocks the dashboard until signed in.

## Method notes & limitations
* This repo's dataset is **synthetic** (M5-schema-compatible, not real Walmart data). Swap in the
  real Kaggle CSVs before using this for real business decisions.
* The LightGBM 30-day forecast is **recursive**: each day's prediction feeds the next day's lag
  features, so errors can compound over the horizon - normal for iterative multi-step forecasting,
  and part of why the champion-per-series selection matters.
* The what-if price simulator reflects whatever price-demand relationship the model learned from
  this dataset's built-in elasticity; validate against real price experiments before using it to
  set prices.
* Verified: full pipeline (`python -m src.pipeline`) rebuilt clean end-to-end, 7 pytest tests green,
  14 dbt tests green, and the Streamlit app tested headlessly (base view, what-if slider, and both
  access-control paths) with zero exceptions.

## GitHub protocol
See `docs/GIT_WORKFLOW.md` (daily commits, branching, commit-message conventions).
