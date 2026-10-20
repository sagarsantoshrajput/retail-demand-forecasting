# Architecture

```
M5-schema CSVs (data/raw/)
        |
        v
DuckDB "raw" schema  (src/extract_load.py)  <- stands in for BigQuery/Snowflake
        |
        v
dbt: staging -> intermediate -> marts   (dbt_project/)
        |
        +--> mart_sales_daily / weekly / monthly / item_store_summary
        |
        +---------------------+
        v                     v
Facebook Prophet        LightGBM (multi-variable,
(high-volume items,     all series, lag/rolling/
seasonality+holidays)   price/calendar features)
        |                     |
        +----------+----------+
                   v
   Per-series champion picked by holdout WMAPE
   -> forecast_results / forecast_model_selection / forecast_model_accuracy (DuckDB)
                   |
                   v
        Streamlit app (streamlit_app/app.py)
   store/category/item selectors, 30-day forecast chart, what-if price slider
```

## Swapping DuckDB for a real BigQuery/Snowflake warehouse
1. `pip install dbt-bigquery` (or `dbt-snowflake`).
2. Add a new target to `~/.dbt/profiles.yml` with your project/dataset (or account/warehouse) credentials.
3. Point `src/warehouse.py`'s connection helper at that target's client instead of `duckdb.connect`, OR
   run the src ingestion scripts against BigQuery via `pandas.to_gbq` / the Snowflake connector.
4. `dbt run --target <new_target>` - no model SQL changes needed; it's ANSI-SQL compatible.
