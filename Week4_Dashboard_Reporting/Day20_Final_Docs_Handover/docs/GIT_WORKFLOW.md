# GitHub protocol (required by Zaalima)

1. Create the GitHub repository before starting development.
2. Push commits **daily** for every week of the project lifecycle - the commit history is the daily-productivity metric.
3. Use Conventional Commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`), logical branch management
   (`main` <- `develop` <- `feature/<topic>`) and tag weekly releases (`v0.1` ... `v1.0`).

## 20-workday plan (4 weeks x 5 days)
| Day | Week | Work | Commit message |
|---|---|---|---|
| 01 | 1 | Project skeleton, requirements, config | `chore: project skeleton and configuration` |
| 02 | 1 | M5-schema dataset generator + raw data | `feat(data): M5-schema-compatible dataset generator` |
| 03 | 1 | Warehouse connection + raw extract/load | `feat(warehouse): DuckDB connection and raw CSV extract/load` |
| 04 | 1 | Data quality checks | `feat(quality): raw-layer data quality checks` |
| 05 | 1 | Baseline report + first tests | `docs: week 1 baseline report; test: raw-layer tests` |
| 06 | 2 | dbt project, profile, sources | `chore(dbt): initialize dbt project and sources` |
| 07 | 2 | Staging: calendar, prices | `feat(dbt): staging models for calendar and prices` |
| 08 | 2 | Staging: sales (unpivot) + staging tests | `feat(dbt): staging model for daily sales` |
| 09 | 2 | Intermediate + marts | `feat(dbt): enriched join and daily/weekly/monthly marts` |
| 10 | 2 | Mart tests/docs, architecture, mart tests | `test(dbt): schema tests and lineage documentation` |
| 11 | 3 | Feature engineering | `feat(features): lag/rolling features, high-volume split` |
| 12 | 3 | Prophet | `feat(forecast): Facebook Prophet for high-volume items` |
| 13 | 3 | LightGBM training | `feat(forecast): LightGBM multi-variable training` |
| 14 | 3 | LightGBM recursive forecasting | `feat(forecast): 30-day recursive forecasting` |
| 15 | 3 | Evaluate, store forecasts, pipeline runner | `feat(warehouse): store forecast outputs; add pipeline runner` |
| 16 | 4 | Streamlit skeleton | `feat(app): Streamlit skeleton with selectors` |
| 17 | 4 | Forecast chart | `feat(app): 30-day demand forecast chart` |
| 18 | 4 | What-if scenario | `feat(app): what-if price scenario` |
| 19 | 4 | Access control + final tests | `feat(app): access control; test: what-if sanity test` |
| 20 | 4 | Final README + handover | `docs: final documentation and handover` |
