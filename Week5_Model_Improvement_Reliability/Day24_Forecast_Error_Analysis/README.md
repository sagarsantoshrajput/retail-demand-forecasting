\# Day 24 – Forecast Error Analysis



\## Objective

Analyze forecast errors across stores, products and forecasting models to identify underperforming demand forecasts and generate improvement recommendations.



\## Implementation



1\. \*\*Store-Level Analysis\*\*

&#x20;  - Average WMAPE by store.

&#x20;  - MAE and RMSE comparisons.

&#x20;  - Average forecast bias.



2\. \*\*Product-Level Analysis\*\*

&#x20;  - Identify products with the highest average WMAPE.

&#x20;  - Compare forecast errors across products and stores.



3\. \*\*Error Severity Classification\*\*

&#x20;  - Low: WMAPE below 20%.

&#x20;  - Moderate: WMAPE from 20% to below 40%.

&#x20;  - High: WMAPE of 40% or above.



4\. \*\*Forecast Bias Analysis\*\*

&#x20;  - Overforecasting.

&#x20;  - Underforecasting.

&#x20;  - Balanced forecasts.



5\. \*\*Recommendations\*\*

&#x20;  - Identify high-error records.

&#x20;  - Highlight stores requiring investigation.

&#x20;  - Suggest areas for model and feature improvement.



\## Key Findings

\- 180 model-series records analyzed.

\- 144 unique store-product combinations.

\- 60 records classified as high error.

\- 95 records showed underforecasting.

\- 85 records showed overforecasting.

\- STORE\_3 recorded the highest average WMAPE at 40.52%.



\## Technologies

\- Python

\- Pandas

\- NumPy

\- Pytest



\## Execution



```bash

python -m src.forecast\_error\_analysis

```



\## Testing



```bash

python -m pytest tests/test\_forecast\_error\_analysis.py -v

python -m pytest -q

```



\*\*Test Results:\*\* 8 dedicated tests passed; 24 total project tests passed.



\## Files

\- `src/forecast\_error\_analysis.py`

\- `tests/test\_forecast\_error\_analysis.py`

\- `reports/` – JSON and CSV diagnostic outputs.



\## Outcome

Created a structured forecast error diagnostic process to identify high-error products, store-level performance differences and forecast bias patterns.

