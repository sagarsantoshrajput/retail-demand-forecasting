\# Day 22: Forecast Performance Analysis



\## Objective 

Enhance forecast evaluation by introducing detailed performance metrics for Prophet and LightGBM.



\## Metrics Implemented

\- Mean Absolute Error (MAE)

\- Root Mean Squared Error (RMSE)

\- Mean Absolute Percentage Error (MAPE)

\- Weighted Mean Absolute Percentage Error (WMAPE)

\- Forecast Bias



\## Features

\- Prophet and LightGBM performance comparison

\- Individual store-item series evaluation

\- Forecast overprediction and underprediction analysis

\- JSON and CSV report generation



\## Files

\- src/forecast\_performance.py

\- tests/test\_forecast\_performance.py

\- reports/day22\_forecast\_performance.json

\- reports/day22\_model\_performance.csv

\- reports/day22\_series\_performance.csv



\## Execution

python -m src.forecast\_performance



\## Testing

python -m pytest -q



\## Result

All 16 project tests passed.

Performance reports generated successfully.

