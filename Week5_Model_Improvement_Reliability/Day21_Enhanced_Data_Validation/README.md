\# Day 21: Enhanced Data Validation



\## Objective

Improve raw dataset reliability through automated validation.



\## Datasets Validated

\- calendar.csv

\- sell\_prices.csv

\- sales\_train\_validation.csv

\- actuals\_holdout.csv



\## Features

\- Missing and invalid value detection

\- Duplicate record detection

\- Non-positive price detection

\- Negative sales detection

\- Required column validation

\- Numeric data validation



\## Files

\- src/data\_validation.py

\- tests/test\_data\_validation.py

\- reports/day21\_data\_validation\_report.json



\## Execution

python -m src.data\_validation



\## Testing

python -m pytest tests/test\_data\_validation.py -q



\## Result

All four datasets passed validation.

All four automated tests passed.

