\# Day 25 – Inventory Reorder Recommendations



\## Objective

Develop an inventory recommendation module using historical sales data to calculate replenishment requirements.



\## Implementation

\- Analysed the latest 28 days of sales history.

\- Calculated average daily demand for each store-product combination.

\- Calculated demand variability and safety stock.

\- Calculated reorder points and target stock levels.

\- Classified inventory status and stockout risk.

\- Generated store-level inventory summaries.



\## Inventory Parameters

\- Lead time: 7 days

\- Review period: 7 days

\- Service factor: 1.65



\## Output Reports

\- `day25\_inventory\_recommendations.csv`

\- `day25\_inventory\_summary.csv`



\## Inventory Data Requirement

The project creates `inventory\_on\_hand.csv` as an input template. Actual stock quantities must be entered before calculating actionable reorder quantities.



\## Outcome

Successfully analysed 144 store-product combinations across 3 stores and generated inventory recommendation reports.

