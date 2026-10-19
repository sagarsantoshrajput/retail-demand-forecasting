-- Week 2 (Day 4-6): clean, analysis-ready daily grain data mart. This is what
-- the forecasting models (Week 3) and the Streamlit app (Week 4) read from.
select * from {{ ref('int_sales_enriched') }}
