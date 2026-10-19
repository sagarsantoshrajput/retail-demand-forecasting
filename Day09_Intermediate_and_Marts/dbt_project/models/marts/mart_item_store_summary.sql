-- Week 2 (Day 4-6): one row per store-item - a clean data mart for BI /
-- Streamlit dropdowns and for picking which series go to Prophet vs LightGBM.
select
    store_id, item_id, dept_id, cat_id, state_id,
    count(*)                 as days_observed,
    sum(units_sold)          as total_units,
    avg(units_sold)          as avg_daily_units,
    stddev(units_sold)       as stddev_daily_units,
    sum(revenue)             as total_revenue,
    avg(sell_price)          as avg_sell_price
from {{ ref('mart_sales_daily') }}
group by store_id, item_id, dept_id, cat_id, state_id
