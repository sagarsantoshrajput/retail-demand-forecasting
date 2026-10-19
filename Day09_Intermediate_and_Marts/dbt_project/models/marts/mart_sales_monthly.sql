-- Week 2 (Day 4-6): monthly aggregation, one row per store-item-month.
select
    store_id, item_id, dept_id, cat_id, state_id, year_num, month_num,
    sum(units_sold)  as units_sold,
    sum(revenue)     as revenue,
    avg(sell_price)  as avg_sell_price
from {{ ref('mart_sales_daily') }}
group by store_id, item_id, dept_id, cat_id, state_id, year_num, month_num
