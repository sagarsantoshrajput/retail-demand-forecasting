-- Week 2 (Day 4-6): weekly aggregation, one row per store-item-week.
select
    store_id, item_id, dept_id, cat_id, state_id, year_week_id,
    min(calendar_date)              as week_start_date,
    sum(units_sold)                 as units_sold,
    sum(revenue)                    as revenue,
    avg(sell_price)                 as avg_sell_price,
    max(case when event_name is not null then 1 else 0 end) as had_event,
    max(snap_active)                as any_snap_day
from {{ ref('mart_sales_daily') }}
group by store_id, item_id, dept_id, cat_id, state_id, year_week_id
