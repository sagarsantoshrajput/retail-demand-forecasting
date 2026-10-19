-- Week 2 (Day 4-6): join daily sales with calendar and price context.
-- This is the row-level "enriched" table other marts build on.
select
    s.series_id,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    c.calendar_date,
    c.year_week_id,
    c.year_num,
    c.month_num,
    c.weekday_num,
    c.weekday_name,
    c.event_name,
    c.event_type,
    case s.state_id when 'CA' then c.snap_ca when 'TX' then c.snap_tx when 'WI' then c.snap_wi end as snap_active,
    s.units_sold,
    p.sell_price,
    s.units_sold * p.sell_price as revenue
from {{ ref('stg_sales') }} s
join {{ ref('stg_calendar') }} c on c.day_id = s.day_id
left join {{ ref('stg_sell_prices') }} p
    on p.store_id = s.store_id and p.item_id = s.item_id and p.year_week_id = c.year_week_id
