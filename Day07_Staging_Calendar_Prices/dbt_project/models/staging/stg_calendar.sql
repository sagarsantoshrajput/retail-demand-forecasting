-- Week 2 (Day 1-3): staging model - typed, renamed, one row per date.
select
    cast(date as date)         as calendar_date,
    wm_yr_wk                   as year_week_id,
    weekday                    as weekday_name,
    cast(wday as integer)      as weekday_num,
    cast(month as integer)     as month_num,
    cast(year as integer)      as year_num,
    d                          as day_id,
    event_name_1               as event_name,
    event_type_1                as event_type,
    snap_ca, snap_tx, snap_wi
from {{ source('raw', 'calendar') }}
