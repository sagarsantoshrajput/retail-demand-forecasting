-- Week 2 (Day 1-3): staging model - typed weekly prices.
select
    store_id,
    item_id,
    year_week_id,
    cast(sell_price as double) as sell_price
from (
    select store_id, item_id, wm_yr_wk as year_week_id, sell_price
    from {{ source('raw', 'sell_prices') }}
)
where sell_price is not null and sell_price > 0
