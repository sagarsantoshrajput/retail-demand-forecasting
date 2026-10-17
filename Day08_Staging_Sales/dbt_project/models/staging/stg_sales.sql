-- Week 2 (Day 1-3): staging model - unpivot the wide d_1..d_N sales matrix
-- (one row per store-item, one column per day, as M5 ships it) into one row
-- per (store, item, day), which is the grain every downstream model expects.
select
    id           as series_id,
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    day_id,
    cast(units_sold as integer) as units_sold
from {{ source('raw', 'sales_train_validation') }}
unpivot (units_sold for day_id in (columns('d_\d+')))
