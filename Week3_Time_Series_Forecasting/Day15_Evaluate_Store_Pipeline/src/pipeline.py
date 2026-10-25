"""One-command orchestration for the whole 4-week pipeline:

    WAREHOUSE_PATH=warehouse/warehouse.duckdb python -m src.pipeline

Runs: generate data -> extract/load -> data quality -> dbt run/test ->
Prophet -> LightGBM -> evaluate & store. The Streamlit app (Week 4) is run
separately with `streamlit run streamlit_app/app.py` since it's a server, not
a batch step.
"""
import logging
import subprocess
import sys
import time
from pathlib import Path

from src import config, data_quality, extract_load, forecast_evaluate_and_store
from src import forecast_lightgbm, forecast_prophet, generate_m5_style_data

log = logging.getLogger("pipeline")


def run_dbt():
    dbt_dir = config.ROOT / "dbt_project"
    for args in (["dbt", "run"], ["dbt", "test"]):
        log.info("$ %s (in %s)", " ".join(args), dbt_dir)
        subprocess.run(args, cwd=dbt_dir, check=True)


STEPS = {
    "generate_data": generate_m5_style_data.run,
    "extract_load": extract_load.run,
    "data_quality": data_quality.run,
    "dbt": run_dbt,
    "prophet": forecast_prophet.run,
    "lightgbm": forecast_lightgbm.run,
    "evaluate_and_store": forecast_evaluate_and_store.run,
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    for name in args.steps:
        t0 = time.time()
        log.info("=== %s ===", name)
        STEPS[name]()
        log.info("=== %s done in %.1fs ===", name, time.time() - t0)


if __name__ == "__main__":
    main()
