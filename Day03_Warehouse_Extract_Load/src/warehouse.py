"""Thin DuckDB connection helper (the local BigQuery/Snowflake stand-in)."""
import duckdb

from src import config


def get_conn():
    return duckdb.connect(config.WAREHOUSE_PATH)


def run_sql_file(path, conn=None):
    """Execute a .sql file that may contain several ';'-separated statements."""
    own = conn is None
    conn = conn or get_conn()
    stmts = [s.strip() for s in path.read_text().split(";") if s.strip() and not s.strip().startswith("--")]
    for s in stmts:
        conn.execute(s)
    if own:
        conn.close()
