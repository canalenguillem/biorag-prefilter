import html
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def load_and_clean(paths: list[Path]) -> pl.DataFrame:
    """Read, concatenate, and clean the raw CSV files."""
    frames = [pl.scan_csv(p) for p in paths]
    lf = pl.concat(frames)
    lf = lf.with_columns([
        pl.col("date").str.to_date(format="%d-%b-%y"),
        pl.col("review").map_elements(html.unescape, return_dtype=pl.String)

    ])
    return lf.collect()


def write_parquet(df: pl.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    logger.info("wrote %d rows to %s", len(df), dest)


def verify(df: pl.DataFrame) -> None:
    logger.info("schema:\n%s", df.schema)
    logger.info("row count: %d", len(df))
