import logging
from pathlib import Path

from biorag.ingest import load_and_clean, verify, write_parquet

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RAW_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).parents[2] / "data" / "processed"

if __name__ == "__main__":
    paths = sorted(RAW_DIR.glob("*.csv"))
    df = load_and_clean(paths)
    verify(df)
    write_parquet(df, PROCESSED_DIR / "reviews.parquet")
