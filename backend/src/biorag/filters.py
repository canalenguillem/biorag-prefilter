import logging
from datetime import date, timedelta

import polars as pl
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Filters(BaseModel):
    drug_names: list[str] | None = None
    conditions: list[str] | None = None
    rating_min: int | None = None
    rating_max: int | None = None
    date_from: date | None = None
    date_to: date | None = None
    rating_drop_threshold: float | None = None
    rating_drop_window_months: int = 12


def apply_drug_filter(lf: pl.LazyFrame, filters: Filters) -> pl.LazyFrame:
    if filters.drug_names is None:
        return lf
    return lf.filter(pl.col("drugName").is_in(filters.drug_names))



def apply_condition_filter(lf: pl.LazyFrame, filters: Filters) -> pl.LazyFrame:
    if filters.conditions is None:
        return lf
    return lf.filter(pl.col("condition").is_in(filters.conditions))



def apply_rating_filter(lf: pl.LazyFrame, filters: Filters) -> pl.LazyFrame:
    if filters.rating_min is not None:
        lf = lf.filter(pl.col("rating") >= filters.rating_min)
    if filters.rating_max is not None:
        lf = lf.filter(pl.col("rating") <= filters.rating_max)
    return lf



def apply_date_filter(lf: pl.LazyFrame, filters: Filters) -> pl.LazyFrame:
    if filters.date_from is not None:
        lf = lf.filter(pl.col("date") >= filters.date_from)
    if filters.date_to is not None:
        lf = lf.filter(pl.col("date") <= filters.date_to)
    return lf



def apply_rating_drop_filter(lf: pl.LazyFrame, filters: Filters) -> pl.LazyFrame:
    """Keep only reviews belonging to drugs whose average rating dropped
    more than rating_drop_threshold points in the last rating_drop_window_months
    compared to the preceding period of the same length.
    """
    if filters.rating_drop_threshold is None:
        return lf

    window_days = filters.rating_drop_window_months * 30
    today = date.today()
    recent_start = today - timedelta(days=window_days)
    prior_start = recent_start - timedelta(days=window_days)

    recent_avg = (
        lf.filter(pl.col("date") >= recent_start)
        .group_by("drugName")
        .agg(pl.col("rating").mean().alias("recent_avg"))
    )

    prior_avg = (
        lf.filter((pl.col("date") >= prior_start) & (pl.col("date") < recent_start))
        .group_by("drugName")
        .agg(pl.col("rating").mean().alias("prior_avg"))
    )

    qualifying_drugs = (
        recent_avg.join(prior_avg, on="drugName", how="inner")
        .filter(pl.col("prior_avg") - pl.col("recent_avg") > filters.rating_drop_threshold)
        .select("drugName")
    )

    return lf.join(qualifying_drugs, on="drugName", how="inner")



def apply_filters(lf: pl.LazyFrame, filters: Filters) -> list[int]:
    """Apply all filters in sequence and return matching review IDs."""
    lf = apply_drug_filter(lf, filters)
    lf = apply_condition_filter(lf, filters)
    lf = apply_rating_filter(lf, filters)
    lf = apply_date_filter(lf, filters)
    lf = apply_rating_drop_filter(lf, filters)
    return lf.select("uniqueID").collect()["uniqueID"].to_list()
