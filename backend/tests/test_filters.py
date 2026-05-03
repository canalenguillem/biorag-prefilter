from datetime import date

import polars as pl
import pytest

from biorag.filters import Filters, apply_condition_filter, apply_date_filter, apply_drug_filter, apply_filters, apply_rating_filter, apply_rating_drop_filter


@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.LazyFrame({
        "uniqueID": [1, 2, 3, 4],
        "drugName": ["Sertraline", "Sertraline", "Mirtazapine", "Mirtazapine"],
        "condition": ["Depression", "Depression", "Depression", "Anxiety"],
        "rating": [8, 6, 9, 4],
        "date": [date(2023, 1, 1), date(2024, 6, 1), date(2023, 3, 1), date(2024, 9, 1)],
        "usefulCount": [10, 5, 20, 3],
    })


def test_apply_drug_filter(lf: pl.LazyFrame) -> None:
    filters = Filters(drug_names=["Sertraline"])
    result = apply_drug_filter(lf, filters).collect()["uniqueID"].to_list()
    assert sorted(result) == [1, 2]


def test_apply_drug_filter_none_is_noop(lf: pl.LazyFrame) -> None:
    filters = Filters()
    result = apply_drug_filter(lf, filters).collect()["uniqueID"].to_list()
    assert len(result) == 4


def test_apply_condition_filter(lf: pl.LazyFrame) -> None:
    filters = Filters(conditions=["Anxiety"])
    result = apply_condition_filter(lf, filters).collect()["uniqueID"].to_list()
    assert result == [4]


def test_apply_rating_filter_min(lf: pl.LazyFrame) -> None:
    filters = Filters(rating_min=7)
    result = apply_rating_filter(lf, filters).collect()["uniqueID"].to_list()
    assert sorted(result) == [1, 3]


def test_apply_rating_filter_range(lf: pl.LazyFrame) -> None:
    filters = Filters(rating_min=6, rating_max=8)
    result = apply_rating_filter(lf, filters).collect()["uniqueID"].to_list()
    assert sorted(result) == [1, 2]


def test_apply_date_filter(lf: pl.LazyFrame) -> None:
    filters = Filters(date_from=date(2024, 1, 1))
    result = apply_date_filter(lf, filters).collect()["uniqueID"].to_list()
    assert sorted(result) == [2, 4]


# --- rating drop fixture and test ---

@pytest.fixture
def lf_drop() -> pl.LazyFrame:
    """
    Sertraline:  prior avg = 8.0, recent avg = 5.0  → drop 3.0  (qualifies at threshold 1.5)
    Mirtazapine: prior avg = 7.0, recent avg = 7.0  → drop 0.0  (does not qualify)

    "recent" = last 12 months from today, approximated as last 360 days.
    Dates are hardcoded relative to 2026-05-03 (current date in this project).
    """
    return pl.LazyFrame({
        "uniqueID": [1, 2, 3, 4, 5, 6, 7, 8],
        "drugName": [
            "Sertraline", "Sertraline",       # prior window
            "Sertraline", "Sertraline",       # recent window
            "Mirtazapine", "Mirtazapine",     # prior window
            "Mirtazapine", "Mirtazapine",     # recent window
        ],
        "condition": ["Depression"] * 8,
        "rating": [8, 8, 5, 5, 7, 7, 7, 7],
        "date": [
            date(2024, 7, 1), date(2024, 7, 2),   # Sertraline prior  (prior window: ~2024-05-13 to ~2025-05-08)
            date(2025, 6, 1), date(2025, 6, 2),   # Sertraline recent (recent window: ~2025-05-08 to today)
            date(2024, 7, 1), date(2024, 7, 2),   # Mirtazapine prior
            date(2025, 6, 1), date(2025, 6, 2),   # Mirtazapine recent
        ],
        "usefulCount": [1] * 8,
    })


def test_apply_rating_drop_filter(lf_drop: pl.LazyFrame) -> None:
    filters = Filters(rating_drop_threshold=1.5, rating_drop_window_months=12)
    result = apply_rating_drop_filter(lf_drop, filters).collect()["drugName"].unique().to_list()
    assert result == ["Sertraline"]
