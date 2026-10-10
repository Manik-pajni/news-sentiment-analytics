from datetime import date

import pytest

from news_analytics.pipeline import DEFAULT_FETCH_DAYS, build_parser, fetch_window


def test_fetch_defaults():
    args = build_parser().parse_args(["--fetch"])
    assert args.fetch is True
    assert args.fetch_days == DEFAULT_FETCH_DAYS == 7


def test_fetch_days_override():
    args = build_parser().parse_args(["--fetch", "--fetch-days", "3"])
    assert args.fetch is True
    assert args.fetch_days == 3


def test_default_pipeline_does_not_fetch():
    args = build_parser().parse_args([])
    assert args.fetch is False
    assert args.steps == "all"
    assert args.fetch_days == 7


def test_fetch_window_inclusive():
    start, end = fetch_window(7, today=date(2026, 10, 10))
    assert start == "2026-10-04"
    assert end == "2026-10-10"


def test_fetch_window_rejects_non_positive():
    with pytest.raises(ValueError):
        fetch_window(0)
