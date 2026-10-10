from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

from news_analytics.analytics.aggregate import (
    aggregate_daily_metrics,
    attach_next_day_prices,
    build_daily_features,
)
from news_analytics.analytics.events import detect_events
from news_analytics.backtest.engine import run_backtest
from news_analytics.config import load_settings
from news_analytics.db import get_connection
from news_analytics.entity.resolver import load_universe
from news_analytics.ingest.gdelt import download_gdelt_daily, ingest_gdelt
from news_analytics.ingest.prices import ingest_stock_prices, mapped_tickers
from news_analytics.ingest.universe import seed_universe
from news_analytics.logging_utils import setup_logging
from news_analytics.schema import create_tables
from news_analytics.study.event_study import run_event_study

DEFAULT_FETCH_DAYS = 7


def _log_run(db_path: Path, step: str, status: str, message: str, records: int | None, started: str) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs (step, status, message, records, started_at, finished_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            """,
            (step, status, message, records, started),
        )


def fetch_window(days: int, *, today: date | None = None) -> tuple[str, str]:
    """Inclusive calendar window of `days` days ending on `today` (default: local today)."""
    if days < 1:
        raise ValueError("fetch_days must be >= 1")
    end = today or date.today()
    start = end - timedelta(days=days - 1)
    return start.isoformat(), end.isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="News Sentiment Analytics & Market Impact pipeline"
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Comma-separated: init,ingest-gdelt,ingest-prices,aggregate,events,study,backtest,all",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Download official GDELT daily event files for the last --fetch-days (default 7)",
    )
    parser.add_argument(
        "--fetch-days",
        type=int,
        default=DEFAULT_FETCH_DAYS,
        metavar="N",
        help="Calendar days to download with --fetch (default: 7)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate all tables (destroys existing database rows)",
    )
    return parser


def run_pipeline(
    steps: list[str] | None = None,
    *,
    fetch: bool = False,
    fetch_days: int = DEFAULT_FETCH_DAYS,
    reset: bool = False,
) -> None:
    settings = load_settings()
    logger = setup_logging(settings.log_level, settings.log_file)
    wanted = steps or [
        "init",
        "ingest-gdelt",
        "ingest-prices",
        "aggregate",
        "events",
        "study",
        "backtest",
    ]

    def go(name: str, fn):
        if name not in wanted and "all" not in wanted:
            return
        t0 = datetime.utcnow().isoformat()
        logger.info("Starting step %s", name)
        try:
            result = fn()
            records = result if isinstance(result, int) else None
            if isinstance(result, dict):
                records = int(result.get("articles") or result.get("n_events") or result.get("price_rows") or result.get("n_trades") or 0)
            _log_run(settings.database_path, name, "ok", str(result), records, t0)
            logger.info("Finished step %s: %s", name, result)
        except Exception as exc:
            _log_run(settings.database_path, name, "error", str(exc), None, t0)
            logger.exception("Step %s failed", name)
            raise

    if "init" in wanted or "all" in wanted:
        t0 = datetime.utcnow().isoformat()
        create_tables(settings.database_path, reset=reset)
        n = seed_universe(settings.database_path, settings.ticker_universe_path, settings.alias_path)
        _log_run(settings.database_path, "init", "ok", f"tickers={n}", n, t0)

    universe = load_universe(settings.ticker_universe_path, settings.alias_path)

    if fetch:
        start, end = fetch_window(fetch_days)
        logger.info("Fetching GDELT daily files for %s .. %s (existing files skipped)", start, end)
        download_gdelt_daily(settings.gdelt_dir, start, end)

    def _ingest_news():
        return ingest_gdelt(
            settings.database_path,
            settings.gdelt_dir,
            universe,
            fuzzy_min_score=settings.fuzzy_min_score,
            ambiguity_gap=settings.fuzzy_ambiguity_gap,
            min_name_length=settings.fuzzy_min_name_length,
        )

    go("ingest-gdelt", _ingest_news)

    def _prices():
        with get_connection(settings.database_path) as conn:
            row = conn.execute(
                "SELECT MIN(publish_date) AS a, MAX(publish_date) AS b FROM news_articles"
            ).fetchone()
        if not row["a"] or not row["b"]:
            logger.warning("No articles; skipping prices.")
            return {"price_rows": 0}
        pstart = (datetime.strptime(row["a"], "%Y-%m-%d") - timedelta(days=90)).date().isoformat()
        pend = (datetime.strptime(row["b"], "%Y-%m-%d") + timedelta(days=10)).date().isoformat()
        tickers = mapped_tickers(settings.database_path, settings.benchmark_ticker)
        if len(tickers) <= 1:
            logger.warning("No mapped equities; downloading benchmark only.")
        return ingest_stock_prices(
            settings.database_path,
            tickers,
            pstart,
            pend,
            timeout=settings.yf_timeout_seconds,
            max_retries=settings.yf_max_retries,
        )

    go("ingest-prices", _prices)

    def _agg():
        n = aggregate_daily_metrics(settings.database_path)
        f = build_daily_features(settings.database_path, settings.benchmark_ticker)
        attach_next_day_prices(settings.database_path, settings.benchmark_ticker)
        return {"daily_metrics": n, "feature_rows": f}

    go("aggregate", _agg)

    go(
        "events",
        lambda: detect_events(
            settings.database_path, settings.benchmark_ticker, settings.train_fraction
        ),
    )
    go(
        "study",
        lambda: run_event_study(
            settings.database_path, settings.benchmark_ticker, settings.train_fraction
        ),
    )
    go(
        "backtest",
        lambda: run_backtest(
            settings.database_path,
            settings.benchmark_ticker,
            train_fraction=settings.train_fraction,
            cost_bps=settings.transaction_cost_bps,
            slippage_bps=settings.slippage_bps,
            max_positions=settings.max_positions,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fetch_days < 1:
        parser.error("--fetch-days must be >= 1")
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    run_pipeline(steps=steps, fetch=args.fetch, fetch_days=args.fetch_days, reset=args.reset)


if __name__ == "__main__":
    main()
