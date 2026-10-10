from __future__ import annotations

from pathlib import Path

import pandas as pd

from news_analytics.analytics.calendar import next_trading_day, trading_days_from_prices
from news_analytics.db import get_connection


def chronological_split(dates: pd.Series, train_fraction: float) -> pd.Timestamp:
    uniq = pd.to_datetime(sorted(dates.dropna().unique()))
    if len(uniq) == 0:
        raise ValueError("No dates available for train/test split")
    cut = max(int(len(uniq) * train_fraction) - 1, 0)
    return pd.Timestamp(uniq[cut])


def detect_events(db_path: Path, benchmark: str, train_fraction: float = 0.7) -> dict[str, float | int]:
    """
    High-impact events: event_score at or above the train-period 90th percentile,
    requiring positive volume z (unusual news volume vs the ticker's own history)
    and non-null sentiment. Threshold never uses test-period or future scores.
    """
    with get_connection(db_path) as conn:
        feat = pd.read_sql_query("SELECT * FROM daily_features", conn)
        prices = pd.read_sql_query("SELECT ticker, trade_date FROM stock_prices", conn)

    if feat.empty:
        with get_connection(db_path) as conn:
            conn.execute("DELETE FROM market_events;")
        return {"n_events": 0, "threshold": None}

    feat["date"] = pd.to_datetime(feat["date"])
    scored = feat[feat["event_score"].notna() & (feat["article_count"] > 0)].copy()
    if scored.empty:
        with get_connection(db_path) as conn:
            conn.execute("DELETE FROM market_events;")
        return {"n_events": 0, "threshold": None}

    train_end = chronological_split(scored["date"], train_fraction)
    train_scores = scored.loc[scored["date"] <= train_end, "event_score"]
    if train_scores.empty:
        train_scores = scored["event_score"]
    threshold = float(train_scores.quantile(0.90))
    # A non-positive train percentile is not "unusually high"; require the
    # 90th percentile among strictly positive train scores when possible.
    if threshold <= 0:
        positive = train_scores[train_scores > 0]
        if len(positive) >= 5:
            threshold = float(positive.quantile(0.90))
        else:
            threshold = float("inf")

    trading_days = trading_days_from_prices(
        prices.rename(columns={"trade_date": "trade_date"}).assign(
            ticker=prices["ticker"]
        )
        if not prices.empty
        else pd.DataFrame(columns=["ticker", "trade_date"]),
        benchmark,
    )
    if prices.empty:
        trading_days = pd.DatetimeIndex(sorted(feat["date"].unique()))

    rows = []
    for r in feat.itertuples(index=False):
        exec_date = next_trading_day(trading_days, r.date)
        exec_iso = None if exec_date is None else exec_date.date().isoformat()
        d = pd.Timestamp(r.date).date().isoformat()
        if r.mean_sentiment is not None and not pd.isna(r.mean_sentiment):
            if r.mean_sentiment <= -0.5:
                rows.append((r.ticker, d, "STRONG_NEGATIVE_NEWS", r.event_score, r.volume_z_20, r.mean_sentiment, exec_iso))
            if r.mean_sentiment >= 0.5:
                rows.append((r.ticker, d, "STRONG_POSITIVE_NEWS", r.event_score, r.volume_z_20, r.mean_sentiment, exec_iso))
        if r.volume_z_20 is not None and not pd.isna(r.volume_z_20) and r.volume_z_20 >= 2.0:
            rows.append((r.ticker, d, "HIGH_MEDIA_ATTENTION", r.event_score, r.volume_z_20, r.mean_sentiment, exec_iso))
        if (
            r.event_score is not None
            and not pd.isna(r.event_score)
            and r.article_count > 0
            and r.event_score > 0
            and r.event_score >= threshold
            and r.volume_z_20 is not None
            and not pd.isna(r.volume_z_20)
            and r.volume_z_20 > 0
        ):
            rows.append((r.ticker, d, "HIGH_IMPACT", r.event_score, r.volume_z_20, r.mean_sentiment, exec_iso))

    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM market_events;")
        conn.executemany(
            """
            INSERT OR REPLACE INTO market_events
                (company, date, event_type, event_score, volume_z_20, mean_sentiment, execution_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM market_events WHERE event_type = 'HIGH_IMPACT'"
        ).fetchone()["n"]
    return {"n_events": int(n), "threshold": threshold, "train_end": train_end.date().isoformat()}
