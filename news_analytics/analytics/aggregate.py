from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from news_analytics.analytics.calendar import next_trading_day, trading_days_from_prices
from news_analytics.db import get_connection

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def _median_map(db_path: Path) -> dict[tuple[str, str], float]:
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT cm.company, na.publish_date AS date, na.sentiment
            FROM news_articles na
            JOIN company_mentions cm ON na.article_id = cm.article_id
            """,
            conn,
        )
    if df.empty:
        return {}
    med = df.groupby(["company", "date"], as_index=True)["sentiment"].median()
    return {(str(k[0]), str(k[1])): float(v) for k, v in med.items()}


def aggregate_daily_metrics(db_path: Path) -> int:
    sql = (SQL_DIR / "daily_aggregation.sql").read_text()
    with get_connection(db_path) as conn:
        conn.executescript(sql)
        medians = _median_map(db_path)
        conn.executemany(
            """
            UPDATE daily_metrics
            SET median_sentiment = ?
            WHERE company = ? AND date = ?
            """,
            [(v, c, d) for (c, d), v in medians.items()],
        )
        n = conn.execute("SELECT COUNT(*) AS n FROM daily_metrics").fetchone()["n"]
    return int(n)


def _causal_rolling_mean(values: pd.Series, window: int) -> pd.Series:
    """Rolling mean that only looks backward."""
    return values.rolling(window=window, min_periods=max(2, window // 2)).mean()


def _causal_rolling_std(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=max(3, window // 2)).std(ddof=1)


def build_daily_features(db_path: Path, benchmark: str) -> int:
    """Daily features on the trading calendar. Rolling stats don't peek ahead."""
    with get_connection(db_path) as conn:
        daily = pd.read_sql_query("SELECT * FROM daily_metrics", conn)
        prices = pd.read_sql_query("SELECT ticker, trade_date, adj_close FROM stock_prices", conn)
        mapped = pd.read_sql_query("SELECT DISTINCT company AS ticker FROM company_mentions", conn)

    if mapped.empty:
        with get_connection(db_path) as conn:
            conn.execute("DELETE FROM daily_features;")
        return 0

    trading_days = trading_days_from_prices(prices, benchmark) if not prices.empty else pd.DatetimeIndex(
        sorted(pd.to_datetime(daily["date"].unique())) if not daily.empty else []
    )
    if len(trading_days) == 0:
        with get_connection(db_path) as conn:
            conn.execute("DELETE FROM daily_features;")
        return 0

    tickers = sorted(set(mapped["ticker"].tolist()) - {benchmark, "SPY"})
    frames = []
    daily = daily.copy()
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])

    for ticker in tickers:
        spine = pd.DataFrame({"date": trading_days, "ticker": ticker})
        sub = daily[daily["company"] == ticker] if not daily.empty else pd.DataFrame()
        if sub.empty:
            merged = spine
            merged["article_count"] = 0
            merged["mean_sentiment"] = np.nan
            merged["median_sentiment"] = np.nan
            merged["pos_count"] = 0
            merged["neg_count"] = 0
            merged["weighted_sentiment"] = np.nan
        else:
            merged = spine.merge(
                sub.rename(
                    columns={
                        "news_volume": "article_count",
                        "avg_sentiment": "mean_sentiment",
                    }
                ),
                left_on="date",
                right_on="date",
                how="left",
            )
            merged["article_count"] = merged["article_count"].fillna(0).astype(int)
            for col in ("pos_count", "neg_count"):
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0).astype(int)
                else:
                    merged[col] = 0

        merged = merged.sort_values("date")
        vol = merged["article_count"].astype(float)
        merged["rolling_sent_5"] = _causal_rolling_mean(merged["mean_sentiment"], 5)
        merged["rolling_sent_20"] = _causal_rolling_mean(merged["mean_sentiment"], 20)
        merged["volume_mean_20"] = _causal_rolling_mean(vol, 20)
        merged["volume_std_20"] = _causal_rolling_std(vol, 20)
        std = merged["volume_std_20"]
        mean = merged["volume_mean_20"]
        z = (vol - mean) / std
        z = z.where(std.notna() & (std > 1e-12) & mean.notna())
        merged["volume_z_20"] = z
        # score = sentiment × unusual volume (today only)
        sent = merged["mean_sentiment"].fillna(0.0)
        merged["event_score"] = sent * merged["volume_z_20"].clip(lower=0)
        merged["is_trading_day"] = 1
        frames.append(
            merged[
                [
                    "ticker",
                    "date",
                    "article_count",
                    "mean_sentiment",
                    "median_sentiment",
                    "pos_count",
                    "neg_count",
                    "weighted_sentiment",
                    "rolling_sent_5",
                    "rolling_sent_20",
                    "volume_mean_20",
                    "volume_std_20",
                    "volume_z_20",
                    "event_score",
                    "is_trading_day",
                ]
            ]
        )

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    rows = []
    for r in out.itertuples(index=False):
        rows.append(
            (
                r.ticker,
                pd.Timestamp(r.date).date().isoformat(),
                int(r.article_count) if not pd.isna(r.article_count) else 0,
                None if pd.isna(r.mean_sentiment) else float(r.mean_sentiment),
                None if pd.isna(r.median_sentiment) else float(r.median_sentiment),
                int(r.pos_count) if not pd.isna(r.pos_count) else 0,
                int(r.neg_count) if not pd.isna(r.neg_count) else 0,
                None if pd.isna(r.weighted_sentiment) else float(r.weighted_sentiment),
                None if pd.isna(r.rolling_sent_5) else float(r.rolling_sent_5),
                None if pd.isna(r.rolling_sent_20) else float(r.rolling_sent_20),
                None if pd.isna(r.volume_mean_20) else float(r.volume_mean_20),
                None if pd.isna(r.volume_std_20) else float(r.volume_std_20),
                None if pd.isna(r.volume_z_20) else float(r.volume_z_20),
                None if pd.isna(r.event_score) else float(r.event_score),
                int(r.is_trading_day),
            )
        )
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM daily_features;")
        conn.executemany(
            """
            INSERT INTO daily_features (
                ticker, date, article_count, mean_sentiment, median_sentiment,
                pos_count, neg_count, weighted_sentiment, rolling_sent_5, rolling_sent_20,
                volume_mean_20, volume_std_20, volume_z_20, event_score, is_trading_day
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def attach_next_day_prices(db_path: Path, benchmark: str) -> int:
    """Fill next-session close on daily_metrics (raw change, not vs market)."""
    with get_connection(db_path) as conn:
        daily = pd.read_sql_query("SELECT company, date FROM daily_metrics", conn)
        prices = pd.read_sql_query(
            "SELECT ticker, trade_date, adj_close FROM stock_prices", conn
        )
    if daily.empty or prices.empty:
        return 0
    trading_days = trading_days_from_prices(prices, benchmark)
    prices = prices.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    px = {
        t: g.set_index("trade_date")["adj_close"].sort_index()
        for t, g in prices.groupby("ticker")
    }
    updates = []
    for r in daily.itertuples(index=False):
        series = px.get(r.company)
        if series is None:
            updates.append((None, None, None, r.company, r.date))
            continue
        d0 = pd.Timestamp(r.date)
        # close on or after the news date
        pos = series.index.searchsorted(d0)
        if pos >= len(series.index):
            updates.append((None, None, None, r.company, r.date))
            continue
        t0 = series.index[pos]
        p0 = float(series.loc[t0])
        nxt = next_trading_day(trading_days, t0)
        if nxt is None or nxt not in series.index or p0 == 0:
            updates.append((p0, None, None, r.company, r.date))
            continue
        p1 = float(series.loc[nxt])
        pct = ((p1 - p0) / p0) * 100.0
        updates.append((p0, p1, pct, r.company, r.date))
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            UPDATE daily_metrics
            SET close_price = ?, next_close_price = ?, pct_change_next_day = ?
            WHERE company = ? AND date = ?
            """,
            updates,
        )
    return len(updates)
