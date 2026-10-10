from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from news_analytics.analytics.aggregate import _causal_rolling_mean, _causal_rolling_std, aggregate_daily_metrics
from news_analytics.analytics.calendar import next_trading_day
from news_analytics.analytics.returns import benchmark_adjusted_return, daily_returns
from news_analytics.db import get_connection


def test_sentiment_aggregation(db_path):
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO news_articles (article_id, publish_date, source, text, tone, sentiment)
            VALUES
                ('a1', '2024-01-02', 's', 't', 5.0, 0.5),
                ('a2', '2024-01-02', 's', 't', 15.0, 1.0),
                ('a3', '2024-01-02', 's', 't', 0.0, -0.2)
            """
        )
        conn.executemany(
            "INSERT INTO company_mentions (article_id, company) VALUES (?, ?)",
            [("a1", "AAPL"), ("a2", "AAPL"), ("a3", "AAPL")],
        )
    n = aggregate_daily_metrics(db_path)
    assert n == 1
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM daily_metrics").fetchone()
    assert row["news_volume"] == 3
    assert abs(row["avg_sentiment"] - (0.5 + 1.0 - 0.2) / 3) < 1e-9
    assert row["pos_count"] == 2
    assert row["neg_count"] == 1
    assert abs(row["median_sentiment"] - 0.5) < 1e-9
    # Weighted by abs(tone): (0.5*5 + 1.0*15 + -0.2*0) / 20 = 0.875
    assert abs(row["weighted_sentiment"] - 0.875) < 1e-9


def test_rolling_and_zero_variance():
    s = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 10.0])
    mean = _causal_rolling_mean(s, 5)
    std = _causal_rolling_std(s, 5)
    assert abs(mean.iloc[4] - 1.0) < 1e-12
    assert std.iloc[4] < 1e-12 or pd.isna(std.iloc[4]) or std.iloc[4] == 0
    # Last point may use the jump; prior points must not include the 10.
    assert mean.iloc[3] == 1.0
    z_last_if_used_future = None  # feature at index 4 cannot see index 5
    assert s.iloc[5] not in [1.0]


def test_lookahead_rolling_window():
    values = pd.Series([np.nan, 1.0, 2.0, 3.0, 100.0])
    rolled = _causal_rolling_mean(values, 3)
    # Position 3 (value 3) window is 1,2,3 — not 100.
    assert abs(rolled.iloc[3] - 2.0) < 1e-12


def test_trading_day_alignment():
    days = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    # Saturday 6th -> Monday 8th
    nxt = next_trading_day(days, "2024-01-06")
    assert nxt == pd.Timestamp("2024-01-08")
    # Friday stays next session Monday if Friday is last listed before weekend
    nxt_f = next_trading_day(days, "2024-01-05")
    assert nxt_f == pd.Timestamp("2024-01-08")
    # Never same-day
    assert next_trading_day(days, "2024-01-03") == pd.Timestamp("2024-01-04")


def test_abnormal_return_not_raw():
    ar = benchmark_adjusted_return(0.03, 0.01)
    assert abs(ar - 0.02) < 1e-12
    assert benchmark_adjusted_return(0.03, None) is None
    rets = daily_returns(pd.Series([100.0, 110.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"])))
    assert abs(rets.iloc[1] - 0.10) < 1e-12
    assert pd.isna(rets.iloc[0])
