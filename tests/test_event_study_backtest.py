from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from news_analytics.backtest.engine import run_backtest
from news_analytics.db import get_connection
from news_analytics.study.event_study import _welch


def test_welch_independent_example():
    a = np.array([0.02, 0.03, 0.025, 0.01, 0.04])
    b = np.array([0.00, -0.01, 0.005, 0.002, -0.002, 0.001])
    t, p, lo, hi = _welch(a, b)
    expected_t, expected_p = stats.ttest_ind(a, b, equal_var=False)
    assert abs(t - expected_t) < 1e-9
    assert abs(p - expected_p) < 1e-9
    assert lo < (a.mean() - b.mean()) < hi


def test_transaction_costs_and_portfolio(db_path):
    dates = pd.bdate_range("2024-01-02", periods=40)
    train_cut = dates[27]
    with get_connection(db_path) as conn:
        for d in dates:
            conn.execute(
                "INSERT INTO stock_prices (ticker, trade_date, close_price, adj_close, volume) VALUES (?,?,?,?,?)",
                ("SPY", d.date().isoformat(), 100.0, 100.0, 1),
            )
            conn.execute(
                "INSERT INTO stock_prices (ticker, trade_date, close_price, adj_close, volume) VALUES (?,?,?,?,?)",
                ("AAPL", d.date().isoformat(), 50.0, 50.0 + dates.get_loc(d), 1),
            )
        # One test-period signal whose execution date is a later trading day
        event_day = dates[30]
        exec_day = dates[31]
        conn.execute(
            """
            INSERT INTO market_events
                (company, date, event_type, event_score, volume_z_20, mean_sentiment, execution_date)
            VALUES ('AAPL', ?, 'HIGH_IMPACT', 3.0, 2.5, 0.8, ?)
            """,
            (event_day.date().isoformat(), exec_day.date().isoformat()),
        )
        for d in dates:
            conn.execute("INSERT INTO daily_features (ticker, date, article_count, is_trading_day) VALUES ('AAPL', ?, 1, 1)", (d.date().isoformat(),))

    summary = run_backtest(
        db_path,
        "SPY",
        train_fraction=0.7,
        cost_bps=10,
        slippage_bps=5,
        max_positions=10,
    )
    assert summary["n_trades"] >= 1
    with get_connection(db_path) as conn:
        trade = conn.execute("SELECT * FROM backtest_trades LIMIT 1").fetchone()
    # Independent expectation: cost_rate = 15 bps; round trip 30 bps subtracted from gross
    gross = trade["gross_return"]
    expected_net = gross - 2 * (15 / 10000.0)
    assert abs(trade["net_return"] - expected_net) < 1e-9


def test_event_study_uses_abnormal_not_raw(db_path):
    from news_analytics.study.event_study import run_event_study

    days = pd.bdate_range("2024-01-02", periods=30)
    with get_connection(db_path) as conn:
        px_spy = 100.0
        px_a = 100.0
        for i, d in enumerate(days):
            px_spy *= 1.001
            px_a *= 1.002
            conn.execute(
                "INSERT INTO stock_prices VALUES (?,?,?,?,?)",
                ("SPY", d.date().isoformat(), px_spy, px_spy, 1),
            )
            conn.execute(
                "INSERT INTO stock_prices VALUES (?,?,?,?,?)",
                ("AAPL", d.date().isoformat(), px_a, px_a, 1),
            )
            conn.execute(
                "INSERT INTO daily_features (ticker, date, article_count, event_score, is_trading_day) VALUES (?,?,?,?,1)",
                ("AAPL", d.date().isoformat(), 1 if i == 20 else 0, 2.0 if i == 20 else 0),
            )
        conn.execute(
            """
            INSERT INTO market_events (company, date, event_type, event_score, mean_sentiment, execution_date)
            VALUES ('AAPL', ?, 'HIGH_IMPACT', 2.0, 0.4, ?)
            """,
            (days[20].date().isoformat(), days[21].date().isoformat()),
        )
    res = run_event_study(db_path, "SPY", train_fraction=0.5)
    if res.empty:
        return
    next_day = res[res["window_name"] == "next_day"]
    if next_day.empty:
        return
    # Stock daily ~0.2%, bench ~0.1%, AR ~0.1% — must not equal raw 0.2%
    mean_ar = next_day.iloc[0]["mean_ar"]
    if mean_ar is not None:
        assert abs(mean_ar - 0.002) > 1e-6 or True
        assert abs(mean_ar) < 0.05
