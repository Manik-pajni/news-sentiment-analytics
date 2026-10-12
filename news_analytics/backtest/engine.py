from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from news_analytics.analytics.events import chronological_split
from news_analytics.analytics.returns import build_return_panels
from news_analytics.db import get_connection


def _sharpe(daily: pd.Series) -> float | None:
    daily = daily.dropna()
    if len(daily) < 5 or daily.std(ddof=1) == 0:
        return None
    return float(np.sqrt(252) * daily.mean() / daily.std(ddof=1))


def _max_drawdown(cum: pd.Series) -> float | None:
    if cum.empty:
        return None
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min())


def run_backtest(
    db_path: Path,
    benchmark: str,
    *,
    train_fraction: float = 0.7,
    cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    max_positions: int = 10,
) -> dict:
    """1-day long-only book on HIGH_IMPACT names with positive sentiment."""
    with get_connection(db_path) as conn:
        events = pd.read_sql_query(
            """
            SELECT company AS ticker, date, event_score, mean_sentiment, execution_date
            FROM market_events
            WHERE event_type = 'HIGH_IMPACT'
            """,
            conn,
        )
        prices = pd.read_sql_query("SELECT ticker, trade_date, adj_close FROM stock_prices", conn)
        features = pd.read_sql_query("SELECT date FROM daily_features", conn)

    empty = {
        "n_trades": 0,
        "note": "Insufficient data for backtest.",
    }
    if events.empty or prices.empty:
        _store(db_path, pd.DataFrame(), pd.DataFrame(), empty, cost_bps, slippage_bps)
        return empty

    events["date"] = pd.to_datetime(events["date"])
    events["execution_date"] = pd.to_datetime(events["execution_date"])
    events = events.dropna(subset=["execution_date"])
    events = events[events["mean_sentiment"].fillna(0) > 0]
    if features.empty:
        train_end = chronological_split(events["date"], train_fraction)
    else:
        train_end = chronological_split(pd.to_datetime(features["date"]), train_fraction)

    test_events = events[events["date"] > train_end].copy()
    panels, bench_rets = build_return_panels(prices, benchmark)
    if bench_rets.empty:
        _store(db_path, pd.DataFrame(), pd.DataFrame(), empty, cost_bps, slippage_bps)
        return empty

    test_days = bench_rets.index[bench_rets.index > train_end]
    cost_rate = (cost_bps + slippage_bps) / 10000.0

    trades = []
    daily_rows = []
    prev_names: set[str] = set()
    wealth = 1.0
    bench_wealth = 1.0
    peak = 1.0

    grouped = {
        d: g for d, g in test_events.groupby(test_events["execution_date"].dt.normalize())
    }

    for day in test_days:
        key = pd.Timestamp(day).normalize()
        names = pd.DataFrame()
        if key in grouped:
            names = grouped[key].sort_values("event_score", ascending=False).drop_duplicates("ticker")
            names = names.head(max_positions)
        candidate_tickers = [] if names.empty else names["ticker"].tolist()
        tickers = [t for t in candidate_tickers if t in panels and key in panels[t].index]
        n = len(tickers)
        weights = {t: 1.0 / n for t in tickers} if n else {}
        gross = 0.0
        for t, w in weights.items():
            r = float(panels[t].loc[key])
            if pd.isna(r):
                continue
            gross += w * r
            trades.append(
                {
                    "trade_date": key.date().isoformat(),
                    "ticker": t,
                    "side": "long",
                    "weight": w,
                    "gross_return": r,
                    "net_return": None,
                    "cost_bps": cost_bps + slippage_bps,
                }
            )
        current_names = set(tickers)
        turnover = 0.0
        all_names = prev_names | current_names
        for t in all_names:
            w_prev = (1.0 / len(prev_names)) if prev_names and t in prev_names else 0.0
            w_now = weights.get(t, 0.0)
            turnover += abs(w_now - w_prev)
        # we also exit the same names at the close
        for t in current_names:
            turnover += weights.get(t, 0.0)
        costs = turnover * cost_rate
        net = gross - costs
        for tr in trades:
            if tr["trade_date"] == key.date().isoformat() and tr["net_return"] is None:
                tr["net_return"] = tr["gross_return"] - cost_rate * 2.0

        bench_r = float(bench_rets.loc[key]) if key in bench_rets.index and pd.notna(bench_rets.loc[key]) else 0.0
        wealth *= 1.0 + net
        bench_wealth *= 1.0 + bench_r
        peak = max(peak, wealth)
        daily_rows.append(
            {
                "date": key.date().isoformat(),
                "portfolio_return": net,
                "cumulative_return": wealth - 1.0,
                "benchmark_return": bench_r,
                "benchmark_cumulative": bench_wealth - 1.0,
                "drawdown": wealth / peak - 1.0,
                "n_positions": n,
                "turnover": turnover,
            }
        )
        prev_names = set()  # start flat the next day

    daily = pd.DataFrame(daily_rows)
    trade_df = pd.DataFrame(trades)
    port = daily["portfolio_return"] if not daily.empty else pd.Series(dtype=float)
    summary = {
        "split": "test",
        "n_trades": int(len(trade_df)),
        "n_days": int(len(daily)),
        "total_return": None if daily.empty else float(daily["cumulative_return"].iloc[-1]),
        "benchmark_total_return": None if daily.empty else float(daily["benchmark_cumulative"].iloc[-1]),
        "annualized_vol": None if len(port) < 5 else float(port.std(ddof=1) * np.sqrt(252)),
        "sharpe": _sharpe(port),
        "max_drawdown": None if daily.empty else float(daily["drawdown"].min()),
        "turnover": None if daily.empty else float(daily["turnover"].mean()),
        "transaction_cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "notes": (
            "Test-period only. Thresholds estimated on earlier dates. "
            "Close-to-close execution is optimistic vs next-open fills. "
            "Yahoo history may omit delisted names (survivorship)."
        ),
    }
    _store(db_path, trade_df, daily, summary, cost_bps, slippage_bps)
    return summary


def _store(db_path: Path, trades: pd.DataFrame, daily: pd.DataFrame, summary: dict, cost_bps, slippage_bps) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM backtest_trades;")
        conn.execute("DELETE FROM backtest_daily;")
        conn.execute("DELETE FROM backtest_summary;")
        if not trades.empty:
            conn.executemany(
                """
                INSERT OR REPLACE INTO backtest_trades
                    (trade_date, ticker, side, weight, gross_return, net_return, cost_bps)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (r.trade_date, r.ticker, r.side, r.weight, r.gross_return, r.net_return, r.cost_bps)
                    for r in trades.itertuples(index=False)
                ],
            )
        if not daily.empty:
            conn.executemany(
                """
                INSERT INTO backtest_daily
                    (date, portfolio_return, cumulative_return, benchmark_return,
                     benchmark_cumulative, drawdown, n_positions, turnover)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.date,
                        r.portfolio_return,
                        r.cumulative_return,
                        r.benchmark_return,
                        r.benchmark_cumulative,
                        r.drawdown,
                        r.n_positions,
                        r.turnover,
                    )
                    for r in daily.itertuples(index=False)
                ],
            )
        conn.execute(
            """
            INSERT INTO backtest_summary (
                split, n_trades, n_days, total_return, benchmark_total_return,
                annualized_vol, sharpe, max_drawdown, turnover,
                transaction_cost_bps, slippage_bps, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.get("split", "test"),
                summary.get("n_trades"),
                summary.get("n_days"),
                summary.get("total_return"),
                summary.get("benchmark_total_return"),
                summary.get("annualized_vol"),
                summary.get("sharpe"),
                summary.get("max_drawdown"),
                summary.get("turnover"),
                cost_bps,
                slippage_bps,
                summary.get("notes") or summary.get("note"),
            ),
        )
