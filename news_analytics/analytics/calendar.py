from __future__ import annotations

import pandas as pd


def next_trading_day(trading_days: pd.DatetimeIndex, asof) -> pd.Timestamp | None:
    """First trading session strictly after asof. Never returns asof itself."""
    days = trading_days.sort_values().unique()
    ts = pd.Timestamp(asof).normalize()
    pos = days.searchsorted(ts, side="right")
    if pos >= len(days):
        return None
    return pd.Timestamp(days[pos])


def trading_days_from_prices(prices: pd.DataFrame, benchmark: str) -> pd.DatetimeIndex:
    bench = prices[prices["ticker"] == benchmark]
    if bench.empty:
        idx = pd.to_datetime(prices["trade_date"].unique())
        return pd.DatetimeIndex(sorted(idx))
    return pd.DatetimeIndex(sorted(pd.to_datetime(bench["trade_date"].unique())))


def window_trading_days(
    trading_days: pd.DatetimeIndex, start_exclusive, n_days: int
) -> list[pd.Timestamp]:
    """Next n trading days strictly after start_exclusive."""
    out = []
    cursor = start_exclusive
    for _ in range(n_days):
        nxt = next_trading_day(trading_days, cursor)
        if nxt is None:
            break
        out.append(nxt)
        cursor = nxt
    return out
