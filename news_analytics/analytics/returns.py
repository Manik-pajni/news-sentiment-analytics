from __future__ import annotations

import numpy as np
import pandas as pd

from news_analytics.analytics.calendar import next_trading_day, window_trading_days


def daily_returns(price: pd.Series) -> pd.Series:
    price = price.sort_index().astype(float)
    return price.pct_change()


def benchmark_adjusted_return(stock_ret: float | None, bench_ret: float | None) -> float | None:
    if stock_ret is None or bench_ret is None or pd.isna(stock_ret) or pd.isna(bench_ret):
        return None
    return float(stock_ret) - float(bench_ret)


def market_model_abnormal(
    stock_rets: pd.Series,
    bench_rets: pd.Series,
    event_date,
    estimation_window: int = 60,
    gap: int = 1,
) -> float | None:
    """
    Estimate alpha/beta on dates strictly before the event (with a gap),
    then AR = R - (alpha + beta * Rm) on the execution date.
    Distinct from simple benchmark-adjusted returns (R - Rm).
    """
    aligned = pd.concat([stock_rets.rename("r"), bench_rets.rename("m")], axis=1).dropna()
    if aligned.empty:
        return None
    event = pd.Timestamp(event_date)
    # Estimation sample: last `estimation_window` sessions ending `gap` days before event
    eligible = aligned.index[aligned.index < event]
    if len(eligible) < gap + 10:
        return None
    if gap > 0:
        eligible = eligible[:-gap] if len(eligible) > gap else eligible[:0]
    sample = aligned.loc[eligible].tail(estimation_window)
    if len(sample) < 20:
        return None
    x = sample["m"].to_numpy()
    y = sample["r"].to_numpy()
    var = np.var(x, ddof=1)
    if var < 1e-16:
        return None
    beta = float(np.cov(y, x, ddof=1)[0, 1] / var)
    alpha = float(y.mean() - beta * x.mean())
    if event not in aligned.index:
        return None
    r = float(aligned.loc[event, "r"])
    m = float(aligned.loc[event, "m"])
    return r - (alpha + beta * m)


def build_return_panels(prices: pd.DataFrame, benchmark: str) -> tuple[dict[str, pd.Series], pd.Series]:
    prices = prices.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    panels: dict[str, pd.Series] = {}
    for ticker, g in prices.groupby("ticker"):
        s = g.drop_duplicates("trade_date").set_index("trade_date")["adj_close"].sort_index()
        panels[str(ticker)] = daily_returns(s)
    bench_key = benchmark if benchmark in panels else "SPY"
    bench = panels.get(bench_key, pd.Series(dtype=float))
    return panels, bench


def execution_abnormal_return(
    stock_rets: pd.Series,
    bench_rets: pd.Series,
    publish_date,
    trading_days: pd.DatetimeIndex,
    horizon: int = 1,
) -> float | None:
    """
    Signal is treated as observable at the end of publish_date (no intraday stamp).
    Trade/measure on subsequent trading sessions only.
    horizon=1 -> next trading day; >1 -> cumulative abnormal return over next N days.
    """
    exec0 = next_trading_day(trading_days, publish_date)
    if exec0 is None:
        return None
    days = [exec0] + window_trading_days(trading_days, exec0, horizon - 1) if horizon > 1 else [exec0]
    ars = []
    for d in days:
        if d not in stock_rets.index or d not in bench_rets.index:
            return None
        ar = benchmark_adjusted_return(float(stock_rets.loc[d]), float(bench_rets.loc[d]))
        if ar is None:
            return None
        ars.append(ar)
    return float(np.sum(ars))
