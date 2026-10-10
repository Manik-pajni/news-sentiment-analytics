from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from news_analytics.db import get_connection

logger = logging.getLogger("news_analytics")


def _yf_symbol(symbol: str) -> str:
    return (symbol or "").replace(".", "-")


def mapped_tickers(db_path: Path, benchmark: str) -> list[str]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT company FROM company_mentions WHERE company IS NOT NULL"
        ).fetchall()
        tickers = [r["company"] for r in rows]
        bench = conn.execute(
            "SELECT yfinance_symbol FROM tickers WHERE ticker = ? OR is_benchmark = 1",
            (benchmark,),
        ).fetchall()
    universe = set(tickers)
    universe.add(benchmark)
    for row in bench:
        universe.add(row["yfinance_symbol"].replace("-", "."))
        universe.add(row["yfinance_symbol"])
    universe.add("SPY")
    return sorted(universe)


def ingest_stock_prices(
    db_path: Path,
    tickers: list[str],
    start: str,
    end: str,
    *,
    timeout: int = 30,
    max_retries: int = 3,
) -> dict[str, int]:
    """
    Store Yahoo Finance daily bars. Prefer Adj Close for returns.

    Limitation: Yahoo adjusted prices can be revised retrospectively and omit
    some delisted names (survivorship). Missing symbols are logged, never filled
    with simulated prices.
    """
    inserted = 0
    failed = 0
    # yfinance end is exclusive for some versions; pad one day.
    end_pad = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    unique = []
    seen = set()
    for t in tickers:
        if t and t not in seen:
            unique.append(t)
            seen.add(t)

    with get_connection(db_path) as conn:
        for ticker in unique:
            symbol = _yf_symbol(ticker)
            hist = pd.DataFrame()
            last_err = None
            for attempt in range(max_retries):
                try:
                    hist = yf.download(
                        symbol,
                        start=start,
                        end=end_pad,
                        progress=False,
                        auto_adjust=False,
                        threads=False,
                        timeout=timeout,
                    )
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    time.sleep(1.5 * (attempt + 1))
            if hist is None or hist.empty:
                failed += 1
                conn.execute(
                    "INSERT INTO ingestion_errors (source, context, error) VALUES (?, ?, ?)",
                    ("yfinance", ticker, str(last_err or "empty_history")),
                )
                logger.warning("No prices for %s", ticker)
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            adj = hist["Adj Close"] if "Adj Close" in hist.columns else hist.get("Close")
            close = hist["Close"] if "Close" in hist.columns else adj
            vol = hist["Volume"] if "Volume" in hist.columns else None
            rows = []
            for idx, adj_px in adj.dropna().items():
                trade_date = pd.Timestamp(idx).date().isoformat()
                c = None if pd.isna(close.loc[idx]) else float(close.loc[idx])
                v = None if vol is None or pd.isna(vol.loc[idx]) else float(vol.loc[idx])
                rows.append((ticker, trade_date, c, float(adj_px), v))
            conn.executemany(
                """
                INSERT INTO stock_prices (ticker, trade_date, close_price, adj_close, volume)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker, trade_date) DO UPDATE SET
                    close_price = excluded.close_price,
                    adj_close = excluded.adj_close,
                    volume = excluded.volume
                """,
                rows,
            )
            inserted += len(rows)
            logger.info("Stored %s price rows for %s", len(rows), ticker)
    return {"price_rows": inserted, "failed_tickers": failed}
