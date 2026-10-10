from __future__ import annotations

from pathlib import Path

import pandas as pd

from news_analytics.db import get_connection
from news_analytics.entity.normalize import clean_mention


def seed_universe(db_path: Path, ticker_path: Path, alias_path: Path) -> int:
    tickers = pd.read_csv(ticker_path)
    aliases = pd.read_csv(alias_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM company_aliases;")
        conn.execute("DELETE FROM tickers;")
        conn.executemany(
            """
            INSERT INTO tickers (ticker, company_name, sector, yfinance_symbol, is_benchmark)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    str(r.ticker),
                    str(r.company_name),
                    None if pd.isna(r.sector) else str(r.sector),
                    str(r.yfinance_symbol),
                    1 if str(r.ticker) == "SPY" or str(r.sector) == "Benchmark" else 0,
                )
                for r in tickers.itertuples(index=False)
            ],
        )
        rows = []
        for r in aliases.itertuples(index=False):
            raw = str(r.alias)
            ticker = str(r.ticker)
            rows.append((clean_mention(raw), raw, ticker))
        conn.executemany(
            """
            INSERT OR REPLACE INTO company_aliases (alias_norm, alias_raw, ticker)
            VALUES (?, ?, ?)
            """,
            rows,
        )
    return len(tickers)
