from __future__ import annotations

from pathlib import Path

import pandas as pd

from news_analytics.db import connect


def q(db_path: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def overview(db_path: Path) -> dict:
    def scalar(sql: str, default=0):
        df = q(db_path, sql)
        if df.empty:
            return default
        val = df.iloc[0, 0]
        return default if val is None or (isinstance(val, float) and pd.isna(val)) else val

    return {
        "n_articles": int(scalar("SELECT COUNT(*) FROM news_articles")),
        "n_mapped_mentions": int(scalar("SELECT COUNT(*) FROM company_mentions")),
        "n_tickers_mapped": int(scalar("SELECT COUNT(DISTINCT company) FROM company_mentions")),
        "n_unmatched": int(scalar("SELECT COUNT(*) FROM unmatched_mentions")),
        "n_high_impact": int(
            scalar("SELECT COUNT(*) FROM market_events WHERE event_type='HIGH_IMPACT'")
        ),
        "date_min": scalar("SELECT MIN(publish_date) FROM news_articles", None),
        "date_max": scalar("SELECT MAX(publish_date) FROM news_articles", None),
        "n_price_rows": int(scalar("SELECT COUNT(*) FROM stock_prices")),
        "last_run": scalar(
            "SELECT step || ' ' || status || ' ' || COALESCE(finished_at,'') FROM pipeline_runs ORDER BY id DESC LIMIT 1",
            None,
        ),
    }


def mapping_stats(db_path: Path) -> dict:
    df = q(
        db_path,
        """
        SELECT status, COUNT(*) AS n
        FROM entity_match_audit
        GROUP BY status
        """,
    )
    uniq = q(db_path, "SELECT COUNT(DISTINCT mention_norm) AS n FROM entity_match_audit")
    counts = {r.status: int(r.n) for r in df.itertuples(index=False)} if not df.empty else {}
    total = sum(counts.values())
    mapped = counts.get("mapped", 0)
    unique_mentions = int(uniq.iloc[0, 0]) if not uniq.empty else 0
    return {
        "unique_mentions": unique_mentions,
        "mapped": mapped,
        "review": counts.get("review", 0),
        "unmatched": counts.get("unmatched", 0),
        "coverage": (mapped / total) if total else None,
    }
