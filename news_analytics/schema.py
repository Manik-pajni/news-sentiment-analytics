from __future__ import annotations

from pathlib import Path

from news_analytics.db import get_connection

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS news_articles (
    article_id TEXT PRIMARY KEY,
    publish_date DATE NOT NULL,
    source TEXT,
    text TEXT,
    tone REAL,
    sentiment REAL,
    actor1 TEXT,
    actor2 TEXT,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS company_mentions (
    article_id TEXT NOT NULL,
    company TEXT NOT NULL,
    raw_mention TEXT,
    match_method TEXT,
    match_confidence REAL,
    PRIMARY KEY (article_id, company),
    FOREIGN KEY (article_id) REFERENCES news_articles(article_id)
);

CREATE TABLE IF NOT EXISTS tickers (
    ticker TEXT PRIMARY KEY,
    company_name TEXT NOT NULL,
    sector TEXT,
    yfinance_symbol TEXT NOT NULL,
    is_benchmark INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS company_aliases (
    alias_norm TEXT PRIMARY KEY,
    alias_raw TEXT NOT NULL,
    ticker TEXT NOT NULL,
    FOREIGN KEY (ticker) REFERENCES tickers(ticker)
);

CREATE TABLE IF NOT EXISTS entity_match_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id TEXT,
    raw_mention TEXT NOT NULL,
    mention_norm TEXT,
    ticker TEXT,
    match_method TEXT NOT NULL,
    match_confidence REAL,
    status TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS unmatched_mentions (
    mention_norm TEXT PRIMARY KEY,
    raw_mention TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 0,
    last_seen DATE,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS stock_prices (
    ticker TEXT NOT NULL,
    trade_date DATE NOT NULL,
    close_price REAL,
    adj_close REAL NOT NULL,
    volume REAL,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS daily_metrics (
    company TEXT NOT NULL,
    date DATE NOT NULL,
    avg_sentiment REAL,
    news_volume INTEGER,
    median_sentiment REAL,
    pos_count INTEGER,
    neg_count INTEGER,
    weighted_sentiment REAL,
    close_price REAL,
    next_close_price REAL,
    pct_change_next_day REAL,
    PRIMARY KEY (company, date)
);

CREATE TABLE IF NOT EXISTS daily_features (
    ticker TEXT NOT NULL,
    date DATE NOT NULL,
    article_count INTEGER,
    mean_sentiment REAL,
    median_sentiment REAL,
    pos_count INTEGER,
    neg_count INTEGER,
    weighted_sentiment REAL,
    rolling_sent_5 REAL,
    rolling_sent_20 REAL,
    volume_mean_20 REAL,
    volume_std_20 REAL,
    volume_z_20 REAL,
    event_score REAL,
    is_trading_day INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS market_events (
    company TEXT NOT NULL,
    date DATE NOT NULL,
    event_type TEXT NOT NULL,
    event_score REAL,
    volume_z_20 REAL,
    mean_sentiment REAL,
    execution_date DATE,
    PRIMARY KEY (company, date, event_type)
);

CREATE TABLE IF NOT EXISTS event_study_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    split TEXT NOT NULL,
    window_name TEXT NOT NULL,
    n_events INTEGER,
    n_tickers INTEGER,
    mean_ar REAL,
    median_ar REAL,
    mean_ar_control REAL,
    median_ar_control REAL,
    diff_mean REAL,
    ci_low REAL,
    ci_high REAL,
    welch_t REAL,
    welch_p REAL,
    mwu_u REAL,
    mwu_p REAL,
    notes TEXT,
    computed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS event_study_observations (
    ticker TEXT NOT NULL,
    event_date DATE NOT NULL,
    execution_date DATE,
    window_name TEXT NOT NULL,
    is_event INTEGER NOT NULL,
    abnormal_return REAL,
    split TEXT NOT NULL,
    PRIMARY KEY (ticker, event_date, window_name, split)
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    trade_date DATE NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    weight REAL,
    gross_return REAL,
    net_return REAL,
    cost_bps REAL,
    PRIMARY KEY (trade_date, ticker)
);

CREATE TABLE IF NOT EXISTS backtest_daily (
    date DATE PRIMARY KEY,
    portfolio_return REAL,
    cumulative_return REAL,
    benchmark_return REAL,
    benchmark_cumulative REAL,
    drawdown REAL,
    n_positions INTEGER,
    turnover REAL
);

CREATE TABLE IF NOT EXISTS backtest_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    split TEXT NOT NULL,
    n_trades INTEGER,
    n_days INTEGER,
    total_return REAL,
    benchmark_total_return REAL,
    annualized_vol REAL,
    sharpe REAL,
    max_drawdown REAL,
    turnover REAL,
    transaction_cost_bps REAL,
    slippage_bps REAL,
    notes TEXT,
    computed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    records INTEGER,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    context TEXT,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_articles_date ON news_articles(publish_date);
CREATE UNIQUE INDEX IF NOT EXISTS idx_mentions_unique ON company_mentions(article_id, company);
CREATE INDEX IF NOT EXISTS idx_mentions_company ON company_mentions(company);
CREATE INDEX IF NOT EXISTS idx_audit_status ON entity_match_audit(status);
CREATE INDEX IF NOT EXISTS idx_prices_date ON stock_prices(trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_metrics(date);
CREATE INDEX IF NOT EXISTS idx_features_score ON daily_features(event_score);
CREATE INDEX IF NOT EXISTS idx_events_date ON market_events(date);
"""


def create_tables(db_path: str | Path, reset: bool = False) -> None:
    """Create schema idempotently. Never drops data unless reset=True."""
    path = Path(db_path)
    with get_connection(path) as conn:
        if reset:
            tables = [
                "backtest_summary",
                "backtest_daily",
                "backtest_trades",
                "event_study_observations",
                "event_study_results",
                "market_events",
                "daily_features",
                "daily_metrics",
                "stock_prices",
                "unmatched_mentions",
                "entity_match_audit",
                "company_mentions",
                "company_aliases",
                "tickers",
                "news_articles",
                "pipeline_runs",
                "ingestion_errors",
            ]
            for table in tables:
                conn.execute(f"DROP TABLE IF EXISTS {table};")
        _migrate_legacy_stock_prices(conn)
        conn.executescript(SCHEMA_SQL)
        _migrate_columns(conn)


def _migrate_legacy_stock_prices(conn) -> None:
    """Older builds used stock_prices(company, date, close_price)."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='stock_prices'"
    ).fetchone()
    if not row:
        return
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(stock_prices)").fetchall()}
    if "trade_date" in cols and "ticker" in cols and "adj_close" in cols:
        return
    conn.execute("ALTER TABLE stock_prices RENAME TO stock_prices_legacy;")
    conn.execute(
        """
        CREATE TABLE stock_prices (
            ticker TEXT NOT NULL,
            trade_date DATE NOT NULL,
            close_price REAL,
            adj_close REAL NOT NULL,
            volume REAL,
            PRIMARY KEY (ticker, trade_date)
        );
        """
    )
    legacy_cols = {r["name"] for r in conn.execute("PRAGMA table_info(stock_prices_legacy)").fetchall()}
    ticker_col = "ticker" if "ticker" in legacy_cols else "company"
    date_col = "trade_date" if "trade_date" in legacy_cols else "date"
    close_col = "close_price" if "close_price" in legacy_cols else "close"
    adj_expr = "adj_close" if "adj_close" in legacy_cols else close_col
    vol_expr = "volume" if "volume" in legacy_cols else "NULL"
    conn.execute(
        f"""
        INSERT OR IGNORE INTO stock_prices (ticker, trade_date, close_price, adj_close, volume)
        SELECT {ticker_col}, {date_col}, {close_col}, COALESCE({adj_expr}, {close_col}), {vol_expr}
        FROM stock_prices_legacy
        WHERE {ticker_col} IS NOT NULL AND {date_col} IS NOT NULL
          AND COALESCE({adj_expr}, {close_col}) IS NOT NULL
        """
    )
    conn.execute("DROP TABLE stock_prices_legacy;")


def _migrate_columns(conn) -> None:
    """Add columns to legacy tables without dropping existing rows."""
    specs = {
        "news_articles": {
            "actor1": "TEXT",
            "actor2": "TEXT",
            "ingested_at": "TEXT",
        },
        "company_mentions": {
            "raw_mention": "TEXT",
            "match_method": "TEXT",
            "match_confidence": "REAL",
        },
        "daily_metrics": {
            "median_sentiment": "REAL",
            "pos_count": "INTEGER",
            "neg_count": "INTEGER",
            "weighted_sentiment": "REAL",
        },
        "market_events": {
            "event_score": "REAL",
            "volume_z_20": "REAL",
            "mean_sentiment": "REAL",
            "execution_date": "DATE",
        },
    }
    for table, cols in specs.items():
        existing = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for col, decl in cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl};")
