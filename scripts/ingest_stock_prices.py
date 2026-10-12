"""Old entrypoint for Yahoo prices."""

from datetime import datetime, timedelta

from news_analytics.analytics.aggregate import attach_next_day_prices
from news_analytics.config import load_settings
from news_analytics.db import get_connection
from news_analytics.ingest.prices import ingest_stock_prices, mapped_tickers

DATABASE = "db/news.db"


def update_daily_market_impact(db_path: str = DATABASE) -> None:
    settings = load_settings()
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT MIN(publish_date) AS a, MAX(publish_date) AS b FROM news_articles"
        ).fetchone()
    if not row["a"]:
        print("No articles; skipping prices.")
        return
    start = (datetime.strptime(row["a"], "%Y-%m-%d") - timedelta(days=90)).date().isoformat()
    end = (datetime.strptime(row["b"], "%Y-%m-%d") + timedelta(days=10)).date().isoformat()
    tickers = mapped_tickers(db_path, settings.benchmark_ticker)
    stats = ingest_stock_prices(db_path, tickers, start, end)
    attach_next_day_prices(db_path, settings.benchmark_ticker)
    print("Market prices updated.", stats)


if __name__ == "__main__":
    update_daily_market_impact()
