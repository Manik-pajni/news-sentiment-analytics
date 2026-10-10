"""Backward-compatible GDELT ingest entrypoint (no synthetic duplication)."""

from news_analytics.config import load_settings
from news_analytics.entity.resolver import load_universe
from news_analytics.ingest.gdelt import ingest_gdelt as _ingest
from news_analytics.schema import create_tables

DATABASE = "db/news.db"


def get_company_universe():
    settings = load_settings()
    u = load_universe(settings.ticker_universe_path, settings.alias_path)
    return dict(zip(u.tickers["ticker"], u.tickers["ticker"]))


def ingest_gdelt(db_path: str = DATABASE) -> None:
    settings = load_settings()
    create_tables(db_path)
    universe = load_universe(settings.ticker_universe_path, settings.alias_path)
    stats = _ingest(
        db_path,
        settings.gdelt_dir,
        universe,
        fuzzy_min_score=settings.fuzzy_min_score,
        ambiguity_gap=settings.fuzzy_ambiguity_gap,
        min_name_length=settings.fuzzy_min_name_length,
    )
    print("Ingestion complete.", stats)


if __name__ == "__main__":
    ingest_gdelt()
