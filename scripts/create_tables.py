from news_analytics.config import load_settings
from news_analytics.ingest.universe import seed_universe
from news_analytics.schema import create_tables as _create


DATABASE = "db/news.db"


def create_tables(db_path: str = DATABASE, reset: bool = False) -> None:
    settings = load_settings()
    path = db_path
    _create(path, reset=reset)
    seed_universe(path, settings.ticker_universe_path, settings.alias_path)


if __name__ == "__main__":
    create_tables()
    print("All tables created successfully.")
