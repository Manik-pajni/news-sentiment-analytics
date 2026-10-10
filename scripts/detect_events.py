from news_analytics.analytics.aggregate import build_daily_features
from news_analytics.analytics.events import detect_events as _detect
from news_analytics.config import load_settings

DATABASE = "db/news.db"


def detect_events(db_path: str = DATABASE) -> None:
    settings = load_settings()
    build_daily_features(db_path, settings.benchmark_ticker)
    stats = _detect(db_path, settings.benchmark_ticker, settings.train_fraction)
    print("Event detection completed.", stats)


if __name__ == "__main__":
    detect_events()
