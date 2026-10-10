from news_analytics.analytics.aggregate import aggregate_daily_metrics as _agg

DATABASE = "db/news.db"


def aggregate_daily_metrics(db_path: str = DATABASE) -> None:
    n = _agg(db_path)
    print(f"Daily metrics aggregation completed ({n} rows).")


if __name__ == "__main__":
    aggregate_daily_metrics()
