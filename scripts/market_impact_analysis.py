from news_analytics.config import load_settings
from news_analytics.reporting import q

DATABASE = "db/news.db"


def print_market_impact(db_path: str = DATABASE, limit: int = 20) -> None:
    df = q(
        db_path,
        """
        SELECT
            dm.company,
            dm.date,
            dm.avg_sentiment,
            dm.news_volume,
            dm.pct_change_next_day,
            GROUP_CONCAT(me.event_type, '|') AS events
        FROM daily_metrics dm
        LEFT JOIN market_events me
            ON dm.company = me.company AND dm.date = me.date
        GROUP BY dm.company, dm.date
        ORDER BY dm.news_volume DESC, ABS(dm.avg_sentiment) DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    if df.empty:
        print("No results to display.")
        return
    print(df.to_string(index=False))


if __name__ == "__main__":
    print_market_impact()
