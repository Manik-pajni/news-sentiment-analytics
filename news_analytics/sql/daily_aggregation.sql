-- Primary daily aggregation (SQLite). Median is computed in Python because
-- SQLite has no PERCENTILE_CONT/MEDIAN aggregate.
-- Weighted sentiment: tone-weighted mean using abs(tone) as weight.

DELETE FROM daily_metrics;

INSERT INTO daily_metrics (
    company,
    date,
    avg_sentiment,
    news_volume,
    pos_count,
    neg_count,
    weighted_sentiment
)
SELECT
    cm.company AS company,
    na.publish_date AS date,
    AVG(na.sentiment) AS avg_sentiment,
    COUNT(*) AS news_volume,
    SUM(CASE WHEN na.sentiment > 0 THEN 1 ELSE 0 END) AS pos_count,
    SUM(CASE WHEN na.sentiment < 0 THEN 1 ELSE 0 END) AS neg_count,
    CASE
        WHEN SUM(ABS(COALESCE(na.tone, 0))) > 0
        THEN SUM(na.sentiment * ABS(COALESCE(na.tone, 0))) / SUM(ABS(COALESCE(na.tone, 0)))
        ELSE AVG(na.sentiment)
    END AS weighted_sentiment
FROM news_articles na
JOIN company_mentions cm
    ON na.article_id = cm.article_id
GROUP BY cm.company, na.publish_date;
