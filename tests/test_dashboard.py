from pathlib import Path

from news_analytics.reporting import mapping_stats, overview
from news_analytics.schema import create_tables


def test_dashboard_loads_empty(tmp_path: Path):
    db = tmp_path / "empty.db"
    create_tables(db)
    stats = overview(db)
    assert stats["n_articles"] == 0
    assert stats["n_high_impact"] == 0
    m = mapping_stats(db)
    assert m["mapped"] == 0
    assert m["coverage"] is None
