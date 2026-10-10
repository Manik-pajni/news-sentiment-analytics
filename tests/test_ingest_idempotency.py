from pathlib import Path

from news_analytics.db import connect
from news_analytics.ingest.gdelt import download_gdelt_daily, ingest_gdelt


def _row(event_id: str, actor: str, tone: str, ymd: str = "20240102") -> str:
    cols = [""] * 58
    cols[0] = event_id
    cols[1] = ymd
    cols[6] = actor
    cols[34] = tone
    cols[57] = f"http://example.com/{event_id}"
    return "\t".join(cols)


def _write_csv(folder: Path) -> None:
    text = "\n".join(
        [
            _row("1", "Apple", "2.0"),
            _row("2", "Microsoft", "-1.0"),
        ]
    )
    (folder / "20240102.export.CSV").write_text(text + "\n")


def test_ingest_idempotent(db_path, universe, tmp_path):
    gdelt_dir = tmp_path / "gdelt"
    gdelt_dir.mkdir()
    _write_csv(gdelt_dir)
    ingest_gdelt(db_path, gdelt_dir, universe, fuzzy_min_score=92, ambiguity_gap=5, min_name_length=6)
    ingest_gdelt(db_path, gdelt_dir, universe, fuzzy_min_score=92, ambiguity_gap=5, min_name_length=6)
    conn = connect(db_path)
    n_articles = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
    n_mentions = conn.execute("SELECT COUNT(*) FROM company_mentions").fetchone()[0]
    conn.close()
    assert n_articles == 2
    assert n_mentions == 2


def test_ingest_all_local_csvs_no_date_filter(db_path, universe, tmp_path):
    gdelt_dir = tmp_path / "gdelt"
    gdelt_dir.mkdir()
    (gdelt_dir / "20240102.export.CSV").write_text(_row("1", "Apple", "2.0", "20240102") + "\n")
    (gdelt_dir / "20240615.export.CSV").write_text(_row("2", "Microsoft", "-1.0", "20240615") + "\n")
    ingest_gdelt(db_path, gdelt_dir, universe, fuzzy_min_score=92, ambiguity_gap=5, min_name_length=6)
    conn = connect(db_path)
    dates = {row[0] for row in conn.execute("SELECT publish_date FROM news_articles")}
    conn.close()
    assert dates == {"2024-01-02", "2024-06-15"}


def test_download_skips_existing_files(tmp_path):
    gdelt_dir = tmp_path / "gdelt"
    gdelt_dir.mkdir()
    dest = gdelt_dir / "20240102.export.CSV"
    dest.write_text("already here")
    saved = download_gdelt_daily(gdelt_dir, "2024-01-02", "2024-01-02")
    assert saved == 1
    assert dest.read_text() == "already here"
