from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

from news_analytics.db import get_connection
from news_analytics.entity.resolver import Universe, load_universe, resolve_mention

logger = logging.getLogger("news_analytics")

# official daily events file: tab-separated, no header
COL_EVENT_ID = 0
COL_SQLDATE = 1
COL_ACTOR1 = 6
COL_ACTOR2 = 16
COL_AVGTONE = 34
COL_SOURCE_URL = 57

GDELT_EVENTS_URL = "http://data.gdeltproject.org/events/{ymd}.export.CSV.zip"


def _parse_gdelt_date(value) -> str | None:
    s = str(value)
    if len(s) < 8 or not s[:8].isdigit():
        return None
    try:
        datetime.strptime(s[:8], "%Y%m%d")
    except ValueError:
        return None
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def _tone_to_sentiment(tone: float) -> float:
    """AvgTone / 10, then clip to [-1, 1]."""
    try:
        return max(-1.0, min(1.0, float(tone) / 10.0))
    except (TypeError, ValueError):
        return 0.0


def date_range(start: str, end: str) -> list[date]:
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    if d1 < d0:
        raise ValueError("end_date must be on or after start_date")
    days = []
    cur = d0
    while cur <= d1:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def download_gdelt_daily(gdelt_dir: Path, start: str, end: str) -> int:
    """Pull official GDELT daily files."""
    gdelt_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for d in date_range(start, end):
        ymd = d.strftime("%Y%m%d")
        dest = gdelt_dir / f"{ymd}.export.CSV"
        if dest.exists():
            saved += 1
            continue  # skip days we already have
        url = GDELT_EVENTS_URL.format(ymd=ymd)
        try:
            with urlopen(url, timeout=60) as resp:
                payload = resp.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                name = zf.namelist()[0]
                dest.write_bytes(zf.read(name))
            saved += 1
            logger.info("Downloaded GDELT events for %s", ymd)
        except (HTTPError, URLError, zipfile.BadZipFile, OSError) as exc:
            logger.warning("Could not download GDELT %s: %s", ymd, exc)
    return saved


def load_raw_records(gdelt_dir: Path) -> pd.DataFrame:
    rows: list[tuple] = []
    if not gdelt_dir.is_dir():
        return pd.DataFrame(
            columns=["article_id", "publish_date", "actor1", "actor2", "tone", "source"]
        )
    files = sorted(p for p in gdelt_dir.iterdir() if p.suffix.upper() == ".CSV")
    for file_path in files:
        try:
            df = pd.read_csv(
                file_path,
                sep="\t",
                header=None,
                low_memory=False,
                dtype=str,
                on_bad_lines="skip",
            )
        except Exception as exc:
            logger.warning("Skipping malformed GDELT file %s: %s", file_path, exc)
            continue
        if df.empty:
            continue
        for _, r in df.iterrows():
            article_id = str(r.get(COL_EVENT_ID, "") or "").strip()
            publish_date = _parse_gdelt_date(r.get(COL_SQLDATE, ""))
            if not article_id or not publish_date:
                continue
            try:
                tone = float(r.get(COL_AVGTONE, 0.0) or 0.0)
            except (TypeError, ValueError):
                tone = 0.0
            actor1 = str(r.get(COL_ACTOR1, "") or "")
            actor2 = str(r.get(COL_ACTOR2, "") or "")
            source = str(r.get(COL_SOURCE_URL, "") or "")
            rows.append((article_id, publish_date, actor1, actor2, tone, source))
    return pd.DataFrame(rows, columns=["article_id", "publish_date", "actor1", "actor2", "tone", "source"])


def _upsert_unmatched(conn, mention_norm: str, raw: str, reason: str, publish_date: str) -> None:
    conn.execute(
        """
        INSERT INTO unmatched_mentions (mention_norm, raw_mention, mention_count, last_seen, reason)
        VALUES (?, ?, 1, ?, ?)
        ON CONFLICT(mention_norm) DO UPDATE SET
            mention_count = mention_count + 1,
            last_seen = excluded.last_seen,
            reason = excluded.reason
        """,
        (mention_norm, raw, publish_date, reason),
    )


def ingest_gdelt(
    db_path: Path,
    gdelt_dir: Path,
    universe: Universe,
    *,
    fuzzy_min_score: float,
    ambiguity_gap: float,
    min_name_length: int,
) -> dict[str, int]:
    """Read every local GDELT CSV into SQLite."""
    raw = load_raw_records(gdelt_dir)
    if raw.empty:
        logger.warning("No GDELT CSV files found in %s. Ingesting 0 events.", gdelt_dir)
        return {"files_records": 0, "articles": 0, "mapped_mentions": 0, "unmatched": 0}

    raw = raw.drop_duplicates(subset=["article_id"])

    articles_inserted = 0
    mapped = 0
    unmatched = 0
    with get_connection(db_path) as conn:
        article_rows = []
        for r in raw.itertuples(index=False):
            text = f"{r.actor1} {r.actor2} {r.source}".strip()
            article_rows.append(
                (
                    r.article_id,
                    r.publish_date,
                    r.source,
                    text,
                    float(r.tone),
                    _tone_to_sentiment(float(r.tone)),
                    r.actor1,
                    r.actor2,
                )
            )
        existing = {
            row["article_id"]
            for row in conn.execute("SELECT article_id FROM news_articles").fetchall()
        }
        conn.executemany(
            """
            INSERT OR IGNORE INTO news_articles
                (article_id, publish_date, source, text, tone, sentiment, actor1, actor2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            article_rows,
        )
        new_ids = {r.article_id for r in raw.itertuples(index=False) if r.article_id not in existing}
        articles_inserted = len(new_ids)

        for r in raw.itertuples(index=False):
            if r.article_id not in new_ids:
                continue
            for actor in (r.actor1, r.actor2):
                if not str(actor).strip() or str(actor).lower() == "nan":
                    continue
                result = resolve_mention(
                    str(actor),
                    universe,
                    fuzzy_min_score=fuzzy_min_score,
                    ambiguity_gap=ambiguity_gap,
                    min_name_length=min_name_length,
                )
                conn.execute(
                    """
                    INSERT INTO entity_match_audit
                        (article_id, raw_mention, mention_norm, ticker, match_method, match_confidence, status, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r.article_id,
                        result.raw_mention,
                        result.mention_norm,
                        result.ticker,
                        result.method,
                        result.confidence,
                        result.status,
                        result.reason,
                    ),
                )
                if result.status == "mapped" and result.ticker:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO company_mentions
                            (article_id, company, raw_mention, match_method, match_confidence)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            r.article_id,
                            result.ticker,
                            result.raw_mention,
                            result.method,
                            result.confidence,
                        ),
                    )
                    mapped += 1
                else:
                    unmatched += 1
                    _upsert_unmatched(
                        conn,
                        result.mention_norm or str(actor),
                        str(actor),
                        result.reason,
                        r.publish_date,
                    )

    stats = {
        "files_records": int(len(raw)),
        "articles": int(articles_inserted),
        "mapped_mentions": int(mapped),
        "unmatched": int(unmatched),
    }
    logger.info("GDELT ingest stats: %s", stats)
    return stats
