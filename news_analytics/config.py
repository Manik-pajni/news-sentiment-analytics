"""Environment-driven configuration. Secrets never belong in source control."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return default if raw is None else float(raw)


@dataclass(frozen=True)
class Settings:
    database_path: Path
    gdelt_dir: Path
    ticker_universe_path: Path
    alias_path: Path
    benchmark_ticker: str
    yf_timeout_seconds: int
    yf_max_retries: int
    fuzzy_min_score: float
    fuzzy_ambiguity_gap: float
    fuzzy_min_name_length: int
    transaction_cost_bps: float
    slippage_bps: float
    max_positions: int
    train_fraction: float
    log_level: str
    log_file: Path


def load_settings() -> Settings:
    db = Path(_env("DATABASE_PATH", str(ROOT / "db" / "news.db")))
    if not db.is_absolute():
        db = ROOT / db
    gdelt = Path(_env("GDELT_DIR", str(ROOT / "data" / "raw_gdelt")))
    if not gdelt.is_absolute():
        gdelt = ROOT / gdelt
    log_file = Path(_env("LOG_FILE", str(ROOT / "logs" / "pipeline.log")))
    if not log_file.is_absolute():
        log_file = ROOT / log_file
    return Settings(
        database_path=db,
        gdelt_dir=gdelt,
        ticker_universe_path=ROOT / "config" / "tickers.csv",
        alias_path=ROOT / "config" / "aliases.csv",
        benchmark_ticker=_env("BENCHMARK_TICKER", "SPY") or "SPY",
        yf_timeout_seconds=_env_int("YF_TIMEOUT_SECONDS", 30),
        yf_max_retries=_env_int("YF_MAX_RETRIES", 3),
        fuzzy_min_score=_env_float("FUZZY_MIN_SCORE", 92.0),
        fuzzy_ambiguity_gap=_env_float("FUZZY_AMBIGUITY_GAP", 5.0),
        fuzzy_min_name_length=_env_int("FUZZY_MIN_NAME_LENGTH", 6),
        transaction_cost_bps=_env_float("TRANSACTION_COST_BPS", 10.0),
        slippage_bps=_env_float("SLIPPAGE_BPS", 5.0),
        max_positions=_env_int("MAX_POSITIONS", 10),
        train_fraction=_env_float("TRAIN_FRACTION", 0.7),
        log_level=_env("LOG_LEVEL", "INFO") or "INFO",
        log_file=log_file,
    )
