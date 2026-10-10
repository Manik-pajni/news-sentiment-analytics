from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from news_analytics.config import ROOT, load_settings
from news_analytics.entity.resolver import load_universe
from news_analytics.ingest.universe import seed_universe
from news_analytics.schema import create_tables


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    create_tables(path)
    settings = load_settings()
    seed_universe(path, settings.ticker_universe_path, settings.alias_path)
    return path


@pytest.fixture
def universe():
    settings = load_settings()
    return load_universe(settings.ticker_universe_path, settings.alias_path)


@pytest.fixture
def project_root() -> Path:
    return ROOT
