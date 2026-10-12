from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from news_analytics.entity.normalize import clean_mention, is_generic

# don't fuzzy-match 1-2 letter tickers
SHORT_TICKER_MAX_LEN = 2


@dataclass(frozen=True)
class MatchResult:
    ticker: str | None
    method: str
    confidence: float
    status: str
    reason: str
    raw_mention: str
    mention_norm: str


@dataclass
class Universe:
    tickers: pd.DataFrame
    alias_to_ticker: dict[str, str]
    names: list[tuple[str, str]]  # (normalized company name, ticker)


def load_universe(ticker_path: Path, alias_path: Path) -> Universe:
    tickers = pd.read_csv(ticker_path)
    aliases = pd.read_csv(alias_path)
    alias_map: dict[str, str] = {}
    for row in aliases.itertuples(index=False):
        alias_map[clean_mention(str(row.alias))] = str(row.ticker)
    for row in tickers.itertuples(index=False):
        alias_map[clean_mention(str(row.ticker))] = str(row.ticker)
        alias_map[clean_mention(str(row.company_name))] = str(row.ticker)
    names = [
        (clean_mention(str(row.company_name)), str(row.ticker))
        for row in tickers.itertuples(index=False)
        if str(row.ticker) != "SPY"
    ]
    return Universe(tickers=tickers, alias_to_ticker=alias_map, names=names)


def resolve_mention(
    raw: str,
    universe: Universe,
    *,
    fuzzy_min_score: float = 92.0,
    ambiguity_gap: float = 5.0,
    min_name_length: int = 6,
) -> MatchResult:
    mention_norm = clean_mention(raw)
    if is_generic(mention_norm):
        return MatchResult(None, "rejected", 0.0, "unmatched", "generic_or_empty", raw, mention_norm)

    if mention_norm in universe.alias_to_ticker:
        ticker = universe.alias_to_ticker[mention_norm]
        # exact alias match is fine even for short tickers
        return MatchResult(ticker, "deterministic_alias", 1.0, "mapped", "exact_alias", raw, mention_norm)

    tokens = mention_norm.split()
    if len(tokens) == 1 and mention_norm in universe.alias_to_ticker:
        ticker = universe.alias_to_ticker[mention_norm]
        return MatchResult(ticker, "deterministic_alias", 1.0, "mapped", "exact_token", raw, mention_norm)

    # skip fuzzy if the mention is too short
    if len(mention_norm) < min_name_length or (len(tokens) == 1 and len(mention_norm) <= SHORT_TICKER_MAX_LEN):
        return MatchResult(
            None, "rejected", 0.0, "unmatched", "too_short_for_fuzzy", raw, mention_norm
        )

    name_choices = [n for n, _ in universe.names if len(n) >= min_name_length]
    if not name_choices:
        return MatchResult(None, "unmatched", 0.0, "unmatched", "empty_universe", raw, mention_norm)

    scored = process.extract(mention_norm, name_choices, scorer=fuzz.token_sort_ratio, limit=3)
    if not scored:
        return MatchResult(None, "unmatched", 0.0, "unmatched", "no_fuzzy_candidates", raw, mention_norm)

    best_name, best_score, _ = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    name_to_ticker = {n: t for n, t in universe.names}

    if best_score < fuzzy_min_score:
        return MatchResult(
            None, "fuzzy", float(best_score) / 100.0, "unmatched", "below_threshold", raw, mention_norm
        )
    if best_score - second_score < ambiguity_gap and second_score >= fuzzy_min_score - 2:
        return MatchResult(
            None,
            "fuzzy",
            float(best_score) / 100.0,
            "review",
            "ambiguous_fuzzy",
            raw,
            mention_norm,
        )
    ticker = name_to_ticker[best_name]
    return MatchResult(
        ticker, "fuzzy", float(best_score) / 100.0, "mapped", "fuzzy_name", raw, mention_norm
    )
