from news_analytics.entity.resolver import resolve_mention


def test_deterministic_alias(universe):
    m = resolve_mention("Apple Inc", universe)
    assert m.ticker == "AAPL"
    assert m.method == "deterministic_alias"
    assert m.status == "mapped"
    assert m.confidence == 1.0


def test_short_token_not_fuzzy_matched(universe):
    m = resolve_mention("US", universe)
    assert m.status == "unmatched"
    assert m.ticker is None


def test_generic_rejected(universe):
    m = resolve_mention("United States", universe)
    assert m.status == "unmatched"
    assert m.reason == "generic_or_empty"


def test_ambiguous_or_unmatched_noise(universe):
    m = resolve_mention("Completely Unknown Conglomerate Holdings Worldwide", universe)
    assert m.status in {"unmatched", "review"}
    assert m.ticker is None


def test_fuzzy_microsoft(universe):
    m = resolve_mention("Microsoft Corporaton", universe, fuzzy_min_score=88, min_name_length=6)
    assert m.ticker == "MSFT"
    assert m.method == "fuzzy"
    assert m.status == "mapped"
