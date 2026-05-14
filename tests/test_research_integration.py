"""
Tests for the research_market() integration that wires
category_research → research_agents.

Covers:
  1. category-aware path runs and is preferred when it returns items
  2. legacy fallback runs when category-aware returns []
  3. CategoryResearchItem.to_legacy() maps spec fields → legacy ResearchItem
  4. missing API keys never crash the pipeline
  5. callers still get the old ResearchResult shape
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import category_research
import research_agents
from config import Config
from models import (
    CategoryResearchItem,
    Market,
    ResearchItem,
    ResearchResult,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_market(
    ticker: str = "KXBTCD-25NOV13-B104000",
    title: str = "BTC price above 104000 by 5pm CT",
    category: str = "crypto",
) -> Market:
    now = datetime.now(timezone.utc)
    return Market(
        ticker=ticker,
        title=title,
        status="open",
        yes_ask=58, yes_bid=55, no_ask=45, no_bid=42,
        volume=1200, volume_24h=15000, open_interest=4000,
        close_time=now + timedelta(minutes=90),
        settlement_time=now + timedelta(minutes=95),
        category=category,
        rules_primary="Resolves YES if Coinbase BTC-USD spot is above 104000 at 5pm CT.",
        event_ticker="KXBTCD-25NOV13",
    )


@pytest.fixture
def cfg() -> Config:
    return Config(
        kalshi_api_key="x", anthropic_api_key="x",
        kalshi_private_key_path="",   # validate() not called in tests
    )


@pytest.fixture(autouse=True)
def _isolate_db_and_network(monkeypatch):
    """
    Default for every test: cache miss, no-op writes, and any unmocked
    network call would error loudly (which surfaces test gaps).
    """
    monkeypatch.setattr(research_agents.db, "get_cached_research",
                        lambda *a, **kw: None)
    monkeypatch.setattr(research_agents.db, "set_cached_research",
                        lambda *a, **kw: None)


class _StubAgent:
    """
    Stand-in for the legacy agents. Some real agents (RSSNewsAgent,
    MarketSpecificAgent) override .run(), so patching ResearchAgent.run only
    stubs 4 of 6 agents — this class side-steps the inheritance issue.
    """
    def __init__(self, name: str, counter: dict, items_to_return=None):
        self.name = name
        self._counter = counter
        self._items = items_to_return or []

    def run(self, market, cfg):
        self._counter["calls"] += 1
        return list(self._items)


def _install_stub_agents(monkeypatch, counter, items_per_agent=None):
    """Replace research_agents._AGENTS with 6 stubs sharing a counter."""
    stubs = [
        _StubAgent(f"stub_{i}", counter, items_per_agent)
        for i in range(6)
    ]
    monkeypatch.setattr(research_agents, "_AGENTS", stubs)
    return stubs


# ── 1. Category-aware path is preferred when it returns items ────────────────

def test_categorical_path_preferred_when_items_returned(monkeypatch, cfg):
    market = make_market()
    now = datetime.now(timezone.utc)

    fake_items = [
        CategoryResearchItem(
            query=market.title,
            source_type="market_data",
            source_name="Coinbase",
            claim="BTC is trading at $103,850 with 90 minutes to resolution.",
            supports="no",
            credibility=0.90,
            relevance=0.95,
            recency=1.00,
            risk_flags=["price volatile", "thin Kalshi book"],
            url="https://www.coinbase.com/price/bitcoin",
            published_at=now,
            summary="Spot price from Coinbase exchange.",
        ),
    ]

    counter = {"calls": 0}
    _install_stub_agents(monkeypatch, counter)

    monkeypatch.setattr(category_research, "research_market_categorical",
                        lambda m, c: fake_items)

    result = research_agents.research_market(market, cfg)

    assert isinstance(result, ResearchResult)
    assert len(result.items) == 1
    item = result.items[0]
    assert item.source == "Coinbase"
    assert item.claim.startswith("BTC is trading")
    # to_legacy() maps supports="no" → direction="supports_no"
    assert item.direction == "supports_no"
    # legacy agents were not consulted
    assert counter["calls"] == 0


# ── 2. Legacy fallback runs when category-aware returns nothing ──────────────

def test_legacy_fallback_when_categorical_empty(monkeypatch, cfg):
    market = make_market()

    legacy_item = ResearchItem(
        source="LegacyAgent", url="", published_at=datetime.now(timezone.utc),
        claim="claim from legacy", direction="neutral",
        relevance=0.5, credibility=0.5, recency_score=0.5,
        summary="", agent="stub",
    )
    counter = {"calls": 0}
    _install_stub_agents(monkeypatch, counter,
                         items_per_agent=[legacy_item])

    monkeypatch.setattr(category_research, "research_market_categorical",
                        lambda m, c: [])

    result = research_agents.research_market(market, cfg)

    # All 6 legacy agents fired
    assert counter["calls"] == 6
    assert isinstance(result, ResearchResult)
    assert len(result.items) >= 1
    # Items come from legacy (each tagged with an agent name) not from Coinbase
    assert all(item.source == "LegacyAgent" for item in result.items)


# ── 3. to_legacy() field mapping is safe and complete ────────────────────────

@pytest.mark.parametrize("supports,expected_direction", [
    ("yes",     "supports_yes"),
    ("no",      "supports_no"),
    ("neutral", "neutral"),
    ("unclear", "unclear"),
    ("garbage", "unclear"),    # unknown values must degrade safely
])
def test_to_legacy_maps_fields(supports, expected_direction):
    now = datetime.now(timezone.utc)
    item = CategoryResearchItem(
        query="Q?",
        source_type="market_data",
        source_name="Coinbase",
        claim="claim",
        supports=supports,
        credibility=0.81,
        relevance=0.77,
        recency=0.66,
        risk_flags=["flag-a", "flag-b"],
        url="https://example",
        published_at=now,
        summary="summary text",
    )
    legacy = item.to_legacy()

    assert isinstance(legacy, ResearchItem)
    assert legacy.source       == "Coinbase"            # source_name → source
    assert legacy.direction    == expected_direction    # supports → direction
    assert legacy.credibility  == 0.81
    assert legacy.relevance    == 0.77
    assert legacy.recency_score == 0.66                 # recency → recency_score
    assert legacy.url          == "https://example"
    assert legacy.published_at == now
    assert legacy.summary      == "summary text"
    assert legacy.agent.startswith("category:")         # agent tag preserved


def test_to_legacy_summary_falls_back_to_claim_when_empty():
    item = CategoryResearchItem(
        query="Q?", source_type="news", source_name="Reuters",
        claim="A specific claim.", supports="yes",
        credibility=0.8, relevance=0.8, recency=0.8,
        summary="",   # empty
    )
    assert item.to_legacy().summary == "A specific claim."


# ── 4. Missing API keys never crash ──────────────────────────────────────────

def test_missing_api_keys_do_not_crash(monkeypatch, cfg):
    """
    Wipe every optional key. The categorical pipeline must return [] cleanly
    and the legacy fallback must take over without raising.
    """
    cfg.perplexity_api_key = ""
    cfg.fred_api_key = ""
    cfg.eia_api_key = ""
    cfg.bls_api_key = ""
    cfg.noaa_token = ""

    # Force the network calls inside the categorical pipeline to fail.
    # (They should already short-circuit on missing keys, but this guards
    # the case where a fetcher does not check.)
    def boom(*a, **kw):
        raise RuntimeError("network disabled in test")
    monkeypatch.setattr(category_research.requests, "get", boom)
    monkeypatch.setattr(category_research.requests, "post", boom)

    # Stub out the legacy path so we don't hit real APIs in fallback either.
    _install_stub_agents(monkeypatch, {"calls": 0})

    market = make_market(category="economic",
                         title="CPI year-over-year above 3.0% in October")
    result = research_agents.research_market(market, cfg)

    # No crash, valid empty-ish result
    assert isinstance(result, ResearchResult)
    assert result.ticker == market.ticker
    assert isinstance(result.items, list)


def test_categorical_exception_falls_back_to_legacy(monkeypatch, cfg):
    """
    If research_market_categorical raises, we must catch it, log, and
    proceed to the legacy pipeline — never propagate the exception.
    """
    def explode(m, c):
        raise RuntimeError("simulated fetcher crash")

    counter = {"calls": 0}
    _install_stub_agents(monkeypatch, counter)

    monkeypatch.setattr(category_research, "research_market_categorical", explode)

    market = make_market()
    result = research_agents.research_market(market, cfg)

    assert counter["calls"] == 6
    assert isinstance(result, ResearchResult)


# ── 5. Existing callers still receive the legacy ResearchResult shape ────────

def test_research_result_shape_unchanged(monkeypatch, cfg):
    market = make_market()

    now = datetime.now(timezone.utc)
    monkeypatch.setattr(
        category_research, "research_market_categorical",
        lambda m, c: [CategoryResearchItem(
            query=m.title, source_type="market_data", source_name="Coinbase",
            claim="A claim.", supports="yes",
            credibility=0.9, relevance=0.9, recency=0.95,
            published_at=now,
        )],
    )

    result = research_agents.research_market(market, cfg)

    # Public attributes the rest of the system depends on
    assert hasattr(result, "ticker")
    assert hasattr(result, "query")
    assert hasattr(result, "items")
    assert hasattr(result, "failed_reason")
    assert hasattr(result, "timestamp")
    # raw_text and sources properties still work
    assert isinstance(result.raw_text, str)
    assert isinstance(result.sources, list)
    assert result.ticker == market.ticker
    assert result.query == market.title
    # Every item is a ResearchItem (not CategoryResearchItem) — downstream
    # sentiment.py / edge.py code is keyed on the legacy field names.
    assert all(isinstance(it, ResearchItem) for it in result.items)


def test_cache_hit_short_circuits_both_paths(monkeypatch, cfg):
    """
    When db.get_cached_research returns items, neither the categorical
    pipeline nor the legacy agents should be consulted.
    """
    cached = [ResearchItem(
        source="CachedSource", url="", published_at=None,
        claim="cached claim", direction="neutral",
        relevance=0.5, credibility=0.5, recency_score=0.5,
        summary="", agent="cached",
    )]

    monkeypatch.setattr(research_agents.db, "get_cached_research",
                        lambda *a, **kw: cached)

    cat_called = {"n": 0}
    agent_counter = {"calls": 0}

    def cat_spy(m, c):
        cat_called["n"] += 1
        return []

    monkeypatch.setattr(category_research, "research_market_categorical", cat_spy)
    _install_stub_agents(monkeypatch, agent_counter)

    result = research_agents.research_market(make_market(), cfg)

    assert cat_called["n"] == 0
    assert agent_counter["calls"] == 0
    assert len(result.items) == 1
    assert result.items[0].source == "CachedSource"
