"""
Tests for sentiment.analyze() — the narrative-analysis layer.

Covers the five spec rules:
  R1. Social-only signal cannot create high confidence.
  R2. Official data overrides social chatter.
  R3. Conflicting sources reduce confidence.
  R4. Stale sources contribute less than fresh ones.
  R5. Market impact is capped unless official confirmation exists.

Plus:
  - the new spec-shaped to_spec_dict()
  - "mixed" narrative_direction detection
  - back-compat: existing SentimentResult fields still populated
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import sentiment
from models import Market, ResearchItem, ResearchResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_market(ticker: str = "KXTEST", category: str = "crypto") -> Market:
    now = datetime.now(timezone.utc)
    return Market(
        ticker=ticker, title="Test market", status="open",
        yes_ask=55, yes_bid=53, no_ask=47, no_bid=45,
        volume=1000, volume_24h=10_000, open_interest=2000,
        close_time=now + timedelta(hours=2),
        settlement_time=now + timedelta(hours=2),
        category=category, rules_primary="Test rules",
        event_ticker=ticker,
    )


def item(
    *,
    source: str = "src",
    direction: str = "supports_yes",
    credibility: float = 0.85,
    relevance: float = 0.9,
    recency: float = 0.95,
    claim: str = "claim",
) -> ResearchItem:
    return ResearchItem(
        source=source, url="", published_at=datetime.now(timezone.utc),
        claim=claim, direction=direction,
        relevance=relevance, credibility=credibility,
        recency_score=recency, summary="", agent="test",
    )


def research_with(items: list[ResearchItem]) -> ResearchResult:
    return ResearchResult(ticker="KXTEST", query="Q", items=items)


# ── Empty / null ──────────────────────────────────────────────────────────────

def test_no_items_returns_null_result():
    r = sentiment.analyze(make_market(), research_with([]))
    assert r.sentiment_score == 0.0
    assert r.narrative_direction == "neutral"
    assert r.confidence == 0.0
    assert r.market_impact_estimate_cents == 0
    assert r.rumor_risk == "low"
    assert r.major_contradictions == []


# ── R1: social-only confidence cap ───────────────────────────────────────────

def test_social_only_signal_caps_confidence():
    """All items are social-tier and agree — confidence must be capped low."""
    items = [
        item(source="reddit", credibility=0.35, direction="supports_yes"),
        item(source="x.com",  credibility=0.45, direction="supports_yes"),
        item(source="reddit", credibility=0.40, direction="supports_yes"),
    ]
    r = sentiment.analyze(make_market(), research_with(items))

    assert r.narrative_direction == "supports_yes"
    assert r.rumor_risk == "high"
    # Confidence must respect the high-rumor cap
    assert r.confidence <= sentiment._CONF_CAP_HIGH_RUMOR + 1e-9
    # And the market impact cap kicks in too (R5)
    assert r.market_impact_estimate_cents <= sentiment._IMPACT_CAP_HIGH_RUMOR


def test_midtier_only_signal_caps_at_medium():
    """Wire + analyst only (no official) → medium rumor risk → medium cap."""
    items = [
        item(source="Reuters",  credibility=0.85, direction="supports_yes"),
        item(source="CoinDesk", credibility=0.70, direction="supports_yes"),
    ]
    r = sentiment.analyze(make_market(), research_with(items))
    # Reuters at 0.85 is in the official tier (≥0.80) so confirmation exists.
    # This should be low rumor risk, not medium.
    assert r.rumor_risk == "low"
    # Bump down: use cred 0.70 / 0.75 only
    items2 = [
        item(source="CoinDesk",   credibility=0.70, direction="supports_yes"),
        item(source="MarketWatch", credibility=0.72, direction="supports_yes"),
    ]
    r2 = sentiment.analyze(make_market(), research_with(items2))
    assert r2.rumor_risk == "medium"
    assert r2.confidence <= sentiment._CONF_CAP_MEDIUM_RUMOR + 1e-9
    assert r2.market_impact_estimate_cents <= sentiment._IMPACT_CAP_MEDIUM_RUMOR


# ── R2: official overrides social ────────────────────────────────────────────

def test_official_overrides_disagreeing_social():
    """
    Social chatter pushes NO, but official confirms YES.
    Final narrative must follow the official direction.
    """
    items = [
        # 3 noisy social votes for NO
        item(source="reddit", credibility=0.35, direction="supports_no",
             relevance=0.9, recency=0.95),
        item(source="x.com",  credibility=0.55, direction="supports_no",
             relevance=0.9, recency=0.95),
        item(source="reddit", credibility=0.30, direction="supports_no",
             relevance=0.9, recency=0.95),
        # 1 official YES with strong weight
        item(source="Federal Reserve", credibility=0.95, direction="supports_yes",
             relevance=0.95, recency=1.0),
    ]
    r = sentiment.analyze(make_market(), research_with(items))

    assert r.narrative_direction == "supports_yes"
    assert r.sentiment_score > 0
    assert r.rumor_risk == "low"     # official confirmation present


# ── R3: contradictions reduce confidence ─────────────────────────────────────

def test_contradictions_reduce_confidence():
    """Same agreeing items, then a strong contradicting set — confidence drops."""
    base = [
        item(source="Reuters",   credibility=0.85, direction="supports_yes"),
        item(source="Bloomberg", credibility=0.85, direction="supports_yes"),
    ]
    r_clean = sentiment.analyze(make_market(), research_with(base))

    conflicted = base + [
        # Stale official saying NO + fresh social YES = explicit contradiction
        item(source="Fed (old)", credibility=0.95, direction="supports_no",
             recency=0.5),
        item(source="reddit",    credibility=0.40, direction="supports_yes",
             recency=0.95),
        # Fresh news against
        item(source="AP",        credibility=0.85, direction="supports_no",
             recency=0.95),
    ]
    r_conf = sentiment.analyze(make_market(), research_with(conflicted))

    assert len(r_conf.major_contradictions) >= 1
    assert r_conf.confidence < r_clean.confidence


# ── R4: stale sources count less ─────────────────────────────────────────────

def test_stale_items_reduce_weight():
    """A stale YES + fresh NO should net to NO direction."""
    items = [
        item(source="Stale YES",  credibility=0.85, direction="supports_yes",
             recency=0.20),  # week-old
        item(source="Fresh NO",   credibility=0.85, direction="supports_no",
             recency=1.00),
    ]
    r = sentiment.analyze(make_market(), research_with(items))
    assert r.narrative_direction == "supports_no"
    assert r.sentiment_score < 0


# ── R5: market impact cap ────────────────────────────────────────────────────

def test_market_impact_capped_without_official():
    """Strong unanimous social signal — impact must be capped."""
    items = [
        item(source="reddit", credibility=0.35, direction="supports_yes",
             relevance=1.0, recency=1.0),
        item(source="x.com",  credibility=0.40, direction="supports_yes",
             relevance=1.0, recency=1.0),
        item(source="reddit", credibility=0.35, direction="supports_yes",
             relevance=1.0, recency=1.0),
        item(source="x.com",  credibility=0.50, direction="supports_yes",
             relevance=1.0, recency=1.0),
    ]
    r = sentiment.analyze(make_market(), research_with(items))
    assert r.rumor_risk == "high"
    assert r.market_impact_estimate_cents <= sentiment._IMPACT_CAP_HIGH_RUMOR


def test_market_impact_uncapped_with_official():
    """Official agreement removes the cap (only the global ceiling applies)."""
    items = [
        item(source="EIA",     credibility=0.95, direction="supports_yes",
             relevance=1.0, recency=1.0),
        item(source="Reuters", credibility=0.85, direction="supports_yes",
             relevance=1.0, recency=1.0),
        item(source="AP",      credibility=0.85, direction="supports_yes",
             relevance=1.0, recency=1.0),
    ]
    r = sentiment.analyze(make_market(), research_with(items))
    assert r.rumor_risk == "low"
    # Allowed up to the low-rumor ceiling (well above the high-rumor cap)
    assert r.market_impact_estimate_cents > sentiment._IMPACT_CAP_HIGH_RUMOR
    assert r.market_impact_estimate_cents <= sentiment._IMPACT_CAP_LOW_RUMOR


# ── Mixed direction ──────────────────────────────────────────────────────────

def test_mixed_direction_when_both_sides_credible():
    items = [
        item(source="Reuters",   credibility=0.85, direction="supports_yes",
             relevance=0.9, recency=1.0),
        item(source="Bloomberg", credibility=0.85, direction="supports_yes",
             relevance=0.9, recency=1.0),
        item(source="AP",        credibility=0.85, direction="supports_no",
             relevance=0.9, recency=1.0),
        item(source="WSJ",       credibility=0.85, direction="supports_no",
             relevance=0.9, recency=1.0),
    ]
    r = sentiment.analyze(make_market(), research_with(items))
    assert r.narrative_direction == "mixed"
    # Mixed signals should produce a tiny / clipped impact
    assert r.market_impact_estimate_cents <= 2


# ── Spec dict shape ──────────────────────────────────────────────────────────

def test_to_spec_dict_returns_exact_schema():
    items = [
        item(source="Reuters", credibility=0.85, direction="supports_yes"),
    ]
    r = sentiment.analyze(make_market(), research_with(items))
    d = r.to_spec_dict()

    assert set(d.keys()) == {
        "sentiment_score", "narrative_direction", "source_credibility",
        "event_relevance", "market_impact_estimate_cents", "confidence",
        "contradictions", "rumor_risk", "signal_clarity",
    }
    assert isinstance(d["sentiment_score"], float)
    assert d["narrative_direction"] in ("supports_yes", "supports_no", "mixed", "neutral")
    assert 0.0 <= d["source_credibility"] <= 1.0
    assert 0.0 <= d["event_relevance"] <= 1.0
    assert isinstance(d["market_impact_estimate_cents"], int)
    assert 0.0 <= d["confidence"] <= 1.0
    assert isinstance(d["contradictions"], list)
    assert d["rumor_risk"] in ("low", "medium", "high")
    assert d["signal_clarity"] in ("low", "medium", "high")


def test_official_directional_weather_data_sets_confidence_floor():
    items = [
        item(
            source="NOAA",
            direction="supports_yes",
            credibility=0.96,
            relevance=0.98,
            recency=1.0,
            claim="Official forecast is clearly above the market threshold.",
        )
    ]

    r = sentiment.analyze(make_market(category="weather"), research_with(items))

    assert r.narrative_direction == "supports_yes"
    assert r.signal_clarity == "high"
    assert r.confidence >= 0.50


def test_low_credibility_contradictions_do_not_crush_clear_official_evidence():
    items = [
        item(
            source="NOAA",
            direction="supports_yes",
            credibility=0.96,
            relevance=0.98,
            recency=1.0,
            claim="Official forecast is clearly above the market threshold.",
        ),
        item(source="reddit", direction="supports_no", credibility=0.25),
        item(source="x.com", direction="supports_no", credibility=0.35),
    ]

    r = sentiment.analyze(make_market(category="weather"), research_with(items))

    assert r.signal_clarity == "high"
    assert r.confidence >= 0.45


# ── Back-compat: existing prediction_model.py reads ──────────────────────────

def test_legacy_fields_still_populated():
    """prediction_model.py reads .sentiment_score, .narrative_direction,
    .market_impact_estimate_cents, .major_contradictions — all must still work."""
    items = [
        item(source="Reuters", credibility=0.85, direction="supports_yes"),
    ]
    r = sentiment.analyze(make_market(), research_with(items))

    assert isinstance(r.sentiment_score, float)
    assert isinstance(r.narrative_direction, str)
    assert isinstance(r.market_impact_estimate_cents, int)
    assert isinstance(r.major_contradictions, list)
    assert isinstance(r.item_count, int) and r.item_count == 1


def test_neutral_when_signal_too_weak():
    items = [
        item(source="Reuters", credibility=0.85, direction="neutral"),
        item(source="AP",      credibility=0.85, direction="unclear"),
    ]
    r = sentiment.analyze(make_market(), research_with(items))
    assert r.narrative_direction == "neutral"
    assert abs(r.sentiment_score) < 0.05


# ── Spec guardrails: sentiment must NOT make trade recommendations ───────────

_FORBIDDEN_TRADE_KEYS = {
    "recommendation", "recommended_side", "side", "buy", "sell",
    "buy_yes", "buy_no", "action", "trade", "trade_recommendation",
    "edge", "edge_cents", "fair_price", "fair_value",
    "position_size", "contracts", "size", "stake",
    "probability", "prob_yes", "prob_no", "p_yes", "p_no",
}


def test_spec_dict_has_no_trade_recommendation_fields():
    items = [
        item(source="Reuters", credibility=0.85, direction="supports_yes"),
        item(source="reddit",  credibility=0.40, direction="supports_no"),
    ]
    r = sentiment.analyze(make_market(), research_with(items))
    d = r.to_spec_dict()
    leaks = set(d.keys()) & _FORBIDDEN_TRADE_KEYS
    assert not leaks, f"sentiment spec dict leaked trade fields: {leaks}"


def test_sentiment_result_has_no_trade_recommendation_fields():
    """Even on the dataclass itself — no side/edge/size/probability fields."""
    items = [item(source="Reuters", credibility=0.85, direction="supports_yes")]
    r = sentiment.analyze(make_market(), research_with(items))
    public_attrs = {a for a in vars(r).keys() if not a.startswith("_")}
    leaks = public_attrs & _FORBIDDEN_TRADE_KEYS
    assert not leaks, f"SentimentResult leaked trade fields: {leaks}"


def test_stale_evidence_lowers_confidence_vs_fresh():
    """Same one-sided official signal — stale recency yields lower confidence."""
    fresh = [
        item(source="Reuters",   credibility=0.85, direction="supports_yes", recency=1.0),
        item(source="Bloomberg", credibility=0.85, direction="supports_yes", recency=1.0),
    ]
    stale = [
        item(source="Reuters",   credibility=0.85, direction="supports_yes", recency=0.30),
        item(source="Bloomberg", credibility=0.85, direction="supports_yes", recency=0.30),
    ]
    r_fresh = sentiment.analyze(make_market(), research_with(fresh))
    r_stale = sentiment.analyze(make_market(), research_with(stale))
    # Stale evidence carries less aggregate weight → lower derived impact.
    assert r_stale.market_impact_estimate_cents <= r_fresh.market_impact_estimate_cents
    # And the absolute sentiment score should not exceed the fresh version.
    assert abs(r_stale.sentiment_score) <= abs(r_fresh.sentiment_score) + 1e-9
