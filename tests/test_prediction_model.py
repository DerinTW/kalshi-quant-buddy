"""
Tests for prediction_model.estimate() — the wired-up feature consumer.

Verifies that the feature vector built by features.extract_features is used
in a controlled way: as confidence modifiers, with capped probability
adjustments, and with fail-soft behavior on missing inputs.

The LLM call is monkeypatched in every test so the suite is hermetic and
deterministic. By default the LLM is stubbed to return p_market with
"medium-high" confidence — that gives every test a high enough starting
point that step-downs are measurable.
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import edge
import llm
import prediction_model as pm
from config import Config
from models import (
    Market,
    ProbabilityEstimate,
    ResearchResult,
    ResearchItem,
    SentimentResult,
    WeirdMoveSignal,
)


# ── Fixtures / builders ──────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> Config:
    return Config(
        kalshi_api_key="x", anthropic_api_key="x",
        kalshi_private_key_path="",
    )


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch):
    """
    Default LLM stub: returns the market midpoint with medium-high confidence
    so we can observe rule-layer step-downs cleanly. Tests that need a
    different LLM behavior override this.

    Computes mid from (yes_ask + (100 - no_ask)) / 2, since that mirrors
    p_market in production code.
    """
    def fake(cfg, ticker, title, rules, yes_ask, no_ask, research_text, sentiment_data):
        yes_bid = 100 - no_ask
        mid = (yes_ask + yes_bid) / 2.0 if yes_bid > 0 else float(yes_ask)
        return {
            "yes_probability": mid / 100.0,
            "confidence": "medium-high",
            "reasoning": "stub",
            "assumptions": [],
            "invalidation_conditions": [],
        }
    monkeypatch.setattr(llm, "estimate_probability", fake)


def make_market(
    *,
    ticker: str = "KXBTCD-25NOV13-B104000",
    title: str = "BTC price above 104000 by 5pm CT",
    yes_ask: int = 55,
    yes_bid: int = 53,
    volume_24h: int = 12_000,
    open_interest: int = 4_000,
    orderbook_depth: int = 300,
    liquidity_dollars: float = 5_000.0,
    minutes_to_settlement: float = 90.0,
    price_history: list[dict] | None = None,
    category: str = "crypto",
    event_ticker: str | None = None,
) -> Market:
    if price_history is None:
        price_history = [
            {"yes_price": 54}, {"yes_price": 55}, {"yes_price": 53},
            {"yes_price": 54}, {"yes_price": 55}, {"yes_price": 55},
        ]
    now = datetime.now(timezone.utc)
    m = Market(
        ticker=ticker, title=title, status="open",
        yes_ask=yes_ask, yes_bid=yes_bid,
        no_ask=100 - yes_bid, no_bid=100 - yes_ask,
        volume=volume_24h * 3, volume_24h=volume_24h,
        open_interest=open_interest,
        close_time=now + timedelta(minutes=minutes_to_settlement),
        settlement_time=now + timedelta(minutes=minutes_to_settlement),
        category=category, rules_primary="rules",
        event_ticker=event_ticker or "KXBTCD-25NOV13",
        price_history=price_history,
    )
    m.minutes_to_settlement = minutes_to_settlement
    m.minutes_to_close = minutes_to_settlement
    m.orderbook_depth = orderbook_depth
    m.liquidity_dollars = liquidity_dollars
    return m


def make_research(items: int = 0) -> ResearchResult:
    return ResearchResult(
        ticker="KX", query="Q",
        items=[
            ResearchItem(
                source="Reuters", url="", published_at=datetime.now(timezone.utc),
                claim=f"claim {i}", direction="supports_yes",
                relevance=0.9, credibility=0.85, recency_score=0.95,
                summary="s", agent="a",
            ) for i in range(items)
        ],
    )


def make_sentiment(
    *,
    score: float = 0.4,
    direction: str = "supports_yes",
    impact: int = 5,
    cred: float = 0.85,
    rel: float = 0.9,
    rumor_risk: str = "low",
) -> SentimentResult:
    return SentimentResult(
        ticker="KX",
        sentiment_score=score,
        narrative_direction=direction,
        confidence=0.7,
        market_impact_estimate_cents=impact,
        major_contradictions=[],
        item_count=3,
        contributing_sources=["Reuters"],
        source_credibility=cred,
        event_relevance=rel,
        rumor_risk=rumor_risk,
    )


def make_weird_move(
    *,
    flagged: bool = False,
    classification: str = "none",
    p5: float = 0.0,
    p15: float = 0.0,
    vol_ratio: float = 1.0,
) -> WeirdMoveSignal:
    return WeirdMoveSignal(
        ticker="KX",
        flagged=flagged,
        classification=classification,
        price_change_5m=p5,
        price_change_15m=p15,
        volume_ratio=vol_ratio,
        spread_change=1.0,
        related_disagreement=0.0,
        triggers=[],
        description="",
        confidence="low",
    )


# ── Spec test 1: missing sentiment does not crash ────────────────────────────

def test_missing_sentiment_does_not_crash(cfg):
    market = make_market()
    res = make_research()
    weird = make_weird_move()
    # Pass sentiment=None
    out = pm.estimate(market, res, None, weird, cfg)
    assert isinstance(out, ProbabilityEstimate)
    # No sentiment → p_research collapses to p_market → blend == p_market
    p_market = (market.yes_ask + market.yes_bid) / 2 / 100
    assert out.yes_probability == pytest.approx(p_market, abs=1e-9)


# ── Spec test 2: missing weird_move does not invent movement ─────────────────

def test_missing_weird_move_does_not_create_fake_movement(cfg):
    market = make_market()
    res = make_research()
    sent = make_sentiment()
    out = pm.estimate(market, res, sent, None, cfg)
    assert isinstance(out, ProbabilityEstimate)
    # No weird_move signal → no step-down, no penalty path triggered
    # (Confidence stays at the LLM stub's "medium-high" minus normal step-downs.
    # With a 6-tick history and ample liquidity, none of the microstructure
    # step-downs should fire.)
    assert out.confidence in ("medium-high", "high")


# ── Spec test 3: sparse history reduces confidence ───────────────────────────

def test_sparse_price_history_lowers_confidence(cfg):
    market = make_market(price_history=[{"yes_price": 55}])   # 1 tick
    out = pm.estimate(market, make_research(), make_sentiment(), make_weird_move(), cfg)
    # Sparse history is one step down — and historical_volatility=0 (<3 ticks)
    # so the high-vol step-down does NOT fire.
    assert out.confidence in ("medium-low", "medium", "low")
    assert out.confidence != "medium-high"


# ── Spec test 4: empty related_market_prices is safe ─────────────────────────

def test_related_market_prices_empty_is_safe(cfg):
    market = make_market()
    # Pass an event map containing only this market — siblings list is empty.
    out = pm.estimate(
        market, make_research(), make_sentiment(), make_weird_move(), cfg,
        markets_by_event={market.event_ticker: [market]},
    )
    assert isinstance(out, ProbabilityEstimate)
    # No neighbors → no neighbor-disagreement step-down possible
    assert out.confidence in ("medium-high", "high")


def test_related_market_prices_none_event_map_is_safe(cfg):
    out = pm.estimate(
        make_market(), make_research(), make_sentiment(), make_weird_move(), cfg,
        markets_by_event=None,
    )
    assert isinstance(out, ProbabilityEstimate)


def test_neighbor_disagreement_lowers_confidence(cfg):
    """When siblings strongly disagree with p_model, step down."""
    self_m = make_market(ticker="A-104000", yes_ask=55, yes_bid=53,
                         event_ticker="A")
    # Two siblings priced ~5¢ — our 54¢ p_model is 49pp away from neighbor mean
    sib1 = make_market(ticker="A-200000", yes_ask=6, yes_bid=4,
                       event_ticker="A")
    sib2 = make_market(ticker="A-300000", yes_ask=6, yes_bid=4,
                       event_ticker="A")
    by_event = {"A": [self_m, sib1, sib2]}

    out_quiet = pm.estimate(
        self_m, make_research(), make_sentiment(), make_weird_move(), cfg,
        markets_by_event={"A": [self_m]},   # no siblings — baseline
    )
    out_loud = pm.estimate(
        self_m, make_research(), make_sentiment(), make_weird_move(), cfg,
        markets_by_event=by_event,
    )
    rank = pm._CONF_ORDER.index
    assert rank(out_loud.confidence) < rank(out_quiet.confidence)


# ── Spec test 5: distance_to_threshold = None is safe ────────────────────────

def test_distance_to_threshold_none_is_safe(cfg):
    market = make_market(title="Will Eagles win the Super Bowl?")
    out = pm.estimate(market, make_research(), make_sentiment(),
                      make_weird_move(), cfg, spot_price=None)
    assert isinstance(out, ProbabilityEstimate)


# ── Spec test 6: large weird-move w/o volume ≠ automatic high confidence ────

def test_large_weird_move_without_volume_does_not_create_high_confidence(cfg):
    """
    A 15¢ jump with no volume confirmation is classified by weird_move as
    rumor_move. That must NOT propagate as a high-confidence signal — the
    prediction model treats it as a confidence step-down, not a price input.
    """
    market = make_market()
    weird = make_weird_move(
        flagged=True, classification="rumor_move",
        p5=15.0, p15=20.0, vol_ratio=0.8,   # large move, NO volume spike
    )
    out = pm.estimate(market, make_research(), make_sentiment(), weird, cfg)
    # rumor_move costs one step down from "medium-high" → "medium"
    assert out.confidence != "high"
    assert out.confidence != "medium-high"


def test_stale_book_artifact_severely_penalises_confidence(cfg):
    weird = make_weird_move(flagged=True, classification="stale_book_artifact",
                            p5=5.0, vol_ratio=0.1)
    out = pm.estimate(make_market(), make_research(), make_sentiment(), weird, cfg)
    # stale_book_artifact = 2 steps down from "medium-high" → "medium-low"
    rank = pm._CONF_ORDER.index
    assert rank(out.confidence) <= rank("medium-low")


def test_news_confirmed_move_does_not_boost_confidence(cfg):
    """A valid news_confirmed_move must not upgrade confidence."""
    weird = make_weird_move(flagged=True, classification="news_confirmed_move",
                            p5=10.0, vol_ratio=4.0)
    out_with = pm.estimate(make_market(), make_research(), make_sentiment(),
                           weird, cfg)
    out_without = pm.estimate(make_market(), make_research(), make_sentiment(),
                              make_weird_move(), cfg)
    # No boost: the "with" confidence is not strictly above the "without".
    rank = pm._CONF_ORDER.index
    assert rank(out_with.confidence) <= rank(out_without.confidence)


# ── Spec test 7: high volatility reduces confidence ──────────────────────────

def test_high_volatility_reduces_confidence(cfg):
    # Flat history → hist_vol = 0; volatile → > 4¢
    flat = [{"yes_price": 55}] * 8
    volatile = [
        {"yes_price": p} for p in (40, 70, 35, 75, 30, 80, 25, 85)
    ]
    out_flat = pm.estimate(make_market(price_history=flat),
                           make_research(), make_sentiment(), make_weird_move(), cfg)
    out_vol = pm.estimate(make_market(price_history=volatile),
                          make_research(), make_sentiment(), make_weird_move(), cfg)
    rank = pm._CONF_ORDER.index
    assert rank(out_vol.confidence) < rank(out_flat.confidence)


# ── Spec test 8: LLM adjustment remains capped at ±10pp ──────────────────────

def test_llm_adjustment_capped_at_10pp(monkeypatch, cfg):
    """LLM returns extreme prob — final p_llm contribution must respect ±10pp."""
    # Force LLM to say YES is 99% certain regardless of market
    def extreme(cfg, *args, **kwargs):
        return {
            "yes_probability": 0.99,
            "confidence": "high",
            "reasoning": "extreme",
            "assumptions": [], "invalidation_conditions": [],
        }
    monkeypatch.setattr(llm, "estimate_probability", extreme)

    market = make_market(yes_ask=30, yes_bid=28)   # market thinks ~29%
    out = pm.estimate(market, make_research(), make_sentiment(score=0.0,
                       direction="neutral", impact=0), make_weird_move(), cfg)

    # p_market = 0.29; p_research = p_market (neutral sentiment);
    # p_llm should be clamped to p_market + 0.10 = 0.39
    # Ensemble: 0.70 * 0.29 + 0.25 * 0.29 + 0.05 * 0.39 = 0.295
    p_market = 0.29
    expected_max = _W_MARKET_EXPECTED * p_market + _W_RESEARCH_EXPECTED * p_market + _W_LLM_EXPECTED * (p_market + 0.10)
    assert out.yes_probability <= expected_max + 1e-9


# These mirror the constants in prediction_model — recorded here so the test
# fails loudly if anyone changes ensemble weights without updating expectations.
_W_MARKET_EXPECTED   = 0.70
_W_RESEARCH_EXPECTED = 0.25
_W_LLM_EXPECTED      = 0.05


def test_sentiment_shift_capped_at_10pp(monkeypatch, cfg):
    """Even with huge sentiment.market_impact_estimate_cents, p_research is bounded."""
    # Sentiment claims 50¢ impact — must be clamped to ±10pp = 10¢
    huge = make_sentiment(impact=50, direction="supports_yes")
    market = make_market(yes_ask=50, yes_bid=48)
    out = pm.estimate(market, make_research(), huge, make_weird_move(), cfg)
    # p_market = 0.49, p_research <= 0.59, p_llm ≈ 0.49 (stubbed)
    # blend max = 0.70*0.49 + 0.25*0.59 + 0.05*0.49 = 0.515
    assert out.yes_probability <= 0.52


def test_supports_no_sentiment_pulls_estimate_down(cfg):
    """Bearish sentiment with magnitude 5¢ should shift p_research downward."""
    market = make_market(yes_ask=60, yes_bid=58)
    bull = make_sentiment(impact=5, direction="supports_yes")
    bear = make_sentiment(impact=5, direction="supports_no")
    out_bull = pm.estimate(market, make_research(), bull, make_weird_move(), cfg)
    out_bear = pm.estimate(market, make_research(), bear, make_weird_move(), cfg)
    assert out_bear.yes_probability < out_bull.yes_probability


# ── Spec test 9: result stays in [0.01, 0.99] ────────────────────────────────

def test_estimate_clamped_between_0_01_and_0_99(monkeypatch, cfg):
    """Combine the most extreme inputs we can engineer; result must stay clamped."""
    def extreme(cfg, *args, **kwargs):
        return {"yes_probability": 0.999, "confidence": "high",
                "reasoning": "", "assumptions": [], "invalidation_conditions": []}
    monkeypatch.setattr(llm, "estimate_probability", extreme)

    market = make_market(yes_ask=99, yes_bid=98)
    huge_yes = make_sentiment(impact=50, direction="supports_yes")
    out = pm.estimate(market, make_research(), huge_yes, make_weird_move(), cfg)
    assert 0.01 <= out.yes_probability <= 0.99

    # And the lower bound
    market_low = make_market(yes_ask=1, yes_bid=1)
    huge_no = make_sentiment(impact=50, direction="supports_no")
    out_low = pm.estimate(market_low, make_research(), huge_no, make_weird_move(), cfg)
    assert 0.01 <= out_low.yes_probability <= 0.99


# ── Spec test 10: low-confidence rejected by edge.passes_threshold ───────────

def test_low_confidence_rejected_by_edge_threshold(cfg):
    """
    Wire prediction_model → edge.calculate → edge.passes_threshold.
    A "low" estimate with otherwise tradeable raw edge must be rejected.
    """
    # Engineer a low-confidence estimate: sparse history + near resolution
    # + thin liquidity will stack three step-downs on top of the LLM's
    # medium-high → confidence == "low".
    market = make_market(
        yes_ask=40, yes_bid=38,
        price_history=[],                  # sparse  → step down
        minutes_to_settlement=15,          # near    → step down
        liquidity_dollars=100.0,           # thin    → step down
    )
    # Optimistic sentiment to create a raw edge on YES
    sent = make_sentiment(impact=8, direction="supports_yes")
    est = pm.estimate(market, make_research(), sent, make_weird_move(), cfg)
    assert est.confidence == "low"

    # Run through the edge layer
    edge_result = edge.calculate(market, est, cfg)
    if edge_result is None:
        # No tradable edge at all → trivially rejected
        return
    assert edge.passes_threshold(edge_result, cfg) is False


# ── Bonus: top-level estimate() never raises ─────────────────────────────────

def test_estimate_returns_fallback_on_unexpected_error(monkeypatch, cfg):
    """If something deep crashes, _fallback kicks in — no exception bubbles up."""
    # Make features.extract_features blow up
    import features as features_mod
    def bad(*a, **kw):
        raise RuntimeError("simulated feature crash")
    monkeypatch.setattr(features_mod, "extract_features", bad)

    out = pm.estimate(make_market(), make_research(), make_sentiment(),
                      make_weird_move(), cfg)
    assert isinstance(out, ProbabilityEstimate)
    assert out.confidence == "low"
    assert "fallback" in out.reasoning.lower()


def test_p_research_neutral_when_narrative_mixed(cfg):
    """Mixed narratives must not bias p_research, even with a non-zero impact."""
    market = make_market(yes_ask=55, yes_bid=53)
    sent = make_sentiment(impact=10, direction="mixed")
    out = pm.estimate(market, make_research(), sent, make_weird_move(), cfg)
    p_market = (market.yes_ask + market.yes_bid) / 2 / 100
    # With LLM stubbed to p_market, the blend equals p_market exactly.
    assert out.yes_probability == pytest.approx(p_market, abs=1e-9)
