"""
Tests for the spec-shaped Probability Estimator in llm.run_probability_estimator
and its consumption by prediction_model.

Covers the spec checklist:
  A. JSON shape
  B. Baseline anchoring
  C. LLM cap (deterministic ±10pp clamp in prediction_model)
  D. Social-only caution
  E. Conflicting evidence
  F. Missing data fallback
  G. Output consistency: yes + no = 1
  H. No trade-recommendation fields
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import llm
import prediction_model as pm
from config import Config
from models import (
    Market, ResearchItem, ResearchResult, SentimentResult, WeirdMoveSignal,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> Config:
    return Config(
        kalshi_api_key="x", anthropic_api_key="x",
        kalshi_private_key_path="",
    )


def _make_market(yes_ask: int = 50, yes_bid: int = 48) -> Market:
    now = datetime.now(timezone.utc)
    m = Market(
        ticker="K", title="T", status="open",
        yes_ask=yes_ask, yes_bid=yes_bid,
        no_ask=100 - yes_bid, no_bid=100 - yes_ask,
        volume=2000, volume_24h=12_000, open_interest=4_000,
        close_time=now + timedelta(minutes=120),
        settlement_time=now + timedelta(minutes=120),
        category="crypto", rules_primary="r", event_ticker="K",
        price_history=[
            {"yes_price": 49}, {"yes_price": 50}, {"yes_price": 51},
            {"yes_price": 50}, {"yes_price": 50}, {"yes_price": 50},
        ],
    )
    m.minutes_to_settlement = 120
    m.liquidity_dollars = 5_000
    return m


def _research_with(items: int = 1) -> ResearchResult:
    now = datetime.now(timezone.utc)
    return ResearchResult(
        ticker="K", query="Q",
        items=[
            ResearchItem(
                source="Reuters", url="", published_at=now,
                claim=f"c{i}", direction="supports_yes",
                relevance=0.9, credibility=0.85, recency_score=0.95,
                summary="s", agent="a",
            ) for i in range(items)
        ],
    )


def _sentiment(
    *, direction: str = "supports_yes",
    impact: int = 5, rumor_risk: str = "low",
    contradictions: list[str] | None = None,
    item_count: int = 3,
) -> SentimentResult:
    return SentimentResult(
        ticker="K",
        sentiment_score=0.4 if direction == "supports_yes" else -0.4,
        narrative_direction=direction,
        confidence=0.7,
        market_impact_estimate_cents=impact,
        major_contradictions=contradictions or [],
        item_count=item_count,
        contributing_sources=["Reuters"],
        source_credibility=0.85,
        event_relevance=0.9,
        rumor_risk=rumor_risk,
    )


def _quiet_weird_move() -> WeirdMoveSignal:
    return WeirdMoveSignal(
        ticker="K", flagged=False, classification="none",
        price_change_5m=0.0, price_change_15m=0.0,
        volume_ratio=1.0, spread_change=1.0,
        related_disagreement=0.0, triggers=[],
        description="", confidence="low",
    )


# ── Spec keys ────────────────────────────────────────────────────────────────

_SPEC_KEYS = {
    "estimated_yes_probability", "estimated_no_probability", "confidence",
    "main_factors", "uncertainty_factors", "probability_rationale",
    "overconfidence_warning",
}


# ── A. JSON shape ────────────────────────────────────────────────────────────

def test_run_probability_estimator_returns_exact_spec_shape(monkeypatch, cfg):
    def fake_call_json(*a, **kw):
        return {
            "estimated_yes_probability": 0.55,
            "estimated_no_probability":  0.45,
            "confidence":                0.62,
            "main_factors":              ["official data confirms"],
            "uncertainty_factors":       ["short history"],
            "probability_rationale":     "Slight tilt above 50 from official data.",
            "overconfidence_warning":    False,
        }
    monkeypatch.setattr(llm, "call_json", fake_call_json)

    out = llm.run_probability_estimator(
        cfg,
        ticker="K", title="T", rules="r",
        market_implied_yes_probability=0.50,
        yes_bid=49, yes_ask=51, no_bid=49, no_ask=51,
        spread_cents=2, liquidity_dollars=5000.0, volume_24h=10_000,
        minutes_to_resolution=120.0,
        research_summary="...",
        sentiment_payload={"narrative_direction": "supports_yes"},
    )
    assert set(out.keys()) == _SPEC_KEYS
    assert isinstance(out["estimated_yes_probability"], float)
    assert isinstance(out["estimated_no_probability"], float)
    assert isinstance(out["confidence"], float)
    assert isinstance(out["main_factors"], list)
    assert isinstance(out["uncertainty_factors"], list)
    assert isinstance(out["probability_rationale"], str)
    assert isinstance(out["overconfidence_warning"], bool)


# ── B. Baseline anchoring ────────────────────────────────────────────────────

def test_invalid_llm_output_falls_back_to_market_baseline(monkeypatch, cfg):
    """Garbage output → market-anchored fallback with low confidence."""
    monkeypatch.setattr(llm, "call_json", lambda *a, **kw: {"nonsense": True})

    p_market = 0.42
    out = llm.run_probability_estimator(
        cfg,
        ticker="K", title="T", rules="r",
        market_implied_yes_probability=p_market,
        yes_bid=41, yes_ask=43, no_bid=57, no_ask=59,
        spread_cents=2, liquidity_dollars=5000.0, volume_24h=10_000,
        minutes_to_resolution=120.0,
        research_summary="",
    )
    assert out["estimated_yes_probability"] == pytest.approx(p_market, abs=1e-6)
    assert out["estimated_no_probability"] == pytest.approx(1 - p_market, abs=1e-6)
    assert out["confidence"] == 0.25
    assert out["overconfidence_warning"] is True


def test_llm_failure_falls_back(monkeypatch, cfg):
    def boom(*a, **kw):
        raise RuntimeError("network down")
    monkeypatch.setattr(llm, "call_json", boom)

    out = llm.run_probability_estimator(
        cfg,
        ticker="K", title="T", rules="r",
        market_implied_yes_probability=0.30,
        yes_bid=29, yes_ask=31, no_bid=69, no_ask=71,
        spread_cents=2, liquidity_dollars=5000.0, volume_24h=10_000,
        minutes_to_resolution=120.0, research_summary="",
    )
    assert out["estimated_yes_probability"] == pytest.approx(0.30, abs=1e-6)
    assert out["confidence"] == 0.25
    assert out["overconfidence_warning"] is True


# ── C. LLM cap inside prediction_model is enforced regardless of spec output ─

def test_extreme_spec_output_still_capped_in_prediction_model(monkeypatch, cfg):
    """Even when the estimator yells 0.95, prediction_model clamps ±10pp from p_market."""
    def fake_estimate_probability(
        cfg, ticker, title, rules, yes_ask, no_ask, research_summary, sentiment_data
    ):
        # Legacy-shaped result (this is what _llm_component consumes)
        return {
            "yes_probability":         0.95,
            "confidence":              "high",
            "reasoning":               "spec extreme",
            "assumptions":             ["something big"],
            "invalidation_conditions": ["..."],
        }
    monkeypatch.setattr(llm, "estimate_probability", fake_estimate_probability)

    market = _make_market(yes_ask=50, yes_bid=48)
    out = pm.estimate(
        market, _research_with(1),
        _sentiment(direction="supports_yes", impact=0, item_count=3),
        _quiet_weird_move(), cfg,
    )
    p_market = (50 + 48) / 2 / 100  # 0.49
    # LLM evidence is capped in log-odds space and then weighted at 5%.
    max_shift = max(
        abs(pm._sigmoid(pm._logit(p_market) + sign * 0.05 * pm._LLM_LOGIT_SHIFT_CAP) - p_market)
        for sign in (-1, 1)
    ) + 1e-9
    assert abs(out.yes_probability - p_market) <= max_shift


# ── D. Social-only caution ───────────────────────────────────────────────────

def test_social_only_signal_blocks_high_confidence(monkeypatch, cfg):
    """High rumor_risk → prediction_model steps confidence down at least once."""
    def fake(cfg, *a, **kw):
        return {
            "yes_probability": 0.50, "confidence": "high",
            "reasoning": "", "assumptions": [], "invalidation_conditions": [],
        }
    monkeypatch.setattr(llm, "estimate_probability", fake)

    market = _make_market()
    rumor = _sentiment(rumor_risk="high")
    clean = _sentiment(rumor_risk="low")
    out_rumor = pm.estimate(market, _research_with(1), rumor, _quiet_weird_move(), cfg)
    out_clean = pm.estimate(market, _research_with(1), clean, _quiet_weird_move(), cfg)
    rank = pm._CONF_ORDER.index
    assert rank(out_rumor.confidence) < rank(out_clean.confidence)


# ── E. Conflicting evidence ──────────────────────────────────────────────────

def test_contradictions_lower_confidence(monkeypatch, cfg):
    def fake(cfg, *a, **kw):
        return {
            "yes_probability": 0.50, "confidence": "high",
            "reasoning": "", "assumptions": [], "invalidation_conditions": [],
        }
    monkeypatch.setattr(llm, "estimate_probability", fake)

    market = _make_market()
    clean = _sentiment()
    conflicted = _sentiment(contradictions=["official YES vs social NO"])
    out_clean = pm.estimate(market, _research_with(1), clean, _quiet_weird_move(), cfg)
    out_conf = pm.estimate(market, _research_with(1), conflicted, _quiet_weird_move(), cfg)
    rank = pm._CONF_ORDER.index
    assert rank(out_conf.confidence) < rank(out_clean.confidence)


def test_contradictions_surface_as_uncertainty_factor(monkeypatch, cfg):
    """If the spec layer captures contradictions, they belong in uncertainty_factors."""
    def fake_call_json(*a, **kw):
        return {
            "estimated_yes_probability": 0.50,
            "estimated_no_probability":  0.50,
            "confidence":                0.40,
            "main_factors":              ["mixed signal"],
            "uncertainty_factors":       ["official YES contradicts social NO"],
            "probability_rationale":     "Held at baseline due to contradictions.",
            "overconfidence_warning":    True,
        }
    monkeypatch.setattr(llm, "call_json", fake_call_json)

    out = llm.run_probability_estimator(
        cfg,
        ticker="K", title="T", rules="r",
        market_implied_yes_probability=0.50,
        yes_bid=49, yes_ask=51, no_bid=49, no_ask=51,
        spread_cents=2, liquidity_dollars=5000.0, volume_24h=10_000,
        minutes_to_resolution=120.0, research_summary="",
    )
    assert any("contradict" in f.lower() for f in out["uncertainty_factors"])
    assert out["overconfidence_warning"] is True


# ── F. Missing data falls back safely + lowers confidence ────────────────────

def test_missing_research_lowers_pm_confidence(monkeypatch, cfg):
    def fake(cfg, *a, **kw):
        return {
            "yes_probability": 0.50, "confidence": "high",
            "reasoning": "", "assumptions": [], "invalidation_conditions": [],
        }
    monkeypatch.setattr(llm, "estimate_probability", fake)

    market = _make_market()
    has_research = _sentiment(item_count=3)
    no_research = _sentiment(item_count=0)
    out_yes = pm.estimate(market, _research_with(1), has_research, _quiet_weird_move(), cfg)
    out_no  = pm.estimate(market, _research_with(0), no_research, _quiet_weird_move(), cfg)
    rank = pm._CONF_ORDER.index
    assert rank(out_no.confidence) < rank(out_yes.confidence)


# ── G. yes + no = 1.0 always ─────────────────────────────────────────────────

@pytest.mark.parametrize("p_yes_in", [0.05, 0.50, 0.50001, 0.95])
def test_yes_plus_no_equals_one(monkeypatch, cfg, p_yes_in):
    def fake_call_json(*a, **kw):
        return {
            "estimated_yes_probability": p_yes_in,
            "estimated_no_probability":  0.123,  # garbage — must be overwritten
            "confidence":                0.5,
            "main_factors":              [],
            "uncertainty_factors":       [],
            "probability_rationale":     "x",
            "overconfidence_warning":    False,
        }
    monkeypatch.setattr(llm, "call_json", fake_call_json)

    out = llm.run_probability_estimator(
        cfg,
        ticker="K", title="T", rules="r",
        market_implied_yes_probability=0.50,
        yes_bid=49, yes_ask=51, no_bid=49, no_ask=51,
        spread_cents=2, liquidity_dollars=5000.0, volume_24h=10_000,
        minutes_to_resolution=120.0, research_summary="",
    )
    assert out["estimated_yes_probability"] + out["estimated_no_probability"] == pytest.approx(1.0, abs=1e-6)


def test_probability_clamped_to_safe_range(monkeypatch, cfg):
    """LLM saying 1.0 or 0.0 must be pulled back into [0.01, 0.99]."""
    for raw_yes in (0.0, 1.0, -0.5, 1.5):
        def fake(*a, _yes=raw_yes, **kw):
            return {
                "estimated_yes_probability": _yes,
                "estimated_no_probability":  1 - _yes,
                "confidence":                0.5,
                "main_factors":              [],
                "uncertainty_factors":       [],
                "probability_rationale":     "x",
                "overconfidence_warning":    False,
            }
        monkeypatch.setattr(llm, "call_json", fake)
        out = llm.run_probability_estimator(
            cfg, ticker="K", title="T", rules="r",
            market_implied_yes_probability=0.50,
            yes_bid=49, yes_ask=51, no_bid=49, no_ask=51,
            spread_cents=2, liquidity_dollars=5000.0, volume_24h=10_000,
            minutes_to_resolution=120.0, research_summary="",
        )
        assert 0.01 <= out["estimated_yes_probability"] <= 0.99
        assert 0.01 <= out["estimated_no_probability"] <= 0.99


# ── H. No trade-recommendation fields ────────────────────────────────────────

_FORBIDDEN_TRADE_KEYS = {
    "recommendation", "recommended_side", "side", "buy", "sell",
    "buy_yes", "buy_no", "action", "trade", "trade_recommendation",
    "edge", "edge_cents", "fair_price", "fair_value", "expected_value", "ev",
    "position_size", "contracts", "size", "stake", "entry", "exit",
}


def test_spec_output_has_no_trade_recommendation_fields(monkeypatch, cfg):
    """Even when the LLM tries to smuggle in trade fields, they must not appear."""
    def fake_call_json(*a, **kw):
        return {
            "estimated_yes_probability": 0.55,
            "estimated_no_probability":  0.45,
            "confidence":                0.6,
            "main_factors":              ["x"],
            "uncertainty_factors":       ["y"],
            "probability_rationale":     "r",
            "overconfidence_warning":    False,
            # Forbidden smuggled fields:
            "recommended_side":          "YES",
            "edge":                      0.07,
            "position_size":             5,
        }
    monkeypatch.setattr(llm, "call_json", fake_call_json)

    out = llm.run_probability_estimator(
        cfg, ticker="K", title="T", rules="r",
        market_implied_yes_probability=0.50,
        yes_bid=49, yes_ask=51, no_bid=49, no_ask=51,
        spread_cents=2, liquidity_dollars=5000.0, volume_24h=10_000,
        minutes_to_resolution=120.0, research_summary="",
    )
    leaks = set(out.keys()) & _FORBIDDEN_TRADE_KEYS
    assert not leaks, f"spec output leaked trade fields: {leaks}"
    assert set(out.keys()) == _SPEC_KEYS


def test_system_prompt_has_no_trade_recommendation_vocabulary():
    """Belt-and-braces: the system prompt itself bans recommending trades."""
    sp = llm._PROBABILITY_ESTIMATOR_SYSTEM
    # Required negative-space statements
    for clause in (
        "Do not recommend a trade",
        "Do not calculate edge",
        "Do not suggest side",
        "Do not place orders",
        "Do not override risk",
        "Do not invent missing facts",
    ):
        assert clause in sp, f"missing prompt clause: {clause!r}"


# ── Overconfidence warning auto-fires on large baseline drift ────────────────

def test_overconfidence_warning_set_when_far_from_baseline(monkeypatch, cfg):
    """LLM says 0.80, baseline says 0.50 — overconfidence_warning must be True
    even if the LLM said False."""
    def fake_call_json(*a, **kw):
        return {
            "estimated_yes_probability": 0.80,
            "estimated_no_probability":  0.20,
            "confidence":                0.9,
            "main_factors":              ["bullish narrative"],
            "uncertainty_factors":       [],
            "probability_rationale":     "Way above baseline.",
            "overconfidence_warning":    False,
        }
    monkeypatch.setattr(llm, "call_json", fake_call_json)

    out = llm.run_probability_estimator(
        cfg, ticker="K", title="T", rules="r",
        market_implied_yes_probability=0.50,
        yes_bid=49, yes_ask=51, no_bid=49, no_ask=51,
        spread_cents=2, liquidity_dollars=5000.0, volume_24h=10_000,
        minutes_to_resolution=120.0, research_summary="",
    )
    assert out["overconfidence_warning"] is True


# ── Confidence float-to-label mapping ────────────────────────────────────────

@pytest.mark.parametrize("v,expected", [
    (0.00, "low"),
    (0.39, "low"),
    (0.40, "medium-low"),
    (0.54, "medium-low"),
    (0.55, "medium"),
    (0.69, "medium"),
    (0.70, "medium-high"),
    (0.84, "medium-high"),
    (0.85, "high"),
    (1.00, "high"),
])
def test_confidence_float_to_label(v, expected):
    assert llm._confidence_float_to_label(v) == expected


def test_confidence_float_to_label_handles_garbage():
    assert llm._confidence_float_to_label(None) == "low"
    assert llm._confidence_float_to_label("nope") == "low"
    assert llm._confidence_float_to_label(-0.5) == "low"
    assert llm._confidence_float_to_label(99.0) == "high"
