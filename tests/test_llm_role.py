"""
Tests that codify the LLM role contract documented at the top of llm.py.

These tests are structural — they enforce *which modules can import llm*
and *how its output may be used* — not behavioral. They guard against
future drift like "let me just have the risk manager double-check by
asking the LLM" or "if Perplexity is down, let's just have the LLM
summarize what it knows."

Rules under test:
  A. Direct order placement       — trading.py / kalshi_client.py must
                                    not import llm.
  B. Overriding the risk manager  — risk_manager.py must not import llm.
  C. Uncapped probability jumps   — prediction_model never propagates the
                                    LLM's yes_probability beyond ±10pp
                                    from market mid.
  D. Inventing missing data       — research falls back to empty, never
                                    to LLM-fabricated "research", when the
                                    real-time data source is unavailable.
"""
from __future__ import annotations
import ast
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import llm
import prediction_model as pm
import research_agents
from config import Config
from models import (
    Market, ResearchItem, ResearchResult, SentimentResult, WeirdMoveSignal,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _imports_module(file_path: Path, module_name: str) -> bool:
    """Static check: does `file_path` import `module_name`?"""
    src = file_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name:
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == module_name:
                return True
    return False


# ── Rule A: order placement must not import the LLM ──────────────────────────

@pytest.mark.parametrize("module_file", [
    "trading.py",
    "kalshi_client.py",
    "monitor.py",
])
def test_order_placement_does_not_import_llm(module_file):
    path = _PROJECT_ROOT / module_file
    assert path.exists(), f"{module_file} not found at {path}"
    assert not _imports_module(path, "llm"), (
        f"{module_file} must not import llm — order-placement and "
        f"position-management code is strictly deterministic."
    )


# ── Rule B: the risk manager is fully deterministic ──────────────────────────

def test_risk_manager_does_not_import_llm():
    path = _PROJECT_ROOT / "risk_manager.py"
    assert not _imports_module(path, "llm"), (
        "risk_manager.py must not import llm. Risk decisions are "
        "deterministic — the LLM cannot relax a risk gate."
    )


def test_risk_manager_does_not_import_prediction_model():
    """
    prediction_model.py imports llm. If risk_manager imported the prediction
    model and re-evaluated probabilities at gate time, an LLM hiccup could
    indirectly relax a gate. The arrow is one-way: prediction_model →
    risk_manager via PositionSize / EdgeResult only.
    """
    path = _PROJECT_ROOT / "risk_manager.py"
    assert not _imports_module(path, "prediction_model"), (
        "risk_manager.py must not depend on prediction_model."
    )


# ── Rule C: LLM probability is hard-capped to ±10pp ──────────────────────────

@pytest.fixture
def cfg() -> Config:
    return Config(kalshi_api_key="x", anthropic_api_key="x",
                  kalshi_private_key_path="")


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


def _neutral_sentiment() -> SentimentResult:
    """Neutral sentiment so only the LLM component moves p_model."""
    return SentimentResult(
        ticker="K",
        sentiment_score=0.0,
        narrative_direction="neutral",
        confidence=0.0,
        market_impact_estimate_cents=0,
        major_contradictions=[],
        item_count=0,
        contributing_sources=[],
        source_credibility=0.0,
        event_relevance=0.0,
        rumor_risk="low",
    )


def _quiet_weird_move() -> WeirdMoveSignal:
    return WeirdMoveSignal(
        ticker="K", flagged=False, classification="none",
        price_change_5m=0.0, price_change_15m=0.0,
        volume_ratio=1.0, spread_change=1.0,
        related_disagreement=0.0, triggers=[],
        description="", confidence="low",
    )


def _research() -> ResearchResult:
    return ResearchResult(ticker="K", query="Q", items=[])


@pytest.mark.parametrize("llm_says,expected_within_pp", [
    (0.99, 0.10),
    (0.01, 0.10),
    (0.80, 0.10),
    (0.20, 0.10),
])
def test_llm_probability_cannot_jump_more_than_10pp(monkeypatch, cfg,
                                                    llm_says, expected_within_pp):
    """
    The LLM proposes an extreme yes_probability. The prediction model must
    clamp the LLM's effective contribution to ±10pp from market mid, and the
    ensemble weight on the LLM is only 5%, so the FULL p_model excursion
    cannot exceed 0.5pp from p_market in either direction.
    """
    def extreme(cfg, *a, **kw):
        return {"yes_probability": llm_says, "confidence": "high",
                "reasoning": "", "assumptions": [], "invalidation_conditions": []}
    monkeypatch.setattr(llm, "estimate_probability", extreme)

    market = _make_market(yes_ask=50, yes_bid=48)   # p_market = 0.49
    p_market = (market.yes_ask + market.yes_bid) / 2 / 100

    out = pm.estimate(market, _research(), _neutral_sentiment(),
                      _quiet_weird_move(), cfg)

    # With neutral sentiment, p_research == p_market. The LLM component is
    # the only mover, and its raw value is clamped to p_market ± 0.10. Total
    # ensemble shift from p_market is bounded by 0.05 * 0.10 = 0.005.
    max_shift = 0.05 * 0.10 + 1e-9
    assert abs(out.yes_probability - p_market) <= max_shift, (
        f"LLM shifted p_model by {abs(out.yes_probability - p_market):.4f} "
        f"— must be <= {max_shift:.4f} (5% weight × 10pp cap)"
    )


def test_llm_failure_does_not_change_p_market(monkeypatch, cfg):
    """If the LLM call raises, p_llm defaults to p_market — never propagates."""
    def boom(cfg, *a, **kw):
        raise RuntimeError("LLM down")
    monkeypatch.setattr(llm, "estimate_probability", boom)

    market = _make_market(yes_ask=72, yes_bid=70)
    p_market = (market.yes_ask + market.yes_bid) / 2 / 100

    out = pm.estimate(market, _research(), _neutral_sentiment(),
                      _quiet_weird_move(), cfg)

    # Neutral sentiment + LLM failure → ensemble == p_market exactly
    assert out.yes_probability == pytest.approx(p_market, abs=1e-9)


# ── Rule D: do not invent missing data ───────────────────────────────────────

def test_research_returns_no_items_when_perplexity_unavailable(monkeypatch, cfg):
    """
    Direct test of research_agents._search behavior: with no Perplexity key
    and no LLM fallback, _search must return "" — not LLM-fabricated text.
    """
    cfg.perplexity_api_key = ""

    market = _make_market()
    out = research_agents._search("query", "hint", market, cfg)

    assert out == "", (
        "_search must return an empty string when the real-time API is "
        "unavailable. Falling back to the LLM would invent missing data."
    )


def test_search_does_not_call_llm_when_perplexity_unavailable(monkeypatch, cfg):
    """
    Spy on every LLM entry point. If Perplexity is unavailable, none of
    them may be invoked during _search.
    """
    cfg.perplexity_api_key = ""
    calls: list[str] = []
    monkeypatch.setattr(llm, "call", lambda *a, **kw: calls.append("call") or "")
    monkeypatch.setattr(llm, "call_json", lambda *a, **kw: calls.append("call_json") or {})
    monkeypatch.setattr(llm, "estimate_probability",
                        lambda *a, **kw: calls.append("estimate_probability") or {})
    monkeypatch.setattr(llm, "research_agent_evidence",
                        lambda *a, **kw: calls.append("research_agent_evidence") or {})
    monkeypatch.setattr(llm, "extract_research_items",
                        lambda *a, **kw: calls.append("extract_research_items") or [])
    monkeypatch.setattr(llm, "research", lambda *a, **kw: calls.append("research") or {})
    monkeypatch.setattr(llm, "run_postmortem", lambda *a, **kw: calls.append("run_postmortem") or {})
    monkeypatch.setattr(llm, "analyze_market_structure",
                        lambda *a, **kw: calls.append("analyze_market_structure") or {})

    market = _make_market()
    research_agents._search("query", "hint", market, cfg)

    assert calls == [], (
        f"_search must not call any LLM function when Perplexity is "
        f"unavailable. Got: {calls}"
    )


def test_items_from_raw_short_circuits_on_empty_text(monkeypatch, cfg):
    """
    The LLM extractor is for converting REAL research text into typed
    claims. With empty input, it must not be called — otherwise the LLM
    could be coaxed into producing claims out of thin air.
    """
    called = {"n": 0}
    def spy(cfg, ticker, title, rules, raw_text):
        called["n"] += 1
        return []
    monkeypatch.setattr(llm, "extract_research_items", spy)

    market = _make_market()
    items = research_agents._items_from_raw(
        raw_text="", market=market,
        base_credibility=0.85, agent_name="test", cfg=cfg,
    )
    assert items == []
    assert called["n"] == 0, "extract_research_items must not be called on empty input"


def test_items_from_raw_short_circuits_on_whitespace(monkeypatch, cfg):
    called = {"n": 0}
    def spy(*a, **kw):
        called["n"] += 1
        return []
    monkeypatch.setattr(llm, "extract_research_items", spy)

    market = _make_market()
    items = research_agents._items_from_raw(
        raw_text="   \n  \t ", market=market,
        base_credibility=0.85, agent_name="test", cfg=cfg,
    )
    assert items == []
    assert called["n"] == 0


def test_llm_fallback_helper_no_longer_exists():
    """
    The old _llm_fallback function fabricated research from the LLM's
    training data when Perplexity was unavailable. It has been removed and
    must not reappear under that name or a synonym.
    """
    forbidden_names = [
        "_llm_fallback",   # exact old name
        "llm_fallback",
        "_fabricate_research",
    ]
    for name in forbidden_names:
        assert not hasattr(research_agents, name), (
            f"research_agents.{name} must not exist — it would invent missing data."
        )


# ── Sanity: the allowed callers DO import llm ────────────────────────────────
# This is the inverse of the prohibitions — it documents what the contract
# explicitly permits. If someone removes one of these on purpose, this test
# will fail loudly so they have to update both the contract and the test.

def test_allowed_modules_import_llm():
    allowed = ["prediction_model.py", "research_agents.py", "category_research.py"]
    for fname in allowed:
        path = _PROJECT_ROOT / fname
        assert path.exists(), f"{fname} not found"
        assert _imports_module(path, "llm"), (
            f"{fname} is expected to import llm — the role contract allows it."
        )
