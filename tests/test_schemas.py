"""
Step 18 schema-compatibility tests.

Covers:
  - every blueprint dataclass exists and is importable from models
  - blueprint aliases (time_to_resolution_minutes, PostmortemReport,
    rejection_reasons, proposed_rule_change) are wired
  - to_dict() / to_spec_dict() helpers are JSON-safe (datetime → ISO)
  - probability/credibility/relevance values are clamped to [0.0, 1.0]
  - Market accepts the fields market_scanner.py constructs
  - Postmortem rule changes are proposals only (never auto-approved)
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import models
from models import (
    CategoryResearchItem,
    EdgeResult,
    ExecutionReport,
    FeatureVector,
    Market,
    MarketSnapshot,
    Orderbook,
    PositionSize,
    Postmortem,
    PostmortemReport,
    ProbabilityEstimate,
    ResearchItem,
    ResearchResult,
    RiskAssessment,
    SentimentResult,
    TradeDecision,
    TradeRecord,
    WeirdMoveSignal,
)


# ── Imports / module surface ─────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "Market", "Orderbook", "MarketSnapshot", "WeirdMoveSignal",
    "ResearchItem", "CategoryResearchItem", "ResearchResult",
    "SentimentResult", "FeatureVector", "ProbabilityEstimate",
    "EdgeResult", "RiskAssessment", "PositionSize", "TradeDecision",
    "ExecutionReport", "TradeRecord", "Postmortem", "PostmortemReport",
])
def test_blueprint_schemas_are_exported(name):
    assert hasattr(models, name), f"models.{name} is missing"


def test_postmortem_report_is_alias_of_postmortem():
    assert PostmortemReport is Postmortem


# ── Market construction & aliases ────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_market() -> Market:
    return Market(
        ticker="KXTEST", title="Test", status="open",
        yes_ask=55, yes_bid=53, no_ask=47, no_bid=45,
        volume=2000, volume_24h=12_000, open_interest=4_000,
        close_time=_now() + timedelta(minutes=120),
        settlement_time=_now() + timedelta(minutes=120),
        category="crypto", rules_primary="rules",
        event_ticker="KXTEST",
    )


def test_market_constructs_with_scanner_field_set():
    """Mirrors the keyword arguments market_scanner.py passes."""
    m = _make_market()
    m.minutes_to_close = 90.0
    m.minutes_to_settlement = 120.0
    m.liquidity_dollars = 5_000.0
    m.spread_pct = 0.05
    m.orderbook_depth = 200
    m.last_trade_at = _now()
    assert m.ticker == "KXTEST"


def test_market_time_to_resolution_alias_mirrors_minutes_to_settlement():
    m = _make_market()
    m.minutes_to_settlement = 77.5
    assert m.time_to_resolution_minutes == 77.5
    # mutating the canonical field updates the alias
    m.minutes_to_settlement = 12.0
    assert m.time_to_resolution_minutes == 12.0


def test_market_to_dict_is_json_safe():
    m = _make_market()
    d = m.to_dict()
    # Datetimes serialized as ISO strings
    assert isinstance(d["close_time"], str)
    assert isinstance(d["settlement_time"], str)
    # JSON round-trip works
    json.dumps(d)


def test_market_mid_price_cents_handles_missing_bid():
    m = _make_market()
    m.yes_bid = 0
    m.yes_ask = 50
    assert m.mid_price_cents == 50.0


# ── ResearchItem / CategoryResearchItem ──────────────────────────────────────

def test_research_item_to_dict_clamps_and_serializes():
    item = ResearchItem(
        source="Reuters", url="", published_at=_now(),
        claim="claim", direction="supports_yes",
        relevance=2.0, credibility=-0.3, recency_score=0.7,
        summary="s", agent="a",
    )
    d = item.to_dict()
    # clamped
    assert 0.0 <= d["relevance"] <= 1.0
    assert 0.0 <= d["credibility"] <= 1.0
    # ISO timestamp
    assert isinstance(d["published_at"], str)
    json.dumps(d)


def test_category_research_item_spec_dict_shape():
    cri = CategoryResearchItem(
        query="Will X happen?", source_type="official", source_name="Fed",
        claim="Rate held", supports="yes",
        credibility=0.95, relevance=0.9, recency=1.0,
        risk_flags=["paywalled"],
    )
    d = cri.to_dict()
    assert set(d.keys()) == {
        "query", "source_type", "source_name", "claim", "supports",
        "credibility", "relevance", "recency", "risk_flags",
    }
    legacy = cri.to_legacy()
    assert legacy.direction == "supports_yes"


def test_research_result_to_dict_includes_items():
    items = [
        ResearchItem(
            source="Reuters", url="", published_at=_now(),
            claim="x", direction="supports_yes",
            relevance=0.9, credibility=0.85, recency_score=0.9,
            summary="s", agent="a",
        ),
    ]
    r = ResearchResult(ticker="K", query="q", items=items)
    d = r.to_dict()
    assert d["items"] and isinstance(d["items"][0], dict)
    assert isinstance(d["timestamp"], str)
    json.dumps(d)


# ── SentimentResult ──────────────────────────────────────────────────────────

def test_sentiment_to_spec_dict_keys_unchanged():
    s = SentimentResult(
        ticker="K",
        sentiment_score=0.3, narrative_direction="supports_yes",
        confidence=0.7, market_impact_estimate_cents=5,
        major_contradictions=[], item_count=3,
        contributing_sources=["Reuters"],
        source_credibility=0.85, event_relevance=0.9,
        rumor_risk="low",
    )
    d = s.to_spec_dict()
    assert set(d.keys()) == {
        "sentiment_score", "narrative_direction", "source_credibility",
        "event_relevance", "market_impact_estimate_cents", "confidence",
        "contradictions", "rumor_risk", "signal_clarity",
    }
    # full dict is JSON-safe too
    json.dumps(s.to_dict())


# ── ProbabilityEstimate ──────────────────────────────────────────────────────

def test_probability_estimate_exposes_blueprint_fields():
    p = ProbabilityEstimate(
        ticker="K", yes_probability=0.62, confidence="medium-high",
        reasoning="Solid evidence",
        assumptions=["a1"], invalidation_conditions=["i1"],
    )
    assert hasattr(p, "yes_probability")
    assert hasattr(p, "confidence")
    assert hasattr(p, "reasoning")
    assert hasattr(p, "assumptions")
    assert hasattr(p, "invalidation_conditions")
    # NO probability is derived and consistent
    assert p.no_probability == pytest.approx(1.0 - 0.62, abs=1e-6)
    d = p.to_dict()
    assert d["no_probability"] == pytest.approx(1.0 - d["yes_probability"], abs=1e-6)
    json.dumps(d)


def test_probability_estimate_clamps_in_to_dict():
    p = ProbabilityEstimate(
        ticker="K", yes_probability=1.5, confidence="high",
        reasoning="",
    )
    d = p.to_dict()
    assert 0.0 <= d["yes_probability"] <= 1.0
    assert 0.0 <= d["no_probability"] <= 1.0


# ── RiskAssessment ───────────────────────────────────────────────────────────

def test_risk_assessment_approved_and_rejected_serialize():
    ok = RiskAssessment(
        ticker="K", side="YES", approved=True, reason="ok", mode="paper",
        checks_passed=["a", "b"], checks_failed=[],
    )
    bad = RiskAssessment(
        ticker="K", side="YES", approved=False, reason="duplicate", mode="paper",
        checks_passed=["spread_ok"], checks_failed=["duplicate_position"],
    )
    assert ok.rejection_reasons == []
    assert bad.rejection_reasons == ["duplicate_position"]
    json.dumps(ok.to_dict())
    json.dumps(bad.to_dict())


# ── TradeDecision ────────────────────────────────────────────────────────────

_DECISION_SPEC_KEYS = {
    "action", "ticker", "side", "limit_price_cents", "contracts",
    "dollar_size", "thesis", "edge_cents", "confidence",
    "risk_summary", "monitoring_plan",
}


def test_trade_decision_from_formatter_dict_round_trips():
    raw = {
        "action": "BUY_YES", "ticker": "K", "side": "yes",
        "limit_price_cents": 55, "contracts": 5,
        "dollar_size": 2.75, "thesis": "thesis",
        "edge_cents": 8.0, "confidence": 0.80,
        "risk_summary": "approved (paper, 3 checks passed): ok",
        "monitoring_plan": ["watch spread"],
    }
    td = TradeDecision.from_dict(raw)
    assert td.action == "BUY_YES"
    assert td.side == "yes"
    assert td.is_trade is True
    out = td.to_dict()
    assert set(out.keys()) == _DECISION_SPEC_KEYS
    json.dumps(out)


def test_trade_decision_no_trade_round_trips_with_null_side():
    raw = {
        "action": "NO_TRADE", "ticker": "K", "side": None,
        "limit_price_cents": 0, "contracts": 0,
        "dollar_size": 0.0, "thesis": "thesis",
        "edge_cents": 0.0, "confidence": 0.3,
        "risk_summary": "edge_below_threshold",
        "monitoring_plan": [],
    }
    td = TradeDecision.from_dict(raw)
    assert td.side is None
    assert td.is_trade is False


def test_trade_decision_matches_formatter_output_keys():
    """The dataclass keys must match what decision_formatter emits."""
    import decision_formatter as df
    from config import Config
    cfg = Config(kalshi_api_key="x", anthropic_api_key="x",
                 kalshi_private_key_path="")
    market = _make_market()
    est = ProbabilityEstimate(
        ticker="KXTEST", yes_probability=0.62, confidence="medium-high",
        reasoning="thesis",
    )
    out = df.format_decision(market, est, None, None, None, cfg)
    td = TradeDecision.from_dict(out)
    assert set(td.to_dict().keys()) == set(out.keys()) == _DECISION_SPEC_KEYS


# ── ExecutionReport ──────────────────────────────────────────────────────────

def test_execution_report_to_dict_json_safe():
    er = ExecutionReport(
        ticker="K", mode="paper", status="OPEN",
        trade_id="abc", side="YES",
        contracts_requested=5, contracts_filled=5, contracts_remaining=0,
        limit_price_cents=55, avg_fill_price_cents=55,
        dollars_at_risk=2.75, placed=True,
    )
    d = er.to_dict()
    assert isinstance(d["timestamp"], str)
    json.dumps(d)
    assert er.succeeded is True


def test_execution_report_failed_does_not_count_as_succeeded():
    er = ExecutionReport(
        ticker="K", mode="live", status="FAILED",
        contracts_requested=5, contracts_remaining=5,
        limit_price_cents=55, error="rate_limited",
    )
    assert er.succeeded is False


def test_execution_report_does_not_place_orders():
    """ExecutionReport is a record, not an executor — no client / no calls."""
    # Construction must not require any client/cfg/secret.
    er = ExecutionReport(ticker="K", mode="dry_run", status="DRY_RUN")
    assert er.placed is False
    # And the class has no `place_order` / `execute` method.
    for attr in ("place_order", "execute", "submit"):
        assert not hasattr(er, attr), f"ExecutionReport must not expose {attr}"


# ── Postmortem / PostmortemReport ────────────────────────────────────────────

def test_postmortem_rule_change_is_proposal_not_auto_applied():
    pm = Postmortem(
        trade_id="t", ticker="K",
        original_thesis="x", estimated_yes_prob=0.6,
        market_price_at_entry=55, actual_result="loss",
        was_variance=False, data_was_stale=True,
        resolution_handled_correctly=True, liquidity_hurt=False,
        sizing_appropriate=True,
        analysis="bad fill", rule_change_proposal="raise spread cap to 4¢",
    )
    # Default not approved
    assert pm.human_approved is False
    # Blueprint alias exposes the proposal text
    assert pm.proposed_rule_change == "raise spread cap to 4¢"
    d = pm.to_dict()
    assert d["human_approved"] is False
    assert d["rule_change_proposal"] == "raise spread cap to 4¢"
    json.dumps(d)


def test_postmortem_report_alias_constructs_same_object():
    pm = PostmortemReport(
        trade_id="t", ticker="K",
        original_thesis="", estimated_yes_prob=0.5,
        market_price_at_entry=50, actual_result="win",
        was_variance=True, data_was_stale=False,
        resolution_handled_correctly=True, liquidity_hurt=False,
        sizing_appropriate=True,
        analysis="", rule_change_proposal="none",
    )
    assert isinstance(pm, Postmortem)


# ── TradeRecord / EdgeResult / PositionSize ──────────────────────────────────

def test_trade_record_to_dict_json_safe():
    tr = TradeRecord(
        id="abc", ticker="K", side="YES", contracts=5,
        entry_price_cents=55, dollars_at_risk=2.75, mode="paper",
        thesis="thesis", estimated_yes_prob=0.6, result="open",
    )
    d = tr.to_dict()
    assert isinstance(d["timestamp"], str)
    json.dumps(d)


def test_edge_result_to_dict_json_safe():
    e = EdgeResult(
        ticker="K", side="YES", entry_price_cents=55,
        implied_yes_prob=0.55, estimated_yes_prob=0.62,
        raw_edge_pct=7.0, adjusted_edge_pct=5.0,
        expected_value=0.07, adjusted_ev=0.05,
        confidence="medium-high",
        confidence_adjusted_ev=0.04, confidence_adjusted_edge_pct=4.0,
        spread_cents=2,
    )
    d = e.to_dict()
    json.dumps(d)


def test_position_size_to_dict_json_safe():
    ps = PositionSize(
        ticker="K", side="YES", dollars=2.75, contracts=5,
        entry_price_cents=55, max_loss_dollars=2.75,
    )
    d = ps.to_dict()
    json.dumps(d)
    assert d["contracts"] == 5
    assert d["dollars"] == 2.75


# ── No live trading wired in via the new schemas ────────────────────────────

def test_new_schemas_have_no_order_placement_methods():
    """Per safety constraint: schemas don't place trades or call APIs."""
    forbidden = {"place_order", "execute", "submit", "send_order", "buy", "sell"}
    for cls in (TradeDecision, ExecutionReport):
        attrs = set(dir(cls))
        assert not (attrs & forbidden), f"{cls.__name__} exposed: {attrs & forbidden}"
