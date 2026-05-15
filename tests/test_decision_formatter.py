"""
Tests for decision_formatter.format_decision().

The formatter is a thin, deterministic glue layer: it combines upstream
outputs (estimate / edge / sizing / risk_assessment) into the final
trade-decision JSON. Critical guarantees under test:

  - Any failing upstream gate → NO_TRADE (no exceptions).
  - Risk manager is final authority.
  - Inputs are never mutated.
  - Output is JSON-serializable.
  - Output schema matches the spec exactly.
"""
from __future__ import annotations
import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import decision_formatter as df
from config import Config
from models import (
    EdgeResult,
    Market,
    PositionSize,
    ProbabilityEstimate,
    RiskAssessment,
)


_SPEC_KEYS = {
    "action", "ticker", "side", "limit_price_cents", "contracts",
    "dollar_size", "thesis", "edge_cents", "confidence",
    "risk_summary", "monitoring_plan",
}


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> Config:
    return Config(
        kalshi_api_key="x", anthropic_api_key="x",
        kalshi_private_key_path="",
    )


def _market(ticker: str = "KXTEST") -> Market:
    now = datetime.now(timezone.utc)
    m = Market(
        ticker=ticker, title="t", status="open",
        yes_ask=55, yes_bid=53, no_ask=47, no_bid=45,
        volume=2000, volume_24h=12_000, open_interest=4_000,
        close_time=now + timedelta(minutes=120),
        settlement_time=now + timedelta(minutes=120),
        category="crypto", rules_primary="r", event_ticker=ticker,
        price_history=[{"yes_price": 54}, {"yes_price": 55}],
    )
    m.minutes_to_settlement = 120
    m.liquidity_dollars = 5_000
    return m


def _estimate(confidence: str = "medium-high") -> ProbabilityEstimate:
    return ProbabilityEstimate(
        ticker="KXTEST",
        yes_probability=0.62,
        confidence=confidence,
        reasoning="Fed minutes confirm hawkish path.",
        assumptions=[],
        invalidation_conditions=[],
    )


def _edge_yes(
    *,
    adj_edge_pct: float = 8.0,
    raw_edge_pct: float = 10.0,
    confidence: str = "medium-high",
    spread_cents: int = 2,
) -> EdgeResult:
    return EdgeResult(
        ticker="KXTEST",
        side="YES",
        entry_price_cents=55,
        implied_yes_prob=0.55,
        estimated_yes_prob=0.62,
        raw_edge_pct=raw_edge_pct,
        adjusted_edge_pct=adj_edge_pct,
        expected_value=raw_edge_pct / 100.0,
        adjusted_ev=adj_edge_pct / 100.0,
        confidence=confidence,
        confidence_adjusted_ev=(adj_edge_pct / 100.0) * 0.80,
        confidence_adjusted_edge_pct=adj_edge_pct * 0.80,
        spread_cents=spread_cents,
    )


def _edge_no() -> EdgeResult:
    return EdgeResult(
        ticker="KXTEST",
        side="NO",
        entry_price_cents=47,
        implied_yes_prob=0.55,
        estimated_yes_prob=0.40,
        raw_edge_pct=10.0,
        adjusted_edge_pct=8.0,
        expected_value=0.10,
        adjusted_ev=0.08,
        confidence="medium-high",
        confidence_adjusted_ev=0.064,
        confidence_adjusted_edge_pct=6.4,
        spread_cents=2,
    )


def _sizing(contracts: int = 5, dollars: float = 2.75, entry: int = 55) -> PositionSize:
    return PositionSize(
        ticker="KXTEST", side="YES",
        dollars=dollars, contracts=contracts,
        entry_price_cents=entry, max_loss_dollars=dollars,
    )


def _risk(
    *, approved: bool = True, reason: str = "ok",
    failed: list[str] | None = None,
) -> RiskAssessment:
    return RiskAssessment(
        ticker="KXTEST", side="YES",
        approved=approved, reason=reason, mode="paper",
        checks_passed=["kill_switch_ok", "size_ok", "spread_ok"],
        checks_failed=failed or [],
    )


# ── Schema ───────────────────────────────────────────────────────────────────

def test_output_schema_matches_spec(cfg):
    out = df.format_decision(
        _market(), _estimate(), _edge_yes(), _sizing(), _risk(), cfg,
    )
    assert set(out.keys()) == _SPEC_KEYS


def test_output_is_json_serializable(cfg):
    out = df.format_decision(
        _market(), _estimate(), _edge_yes(), _sizing(), _risk(), cfg,
    )
    json.dumps(out)  # must not raise

    out_block = df.format_decision(
        _market(), _estimate(confidence="low"),
        _edge_yes(), _sizing(), _risk(approved=False, reason="too_risky"), cfg,
    )
    json.dumps(out_block)


# ── Approved trades ──────────────────────────────────────────────────────────

def test_approved_buy_yes_formats_correctly(cfg):
    out = df.format_decision(
        _market(), _estimate(), _edge_yes(), _sizing(), _risk(), cfg,
    )
    assert out["action"] == "BUY_YES"
    assert out["side"] == "yes"
    assert out["ticker"] == "KXTEST"
    assert out["limit_price_cents"] == 55
    assert out["contracts"] == 5
    assert out["dollar_size"] == pytest.approx(2.75)
    assert out["edge_cents"] == pytest.approx(8.0)
    assert 0.0 <= out["confidence"] <= 1.0
    assert "Fed minutes" in out["thesis"]
    assert isinstance(out["monitoring_plan"], list) and out["monitoring_plan"]
    assert "approved" in out["risk_summary"]


def test_approved_buy_no_formats_correctly(cfg):
    out = df.format_decision(
        _market(), _estimate(), _edge_no(),
        PositionSize(ticker="KXTEST", side="NO", dollars=2.35, contracts=5,
                     entry_price_cents=47, max_loss_dollars=2.35),
        _risk(),
        cfg,
    )
    assert out["action"] == "BUY_NO"
    assert out["side"] == "no"
    assert out["limit_price_cents"] == 47
    assert out["contracts"] == 5


# ── Gating: every blocker yields NO_TRADE ────────────────────────────────────

def _assert_no_trade(out: dict) -> None:
    assert out["action"] == "NO_TRADE"
    assert out["side"] is None
    assert out["limit_price_cents"] == 0
    assert out["contracts"] == 0
    assert out["dollar_size"] == 0.0
    assert out["monitoring_plan"] == []


def test_risk_rejection_forces_no_trade_even_with_good_edge(cfg):
    out = df.format_decision(
        _market(), _estimate(), _edge_yes(), _sizing(),
        _risk(approved=False, reason="duplicate_position",
              failed=["duplicate_position: KXTEST YES already open"]),
        cfg,
    )
    _assert_no_trade(out)
    assert "risk_rejected" in out["risk_summary"]
    assert "duplicate_position" in out["risk_summary"]


def test_below_threshold_edge_forces_no_trade(cfg):
    # adjusted_edge well below cfg.min_adjusted_edge_pct=5
    weak = _edge_yes(adj_edge_pct=0.5, raw_edge_pct=1.0)
    out = df.format_decision(
        _market(), _estimate(), weak, _sizing(), _risk(), cfg,
    )
    _assert_no_trade(out)
    assert "edge_below_threshold" in out["risk_summary"]


def test_low_confidence_forces_no_trade(cfg):
    # cfg.min_confidence = 0.65 → "low" (0.30) fails
    out = df.format_decision(
        _market(), _estimate(confidence="low"),
        _edge_yes(confidence="low"), _sizing(), _risk(), cfg,
    )
    _assert_no_trade(out)
    assert "confidence_below_threshold" in out["risk_summary"]


def test_zero_contracts_forces_no_trade(cfg):
    out = df.format_decision(
        _market(), _estimate(), _edge_yes(),
        PositionSize(ticker="KXTEST", side="YES", dollars=0.0, contracts=0,
                     entry_price_cents=55, max_loss_dollars=0.0),
        _risk(), cfg,
    )
    _assert_no_trade(out)
    assert "zero_size" in out["risk_summary"]


def test_zero_dollars_forces_no_trade(cfg):
    out = df.format_decision(
        _market(), _estimate(), _edge_yes(),
        PositionSize(ticker="KXTEST", side="YES", dollars=0.0, contracts=3,
                     entry_price_cents=55, max_loss_dollars=0.0),
        _risk(), cfg,
    )
    _assert_no_trade(out)
    assert "zero_size" in out["risk_summary"]


def test_missing_edge_forces_no_trade(cfg):
    out = df.format_decision(
        _market(), _estimate(), None, _sizing(), _risk(), cfg,
    )
    _assert_no_trade(out)
    assert "no_edge" in out["risk_summary"]


def test_missing_sizing_forces_no_trade(cfg):
    out = df.format_decision(
        _market(), _estimate(), _edge_yes(), None, _risk(), cfg,
    )
    _assert_no_trade(out)
    assert "no_sizing" in out["risk_summary"]


def test_missing_risk_assessment_forces_no_trade(cfg):
    out = df.format_decision(
        _market(), _estimate(), _edge_yes(), _sizing(), None, cfg,
    )
    _assert_no_trade(out)
    assert "no_risk_assessment" in out["risk_summary"]


# ── Risk manager remains final authority ─────────────────────────────────────

def test_risk_manager_overrides_strong_inputs(cfg):
    """Even with strong edge + ample sizing + high confidence, a risk veto wins."""
    out = df.format_decision(
        _market(), _estimate(confidence="high"),
        _edge_yes(adj_edge_pct=20.0, raw_edge_pct=22.0, confidence="high"),
        _sizing(contracts=20, dollars=11.0),
        _risk(approved=False, reason="kill_switch_active",
              failed=["kill_switch_active"]),
        cfg,
    )
    assert out["action"] == "NO_TRADE"
    assert "kill_switch" in out["risk_summary"]


# ── Input non-mutation ───────────────────────────────────────────────────────

def test_formatter_does_not_mutate_inputs(cfg):
    market = _market()
    estimate = _estimate()
    edge = _edge_yes()
    sizing = _sizing()
    risk = _risk()

    market_copy = copy.deepcopy(market)
    estimate_copy = copy.deepcopy(estimate)
    edge_copy = copy.deepcopy(edge)
    sizing_copy = copy.deepcopy(sizing)
    risk_copy = copy.deepcopy(risk)

    df.format_decision(market, estimate, edge, sizing, risk, cfg)

    assert market == market_copy
    assert estimate == estimate_copy
    assert edge == edge_copy
    assert sizing == sizing_copy
    assert risk == risk_copy


# ── Monitoring plan content (deterministic, no new facts) ────────────────────

def test_monitoring_plan_references_only_known_fields(cfg):
    out = df.format_decision(
        _market(), _estimate(), _edge_yes(spread_cents=3), _sizing(), _risk(), cfg,
    )
    plan = out["monitoring_plan"]
    joined = " ".join(plan)
    assert "spread" in joined.lower()
    assert "thesis" in joined.lower()
    # No invented URLs or sources
    assert "http" not in joined.lower()


def test_no_trade_has_empty_monitoring_plan(cfg):
    out = df.format_decision(
        _market(), _estimate(), None, _sizing(), _risk(), cfg,
    )
    assert out["monitoring_plan"] == []


# ── Edge side translation ────────────────────────────────────────────────────

def test_unknown_edge_side_falls_back_to_no_trade(cfg):
    weird_side = EdgeResult(
        ticker="KXTEST", side="??", entry_price_cents=55,
        implied_yes_prob=0.55, estimated_yes_prob=0.62,
        raw_edge_pct=10.0, adjusted_edge_pct=8.0,
        expected_value=0.10, adjusted_ev=0.08,
        confidence="medium-high",
        confidence_adjusted_ev=0.064,
        confidence_adjusted_edge_pct=6.4,
        spread_cents=2,
    )
    out = df.format_decision(
        _market(), _estimate(), weird_side, _sizing(), _risk(), cfg,
    )
    assert out["action"] == "NO_TRADE"
    assert out["side"] is None


# ── Confidence in output is the numeric weight, not the label ────────────────

def test_confidence_is_numeric_weight(cfg):
    for label, weight in [("low", 0.30), ("medium", 0.65), ("high", 0.90)]:
        # Build inputs that would otherwise pass; only label varies
        risk = _risk()
        sizing = _sizing()
        edge = _edge_yes(confidence=label, adj_edge_pct=20.0, raw_edge_pct=22.0)
        out = df.format_decision(
            _market(), _estimate(confidence=label), edge, sizing, risk, cfg,
        )
        assert out["confidence"] == pytest.approx(weight)


# ── Stacked blockers don't crash; all reasons are reported ───────────────────

def test_multiple_block_reasons_all_recorded(cfg):
    weak_edge = _edge_yes(adj_edge_pct=0.5, raw_edge_pct=1.0, confidence="low")
    out = df.format_decision(
        _market(), _estimate(confidence="low"),
        weak_edge,
        PositionSize(ticker="KXTEST", side="YES", dollars=0.0, contracts=0,
                     entry_price_cents=55, max_loss_dollars=0.0),
        _risk(approved=False, reason="size_too_large"),
        cfg,
    )
    _assert_no_trade(out)
    summary = out["risk_summary"]
    # Each blocker should be referenced once
    for token in ("edge_below_threshold", "confidence_below_threshold",
                  "zero_size", "risk_rejected"):
        assert token in summary
