from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import edge
import risk_manager
from config import Config
from models import EdgeResult, Market, PositionSize, ProbabilityEstimate


def cfg(**overrides) -> Config:
    base = dict(
        kalshi_api_key="x",
        anthropic_api_key="x",
        kalshi_private_key_path="",
        kill_switch=False,
        trading_mode="paper",
        min_edge_pct=5.0,
        min_adjusted_edge_pct=5.0,
        slippage_cents=0,
        fee_pct=0.0,
        min_confidence=0.65,
        min_confidence_adjusted_edge_cents=4.0,
        max_spread_cents=20,
        max_spread_cents_edge=6,
        min_liquidity_dollars=500,
    )
    base.update(overrides)
    return Config(**base)


def market(
    *,
    yes_ask: int = 40,
    yes_bid: int = 38,
    no_ask: int = 62,
    no_bid: int = 60,
    ticker: str = "KXEDGE-TEST",
) -> Market:
    now = datetime.now(timezone.utc)
    m = Market(
        ticker=ticker,
        title="Test market",
        status="open",
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=no_ask,
        no_bid=no_bid,
        volume=10_000,
        volume_24h=5_000,
        open_interest=2_000,
        close_time=now + timedelta(hours=2),
        settlement_time=now + timedelta(hours=2),
        category="crypto",
        rules_primary="rules",
    )
    m.minutes_to_close = 120
    m.minutes_to_settlement = 120
    m.liquidity_dollars = 5_000
    return m


def estimate(p_yes: float, confidence: str = "medium-high") -> ProbabilityEstimate:
    return ProbabilityEstimate(
        ticker="KXEDGE-TEST",
        yes_probability=p_yes,
        confidence=confidence,
        reasoning="test",
    )


def test_yes_edge_uses_yes_ask():
    result = edge.calculate(
        market(yes_ask=45, yes_bid=44, no_ask=60, no_bid=59),
        estimate(0.55),
        cfg(),
    )

    assert result is not None
    assert result.side == "YES"
    assert result.entry_price_cents == 45
    assert result.raw_edge_pct == pytest.approx(10.0)
    assert result.expected_value == pytest.approx(0.10)


def test_no_edge_uses_no_ask_and_one_minus_yes_probability():
    result = edge.calculate(
        market(yes_ask=80, yes_bid=79, no_ask=25, no_bid=24),
        estimate(0.60),
        cfg(),
    )

    assert result is not None
    assert result.side == "NO"
    assert result.entry_price_cents == 25
    assert result.raw_edge_pct == pytest.approx(15.0)
    assert result.expected_value == pytest.approx(0.15)


def test_ev_simplifies_to_probability_minus_price():
    result = edge.calculate(
        market(yes_ask=37, yes_bid=36, no_ask=66, no_bid=65),
        estimate(0.52),
        cfg(),
    )

    assert result is not None
    assert result.side == "YES"
    assert result.expected_value == pytest.approx(0.52 - 0.37)
    assert result.adjusted_ev == pytest.approx(result.adjusted_edge_pct / 100.0)


def test_slippage_lowers_adjusted_edge_and_ev():
    m = market(yes_ask=40, yes_bid=39, no_ask=65, no_bid=64)
    no_slip = edge.calculate(m, estimate(0.60), cfg(slippage_cents=0))
    with_slip = edge.calculate(m, estimate(0.60), cfg(slippage_cents=2))

    assert no_slip is not None and with_slip is not None
    assert with_slip.adjusted_ev == pytest.approx(no_slip.adjusted_ev - 0.02)
    assert with_slip.adjusted_edge_pct == pytest.approx(no_slip.adjusted_edge_pct - 2.0)


def test_fees_lower_adjusted_edge_and_ev():
    m = market(yes_ask=40, yes_bid=39, no_ask=65, no_bid=64)
    no_fee = edge.calculate(m, estimate(0.60), cfg(fee_pct=0.0))
    with_fee = edge.calculate(m, estimate(0.60), cfg(fee_pct=2.0))

    assert no_fee is not None and with_fee is not None
    assert with_fee.adjusted_ev == pytest.approx(no_fee.adjusted_ev - 0.02)
    assert with_fee.adjusted_edge_pct == pytest.approx(no_fee.adjusted_edge_pct - 2.0)


def test_spread_cost_is_spread_cents_divided_by_200():
    result = edge.calculate(
        market(yes_ask=40, yes_bid=36, no_ask=65, no_bid=64),
        estimate(0.60),
        cfg(),
    )

    assert result is not None
    assert result.raw_edge_pct == pytest.approx(20.0)
    assert result.adjusted_ev == pytest.approx(0.20 - (4 / 200))
    assert result.adjusted_edge_pct == pytest.approx(18.0)


def test_confidence_adjusted_edge_decreases_with_lower_confidence():
    m = market(yes_ask=40, yes_bid=39, no_ask=65, no_bid=64)
    high = edge.calculate(m, estimate(0.60, "high"), cfg())
    medium = edge.calculate(m, estimate(0.60, "medium"), cfg())

    assert high is not None and medium is not None
    assert medium.confidence_adjusted_ev < high.confidence_adjusted_ev
    assert medium.confidence_adjusted_edge_pct < high.confidence_adjusted_edge_pct


def test_low_confidence_fails_threshold_even_with_positive_raw_edge():
    result = edge.calculate(
        market(yes_ask=40, yes_bid=39, no_ask=65, no_bid=64),
        estimate(0.65, "low"),
        cfg(min_confidence_adjusted_edge_cents=1.0),
    )

    assert result is not None
    assert result.adjusted_edge_pct > 0
    assert edge.passes_threshold(result, cfg(min_confidence_adjusted_edge_cents=1.0)) is False


def test_adjusted_edge_less_than_or_equal_to_zero_fails():
    result = EdgeResult(
        ticker="KXEDGE-TEST",
        side="YES",
        entry_price_cents=50,
        implied_yes_prob=0.50,
        estimated_yes_prob=0.60,
        raw_edge_pct=10.0,
        adjusted_edge_pct=0.0,
        expected_value=0.10,
        adjusted_ev=0.0,
        confidence="high",
        confidence_adjusted_ev=0.0,
        confidence_adjusted_edge_pct=0.0,
        spread_cents=1,
    )

    assert edge.passes_threshold(result, cfg(min_adjusted_edge_pct=0.0)) is False


def test_raw_edge_below_min_edge_fails():
    result = edge.calculate(
        market(yes_ask=40, yes_bid=39, no_ask=65, no_bid=64),
        estimate(0.445),
        cfg(min_edge_pct=5.0, min_adjusted_edge_pct=1.0, min_confidence_adjusted_edge_cents=1.0),
    )

    assert result is not None
    assert result.raw_edge_pct == pytest.approx(4.5)
    assert edge.passes_threshold(
        result,
        cfg(min_edge_pct=5.0, min_adjusted_edge_pct=1.0, min_confidence_adjusted_edge_cents=1.0),
    ) is False


def test_confidence_adjusted_edge_below_threshold_fails():
    result = edge.calculate(
        market(yes_ask=40, yes_bid=39, no_ask=65, no_bid=64),
        estimate(0.455, "medium"),
        cfg(min_adjusted_edge_pct=1.0),
    )

    assert result is not None
    assert result.confidence_adjusted_ev * 100 < 4.0
    assert edge.passes_threshold(result, cfg(min_adjusted_edge_pct=1.0)) is False


def test_wide_spread_fails():
    result = edge.calculate(
        market(yes_ask=40, yes_bid=20, no_ask=80, no_bid=79),
        estimate(0.90, "high"),
        cfg(max_spread_cents_edge=6, min_confidence_adjusted_edge_cents=1.0),
    )

    assert result is not None
    assert result.spread_cents == 20
    assert edge.passes_threshold(
        result,
        cfg(max_spread_cents_edge=6, min_confidence_adjusted_edge_cents=1.0),
    ) is False


def test_returns_none_when_neither_side_has_positive_adjusted_edge():
    result = edge.calculate(
        market(yes_ask=60, yes_bid=59, no_ask=60, no_bid=59),
        estimate(0.50),
        cfg(),
    )

    assert result is None


def test_chooses_better_adjusted_side_when_both_yes_and_no_are_positive():
    result = edge.calculate(
        market(yes_ask=40, yes_bid=38, no_ask=40, no_bid=39),
        estimate(0.54),
        cfg(),
    )

    assert result is not None
    assert result.side == "YES"
    assert result.adjusted_edge_pct == pytest.approx(13.0)


def test_probability_and_ev_fields_are_sane_and_bounded():
    result = edge.calculate(
        market(yes_ask=40, yes_bid=39, no_ask=65, no_bid=64),
        estimate(0.60, "medium-high"),
        cfg(),
    )

    assert result is not None
    assert 0 <= result.estimated_yes_prob <= 1
    assert 0 <= result.implied_yes_prob <= 1
    assert 0 <= result.entry_price_cents <= 100
    assert -1 <= result.expected_value <= 1
    assert -1 <= result.adjusted_ev <= 1
    assert result.adjusted_ev == pytest.approx(result.adjusted_edge_pct / 100.0)
    assert result.confidence_adjusted_ev == pytest.approx(
        result.confidence_adjusted_edge_pct / 100.0
    )


def test_risk_manager_does_not_approve_trade_rejected_by_edge_layer():
    c = cfg(min_confidence_adjusted_edge_cents=1.0)
    m = market(yes_ask=40, yes_bid=39, no_ask=65, no_bid=64)
    result = edge.calculate(m, estimate(0.65, "low"), c)

    assert result is not None
    assert edge.passes_threshold(result, c) is False

    sizing = PositionSize(
        ticker=result.ticker,
        side=result.side,
        dollars=10.0,
        contracts=25,
        entry_price_cents=result.entry_price_cents,
        max_loss_dollars=10.0,
    )
    decision = risk_manager.assess(
        sizing,
        m,
        result,
        open_positions=[],
        daily_pnl=0.0,
        trades_today=0,
        bankroll=1_000.0,
        cfg=c,
    )

    assert decision.approved is False
    assert "edge_thresholds_failed" in decision.checks_failed
