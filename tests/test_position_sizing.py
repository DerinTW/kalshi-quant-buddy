from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import position_sizing
import trading
from config import Config
from models import EdgeResult, Market, ProbabilityEstimate


def cfg(**overrides) -> Config:
    base = dict(
        kalshi_api_key="x",
        anthropic_api_key="x",
        kalshi_private_key_path="",
        kill_switch=False,
        trading_mode="paper",
        max_trade_dollars=10.0,
        max_live_dollars_per_trade=1.0,
        paper_bankroll=100_000.0,
        max_position_pct_of_bankroll=10.0,
    )
    base.update(overrides)
    return Config(**base)


def market(
    *,
    liquidity_dollars: float = 10_000.0,
    orderbook_depth: int = 0,
    minutes_to_settlement: float = 120.0,
) -> Market:
    now = datetime.now(timezone.utc)
    m = Market(
        ticker="KXSIZE-TEST",
        title="Sizing test market",
        status="open",
        yes_ask=40,
        yes_bid=39,
        no_ask=61,
        no_bid=60,
        volume=10_000,
        volume_24h=5_000,
        open_interest=2_000,
        close_time=now + timedelta(minutes=minutes_to_settlement),
        settlement_time=now + timedelta(minutes=minutes_to_settlement),
        category="crypto",
        rules_primary="rules",
    )
    m.minutes_to_close = minutes_to_settlement
    m.minutes_to_settlement = minutes_to_settlement
    m.liquidity_dollars = liquidity_dollars
    m.orderbook_depth = orderbook_depth
    return m


def edge_result(**overrides) -> EdgeResult:
    base = dict(
        ticker="KXSIZE-TEST",
        side="YES",
        entry_price_cents=40,
        implied_yes_prob=0.40,
        estimated_yes_prob=0.60,
        raw_edge_pct=20.0,
        adjusted_edge_pct=15.0,
        expected_value=0.20,
        adjusted_ev=0.15,
        confidence="high",
        confidence_adjusted_ev=0.135,
        confidence_adjusted_edge_pct=13.5,
        spread_cents=1,
    )
    base.update(overrides)
    return EdgeResult(**base)


def estimate(**overrides) -> ProbabilityEstimate:
    base = dict(
        ticker="KXSIZE-TEST",
        yes_probability=0.60,
        confidence="high",
        reasoning="test",
    )
    base.update(overrides)
    return ProbabilityEstimate(**base)


@pytest.fixture(autouse=True)
def _no_open_exposure(monkeypatch):
    monkeypatch.setattr(position_sizing.db, "total_open_exposure", lambda: 0.0)


def test_caps_at_max_trade_dollars():
    size = position_sizing.compute(market(), edge_result(entry_price_cents=40), estimate(), cfg())

    assert size.contracts == 25
    assert size.dollars == pytest.approx(10.0)


def test_caps_at_bankroll_percentage():
    c = cfg(paper_bankroll=1_000.0, max_position_pct_of_bankroll=0.5, max_trade_dollars=10.0)

    size = position_sizing.compute(market(), edge_result(entry_price_cents=40), estimate(), c)

    assert size.dollars <= 5.0
    assert size.contracts == 12
    assert size.dollars == pytest.approx(4.8)


def test_caps_at_liquidity_limit():
    size = position_sizing.compute(
        market(liquidity_dollars=10.0),
        edge_result(entry_price_cents=40),
        estimate(),
        cfg(),
    )

    assert size.dollars == pytest.approx(2.0)
    assert size.contracts == 5


def test_caps_at_visible_orderbook_depth_when_available():
    size = position_sizing.compute(
        market(liquidity_dollars=10_000.0, orderbook_depth=10),
        edge_result(entry_price_cents=50),
        estimate(),
        cfg(),
    )

    assert size.dollars == pytest.approx(1.0)
    assert size.contracts == 2


def test_returns_zero_contracts_when_size_below_one_contract():
    size = position_sizing.compute(
        market(liquidity_dollars=2.5),
        edge_result(entry_price_cents=90),
        estimate(),
        cfg(),
    )

    assert size.contracts == 0
    assert size.dollars == 0.0


def test_zero_contract_size_does_not_execute_trade(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.orders = []

        def place_order(self, **kwargs):
            self.orders.append(kwargs)
            return {"order": {"order_id": "x"}}

    client = FakeClient()
    zero = position_sizing.compute(
        market(liquidity_dollars=2.5),
        edge_result(entry_price_cents=90),
        estimate(),
        cfg(),
    )
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: None)

    record = trading.execute(zero, estimate(), edge_result(), cfg(), client=client)

    assert record is None
    assert client.orders == []


def test_does_not_use_kelly_confidence_or_edge_scaling():
    m = market()
    c = cfg()
    conservative = position_sizing.compute(
        m,
        edge_result(adjusted_edge_pct=5.0, adjusted_ev=0.05, confidence="medium"),
        estimate(yes_probability=0.52, confidence="medium"),
        c,
    )
    aggressive = position_sizing.compute(
        m,
        edge_result(adjusted_edge_pct=50.0, adjusted_ev=0.50, confidence="high"),
        estimate(yes_probability=0.95, confidence="high"),
        c,
    )

    assert conservative.contracts == aggressive.contracts
    assert conservative.dollars == pytest.approx(aggressive.dollars)


def test_execution_risk_penalty_reduces_size_directly():
    size = position_sizing.compute(
        market(),
        edge_result(entry_price_cents=40),
        estimate(confidence_breakdown={"execution_risk_penalties": ["near_resolution"]}),
        cfg(),
    )

    assert size.contracts == 12
    assert size.dollars == pytest.approx(4.8)


def test_time_to_resolution_haircut_reduces_long_horizon_size():
    size = position_sizing.compute(
        market(minutes_to_settlement=4 * 24 * 60),
        edge_result(entry_price_cents=40),
        estimate(),
        cfg(),
    )

    assert position_sizing.time_to_resolution_size_multiplier(
        market(minutes_to_settlement=4 * 24 * 60),
        cfg(),
    ) == pytest.approx(0.5)
    assert size.contracts == 12
    assert size.dollars == pytest.approx(4.8)


def test_time_to_resolution_haircut_does_not_boost_intraday_size():
    size = position_sizing.compute(
        market(minutes_to_settlement=6 * 60),
        edge_result(entry_price_cents=40),
        estimate(),
        cfg(),
    )

    assert size.contracts == 25
    assert size.dollars == pytest.approx(10.0)


def test_handles_yes_entry_price():
    size = position_sizing.compute(market(), edge_result(side="YES", entry_price_cents=40), estimate(), cfg())

    assert size.side == "YES"
    assert size.entry_price_cents == 40
    assert size.contracts == 25


def test_handles_no_entry_price():
    size = position_sizing.compute(market(), edge_result(side="NO", entry_price_cents=35), estimate(), cfg())

    assert size.side == "NO"
    assert size.entry_price_cents == 35
    assert size.contracts == 28
    assert size.dollars == pytest.approx(9.8)


def test_live_mode_caps_at_live_test_max():
    c = cfg(trading_mode="live", max_trade_dollars=10.0, max_live_dollars_per_trade=1.0)

    size = position_sizing.compute(market(), edge_result(entry_price_cents=40), estimate(), c)

    assert size.dollars <= 1.0
    assert size.contracts == 2
    assert size.dollars == pytest.approx(0.8)
