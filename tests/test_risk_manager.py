from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import risk_manager
from config import Config
from models import EdgeResult, Market, PositionSize, TradeRecord


def cfg(**overrides) -> Config:
    base = dict(
        kalshi_api_key="x",
        anthropic_api_key="x",
        kalshi_private_key_path="",
        kill_switch=False,
        trading_mode="paper",
        paper_only=True,
        live_trading_enabled=False,
        max_trade_dollars=10.0,
        max_live_dollars_per_trade=1.0,
        max_position_pct_of_bankroll=0.5,
        max_daily_loss_dollars=20.0,
        max_trades_per_day=5,
        max_category_exposure_dollars=25.0,
        max_correlated_exposure_dollars=15.0,
        max_spread_cents=6,
        min_liquidity_dollars=500.0,
        min_edge_pct=5.0,
        min_adjusted_edge_pct=5.0,
        min_confidence=0.65,
        min_confidence_adjusted_edge_cents=1.0,
        max_spread_cents_edge=6,
    )
    base.update(overrides)
    return Config(**base)


def market(
    *,
    ticker: str = "KXRISK-TEST",
    event_ticker: str = "KXRISK",
    side_spread: int = 1,
    minutes_to_close: float = 120,
    liquidity_dollars: float = 5_000,
    category: str = "crypto",
) -> Market:
    now = datetime.now(timezone.utc)
    m = Market(
        ticker=ticker,
        title="Risk test market",
        status="open",
        yes_ask=40,
        yes_bid=40 - side_spread,
        no_ask=61,
        no_bid=60,
        volume=10_000,
        volume_24h=5_000,
        open_interest=2_000,
        close_time=now + timedelta(minutes=minutes_to_close),
        settlement_time=now + timedelta(minutes=minutes_to_close),
        category=category,
        rules_primary="rules",
        event_ticker=event_ticker,
    )
    m.minutes_to_close = minutes_to_close
    m.minutes_to_settlement = minutes_to_close
    m.liquidity_dollars = liquidity_dollars
    return m


def sizing(**overrides) -> PositionSize:
    base = dict(
        ticker="KXRISK-TEST",
        side="YES",
        dollars=5.0,
        contracts=10,
        entry_price_cents=40,
        max_loss_dollars=5.0,
    )
    base.update(overrides)
    return PositionSize(**base)


def edge_result(**overrides) -> EdgeResult:
    base = dict(
        ticker="KXRISK-TEST",
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


def trade(
    *,
    ticker: str = "KXRISK-TEST",
    side: str = "YES",
    mode: str = "paper",
) -> TradeRecord:
    return TradeRecord(
        id=f"{ticker}-{side}",
        ticker=ticker,
        side=side,
        contracts=5,
        entry_price_cents=40,
        dollars_at_risk=2.0,
        mode=mode,
        result="open",
    )


def assess(
    *,
    c: Config | None = None,
    m: Market | None = None,
    s: PositionSize | None = None,
    e: EdgeResult | None = None,
    open_positions: list[TradeRecord] | None = None,
    daily_pnl: float = 0.0,
    trades_today: int = 0,
    bankroll: float = 1_000.0,
    category_exposure: float = 0.0,
    correlated_exposure: float = 0.0,
    action_type: str = "entry",
    live_buy_guard: set[str] | None = None,
):
    return risk_manager.assess(
        s or sizing(),
        m or market(),
        e or edge_result(),
        open_positions=open_positions or [],
        daily_pnl=daily_pnl,
        trades_today=trades_today,
        bankroll=bankroll,
        cfg=c or cfg(),
        category_exposure=category_exposure,
        correlated_exposure=correlated_exposure,
        action_type=action_type,
        live_buy_guard=live_buy_guard,
    )


def assert_rejected(decision, token: str) -> None:
    assert decision.approved is False
    assert any(token in failure for failure in decision.checks_failed)


def test_default_config_has_safe_defaults(monkeypatch):
    for name in (
        "KILL_SWITCH",
        "TRADING_MODE",
        "LIVE_TRADING_ENABLED",
        "MAX_DOLLARS_PER_TRADE",
        "MAX_TRADE_DOLLARS",
        "MAX_LIVE_TRADE_DOLLARS",
        "MAX_LIVE_DOLLARS_PER_TRADE",
        "MAX_BANKROLL_PCT_PER_TRADE",
        "MAX_POSITION_PCT_OF_BANKROLL",
        "MAX_DAILY_LOSS",
        "MAX_DAILY_LOSS_DOLLARS",
        "MAX_TRADES_PER_DAY",
        "MAX_CATEGORY_EXPOSURE",
        "MAX_CATEGORY_EXPOSURE_DOLLARS",
        "MAX_CORRELATED_EXPOSURE",
        "MAX_CORRELATED_EXPOSURE_DOLLARS",
        "MAX_SPREAD_CENTS",
        "MIN_LIQUIDITY",
        "MIN_LIQUIDITY_DOLLARS",
        "PAPER_ONLY",
        "ALLOW_LIVE_ORDERS",
        "LIVE_CONFIRMATION_PHRASE",
        "DRY_RUN_LOG_ONLY",
        "MIN_PAPER_TRADES_BEFORE_LIVE",
        "MIN_PAPER_DAYS_BEFORE_LIVE",
        "MIN_PAPER_TRADING_DAYS_BEFORE_LIVE",
    ):
        monkeypatch.delenv(name, raising=False)

    c = Config(kalshi_api_key="x", anthropic_api_key="x", kalshi_private_key_path="")

    assert c.kill_switch is True
    assert c.trading_mode == "paper"
    assert c.paper_only is True
    assert c.live_trading_enabled is False
    assert c.allow_live_orders is False
    assert c.live_confirmation_phrase == ""
    assert c.dry_run_log_only is True
    assert c.max_trade_dollars == 10.0
    assert c.max_live_dollars_per_trade == 1.0
    assert c.max_live_trade_dollars == 1.0
    assert c.max_position_pct_of_bankroll == 0.5
    assert c.max_daily_loss_dollars == 20.0
    assert c.max_trades_per_day == 5
    assert c.max_category_exposure_dollars == 25.0
    assert c.max_correlated_exposure_dollars == 15.0
    assert c.max_spread_cents == 10
    assert c.min_liquidity_dollars == 25.0
    assert c.min_paper_days_before_live == 7
    assert c.min_paper_trades_before_live == 100


@pytest.mark.parametrize(
    "kwargs,token",
    [
        (
            {"c": cfg(kill_switch=True, trading_mode="live", live_trading_enabled=True)},
            "kill_switch_engaged",
        ),
        ({"c": cfg(trading_mode="chaos")}, "invalid_trading_mode"),
        ({"c": cfg(trading_mode="live", live_trading_enabled=False)}, "live_mode_not_enabled"),
        ({"daily_pnl": -20.0}, "daily_loss_limit"),
        ({"trades_today": 5}, "daily_trade_limit"),
        ({"s": sizing(dollars=11.0)}, "position_too_large"),
        ({"s": sizing(dollars=6.0)}, "bankroll_pct_exceeded"),
        ({"category_exposure": 25.01}, "category_exposure_cap"),
        ({"correlated_exposure": 15.01}, "correlated_exposure_cap"),
        ({"m": market(side_spread=7), "e": edge_result(spread_cents=7)}, "spread_too_wide"),
        ({"m": market(liquidity_dollars=499.0)}, "insufficient_liquidity"),
        ({"s": sizing(contracts=0)}, "non_positive_contracts"),
        ({"s": sizing(dollars=0.0)}, "non_positive_dollars"),
    ],
)
def test_hard_reject_gates(kwargs, token):
    assert_rejected(assess(**kwargs), token)


def test_spread_gate_uses_selected_no_side_spread():
    decision = assess(
        m=market(side_spread=20),
        s=sizing(side="NO", entry_price_cents=25),
        e=edge_result(
            side="NO",
            entry_price_cents=25,
            spread_cents=1,
            raw_edge_pct=12.0,
            adjusted_edge_pct=8.0,
            adjusted_ev=0.08,
            confidence_adjusted_ev=0.072,
            confidence_adjusted_edge_pct=7.2,
        ),
    )

    assert decision.approved is True


def test_duplicate_ticker_and_side_rejects():
    decision = assess(open_positions=[trade(ticker="KXRISK-TEST", side="YES")])

    assert_rejected(decision, "duplicate_position")


def test_same_event_ticker_related_group_duplicate_rejects():
    decision = assess(
        m=market(ticker="KXRISK-TEST", event_ticker="KXRISK"),
        open_positions=[trade(ticker="KXRISK-OTHER", side="YES")],
    )

    assert_rejected(decision, "related_group_position_open")


def test_opposite_side_same_market_rejects_new_entry():
    decision = assess(open_positions=[trade(ticker="KXRISK-TEST", side="NO")])

    assert_rejected(decision, "opposite_side_open")


def test_less_than_five_minutes_rejects_entry():
    decision = assess(m=market(minutes_to_close=4))

    assert_rejected(decision, "time_entry_rejected")


def test_five_to_twenty_minutes_rejects_new_entry():
    decision = assess(m=market(minutes_to_close=10))

    assert_rejected(decision, "time_entry_rejected")


def test_twenty_to_sixty_minutes_requires_medium_high_confidence():
    decision = assess(
        m=market(minutes_to_close=45),
        e=edge_result(confidence="medium", confidence_adjusted_ev=0.0975),
    )

    assert_rejected(decision, "near_resolution_confidence_low")


def test_twenty_to_sixty_minutes_requires_smaller_size():
    decision = assess(m=market(minutes_to_close=45), s=sizing(dollars=6.0))

    assert_rejected(decision, "near_resolution_size_too_large")


def test_one_to_twenty_four_hours_can_approve_when_all_checks_pass():
    decision = assess()

    assert decision.approved is True


def test_kill_switch_does_not_block_valid_paper_trade():
    decision = assess(c=cfg(kill_switch=True, trading_mode="paper"))

    assert decision.approved is True
    assert any("kill_switch_live_only" in check for check in decision.checks_passed)


def test_long_horizon_rejects_unsupported_categories():
    decision = assess(m=market(minutes_to_close=73 * 60, category="politics"))

    assert_rejected(decision, "unsupported_long_horizon")


def test_edge_after_costs_still_independently_rejects():
    decision = assess(e=edge_result(adjusted_edge_pct=4.0, adjusted_ev=0.04))

    assert_rejected(decision, "edge_too_small")


def test_risk_manager_does_not_approve_edge_layer_rejection():
    decision = assess(e=edge_result(confidence="low", confidence_adjusted_ev=0.045))

    assert_rejected(decision, "edge_thresholds_failed")


def test_live_buy_guard_rejects_duplicate_live_key(monkeypatch):
    monkeypatch.setattr(risk_manager.db, "count_paper_trading_days", lambda: 0)
    monkeypatch.setattr(risk_manager.db, "count_completed_trades", lambda: 0)
    monkeypatch.setattr(risk_manager.db, "total_paper_pnl", lambda: 0.0)

    c = cfg(
        trading_mode="live",
        paper_only=False,
        live_trading_enabled=True,
        allow_live_orders=True,
        live_confirmation_phrase="I_UNDERSTAND_THIS_CAN_LOSE_MONEY",
        min_paper_days_before_live=0,
        min_paper_trades_before_live=0,
        min_paper_pnl_before_live=0.0,
    )

    decision = assess(
        c=c,
        s=sizing(dollars=1.0),
        live_buy_guard={"KXRISK-TEST:YES"},
    )

    assert_rejected(decision, "live_buy_guard_duplicate")


def test_live_prerequisite_failure_does_not_query_paper_history(monkeypatch):
    monkeypatch.setattr(
        risk_manager.db,
        "count_paper_trading_days",
        lambda: (_ for _ in ()).throw(AssertionError("paper history queried")),
    )
    c = cfg(trading_mode="live", live_trading_enabled=True, paper_only=True)

    decision = assess(c=c, s=sizing(dollars=1.0))

    assert_rejected(decision, "PAPER_ONLY")
