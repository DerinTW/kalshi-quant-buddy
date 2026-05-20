from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import risk_manager
from config import Config
from models import EdgeResult, Market, PositionSize


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_DEFAULT_ENV_NAMES = (
    "KILL_SWITCH",
    "TRADING_MODE",
    "PAPER_ONLY",
    "LIVE_TRADING_ENABLED",
    "ALLOW_LIVE_ORDERS",
    "LIVE_CONFIRMATION_PHRASE",
    "DRY_RUN_LOG_ONLY",
    "MAX_LIVE_TRADE_DOLLARS",
    "MAX_LIVE_DOLLARS_PER_TRADE",
    "MIN_PAPER_TRADES_BEFORE_LIVE",
    "MIN_PAPER_TRADING_DAYS_BEFORE_LIVE",
    "MIN_PAPER_DAYS_BEFORE_LIVE",
)


def _env_value(text: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}=(.*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


@pytest.fixture(autouse=True)
def clear_safety_env(monkeypatch):
    for name in REQUIRED_DEFAULT_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in (
        "MIN_VOLUME",
        "MIN_VOLUME_24H",
        "MIN_TIME_TO_RESOLUTION_MINUTES",
        "MIN_MINUTES_TO_EXPIRY",
        "MAX_TIME_TO_RESOLUTION_HOURS",
        "MAX_MINUTES_TO_EXPIRY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_config_default_safety_values():
    c = Config(kalshi_api_key="x", anthropic_api_key="x", kalshi_private_key_path="")

    assert c.kill_switch is True
    assert c.trading_mode == "paper"
    assert c.paper_only is True
    assert c.live_trading_enabled is False
    assert c.allow_live_orders is False
    assert c.live_confirmation_phrase == ""
    assert c.dry_run_log_only is True
    assert c.max_live_dollars_per_trade == 1.0
    assert c.min_paper_days_before_live == 7
    assert c.min_paper_trades_before_live == 100
    assert c.max_spread_pct == 20.0
    assert c.min_volume_24h == 500


def test_config_backwards_compatible_env_aliases(monkeypatch):
    monkeypatch.setenv("MAX_LIVE_TRADE_DOLLARS", "0.75")
    monkeypatch.setenv("MIN_PAPER_TRADING_DAYS_BEFORE_LIVE", "9")
    monkeypatch.setenv("MIN_VOLUME", "2222")
    monkeypatch.setenv("MIN_TIME_TO_RESOLUTION_MINUTES", "33")
    monkeypatch.setenv("MAX_TIME_TO_RESOLUTION_HOURS", "48")

    c = Config(kalshi_api_key="x", anthropic_api_key="x", kalshi_private_key_path="")

    assert c.max_live_dollars_per_trade == 0.75
    assert c.max_live_trade_dollars == 0.75
    assert c.min_paper_days_before_live == 9
    assert c.min_volume_24h == 2222
    assert c.min_minutes_to_expiry == 33
    assert c.max_minutes_to_expiry == 48 * 60


def test_env_example_contains_required_safety_variables():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert _env_value(text, "TRADING_MODE") == "paper"
    assert _env_value(text, "KILL_SWITCH") == "true"
    assert _env_value(text, "PAPER_ONLY") == "true"
    assert _env_value(text, "LIVE_TRADING_ENABLED") == "false"
    assert _env_value(text, "ALLOW_LIVE_ORDERS") == "false"
    assert _env_value(text, "LIVE_CONFIRMATION_PHRASE") == ""
    assert _env_value(text, "DRY_RUN_LOG_ONLY") == "true"
    assert _env_value(text, "MAX_LIVE_TRADE_DOLLARS") == "1"
    assert _env_value(text, "MIN_PAPER_TRADES_BEFORE_LIVE") == "100"
    assert _env_value(text, "DEBUG") == "false"
    assert _env_value(text, "MAX_SPREAD_PCT") == "20"
    assert _env_value(text, "MIN_VOLUME_24H") == "500"


def test_env_example_contains_weird_move_documented_constants():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "weird_move.py currently uses module constants" in text
    assert _env_value(text, "PRICE_JUMP_5M_CENTS") == "8"
    assert _env_value(text, "PRICE_JUMP_15M_CENTS") == "12"
    assert _env_value(text, "VOLUME_SPIKE_RATIO") == "3.0"
    assert _env_value(text, "SPREAD_WIDEN_RATIO") == "2.0"
    assert _env_value(text, "RELATED_MARKET_DISAGREEMENT_CENTS") == "10"


def test_live_mode_rejected_until_all_gates_are_explicitly_satisfied(monkeypatch):
    monkeypatch.setattr(risk_manager.db, "count_paper_trading_days", lambda: 7)
    monkeypatch.setattr(risk_manager.db, "count_completed_trades", lambda: 100)
    monkeypatch.setattr(risk_manager.db, "total_paper_pnl", lambda: 0.0)

    base = _live_config()
    rejected = risk_manager.assess(
        _sizing(),
        _market(),
        _edge(),
        open_positions=[],
        daily_pnl=0.0,
        trades_today=0,
        bankroll=1_000.0,
        cfg=base,
    )

    assert rejected.approved is False
    assert any("ALLOW_LIVE_ORDERS" in failure for failure in rejected.checks_failed)

    approved = risk_manager.assess(
        _sizing(dollars=1.0),
        _market(),
        _edge(),
        open_positions=[],
        daily_pnl=0.0,
        trades_today=0,
        bankroll=1_000.0,
        cfg=_live_config(
            allow_live_orders=True,
            live_confirmation_phrase="I_UNDERSTAND_THIS_CAN_LOSE_MONEY",
        ),
    )

    assert approved.approved is True


def _live_config(**overrides) -> Config:
    base = dict(
        kalshi_api_key="x",
        anthropic_api_key="x",
        kalshi_private_key_path="",
        kill_switch=False,
        trading_mode="live",
        paper_only=False,
        live_trading_enabled=True,
        allow_live_orders=False,
        live_confirmation_phrase="",
        max_trade_dollars=1.0,
        max_live_dollars_per_trade=1.0,
        min_paper_days_before_live=7,
        min_paper_trades_before_live=100,
        min_paper_pnl_before_live=0.0,
        max_position_pct_of_bankroll=0.5,
        min_adjusted_edge_pct=5,
        min_confidence_adjusted_edge_cents=1,
    )
    base.update(overrides)
    return Config(**base)


def _sizing(**overrides) -> PositionSize:
    base = dict(
        ticker="KXCONFIG-TEST",
        side="YES",
        dollars=1.0,
        contracts=2,
        entry_price_cents=40,
        max_loss_dollars=1.0,
    )
    base.update(overrides)
    return PositionSize(**base)


def _market() -> Market:
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    m = Market(
        ticker="KXCONFIG-TEST",
        title="Config live gates test",
        status="open",
        yes_ask=40,
        yes_bid=39,
        no_ask=61,
        no_bid=60,
        volume=10_000,
        volume_24h=5_000,
        open_interest=2_000,
        close_time=now + timedelta(minutes=120),
        settlement_time=now + timedelta(minutes=120),
        category="crypto",
        rules_primary="rules",
        event_ticker="KXCONFIG",
        liquidity_dollars=5_000,
        minutes_to_close=120,
    )
    return m


def _edge() -> EdgeResult:
    return EdgeResult(
        ticker="KXCONFIG-TEST",
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
