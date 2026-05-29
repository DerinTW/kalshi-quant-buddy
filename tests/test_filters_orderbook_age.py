from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import filters
from config import Config
from models import Market


def make_cfg() -> Config:
    return Config(
        kalshi_api_key="x",
        anthropic_api_key="x",
        kalshi_private_key_path="",
        min_liquidity_dollars=100,
        max_spread_pct=25,
        min_minutes_to_expiry=20,
        max_minutes_to_expiry=24 * 60,
        min_volume_24h=100,
        min_yes_price=10,
        max_yes_price=90,
        max_spread_cents=10,
        max_orderbook_age_seconds=60,
        min_orderbook_depth_at_limit=100,
        category_allowlist=["crypto", "economic", "financial", "weather"],
        blocked_tickers=set(),
    )


def valid_market(**overrides) -> Market:
    now = datetime.now(timezone.utc)
    values = {
        "ticker": "KXTEST-26MAY19",
        "title": "Will the test market resolve yes?",
        "status": "open",
        "yes_ask": 45,
        "yes_bid": 42,
        "no_ask": 58,
        "no_bid": 55,
        "volume": 5000,
        "volume_24h": 2500,
        "open_interest": 5000,
        "close_time": now + timedelta(hours=2),
        "settlement_time": now + timedelta(hours=2, minutes=5),
        "category": "crypto",
        "rules_primary": "Test rules.",
        "spread_pct": 6.9,
        "minutes_to_close": 120.0,
        "minutes_to_settlement": 125.0,
        "liquidity_dollars": 1500.0,
        "is_unsafe": False,
        "event_ticker": "KXTEST-26MAY19",
        "fetched_at": now - timedelta(seconds=5),
        "last_trade_at": None,
        "orderbook_depth": 200,
        "orderbook_depth_fetched": True,
    }
    values.update(overrides)
    return Market(**values)


def test_check_orderbook_age_returns_none_when_fetched_at_is_recent():
    market = valid_market(fetched_at=datetime.now(timezone.utc) - timedelta(seconds=5))

    assert filters._check_orderbook_age(market, make_cfg()) is None


def test_check_orderbook_age_rejects_when_fetched_at_is_stale():
    market = valid_market(fetched_at=datetime.now(timezone.utc) - timedelta(seconds=120))

    reason = filters._check_orderbook_age(market, make_cfg())

    assert reason is not None
    assert "snapshot_stale" in reason
    assert reason.startswith("snapshot_stale")


def test_check_orderbook_age_returns_none_when_fetched_at_is_none():
    market = valid_market(fetched_at=None)

    assert filters._check_orderbook_age(market, make_cfg()) is None


def test_stale_last_trade_at_alone_does_not_reject_when_fetched_at_is_recent():
    old_trade_time = datetime.now(timezone.utc) - timedelta(hours=6)
    market = valid_market(
        fetched_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        last_trade_at=old_trade_time,
    )

    assert filters._check_orderbook_age(market, make_cfg()) is None
    assert market.last_trade_at == old_trade_time


def test_none_last_trade_at_does_not_prevent_rejection_when_fetched_at_is_stale():
    market = valid_market(
        fetched_at=datetime.now(timezone.utc) - timedelta(seconds=120),
        last_trade_at=None,
    )

    reason = filters._check_orderbook_age(market, make_cfg())

    assert reason is not None
    assert reason.startswith("snapshot_stale")


def test_run_rejects_stale_fetched_at_and_groups_reason_under_snapshot_stale():
    market = valid_market(fetched_at=datetime.now(timezone.utc) - timedelta(seconds=120))

    result = filters.run([market], make_cfg())

    assert result.passed == []
    assert result.rejected == [(market, result.rejected[0][1])]
    assert result.skip_reason_counts == {"snapshot_stale": 1}


def test_run_passes_recent_fetched_at_when_other_filter_criteria_pass():
    market = valid_market(fetched_at=datetime.now(timezone.utc) - timedelta(seconds=5))

    result = filters.run([market], make_cfg())

    assert result.passed == [market]
    assert result.rejected == []


def test_naive_fetched_at_datetimes_are_treated_as_utc():
    market = valid_market(fetched_at=datetime.now() - timedelta(seconds=120))

    reason = filters._check_orderbook_age(market, make_cfg())

    assert reason is not None
    assert reason.startswith("snapshot_stale")


def test_regression_stale_snapshot_rejected_even_when_last_trade_at_is_none():
    market = valid_market(
        fetched_at=datetime.now(timezone.utc) - timedelta(seconds=120),
        last_trade_at=None,
    )

    reason = filters._check_orderbook_age(market, make_cfg())

    assert reason is not None
    assert reason.startswith("snapshot_stale")


def test_rejection_reason_no_longer_starts_with_orderbook_stale():
    market = valid_market(fetched_at=datetime.now(timezone.utc) - timedelta(seconds=120))

    reason = filters._check_orderbook_age(market, make_cfg())

    assert reason is not None
    assert not reason.startswith("orderbook_stale")


def test_last_trade_at_can_exist_independently_on_market():
    last_trade_at = datetime.now(timezone.utc) - timedelta(hours=3)
    fetched_at = datetime.now(timezone.utc) - timedelta(seconds=5)

    market = valid_market(fetched_at=fetched_at, last_trade_at=last_trade_at)

    assert market.fetched_at == fetched_at
    assert market.last_trade_at == last_trade_at
    assert filters._check_orderbook_age(market, make_cfg()) is None
