from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import weird_move
from models import Market, MarketSnapshot


@pytest.fixture(autouse=True)
def reset_snapshot_store(monkeypatch):
    monkeypatch.setattr(weird_move, "_store", weird_move.SnapshotStore())


def make_market(
    ticker: str = "KXTEST-26MAY-T1",
    *,
    yes_ask: int = 52,
    yes_bid: int = 50,
    volume: int = 1_000,
    volume_24h: int = 1_000,
    open_interest: int = 1_000,
    event_ticker: str = "KXTEST-26MAY",
    price_history: list[dict] | None = None,
) -> Market:
    now = datetime.now(timezone.utc)
    return Market(
        ticker=ticker,
        title="Test market",
        status="open",
        yes_ask=yes_ask,
        yes_bid=yes_bid,
        no_ask=100 - yes_bid,
        no_bid=100 - yes_ask,
        volume=volume,
        volume_24h=volume_24h,
        open_interest=open_interest,
        close_time=now + timedelta(hours=2),
        settlement_time=now + timedelta(hours=2),
        category="crypto",
        rules_primary="Test rules",
        event_ticker=event_ticker,
        price_history=price_history or [],
    )


def add_snapshot(ticker: str, minutes_ago: float, *, mid_yes: float, spread_cents: int = 2, volume: int = 1_000) -> None:
    weird_move.get_store().add(
        MarketSnapshot(
            ticker=ticker,
            mid_yes=mid_yes,
            spread_cents=spread_cents,
            volume=volume,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        )
    )


def recent_trade(minutes_ago: float = 1, **size_fields) -> dict:
    return {
        "created_time": (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **size_fields,
    }


def test_weird_move_detects_5m_price_jump():
    market = make_market(yes_ask=62, yes_bid=60)  # mid 61
    add_snapshot(market.ticker, 5.0, mid_yes=52)

    signal = weird_move.detect(market, [market])

    assert signal.flagged is True
    assert "price_5m" in signal.triggers
    assert signal.price_change_5m >= 8


def test_weird_move_detects_15m_price_jump():
    market = make_market(yes_ask=64, yes_bid=62)  # mid 63
    add_snapshot(market.ticker, 15.0, mid_yes=50)

    signal = weird_move.detect(market, [market])

    assert signal.flagged is True
    assert "price_15m" in signal.triggers
    assert signal.price_change_15m >= 12
    assert "price_5m" not in signal.triggers


def test_weird_move_detects_volume_spike():
    market = make_market(
        volume_24h=100,
        price_history=[recent_trade(count=4)],  # avg baseline is clamped to 1.0, ratio = 4.0
    )

    signal = weird_move.detect(market, [market])

    assert signal.flagged is True
    assert "volume_ratio" in signal.triggers
    assert signal.volume_ratio >= 3.0


def test_weird_move_detects_spread_widening():
    market = make_market(yes_ask=55, yes_bid=50)  # current spread 5
    for minutes_ago in (50, 30, 10):
        add_snapshot(market.ticker, minutes_ago, mid_yes=51, spread_cents=2)

    signal = weird_move.detect(market, [market])

    assert signal.flagged is True
    assert "spread_change" in signal.triggers
    assert signal.spread_change >= 2.0


def test_weird_move_detects_related_market_dislocation():
    market = make_market(ticker="KXTEST-26MAY-A", yes_ask=52, yes_bid=50, event_ticker="KXTEST-26MAY")
    sibling = make_market(ticker="KXTEST-26MAY-B", yes_ask=72, yes_bid=70, event_ticker="KXTEST-26MAY")

    signal = weird_move.detect(market, [market, sibling])

    assert signal.flagged is True
    assert "related_disagreement" in signal.triggers
    assert signal.related_disagreement >= 10
    assert signal.classification == "related_market_dislocation"


def test_weird_move_classifies_thin_book_jump_as_liquidity_gap():
    market = make_market(yes_ask=62, yes_bid=60, open_interest=50)  # thin book, mid 61
    add_snapshot(market.ticker, 5.0, mid_yes=52)

    signal = weird_move.detect(market, [market])

    assert signal.flagged is True
    assert "price_5m" in signal.triggers
    assert signal.classification == "liquidity_gap_move"


def test_weird_move_classifies_spread_no_volume_as_stale_book_artifact():
    market = make_market(yes_ask=56, yes_bid=50, price_history=[])  # current spread 6, no recent trades
    for minutes_ago in (50, 30, 10):
        add_snapshot(market.ticker, minutes_ago, mid_yes=51, spread_cents=2)

    signal = weird_move.detect(market, [market])

    assert signal.flagged is True
    assert "spread_change" in signal.triggers
    assert signal.classification == "stale_book_artifact"


def test_batch_detect_returns_signal_for_each_market():
    markets = [
        make_market(ticker="KXTEST-26MAY-A", event_ticker="KXTEST-26MAY"),
        make_market(ticker="KXTEST-26MAY-B", event_ticker="KXTEST-26MAY"),
    ]

    results = weird_move.batch_detect(markets, markets)

    assert set(results) == {m.ticker for m in markets}
    assert all(hasattr(signal, "flagged") for signal in results.values())
