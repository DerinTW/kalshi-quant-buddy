"""
Tests for features.extract_features() — the probability-model feature
vector builder.

Covers every listed feature plus fail-soft behavior when optional inputs
(sentiment, weird_move, spot_price, markets_by_event) are absent.
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import features
from models import (
    FeatureVector,
    Market,
    SentimentResult,
    WeirdMoveSignal,
)


# ── Builders ─────────────────────────────────────────────────────────────────

def make_market(
    ticker: str = "KXBTCD-25NOV13-B104000",
    *,
    title: str = "BTC price above 104000 by 5pm CT",
    category: str = "crypto",
    yes_ask: int = 55,
    yes_bid: int = 52,
    volume_24h: int = 12_000,
    open_interest: int = 4_000,
    orderbook_depth: int = 300,
    minutes_to_settlement: float = 90.0,
    price_history: list[dict] | None = None,
    event_ticker: str | None = None,
) -> Market:
    now = datetime.now(timezone.utc)
    m = Market(
        ticker=ticker,
        title=title,
        status="open",
        yes_ask=yes_ask, yes_bid=yes_bid,
        no_ask=100 - yes_bid, no_bid=100 - yes_ask,
        volume=volume_24h * 3, volume_24h=volume_24h,
        open_interest=open_interest,
        close_time=now + timedelta(minutes=minutes_to_settlement),
        settlement_time=now + timedelta(minutes=minutes_to_settlement),
        category=category, rules_primary="rules",
        event_ticker=event_ticker or ticker.rsplit("-", 1)[0],
        price_history=price_history or [],
    )
    m.minutes_to_settlement = minutes_to_settlement
    m.minutes_to_close = minutes_to_settlement
    m.orderbook_depth = orderbook_depth
    return m


def make_sentiment(
    score: float = 0.4,
    cred: float = 0.85,
    rel: float = 0.9,
) -> SentimentResult:
    return SentimentResult(
        ticker="KX",
        sentiment_score=score,
        narrative_direction="supports_yes" if score > 0 else "supports_no",
        confidence=0.6,
        market_impact_estimate_cents=5,
        major_contradictions=[],
        item_count=3,
        contributing_sources=["Reuters"],
        source_credibility=cred,
        event_relevance=rel,
        rumor_risk="low",
    )


def make_weird_move(
    p5: float = 1.5,
    p15: float = -0.5,
    vol_ratio: float = 2.3,
) -> WeirdMoveSignal:
    return WeirdMoveSignal(
        ticker="KX",
        flagged=False,
        classification="none",
        price_change_5m=p5,
        price_change_15m=p15,
        volume_ratio=vol_ratio,
        spread_change=1.0,
        related_disagreement=0.0,
        triggers=[],
        description="",
        confidence="low",
    )


# ── Microstructure features ──────────────────────────────────────────────────

def test_microstructure_fields():
    m = make_market(yes_ask=58, yes_bid=54, volume_24h=15_000,
                    open_interest=4_200, orderbook_depth=275,
                    minutes_to_settlement=90)
    fv = features.extract_features(m, make_sentiment(), make_weird_move())

    assert fv.ticker == m.ticker
    assert fv.current_yes_ask == 58
    assert fv.current_yes_bid == 54
    assert fv.mid_price == pytest.approx(56.0)
    assert fv.spread == 4
    assert fv.volume == 15_000
    assert fv.open_interest == 4_200
    assert fv.order_book_depth == 275
    assert fv.time_to_resolution_minutes == pytest.approx(90.0)
    assert fv.category == "crypto"


def test_spread_handles_missing_bid():
    """yes_bid=0 should not give a negative spread."""
    m = make_market(yes_ask=60, yes_bid=0)
    fv = features.extract_features(m, make_sentiment(), make_weird_move())
    assert fv.spread >= 0


# ── Price dynamics from weird_move ───────────────────────────────────────────

def test_price_dynamics_from_weird_move():
    fv = features.extract_features(
        make_market(),
        make_sentiment(),
        make_weird_move(p5=2.0, p15=-1.5, vol_ratio=3.4),
    )
    assert fv.price_change_5m == 2.0
    assert fv.price_change_15m == -1.5
    assert fv.volume_spike_ratio == 3.4


def test_price_dynamics_default_when_weird_move_missing():
    fv = features.extract_features(make_market(), make_sentiment(), weird_move=None)
    assert fv.price_change_5m == 0.0
    assert fv.price_change_15m == 0.0
    # Volume spike defaults to 1.0 (no spike), not 0.0
    assert fv.volume_spike_ratio == 1.0


# ── Narrative features from sentiment ────────────────────────────────────────

def test_narrative_features_from_sentiment():
    s = make_sentiment(score=-0.6, cred=0.92, rel=0.75)
    fv = features.extract_features(make_market(), s, make_weird_move())
    assert fv.sentiment_score == pytest.approx(-0.6)
    assert fv.source_credibility == pytest.approx(0.92)
    assert fv.event_relevance == pytest.approx(0.75)


def test_narrative_defaults_when_sentiment_missing():
    fv = features.extract_features(make_market(), sentiment=None,
                                   weird_move=make_weird_move())
    assert fv.sentiment_score == 0.0
    assert fv.source_credibility == 0.0
    assert fv.event_relevance == 0.0


# ── Historical volatility ────────────────────────────────────────────────────

def test_historical_volatility_from_price_history():
    history = [
        {"yes_price": 50},
        {"yes_price": 52},
        {"yes_price": 49},
        {"yes_price": 55},
        {"yes_price": 51},
    ]
    fv = features.extract_features(
        make_market(price_history=history),
        make_sentiment(), make_weird_move(),
    )
    # Diffs: +2, -3, +6, -4 → stdev > 0
    assert fv.historical_volatility > 0
    # Sanity: should be less than max |diff|
    assert fv.historical_volatility < 6.0


def test_historical_volatility_flat_history_is_zero():
    history = [{"yes_price": 50}] * 5
    fv = features.extract_features(
        make_market(price_history=history),
        make_sentiment(), make_weird_move(),
    )
    assert fv.historical_volatility == 0.0


def test_historical_volatility_empty_history_is_zero():
    fv = features.extract_features(
        make_market(price_history=[]),
        make_sentiment(), make_weird_move(),
    )
    assert fv.historical_volatility == 0.0


def test_historical_volatility_handles_alt_field_names():
    """Kalshi history rows vary; the extractor should try yes_ask + yes_bid too."""
    history = [
        {"yes_ask": 51, "yes_bid": 49},
        {"yes_ask": 55, "yes_bid": 53},
        {"yes_ask": 52, "yes_bid": 50},
    ]
    fv = features.extract_features(
        make_market(price_history=history),
        make_sentiment(), make_weird_move(),
    )
    assert fv.historical_volatility > 0


# ── Distance to threshold ────────────────────────────────────────────────────

def test_distance_to_threshold_when_spot_and_strike_known():
    m = make_market(title="BTC price above 104000 by 5pm CT")
    fv = features.extract_features(m, make_sentiment(), make_weird_move(),
                                   spot_price=103_000.0)
    # |103000 - 104000| / 104000 ≈ 0.00962
    assert fv.distance_to_threshold == pytest.approx(0.00961, abs=1e-3)


def test_distance_to_threshold_handles_dollars_and_commas():
    m = make_market(title="ETH close above $3,500 on Friday")
    fv = features.extract_features(m, make_sentiment(), make_weird_move(),
                                   spot_price=3_650.0)
    assert fv.distance_to_threshold == pytest.approx(abs(3650 - 3500) / 3500)


def test_distance_to_threshold_none_without_spot():
    m = make_market(title="BTC price above 104000")
    fv = features.extract_features(m, make_sentiment(), make_weird_move(),
                                   spot_price=None)
    assert fv.distance_to_threshold is None


def test_distance_to_threshold_none_without_strike():
    m = make_market(title="Will the Eagles win the Super Bowl?")
    fv = features.extract_features(m, make_sentiment(), make_weird_move(),
                                   spot_price=42.0)
    assert fv.distance_to_threshold is None


# ── Related market prices (siblings) ─────────────────────────────────────────

def test_related_market_prices_from_event_siblings():
    self_market = make_market(ticker="KXBTCD-25NOV13-B104000",
                              event_ticker="KXBTCD-25NOV13",
                              yes_ask=55, yes_bid=53)
    sibling_a = make_market(ticker="KXBTCD-25NOV13-B100000",
                            event_ticker="KXBTCD-25NOV13",
                            yes_ask=82, yes_bid=80)
    sibling_b = make_market(ticker="KXBTCD-25NOV13-B108000",
                            event_ticker="KXBTCD-25NOV13",
                            yes_ask=28, yes_bid=25)
    by_event = {"KXBTCD-25NOV13": [self_market, sibling_a, sibling_b]}

    fv = features.extract_features(self_market, make_sentiment(), make_weird_move(),
                                   markets_by_event=by_event)
    # Self excluded; siblings included as implied YES probabilities (mid/100)
    assert len(fv.related_market_prices) == 2
    # Sibling A mid = 81 → 0.81; B mid = 26.5 → 0.265
    assert pytest.approx(0.81, abs=0.01) in fv.related_market_prices
    assert pytest.approx(0.265, abs=0.01) in fv.related_market_prices


def test_related_market_prices_empty_without_event_map():
    fv = features.extract_features(make_market(), make_sentiment(), make_weird_move(),
                                   markets_by_event=None)
    assert fv.related_market_prices == []


# ── Fail-soft top-level: minimal inputs ──────────────────────────────────────

def test_extract_with_all_optionals_missing_does_not_crash():
    fv = features.extract_features(make_market())   # no sentiment, no weird_move
    assert isinstance(fv, FeatureVector)
    assert fv.sentiment_score == 0.0
    assert fv.price_change_5m == 0.0
    assert fv.volume_spike_ratio == 1.0
    assert fv.distance_to_threshold is None
    assert fv.related_market_prices == []


# ── Spec dict shape ──────────────────────────────────────────────────────────

def test_to_dict_contains_all_listed_features():
    fv = features.extract_features(
        make_market(),
        make_sentiment(),
        make_weird_move(),
        spot_price=103_000.0,
    )
    d = fv.to_dict()
    required = {
        "current_yes_bid", "current_yes_ask", "mid_price", "spread",
        "volume", "open_interest", "order_book_depth",
        "time_to_resolution_minutes",
        "price_change_5m", "price_change_15m", "volume_spike_ratio",
        "category", "related_market_prices",
        "sentiment_score", "source_credibility", "event_relevance",
        "historical_volatility", "distance_to_threshold",
    }
    assert required.issubset(d.keys())


# ── batch_extract ────────────────────────────────────────────────────────────

def test_batch_extract_returns_one_vector_per_market():
    m1 = make_market(ticker="A-100", event_ticker="A")
    m2 = make_market(ticker="A-200", event_ticker="A", yes_ask=30, yes_bid=28)
    m3 = make_market(ticker="B-50",  event_ticker="B")

    s_map = {"A-100": make_sentiment(0.3), "A-200": make_sentiment(-0.2)}
    w_map = {"A-100": make_weird_move(), "B-50": make_weird_move(p5=4.0)}

    out = features.batch_extract([m1, m2, m3], s_map, w_map)

    assert set(out.keys()) == {"A-100", "A-200", "B-50"}
    # Missing sentiment → defaults
    assert out["B-50"].sentiment_score == 0.0
    # Missing weird_move → defaults
    assert out["A-200"].price_change_5m == 0.0
    # B-50 got its real weird_move
    assert out["B-50"].price_change_5m == 4.0
    # Event A has two markets → each sees the other as a sibling
    assert len(out["A-100"].related_market_prices) == 1
    assert len(out["A-200"].related_market_prices) == 1
    # Event B has only one market → no siblings
    assert out["B-50"].related_market_prices == []
