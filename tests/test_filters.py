"""
Filter-pipeline correctness tests.

Covers the structural filter pass before any LLM/research/edge work:

  * a structurally valid fake market passes
  * invalid markets fail for the correct reason
  * skip_reason_counts / skip_reason_examples are populated usefully
  * category filtering accepts Kalshi's human-friendly category strings
    ("Crypto", "Climate and Weather", "Economics", "Financials") when the
    allowlist uses the short lowercase form (regression for the case where
    pass_rate dropped to 0 because of a case/alias mismatch)
  * orderbook-depth gate uses the `orderbook_depth_fetched` flag so that
    markets where enrichment never ran are rejected before downstream work
  * orderbook freshness uses fetched_at (not last_trade_at)
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import filters
from config import Config
from models import Market


def make_cfg(**overrides) -> Config:
    base = dict(
        kalshi_api_key="x",
        anthropic_api_key="x",
        kalshi_private_key_path="",
        min_liquidity_dollars=500,
        max_spread_pct=20,
        min_minutes_to_expiry=20,
        max_minutes_to_expiry=4320,
        min_volume_24h=500,
        min_yes_price=15,
        max_yes_price=85,
        max_spread_cents=6,
        max_orderbook_age_seconds=60,
        min_orderbook_depth_at_limit=100,
        category_allowlist=[
            "crypto",
            "economic",
            "financial",
            "commodities",
            "weather",
            "science and technology",
            "culture",
        ],
        blocked_tickers=set(),
    )
    base.update(overrides)
    return Config(**base)


def valid_market(**overrides) -> Market:
    now = datetime.now(timezone.utc)
    values = dict(
        ticker="KXBTCD-26MAY20-B100000",
        title="Will BTC be above $100,000 today?",
        status="open",
        yes_ask=45,
        yes_bid=42,
        no_ask=58,
        no_bid=55,
        volume=10_000,
        volume_24h=5_000,
        open_interest=5_000,
        close_time=now + timedelta(hours=2),
        settlement_time=now + timedelta(hours=2, minutes=5),
        category="crypto",
        rules_primary="Resolves YES if BTC > $100k at close.",
        spread_pct=6.9,
        minutes_to_close=120.0,
        minutes_to_settlement=125.0,
        liquidity_dollars=1500.0,
        is_unsafe=False,
        event_ticker="KXBTCD-26MAY20",
        fetched_at=now - timedelta(seconds=5),
        last_trade_at=now - timedelta(minutes=2),
        orderbook_depth=500,
        orderbook_depth_fetched=True,
    )
    values.update(overrides)
    return Market(**values)


# ── Smoke: structurally valid market passes ──────────────────────────────────

def test_structurally_valid_market_passes_all_checks():
    result = filters.run([valid_market()], make_cfg())

    assert len(result.passed) == 1
    assert result.rejected == []
    assert result.pass_rate == 1.0


# ── Per-check: each invalid market fails for the correct reason ──────────────

def test_status_not_open_rejected():
    m = valid_market(status="closed")
    result = filters.run([m], make_cfg())
    assert result.passed == []
    assert list(result.skip_reason_counts.keys()) == ["status"]


def test_unsafe_market_rejected_with_reason():
    m = valid_market(is_unsafe=True, unsafe_reason="missing_title")
    result = filters.run([m], make_cfg())
    assert result.passed == []
    assert "unsafe" in result.skip_reason_counts


def test_blocked_ticker_rejected():
    m = valid_market(ticker="KXBLOCK-1")
    cfg = make_cfg(blocked_tickers={"KXBLOCK-1"})
    result = filters.run([m], cfg)
    assert result.passed == []
    assert result.skip_reason_counts == {"blocked_ticker": 1}


def test_too_close_to_close_rejected():
    now = datetime.now(timezone.utc)
    m = valid_market(
        close_time=now + timedelta(minutes=5),
        settlement_time=now + timedelta(minutes=5),
        minutes_to_close=5.0,
        minutes_to_settlement=5.0,
    )
    result = filters.run([m], make_cfg())
    assert result.passed == []
    assert list(result.skip_reason_counts.keys()) == ["too_close_to_close"]


def test_too_far_to_close_rejected():
    now = datetime.now(timezone.utc)
    m = valid_market(
        close_time=now + timedelta(days=30),
        settlement_time=now + timedelta(days=30),
        minutes_to_close=30 * 24 * 60.0,
        minutes_to_settlement=30 * 24 * 60.0,
    )
    result = filters.run([m], make_cfg())
    assert result.passed == []
    assert list(result.skip_reason_counts.keys()) == ["too_far_to_close"]


def test_price_out_of_range_rejected():
    m_low = valid_market(yes_ask=5, yes_bid=3)
    m_high = valid_market(ticker="KXHI-1", yes_ask=95, yes_bid=93)
    result = filters.run([m_low, m_high], make_cfg())
    assert result.passed == []
    counts = result.skip_reason_counts
    # Both come back under the same "yes_ask" prefix
    assert sum(counts.values()) == 2


def test_zero_or_negative_yes_ask_rejected():
    m = valid_market(yes_ask=0)
    result = filters.run([m], make_cfg())
    assert result.passed == []
    assert "no_valid_price" in result.skip_reason_counts


def test_wide_absolute_spread_rejected():
    m = valid_market(yes_ask=60, yes_bid=40)  # 20¢ spread > 6¢ cap
    result = filters.run([m], make_cfg())
    assert result.passed == []
    assert "spread" in result.skip_reason_counts


def test_low_volume_24h_rejected():
    m = valid_market(volume_24h=50)
    result = filters.run([m], make_cfg(min_volume_24h=500))
    assert result.passed == []
    assert "volume_24h" in result.skip_reason_counts


def test_low_liquidity_rejected():
    m = valid_market(liquidity_dollars=10.0)
    result = filters.run([m], make_cfg(min_liquidity_dollars=500))
    assert result.passed == []
    assert "liquidity" in result.skip_reason_counts


# ── Orderbook snapshot freshness uses fetched_at (NOT last_trade_at) ─────────

def test_orderbook_freshness_uses_fetched_at_not_last_trade_at():
    now = datetime.now(timezone.utc)
    # Stale fetched_at must reject even when last_trade_at is recent.
    stale_snapshot = valid_market(
        fetched_at=now - timedelta(minutes=10),
        last_trade_at=now - timedelta(seconds=2),
    )
    result = filters.run([stale_snapshot], make_cfg(max_orderbook_age_seconds=60))
    assert result.passed == []
    assert "snapshot_stale" in result.skip_reason_counts

    # Conversely, a stale last_trade_at must NOT alone trigger rejection
    # when the fetched_at snapshot is fresh.
    stale_trades_only = valid_market(
        fetched_at=now - timedelta(seconds=5),
        last_trade_at=now - timedelta(hours=6),
    )
    result = filters.run([stale_trades_only], make_cfg(max_orderbook_age_seconds=60))
    assert len(result.passed) == 1


def test_fetched_at_none_is_lenient():
    # Legacy / hand-built markets without a fetched_at must not be rejected
    # by the snapshot-age gate alone (they will still fail other gates if
    # the data is bad).
    m = valid_market(fetched_at=None)
    assert filters._check_orderbook_age(m, make_cfg()) is None


# ── Orderbook depth gate respects orderbook_depth_fetched flag ───────────────

def test_depth_below_min_rejected_when_fetched():
    m = valid_market(orderbook_depth=10, orderbook_depth_fetched=True)
    result = filters.run([m], make_cfg(min_orderbook_depth_at_limit=100))
    assert result.passed == []
    assert "depth" in result.skip_reason_counts


def test_zero_depth_after_successful_fetch_is_a_rejection_not_a_pass():
    # Critical: previously orderbook_depth==0 silently passed. Now if the
    # fetch ran (orderbook_depth_fetched=True), depth=0 means an empty book
    # and must be rejected.
    m = valid_market(orderbook_depth=0, orderbook_depth_fetched=True)
    result = filters.run([m], make_cfg(min_orderbook_depth_at_limit=100))
    assert result.passed == []
    assert "depth" in result.skip_reason_counts


def test_depth_check_rejects_when_not_yet_fetched():
    # A live orderbook fetch is a hard prerequisite for downstream analysis.
    # Markets that reach filters without enrichment must not pass silently.
    m = valid_market(orderbook_depth=0, orderbook_depth_fetched=False)
    result = filters.run([m], make_cfg(min_orderbook_depth_at_limit=100))
    assert result.passed == []
    assert "orderbook_unfetched" in result.skip_reason_counts


# ── Category filtering works end-to-end ─────────────────────────────────────

def test_category_allowlist_accepts_lowercase_short_form():
    m = valid_market(category="crypto")
    result = filters.run([m], make_cfg(category_allowlist=["crypto"]))
    assert len(result.passed) == 1


def test_category_allowlist_accepts_kalshi_capitalised_form():
    # Regression: Kalshi returns "Crypto" / "Climate and Weather" / etc.
    # The filter must normalise both sides so the default lowercase
    # allowlist matches.
    cases = [
        ("Crypto", "crypto"),
        ("Climate and Weather", "weather"),
        ("Economics", "economic"),
        ("Financials", "financial"),
        ("Commodities", "commodities"),
        ("Science and Technology", "science and technology"),
        ("Culture", "culture"),
    ]
    cfg = make_cfg(
        category_allowlist=[
            "crypto",
            "economic",
            "financial",
            "commodities",
            "weather",
            "science and technology",
            "culture",
        ]
    )
    for raw_category, _short in cases:
        m = valid_market(ticker=f"KX-{raw_category}", category=raw_category)
        result = filters.run([m], cfg)
        assert len(result.passed) == 1, (
            f"category {raw_category!r} should match allowlist but was rejected: "
            f"{result.skip_reason_counts}"
        )


def test_category_outside_allowlist_rejected():
    m = valid_market(category="Politics")
    cfg = make_cfg(category_allowlist=["crypto", "weather"])
    result = filters.run([m], cfg)
    assert result.passed == []
    assert "category" in result.skip_reason_counts


def test_category_allowlist_accepts_requested_user_facing_names():
    cases = [
        ("Financials", "finance"),
        ("Economics", "economics"),
        ("Climate and Weather", "climate"),
        ("Science and Technology", "tech & science"),
    ]
    for raw_category, allowed_category in cases:
        m = valid_market(ticker=f"KX-{allowed_category}", category=raw_category)
        result = filters.run([m], make_cfg(category_allowlist=[allowed_category]))
        assert len(result.passed) == 1


def test_empty_allowlist_accepts_all_categories():
    m = valid_market(category="anything")
    result = filters.run([m], make_cfg(category_allowlist=[]))
    assert len(result.passed) == 1


# ── Skip reason counts / examples are useful for diagnosing pass_rate=0 ─────

def test_skip_reason_counts_aggregates_by_reason_key():
    cfg = make_cfg()
    markets = [
        valid_market(ticker="A", status="closed"),
        valid_market(ticker="B", status="settled"),
        valid_market(ticker="C", volume_24h=10),
        valid_market(ticker="D", category="Politics"),
    ]
    result = filters.run(markets, cfg)

    counts = result.skip_reason_counts
    assert counts.get("status", 0) == 2
    assert counts.get("volume_24h", 0) == 1
    assert counts.get("category", 0) == 1
    assert sum(counts.values()) == 4


def test_skip_reason_examples_capped_per_reason():
    cfg = make_cfg()
    # 10 markets all failing for the same reason; example list should cap
    # at filters._MAX_EXAMPLES.
    markets = [valid_market(ticker=f"BAD-{i}", status="closed") for i in range(10)]
    result = filters.run(markets, cfg)

    examples = result.skip_reason_examples
    assert "status" in examples
    assert 1 <= len(examples["status"]) <= filters._MAX_EXAMPLES


def test_pass_rate_reflects_passed_over_total():
    cfg = make_cfg()
    markets = [
        valid_market(ticker="GOOD"),
        valid_market(ticker="BAD-1", status="closed"),
        valid_market(ticker="BAD-2", volume_24h=10),
    ]
    result = filters.run(markets, cfg)
    assert result.pass_rate == 1 / 3


def test_rejected_market_cannot_also_appear_in_passed_list():
    good = valid_market(ticker="GOOD")
    rejected = valid_market(ticker="BAD", volume_24h=10)

    result = filters.run([good, rejected], make_cfg())

    passed_tickers = {market.ticker for market in result.passed}
    rejected_tickers = {market.ticker for market, _ in result.rejected}
    assert passed_tickers == {"GOOD"}
    assert rejected_tickers == {"BAD"}
    assert passed_tickers.isdisjoint(rejected_tickers)


# ── Scanner enrichment sets the orderbook_depth_fetched flag ────────────────

def test_enrich_with_orderbook_depth_sets_fetched_flag(monkeypatch):
    """Successful enrichment must set orderbook_depth_fetched so the filter
    can distinguish 'we asked, got 0' from 'we never asked'."""
    import market_scanner

    market = valid_market(orderbook_depth=0, orderbook_depth_fetched=False)

    class FakeClient:
        def get_orderbook(self, ticker, depth=10):
            return {"orderbook": {"yes": {"ask": [[45, 250]], "bid": [[42, 250]]}}}

    market_scanner.enrich_with_orderbook_depth(
        [market], FakeClient(), depth=10, delay_seconds=0.0
    )

    assert market.orderbook_depth_fetched is True
    assert market.orderbook_depth == 250
    assert market.fetched_at is not None


def test_enrich_with_orderbook_depth_parses_current_orderbook_fp_shape(monkeypatch):
    """Kalshi REST orderbooks currently return bid-only fixed-point arrays.
    The best bid is the last level; NO bid depth is executable YES-ask depth."""
    import market_scanner

    market = valid_market(orderbook_depth=0, orderbook_depth_fetched=False)

    class FixedPointBookClient:
        def get_orderbook(self, ticker, depth=10):
            return {
                "orderbook_fp": {
                    "yes_dollars": [["0.0100", "200.00"], ["0.4200", "13.00"]],
                    "no_dollars": [["0.0100", "100.00"], ["0.5600", "117.00"]],
                }
            }

    market_scanner.enrich_with_orderbook_depth(
        [market], FixedPointBookClient(), depth=10, delay_seconds=0.0
    )

    assert market.orderbook_depth_fetched is True
    assert market.orderbook_depth == 117


def test_enrich_with_orderbook_depth_supports_legacy_bid_only_orderbook(monkeypatch):
    import market_scanner

    market = valid_market(orderbook_depth=0, orderbook_depth_fetched=False)

    class LegacyBidOnlyClient:
        def get_orderbook(self, ticker, depth=10):
            return {"orderbook": {"yes": [[1, 200], [42, 13]], "no": [[1, 100], [56, 117]]}}

    market_scanner.enrich_with_orderbook_depth(
        [market], LegacyBidOnlyClient(), depth=10, delay_seconds=0.0
    )

    assert market.orderbook_depth_fetched is True
    assert market.orderbook_depth == 117


def test_enrich_with_orderbook_depth_marks_empty_book_as_fetched(monkeypatch):
    """An empty asks list still counts as a successful fetch; the depth
    filter must then reject the market rather than pass it silently."""
    import market_scanner

    market = valid_market(orderbook_depth=10, orderbook_depth_fetched=False)

    class EmptyBookClient:
        def get_orderbook(self, ticker, depth=10):
            return {"orderbook": {"yes": {"ask": [], "bid": []}}}

    market_scanner.enrich_with_orderbook_depth(
        [market], EmptyBookClient(), depth=10, delay_seconds=0.0
    )

    assert market.orderbook_depth == 0
    assert market.orderbook_depth_fetched is True

    result = filters.run([market], make_cfg(min_orderbook_depth_at_limit=100))
    assert result.passed == []
    assert "depth" in result.skip_reason_counts


def test_enrich_with_orderbook_depth_does_not_mark_unrecognized_payload_as_fetched(monkeypatch):
    import market_scanner

    market = valid_market(orderbook_depth=10, orderbook_depth_fetched=False)

    class BadShapeClient:
        def get_orderbook(self, ticker, depth=10):
            return {"unexpected": {"yes_dollars": [["0.4200", "13.00"]]}}

    market_scanner.enrich_with_orderbook_depth(
        [market], BadShapeClient(), depth=10, delay_seconds=0.0
    )

    assert market.orderbook_depth == 10
    assert market.orderbook_depth_fetched is False
    assert market.is_unsafe is True
    assert market.unsafe_reason.startswith("orderbook_fetch_failed:")


def test_enrich_with_orderbook_depth_does_not_mark_fetch_exception_as_fetched(monkeypatch):
    import market_scanner

    market = valid_market(orderbook_depth=10, orderbook_depth_fetched=False)

    class FailingClient:
        def get_orderbook(self, ticker, depth=10):
            raise RuntimeError("network down")

    market_scanner.enrich_with_orderbook_depth(
        [market], FailingClient(), depth=10, delay_seconds=0.0
    )

    assert market.orderbook_depth == 10
    assert market.orderbook_depth_fetched is False
    assert market.is_unsafe is True
    assert market.unsafe_reason.startswith("orderbook_fetch_failed:")


# ── Dedup pass keeps the best market per event group ────────────────────────

def test_dedup_keeps_highest_volume_24h_per_event_group():
    cfg = make_cfg()
    a = valid_market(ticker="EVT-A", event_ticker="EVT", volume_24h=1000)
    b = valid_market(ticker="EVT-B", event_ticker="EVT", volume_24h=5000)
    c = valid_market(ticker="EVT-C", event_ticker="EVT", volume_24h=3000)
    result = filters.run([a, b, c], cfg)

    assert [m.ticker for m in result.passed] == ["EVT-B"]
    dedup_rejected = [r for _, r in result.rejected if "duplicate_event_group" in r]
    assert len(dedup_rejected) == 2
