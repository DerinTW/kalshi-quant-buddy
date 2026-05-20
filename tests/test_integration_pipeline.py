from __future__ import annotations

from datetime import datetime, timedelta, timezone

import decision_formatter
import edge
import filters
import position_sizing
import prediction_model
import risk_manager
import sentiment
import weird_move
from config import Config
from models import Market, ResearchItem, ResearchResult


def _cfg() -> Config:
    cfg = Config()
    cfg.kill_switch = True
    cfg.trading_mode = "paper"
    cfg.live_trading_enabled = False
    cfg.min_liquidity_dollars = 500
    cfg.min_volume_24h = 500
    cfg.max_spread_cents = 6
    cfg.max_spread_pct = 20
    cfg.min_minutes_to_expiry = 20
    cfg.max_minutes_to_expiry = 4320
    cfg.min_yes_price = 15
    cfg.max_yes_price = 85
    cfg.category_allowlist = ["crypto", "economic", "financial", "weather"]
    cfg.max_trade_dollars = 10
    cfg.paper_bankroll = 1000
    cfg.max_position_pct_of_bankroll = 0.5
    cfg.max_daily_loss_dollars = 20
    cfg.max_trades_per_day = 5
    cfg.max_category_exposure_dollars = 25
    cfg.max_correlated_exposure_dollars = 15
    cfg.min_edge_pct = 7
    cfg.min_adjusted_edge_pct = 5
    cfg.min_confidence = 0.65
    cfg.min_confidence_adjusted_edge_cents = 4
    cfg.slippage_cents = 0
    cfg.fee_pct = 0.0
    return cfg


def _market() -> Market:
    now = datetime.now(timezone.utc)
    return Market(
        ticker="TEST-BTC-ABOVE-100K",
        title="Will Bitcoin be above $100,000 today?",
        status="open",
        yes_ask=45,
        yes_bid=42,
        no_ask=58,
        no_bid=55,
        volume=5000,
        volume_24h=2500,
        open_interest=3000,
        close_time=now + timedelta(hours=2),
        settlement_time=now + timedelta(hours=2, minutes=5),
        category="crypto",
        rules_primary="Resolves YES if BTC is above $100,000 at the specified time.",
        spread_pct=6.9,
        minutes_to_close=120,
        minutes_to_settlement=125,
        liquidity_dollars=1500,
        event_ticker="TEST-BTC",
        orderbook_depth=500,
    )


def test_full_pipeline_blocks_trade_when_kill_switch_enabled():
    cfg = _cfg()
    market = _market()

    # 1. Filter should accept this structurally decent fake market.
    filter_result = filters.run([market], cfg)
    assert len(filter_result.passed) == 1

    # 2. Weird move detector should return a signal object without crashing.
    weird_signals = weird_move.batch_detect([market], [market])
    weird_signal = weird_signals[market.ticker]
    assert weird_signal.ticker == market.ticker

    # 3. Research + sentiment.
    research = ResearchResult(
        ticker=market.ticker,
        query=market.title,
        items=[
            ResearchItem(
                source="Test Source",
                url="",
                published_at=datetime.now(timezone.utc),
                claim="BTC is trading above the relevant threshold in this test fixture.",
                direction="supports_yes",
                relevance=0.95,
                credibility=0.90,
                recency_score=1.0,
                summary="Synthetic test evidence.",
                agent="test",
            )
        ],
    )

    sentiment_result = sentiment.analyze(market, research)
    assert sentiment_result.ticker == market.ticker

    # 4. Probability estimate should return a valid probability object.
    estimate = prediction_model.estimate(
        market=market,
        research=research,
        sentiment=sentiment_result,
        weird_move=weird_signal,
        cfg=cfg,
    )
    assert 0.0 < estimate.yes_probability < 1.0

    # 5. Edge calculation may or may not find edge, but should not crash.
    edge_result = edge.calculate(market, estimate, cfg)

    if edge_result is None:
        decision = decision_formatter.format_decision(
            market=market,
            estimate=estimate,
            edge=None,
            sizing=None,
            risk_assessment=None,
            cfg=cfg,
        )
        assert decision["action"] == "NO_TRADE"
        return

    # 6. Position sizing.
    size = position_sizing.compute(market, edge_result, estimate, cfg)

    # 7. Risk manager MUST reject because kill switch is enabled.
    risk = risk_manager.assess(
        decision=size,
        market=market,
        edge=edge_result,
        open_positions=[],
        daily_pnl=0.0,
        trades_today=0,
        bankroll=cfg.paper_bankroll,
        cfg=cfg,
    )
    assert risk.approved is False

    # 8. Final formatter must produce NO_TRADE because risk rejected.
    decision = decision_formatter.format_decision(
        market=market,
        estimate=estimate,
        edge=edge_result,
        sizing=size,
        risk_assessment=risk,
        cfg=cfg,
    )
    assert decision["action"] == "NO_TRADE"