from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import db
from config import Config
from models import (
    EdgeResult,
    PositionSize,
    ProbabilityEstimate,
    ResearchResult,
    RiskAssessment,
    SentimentResult,
)
from scripts import paper_scanner_run as scanner


class FakeKalshiDataClient:
    def __init__(self, raw_markets: list[dict]):
        self.raw_markets = raw_markets
        self.place_order_called = False

    def get_all_markets(self, status: str = "open") -> list[dict]:
        assert status == "open"
        return list(self.raw_markets)

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        return {
            "orderbook": {
                "yes": {
                    "ask": [[45, 500]],
                    "bid": [[42, 500]],
                }
            }
        }

    def get_market_history(self, ticker: str, limit: int = 50) -> dict:
        return {"history": []}

    def place_order(self, **kwargs):
        self.place_order_called = True
        raise AssertionError("place_order must not be called by paper scanner")


class FakeEnvelopeKalshiDataClient(FakeKalshiDataClient):
    def get_markets(self, status: str = "open", limit: int = 25, category=None) -> dict:
        assert status == "open"
        return {"markets": list(self.raw_markets), "cursor": ""}


def cfg(tmp_path: Path, **overrides) -> Config:
    base = dict(
        kalshi_api_key="x",
        kalshi_private_key_path="",
        anthropic_api_key="x",
        kill_switch=False,
        trading_mode="paper",
        paper_only=True,
        live_trading_enabled=False,
        allow_live_orders=False,
        max_trade_dollars=10.0,
        paper_bankroll=1000.0,
        max_position_pct_of_bankroll=1.0,
        min_liquidity_dollars=100.0,
        min_volume_24h=100,
        max_spread_cents=6,
        max_spread_pct=20,
        min_yes_price=10,
        max_yes_price=90,
        min_edge_pct=7.0,
        min_adjusted_edge_pct=5.0,
        min_confidence=0.65,
        min_confidence_adjusted_edge_cents=4.0,
        slippage_cents=0,
        fee_pct=0.0,
        log_dir=str(tmp_path / "logs"),
        db_path=str(tmp_path / "paper_scan.sqlite"),
    )
    base.update(overrides)
    return Config(**base)


def raw_market(ticker: str = "KXTEST-26MAY15") -> dict:
    now = datetime.now(timezone.utc)
    close = now + timedelta(hours=2)
    return {
        "ticker": ticker,
        "title": "Will the test market resolve yes?",
        "status": "open",
        "yes_ask": 45,
        "yes_bid": 42,
        "no_ask": 57,
        "no_bid": 55,
        "volume": 5000,
        "volume_24h": 2500,
        "open_interest": 5000,
        "close_time": close.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settlement_time": close.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "category": "crypto",
        "rules_primary": "Test rules.",
    }


def raw_market_new_schema(ticker: str = "KXTEST-26MAY15") -> dict:
    raw = raw_market(ticker)
    return {
        "ticker": raw["ticker"],
        "event_ticker": "KXTEST",
        "title": raw["title"],
        "status": "active",
        "yes_ask_dollars": "0.4500",
        "yes_bid_dollars": "0.4200",
        "no_ask_dollars": "0.5700",
        "no_bid_dollars": "0.5500",
        "volume_fp": "5000.00",
        "volume_24h_fp": "2500.00",
        "open_interest_fp": "5000.00",
        "close_time": raw["close_time"],
        "settlement_time": raw["settlement_time"],
        "category": raw["category"],
        "rules_primary": raw["rules_primary"],
    }


def fake_estimate() -> ProbabilityEstimate:
    return ProbabilityEstimate(
        ticker="KXTEST-26MAY15",
        yes_probability=0.70,
        confidence="high",
        reasoning="Synthetic approved paper-scan thesis.",
    )


def fake_edge() -> EdgeResult:
    return EdgeResult(
        ticker="KXTEST-26MAY15",
        side="YES",
        entry_price_cents=45,
        implied_yes_prob=0.45,
        estimated_yes_prob=0.70,
        raw_edge_pct=25.0,
        adjusted_edge_pct=20.0,
        expected_value=0.25,
        adjusted_ev=0.20,
        confidence="high",
        confidence_adjusted_ev=0.18,
        confidence_adjusted_edge_pct=18.0,
        spread_cents=3,
    )


def fake_sizing() -> PositionSize:
    return PositionSize(
        ticker="KXTEST-26MAY15",
        side="YES",
        dollars=4.50,
        contracts=10,
        entry_price_cents=45,
        max_loss_dollars=4.50,
    )


def fake_risk(approved: bool = True) -> RiskAssessment:
    return RiskAssessment(
        ticker="KXTEST-26MAY15",
        side="YES",
        approved=approved,
        reason="ok" if approved else "rejected",
        mode="paper",
        checks_passed=["test"] if approved else [],
        checks_failed=[] if approved else ["test_rejected"],
    )


@pytest.fixture
def patched_pipeline(monkeypatch):
    monkeypatch.setattr(
        scanner.category_research,
        "research_market_categorical",
        lambda market, cfg: [],
    )
    monkeypatch.setattr(
        scanner.research_agents,
        "research_market",
        lambda market, cfg: ResearchResult(ticker=market.ticker, query=market.title),
    )
    monkeypatch.setattr(
        scanner.sentiment,
        "analyze",
        lambda market, research: SentimentResult(
            ticker=market.ticker,
            sentiment_score=0.5,
            narrative_direction="supports_yes",
            confidence=0.9,
            market_impact_estimate_cents=8,
            major_contradictions=[],
            item_count=1,
            contributing_sources=["test"],
            source_credibility=0.9,
            event_relevance=0.9,
            rumor_risk="low",
        ),
    )
    monkeypatch.setattr(
        scanner.prediction_model,
        "estimate",
        lambda **kwargs: fake_estimate(),
    )
    monkeypatch.setattr(scanner.edge, "calculate", lambda market, estimate, cfg: fake_edge())
    monkeypatch.setattr(
        scanner.position_sizing,
        "compute",
        lambda market, edge, estimate, cfg: fake_sizing(),
    )
    monkeypatch.setattr(
        scanner.risk_manager,
        "assess",
        lambda *args, **kwargs: fake_risk(True),
    )
    monkeypatch.setattr(scanner.weird_move, "batch_detect", lambda markets, all_markets: {})


def test_paper_scanner_refuses_when_live_trading_enabled(tmp_path):
    with pytest.raises(RuntimeError, match="LIVE_TRADING_ENABLED=true"):
        scanner.run_scan(
            cfg=cfg(tmp_path, live_trading_enabled=True),
            client=FakeKalshiDataClient([raw_market()]),
        )


def test_paper_scanner_refuses_when_allow_live_orders_enabled(tmp_path):
    with pytest.raises(RuntimeError, match="ALLOW_LIVE_ORDERS=true"):
        scanner.run_scan(
            cfg=cfg(tmp_path, allow_live_orders=True),
            client=FakeKalshiDataClient([raw_market()]),
        )


def test_paper_scanner_refuses_when_paper_only_false(tmp_path):
    with pytest.raises(RuntimeError, match="PAPER_ONLY=false"):
        scanner.run_scan(
            cfg=cfg(tmp_path, paper_only=False),
            client=FakeKalshiDataClient([raw_market()]),
        )


def test_paper_scanner_never_calls_client_place_order(tmp_path, patched_pipeline):
    c = cfg(tmp_path)
    client = FakeKalshiDataClient([raw_market()])

    summary = scanner.run_scan(
        cfg=c,
        client=client,
        limit=1,
        execute_paper=True,
    )

    assert client.place_order_called is False
    assert summary["paper_trades_inserted"] == 1
    assert summary["errors"] == []


def test_default_run_does_not_insert_paper_trades(tmp_path, patched_pipeline):
    c = cfg(tmp_path)

    summary = scanner.run_scan(
        cfg=c,
        client=FakeKalshiDataClient([raw_market()]),
        limit=1,
    )

    assert summary["paper_trades_inserted"] == 0
    assert summary["decisions"][0]["action"] == "BUY_YES"
    assert summary["decisions"][0]["execution_skip_reason"] == "execute_paper_not_requested"
    assert db.get_open_trades() == []


def test_execute_paper_inserts_only_paper_trades_when_risk_approved(
    tmp_path,
    patched_pipeline,
):
    c = cfg(tmp_path)

    summary = scanner.run_scan(
        cfg=c,
        client=FakeKalshiDataClient([raw_market()]),
        limit=1,
        execute_paper=True,
    )

    assert summary["paper_trades_inserted"] == 1
    trades = db.get_open_trades()
    assert len(trades) == 1
    assert trades[0].mode == "paper"
    assert trades[0].ticker == "KXTEST-26MAY15"


def test_paper_scanner_unwraps_kalshi_market_envelope_and_new_schema(
    tmp_path,
    patched_pipeline,
):
    c = cfg(tmp_path)

    summary = scanner.run_scan(
        cfg=c,
        client=FakeEnvelopeKalshiDataClient([raw_market_new_schema()]),
        limit=1,
        execute_paper=True,
    )

    assert summary["raw_markets"] == 1
    assert summary["normalized_markets"] == 1
    assert summary["passed_count"] == 1
    assert summary["paper_trades_inserted"] == 1
