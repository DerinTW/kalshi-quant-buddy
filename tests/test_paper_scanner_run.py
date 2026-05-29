from __future__ import annotations

import json
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

    def get_all_markets(
        self,
        status: str = "open",
        category=None,
        max_markets=None,
    ) -> list[dict]:
        assert status == "open"
        markets = list(self.raw_markets)
        return markets[:max_markets] if max_markets is not None else markets

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
    def get_markets(
        self,
        status: str = "open",
        limit: int = 25,
        category=None,
        mve_filter=None,
        cursor=None,
    ) -> dict:
        assert status == "open"
        return {"markets": list(self.raw_markets), "cursor": ""}


class FakeCategoryPagingClient(FakeKalshiDataClient):
    def __init__(self, raw_markets: list[dict]):
        super().__init__(raw_markets)
        self.market_calls: list[dict] = []

    def get_series_list(self) -> list[dict]:
        return [{"ticker": "KXCRYPTO", "category": "Crypto"}]

    def get_markets(self, **kwargs) -> dict:
        self.market_calls.append(dict(kwargs))
        assert kwargs["limit"] <= 200
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


def agent_records(tmp_path: Path) -> list[dict]:
    path = tmp_path / "logs" / "agent.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
    assert summary["filter_config"]["min_volume_24h"] == c.min_volume_24h
    assert summary["min_volume_24h"] == c.min_volume_24h
    assert summary["filter_config"]["min_liquidity"] == c.min_liquidity_dollars
    assert summary["filter_config"]["max_spread_cents"] == c.max_spread_cents
    assert summary["filter_config"]["min_yes_price"] == c.min_yes_price
    assert summary["filter_config"]["max_minutes_to_expiry"] == c.max_minutes_to_expiry
    assert (
        summary["filter_config"]["min_orderbook_depth_at_limit"]
        == c.min_orderbook_depth_at_limit
    )
    summaries = [r for r in agent_records(tmp_path) if r.get("stage") == "scan_summary"]
    assert summaries[-1]["status"] == "success"
    assert summaries[-1]["fetched_count"] == 1
    assert summaries[-1]["normalized_count"] == 1
    records = agent_records(tmp_path)
    assert not [
        r for r in records
        if r.get("stage") in {"fetched", "normalized"} and r.get("outcome") == "passed"
    ]


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


def test_category_fetch_pages_with_api_safe_limit(tmp_path, patched_pipeline):
    c = cfg(tmp_path)
    client = FakeCategoryPagingClient([raw_market_new_schema()])

    summary = scanner.run_scan(
        cfg=c,
        client=client,
        limit=4000,
        execute_paper=True,
        category="crypto",
    )

    assert client.market_calls
    assert all(call["limit"] <= 200 for call in client.market_calls)
    assert summary["raw_markets"] == 1
    assert summary["passed_count"] == 1


def test_paper_scanner_summary_pass_rate_and_examples_are_filter_based(
    tmp_path,
    patched_pipeline,
):
    bad = raw_market("KXBAD-26MAY15")
    bad["yes_ask"] = 0
    bad["yes_bid"] = 0
    bad["no_ask"] = 0
    bad["no_bid"] = 0

    summary = scanner.run_scan(
        cfg=cfg(tmp_path),
        client=FakeKalshiDataClient([raw_market("KXGOOD-26MAY15"), bad]),
        limit=2,
    )

    assert summary["normalized_markets"] == 2
    assert summary["passed_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["top_bottleneck_filter"]["reason"] == "unsafe"
    assert summary["passed_examples"][0]["ticker"] == "KXGOOD-26MAY15"

    records = agent_records(tmp_path)
    filter_passed = {
        r["ticker"] for r in records if r.get("stage") == "filter_passed"
    }
    filter_skipped = {
        r["ticker"] for r in records if r.get("stage") == "filter_skipped"
    }
    assert filter_passed.isdisjoint(filter_skipped)


def test_normalization_exception_is_logged_and_scan_continues(
    tmp_path,
    monkeypatch,
    patched_pipeline,
):
    c = cfg(tmp_path)
    original_normalize = scanner.market_scanner.normalize

    def flaky_normalize(raw):
        if raw.get("ticker") == "KXBAD-26MAY15":
            raise ValueError("boom while normalizing")
        return original_normalize(raw)

    monkeypatch.setattr(scanner.market_scanner, "normalize", flaky_normalize)

    summary = scanner.run_scan(
        cfg=c,
        client=FakeKalshiDataClient(
            [raw_market("KXBAD-26MAY15"), raw_market("KXGOOD-26MAY15")]
        ),
        limit=2,
    )

    assert summary["raw_markets"] == 2
    assert summary["normalized_markets"] == 2
    assert summary["normalization_error_count"] == 1
    records = agent_records(tmp_path)
    errors = [r for r in records if r.get("stage") == "normalization_error"]
    assert len(errors) == 1
    assert errors[0]["ticker"] == "KXBAD-26MAY15"
    assert "boom while normalizing" in errors[0]["err"]
    assert "Traceback" in errors[0]["traceback"]
    assert [r for r in records if r.get("stage") == "normalized" and r.get("ticker") == "KXGOOD-26MAY15"]
    assert [r for r in records if r.get("stage") == "scan_summary"][-1]["status"] == "partial"


def test_every_fetched_market_gets_one_prefilter_or_normalization_followup(
    tmp_path,
    patched_pipeline,
):
    missing = raw_market("")
    missing.pop("ticker")
    sports = raw_market("KXATP-26MAY15")

    scanner.run_scan(
        cfg=cfg(tmp_path),
        client=FakeKalshiDataClient(
            [raw_market("KXGOOD-26MAY15"), missing, sports]
        ),
        limit=3,
    )

    records = agent_records(tmp_path)
    fetched = [r for r in records if r.get("stage") == "fetched"]
    followups = [
        r for r in records
        if r.get("stage") in {"prefilter_skipped", "normalized", "normalization_error"}
    ]
    assert len(fetched) == 3
    assert len(followups) == 3
    reasons = {r.get("skip_reason") for r in followups if r.get("stage") == "prefilter_skipped"}
    assert "prefilter_missing_ticker" in reasons
    assert any(str(reason).startswith("prefilter_blocked_event_prefix=") for reason in reasons)


def test_pipeline_integrity_assertion_logs_and_raises(tmp_path):
    logger_dir = tmp_path / "logs"
    import logger as logger_mod

    logger_mod.init(str(logger_dir))

    with pytest.raises(scanner.PipelineIntegrityError):
        scanner._assert_pipeline_integrity(
            scan_run_id="scan-x",
            fetched_count=3,
            prefilter_skipped_count=1,
            normalized_count=1,
        )

    records = agent_records(tmp_path)
    assert records[-2]["stage"] == "pipeline_integrity_error"
    assert records[-2]["mismatch"] == 1


def test_failed_scan_writes_failed_summary(tmp_path, monkeypatch, patched_pipeline):
    monkeypatch.setattr(scanner, "_normalize_markets", lambda raw, scan_run_id="": [])

    with pytest.raises(scanner.PipelineIntegrityError):
        scanner.run_scan(
            cfg=cfg(tmp_path),
            client=FakeKalshiDataClient([raw_market("KXGOOD-26MAY15")]),
            limit=1,
        )

    summaries = [r for r in agent_records(tmp_path) if r.get("stage") == "scan_summary"]
    assert summaries[-1]["status"] == "failed"
    assert summaries[-1]["fetched_count"] == 1
    assert "count mismatch" in summaries[-1]["error_message"]


def test_startup_warns_when_previous_scan_lacked_summary(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "agent.jsonl").write_text(
        json.dumps(
            {
                "stage": "scan_started",
                "event": "scan_started",
                "scan_run_id": "scan-old",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    import logger as logger_mod

    logger_mod.init(str(log_dir))
    scanner._warn_on_previous_incomplete_scan(str(log_dir))

    records = agent_records(tmp_path)
    assert records[-1]["event"] == "previous_scan_missing_summary"
    assert records[-1]["scan_run_id"] == "scan-old"


def test_startup_warns_when_previous_scan_failed(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "agent.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "stage": "scan_started",
                        "event": "scan_started",
                        "scan_run_id": "scan-old",
                    }
                ),
                json.dumps(
                    {
                        "stage": "scan_summary",
                        "event": "scan_summary",
                        "scan_run_id": "scan-old",
                        "status": "failed",
                        "error_message": "boom",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    import logger as logger_mod

    logger_mod.init(str(log_dir))
    scanner._warn_on_previous_incomplete_scan(str(log_dir))

    records = agent_records(tmp_path)
    assert records[-1]["event"] == "previous_scan_failed"
    assert records[-1]["scan_run_id"] == "scan-old"
