"""
Step 21 — safety/reality-check coverage.

Three independent guarantees are pinned down here:

  1. ``pipeline.compute_open_exposures`` deterministically aggregates open
     dollars-at-risk and fails safe when a peer's category is unknown.
  2. ``pipeline.run_once`` feeds those exposures into ``risk_manager.assess``
     and the run rejects (no execute) when either cap is already exceeded.
  3. ``python main.py --monitor-only`` runs at most one monitor pass and
     never enters a loop or places a live order.

These tests do not touch real APIs, real Kalshi, or a real LLM.
"""
from __future__ import annotations
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import filters as filters_mod
import pipeline
from config import Config
from models import (
    EdgeResult,
    Market,
    PositionSize,
    ProbabilityEstimate,
    ResearchResult,
    RiskAssessment,
    SentimentResult,
    TradeRecord,
    WeirdMoveSignal,
)


# ── Fixtures (mirroring tests/test_pipeline.py for consistency) ──────────────

@pytest.fixture
def cfg() -> Config:
    return Config(
        kalshi_api_key="x", anthropic_api_key="x",
        kalshi_private_key_path="",
    )


def _market(
    ticker: str = "KXTEST",
    *,
    event_ticker: str | None = None,
    category: str = "crypto",
) -> Market:
    now = datetime.now(timezone.utc)
    m = Market(
        ticker=ticker, title="t", status="open",
        yes_ask=55, yes_bid=53, no_ask=47, no_bid=45,
        volume=2000, volume_24h=12_000, open_interest=4_000,
        close_time=now + timedelta(minutes=120),
        settlement_time=now + timedelta(minutes=120),
        category=category, rules_primary="rules",
        event_ticker=event_ticker if event_ticker is not None else ticker,
    )
    m.minutes_to_close = 120.0
    m.minutes_to_settlement = 120.0
    m.liquidity_dollars = 5_000.0
    m.orderbook_depth = 200
    return m


def _trade(
    ticker: str = "KXTEST",
    *,
    dollars: float = 5.0,
    side: str = "YES",
    trade_id: str = "t",
) -> TradeRecord:
    return TradeRecord(
        id=trade_id, ticker=ticker, side=side, contracts=10,
        entry_price_cents=50, dollars_at_risk=dollars,
        mode="paper", thesis="thesis", estimated_yes_prob=0.6,
        result="open",
    )


# ── 1. compute_open_exposures — pure helper coverage ─────────────────────────

def test_compute_exposures_zero_when_no_open_trades():
    cat, corr = pipeline.compute_open_exposures(_market(), open_positions=[])
    assert cat == 0.0
    assert corr == 0.0


def test_compute_exposures_counts_same_event_as_correlated():
    """Open trade in the same event group → correlated_exposure increases."""
    candidate = _market("KXBTCD-26MAY31-B50000", event_ticker="KXBTCD-26MAY31")
    open_trades = [
        _trade("KXBTCD-26MAY31-B55000", dollars=4.0, trade_id="a"),
        _trade("KXBTCD-26MAY31-B60000", dollars=3.0, trade_id="b"),
    ]
    by_ticker = {
        t.ticker: _market(t.ticker, event_ticker="KXBTCD-26MAY31",
                          category=candidate.category)
        for t in open_trades
    }
    _, corr = pipeline.compute_open_exposures(candidate, open_trades, by_ticker)
    assert corr == pytest.approx(7.0)


def test_compute_exposures_unrelated_event_not_counted():
    candidate = _market("KXBTCD-26MAY31-B50000", event_ticker="KXBTCD-26MAY31")
    open_trades = [_trade("KXETHD-26MAY31-B3000", dollars=4.0)]
    by_ticker = {
        "KXETHD-26MAY31-B3000": _market("KXETHD-26MAY31-B3000",
                                        event_ticker="KXETHD-26MAY31",
                                        category="crypto"),
    }
    _, corr = pipeline.compute_open_exposures(candidate, open_trades, by_ticker)
    assert corr == 0.0


def test_compute_exposures_category_matched_when_peer_known():
    candidate = _market("KXBTC", category="crypto")
    open_trades = [
        _trade("KXETH", dollars=4.0, trade_id="a"),
        _trade("KXGDP", dollars=6.0, trade_id="b"),
    ]
    by_ticker = {
        "KXETH": _market("KXETH", category="crypto"),
        "KXGDP": _market("KXGDP", category="economic"),
    }
    cat, _ = pipeline.compute_open_exposures(candidate, open_trades, by_ticker)
    assert cat == pytest.approx(4.0), "only same-category peer is counted"


def test_compute_exposures_unknown_peer_category_fails_safe():
    """Open trade whose market is no longer in the current scan → counted
    toward category exposure (fail-safe / conservative)."""
    candidate = _market("KXBTC", category="crypto")
    open_trades = [_trade("KXOLD", dollars=9.0)]
    cat, _ = pipeline.compute_open_exposures(candidate, open_trades, markets_by_ticker={})
    assert cat == pytest.approx(9.0)


def test_compute_exposures_ignores_zero_and_negative_dollars():
    candidate = _market("KXBTC", category="crypto")
    open_trades = [
        _trade("KXBTC", dollars=0.0, trade_id="z"),
        _trade("KXBTC", dollars=-3.0, trade_id="n"),
        _trade("KXBTC", dollars=2.5, trade_id="ok"),
    ]
    by_ticker = {"KXBTC": _market("KXBTC", category="crypto")}
    cat, corr = pipeline.compute_open_exposures(candidate, open_trades, by_ticker)
    assert cat == pytest.approx(2.5)
    assert corr == pytest.approx(2.5)


# ── 2. pipeline.run_once wires real exposure into risk_manager.assess ────────

def _research() -> ResearchResult:
    return ResearchResult(ticker="KXTEST", query="q", items=[])


def _sentiment() -> SentimentResult:
    return SentimentResult(
        ticker="KXTEST", sentiment_score=0.0,
        narrative_direction="neutral", confidence=0.5,
        market_impact_estimate_cents=0, major_contradictions=[],
        item_count=0, contributing_sources=[],
        source_credibility=0.6, event_relevance=0.6, rumor_risk="low",
    )


def _estimate() -> ProbabilityEstimate:
    return ProbabilityEstimate(
        ticker="KXTEST", yes_probability=0.62,
        confidence="medium-high", reasoning="thesis",
    )


def _edge() -> EdgeResult:
    return EdgeResult(
        ticker="KXTEST", side="YES", entry_price_cents=55,
        implied_yes_prob=0.55, estimated_yes_prob=0.62,
        raw_edge_pct=10.0, adjusted_edge_pct=8.0,
        expected_value=0.10, adjusted_ev=0.08,
        confidence="medium-high",
        confidence_adjusted_ev=0.064,
        confidence_adjusted_edge_pct=6.4,
        spread_cents=2,
    )


def _sizing() -> PositionSize:
    return PositionSize(
        ticker="KXTEST", side="YES",
        dollars=2.75, contracts=5,
        entry_price_cents=55, max_loss_dollars=2.75,
    )


def _weird() -> WeirdMoveSignal:
    return WeirdMoveSignal(
        ticker="KXTEST", flagged=False, classification="none",
        price_change_5m=0.0, price_change_15m=0.0,
        volume_ratio=1.0, spread_change=1.0,
        related_disagreement=0.0, triggers=[],
        description="", confidence="low",
    )


class _RiskSpy:
    """Captures the category_exposure / correlated_exposure passed to risk_assess."""
    def __init__(self, approve: bool, reason: str = ""):
        self.calls: list[dict] = []
        self.approve = approve
        self.reason = reason

    def __call__(self, sizing, market, edge, open_positions, daily_pnl,
                 trades_today, bankroll, cfg, **kw):
        self.calls.append({
            "ticker": sizing.ticker,
            "category_exposure": kw.get("category_exposure"),
            "correlated_exposure": kw.get("correlated_exposure"),
        })
        return RiskAssessment(
            ticker=sizing.ticker, side=sizing.side,
            approved=self.approve, reason=self.reason or "ok",
            mode=cfg.trading_mode,
            checks_passed=["test"],
            checks_failed=[] if self.approve else [self.reason or "rejected"],
        )


class _ExecRecorder:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, sizing, estimate, edge, cfg, *, client=None,
                 risk_approved=False, **kw):
        self.calls.append({"ticker": sizing.ticker, "risk_approved": risk_approved})
        return None


def _deps_with_spy(*, markets, open_positions, risk_spy, executor):
    return pipeline.PipelineDeps(
        scan=lambda c, cfg: list(markets),
        filter_run=lambda ms, cfg: filters_mod.FilterResult(
            passed=list(ms), rejected=[],
        ),
        weird_move_detect=lambda m, all_m: _weird(),
        research=lambda m, cfg: _research(),
        sentiment_analyze=lambda m, r: _sentiment(),
        estimate=lambda m, r, s, w, cfg, **kw: _estimate(),
        edge_calc=lambda m, est, cfg: _edge(),
        sizing=lambda m, e, est, cfg: _sizing(),
        risk_assess=risk_spy,
        format_decision=lambda *a, **kw: __import__("decision_formatter").format_decision(*a, **kw),
        execute=executor,
        open_trades=lambda: list(open_positions),
    )


def test_run_once_passes_real_exposures_to_risk(cfg):
    """Regression: the values must not be the old hard-coded 0.0/0.0."""
    candidate = _market("KXTEST", category="crypto")
    open_positions = [
        _trade("KXTEST", dollars=4.0, side="YES", trade_id="open-a"),
    ]
    spy = _RiskSpy(approve=False, reason="duplicate_position")  # rejection irrelevant here
    executor = _ExecRecorder()
    deps = _deps_with_spy(
        markets=[candidate], open_positions=open_positions,
        risk_spy=spy, executor=executor,
    )
    pipeline.run_once(cfg, client=object(), deps=deps,
                     markets_override=[candidate])
    assert len(spy.calls) == 1
    call = spy.calls[0]
    # Same event_ticker → correlated should reflect the 4.0 open trade.
    assert call["correlated_exposure"] == pytest.approx(4.0)
    # Same category (crypto) and peer in current scan → category counted too.
    assert call["category_exposure"] == pytest.approx(4.0)


def test_run_once_rejects_when_category_cap_already_exceeded(cfg):
    """If category exposure is already above MAX_CATEGORY_EXPOSURE_DOLLARS,
    risk_manager rejects and no execution is attempted."""
    # The kill switch short-circuits risk_manager before any exposure check.
    # We disable it for this assertion only; the safety default itself is not
    # touched in source/.env.
    cfg.kill_switch = False
    cfg.max_category_exposure_dollars = 5.0
    cfg.max_correlated_exposure_dollars = 999.0  # not the binding cap

    candidate = _market("KXNEW", category="crypto",
                        event_ticker="KXNEW-EVT")
    # One old trade in same category, different event group → category-only.
    old_peer = _market("KXBTC", category="crypto",
                       event_ticker="KXBTC-EVT")
    open_positions = [_trade("KXBTC", dollars=20.0, trade_id="big")]

    # Use the REAL risk_manager so the assertion is end-to-end.
    import risk_manager
    executor = _ExecRecorder()
    deps = pipeline.PipelineDeps(
        scan=lambda c, cfg: [candidate, old_peer],
        filter_run=lambda ms, cfg: filters_mod.FilterResult(
            passed=[candidate], rejected=[],
        ),
        weird_move_detect=lambda m, all_m: _weird(),
        research=lambda m, cfg: _research(),
        sentiment_analyze=lambda m, r: _sentiment(),
        estimate=lambda m, r, s, w, cfg, **kw: _estimate(),
        edge_calc=lambda m, est, cfg: _edge(),
        sizing=lambda m, e, est, cfg: _sizing(),
        risk_assess=risk_manager.assess,
        format_decision=lambda *a, **kw: __import__("decision_formatter").format_decision(*a, **kw),
        execute=executor,
        open_trades=lambda: list(open_positions),
    )

    summary = pipeline.run_once(
        cfg, client=object(), deps=deps,
        markets_override=[candidate, old_peer],
    )
    assert summary["executions_attempted"] == 0
    assert executor.calls == []
    record = summary["decisions"][0]
    skip = record["execution_skip_reason"] or ""
    assert "risk_rejected" in skip
    assert "category_exposure" in skip


def test_run_once_rejects_when_correlated_cap_already_exceeded(cfg):
    """If correlated exposure in the same event group is already above the
    cap, risk_manager rejects and no execution is attempted.

    Tickers are shaped so that risk_manager's separate ``related_group``
    duplicate check (which uses a prefix match on the candidate's
    ``event_ticker``) does NOT fire — this lets us prove the exposure
    cap itself is the rejecting check, not the cheaper duplicate guard.
    """
    cfg.kill_switch = False  # see note above
    cfg.max_category_exposure_dollars = 999.0  # not the binding cap
    cfg.max_correlated_exposure_dollars = 5.0

    # Real-Kalshi-style tickers: _derive_event_ticker strips a `-B<digits>`
    # suffix, so both resolve to "KXBTCD-26MAY31". Leaving event_ticker=""
    # on the candidate keeps the related_group check (which uses
    # `event_ticker or ticker`) from firing.
    candidate = _market("KXBTCD-26MAY31-B50000",
                        event_ticker="", category="crypto")
    sibling = _market("KXBTCD-26MAY31-B60000",
                      event_ticker="", category="crypto")
    open_positions = [_trade("KXBTCD-26MAY31-B60000",
                             dollars=8.0, trade_id="sib")]

    import risk_manager
    executor = _ExecRecorder()
    deps = pipeline.PipelineDeps(
        scan=lambda c, cfg: [candidate, sibling],
        filter_run=lambda ms, cfg: filters_mod.FilterResult(
            passed=[candidate], rejected=[],
        ),
        weird_move_detect=lambda m, all_m: _weird(),
        research=lambda m, cfg: _research(),
        sentiment_analyze=lambda m, r: _sentiment(),
        estimate=lambda m, r, s, w, cfg, **kw: _estimate(),
        edge_calc=lambda m, est, cfg: _edge(),
        sizing=lambda m, e, est, cfg: _sizing(),
        risk_assess=risk_manager.assess,
        format_decision=lambda *a, **kw: __import__("decision_formatter").format_decision(*a, **kw),
        execute=executor,
        open_trades=lambda: list(open_positions),
    )

    summary = pipeline.run_once(
        cfg, client=object(), deps=deps,
        markets_override=[candidate, sibling],
    )
    assert summary["executions_attempted"] == 0
    assert executor.calls == []
    skip = summary["decisions"][0]["execution_skip_reason"] or ""
    assert "risk_rejected" in skip
    assert "correlated_exposure" in skip


# ── 3. --monitor-only path: one-shot, safe, never places live orders ─────────

def test_monitor_only_calls_check_positions_exactly_once(cfg, monkeypatch, tmp_path):
    """`python main.py --monitor-only` must call monitor.check_positions
    exactly once and then return — no loop."""
    import main
    import monitor as monitor_mod

    cfg.kalshi_api_key = "x"
    cfg.kalshi_private_key_path = ""  # no client; we'll fake it below
    cfg.log_dir = str(tmp_path / "logs")
    cfg.db_path = str(tmp_path / "db.sqlite")

    monkeypatch.setattr(sys, "argv", ["main.py", "--monitor-only"])
    monkeypatch.setattr(main, "get_config", lambda: cfg)

    # Pretend credentials are present so the client builder runs.
    monkeypatch.setattr(main, "_build_client_if_credentials_present",
                        lambda c: object())

    calls = {"n": 0}

    def fake_check(client, cfg_in):
        calls["n"] += 1

    monkeypatch.setattr(monitor_mod, "check_positions", fake_check)

    main.main()
    assert calls["n"] == 1, "--monitor-only must call check_positions exactly once"


def test_monitor_only_does_nothing_without_client(cfg, monkeypatch, tmp_path):
    """Missing credentials → no monitor pass, no crash, no live orders."""
    import main
    import monitor as monitor_mod

    cfg.kalshi_api_key = ""
    cfg.kalshi_private_key_path = ""
    cfg.log_dir = str(tmp_path / "logs")
    cfg.db_path = str(tmp_path / "db.sqlite")

    monkeypatch.setattr(sys, "argv", ["main.py", "--monitor-only"])
    monkeypatch.setattr(main, "get_config", lambda: cfg)

    calls = {"n": 0}

    def fake_check(client, cfg_in):
        calls["n"] += 1

    monkeypatch.setattr(monitor_mod, "check_positions", fake_check)

    main.main()
    assert calls["n"] == 0, "no credentials → monitor pass must be skipped"


def test_monitor_only_swallows_check_positions_exceptions(cfg, monkeypatch, tmp_path):
    """A crash in monitor.check_positions must NOT propagate out of main()."""
    import main
    import monitor as monitor_mod

    cfg.log_dir = str(tmp_path / "logs")
    cfg.db_path = str(tmp_path / "db.sqlite")

    monkeypatch.setattr(sys, "argv", ["main.py", "--monitor-only"])
    monkeypatch.setattr(main, "get_config", lambda: cfg)
    monkeypatch.setattr(main, "_build_client_if_credentials_present",
                        lambda c: object())

    def boom(*a, **kw):
        raise RuntimeError("monitor crashed")

    monkeypatch.setattr(monitor_mod, "check_positions", boom)

    main.main()  # must not raise


def test_monitor_only_subprocess_exits_cleanly(tmp_path):
    """End-to-end: `python main.py --monitor-only` exits 0 promptly even
    when no credentials are set (no loop, no hang)."""
    env_overrides = {
        "LOG_DIR": str(tmp_path / "logs"),
        "DB_PATH": str(tmp_path / "db.sqlite"),
        "KALSHI_API_KEY": "",
        "ANTHROPIC_API_KEY": "x",
        "KALSHI_PRIVATE_KEY_PATH": "",
    }
    import os
    env = {**os.environ, **env_overrides}
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project_root / "main.py"), "--monitor-only"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, (
        f"main.py --monitor-only failed: stderr={result.stderr!r}"
    )
