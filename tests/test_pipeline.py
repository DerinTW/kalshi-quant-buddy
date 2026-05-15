"""
Step 19 — single-cycle pipeline runner tests.

Every test injects fakes for the network/LLM/execution layers via
PipelineDeps. No test touches real APIs or real LLMs.

Safety guarantees under test:
  - --run-once never bypasses risk_manager.assess
  - execute is never called with risk_approved=True unless risk_assessment
    actually approved
  - kill_switch + live mode blocks execution
  - missing edge / missing sizing / risk rejection all produce NO_TRADE
  - one bad market does not crash the run
  - candidate cap is honoured
  - default `main()` (no CLI flags) does NOT start the pipeline
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


# ── Fixtures / builders ──────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> Config:
    return Config(
        kalshi_api_key="x", anthropic_api_key="x",
        kalshi_private_key_path="",
    )


def _market(ticker: str = "KXTEST") -> Market:
    now = datetime.now(timezone.utc)
    m = Market(
        ticker=ticker, title="t", status="open",
        yes_ask=55, yes_bid=53, no_ask=47, no_bid=45,
        volume=2000, volume_24h=12_000, open_interest=4_000,
        close_time=now + timedelta(minutes=120),
        settlement_time=now + timedelta(minutes=120),
        category="crypto", rules_primary="rules",
        event_ticker=ticker,
        price_history=[{"yes_price": 54}, {"yes_price": 55}],
    )
    m.minutes_to_close = 120.0
    m.minutes_to_settlement = 120.0
    m.liquidity_dollars = 5_000.0
    m.orderbook_depth = 200
    return m


def _research() -> ResearchResult:
    return ResearchResult(ticker="KXTEST", query="q", items=[])


def _sentiment() -> SentimentResult:
    return SentimentResult(
        ticker="KXTEST", sentiment_score=0.3,
        narrative_direction="supports_yes", confidence=0.7,
        market_impact_estimate_cents=5, major_contradictions=[],
        item_count=3, contributing_sources=["Reuters"],
        source_credibility=0.85, event_relevance=0.9, rumor_risk="low",
    )


def _estimate() -> ProbabilityEstimate:
    return ProbabilityEstimate(
        ticker="KXTEST", yes_probability=0.62,
        confidence="medium-high", reasoning="thesis",
    )


def _edge(adj_pct: float = 8.0) -> EdgeResult:
    return EdgeResult(
        ticker="KXTEST", side="YES", entry_price_cents=55,
        implied_yes_prob=0.55, estimated_yes_prob=0.62,
        raw_edge_pct=10.0, adjusted_edge_pct=adj_pct,
        expected_value=0.10, adjusted_ev=adj_pct / 100.0,
        confidence="medium-high",
        confidence_adjusted_ev=(adj_pct / 100.0) * 0.80,
        confidence_adjusted_edge_pct=adj_pct * 0.80,
        spread_cents=2,
    )


def _sizing(contracts: int = 5, dollars: float = 2.75) -> PositionSize:
    return PositionSize(
        ticker="KXTEST", side="YES",
        dollars=dollars, contracts=contracts,
        entry_price_cents=55, max_loss_dollars=dollars,
    )


def _risk(approved: bool = True, reason: str = "ok") -> RiskAssessment:
    return RiskAssessment(
        ticker="KXTEST", side="YES",
        approved=approved, reason=reason, mode="paper",
        checks_passed=["spread_ok"],
        checks_failed=[] if approved else ["duplicate_position"],
    )


def _weird() -> WeirdMoveSignal:
    return WeirdMoveSignal(
        ticker="KXTEST", flagged=False, classification="none",
        price_change_5m=0.0, price_change_15m=0.0,
        volume_ratio=1.0, spread_change=1.0,
        related_disagreement=0.0, triggers=[],
        description="", confidence="low",
    )


def _trade(ticker: str = "KXTEST") -> TradeRecord:
    return TradeRecord(
        id="t1", ticker=ticker, side="YES", contracts=5,
        entry_price_cents=55, dollars_at_risk=2.75,
        mode="paper", thesis="thesis", estimated_yes_prob=0.6,
        result="open",
    )


class _ExecutionRecorder:
    """Tracks how execute() was called so tests can assert it ran (or didn't)."""
    def __init__(self, return_value=None):
        self.calls: list[dict] = []
        self.return_value = return_value

    def __call__(self, sizing, estimate, edge, cfg, *, client=None, risk_approved=False, **kw):
        self.calls.append({
            "ticker": sizing.ticker,
            "risk_approved": risk_approved,
            "mode": cfg.trading_mode,
        })
        return self.return_value


def _deps(
    *,
    markets: list[Market],
    edge=None, sizing=None, risk=None,
    execute=None,
    research=None,
    sentiment_res=None,
    estimate=None,
    weird=None,
) -> pipeline.PipelineDeps:
    """Build a PipelineDeps with simple fakes that ignore real signatures."""
    return pipeline.PipelineDeps(
        scan=lambda c, cfg: list(markets),
        filter_run=lambda ms, cfg: filters_mod.FilterResult(
            passed=list(ms), rejected=[],
        ),
        weird_move_detect=lambda m, all_m: (weird if weird is not None else _weird()),
        research=lambda m, cfg: (research if research is not None else _research()),
        sentiment_analyze=lambda m, r: (sentiment_res if sentiment_res is not None else _sentiment()),
        estimate=lambda m, r, s, w, cfg, **kw: (estimate if estimate is not None else _estimate()),
        edge_calc=lambda m, est, cfg: edge,
        sizing=lambda m, e, est, cfg: sizing,
        risk_assess=lambda *a, **kw: risk,
        format_decision=lambda *a, **kw: __import__("decision_formatter").format_decision(*a, **kw),
        execute=(execute if execute is not None else _ExecutionRecorder()),
        open_trades=lambda: [],
    )


# ── 1. Default main() does not start the pipeline ────────────────────────────

def test_main_default_does_not_start_pipeline(cfg, monkeypatch, tmp_path):
    """Running `python main.py` (no flags) must not invoke pipeline.run_once."""
    import main
    monkeypatch.setattr(sys, "argv", ["main.py"])
    monkeypatch.setattr(main, "get_config", lambda: cfg)
    cfg.log_dir = str(tmp_path / "logs")
    cfg.db_path = str(tmp_path / "db.sqlite")

    called = {"n": 0}
    import pipeline as pipe_mod
    monkeypatch.setattr(pipe_mod, "run_once",
                        lambda *a, **kw: called.__setitem__("n", called["n"] + 1) or {})

    main.main()
    assert called["n"] == 0, "default main() must not start the pipeline"


def test_main_init_only_still_works(cfg, monkeypatch, tmp_path):
    import main
    monkeypatch.setattr(sys, "argv", ["main.py", "--init-only"])
    monkeypatch.setattr(main, "get_config", lambda: cfg)
    cfg.log_dir = str(tmp_path / "logs")
    cfg.db_path = str(tmp_path / "db.sqlite")

    import pipeline as pipe_mod
    called = {"n": 0}
    monkeypatch.setattr(pipe_mod, "run_once",
                        lambda *a, **kw: called.__setitem__("n", called["n"] + 1) or {})

    main.main()
    assert called["n"] == 0


def test_main_run_once_starts_single_cycle(cfg, monkeypatch, tmp_path, capsys):
    import main
    monkeypatch.setattr(sys, "argv", ["main.py", "--run-once"])
    monkeypatch.setattr(main, "get_config", lambda: cfg)
    cfg.log_dir = str(tmp_path / "logs")
    cfg.db_path = str(tmp_path / "db.sqlite")

    import pipeline as pipe_mod
    calls = {"n": 0}

    def fake_run_once(cfg_in, **kwargs):
        calls["n"] += 1
        return {
            "markets_fetched": 0, "markets_passed_filters": 0,
            "candidates_analyzed": 0, "decisions": [],
            "executions_attempted": 0, "errors": [],
        }
    monkeypatch.setattr(pipe_mod, "run_once", fake_run_once)

    main.main()
    assert calls["n"] == 1, "--run-once must invoke pipeline.run_once exactly once"


# ── 2. run_once safety: no client → no work, no errors ───────────────────────

def test_run_once_with_no_client_returns_empty_summary(cfg):
    summary = pipeline.run_once(cfg, client=None)
    assert summary["markets_fetched"] == 0
    assert summary["executions_attempted"] == 0
    assert summary["errors"] == []


# ── 3. Approved BUY_YES is executed exactly once ─────────────────────────────

def test_run_once_executes_when_all_gates_pass(cfg):
    recorder = _ExecutionRecorder(return_value=_trade())
    deps = _deps(
        markets=[_market()],
        edge=_edge(), sizing=_sizing(), risk=_risk(approved=True),
        execute=recorder,
    )
    summary = pipeline.run_once(
        cfg, client=object(), deps=deps, markets_override=[_market()],
    )
    assert summary["executions_attempted"] == 1
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["risk_approved"] is True
    decision = summary["decisions"][0]["decision"]
    assert decision["action"] == "BUY_YES"
    assert summary["decisions"][0]["executed"] is True


# ── 4. Risk rejection forces NO_TRADE, never calls execute ───────────────────

def test_run_once_does_not_execute_when_risk_rejects(cfg):
    recorder = _ExecutionRecorder()
    deps = _deps(
        markets=[_market()],
        edge=_edge(), sizing=_sizing(),
        risk=_risk(approved=False, reason="duplicate_position"),
        execute=recorder,
    )
    summary = pipeline.run_once(
        cfg, client=object(), deps=deps, markets_override=[_market()],
    )
    assert summary["executions_attempted"] == 0
    assert recorder.calls == []
    record = summary["decisions"][0]
    assert record["decision"]["action"] == "NO_TRADE"
    assert "risk_rejected" in record["execution_skip_reason"]


# ── 5. Risk is consulted BEFORE execute (ordering safety) ────────────────────

def test_run_once_calls_risk_before_trading(cfg):
    """If risk_assess raises, execute must not have been called."""
    call_order: list[str] = []

    def risk_boom(*a, **kw):
        call_order.append("risk")
        raise RuntimeError("risk crashed")

    def execute_spy(*a, **kw):
        call_order.append("execute")
        return None

    deps = _deps(
        markets=[_market()],
        edge=_edge(), sizing=_sizing(),
        risk=None,                 # not used since we override risk_assess below
        execute=execute_spy,
    )
    # Replace risk and execute with our ordering spies
    deps.risk_assess = risk_boom
    deps.execute = execute_spy

    summary = pipeline.run_once(
        cfg, client=object(), deps=deps, markets_override=[_market()],
    )
    assert "risk" in call_order
    assert "execute" not in call_order, (
        "execute must not run when risk_assess raised — there is no risk approval"
    )
    assert summary["executions_attempted"] == 0


# ── 6. Edge missing → NO_TRADE, no execute ───────────────────────────────────

def test_run_once_formats_no_trade_when_edge_missing(cfg):
    recorder = _ExecutionRecorder()
    deps = _deps(
        markets=[_market()],
        edge=None, sizing=None, risk=None,
        execute=recorder,
    )
    summary = pipeline.run_once(
        cfg, client=object(), deps=deps, markets_override=[_market()],
    )
    assert recorder.calls == []
    assert summary["executions_attempted"] == 0
    decision = summary["decisions"][0]["decision"]
    assert decision["action"] == "NO_TRADE"


# ── 7. Zero contracts → NO_TRADE, no execute ─────────────────────────────────

def test_run_once_zero_size_forces_no_trade(cfg):
    recorder = _ExecutionRecorder()
    deps = _deps(
        markets=[_market()],
        edge=_edge(), sizing=_sizing(contracts=0, dollars=0.0),
        risk=_risk(approved=True),
        execute=recorder,
    )
    summary = pipeline.run_once(
        cfg, client=object(), deps=deps, markets_override=[_market()],
    )
    assert recorder.calls == []
    decision = summary["decisions"][0]["decision"]
    assert decision["action"] == "NO_TRADE"


# ── 8. Kill switch + live mode blocks execution ──────────────────────────────

def test_run_once_does_not_execute_when_kill_switch_true(cfg):
    """Even with otherwise-passing gates, kill_switch + live mode must block."""
    cfg.kill_switch = True
    cfg.trading_mode = "live"
    recorder = _ExecutionRecorder(return_value=_trade())
    deps = _deps(
        markets=[_market()],
        edge=_edge(), sizing=_sizing(), risk=_risk(approved=True),
        execute=recorder,
    )
    summary = pipeline.run_once(
        cfg, client=object(), deps=deps, markets_override=[_market()],
    )
    assert recorder.calls == [], "kill_switch must block live execution"
    record = summary["decisions"][0]
    assert "kill_switch" in (record["execution_skip_reason"] or "")


def test_run_once_kill_switch_in_paper_still_runs_risk_path(cfg):
    """Kill switch in paper mode does not silently skip risk; the pipeline
    relies on trading.execute's own gates for kill-switch handling there."""
    cfg.kill_switch = True
    cfg.trading_mode = "paper"
    recorder = _ExecutionRecorder(return_value=_trade())
    deps = _deps(
        markets=[_market()],
        edge=_edge(), sizing=_sizing(), risk=_risk(approved=True),
        execute=recorder,
    )
    # Pipeline allows the call through — trading.execute is responsible for
    # mode-specific kill-switch semantics; pipeline must not invent its own.
    summary = pipeline.run_once(
        cfg, client=object(), deps=deps, markets_override=[_market()],
    )
    # The call reached execute (so the test confirms we didn't short-circuit
    # incorrectly). Whether the *real* trading.execute would have placed
    # an order is a question for tests in test_trading_modes.py, not here.
    assert summary["executions_attempted"] == 1


# ── 9. Candidate cap is honoured ─────────────────────────────────────────────

def test_run_once_limits_candidates(cfg):
    markets = [_market(f"KX{i:02d}") for i in range(10)]
    recorder = _ExecutionRecorder()
    deps = _deps(
        markets=markets,
        edge=None, sizing=None, risk=None,
        execute=recorder,
    )
    summary = pipeline.run_once(
        cfg, client=object(), deps=deps,
        markets_override=markets, max_candidates=2,
    )
    assert summary["candidates_analyzed"] == 2
    assert summary["markets_passed_filters"] == 10
    # Decisions are recorded only for analyzed candidates
    assert len(summary["decisions"]) == 2


# ── 10. One bad market does not crash the whole run ─────────────────────────

def test_one_market_failure_does_not_crash_run(cfg):
    markets = [_market("GOOD"), _market("BAD")]

    def edge_fn(m, est, cfg):
        if m.ticker == "BAD":
            raise RuntimeError("edge calc crashed for BAD")
        return _edge()

    deps = _deps(
        markets=markets, edge=_edge(),
        sizing=_sizing(), risk=_risk(approved=True),
        execute=_ExecutionRecorder(return_value=_trade(ticker="GOOD")),
    )
    deps.edge_calc = edge_fn

    summary = pipeline.run_once(
        cfg, client=object(), deps=deps, markets_override=markets,
        max_candidates=10,
    )
    # Errors list should contain BAD's failure but the run still finished
    tickers_with_errors = {e.get("ticker") for e in summary["errors"]}
    assert "BAD" in tickers_with_errors
    # GOOD's decision is recorded
    tickers_with_decisions = [d["ticker"] for d in summary["decisions"]]
    assert "GOOD" in tickers_with_decisions


# ── 11. Missing research does not crash and does not fabricate signal ────────

def test_missing_research_degrades_but_does_not_crash(cfg):
    """research_fn raises; pipeline substitutes an empty ResearchResult."""
    def research_boom(m, c):
        raise RuntimeError("perplexity down")

    deps = _deps(
        markets=[_market()],
        edge=_edge(), sizing=_sizing(), risk=_risk(approved=True),
        execute=_ExecutionRecorder(return_value=_trade()),
    )
    deps.research = research_boom

    summary = pipeline.run_once(
        cfg, client=object(), deps=deps, markets_override=[_market()],
    )
    # We get a decision (could be BUY_YES if everything else still passes,
    # or NO_TRADE — we don't assert action; we assert the run survived)
    assert summary["candidates_analyzed"] == 1
    # And the failure was reported
    research_errors = [e for e in summary["errors"] if e.get("stage") == "research"]
    assert research_errors


# ── 12. Summary is JSON-safe ─────────────────────────────────────────────────

def test_summary_is_json_safe(cfg):
    import json
    recorder = _ExecutionRecorder(return_value=_trade())
    deps = _deps(
        markets=[_market()],
        edge=_edge(), sizing=_sizing(), risk=_risk(approved=True),
        execute=recorder,
    )
    summary = pipeline.run_once(
        cfg, client=object(), deps=deps, markets_override=[_market()],
    )
    json.dumps(summary)  # must not raise


# ── 13. The pipeline never calls execute with risk_approved=True unless
#       risk_assessment.approved is actually True ────────────────────────────

def test_execute_risk_approved_argument_matches_risk_assessment(cfg):
    """Sweep: for every risk outcome, check the risk_approved flag passed to execute."""
    for approved, expect_calls in [(True, 1), (False, 0)]:
        recorder = _ExecutionRecorder(return_value=_trade())
        deps = _deps(
            markets=[_market()],
            edge=_edge(), sizing=_sizing(),
            risk=_risk(approved=approved),
            execute=recorder,
        )
        summary = pipeline.run_once(
            cfg, client=object(), deps=deps, markets_override=[_market()],
        )
        assert len(recorder.calls) == expect_calls, (
            f"approved={approved}: expected {expect_calls} execute call(s), "
            f"got {len(recorder.calls)}"
        )
        if approved:
            assert recorder.calls[0]["risk_approved"] is True


# ── 14. CLI safety smoke test: subprocess --init-only ────────────────────────

def test_init_only_subprocess_exits_cleanly(tmp_path):
    """Best-effort: running `python main.py --init-only` exits 0 quickly."""
    env_overrides = {
        "LOG_DIR": str(tmp_path / "logs"),
        "DB_PATH": str(tmp_path / "db.sqlite"),
        "KALSHI_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
        "KALSHI_PRIVATE_KEY_PATH": "",
    }
    import os
    env = {**os.environ, **env_overrides}
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project_root / "main.py"), "--init-only"],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, (
        f"main.py --init-only failed: stderr={result.stderr!r}"
    )
