"""
Single-cycle pipeline runner.

This module is glue code only. It:
  - fetches and normalizes markets via existing helpers,
  - applies the existing safety filters,
  - runs weird-move detection,
  - caps the number of markets that proceed to expensive analysis,
  - runs research → sentiment → probability → edge → sizing → risk,
  - asks decision_formatter for a final structured decision,
  - and only then — if and only if risk_manager.approved is True — passes
    the trade to trading.execute() (which itself enforces dry_run/paper/live
    gates again).

It DOES NOT:
  - start a loop, scheduler, or background task,
  - bypass risk_manager,
  - place orders directly,
  - call external APIs in tests (callers can inject fakes via PipelineDeps),
  - fabricate research when a real-time source is unavailable.

Use:
    summary = pipeline.run_once(cfg, client=client)
    # summary is a JSON-safe dict (see RunSummary keys).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import db
import decision_formatter
import edge as edge_mod
import filters as filters_mod
import logger
import market_scanner
import position_sizing
import prediction_model
import research_agents
import risk_manager
import sentiment
import trading
import weird_move
from config import Config
from kalshi_client import KalshiClient
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

_MODULE = "pipeline"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Hard, deterministic caps. These apply on top of any cfg limits and cannot
# be relaxed by the LLM or by individual market signals.
_DEFAULT_MAX_CANDIDATES = _env_int("MAX_CANDIDATES_PER_RUN", 3)
_DEFAULT_MAX_RESEARCH = _env_int("MAX_RESEARCH_MARKETS_PER_RUN", 3)


# ── Dependency injection (for tests) ─────────────────────────────────────────

@dataclass
class PipelineDeps:
    """
    Injection seam for tests. Every field defaults to the real production
    function. Tests override only what they need; nothing in this module
    reaches out to a network/LLM unless the underlying function does.
    """
    scan: Callable[[KalshiClient, Config], list[Market]] = market_scanner.scan
    filter_run: Callable[[list[Market], Config], filters_mod.FilterResult] = filters_mod.run
    weird_move_detect: Callable[[Market, list[Market]], WeirdMoveSignal] = weird_move.detect
    research: Callable[[Market, Config], ResearchResult] = research_agents.research_market
    sentiment_analyze: Callable[[Market, ResearchResult], SentimentResult] = sentiment.analyze
    estimate: Callable[..., ProbabilityEstimate] = prediction_model.estimate
    edge_calc: Callable[[Market, ProbabilityEstimate, Config], Optional[EdgeResult]] = edge_mod.calculate
    sizing: Callable[[Market, EdgeResult, ProbabilityEstimate, Config], PositionSize] = position_sizing.compute
    risk_assess: Callable[..., RiskAssessment] = risk_manager.assess
    format_decision: Callable[..., dict] = decision_formatter.format_decision
    execute: Callable[..., Optional[TradeRecord]] = trading.execute
    # data helpers
    open_trades: Callable[[], list[TradeRecord]] = db.get_open_trades


# ── Main entry point ─────────────────────────────────────────────────────────

def run_once(
    cfg: Config,
    *,
    client: Optional[KalshiClient] = None,
    deps: Optional[PipelineDeps] = None,
    max_candidates: Optional[int] = None,
    max_research: Optional[int] = None,
    markets_override: Optional[list[Market]] = None,
) -> dict[str, Any]:
    """
    Run exactly one scan→evaluate→(maybe execute) cycle and return a
    JSON-safe RunSummary dict.

    Live trading is impossible unless cfg already permits it; trading.execute()
    re-checks every live gate. This function additionally refuses to even
    attempt execution when risk_assessment.approved is False.
    """
    deps = deps or PipelineDeps()
    cap_candidates = _coerce_positive_int(max_candidates, _DEFAULT_MAX_CANDIDATES)
    cap_research = _coerce_positive_int(max_research, _DEFAULT_MAX_RESEARCH)

    summary: dict[str, Any] = {
        "markets_fetched":         0,
        "markets_passed_filters":  0,
        "candidates_analyzed":     0,
        "decisions":               [],
        "executions_attempted":    0,
        "errors":                  [],
    }

    logger.info(_MODULE, "run_once_start",
                trading_mode=cfg.trading_mode,
                kill_switch=cfg.kill_switch,
                live_trading_enabled=cfg.live_trading_enabled,
                max_candidates=cap_candidates,
                max_research=cap_research)

    # ── 1. Fetch markets ─────────────────────────────────────────────────
    try:
        if markets_override is not None:
            markets = list(markets_override)
        elif client is None:
            logger.warn(_MODULE, "no_client_skip_fetch")
            return summary
        else:
            markets = deps.scan(client, cfg)
    except Exception as exc:
        logger.error(_MODULE, "scan_failed", err=str(exc))
        summary["errors"].append({"stage": "scan", "err": str(exc)})
        return summary

    summary["markets_fetched"] = len(markets)
    logger.info(_MODULE, "markets_fetched", count=len(markets))

    # ── 2. Filter ────────────────────────────────────────────────────────
    try:
        filtered = deps.filter_run(markets, cfg)
    except Exception as exc:
        logger.error(_MODULE, "filter_failed", err=str(exc))
        summary["errors"].append({"stage": "filter", "err": str(exc)})
        return summary

    passed = list(filtered.passed)
    summary["markets_passed_filters"] = len(passed)
    logger.info(_MODULE, "filters_done",
                passed=len(passed), rejected=len(filtered.rejected))

    if not passed:
        logger.info(_MODULE, "run_once_end_no_candidates")
        return summary

    # ── 3. Weird-move detection (informational; capped step-down later) ──
    weird_map: dict[str, WeirdMoveSignal] = {}
    for market in passed:
        try:
            weird_map[market.ticker] = deps.weird_move_detect(market, passed)
        except Exception as exc:
            logger.error(_MODULE, "weird_move_failed",
                         ticker=market.ticker, err=str(exc))
            summary["errors"].append({
                "stage": "weird_move",
                "ticker": market.ticker, "err": str(exc),
            })

    # ── 4. Pick a small capped set of candidates ─────────────────────────
    # Prefer markets with sane liquidity and short time-to-resolution among
    # those that passed the filters. We use simple deterministic sorting,
    # no LLM ranking.
    candidates = _select_candidates(passed, cap_candidates)
    logger.info(_MODULE, "candidates_selected",
                count=len(candidates),
                tickers=[m.ticker for m in candidates])

    # ── 5. Per-market: research → sentiment → estimate → edge → … ───────
    # Bank-roll / position bookkeeping is read once; risk_manager.assess
    # re-checks every limit anyway.
    try:
        open_positions = deps.open_trades()
    except Exception as exc:
        logger.error(_MODULE, "open_trades_lookup_failed", err=str(exc))
        open_positions = []
    bankroll = cfg.paper_bankroll
    markets_by_event = _group_by_event(passed)

    research_budget = cap_research

    for market in candidates:
        ticker = market.ticker
        record = {
            "ticker": ticker,
            "decision": None,
            "executed": False,
            "execution_skip_reason": None,
        }
        try:
            # Research (LLM-assisted but capped)
            research_result: Optional[ResearchResult] = None
            if research_budget > 0:
                research_budget -= 1
                research_result = _safe_call(
                    deps.research, market, cfg,
                    summary=summary, stage="research", ticker=ticker,
                )
            else:
                logger.info(_MODULE, "research_budget_exhausted", ticker=ticker)

            # Sentiment requires a ResearchResult; if missing, build an empty
            # one so the downstream signal is honestly weak rather than absent.
            if research_result is None:
                research_result = ResearchResult(
                    ticker=ticker, query=market.title,
                    failed_reason="research_budget_or_error",
                )

            sentiment_result = _safe_call(
                deps.sentiment_analyze, market, research_result,
                summary=summary, stage="sentiment", ticker=ticker,
            )

            weird = weird_map.get(ticker)

            estimate = _safe_call(
                deps.estimate, market, research_result, sentiment_result, weird, cfg,
                kwargs={"markets_by_event": markets_by_event},
                summary=summary, stage="estimate", ticker=ticker,
            )
            if estimate is None:
                record["decision"] = _no_trade_decision(ticker, "estimate_failed")
                record["execution_skip_reason"] = "estimate_failed"
                summary["decisions"].append(record)
                summary["candidates_analyzed"] += 1
                continue

            edge_result = _safe_call(
                deps.edge_calc, market, estimate, cfg,
                summary=summary, stage="edge", ticker=ticker,
            )

            sizing_result: Optional[PositionSize] = None
            risk_result: Optional[RiskAssessment] = None
            if edge_result is not None:
                sizing_result = _safe_call(
                    deps.sizing, market, edge_result, estimate, cfg,
                    summary=summary, stage="sizing", ticker=ticker,
                )
                if sizing_result is not None:
                    risk_result = _safe_call(
                        deps.risk_assess,
                        sizing_result, market, edge_result,
                        open_positions, 0.0, 0, bankroll, cfg,
                        kwargs={
                            "category_exposure": 0.0,
                            "correlated_exposure": 0.0,
                            "action_type": "entry",
                            "live_buy_guard": trading.live_buy_guard,
                        },
                        summary=summary, stage="risk", ticker=ticker,
                    )

            decision = deps.format_decision(
                market, estimate, edge_result, sizing_result, risk_result, cfg,
            )
            record["decision"] = decision
            logger.decision(
                _MODULE, ticker, "trade_decision",
                outcome=decision.get("action", "NO_TRADE"),
                edge_cents=decision.get("edge_cents"),
                confidence=decision.get("confidence"),
                risk_summary=decision.get("risk_summary", "")[:200],
            )

            # ── 6. Execution gating (every gate is positive, none bypassed) ──
            skip_reason = _execution_skip_reason(
                decision, risk_result, sizing_result, cfg,
            )
            if skip_reason:
                record["execution_skip_reason"] = skip_reason
                logger.info(_MODULE, "execution_skipped",
                            ticker=ticker, reason=skip_reason,
                            action=decision.get("action"))
                summary["decisions"].append(record)
                summary["candidates_analyzed"] += 1
                continue

            # Only reach here if action is BUY_*, risk approved, sizing > 0
            assert edge_result is not None
            assert sizing_result is not None
            assert risk_result is not None and risk_result.approved

            summary["executions_attempted"] += 1
            try:
                trade = deps.execute(
                    sizing_result, estimate, edge_result, cfg,
                    client=client,
                    risk_approved=True,
                )
                record["executed"] = bool(trade is not None)
                if trade is not None:
                    record["trade_id"] = trade.id
                    logger.info(_MODULE, "execution_done",
                                ticker=ticker, mode=trade.mode,
                                trade_id=trade.id)
                else:
                    record["execution_skip_reason"] = "trading_execute_returned_none"
                    logger.info(_MODULE, "execution_returned_none", ticker=ticker)
            except Exception as exc:
                logger.error(_MODULE, "execution_raised",
                             ticker=ticker, err=str(exc))
                summary["errors"].append({
                    "stage": "execute", "ticker": ticker, "err": str(exc),
                })
                record["execution_skip_reason"] = f"execute_raised:{exc}"

        except Exception as exc:
            # Belt-and-braces: one market should not crash the run.
            logger.error(_MODULE, "candidate_crashed",
                         ticker=ticker, err=str(exc))
            summary["errors"].append({
                "stage": "candidate", "ticker": ticker, "err": str(exc),
            })
            record["decision"] = record["decision"] or _no_trade_decision(
                ticker, f"candidate_crashed:{exc}",
            )

        summary["decisions"].append(record)
        summary["candidates_analyzed"] += 1

    logger.info(
        _MODULE, "run_once_end",
        fetched=summary["markets_fetched"],
        passed=summary["markets_passed_filters"],
        analyzed=summary["candidates_analyzed"],
        executions=summary["executions_attempted"],
        errors=len(summary["errors"]),
    )
    return summary


# ── Helpers ──────────────────────────────────────────────────────────────────

def _coerce_positive_int(value: Optional[int], default: int) -> int:
    if value is None:
        return max(0, int(default))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _select_candidates(passed: list[Market], cap: int) -> list[Market]:
    """
    Deterministic candidate selection: prefer markets with the most
    liquidity, then the shortest time-to-resolution. No LLM ranking.
    """
    if cap <= 0 or not passed:
        return []
    ranked = sorted(
        passed,
        key=lambda m: (
            -float(getattr(m, "liquidity_dollars", 0.0) or 0.0),
            float(getattr(m, "minutes_to_settlement", 0.0) or 0.0),
            m.ticker,  # stable tiebreak
        ),
    )
    return ranked[:cap]


def _group_by_event(markets: list[Market]) -> dict[str, list[Market]]:
    by_event: dict[str, list[Market]] = {}
    for m in markets:
        key = m.event_ticker or m.ticker
        by_event.setdefault(key, []).append(m)
    return by_event


def _no_trade_decision(ticker: str, reason: str) -> dict[str, Any]:
    """Spec-shaped NO_TRADE record for runs that bail before formatting."""
    return {
        "action":            "NO_TRADE",
        "ticker":            ticker,
        "side":              None,
        "limit_price_cents": 0,
        "contracts":         0,
        "dollar_size":       0.0,
        "thesis":            "",
        "edge_cents":        0.0,
        "confidence":        0.0,
        "risk_summary":      reason,
        "monitoring_plan":   [],
    }


def _execution_skip_reason(
    decision: dict[str, Any],
    risk_result: Optional[RiskAssessment],
    sizing_result: Optional[PositionSize],
    cfg: Config,
) -> Optional[str]:
    """
    Positive-list gate: returns None ONLY when every gate to execute is met.
    Any failure returns a short string reason for logging.

    This deliberately does NOT relax any deterministic check — it only
    confirms that all upstream gates are already green.
    """
    # Report the *root* cause first — risk rejection / missing sizing / etc.
    # — before the derived "non-trade action" surface, so logs preserve why
    # we skipped, not just that the action wasn't BUY_*.
    if risk_result is None:
        return "no_risk_assessment"
    if not risk_result.approved:
        return f"risk_rejected:{risk_result.reason}"

    if sizing_result is None:
        return "no_sizing"
    if sizing_result.contracts <= 0 or sizing_result.dollars <= 0:
        return "zero_size"

    action = decision.get("action")
    if action not in ("BUY_YES", "BUY_NO"):
        return f"non_trade_action:{action}"

    if cfg.kill_switch and cfg.trading_mode == "live":
        return "kill_switch_blocks_live"

    if cfg.paper_only and cfg.trading_mode == "live":
        return "paper_only_blocks_live"

    return None


def _safe_call(
    fn: Callable[..., Any],
    *args: Any,
    summary: dict[str, Any],
    stage: str,
    ticker: str,
    kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Call `fn(*args, **kwargs)` and translate any exception into a logged,
    summary-tracked entry. Returns None on failure.
    """
    kwargs = kwargs or {}
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.error(_MODULE, f"{stage}_failed",
                     ticker=ticker, err=str(exc))
        summary["errors"].append({
            "stage": stage, "ticker": ticker, "err": str(exc),
        })
        return None
