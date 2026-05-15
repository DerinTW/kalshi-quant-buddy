"""
Trade decision formatter.

Combines the deterministic outputs of:
  - prediction_model.estimate       (ProbabilityEstimate)
  - edge.calculate                  (EdgeResult | None)
  - position_sizing.compute         (PositionSize)
  - risk_manager.assess             (RiskAssessment)

…into a single structured JSON-compatible decision record.

This layer is intentionally thin:
  - It does NOT invent facts.
  - It does NOT call the LLM or any external service.
  - It does NOT place orders.
  - It does NOT override or relax any deterministic gate. If any upstream
    gate said "no", the action is NO_TRADE, period.
  - Risk manager is the final authority — even a positive edge and a non-zero
    sizing must defer to risk_assessment.approved.

Output schema:
  {
    "action": "BUY_YES|BUY_NO|NO_TRADE|EXIT|HOLD",
    "ticker": "...",
    "side": "yes|no|null",
    "limit_price_cents": int,
    "contracts": int,
    "dollar_size": float,
    "thesis": "...",
    "edge_cents": float,
    "confidence": float,
    "risk_summary": "...",
    "monitoring_plan": [str]
  }
"""
from __future__ import annotations
from typing import Any, Optional

import edge as edge_module
import logger
from config import Config
from models import (
    EdgeResult,
    Market,
    PositionSize,
    ProbabilityEstimate,
    RiskAssessment,
)

_MODULE = "decision_formatter"

# Same mapping used by edge.py / prediction_model.py — duplicated to keep this
# module dependency-light (no import from prediction_model).
_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "low":         0.30,
    "medium-low":  0.50,
    "medium":      0.65,
    "medium-high": 0.80,
    "high":        0.90,
}


def format_decision(
    market: Market,
    estimate: ProbabilityEstimate,
    edge: Optional[EdgeResult],
    sizing: Optional[PositionSize],
    risk_assessment: Optional[RiskAssessment],
    cfg: Config,
) -> dict[str, Any]:
    """
    Build the final structured trade-decision record.

    Returns a plain dict (JSON-serializable). Never raises; defaults toward
    NO_TRADE whenever an upstream input is missing or any gate is unclear.
    Does not mutate any input.
    """
    ticker = market.ticker
    thesis = _safe_str(getattr(estimate, "reasoning", "")) or "No thesis available."
    confidence_label = getattr(estimate, "confidence", "low")
    confidence_float = _CONFIDENCE_WEIGHTS.get(confidence_label, 0.0)

    # ── Collect blocking reasons; if any, return NO_TRADE ────────────────
    block_reasons: list[str] = []

    if edge is None:
        block_reasons.append("no_edge")
    if sizing is None:
        block_reasons.append("no_sizing")
    if risk_assessment is None:
        block_reasons.append("no_risk_assessment")

    if edge is not None and not _edge_passes(edge, cfg):
        block_reasons.append("edge_below_threshold")

    if confidence_float < cfg.min_confidence:
        block_reasons.append(
            f"confidence_below_threshold:{confidence_label}({confidence_float:.2f})"
            f"<{cfg.min_confidence:.2f}"
        )

    if sizing is not None and (sizing.contracts <= 0 or sizing.dollars <= 0):
        block_reasons.append("zero_size")

    if risk_assessment is not None and not risk_assessment.approved:
        block_reasons.append(
            f"risk_rejected:{_safe_str(risk_assessment.reason) or 'unknown'}"
        )

    if block_reasons:
        decision = _no_trade(
            ticker=ticker,
            thesis=thesis,
            confidence_float=confidence_float,
            edge=edge,
            risk_assessment=risk_assessment,
            block_reasons=block_reasons,
        )
        logger.info(_MODULE, "no_trade", ticker=ticker,
                    reasons=block_reasons,
                    confidence=confidence_label)
        return decision

    # ── All gates passed → BUY_YES / BUY_NO ──────────────────────────────
    # Mypy/runtime safety: edge / sizing / risk_assessment are non-None here
    # because the block_reasons checks above would have caught them.
    assert edge is not None
    assert sizing is not None
    assert risk_assessment is not None

    action, side = _action_for_side(edge.side)
    limit_price = (
        sizing.entry_price_cents
        if sizing.entry_price_cents > 0
        else edge.entry_price_cents
    )

    decision: dict[str, Any] = {
        "action":            action,
        "ticker":            ticker,
        "side":              side,
        "limit_price_cents": int(limit_price),
        "contracts":         int(sizing.contracts),
        "dollar_size":       round(float(sizing.dollars), 2),
        "thesis":            thesis,
        "edge_cents":        round(float(edge.adjusted_edge_pct), 4),
        "confidence":        round(confidence_float, 4),
        "risk_summary":      _approved_risk_summary(risk_assessment),
        "monitoring_plan":   _build_monitoring_plan(market, edge, cfg),
    }

    logger.info(
        _MODULE, "decision_formatted",
        ticker=ticker,
        action=action,
        side=side,
        contracts=decision["contracts"],
        dollars=f"${decision['dollar_size']:.2f}",
        edge_cents=f"{decision['edge_cents']:+.2f}",
        confidence=confidence_label,
    )
    return decision


# ── Helpers ──────────────────────────────────────────────────────────────────

def _edge_passes(edge: EdgeResult, cfg: Config) -> bool:
    """Delegate to edge.passes_threshold; never duplicate threshold logic."""
    try:
        return bool(edge_module.passes_threshold(edge, cfg))
    except Exception as exc:
        logger.error(_MODULE, "edge_threshold_check_failed",
                     ticker=getattr(edge, "ticker", ""), err=str(exc))
        return False


def _action_for_side(side: str) -> tuple[str, Optional[str]]:
    """Map edge.side ('YES'/'NO') → (action, output side)."""
    s = (side or "").strip().upper()
    if s == "YES":
        return "BUY_YES", "yes"
    if s == "NO":
        return "BUY_NO", "no"
    return "NO_TRADE", None


def _no_trade(
    *,
    ticker: str,
    thesis: str,
    confidence_float: float,
    edge: Optional[EdgeResult],
    risk_assessment: Optional[RiskAssessment],
    block_reasons: list[str],
) -> dict[str, Any]:
    edge_cents = (
        round(float(edge.adjusted_edge_pct), 4) if edge is not None else 0.0
    )
    risk_summary = _no_trade_risk_summary(risk_assessment, block_reasons)
    return {
        "action":            "NO_TRADE",
        "ticker":            ticker,
        "side":              None,
        "limit_price_cents": 0,
        "contracts":         0,
        "dollar_size":       0.0,
        "thesis":            thesis,
        "edge_cents":        edge_cents,
        "confidence":        round(confidence_float, 4),
        "risk_summary":      risk_summary,
        "monitoring_plan":   [],
    }


def _approved_risk_summary(risk_assessment: RiskAssessment) -> str:
    """One-line summary for an approved trade — mode plus pass count."""
    passes = len(risk_assessment.checks_passed or [])
    reason = _safe_str(risk_assessment.reason) or "approved"
    mode = _safe_str(getattr(risk_assessment, "mode", "")) or "unknown"
    return f"approved ({mode}, {passes} checks passed): {reason}"


def _no_trade_risk_summary(
    risk_assessment: Optional[RiskAssessment],
    block_reasons: list[str],
) -> str:
    parts: list[str] = []
    parts.extend(block_reasons)
    if risk_assessment is not None and risk_assessment.checks_failed:
        parts.extend(
            _safe_str(c) for c in risk_assessment.checks_failed
            if _safe_str(c)
        )
    # Keep the summary stable and short
    return "; ".join(dict.fromkeys(p for p in parts if p))[:600]


def _build_monitoring_plan(
    market: Market,
    edge: EdgeResult,
    cfg: Config,
) -> list[str]:
    """
    Deterministic monitoring checklist for an approved trade. Uses only
    fields already on the market/edge/config — no new facts.
    """
    entry = int(edge.entry_price_cents)
    plan: list[str] = [
        f"Watch spread (current {edge.spread_cents}¢, max "
        f"{cfg.max_spread_cents_edge}¢ before re-evaluating)",
        "Re-check thesis on material news or resolution-rule clarification",
    ]
    if cfg.stop_loss_cents > 0:
        if edge.side == "YES":
            stop = max(1, entry - int(cfg.stop_loss_cents))
        else:
            stop = max(1, entry - int(cfg.stop_loss_cents))
        plan.append(
            f"Stop-loss if entry-side price falls {cfg.stop_loss_cents}¢ "
            f"below {entry}¢ (≈ {stop}¢)"
        )
    if cfg.take_profit_cents > 0:
        target = min(99, entry + int(cfg.take_profit_cents))
        plan.append(
            f"Take-profit at +{cfg.take_profit_cents}¢ from entry "
            f"(≈ {target}¢)"
        )
    if cfg.exit_if_edge_below_cents > 0:
        plan.append(
            f"Exit if adjusted edge drops below {cfg.exit_if_edge_below_cents}¢"
        )
    if getattr(market, "minutes_to_settlement", None) is not None:
        plan.append(
            "Force review in last "
            f"{cfg.force_review_last_minutes} minutes before settlement"
        )
    return plan


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:
        return ""
