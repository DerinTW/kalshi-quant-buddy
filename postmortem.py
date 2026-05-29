from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import db
import llm
import logger
from config import Config, get_config
from models import Postmortem, TradeRecord

_MODULE = "postmortem"
_PENDING_RULES_PATH = Path("rules") / "rules_pending_review.json"
_processed_trade_ids: set[str] = set()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_for_trade(
    trade: TradeRecord,
    *,
    cfg: Config | None = None,
    pending_rules_path: str | Path = _PENDING_RULES_PATH,
) -> Postmortem | None:
    """
    Create a postmortem for a losing paper trade.

    Rule changes are suggestions only and are written to a pending-review file.
    This function never edits config, .env, or active trading rules.
    """
    if trade.result != "loss":
        return None
    if trade.id in _processed_trade_ids:
        return None

    try:
        if db.postmortem_exists(trade.id):
            _processed_trade_ids.add(trade.id)
            return None
    except Exception as exc:
        logger.warn(_MODULE, "postmortem_duplicate_check_failed", trade_id=trade.id, err=str(exc))

    cfg = cfg or get_config()
    logger.info(_MODULE, "postmortem_started", ticker=trade.ticker, trade_id=trade.id)

    try:
        report = _build_report(trade, cfg)
        pm = _to_model(trade, report)
        db.insert_postmortem(pm)
        _processed_trade_ids.add(trade.id)
        logger.info(_MODULE, "postmortem_saved", ticker=trade.ticker, trade_id=trade.id)

        rule_changes = report.get("rule_changes_proposed") or []
        if rule_changes and report.get("should_update_rules_file", True):
            _write_pending_rule_changes(trade, report, pending_rules_path)
        return pm
    except Exception as exc:
        logger.error(_MODULE, "postmortem_failed", ticker=trade.ticker, trade_id=trade.id, err=str(exc))
        return None


def _build_report(trade: TradeRecord, cfg: Config) -> dict[str, Any]:
    llm_result: dict[str, Any] | None = None
    if cfg.anthropic_api_key:
        try:
            llm_result = llm.run_postmortem(
                cfg,
                trade.ticker,
                trade.ticker,
                trade.thesis,
                trade.estimated_yes_prob,
                trade.entry_price_cents,
                _actual_outcome(trade),
                trade_id=trade.id,
                side=trade.side,
                contracts=trade.contracts,
                exit_price_cents=trade.exit_price_cents or 0,
                pnl_dollars=float(trade.pnl_dollars or 0.0),
                result=trade.result or "loss",
                execution_log={
                    "mode": trade.mode,
                    "dollars_at_risk": trade.dollars_at_risk,
                },
            )
        except Exception as exc:
            logger.warn(_MODULE, "postmortem_failed", ticker=trade.ticker, trade_id=trade.id, err=str(exc))

    if not llm_result:
        return _fallback_report(trade, cfg)
    return _normalize_report(trade, cfg, llm_result)


def _fallback_report(trade: TradeRecord, cfg: Config) -> dict[str, Any]:
    root_causes: list[str] = []
    if not trade.thesis.strip():
        root_causes.append("Original thesis was missing or incomplete")
    if not (0 < trade.estimated_yes_prob < 1):
        root_causes.append("Estimated probability was missing or invalid")
    if trade.dollars_at_risk > cfg.max_trade_dollars:
        root_causes.append("Position size exceeded current per-trade cap")
    if _actual_outcome(trade).startswith("UNKNOWN"):
        root_causes.append("Trade was exited before final outcome; review exit timing and market structure")
    if not root_causes:
        root_causes.append("Trade thesis did not survive the realized outcome")

    rule_changes = [
        {
            "rule": "Review this losing trade before changing production rules",
            "priority": "medium",
            "requires_human_approval": True,
        }
    ]
    return {
        "trade_id": trade.id,
        "ticker": trade.ticker,
        "thesis": trade.thesis,
        "actual_outcome": _actual_outcome(trade),
        "loss_amount": abs(float(trade.pnl_dollars or 0.0)),
        "root_causes": root_causes,
        "good_process_bad_outcome": False,
        "rule_changes_proposed": rule_changes,
        "should_update_rules_file": True,
        "was_variance": False,
        "data_was_stale": "missing or invalid" in " ".join(root_causes).lower(),
        "resolution_handled_correctly": not _actual_outcome(trade).startswith("UNKNOWN"),
        "liquidity_hurt": _actual_outcome(trade).startswith("UNKNOWN"),
        "sizing_appropriate": trade.dollars_at_risk <= cfg.max_trade_dollars,
        "analysis": _fallback_analysis(trade, root_causes),
    }


def _normalize_report(trade: TradeRecord, cfg: Config, raw: dict[str, Any]) -> dict[str, Any]:
    root_causes = _as_str_list(raw.get("root_causes"))
    if not root_causes:
        root_causes = _root_causes_from_legacy(raw)
    rule_changes = _normalize_rule_changes(raw.get("proposed_rule_changes"))
    if not rule_changes:
        rule_changes = _normalize_rule_changes(raw.get("rule_changes_proposed"))
    if not rule_changes:
        proposal = str(raw.get("rule_change_proposal", "")).strip()
        if proposal and proposal.lower() != "none":
            rule_changes = [{
                "rule": proposal,
                "reason": "Legacy rule_change_proposal field from postmortem reviewer.",
                "priority": "medium",
                "requires_human_approval": True,
            }]
    if (
        not rule_changes
        and "proposed_rule_changes" not in raw
        and "rule_changes_proposed" not in raw
        and "rule_change_proposal" not in raw
    ):
        rule_changes = _fallback_report(trade, cfg)["rule_changes_proposed"]

    analysis = str(raw.get("analysis") or _fallback_analysis(trade, root_causes))
    should_update_rules_file = bool(raw.get("should_update_rules_file", False)) and bool(rule_changes)
    return {
        "trade_id": trade.id,
        "ticker": trade.ticker,
        "thesis": trade.thesis,
        "actual_outcome": str(raw.get("actual_outcome") or _actual_outcome(trade)),
        "loss_amount": abs(float(trade.pnl_dollars or 0.0)),
        "root_causes": root_causes,
        "data_quality_issues": _as_str_list(raw.get("data_quality_issues")),
        "reasoning_issues": _as_str_list(raw.get("reasoning_issues")),
        "risk_issues": _as_str_list(raw.get("risk_issues")),
        "execution_issues": _as_str_list(raw.get("execution_issues")),
        "market_structure_issues": _as_str_list(raw.get("market_structure_issues")),
        "good_process_bad_outcome": bool(raw.get("good_process_bad_outcome", raw.get("was_variance", False))),
        "proposed_rule_changes": rule_changes,
        "rule_changes_proposed": rule_changes,
        "should_update_rules_file": should_update_rules_file,
        "was_variance": bool(raw.get("was_variance", False)),
        "data_was_stale": bool(raw.get("data_was_stale", False)),
        "resolution_handled_correctly": bool(raw.get("resolution_handled_correctly", True)),
        "liquidity_hurt": bool(raw.get("liquidity_hurt", False)),
        "sizing_appropriate": bool(raw.get("sizing_appropriate", trade.dollars_at_risk <= cfg.max_trade_dollars)),
        "analysis": analysis,
    }


def _to_model(trade: TradeRecord, report: dict[str, Any]) -> Postmortem:
    return Postmortem(
        trade_id=trade.id,
        ticker=trade.ticker,
        original_thesis=trade.thesis,
        estimated_yes_prob=trade.estimated_yes_prob,
        market_price_at_entry=trade.entry_price_cents,
        actual_result=str(report.get("actual_outcome") or _actual_outcome(trade)),
        was_variance=bool(report.get("was_variance", False)),
        data_was_stale=bool(report.get("data_was_stale", False)),
        resolution_handled_correctly=bool(report.get("resolution_handled_correctly", True)),
        liquidity_hurt=bool(report.get("liquidity_hurt", False)),
        sizing_appropriate=bool(report.get("sizing_appropriate", True)),
        analysis=_analysis_with_causes(report),
        rule_change_proposal=json.dumps(report.get("rule_changes_proposed", []), sort_keys=True),
        human_approved=False,
    )


def _write_pending_rule_changes(
    trade: TradeRecord,
    report: dict[str, Any],
    pending_rules_path: str | Path = _PENDING_RULES_PATH,
) -> None:
    path = Path(pending_rules_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]]
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                pending = loaded.get("pending_rule_changes", [])
                existing = pending if isinstance(pending, list) else []
            else:
                existing = loaded if isinstance(loaded, list) else []
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []

    entry = {
        "trade_id": trade.id,
        "ticker": trade.ticker,
        "timestamp": _utc_now_iso(),
        "root_causes": report.get("root_causes", []),
        "rule_changes_proposed": report.get("rule_changes_proposed", []),
        "requires_human_approval": True,
    }
    if not any(item.get("trade_id") == trade.id for item in existing):
        existing.append(entry)
        payload = {"pending_rule_changes": existing}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info(
            _MODULE,
            "pending_rule_change_written",
            ticker=trade.ticker,
            trade_id=trade.id,
            path=str(path),
        )


def _normalize_rule_changes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            rule = item.strip()
            reason = "Legacy rule-change proposal from postmortem reviewer."
            priority = "medium"
        elif isinstance(item, dict):
            rule = str(item.get("rule", "")).strip()
            reason = str(item.get("reason", "")).strip()
            priority = str(item.get("priority", "medium")).strip() or "medium"
        else:
            continue
        if not rule or rule.lower() == "none":
            continue
        if priority not in {"low", "medium", "high"}:
            priority = "medium"
        out.append({
            "rule": rule,
            "reason": reason or "No reason provided by reviewer.",
            "priority": priority,
            "requires_human_approval": True,
        })
    return out


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _root_causes_from_legacy(raw: dict[str, Any]) -> list[str]:
    causes: list[str] = []
    if raw.get("data_was_stale"):
        causes.append("Input data may have been stale, wrong, incomplete, or low credibility")
    if not raw.get("resolution_handled_correctly", True):
        causes.append("Market resolution rule may have been misunderstood")
    if raw.get("liquidity_hurt"):
        causes.append("Liquidity, spread, or slippage hurt execution")
    if not raw.get("sizing_appropriate", True):
        causes.append("Position size was too large for the setup")
    if not causes:
        causes.append("Loss requires human review before rules change")
    return causes


def _analysis_with_causes(report: dict[str, Any]) -> str:
    payload = {
        "analysis": report.get("analysis", ""),
        "root_causes": report.get("root_causes", []),
        "good_process_bad_outcome": bool(report.get("good_process_bad_outcome", False)),
        "loss_amount": report.get("loss_amount", 0.0),
    }
    return json.dumps(payload, sort_keys=True)


def _fallback_analysis(trade: TradeRecord, root_causes: list[str]) -> str:
    return (
        "Deterministic fallback postmortem created because no usable LLM review was available. "
        "Data quality, reasoning, risk, execution, and market structure should be reviewed by a human. "
        f"Detected root causes: {', '.join(root_causes)}."
    )


def _actual_outcome(trade: TradeRecord) -> str:
    if trade.exit_price_cents == 100:
        return "YES"
    if trade.exit_price_cents == 0:
        return "NO"
    return "UNKNOWN_EXIT"


def _clear_processed_for_tests() -> None:
    _processed_trade_ids.clear()
