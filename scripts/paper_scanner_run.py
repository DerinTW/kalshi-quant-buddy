from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import category_research
import db
import decision_formatter
import edge
import filters
import logger
import market_scanner
import position_sizing
import prediction_model
import research_agents
import risk_manager
import sentiment
import trading
import weird_move
from config import Config, get_config
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

_MODULE = "paper_scanner_run"
_SAFE_MODES = {"paper", "dry_run"}


def main() -> None:
    args = _parse_args()
    try:
        summary = run_scan(
            limit=args.limit,
            execute_paper=args.execute_paper,
            dry_run=args.dry_run,
            category=args.category,
        )
    except Exception as exc:
        logger.error(_MODULE, "paper_scan_failed", err=str(exc))
        raise
    print(json.dumps(_jsonable(summary), indent=2, default=str))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one safe paper-only scan against real Kalshi markets."
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument(
        "--execute-paper",
        action="store_true",
        default=False,
        help="Insert approved trades into the paper ledger. Default only logs decisions.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Scan and score without inserting paper trades. This is the default.",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Disable the dry-run label. Paper execution still requires --execute-paper.",
    )
    parser.add_argument("--category", default=None)
    return parser.parse_args()


def run_scan(
    *,
    limit: int = 25,
    execute_paper: bool = False,
    dry_run: bool = True,
    category: Optional[str] = None,
    cfg: Optional[Config] = None,
    client: Optional[object] = None,
) -> dict[str, Any]:
    """
    Run one paper-only scanner pass.

    This function is intentionally stricter than pipeline.run_once:
      - it refuses unsafe live-capable config before fetching markets,
      - it patches client.place_order with a hard RuntimeError tripwire,
      - and the only execution path calls trading.execute(..., mode_override="paper").
    """
    cfg = cfg or get_config()
    limit = max(0, int(limit or 0))

    logger.init(cfg.log_dir)
    db.init(cfg.db_path)
    _refuse_unsafe_config(cfg)
    _log_safety_banner(cfg, execute_paper=execute_paper, dry_run=dry_run)

    client = client or _build_client(cfg)
    _install_live_order_tripwire(client)

    raw_markets = _fetch_raw_markets(client, limit=limit, category=category)
    normalized = _normalize_markets(raw_markets)
    if category:
        desired = filters._normalize_category(category)
        normalized = [
            market for market in normalized
            if filters._normalize_category(market.category) == desired
        ]

    enrichment_subset = _select_candidates(
        [market for market in normalized if _eligible_for_enrichment(market, cfg)],
        limit,
    )
    _enrich_candidates(enrichment_subset, client)

    filter_result = filters.run(normalized, cfg)
    _log_filter_result(raw_markets, normalized, filter_result)

    passed = list(filter_result.passed)
    candidates = _select_candidates(passed, limit)
    weird_signals = _detect_weird_moves(passed)
    markets_by_event = _group_by_event(passed)
    markets_by_ticker = {market.ticker: market for market in normalized}
    open_positions = _safe_open_positions()

    summary: dict[str, Any] = {
        "raw_markets": len(raw_markets),
        "normalized_markets": len(normalized),
        "passed_count": len(filter_result.passed),
        "rejected_count": len(filter_result.rejected),
        "pass_rate": filter_result.pass_rate,
        "skip_reason_counts": filter_result.skip_reason_counts,
        "skip_reason_examples": filter_result.skip_reason_examples,
        "candidates_analyzed": 0,
        "execute_paper": bool(execute_paper),
        "dry_run": bool(dry_run),
        "decisions": [],
        "paper_trades_inserted": 0,
        "errors": [],
    }

    for market in candidates:
        record = _analyze_market(
            market=market,
            cfg=cfg,
            client=client,
            all_markets=passed,
            markets_by_event=markets_by_event,
            markets_by_ticker=markets_by_ticker,
            open_positions=open_positions,
            weird_signal=weird_signals.get(market.ticker),
            execute_paper=execute_paper,
            summary=summary,
        )
        summary["decisions"].append(record)
        summary["candidates_analyzed"] += 1

    logger.info(
        _MODULE,
        "paper_scan_done",
        raw_markets=summary["raw_markets"],
        normalized_markets=summary["normalized_markets"],
        passed=summary["passed_count"],
        rejected=summary["rejected_count"],
        analyzed=summary["candidates_analyzed"],
        paper_trades_inserted=summary["paper_trades_inserted"],
        errors=len(summary["errors"]),
    )
    return summary


def _refuse_unsafe_config(cfg: Config) -> None:
    reasons: list[str] = []
    if cfg.trading_mode not in _SAFE_MODES:
        reasons.append(f"TRADING_MODE={cfg.trading_mode!r}")
    if cfg.live_trading_enabled:
        reasons.append("LIVE_TRADING_ENABLED=true")
    if cfg.allow_live_orders:
        reasons.append("ALLOW_LIVE_ORDERS=true")
    if not cfg.paper_only:
        reasons.append("PAPER_ONLY=false")
    if reasons:
        raise RuntimeError("PAPER_SCAN_REFUSED: " + "; ".join(reasons))


def _log_safety_banner(cfg: Config, *, execute_paper: bool, dry_run: bool) -> None:
    fields = {
        "mode": cfg.trading_mode,
        "kill_switch": cfg.kill_switch,
        "paper_only": cfg.paper_only,
        "live_trading_enabled": cfg.live_trading_enabled,
        "allow_live_orders": cfg.allow_live_orders,
        "max_trade_dollars": cfg.max_trade_dollars,
        "category_allowlist": cfg.category_allowlist,
        "execute_paper": execute_paper,
        "dry_run": dry_run,
    }
    banner = (
        "PAPER SCANNER SAFETY BANNER "
        f"mode={fields['mode']} kill_switch={fields['kill_switch']} "
        f"paper_only={fields['paper_only']} "
        f"live_trading_enabled={fields['live_trading_enabled']} "
        f"allow_live_orders={fields['allow_live_orders']} "
        f"max_trade_dollars={fields['max_trade_dollars']} "
        f"category_allowlist={fields['category_allowlist']}"
    )
    print(banner)
    logger.info(_MODULE, "safety_banner", **fields)


def _build_client(cfg: Config) -> KalshiClient:
    return KalshiClient(
        api_key=cfg.kalshi_api_key,
        private_key_path=cfg.kalshi_private_key_path,
        base_url=cfg.kalshi_base_url,
    )


def _install_live_order_tripwire(client: object) -> None:
    def _blocked_place_order(*args: Any, **kwargs: Any) -> None:
        setattr(client, "_paper_scan_live_order_attempted", True)
        raise RuntimeError("LIVE_ORDER_ATTEMPTED_IN_PAPER_SCAN")

    setattr(client, "_paper_scan_live_order_attempted", False)
    setattr(client, "place_order", _blocked_place_order)


def _assert_no_live_order_attempt(client: object) -> None:
    if getattr(client, "_paper_scan_live_order_attempted", False):
        raise RuntimeError("LIVE_ORDER_ATTEMPTED_IN_PAPER_SCAN")


def _fetch_raw_markets(
    client: object,
    *,
    limit: int,
    category: Optional[str],
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    if category:
        return _fetch_raw_markets_for_category(client, limit=limit, category=category)

    if hasattr(client, "get_markets"):
        response = client.get_markets(  # type: ignore[attr-defined]
            status="open",
            limit=limit,
            mve_filter="exclude",
        )
    elif hasattr(client, "get_all_markets"):
        response = client.get_all_markets(status="open")  # type: ignore[attr-defined]
    else:
        raise AttributeError("client must provide get_markets or get_all_markets")

    return _with_series_metadata(client, _extract_market_list(response))


def _fetch_raw_markets_for_category(
    client: object,
    *,
    limit: int,
    category: str,
) -> list[dict[str, Any]]:
    desired = _normalize_series_category(category)
    series = [
        item for item in _get_series_list(client)
        if _normalize_series_category(item.get("category")) == desired
    ]
    if not series:
        logger.warn(_MODULE, "no_series_for_category", category=category)
        return []

    raw_markets: list[dict[str, Any]] = []
    for item in series:
        series_ticker = str(item.get("ticker") or "").strip()
        if not series_ticker:
            continue
        response = client.get_markets(  # type: ignore[attr-defined]
            status="open",
            limit=max(1, limit - len(raw_markets)),
            series_ticker=series_ticker,
            mve_filter="exclude",
        )
        for raw in _extract_market_list(response):
            enriched = dict(raw)
            enriched.setdefault("series_ticker", series_ticker)
            enriched.setdefault("category", desired)
            raw_markets.append(enriched)
            if len(raw_markets) >= limit:
                return raw_markets
    return raw_markets


def _extract_market_list(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, dict):
        raw_markets = response.get("markets", [])
    else:
        raw_markets = response
    if not isinstance(raw_markets, list):
        raise TypeError("Kalshi markets response did not contain a market list")
    return [_with_market_compat_fields(raw) for raw in raw_markets if isinstance(raw, dict)]


def _with_market_compat_fields(raw: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(raw)
    for cents_field, dollars_field in (
        ("yes_ask", "yes_ask_dollars"),
        ("yes_bid", "yes_bid_dollars"),
        ("no_ask", "no_ask_dollars"),
        ("no_bid", "no_bid_dollars"),
    ):
        if cents_field not in enriched and dollars_field in enriched:
            enriched[cents_field] = _dollars_to_cents(enriched.get(dollars_field))
    return enriched


def _dollars_to_cents(value: Any) -> int:
    try:
        return int(round(float(value) * 100))
    except (TypeError, ValueError):
        return 0


def _get_series_list(client: object) -> list[dict[str, Any]]:
    if hasattr(client, "get_series_list"):
        series = client.get_series_list()  # type: ignore[attr-defined]
    elif hasattr(client, "_get"):
        response = client._get("/series")  # type: ignore[attr-defined]
        series = response.get("series", []) if isinstance(response, dict) else []
    else:
        raise AttributeError("client must provide get_series_list for category scans")
    return [item for item in series if isinstance(item, dict)]


def _with_series_metadata(client: object, raw_markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    series_cache: dict[str, dict[str, Any]] = {}
    enriched_markets: list[dict[str, Any]] = []
    for raw in raw_markets:
        enriched = dict(raw)
        series_ticker = str(enriched.get("series_ticker") or "").strip()
        if not series_ticker:
            series_ticker = _derive_series_ticker(enriched)
            if series_ticker:
                enriched["series_ticker"] = series_ticker
        if series_ticker and not enriched.get("category"):
            if series_ticker not in series_cache:
                series_cache[series_ticker] = _get_series(client, series_ticker)
            category = _normalize_series_category(series_cache[series_ticker].get("category"))
            if category:
                enriched["category"] = category
        enriched_markets.append(enriched)
    return enriched_markets


def _get_series(client: object, series_ticker: str) -> dict[str, Any]:
    try:
        if hasattr(client, "get_series"):
            series = client.get_series(series_ticker)  # type: ignore[attr-defined]
        elif hasattr(client, "_get"):
            response = client._get(f"/series/{series_ticker}")  # type: ignore[attr-defined]
            series = response.get("series", response) if isinstance(response, dict) else {}
        else:
            return {}
    except Exception as exc:
        logger.debug(_MODULE, "series_lookup_failed", series_ticker=series_ticker, err=str(exc))
        return {}
    return series if isinstance(series, dict) else {}


def _derive_series_ticker(raw: dict[str, Any]) -> str:
    event_ticker = str(raw.get("event_ticker") or "").strip()
    if event_ticker:
        return event_ticker.split("-", 1)[0]
    ticker = str(raw.get("ticker") or "").strip()
    return ticker.split("-", 1)[0] if ticker else ""


def _normalize_series_category(category: Any) -> str:
    text = str(category or "").strip().lower()
    aliases = {
        "climate and weather": "weather",
        "economics": "economic",
        "financials": "financial",
    }
    return aliases.get(text, text)


def _normalize_markets(raw_markets: list[dict[str, Any]]) -> list[Market]:
    markets: list[Market] = []
    seen: set[str] = set()
    for raw in raw_markets:
        ticker = str(raw.get("ticker", "")).strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        market = market_scanner.normalize(raw)
        if market is not None:
            markets.append(market)
    return markets


def _eligible_for_enrichment(market: Market, cfg: Config) -> bool:
    if market.status != "open" or market.is_unsafe:
        return False
    if cfg.category_allowlist and market.category not in cfg.category_allowlist:
        return False
    return True


def _enrich_candidates(markets: list[Market], client: object) -> None:
    if not markets:
        return
    if hasattr(market_scanner, "enrich_with_orderbook_depth"):
        try:
            market_scanner.enrich_with_orderbook_depth(markets, client, delay_seconds=0.0)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warn(_MODULE, "orderbook_enrichment_failed", err=str(exc))
    if hasattr(market_scanner, "enrich_with_history"):
        try:
            market_scanner.enrich_with_history(markets, client, delay_seconds=0.0)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warn(_MODULE, "history_enrichment_failed", err=str(exc))


def _log_filter_result(
    raw_markets: list[dict[str, Any]],
    normalized: list[Market],
    result: filters.FilterResult,
) -> None:
    logger.info(
        _MODULE,
        "scanner_filter_summary",
        total_raw_markets=len(raw_markets),
        normalized_markets=len(normalized),
        passed_count=len(result.passed),
        rejected_count=len(result.rejected),
        pass_rate=round(result.pass_rate, 4),
        skip_reason_counts=result.skip_reason_counts,
        skip_reason_examples=result.skip_reason_examples,
    )


def _detect_weird_moves(markets: list[Market]) -> dict[str, WeirdMoveSignal]:
    if not markets:
        return {}
    try:
        if hasattr(weird_move, "batch_detect"):
            return weird_move.batch_detect(markets, markets)
    except Exception as exc:
        logger.warn(_MODULE, "weird_move_batch_failed", err=str(exc))

    signals: dict[str, WeirdMoveSignal] = {}
    for market in markets:
        try:
            signals[market.ticker] = weird_move.detect(market, markets)
        except Exception as exc:
            logger.warn(_MODULE, "weird_move_failed", ticker=market.ticker, err=str(exc))
    return signals


def _analyze_market(
    *,
    market: Market,
    cfg: Config,
    client: object,
    all_markets: list[Market],
    markets_by_event: dict[str, list[Market]],
    markets_by_ticker: dict[str, Market],
    open_positions: list[TradeRecord],
    weird_signal: Optional[WeirdMoveSignal],
    execute_paper: bool,
    summary: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ticker": market.ticker,
        "action": None,
        "executed": False,
        "execution_skip_reason": None,
    }
    try:
        research_result = _research_market(market, cfg)
        sentiment_result = sentiment.analyze(market, research_result)
        estimate = prediction_model.estimate(
            market=market,
            research=research_result,
            sentiment=sentiment_result,
            weird_move=weird_signal,
            cfg=cfg,
            markets_by_event=markets_by_event,
        )
        edge_result = edge.calculate(market, estimate, cfg)
        threshold_passed = (
            edge.passes_threshold(edge_result, cfg)
            if edge_result is not None
            else False
        )

        sizing_result: Optional[PositionSize] = None
        risk_result: Optional[RiskAssessment] = None
        if edge_result is not None:
            sizing_result = position_sizing.compute(market, edge_result, estimate, cfg)
            cat_exposure, corr_exposure = _compute_open_exposures(
                market, open_positions, markets_by_ticker
            )
            risk_result = risk_manager.assess(
                decision=sizing_result,
                market=market,
                edge=edge_result,
                open_positions=open_positions,
                daily_pnl=0.0,
                trades_today=0,
                bankroll=cfg.paper_bankroll,
                cfg=cfg,
                category_exposure=cat_exposure,
                correlated_exposure=corr_exposure,
                action_type="entry",
                live_buy_guard=trading.live_buy_guard,
            )

        decision = decision_formatter.format_decision(
            market=market,
            estimate=estimate,
            edge=edge_result,
            sizing=sizing_result,
            risk_assessment=risk_result,
            cfg=cfg,
        )
        action = decision.get("action", "NO_TRADE")
        record.update(
            {
                "action": action,
                "decision": decision,
                "threshold_passed": threshold_passed,
                "risk_approved": bool(risk_result and risk_result.approved),
            }
        )
        logger.decision(
            _MODULE,
            market.ticker,
            "paper_scan_decision",
            action,
            edge_cents=decision.get("edge_cents"),
            confidence=decision.get("confidence"),
            risk_summary=decision.get("risk_summary", "")[:300],
        )

        if action == "NO_TRADE":
            record["execution_skip_reason"] = "no_trade"
            return record

        if action not in ("BUY_YES", "BUY_NO"):
            record["execution_skip_reason"] = f"unsupported_action:{action}"
            return record

        if not execute_paper:
            record["execution_skip_reason"] = "execute_paper_not_requested"
            return record

        if risk_result is None or not risk_result.approved:
            record["execution_skip_reason"] = "risk_not_approved"
            return record

        if sizing_result is None or edge_result is None:
            record["execution_skip_reason"] = "missing_execution_inputs"
            return record

        trade = trading.execute(
            sizing_result,
            estimate,
            edge_result,
            cfg,
            client=client,  # type: ignore[arg-type]
            mode_override=trading.PAPER,
            risk_approved=True,
        )
        _assert_no_live_order_attempt(client)
        record["executed"] = trade is not None
        if trade is not None:
            record["trade_id"] = trade.id
            record["trade_mode"] = trade.mode
            summary["paper_trades_inserted"] += 1
        else:
            record["execution_skip_reason"] = "trading_execute_returned_none"

    except Exception as exc:
        _assert_no_live_order_attempt(client)
        logger.error(_MODULE, "candidate_failed", ticker=market.ticker, err=str(exc))
        record["action"] = "NO_TRADE"
        record["execution_skip_reason"] = f"candidate_failed:{exc}"
        summary["errors"].append(
            {"stage": "candidate", "ticker": market.ticker, "err": str(exc)}
        )
    return record


def _research_market(market: Market, cfg: Config) -> ResearchResult:
    items = []
    failed: list[str] = []
    try:
        if hasattr(category_research, "research_market_categorical"):
            category_items = category_research.research_market_categorical(market, cfg)
            items.extend(item.to_legacy() for item in category_items)
    except Exception as exc:
        failed.append(f"category_research:{exc}")
        logger.warn(_MODULE, "category_research_failed", ticker=market.ticker, err=str(exc))

    try:
        if hasattr(research_agents, "research_market"):
            agent_result = research_agents.research_market(market, cfg)
            items.extend(agent_result.items)
            if agent_result.failed_reason:
                failed.append(agent_result.failed_reason)
    except Exception as exc:
        failed.append(f"research_agents:{exc}")
        logger.warn(_MODULE, "research_agents_failed", ticker=market.ticker, err=str(exc))

    if not items and failed:
        return ResearchResult(
            ticker=market.ticker,
            query=market.title,
            failed_reason="; ".join(failed),
        )
    return ResearchResult(ticker=market.ticker, query=market.title, items=items)


def _safe_open_positions() -> list[TradeRecord]:
    try:
        return db.get_open_trades()
    except Exception as exc:
        logger.warn(_MODULE, "open_positions_lookup_failed", err=str(exc))
        return []


def _compute_open_exposures(
    market: Market,
    open_positions: list[TradeRecord],
    markets_by_ticker: dict[str, Market],
) -> tuple[float, float]:
    try:
        import pipeline

        return pipeline.compute_open_exposures(
            market, open_positions, markets_by_ticker
        )
    except Exception:
        return 0.0, 0.0


def _select_candidates(markets: list[Market], limit: int) -> list[Market]:
    if limit <= 0:
        return []
    return sorted(
        markets,
        key=lambda m: (
            -float(getattr(m, "liquidity_dollars", 0.0) or 0.0),
            float(getattr(m, "minutes_to_settlement", 0.0) or 0.0),
            m.ticker,
        ),
    )[:limit]


def _group_by_event(markets: list[Market]) -> dict[str, list[Market]]:
    by_event: dict[str, list[Market]] = {}
    for market in markets:
        key = market.event_ticker or market.ticker
        by_event.setdefault(key, []).append(market)
    return by_event


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":
    main()
