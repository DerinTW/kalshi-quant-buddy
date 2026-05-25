from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import category_research
import base_rate_research
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


class PipelineIntegrityError(RuntimeError):
    pass


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
    _warn_on_previous_incomplete_scan(cfg.log_dir)
    db.init(cfg.db_path)
    _refuse_unsafe_config(cfg)
    _log_safety_banner(cfg, execute_paper=execute_paper, dry_run=dry_run)
    scan_run_id = db.create_scan_run(
        mode=cfg.trading_mode,
        category=category,
        limit_requested=limit,
    )
    logger.audit(
        _MODULE,
        "scan_started",
        scan_run_id=scan_run_id,
        limit=limit,
        category=category,
        mode=cfg.trading_mode,
    )

    client = client or _build_client(cfg)
    _install_live_order_tripwire(client)

    raw_markets: list[dict[str, Any]] = []
    prefilter_skips: list[tuple[dict[str, Any], str]] = []
    normalized: list[Market] = []
    summary: dict[str, Any] | None = None
    try:
        raw_markets = _fetch_raw_markets(client, limit=limit, category=category)
        _audit_raw_markets(scan_run_id, raw_markets)
        prefiltered_raw_markets, prefilter_skips = _prefilter_raw_markets(raw_markets, cfg)
        _audit_prefilter_skips(scan_run_id, prefilter_skips)
        normalized = _normalize_markets(prefiltered_raw_markets, scan_run_id)
        _assert_pipeline_integrity(
            scan_run_id=scan_run_id,
            fetched_count=len(raw_markets),
            prefilter_skipped_count=len(prefilter_skips),
            normalized_count=len(normalized),
        )
        _audit_markets(scan_run_id, normalized, stage="normalized", outcome="passed")
        if category:
            desired = filters._normalize_category(category)
            category_mismatches = [
                market for market in normalized
                if filters._normalize_category(market.category) != desired
            ]
            _audit_category_mismatches(scan_run_id, category_mismatches, category)
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
        _audit_filter_result(scan_run_id, filter_result)
        _log_filter_result(raw_markets, normalized, filter_result)

        passed = list(filter_result.passed)
        candidates = _select_candidates(passed, limit)
        weird_signals = _detect_weird_moves(passed)
        markets_by_event = _group_by_event(passed)
        markets_by_ticker = {market.ticker: market for market in normalized}
        open_positions = _safe_open_positions()

        summary = {
            "raw_markets": len(raw_markets),
            "prefilter_skipped": len(prefilter_skips),
            "prefilter_skip_reason_counts": _count_prefilter_reasons(prefilter_skips),
            "normalized_markets": len(normalized),
            "normalization_error_count": sum(
                1 for market in normalized
                if str(market.unsafe_reason).startswith("normalization_error:")
            ),
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
            "filter_config": _filter_config_summary(cfg),
            "min_volume_24h": cfg.min_volume_24h,
            "min_liquidity": cfg.min_liquidity_dollars,
            "max_spread_cents": cfg.max_spread_cents,
            "min_yes_price": cfg.min_yes_price,
            "max_minutes_to_expiry": cfg.max_minutes_to_expiry,
            "min_orderbook_depth_at_limit": cfg.min_orderbook_depth_at_limit,
        }

        for market in candidates:
            record = _analyze_market(
                market=market,
                scan_run_id=scan_run_id,
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

        db.finish_scan_run(scan_run_id, summary)
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
        status = (
            "partial"
            if summary["errors"] or summary["normalization_error_count"]
            else "success"
        )
        _write_scan_summary(scan_run_id, status, summary)
        return summary
    except Exception as exc:
        error_message = str(exc)
        logger.audit(
            _MODULE,
            "scan_failed",
            scan_run_id=scan_run_id,
            error_message=error_message,
            traceback=traceback.format_exc(),
        )
        failed_summary = summary or {
            "raw_markets": len(raw_markets),
            "prefilter_skipped": len(prefilter_skips),
            "normalized_markets": len(normalized),
            "passed_count": 0,
            "candidates_analyzed": 0,
            "paper_trades_inserted": 0,
            "errors": [{"stage": "scan", "err": error_message}],
        }
        _write_scan_summary(scan_run_id, "failed", failed_summary, error_message=error_message)
        raise


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
        "min_volume_24h": cfg.min_volume_24h,
        "min_liquidity": cfg.min_liquidity_dollars,
        "max_spread_cents": cfg.max_spread_cents,
        "min_yes_price": cfg.min_yes_price,
        "max_minutes_to_expiry": cfg.max_minutes_to_expiry,
        "min_orderbook_depth_at_limit": cfg.min_orderbook_depth_at_limit,
        "blocked_event_prefixes": getattr(cfg, "blocked_event_prefixes", []),
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
        f"category_allowlist={fields['category_allowlist']} "
        f"min_volume_24h={fields['min_volume_24h']} "
        f"min_liquidity={fields['min_liquidity']} "
        f"max_spread_cents={fields['max_spread_cents']} "
        f"min_yes_price={fields['min_yes_price']} "
        f"max_minutes_to_expiry={fields['max_minutes_to_expiry']} "
        f"min_orderbook_depth_at_limit={fields['min_orderbook_depth_at_limit']}"
    )
    print(banner)
    logger.info(_MODULE, "safety_banner", **fields)


def _warn_on_previous_incomplete_scan(log_dir: str) -> None:
    path = Path(log_dir) / "agent.jsonl"
    if not path.exists():
        return

    last_started: Optional[dict[str, Any]] = None
    last_summary: Optional[dict[str, Any]] = None
    summaries_by_run: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stage = record.get("stage") or record.get("event")
                scan_run_id = str(record.get("scan_run_id") or "")
                if stage == "scan_started":
                    last_started = record
                elif stage == "scan_summary":
                    last_summary = record
                    if scan_run_id:
                        summaries_by_run[scan_run_id] = record
    except OSError as exc:
        logger.warn(_MODULE, "previous_scan_check_failed", err=str(exc))
        return

    if last_started is not None:
        scan_run_id = str(last_started.get("scan_run_id") or "")
        summary = summaries_by_run.get(scan_run_id)
        if summary is None:
            logger.warn(
                _MODULE,
                "previous_scan_missing_summary",
                scan_run_id=scan_run_id,
            )
            return
        if summary.get("status") == "failed":
            logger.warn(
                _MODULE,
                "previous_scan_failed",
                scan_run_id=scan_run_id,
                error_message=summary.get("error_message", ""),
            )
            return

    if last_summary is not None and last_summary.get("status") == "failed":
        logger.warn(
            _MODULE,
            "previous_scan_failed",
            scan_run_id=last_summary.get("scan_run_id", ""),
            error_message=last_summary.get("error_message", ""),
        )


def _write_scan_summary(
    scan_run_id: str,
    status: str,
    summary: dict[str, Any],
    *,
    error_message: str = "",
) -> None:
    payload = {
        "scan_run_id": scan_run_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "fetched_count": int(summary.get("raw_markets", 0) or 0),
        "prefilter_skipped_count": int(summary.get("prefilter_skipped", 0) or 0),
        "normalized_count": int(summary.get("normalized_markets", 0) or 0),
        "filter_passed_count": int(summary.get("passed_count", 0) or 0),
        "analyzed_count": int(summary.get("candidates_analyzed", 0) or 0),
        "traded_count": int(summary.get("paper_trades_inserted", 0) or 0),
        "error_message": error_message,
    }
    logger.audit(_MODULE, "scan_summary", **payload)


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

    if not hasattr(client, "get_markets"):
        if hasattr(client, "get_all_markets"):
            return client.get_all_markets(  # type: ignore[attr-defined]
                status="open",
                category=None,
                max_markets=limit,
            )
        raise AttributeError("client must provide get_markets")

    collected: list[dict[str, Any]] = []
    cursor: Optional[str] = None
    while len(collected) < limit:
        page_limit = min(200, limit - len(collected))
        response = client.get_markets(  # type: ignore[attr-defined]
            status="open",
            limit=page_limit,
            mve_filter="exclude",
            cursor=cursor,
        )
        batch = _extract_market_list(response)
        if not batch:
            break
        collected.extend(batch)
        cursor = response.get("cursor") if isinstance(response, dict) else None
        if not cursor:
            break

    return _with_series_metadata(client, collected)


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


def _normalize_markets(
    raw_markets: list[dict[str, Any]],
    scan_run_id: str = "",
) -> list[Market]:
    markets: list[Market] = []
    for idx, raw in enumerate(raw_markets):
        ticker = str(raw.get("ticker", "")).strip()
        try:
            market = market_scanner.normalize(raw)
            if market is None:
                market = _normalization_error_market(
                    raw,
                    "normalize_returned_none",
                    idx,
                )
                _audit_normalization_error(
                    scan_run_id,
                    raw,
                    "normalize_returned_none",
                    "",
                    market=market,
                    index=idx,
                )
            else:
                logger.audit(
                    _MODULE,
                    "normalized",
                    scan_run_id=scan_run_id,
                    ticker=market.ticker,
                    event_ticker=market.event_ticker,
                    unsafe=market.is_unsafe,
                    unsafe_reason=market.unsafe_reason,
                )
        except Exception as exc:
            tb = traceback.format_exc()
            market = _normalization_error_market(raw, str(exc), idx)
            _audit_normalization_error(
                scan_run_id,
                raw,
                str(exc),
                tb,
                market=market,
                index=idx,
            )
        if market is not None:
            markets.append(market)
    return markets


def _normalization_error_market(raw: dict[str, Any], err: str, index: int) -> Market:
    ticker = _raw_ticker(raw, index)
    now = datetime.now(timezone.utc)
    market = Market(
        ticker=ticker,
        title=str(raw.get("title") or raw.get("subtitle") or ""),
        status=str(raw.get("status") or ""),
        yes_ask=_raw_cents(raw, "yes_ask"),
        yes_bid=_raw_cents(raw, "yes_bid"),
        no_ask=_raw_cents(raw, "no_ask"),
        no_bid=_raw_cents(raw, "no_bid"),
        volume=_raw_int(raw, "volume"),
        volume_24h=_raw_int(raw, "volume_24h"),
        open_interest=_raw_int(raw, "open_interest"),
        close_time=now,
        settlement_time=now,
        category=str(raw.get("category") or raw.get("event_category") or ""),
        rules_primary=str(raw.get("rules_primary") or raw.get("description") or ""),
        event_ticker=str(raw.get("event_ticker") or raw.get("series_ticker") or ticker),
        fetched_at=now,
    )
    market.is_unsafe = True
    market.unsafe_reason = f"normalization_error: {err[:200]}"
    return market


def _audit_normalization_error(
    scan_run_id: str,
    raw: dict[str, Any],
    err: str,
    traceback_text: str,
    *,
    market: Market,
    index: int,
) -> None:
    logger.audit(
        _MODULE,
        "normalization_error",
        scan_run_id=scan_run_id,
        ticker=_raw_ticker(raw, index),
        err=err,
        traceback=traceback_text,
        unsafe_reason=market.unsafe_reason,
    )
    db.insert_market_audit(
        _market_audit_record(
            scan_run_id,
            market,
            stage="normalization_error",
            outcome="skipped",
            skip_reason=market.unsafe_reason,
            skip_reason_key="normalization_error",
            raw_extra={"err": err, "traceback": traceback_text, "raw": raw},
        )
    )


def _prefilter_raw_markets(
    raw_markets: list[dict[str, Any]],
    cfg: Config,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    kept: list[dict[str, Any]] = []
    skipped: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    for raw in raw_markets:
        ticker = str(raw.get("ticker") or "").strip()
        if not ticker:
            skipped.append((raw, "prefilter_missing_ticker"))
            continue
        if ticker in seen:
            skipped.append((raw, "prefilter_duplicate_ticker"))
            continue
        seen.add(ticker)
        reason = market_scanner.prefilter_raw_market(raw, cfg)
        if reason is None:
            kept.append(raw)
        else:
            skipped.append((raw, reason))
    if skipped:
        logger.info(
            _MODULE,
            "raw_prefilter_done",
            kept=len(kept),
            skipped=len(skipped),
            skip_reason_counts=_count_prefilter_reasons(skipped),
        )
    return kept, skipped


def _count_prefilter_reasons(skips: list[tuple[dict[str, Any], str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, reason in skips:
        key = filters._reason_key(reason)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def _filter_config_summary(cfg: Config) -> dict[str, Any]:
    return {
        "min_volume_24h": cfg.min_volume_24h,
        "min_liquidity": cfg.min_liquidity_dollars,
        "max_spread_cents": cfg.max_spread_cents,
        "min_yes_price": cfg.min_yes_price,
        "max_yes_price": cfg.max_yes_price,
        "max_minutes_to_expiry": cfg.max_minutes_to_expiry,
        "min_orderbook_depth_at_limit": cfg.min_orderbook_depth_at_limit,
        "category_allowlist": list(cfg.category_allowlist),
        "blocked_event_prefixes": list(getattr(cfg, "blocked_event_prefixes", [])),
    }


def _assert_pipeline_integrity(
    *,
    scan_run_id: str,
    fetched_count: int,
    prefilter_skipped_count: int,
    normalized_count: int,
) -> None:
    accounted = prefilter_skipped_count + normalized_count
    mismatch = fetched_count - accounted
    if mismatch == 0:
        return
    payload = {
        "scan_run_id": scan_run_id,
        "fetched_count": fetched_count,
        "prefilter_skipped_count": prefilter_skipped_count,
        "normalized_count": normalized_count,
        "mismatch": mismatch,
    }
    logger.audit(_MODULE, "pipeline_integrity_error", **payload)
    logger.error(_MODULE, "pipeline_integrity_error", **payload)
    raise PipelineIntegrityError(
        "fetch/prefilter/normalization count mismatch: "
        f"fetched={fetched_count} prefilter_skipped={prefilter_skipped_count} "
        f"normalized={normalized_count} mismatch={mismatch}"
    )


def _eligible_for_enrichment(market: Market, cfg: Config) -> bool:
    if market.status != "open" or market.is_unsafe:
        return False
    if cfg.category_allowlist:
        allowed = {filters._normalize_category(c) for c in cfg.category_allowlist}
        if filters._normalize_category(market.category) not in allowed:
            return False
    prefilter_reason = market_scanner.prefilter_raw_market(market.to_dict(), cfg)
    if prefilter_reason is not None:
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
    scan_run_id: str,
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
    _audit_market(scan_run_id, market, stage="analyzed", outcome="passed")
    try:
        research_result = _research_market(market, cfg)
        base_rate_signal = base_rate_research.estimate(market, cfg)
        research_result.base_rate_signal = base_rate_signal
        sentiment_result = sentiment.analyze(market, research_result)
        estimate = prediction_model.estimate(
            market=market,
            research=research_result,
            sentiment=sentiment_result,
            weird_move=weird_signal,
            cfg=cfg,
            markets_by_event=markets_by_event,
            base_rate_signal=base_rate_signal,
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
            _audit_decision(
                scan_run_id,
                market,
                decision=decision,
                estimate=estimate,
                stage=(
                    "risk_rejected"
                    if risk_result is not None and not risk_result.approved
                    else "no_trade"
                ),
                outcome="no_trade",
            )
            record["execution_skip_reason"] = "no_trade"
            return record

        if action not in ("BUY_YES", "BUY_NO"):
            _audit_decision(
                scan_run_id,
                market,
                decision=decision,
                estimate=estimate,
                stage="no_trade",
                outcome="no_trade",
            )
            record["execution_skip_reason"] = f"unsupported_action:{action}"
            return record

        if not execute_paper:
            _audit_decision(
                scan_run_id,
                market,
                decision=decision,
                estimate=estimate,
                stage="no_trade",
                outcome="no_trade",
                raw_extra={"execution_skip_reason": "execute_paper_not_requested"},
            )
            record["execution_skip_reason"] = "execute_paper_not_requested"
            return record

        if risk_result is None or not risk_result.approved:
            _audit_decision(
                scan_run_id,
                market,
                decision=decision,
                estimate=estimate,
                stage="risk_rejected",
                outcome="no_trade",
            )
            record["execution_skip_reason"] = "risk_not_approved"
            return record

        if sizing_result is None or edge_result is None:
            _audit_decision(
                scan_run_id,
                market,
                decision=decision,
                estimate=estimate,
                stage="no_trade",
                outcome="no_trade",
                raw_extra={"execution_skip_reason": "missing_execution_inputs"},
            )
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
            _audit_decision(
                scan_run_id,
                market,
                decision=decision,
                estimate=estimate,
                stage="paper_trade_opened",
                outcome="trade",
                raw_extra={"trade_id": trade.id, "trade_mode": trade.mode},
            )
        else:
            _audit_decision(
                scan_run_id,
                market,
                decision=decision,
                estimate=estimate,
                stage="error",
                outcome="error",
                raw_extra={"execution_skip_reason": "trading_execute_returned_none"},
            )
            record["execution_skip_reason"] = "trading_execute_returned_none"

    except Exception as exc:
        _assert_no_live_order_attempt(client)
        logger.error(_MODULE, "candidate_failed", ticker=market.ticker, err=str(exc))
        _audit_market(
            scan_run_id,
            market,
            stage="error",
            outcome="error",
            raw_extra={"err": str(exc)},
        )
        record["action"] = "NO_TRADE"
        record["execution_skip_reason"] = f"candidate_failed:{exc}"
        summary["errors"].append(
            {"stage": "candidate", "ticker": market.ticker, "err": str(exc)}
        )
    return record


def _raw_ticker(raw: dict[str, Any], index: int = 0) -> str:
    ticker = str(raw.get("ticker") or "").strip()
    audit_index = raw.get("_audit_fetch_index", index)
    return ticker or f"<missing_ticker:{audit_index}>"


def _audit_raw_markets(scan_run_id: str, raw_markets: list[dict[str, Any]]) -> None:
    records = []
    for idx, raw in enumerate(raw_markets):
        raw.setdefault("_audit_fetch_index", idx)
        ticker = _raw_ticker(raw, idx)
        logger.audit(
            _MODULE,
            "fetched",
            scan_run_id=scan_run_id,
            ticker=ticker,
            event_ticker=raw.get("event_ticker") or raw.get("series_ticker"),
        )
        records.append(
            {
                "scan_run_id": scan_run_id,
                "ticker": ticker,
                "event_ticker": raw.get("event_ticker") or raw.get("series_ticker"),
                "title": raw.get("title") or raw.get("subtitle"),
                "category": raw.get("category"),
                "status": raw.get("status"),
                "stage": "fetched",
                "outcome": "passed",
                "yes_bid": _raw_cents(raw, "yes_bid"),
                "yes_ask": _raw_cents(raw, "yes_ask"),
                "no_bid": _raw_cents(raw, "no_bid"),
                "no_ask": _raw_cents(raw, "no_ask"),
                "volume": _raw_int(raw, "volume"),
                "volume_24h": _raw_int(raw, "volume_24h"),
                "open_interest": _raw_int(raw, "open_interest"),
                "raw_json": raw,
            }
        )
    if records:
        db.insert_many_market_audits(records)


def _audit_prefilter_skips(
    scan_run_id: str,
    skips: list[tuple[dict[str, Any], str]],
) -> None:
    records = []
    for idx, (raw, reason) in enumerate(skips):
        ticker = _raw_ticker(raw, idx)
        logger.audit(
            _MODULE,
            "prefilter_skipped",
            scan_run_id=scan_run_id,
            ticker=ticker,
            skip_reason=reason,
            skip_reason_key=filters._reason_key(reason),
        )
        payload = dict(raw)
        payload["orderbook_depth_fetched"] = False
        records.append(
            {
                "scan_run_id": scan_run_id,
                "ticker": ticker,
                "event_ticker": raw.get("event_ticker") or raw.get("series_ticker"),
                "title": raw.get("title") or raw.get("subtitle"),
                "category": raw.get("category") or raw.get("event_category"),
                "status": raw.get("status"),
                "stage": "prefilter_skipped",
                "outcome": "skipped",
                "skip_reason": reason,
                "skip_reason_key": filters._reason_key(reason),
                "yes_bid": _raw_cents(raw, "yes_bid"),
                "yes_ask": _raw_cents(raw, "yes_ask"),
                "no_bid": _raw_cents(raw, "no_bid"),
                "no_ask": _raw_cents(raw, "no_ask"),
                "volume": _raw_int(raw, "volume"),
                "volume_24h": _raw_int(raw, "volume_24h"),
                "open_interest": _raw_int(raw, "open_interest"),
                "orderbook_depth": 0,
                "orderbook_depth_fetched": False,
                "raw_json": payload,
            }
        )
    if records:
        db.insert_many_market_audits(records)


def _audit_markets(
    scan_run_id: str,
    markets: list[Market],
    *,
    stage: str,
    outcome: str,
) -> None:
    records = [
        _market_audit_record(scan_run_id, market, stage=stage, outcome=outcome)
        for market in markets
    ]
    if records:
        db.insert_many_market_audits(records)


def _audit_filter_result(scan_run_id: str, result: filters.FilterResult) -> None:
    records: list[dict[str, Any]] = []
    for market in result.passed:
        records.append(
            _market_audit_record(
                scan_run_id, market, stage="filter_passed", outcome="passed"
            )
        )
    for market, reason in result.rejected:
        records.append(
            _market_audit_record(
                scan_run_id,
                market,
                stage="filter_skipped",
                outcome="skipped",
                skip_reason=reason,
                skip_reason_key=filters._reason_key(reason),
            )
        )
    if records:
        db.insert_many_market_audits(records)


def _audit_category_mismatches(
    scan_run_id: str,
    markets: list[Market],
    requested_category: str,
) -> None:
    records = [
        _market_audit_record(
            scan_run_id,
            market,
            stage="filter_skipped",
            outcome="skipped",
            skip_reason=(
                f"category={market.category!r} did not match requested "
                f"category={requested_category!r}"
            ),
            skip_reason_key="category",
        )
        for market in markets
    ]
    if records:
        db.insert_many_market_audits(records)


def _audit_market(
    scan_run_id: str,
    market: Market,
    *,
    stage: str,
    outcome: str,
    raw_extra: Optional[dict[str, Any]] = None,
) -> None:
    db.insert_market_audit(
        _market_audit_record(
            scan_run_id,
            market,
            stage=stage,
            outcome=outcome,
            raw_extra=raw_extra,
        )
    )


def _audit_decision(
    scan_run_id: str,
    market: Market,
    *,
    decision: dict[str, Any],
    estimate: ProbabilityEstimate,
    stage: str,
    outcome: str,
    raw_extra: Optional[dict[str, Any]] = None,
) -> None:
    raw_payload = {
        "decision": decision,
        "estimate": estimate.to_dict() if hasattr(estimate, "to_dict") else {},
    }
    if raw_extra:
        raw_payload.update(raw_extra)
    db.insert_market_audit(
        _market_audit_record(
            scan_run_id,
            market,
            stage=stage,
            outcome=outcome,
            decision=decision,
            estimated_yes_prob=getattr(estimate, "yes_probability", None),
            raw_extra=raw_payload,
        )
    )


def _market_audit_record(
    scan_run_id: str,
    market: Market,
    *,
    stage: str,
    outcome: str,
    skip_reason: Optional[str] = None,
    skip_reason_key: Optional[str] = None,
    decision: Optional[dict[str, Any]] = None,
    estimated_yes_prob: Optional[float] = None,
    raw_extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    decision = decision or {}
    raw_payload = market.to_dict() if hasattr(market, "to_dict") else {}
    if raw_extra:
        raw_payload.update(raw_extra)
    return {
        "scan_run_id": scan_run_id,
        "ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "title": market.title,
        "category": market.category,
        "status": market.status,
        "stage": stage,
        "outcome": outcome,
        "skip_reason": skip_reason,
        "skip_reason_key": skip_reason_key,
        "decision_action": decision.get("action"),
        "risk_summary": decision.get("risk_summary"),
        "edge_cents": _optional_float(decision.get("edge_cents")),
        "confidence": _optional_float(decision.get("confidence")),
        "yes_bid": market.yes_bid,
        "yes_ask": market.yes_ask,
        "no_bid": market.no_bid,
        "no_ask": market.no_ask,
        "spread_cents": market.yes_ask - market.yes_bid,
        "spread_pct": market.spread_pct,
        "volume": market.volume,
        "volume_24h": market.volume_24h,
        "liquidity_dollars": market.liquidity_dollars,
        "open_interest": market.open_interest,
        "minutes_to_close": market.minutes_to_close,
        "minutes_to_settlement": market.minutes_to_settlement,
        "orderbook_depth": market.orderbook_depth,
        "orderbook_depth_fetched": market.orderbook_depth_fetched,
        "estimated_yes_prob": estimated_yes_prob,
        "side": decision.get("side"),
        "limit_price_cents": decision.get("limit_price_cents"),
        "contracts": decision.get("contracts"),
        "dollar_size": decision.get("dollar_size"),
        "thesis": decision.get("thesis"),
        "raw_json": raw_payload,
    }


def _raw_cents(raw: dict[str, Any], key: str) -> int:
    if raw.get(f"{key}_dollars") is not None:
        return _dollars_to_cents(raw.get(f"{key}_dollars"))
    try:
        return int(round(float(raw.get(key) or 0)))
    except (TypeError, ValueError):
        return 0


def _raw_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(f"{key}_fp", raw.get(key, 0))
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
