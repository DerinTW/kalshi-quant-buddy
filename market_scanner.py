"""
Agent A: Market Scanner

Fetches all active Kalshi markets, normalizes them into Market objects,
marks those with missing or unsafe data, and optionally enriches a subset
with recent trade/price history.

Unsafe markets are NOT dropped here — they are passed through with
is_unsafe=True so the filter stage can log and count them explicitly.
Only completely unparseable records (no ticker) are dropped silently.
"""
from __future__ import annotations
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import logger
from config import Config
from kalshi_client import KalshiClient
from models import Market

_MODULE = "market_scanner"

_UNKNOWN_TIME = datetime(9999, 12, 31, tzinfo=timezone.utc)
_STALE_PRICE_HOURS = 6
_EVENT_SUFFIX = re.compile(r"^[BTbt]?\d")


# ── Event ticker derivation ───────────────────────────────────────────────────

def _derive_event_ticker(ticker: str) -> str:
    """
    Extract the event/series key from a Kalshi ticker.
    e.g. KXBTCD-24DEC31-B50000  →  KXBTCD-24DEC31
         HIGHNY25                →  HIGHNY25  (no price suffix, kept as-is)
    """
    parts = ticker.split("-")
    if len(parts) >= 2 and _EVENT_SUFFIX.match(parts[-1]):
        return "-".join(parts[:-1])
    return ticker


# ── Time parsing ──────────────────────────────────────────────────────────────

def _parse_time(raw: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp. Returns None if unparseable."""
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _cents(raw: Any, dollars_raw: Any = None) -> int:
    if raw not in (None, ""):
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            pass
    if dollars_raw in (None, ""):
        return 0
    try:
        return int(round(float(dollars_raw) * 100))
    except (TypeError, ValueError):
        return 0


def _whole_number(*values: Any) -> int:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return 0


def _normalize_status(raw: Any) -> str:
    status = str(raw or "").strip().lower()
    if status == "active":
        return "open"
    return status


# ── Safety validation ─────────────────────────────────────────────────────────

def _unsafe(market: Market, reason: str) -> Market:
    market.is_unsafe = True
    market.unsafe_reason = reason
    return market


def _validate(market: Market) -> Market:
    """
    Check every field downstream modules depend on.
    Marks unsafe (does NOT raise) so caller always gets a Market back.
    """
    if not market.title:
        return _unsafe(market, "missing_title")
    if not market.rules_primary:
        return _unsafe(market, "missing_rules_primary")
    if market.close_time == _UNKNOWN_TIME:
        return _unsafe(market, "unknown_close_time")
    if market.minutes_to_close <= 0:
        return _unsafe(market, f"already_closed ({market.minutes_to_close:.0f}min)")
    has_yes = market.yes_ask > 0 and market.yes_bid >= 0
    has_no = market.no_ask > 0 and market.no_bid >= 0
    if not has_yes and not has_no:
        return _unsafe(market, "no_usable_bid_ask")
    if market.yes_ask > 0 and market.no_ask > 0:
        price_sum = market.yes_ask + market.no_ask
        if price_sum < 95 or price_sum > 105:
            return _unsafe(market, f"price_sum_anomaly (yes+no={price_sum})")
    if not market.category:
        return _unsafe(market, "missing_category")
    return market


# ── Derived fields ────────────────────────────────────────────────────────────

def _compute_derived(market: Market) -> Market:
    now = datetime.now(timezone.utc)
    market.minutes_to_close = max(0.0, (market.close_time - now).total_seconds() / 60)
    market.minutes_to_settlement = max(0.0, (market.settlement_time - now).total_seconds() / 60)

    if market.yes_ask > 0 and market.yes_bid >= 0:
        mid = (market.yes_ask + market.yes_bid) / 2 if market.yes_bid > 0 else market.yes_ask
        market.spread_pct = ((market.yes_ask - market.yes_bid) / mid * 100) if mid > 0 else 999.0
    else:
        market.spread_pct = 999.0

    if market.open_interest > 0:
        mid_frac = ((market.yes_ask + market.yes_bid) / 2) / 100.0 if market.yes_bid > 0 else market.yes_ask / 100.0
        market.liquidity_dollars = market.open_interest * mid_frac
    else:
        market.liquidity_dollars = 0.0

    return market


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize(raw: dict) -> Optional[Market]:
    """
    Convert a raw Kalshi API dict into a Market.
    Returns None only for records with no ticker.
    Markets with bad data come back with is_unsafe=True.
    """
    ticker = raw.get("ticker", "").strip()
    if not ticker:
        return None

    try:
        yes_ask = _cents(raw.get("yes_ask"), raw.get("yes_ask_dollars"))
        yes_bid = _cents(raw.get("yes_bid"), raw.get("yes_bid_dollars"))
        no_ask  = _cents(raw.get("no_ask"),  raw.get("no_ask_dollars"))
        no_bid  = _cents(raw.get("no_bid"),  raw.get("no_bid_dollars"))

        if yes_ask == 0 and no_bid > 0:
            yes_ask = 100 - no_bid
        if no_ask == 0 and yes_bid > 0:
            no_ask = 100 - yes_bid
        if yes_bid == 0 and no_ask > 0:
            yes_bid = 100 - no_ask
        if no_bid == 0 and yes_ask > 0:
            no_bid = 100 - yes_ask

        close_raw   = raw.get("close_time") or raw.get("expiration_time")
        settle_raw  = raw.get("settlement_time") or raw.get("expected_expiration_time") or close_raw
        close_time  = _parse_time(close_raw) or _UNKNOWN_TIME
        settle_time = _parse_time(settle_raw) or close_time

        # last_trade_at: try several field names the Kalshi API uses
        last_trade_at = (
            _parse_time(raw.get("last_trade_time"))
            or _parse_time(raw.get("last_trade_ts"))
            or _parse_time(raw.get("last_updated"))
        )

        market = Market(
            ticker=ticker,
            title=(raw.get("title") or raw.get("subtitle") or "").strip(),
            status=_normalize_status(raw.get("status")),
            yes_ask=yes_ask,
            yes_bid=yes_bid,
            no_ask=no_ask,
            no_bid=no_bid,
            volume=_whole_number(raw.get("volume"), raw.get("volume_fp")),
            volume_24h=_whole_number(
                raw.get("volume_24h"),
                raw.get("daily_volume"),
                raw.get("volume_24h_fp"),
            ),
            open_interest=_whole_number(raw.get("open_interest"), raw.get("open_interest_fp")),
            close_time=close_time,
            settlement_time=settle_time,
            category=(raw.get("category") or raw.get("event_category") or "").strip(),
            rules_primary=(raw.get("rules_primary") or raw.get("description") or "").strip(),
            result=raw.get("result"),
            event_ticker=(raw.get("event_ticker") or _derive_event_ticker(ticker)).strip(),
            fetched_at=datetime.now(timezone.utc),
            last_trade_at=last_trade_at,
        )

        _compute_derived(market)
        _validate(market)
        return market

    except Exception as exc:
        logger.warn(_MODULE, "normalize_exception", ticker=ticker, err=str(exc))
        fallback = Market(
            ticker=ticker,
            title="",
            status="",
            yes_ask=0, yes_bid=0, no_ask=0, no_bid=0,
            volume=0, volume_24h=0, open_interest=0,
            close_time=_UNKNOWN_TIME,
            settlement_time=_UNKNOWN_TIME,
            category="",
            rules_primary="",
            event_ticker=ticker,
        )
        return _unsafe(fallback, f"parse_exception: {exc}")


# ── Orderbook depth enrichment (post-filter, before depth check) ─────────────

def enrich_with_orderbook_depth(
    markets: list[Market],
    client: KalshiClient,
    depth: int = 10,
    delay_seconds: float = 0.15,
) -> None:
    """
    Fetch the live orderbook for each market and compute top-of-book depth.
    Depth = total contracts available at the single best ask price (YES side).
    Sets market.orderbook_depth in-place. Best-effort — failures leave depth=0.
    Call AFTER the structural filter pass so we only hit the API for candidates.
    """
    for i, market in enumerate(markets):
        try:
            ob_data = client.get_orderbook(market.ticker, depth=depth)
            ob = ob_data.get("orderbook", ob_data)
            asks = ob.get("yes", {}).get("ask", []) or ob.get("yes", {}).get("asks", [])
            if asks:
                best_price = asks[0][0]
                market.orderbook_depth = sum(
                    int(size) for price, size in asks if price == best_price
                )
            else:
                market.orderbook_depth = 0
        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429:
                logger.warn(_MODULE, "rate_limited_orderbook", ticker=market.ticker,
                            msg="429 — backing off 5s")
                time.sleep(5.0)
            else:
                logger.warn(_MODULE, "orderbook_fetch_failed",
                            ticker=market.ticker, err=str(exc))

        if delay_seconds > 0 and i < len(markets) - 1:
            time.sleep(delay_seconds)


# ── History enrichment (post-filter, not during bulk scan) ────────────────────

def enrich_with_history(
    markets: list[Market],
    client: KalshiClient,
    limit: int = 50,
    delay_seconds: float = 0.15,
) -> None:
    """
    Fetch recent trade history for each market and attach to price_history.
    Also updates last_trade_at if more precise data is found.
    Modifies in-place. Best-effort — failures are logged and skipped.
    Only call this AFTER filtering so we're not making 200+ requests per scan.
    """
    for i, market in enumerate(markets):
        try:
            data = client.get_market_history(market.ticker, limit=limit)
            history = data.get("history", [])
            market.price_history = history

            if history:
                last_raw = history[0].get("created_time") or history[0].get("ts") or ""
                last_dt = _parse_time(last_raw)
                if last_dt:
                    market.last_trade_at = last_dt
                    age_hours = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
                    if age_hours > _STALE_PRICE_HOURS:
                        logger.warn(_MODULE, "stale_price_data",
                                    ticker=market.ticker,
                                    last_trade_hours_ago=f"{age_hours:.1f}h")
            else:
                logger.warn(_MODULE, "no_trade_history", ticker=market.ticker)

        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429:
                logger.warn(_MODULE, "rate_limited", ticker=market.ticker,
                            msg="429 — backing off 5s")
                time.sleep(5.0)
            elif code == 404:
                logger.debug(_MODULE, "no_trade_history_endpoint", ticker=market.ticker)
            else:
                logger.warn(_MODULE, "history_fetch_failed",
                            ticker=market.ticker, err=str(exc))

        if delay_seconds > 0 and i < len(markets) - 1:
            time.sleep(delay_seconds)


# ── Main entry point ──────────────────────────────────────────────────────────

def scan(client: KalshiClient, cfg: Config, status: str = "open") -> list[Market]:
    """
    Fetch all active markets, normalize, and return.
    Returns ALL markets including unsafe ones.
    The filter stage handles unsafe rejection and logging.
    """
    categories = list(cfg.category_allowlist)
    fetch_categories = categories if categories else [None]
    max_markets = getattr(cfg, "max_raw_markets_per_scan", None)
    logger.info(
        _MODULE,
        "scan_start",
        "fetching markets from Kalshi API",
        category_count=len(categories),
        max_markets=max_markets,
    )

    raw_list: list[dict] = []
    for cat in fetch_categories:
        try:
            remaining = None
            if max_markets is not None:
                remaining = max(0, int(max_markets) - len(raw_list))
                if remaining <= 0:
                    break
            batch = client.get_all_markets(
                status=status,
                category=cat,
                max_markets=remaining,
            )
            raw_list.extend(batch)
            logger.info(
                _MODULE,
                "category_fetched",
                category=cat or "all",
                count=len(batch),
                total_count=len(raw_list),
            )
        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429:
                logger.error(_MODULE, "rate_limited_on_scan",
                             msg="rate limit hit during market fetch — try again later")
            else:
                logger.error(_MODULE, "api_error",
                             category=cat or "all", err=str(exc))

    if categories and not raw_list:
        logger.warn(
            _MODULE,
            "category_fetch_empty_fallback_all_markets",
            "category-specific fetches returned no markets; fetching all open markets once",
            category_count=len(categories),
        )
        try:
            batch = client.get_all_markets(
                status=status,
                category=None,
                max_markets=max_markets,
            )
            raw_list.extend(batch)
            logger.info(
                _MODULE,
                "category_fetched",
                category="all",
                count=len(batch),
                fallback=True,
            )
        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429:
                logger.error(_MODULE, "rate_limited_on_scan",
                             msg="rate limit hit during market fetch — try again later")
            else:
                logger.error(_MODULE, "api_error",
                             category="all", err=str(exc), fallback=True)

    if not raw_list:
        logger.error(_MODULE, "no_markets_returned",
                     msg="Kalshi returned 0 markets — check API key and network")
        return []

    seen: set[str] = set()
    markets: list[Market] = []
    hard_skipped = 0
    unsafe_count = 0

    for raw in raw_list:
        ticker = raw.get("ticker", "").strip()
        if not ticker or ticker in seen:
            hard_skipped += 1
            continue
        seen.add(ticker)

        m = normalize(raw)
        if m is None:
            hard_skipped += 1
            continue

        if m.is_unsafe:
            unsafe_count += 1
            logger.debug(_MODULE, "unsafe_market",
                         ticker=m.ticker, reason=m.unsafe_reason)

        markets.append(m)

    safe_count = len(markets) - unsafe_count
    logger.info(
        _MODULE, "scan_done",
        f"total={len(markets)}  safe={safe_count}  unsafe={unsafe_count}  skipped={hard_skipped}",
    )
    if unsafe_count > 0:
        logger.warn(_MODULE, "unsafe_summary",
                    msg=f"{unsafe_count} markets will be rejected by filters (unsafe)")

    return markets
