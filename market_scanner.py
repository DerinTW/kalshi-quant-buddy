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
from decimal import Decimal, InvalidOperation
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
_CATEGORY_ALIASES: dict[str, str] = {
    "climate and weather": "weather",
    "climate": "weather",
    "weather": "weather",
    "economics": "economic",
    "economic": "economic",
    "financials": "financial",
    "financial": "financial",
    "finance": "financial",
    "crypto": "crypto",
    "cryptocurrency": "crypto",
    "commodities": "commodities",
    "commodity": "commodities",
    "science and technology": "science and technology",
    "science & technology": "science and technology",
    "science": "science and technology",
    "technology": "science and technology",
    "tech": "science and technology",
    "tech and science": "science and technology",
    "tech & science": "science and technology",
    "culture": "culture",
    "entertainment": "culture",
}
_SPORTS_TITLE_WORDS = (
    " nba ",
    " wnba ",
    " nfl ",
    " nhl ",
    " mlb ",
    " mls ",
    " ufc ",
    " atp ",
    " wta ",
    " tennis ",
    " soccer ",
    " basketball ",
    " football ",
    " baseball ",
    " hockey ",
    " golf ",
)


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


def _normalize_category(value: object) -> str:
    text = str(value or "").strip().lower()
    return _CATEGORY_ALIASES.get(text, text)


def _raw_event_key(raw: dict[str, Any]) -> str:
    event = str(raw.get("event_ticker") or raw.get("series_ticker") or "").strip()
    if event:
        return event
    ticker = str(raw.get("ticker") or "").strip()
    return _derive_event_ticker(ticker) if ticker else ""


def _title_looks_like_sports(raw: dict[str, Any]) -> bool:
    title = f" {raw.get('title') or raw.get('subtitle') or ''} ".lower()
    return any(word in title for word in _SPORTS_TITLE_WORDS)


def prefilter_raw_market(raw: dict[str, Any], cfg: Config) -> Optional[str]:
    """
    Cheap raw-market gate before normalization and enrichment.

    Uses fields already present in the market list response so blocked
    categories and obvious sports event families do not consume orderbook,
    history, research, or LLM budget.
    """
    event_key = _raw_event_key(raw).upper()
    ticker = str(raw.get("ticker") or "").strip().upper()
    status = _normalize_status(raw.get("status"))
    if status and status != "open":
        return f"prefilter_inactive_status={status}"

    missing_fields: list[str] = []
    if not (raw.get("title") or raw.get("subtitle")):
        missing_fields.append("title")
    if not (raw.get("rules_primary") or raw.get("description")):
        missing_fields.append("rules_primary")
    if not (raw.get("category") or raw.get("event_category")):
        missing_fields.append("category")
    if missing_fields:
        return "prefilter_missing_required_fields=" + ",".join(missing_fields)

    for prefix in getattr(cfg, "blocked_event_prefixes", []) or []:
        normalized_prefix = str(prefix or "").strip().upper()
        if not normalized_prefix:
            continue
        if event_key.startswith(normalized_prefix) or ticker.startswith(normalized_prefix):
            return f"prefilter_blocked_event_prefix={normalized_prefix}"

    raw_category = raw.get("category") or raw.get("event_category")
    category = _normalize_category(raw_category)
    if cfg.category_allowlist and category:
        allowed = {_normalize_category(item) for item in cfg.category_allowlist}
        if category not in allowed:
            return f"prefilter_category_not_allowed={category}"

    if not category and _title_looks_like_sports(raw):
        return "prefilter_category_not_allowed=sports_title_fallback"

    return None


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
    if market.yes_ask > 0 and market.no_ask > 0 and market.yes_bid >= 0 and market.no_bid >= 0:
        # yes_mid + no_mid should be ~100. Asks alone always exceed 100 by the spread.
        mid_sum = (market.yes_ask + market.yes_bid + market.no_ask + market.no_bid) / 2
        if mid_sum < 95 or mid_sum > 105:
            return _unsafe(market, f"price_sum_anomaly (mid_sum={mid_sum:.0f})")
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
    Depth = max contracts available at the best YES-bid / implied-YES-ask.
    Kalshi REST orderbooks return bids only: a NO bid is the executable YES
    ask. Sets market.orderbook_depth in-place. Failures leave the previous
    depth, keep orderbook_depth_fetched=False, and mark the market unsafe so
    filters drop it before downstream analysis.
    Call after cheap raw/structural screening and before any downstream
    analysis that depends on a trustworthy depth gate.
    """
    for i, market in enumerate(markets):
        try:
            ob_data = client.get_orderbook(market.ticker, depth=depth)
            market.orderbook_depth = _parse_orderbook_top_depth(ob_data)
            market.orderbook_depth_fetched = True
            market.fetched_at = datetime.now(timezone.utc)
        except Exception as exc:
            if not market.is_unsafe:
                market.is_unsafe = True
                market.unsafe_reason = f"orderbook_fetch_failed: {str(exc)[:200]}"
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


def _parse_orderbook_top_depth(raw: Any) -> int:
    """
    Parse Kalshi orderbook depth across current and legacy response shapes.

    Current REST shape:
      {"orderbook_fp": {"yes_dollars": [["0.4200", "13.00"]],
                        "no_dollars":  [["0.5600", "17.00"]]}}
    The arrays are bid levels sorted ascending, so the best bid is last. Since
    a NO bid is an executable YES ask, generic tradable depth is the larger of
    the best YES-bid and best NO-bid quantities.

    Legacy/test shapes with explicit yes/no bid/ask arrays are still accepted.
    If the payload has no recognizable orderbook container, raise so callers do
    not incorrectly mark depth as fetched.
    """
    if not isinstance(raw, dict):
        raise ValueError("orderbook_response_not_object")

    if isinstance(raw.get("orderbook_fp"), dict):
        ob_fp = raw["orderbook_fp"]
        return max(
            _best_level_depth(ob_fp.get("yes_dollars"), best_at_end=True),
            _best_level_depth(ob_fp.get("no_dollars"), best_at_end=True),
        )

    ob = raw.get("orderbook", raw)
    if not isinstance(ob, dict):
        raise ValueError("orderbook_container_not_object")

    yes = ob.get("yes")
    no = ob.get("no")
    if isinstance(yes, list) or isinstance(no, list):
        return max(
            _best_level_depth(yes, best_at_end=True),
            _best_level_depth(no, best_at_end=True),
        )

    if isinstance(yes, dict) or isinstance(no, dict):
        yes_dict = yes if isinstance(yes, dict) else {}
        no_dict = no if isinstance(no, dict) else {}
        return max(
            _best_level_depth(
                yes_dict.get("ask") or yes_dict.get("asks") or yes_dict.get("bid") or yes_dict.get("bids"),
                best_at_end=False,
            ),
            _best_level_depth(
                no_dict.get("bid") or no_dict.get("bids") or no_dict.get("ask") or no_dict.get("asks"),
                best_at_end=False,
            ),
        )

    if "orderbook" in raw:
        raise ValueError("orderbook_sides_unrecognized")
    raise ValueError("orderbook_container_missing")


def _best_level_depth(levels: Any, *, best_at_end: bool) -> int:
    if levels is None:
        return 0
    if not isinstance(levels, list):
        raise ValueError("orderbook_levels_not_list")
    if not levels:
        return 0

    best = levels[-1] if best_at_end else levels[0]
    price = _level_price(best)
    total = Decimal("0")
    for level in levels:
        if _level_price(level) == price:
            total += _level_size(level)
    return int(total)


def _level_price(level: Any) -> Decimal:
    if isinstance(level, dict):
        value = (
            level.get("price")
            or level.get("price_dollars")
            or level.get("yes_price")
            or level.get("no_price")
        )
    elif isinstance(level, (list, tuple)) and len(level) >= 1:
        value = level[0]
    else:
        raise ValueError("orderbook_level_unrecognized")
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("orderbook_price_unparseable") from exc


def _level_size(level: Any) -> Decimal:
    if isinstance(level, dict):
        value = (
            level.get("count")
            or level.get("count_fp")
            or level.get("size")
            or level.get("quantity")
            or level.get("contracts")
        )
    elif isinstance(level, (list, tuple)) and len(level) >= 2:
        value = level[1]
    else:
        raise ValueError("orderbook_level_unrecognized")
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("orderbook_size_unparseable") from exc


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
    prefilter_skipped = 0
    prefilter_reasons: dict[str, int] = {}
    unsafe_count = 0

    for raw in raw_list:
        ticker = raw.get("ticker", "").strip()
        if not ticker or ticker in seen:
            hard_skipped += 1
            continue
        seen.add(ticker)

        prefilter_reason = prefilter_raw_market(raw, cfg)
        if prefilter_reason is not None:
            prefilter_skipped += 1
            prefilter_reasons[prefilter_reason] = prefilter_reasons.get(prefilter_reason, 0) + 1
            logger.debug(
                _MODULE,
                "prefilter_market_skipped",
                ticker=ticker,
                event_ticker=_raw_event_key(raw),
                reason=prefilter_reason,
            )
            continue

        m = normalize(raw)
        if m is None:
            hard_skipped += 1
            continue

        if m.is_unsafe:
            unsafe_count += 1
            logger.debug(_MODULE, "unsafe_market",
                         ticker=m.ticker, reason=m.unsafe_reason)

        markets.append(m)

    # Orderbook depth is a hard prerequisite for the filter stage. Fetch it for
    # every market that survived raw prefiltering; failed fetches are marked
    # unsafe by enrich_with_orderbook_depth() and rejected by filters.
    enrich_with_orderbook_depth(markets, client, delay_seconds=0.0)

    unsafe_count = sum(1 for market in markets if market.is_unsafe)
    safe_count = len(markets) - unsafe_count
    logger.info(
        _MODULE, "scan_done",
        f"total={len(markets)}  safe={safe_count}  unsafe={unsafe_count}  skipped={hard_skipped}  prefilter_skipped={prefilter_skipped}",
        prefilter_skip_counts=prefilter_reasons,
    )
    if unsafe_count > 0:
        logger.warn(_MODULE, "unsafe_summary",
                    msg=f"{unsafe_count} markets will be rejected by filters (unsafe)")

    return markets
