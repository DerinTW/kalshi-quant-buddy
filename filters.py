"""
Agent B: Filter Pipeline

Removes structurally bad markets before any LLM or research budget is spent.
All 12 rejection criteria run in a fixed priority order.
Deduplication (one market per event group) runs as a final pass.

Returns a FilterResult with:
  - accepted markets
  - every rejection with its reason
  - per-reason counts and up to 3 example tickers for each reason
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import logger
from config import Config
from models import Market

_MODULE = "filters"

# Max example tickers to show per rejection reason in the summary log
_MAX_EXAMPLES = 3

# Category alias table — Kalshi returns human-friendly category strings
# ("Crypto", "Climate and Weather", "Economics", "Financials"); the rest of
# the project, the env var CATEGORY_ALLOWLIST, and downstream code all use
# the short lowercase form. Normalise both sides through this map before
# comparing so the filter is not silently rejecting every market.
_CATEGORY_ALIASES: dict[str, str] = {
    "climate and weather": "weather",
    "climate":             "weather",
    "weather":             "weather",
    "economics":           "economic",
    "economic":            "economic",
    "financials":          "financial",
    "financial":           "financial",
    "crypto":              "crypto",
    "cryptocurrency":      "crypto",
    "politics":            "politics",
    "sports":              "sports",
}


def _normalize_category(value: object) -> str:
    """Lowercase + alias-collapse so 'Climate and Weather' == 'weather'."""
    text = str(value or "").strip().lower()
    return _CATEGORY_ALIASES.get(text, text)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    passed: list[Market]
    rejected: list[tuple[Market, str]]   # (market, full reason string)

    @property
    def pass_rate(self) -> float:
        total = len(self.passed) + len(self.rejected)
        return len(self.passed) / total if total else 0.0

    @property
    def skip_reason_counts(self) -> dict[str, int]:
        """Count of rejections per reason category, sorted most-to-least."""
        counts: dict[str, int] = defaultdict(int)
        for _, reason in self.rejected:
            counts[_reason_key(reason)] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    @property
    def skip_reason_examples(self) -> dict[str, list[str]]:
        """Up to _MAX_EXAMPLES tickers per reason category."""
        examples: dict[str, list[str]] = defaultdict(list)
        for market, reason in self.rejected:
            key = _reason_key(reason)
            if len(examples[key]) < _MAX_EXAMPLES:
                examples[key].append(market.ticker)
        return dict(examples)


def _reason_key(reason: str) -> str:
    """Normalise a full reason string to a stable category key for grouping."""
    # Take everything up to the first '(' or '=' to strip the dynamic value
    for stop in ("(", "=", ":"):
        idx = reason.find(stop)
        if idx > 0:
            return reason[:idx].strip()
    return reason.strip()


# ── Individual checks (return str on fail, None on pass) ─────────────────────

def _check_status(m: Market, cfg: Config) -> Optional[str]:
    if m.status != "open":
        return f"status={m.status}"
    return None


def _check_unsafe(m: Market, cfg: Config) -> Optional[str]:
    if m.is_unsafe:
        return f"unsafe: {m.unsafe_reason}"
    return None


def _check_blocked(m: Market, cfg: Config) -> Optional[str]:
    if m.ticker in cfg.blocked_tickers:
        return "blocked_ticker"
    return None


def _check_category(m: Market, cfg: Config) -> Optional[str]:
    if not cfg.category_allowlist:
        return None
    allow = {_normalize_category(c) for c in cfg.category_allowlist}
    if _normalize_category(m.category) not in allow:
        return f"category={m.category!r} not in allowlist"
    return None


def _check_time_to_close(m: Market, cfg: Config) -> Optional[str]:
    if m.minutes_to_close < cfg.min_minutes_to_expiry:
        return f"too_close_to_close ({m.minutes_to_close:.0f}min < {cfg.min_minutes_to_expiry}min)"
    if m.minutes_to_close > cfg.max_minutes_to_expiry:
        return f"too_far_to_close ({m.minutes_to_close:.0f}min > {cfg.max_minutes_to_expiry}min)"
    return None


def _check_price(m: Market, cfg: Config) -> Optional[str]:
    if m.yes_ask <= 0:
        return "no_valid_price"
    if m.yes_ask < cfg.min_yes_price:
        return f"yes_ask={m.yes_ask}¢ < min {cfg.min_yes_price}¢"
    if m.yes_ask > cfg.max_yes_price:
        return f"yes_ask={m.yes_ask}¢ > max {cfg.max_yes_price}¢"
    return None


def _check_spread(m: Market, cfg: Config) -> Optional[str]:
    spread_cents = m.yes_ask - m.yes_bid
    if spread_cents > cfg.max_spread_cents:
        return f"spread={spread_cents}¢ > max {cfg.max_spread_cents}¢"
    if m.spread_pct > cfg.max_spread_pct:
        return f"spread={m.spread_pct:.1f}% > max {cfg.max_spread_pct}%"
    return None


def _check_volume(m: Market, cfg: Config) -> Optional[str]:
    if m.volume_24h < cfg.min_volume_24h:
        return f"volume_24h={m.volume_24h} < min {cfg.min_volume_24h}"
    return None


def _check_liquidity(m: Market, cfg: Config) -> Optional[str]:
    if m.liquidity_dollars < cfg.min_liquidity_dollars:
        return f"liquidity=${m.liquidity_dollars:.0f} < min ${cfg.min_liquidity_dollars:.0f}"
    return None


def _check_orderbook_age(m: Market, cfg: Config) -> Optional[str]:
    """
    Reject if our fetched market/orderbook snapshot is stale.

    This checks freshness of our own fetched market/orderbook snapshot. This is
    not a last-trade activity check; last_trade_at remains for trade-history /
    weird-move logic.

    Skipped (passes) when fetched_at is None to preserve compatibility with
    hand-built tests, fallback markets, and legacy objects.
    """
    if m.fetched_at is None:
        return None

    fetched_at = m.fetched_at
    if fetched_at.tzinfo is None or fetched_at.utcoffset() is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)

    age_seconds = (datetime.now(timezone.utc) - fetched_at).total_seconds()
    if age_seconds > cfg.max_orderbook_age_seconds:
        age_min = age_seconds / 60
        max_age_min = cfg.max_orderbook_age_seconds / 60
        return f"snapshot_stale ({age_min:.0f}min > {max_age_min:.0f}min)"
    return None


def _check_orderbook_depth(m: Market, cfg: Config) -> Optional[str]:
    """
    Reject if top-of-book depth is below MIN_ORDERBOOK_DEPTH_AT_LIMIT.

    Two distinct states matter:
      * orderbook_depth_fetched == False  → enrichment was never attempted
        (e.g. this market wasn't in the candidate slice). Skip the check;
        the upstream pipeline is responsible for enriching candidates.
      * orderbook_depth_fetched == True   → we asked the API and got an
        answer. Now `orderbook_depth == 0` means an empty book — that is a
        rejection, not a free pass.
    """
    if not m.orderbook_depth_fetched:
        return None
    if m.orderbook_depth < cfg.min_orderbook_depth_at_limit:
        return f"depth={m.orderbook_depth} < min {cfg.min_orderbook_depth_at_limit}"
    return None


# Ordered list of single-market checks — runs top to bottom, stops at first failure
_SINGLE_MARKET_CHECKS = [
    _check_status,
    _check_unsafe,
    _check_blocked,
    _check_category,
    _check_time_to_close,
    _check_price,
    _check_spread,
    _check_volume,
    _check_liquidity,
    _check_orderbook_age,
    _check_orderbook_depth,
]


def _evaluate(m: Market, cfg: Config) -> Optional[str]:
    for check in _SINGLE_MARKET_CHECKS:
        reason = check(m, cfg)
        if reason is not None:
            return reason
    return None


# ── Deduplication pass ────────────────────────────────────────────────────────

def _deduplicate(
    markets: list[Market],
    rejected: list[tuple[Market, str]],
) -> list[Market]:
    """
    Within each event group (same event_ticker), keep only the market with the
    highest volume_24h. Reject the others with "duplicate_event_group".

    Markets with a unique event_ticker pass through unchanged.
    """
    by_event: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        by_event[m.event_ticker].append(m)

    accepted: list[Market] = []
    for event_key, group in by_event.items():
        if len(group) == 1:
            accepted.append(group[0])
            continue

        # Pick winner: highest volume_24h, break ties by tightest spread
        winner = max(group, key=lambda m: (m.volume_24h, -m.spread_pct))
        accepted.append(winner)

        for m in group:
            if m.ticker != winner.ticker:
                reason = f"duplicate_event_group (kept={winner.ticker})"
                rejected.append((m, reason))
                logger.debug(_MODULE, "dedup_rejected",
                             ticker=m.ticker, kept=winner.ticker,
                             event_ticker=event_key)

    return accepted


# ── Main entry point ──────────────────────────────────────────────────────────

def run(markets: list[Market], cfg: Config) -> FilterResult:
    """
    Run the full filter pipeline and return a FilterResult.

    Pass 1: per-market structural checks (10 criteria, in order).
    Pass 2: deduplication — one market per event group.
    """
    if not markets:
        return FilterResult(passed=[], rejected=[])

    passed_p1: list[Market] = []
    rejected: list[tuple[Market, str]] = []

    for m in markets:
        reason = _evaluate(m, cfg)
        if reason is None:
            passed_p1.append(m)
        else:
            rejected.append((m, reason))

    # Deduplication runs only on markets that passed structural checks
    passed_final = _deduplicate(passed_p1, rejected)

    result = FilterResult(passed=passed_final, rejected=rejected)
    _log_summary(result, len(markets))
    return result


def _log_summary(result: FilterResult, total: int) -> None:
    counts = result.skip_reason_counts
    examples = result.skip_reason_examples

    logger.info(
        _MODULE, "filter_summary",
        f"passed={len(result.passed)}  rejected={len(result.rejected)}  "
        f"total={total}  pass_rate={result.pass_rate:.0%}",
        passed=len(result.passed),
        rejected=len(result.rejected),
        total=total,
        pass_rate=round(result.pass_rate, 4),
    )

    if counts:
        logger.info(_MODULE, "rejection_breakdown",
                    msg="skip reason counts (most common first):",
                    counts=counts,
                    examples=examples)
        for reason_key, tickers in examples.items():
            logger.debug(
                _MODULE, "rejection_examples",
                reason=reason_key,
                examples=tickers,
            )
    elif total > 0 and len(result.passed) == 0:
        # Pipeline returned zero candidates with zero rejections — usually
        # means filters never ran or the input list was already empty
        # post-normalisation. Surface it loudly.
        logger.warn(_MODULE, "filter_zero_pass_zero_reject",
                    msg="filters returned 0 passed and 0 rejected — "
                        "check normalisation/category aliasing upstream",
                    total=total)
