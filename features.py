"""
Feature extraction for the probability model.

Single entry point:
    extract_features(market, sentiment, weird_move, *, spot_price=None,
                     markets_by_event=None) -> FeatureVector

Inputs are upstream-module outputs:
  market       (Market)            — from market_scanner, possibly enriched
  sentiment    (SentimentResult)   — from sentiment.analyze
  weird_move   (WeirdMoveSignal)   — from weird_move.evaluate

Optional context:
  spot_price        — underlying spot for distance_to_threshold
  markets_by_event  — dict[event_ticker → list[Market]] for siblings

Everything is fail-soft: missing inputs degrade specific features to safe
defaults; the function never raises on missing data.
"""
from __future__ import annotations
import math
import re
import statistics
from typing import Any, Optional

import logger
from models import FeatureVector, Market, SentimentResult, WeirdMoveSignal

_MODULE = "features"


# ── Strike parsing (for distance_to_threshold) ───────────────────────────────
# Local copy of the regex used in category_research — kept independent so
# the two modules can evolve separately.

_THRESHOLD_RE = re.compile(
    r"(above|over|under|below|at least|greater than|less than|>=|<=|>|<)\s*"
    r"\$?\s*([0-9][0-9,\.]*)\s*(k|thousand|m|million)?",
    re.IGNORECASE,
)


def _extract_strike(title: str) -> Optional[float]:
    m = _THRESHOLD_RE.search(title or "")
    if not m:
        return None
    num = m.group(2).replace(",", "")
    suffix = (m.group(3) or "").lower()
    try:
        value = float(num)
    except ValueError:
        return None
    if suffix in ("k", "thousand"):
        value *= 1_000
    elif suffix in ("m", "million"):
        value *= 1_000_000
    return value


# ── Historical volatility from price_history ─────────────────────────────────

def _extract_mid(entry: dict) -> Optional[float]:
    """
    Pull a YES mid-price (in cents) from a single Kalshi history entry.
    Kalshi's history shape varies; try several known field names.
    """
    if not isinstance(entry, dict):
        return None
    # Try direct fields first
    for key in ("yes_price", "yes_ask", "price", "mid"):
        v = entry.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    # Try yes_bid + yes_ask average
    a = entry.get("yes_ask")
    b = entry.get("yes_bid")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 0 and b >= 0:
        return (float(a) + float(b)) / 2.0
    return None


def _historical_volatility(price_history: list[dict]) -> float:
    """
    Standard deviation of consecutive yes-mid differences, in cents.
    Returns 0.0 when fewer than 3 usable observations are available.
    """
    if not price_history:
        return 0.0
    mids: list[float] = []
    for entry in price_history:
        m = _extract_mid(entry)
        if m is not None:
            mids.append(m)
    if len(mids) < 3:
        return 0.0
    diffs = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
    try:
        return float(statistics.pstdev(diffs))
    except statistics.StatisticsError:
        return 0.0


# ── Distance to threshold ────────────────────────────────────────────────────

def _distance_to_threshold(market: Market, spot_price: Optional[float]) -> Optional[float]:
    if spot_price is None or spot_price <= 0:
        return None
    strike = _extract_strike(market.title)
    if strike is None or strike <= 0:
        return None
    return abs(spot_price - strike) / strike


# ── Related-market prices (sibling strikes in same event group) ──────────────

def _related_market_prices(
    market: Market,
    markets_by_event: Optional[dict[str, list[Market]]],
) -> list[float]:
    """
    Returns implied YES probabilities (0.0-1.0) of every sibling market in
    the same event group, excluding self. Only includes siblings with a
    valid yes_ask (>0).
    """
    if not markets_by_event:
        return []
    event_key = market.event_ticker or market.ticker
    siblings = markets_by_event.get(event_key, [])
    out: list[float] = []
    for sib in siblings:
        if sib.ticker == market.ticker:
            continue
        if sib.yes_ask <= 0:
            continue
        # Use mid when both sides available, else just the ask
        if sib.yes_bid > 0:
            mid = (sib.yes_ask + sib.yes_bid) / 2.0
        else:
            mid = float(sib.yes_ask)
        out.append(round(mid / 100.0, 4))
    return out


# ── Helpers for fail-soft defaults ───────────────────────────────────────────

def _safe_mid(market: Market) -> float:
    if market.yes_bid > 0:
        return (market.yes_ask + market.yes_bid) / 2.0
    return float(market.yes_ask)


def _safe_spread(market: Market) -> int:
    if market.yes_ask <= 0:
        return 0
    if market.yes_bid <= 0:
        return market.yes_ask   # no bid → spread = full ask price (degenerate)
    return max(0, market.yes_ask - market.yes_bid)


# ── Main entry point ─────────────────────────────────────────────────────────

def extract_features(
    market: Market,
    sentiment: Optional[SentimentResult] = None,
    weird_move: Optional[WeirdMoveSignal] = None,
    *,
    spot_price: Optional[float] = None,
    markets_by_event: Optional[dict[str, list[Market]]] = None,
) -> FeatureVector:
    """
    Produce a FeatureVector from upstream outputs. Missing optional inputs
    degrade specific fields to safe defaults — never raises.
    """
    mid = _safe_mid(market)
    spread = _safe_spread(market)

    # Price dynamics (default to flat / no spike when weird_move missing)
    if weird_move is not None:
        price_change_5m  = float(weird_move.price_change_5m)
        price_change_15m = float(weird_move.price_change_15m)
        vol_spike        = float(weird_move.volume_ratio) if weird_move.volume_ratio > 0 else 1.0
    else:
        price_change_5m  = 0.0
        price_change_15m = 0.0
        vol_spike        = 1.0

    # Narrative (default to neutral when sentiment missing)
    if sentiment is not None:
        sentiment_score    = float(sentiment.sentiment_score)
        source_credibility = float(sentiment.source_credibility)
        event_relevance    = float(sentiment.event_relevance)
    else:
        sentiment_score = 0.0
        source_credibility = 0.0
        event_relevance = 0.0

    fv = FeatureVector(
        ticker=market.ticker,
        current_yes_bid=int(market.yes_bid),
        current_yes_ask=int(market.yes_ask),
        mid_price=mid,
        spread=spread,
        volume=int(market.volume_24h),
        open_interest=int(market.open_interest),
        order_book_depth=int(market.orderbook_depth),
        time_to_resolution_minutes=float(market.minutes_to_settlement
                                         or market.minutes_to_close),
        price_change_5m=price_change_5m,
        price_change_15m=price_change_15m,
        volume_spike_ratio=vol_spike,
        category=market.category or "",
        related_market_prices=_related_market_prices(market, markets_by_event),
        sentiment_score=sentiment_score,
        source_credibility=source_credibility,
        event_relevance=event_relevance,
        historical_volatility=_historical_volatility(market.price_history),
        distance_to_threshold=_distance_to_threshold(market, spot_price),
    )

    logger.debug(
        _MODULE, "extracted",
        ticker=market.ticker,
        mid=f"{mid:.1f}",
        spread=spread,
        vol=fv.volume,
        oi=fv.open_interest,
        depth=fv.order_book_depth,
        ttr=f"{fv.time_to_resolution_minutes:.0f}m",
        sent=f"{sentiment_score:+.2f}",
        siblings=len(fv.related_market_prices),
        hist_vol=f"{fv.historical_volatility:.2f}",
        dist=(f"{fv.distance_to_threshold:.3f}"
              if fv.distance_to_threshold is not None else "?"),
    )
    return fv


def batch_extract(
    markets: list[Market],
    sentiment_map: dict[str, SentimentResult],
    weird_move_map: dict[str, WeirdMoveSignal],
    *,
    spot_prices: Optional[dict[str, float]] = None,
    markets_by_event: Optional[dict[str, list[Market]]] = None,
) -> dict[str, FeatureVector]:
    """
    Extract features for many markets at once. Missing per-market sentiment
    or weird_move entries fall through to fail-soft defaults; nothing is
    skipped.
    """
    if markets_by_event is None:
        markets_by_event = _build_event_map(markets)

    spot_prices = spot_prices or {}
    out: dict[str, FeatureVector] = {}
    for m in markets:
        out[m.ticker] = extract_features(
            m,
            sentiment_map.get(m.ticker),
            weird_move_map.get(m.ticker),
            spot_price=spot_prices.get(m.ticker),
            markets_by_event=markets_by_event,
        )
    return out


def _build_event_map(markets: list[Market]) -> dict[str, list[Market]]:
    out: dict[str, list[Market]] = {}
    for m in markets:
        key = m.event_ticker or m.ticker
        out.setdefault(key, []).append(m)
    return out
