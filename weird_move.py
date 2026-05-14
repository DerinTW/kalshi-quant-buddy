"""
Agent C: Weird Price-Move Detector

Identifies markets worth deeper research because price/volume behaviour is unusual.
Maintains a rolling in-memory snapshot store so it can compute genuine time-series
metrics (5m/15m price change, 10m volume ratio, 1h spread median).

Each call to batch_detect():
  1. Takes a new snapshot of every market (updating the historical record).
  2. Detects signals using the snapshot history + attached price_history.
  3. Classifies each signal into one of 5 types.

Signal types:
  news_confirmed_move        — price moved with volume confirmation
  rumor_move                 — price moved without volume confirmation
  liquidity_gap_move         — spread widened or thin book caused a jump
  related_market_dislocation — sibling markets disagree significantly
  stale_book_artifact        — spread moved but no actual trades (bad signal)
"""
from __future__ import annotations
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import logger
from models import Market, MarketSnapshot, WeirdMoveSignal

_MODULE = "weird_move"

# ── Spec thresholds ───────────────────────────────────────────────────────────
PRICE_5M_THRESHOLD        = 8    # cents
PRICE_15M_THRESHOLD       = 12   # cents
VOLUME_RATIO_THRESHOLD    = 3.0
SPREAD_CHANGE_THRESHOLD   = 2.0
RELATED_DISAGREE_THRESHOLD = 10  # cents

# Classification support thresholds (not spec-defined, internal tuning)
_NEWS_VOLUME_MIN_RATIO    = 2.0   # volume_ratio must exceed this to call it "confirmed"
_THIN_BOOK_OI_THRESHOLD   = 100   # open_interest below this = thin book
_STALE_BOOK_MAX_VOLUME    = 2     # volume_last_10m <= this with spread_change = stale artifact


# ── Snapshot store ────────────────────────────────────────────────────────────

class SnapshotStore:
    """
    In-memory rolling window of MarketSnapshots per ticker.
    Retains up to max_age_hours of history; older entries are pruned on write.
    Module-level singleton — persists for the lifetime of the process.
    """

    def __init__(self, max_age_hours: int = 2) -> None:
        self._data: dict[str, list[MarketSnapshot]] = defaultdict(list)
        self._max_age = timedelta(hours=max_age_hours)

    def add(self, snap: MarketSnapshot) -> None:
        snap.timestamp = _aware_utc(snap.timestamp)
        self._data[snap.ticker].append(snap)
        self._data[snap.ticker].sort(key=lambda s: s.timestamp)
        self._prune(snap.ticker)

    def _prune(self, ticker: str) -> None:
        cutoff = datetime.now(timezone.utc) - self._max_age
        self._data[ticker] = [s for s in self._data[ticker] if _aware_utc(s.timestamp) >= cutoff]

    def get_at_or_before(self, ticker: str, target: datetime) -> Optional[MarketSnapshot]:
        """Most recent snapshot at or before `target`."""
        target = _aware_utc(target)
        candidates = [s for s in self._data.get(ticker, []) if _aware_utc(s.timestamp) <= target]
        return max(candidates, key=lambda s: _aware_utc(s.timestamp)) if candidates else None

    def spread_history_1h(self, ticker: str) -> list[int]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        return [s.spread_cents for s in self._data.get(ticker, []) if _aware_utc(s.timestamp) >= cutoff]

    def has_history_older_than(self, ticker: str, minutes: int) -> bool:
        snaps = self._data.get(ticker, [])
        if not snaps:
            return False
        oldest = min(_aware_utc(s.timestamp) for s in snaps)
        return (datetime.now(timezone.utc) - oldest).total_seconds() / 60 >= minutes


# Module-level singleton — shared across all calls in a single process run
_store = SnapshotStore()


def get_store() -> SnapshotStore:
    return _store


# ── Snapshot helpers ──────────────────────────────────────────────────────────

def _mid(market: Market) -> float:
    if market.yes_bid > 0:
        return (market.yes_ask + market.yes_bid) / 2.0
    return float(market.yes_ask)


def _spread_cents(market: Market) -> int:
    return max(0, market.yes_ask - market.yes_bid)


def take_snapshot(market: Market) -> None:
    """Record the current state of a market into the snapshot store."""
    snap = MarketSnapshot(
        ticker=market.ticker,
        mid_yes=_mid(market),
        spread_cents=_spread_cents(market),
        volume=market.volume,
        timestamp=datetime.now(timezone.utc),
    )
    _store.add(snap)


# ── Metric computers ──────────────────────────────────────────────────────────

def _price_change_n_minutes(ticker: str, minutes: int, current_mid: float) -> float:
    """
    Returns current_mid minus the mid price N minutes ago.

    A snapshot must be close to the requested lookback. Without this tolerance,
    a 15-minute-old snapshot could incorrectly trigger the 5-minute detector on
    a sparse first run.
    """
    target = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    snap = _store.get_at_or_before(ticker, target)
    if snap is None:
        return 0.0

    max_age_from_target = timedelta(minutes=2)
    if abs(_aware_utc(snap.timestamp) - target) > max_age_from_target:
        return 0.0

    return current_mid - snap.mid_yes


def _volume_last_10m(price_history: list[dict]) -> int:
    """Count contracts traded in the last 10 minutes using attached price history.

    Kalshi-style history payloads have used several field names for trade size
    across examples/endpoints, so accept the common variants instead of only
    ``count``. Unknown/malformed sizes contribute 0 rather than crashing the
    detector.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
    total = 0
    for trade in price_history:
        ts_raw = trade.get("created_time") or trade.get("created_ts") or trade.get("ts") or trade.get("timestamp") or ""
        ts = _parse_ts(ts_raw)
        if ts and ts >= cutoff:
            total += _trade_size(trade)
    return total


def _avg_volume_10m(market: Market) -> float:
    """Baseline expected volume per 10-min window from 24h data."""
    return max(1.0, market.volume_24h / 144.0)  # 144 = 24h / 10min


def _spread_change_ratio(market: Market) -> float:
    """current_spread / median_spread_last_1h.  Returns 1.0 when no history."""
    history = _store.spread_history_1h(market.ticker)
    if len(history) < 3:
        return 1.0
    median = statistics.median(history)
    if median <= 0:
        return 1.0
    return _spread_cents(market) / median


def _related_disagreement(market: Market, all_markets: list[Market]) -> float:
    """
    Max abs difference between this market's implied YES prob and any sibling
    market's implied YES prob.  Returns 0 if no siblings exist.
    Sibling = same event_ticker, different ticker.
    """
    this_mid = _mid(market)
    max_diff = 0.0
    for other in all_markets:
        if other.ticker == market.ticker:
            continue
        if other.event_ticker != market.event_ticker:
            continue
        other_mid = _mid(other)
        diff = abs(this_mid - other_mid)
        if diff > max_diff:
            max_diff = diff
    return max_diff


# ── Classification ────────────────────────────────────────────────────────────

def _classify(
    triggers: list[str],
    volume_ratio: float,
    spread_change: float,
    volume_last_10m: int,
    market: Market,
) -> tuple[str, str]:
    """
    Returns (classification, confidence).

    Priority order:
      1. stale_book_artifact  — spread widened but zero actual trades
      2. related_market_dislocation
      3. liquidity_gap_move   — spread spike or thin open interest
      4. news_confirmed_move  — price + volume together
      5. rumor_move           — price without volume
    """
    has_price  = "price_5m" in triggers or "price_15m" in triggers
    has_volume = "volume_ratio" in triggers
    has_spread = "spread_change" in triggers
    has_related = "related_disagreement" in triggers

    # 1. Stale book: spread widened but nobody is actually trading
    if has_spread and volume_last_10m <= _STALE_BOOK_MAX_VOLUME:
        return "stale_book_artifact", "low"

    # 2. Related market disagreement
    if has_related:
        return "related_market_dislocation", "medium"

    # 3. Thin book / liquidity gap
    thin_book = market.open_interest < _THIN_BOOK_OI_THRESHOLD
    if has_spread or (has_price and thin_book):
        return "liquidity_gap_move", "low"

    # 4. News-confirmed: price moved AND strong volume
    if has_price and volume_ratio >= _NEWS_VOLUME_MIN_RATIO:
        return "news_confirmed_move", "high"

    # 5. Rumor / unconfirmed
    if has_price or has_volume:
        return "rumor_move", "medium"

    return "none", "low"


# ── Core detector ─────────────────────────────────────────────────────────────

def detect(market: Market, all_markets: list[Market]) -> WeirdMoveSignal:
    """
    Run all 5 metrics and classify the result.
    Caller must have already called take_snapshot(market) this cycle.
    """
    now = datetime.now(timezone.utc)
    current_mid = _mid(market)

    # ── Compute all metrics ───────────────────────────────────────────────────
    pc_5m  = _price_change_n_minutes(market.ticker, 5,  current_mid)
    pc_15m = _price_change_n_minutes(market.ticker, 15, current_mid)

    vol_10m  = _volume_last_10m(market.price_history)
    avg_vol  = _avg_volume_10m(market)
    vol_ratio = vol_10m / avg_vol

    sp_change   = _spread_change_ratio(market)
    related_dis = _related_disagreement(market, all_markets)

    # ── Evaluate thresholds ───────────────────────────────────────────────────
    triggers: list[str] = []
    if abs(pc_5m)  >= PRICE_5M_THRESHOLD:
        triggers.append("price_5m")
    if abs(pc_15m) >= PRICE_15M_THRESHOLD:
        triggers.append("price_15m")
    if vol_ratio   >= VOLUME_RATIO_THRESHOLD:
        triggers.append("volume_ratio")
    if sp_change   >= SPREAD_CHANGE_THRESHOLD:
        triggers.append("spread_change")
    if related_dis >= RELATED_DISAGREE_THRESHOLD:
        triggers.append("related_disagreement")

    flagged = bool(triggers)

    if not flagged:
        return WeirdMoveSignal(
            ticker=market.ticker,
            flagged=False,
            classification="none",
            price_change_5m=pc_5m,
            price_change_15m=pc_15m,
            volume_ratio=vol_ratio,
            spread_change=sp_change,
            related_disagreement=related_dis,
            triggers=[],
            description="No unusual movement detected",
            confidence="low",
            timestamp=now,
        )

    classification, confidence = _classify(
        triggers, vol_ratio, sp_change, vol_10m, market
    )

    description = _build_description(
        market, classification, pc_5m, pc_15m,
        vol_ratio, sp_change, related_dis, triggers
    )

    logger.info(
        _MODULE, "signal_detected",
        ticker=market.ticker,
        classification=classification,
        confidence=confidence,
        triggers=triggers,
        pc_5m=f"{pc_5m:+.1f}¢",
        pc_15m=f"{pc_15m:+.1f}¢",
        vol_ratio=f"{vol_ratio:.1f}x",
        spread_change=f"{sp_change:.1f}x",
        related_dis=f"{related_dis:.1f}¢",
    )

    return WeirdMoveSignal(
        ticker=market.ticker,
        flagged=True,
        classification=classification,
        price_change_5m=pc_5m,
        price_change_15m=pc_15m,
        volume_ratio=vol_ratio,
        spread_change=sp_change,
        related_disagreement=related_dis,
        triggers=triggers,
        description=description,
        confidence=confidence,
        timestamp=now,
    )


def _build_description(
    market: Market,
    classification: str,
    pc_5m: float,
    pc_15m: float,
    vol_ratio: float,
    sp_change: float,
    related_dis: float,
    triggers: list[str],
) -> str:
    parts: list[str] = []
    if "price_5m" in triggers:
        parts.append(f"5m Δ {pc_5m:+.1f}¢")
    if "price_15m" in triggers:
        parts.append(f"15m Δ {pc_15m:+.1f}¢")
    if "volume_ratio" in triggers:
        parts.append(f"vol {vol_ratio:.1f}× avg")
    if "spread_change" in triggers:
        parts.append(f"spread {sp_change:.1f}× 1h median")
    if "related_disagreement" in triggers:
        parts.append(f"related mismatch {related_dis:.1f}¢")

    type_label = {
        "news_confirmed_move":         "News-driven move",
        "rumor_move":                  "Rumor/unconfirmed move",
        "liquidity_gap_move":          "Liquidity gap — thin book",
        "related_market_dislocation":  "Related market disagrees",
        "stale_book_artifact":         "Stale book artifact — bad signal",
    }.get(classification, classification)

    return f"{type_label}: {', '.join(parts)}"


# ── Batch entry point ─────────────────────────────────────────────────────────

def batch_detect(
    markets: list[Market],
    all_markets: list[Market],
) -> dict[str, WeirdMoveSignal]:
    """
    Take snapshots for all markets, then detect signals.
    Snapshots are taken first so the current state is recorded before comparison.

    `markets`     — the filtered candidate list (what we're evaluating)
    `all_markets` — the full market list (needed for related-market checks)
    """
    # Pass 1: snapshot — record current state for future comparisons
    for market in markets:
        take_snapshot(market)

    # Pass 2: detect — compare current state against stored history
    results: dict[str, WeirdMoveSignal] = {}
    for market in markets:
        results[market.ticker] = detect(market, all_markets)

    flagged = sum(1 for s in results.values() if s.flagged)
    by_type: dict[str, int] = defaultdict(int)
    for s in results.values():
        if s.flagged:
            by_type[s.classification] += 1

    logger.info(
        _MODULE, "batch_done",
        f"{flagged}/{len(markets)} markets flagged",
        breakdown=dict(by_type),
    )
    return results


# ── Utilities ─────────────────────────────────────────────────────────────────

def _aware_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime without changing the instant."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _trade_size(trade: dict) -> int:
    """Best-effort extraction of contract count from a trade/history row."""
    for key in ("count", "quantity", "qty", "contracts", "size"):
        if key in trade and trade.get(key) is not None:
            try:
                return max(0, int(trade.get(key) or 0))
            except (TypeError, ValueError):
                return 0

    # Some APIs split YES/NO quantities. Add them if present.
    total = 0
    found_split = False
    for key in ("yes_count", "no_count", "yes_quantity", "no_quantity"):
        if key in trade and trade.get(key) is not None:
            found_split = True
            try:
                total += max(0, int(trade.get(key) or 0))
            except (TypeError, ValueError):
                pass
    return total if found_split else 0


def _parse_ts(raw: str) -> Optional[datetime]:
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
            return _aware_utc(dt)
        except ValueError:
            continue
    return None
