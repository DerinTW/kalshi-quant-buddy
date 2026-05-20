from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ── Module-level JSON helpers (shared by to_dict() methods below) ────────────

def _iso(value: Optional[datetime]) -> Optional[str]:
    """Datetime → ISO 8601 string, None passes through. JSON-safe."""
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return None


def _str_list(value: Any) -> list[str]:
    """Coerce to a list of strings — used for risk-summary / monitoring lists."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _clamp01(value: float, default: float = 0.0) -> float:
    """Clamp probability/credibility/relevance/recency to [0.0, 1.0]."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


@dataclass
class Market:
    ticker: str
    title: str
    status: str
    yes_ask: int              # cents (0-100)
    yes_bid: int
    no_ask: int
    no_bid: int
    volume: int               # total lifetime contracts
    volume_24h: int
    open_interest: int
    close_time: datetime      # when trading stops (no more orders accepted)
    settlement_time: datetime # when market resolves/pays out (may be later than close_time)
    category: str
    rules_primary: str
    result: Optional[str] = None
    # computed by market_scanner
    spread_pct: float = 0.0
    minutes_to_close: float = 0.0
    minutes_to_settlement: float = 0.0
    liquidity_dollars: float = 0.0
    # safety — set by market_scanner when a field is missing or clearly bad
    is_unsafe: bool = False
    unsafe_reason: str = ""
    # recent trade history — attached by enrich_with_history(), empty otherwise
    price_history: list[dict] = field(default_factory=list)
    # event grouping — the series this market belongs to (e.g. "KXBTCD-24DEC31")
    # used by filters to deduplicate markets in the same event group
    event_ticker: str = ""
    # when this market/orderbook snapshot was fetched by the scanner
    fetched_at: Optional[datetime] = None
    # last known trade time - used by trade-history / weird-move activity logic
    last_trade_at: Optional[datetime] = None
    # total contracts available at the best ask/bid — set by enrich_with_orderbook_depth()
    orderbook_depth: int = 0
    # True once enrich_with_orderbook_depth() has been called for this market
    # (even if the book was empty). Distinguishes "we never asked" from
    # "we asked and got 0" so the depth filter can reject empty books.
    orderbook_depth_fetched: bool = False

    # ── Step 18 blueprint compatibility ──────────────────────────────────
    @property
    def time_to_resolution_minutes(self) -> float:
        """Blueprint alias — mirrors minutes_to_settlement (existing field)."""
        return self.minutes_to_settlement

    @property
    def mid_price_cents(self) -> float:
        """Midpoint of best bid/ask in cents; falls back to yes_ask alone."""
        if self.yes_bid > 0 and self.yes_ask > 0:
            return (self.yes_ask + self.yes_bid) / 2.0
        return float(self.yes_ask or 0)

    def to_dict(self) -> dict:
        """JSON-safe dict — datetimes become ISO strings."""
        return {
            "ticker":                self.ticker,
            "title":                 self.title,
            "status":                self.status,
            "yes_ask":               self.yes_ask,
            "yes_bid":               self.yes_bid,
            "no_ask":                self.no_ask,
            "no_bid":                self.no_bid,
            "volume":                self.volume,
            "volume_24h":            self.volume_24h,
            "open_interest":         self.open_interest,
            "close_time":            _iso(self.close_time),
            "settlement_time":       _iso(self.settlement_time),
            "category":              self.category,
            "rules_primary":         self.rules_primary,
            "result":                self.result,
            "spread_pct":            self.spread_pct,
            "minutes_to_close":      self.minutes_to_close,
            "minutes_to_settlement": self.minutes_to_settlement,
            "liquidity_dollars":     self.liquidity_dollars,
            "is_unsafe":             self.is_unsafe,
            "unsafe_reason":         self.unsafe_reason,
            "event_ticker":          self.event_ticker,
            "fetched_at":            _iso(self.fetched_at),
            "last_trade_at":         _iso(self.last_trade_at),
            "orderbook_depth":       self.orderbook_depth,
            "orderbook_depth_fetched": self.orderbook_depth_fetched,
        }


@dataclass
class Orderbook:
    ticker: str
    yes_bids: list[tuple[int, int]]   # [(price_cents, size), ...]
    yes_asks: list[tuple[int, int]]
    fetched_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MarketSnapshot:
    """Point-in-time state of a market, stored by SnapshotStore for trend analysis."""
    ticker: str
    mid_yes: float       # (yes_ask + yes_bid) / 2
    spread_cents: int    # yes_ask - yes_bid
    volume: int          # cumulative total volume at snapshot time
    timestamp: datetime


@dataclass
class WeirdMoveSignal:
    ticker: str
    flagged: bool
    classification: str      # news_confirmed_move | rumor_move | liquidity_gap_move
                             # | related_market_dislocation | stale_book_artifact | none
    price_change_5m: float   # cents moved in last 5 min (positive = up)
    price_change_15m: float  # cents moved in last 15 min
    volume_ratio: float      # volume_last_10m / avg_volume_10m
    spread_change: float     # current_spread / median_spread_1h
    related_disagreement: float  # max abs disagreement with related-market implied prob (¢)
    triggers: list[str]      # which threshold(s) fired
    description: str
    confidence: str          # high | medium | low
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "ticker":               self.ticker,
            "flagged":              bool(self.flagged),
            "classification":       self.classification,
            "price_change_5m":      round(float(self.price_change_5m), 4),
            "price_change_15m":     round(float(self.price_change_15m), 4),
            "volume_ratio":         round(float(self.volume_ratio), 4),
            "spread_change":        round(float(self.spread_change), 4),
            "related_disagreement": round(float(self.related_disagreement), 4),
            "triggers":             _str_list(self.triggers),
            "description":          self.description,
            "confidence":           self.confidence,
            "timestamp":            _iso(self.timestamp),
        }


@dataclass
class ResearchItem:
    """Single piece of evidence found by a research agent."""
    source: str                    # publisher name (Reuters, CoinDesk, …)
    url: str                       # empty string when unknown
    published_at: Optional[datetime]
    claim: str                     # one specific, concrete claim in one sentence
    direction: str                 # supports_yes | supports_no | neutral | unclear
    relevance: float               # 0.0–1.0 — how relevant to this specific market
    credibility: float             # 0.0–1.0 — from source reliability table
    recency_score: float           # 0.0–1.0 — decays with age
    summary: str                   # 2–3 sentence context
    agent: str = ""                # which research agent found this

    def to_dict(self) -> dict:
        """JSON-safe dict — datetimes become ISO strings."""
        return {
            "source":        self.source,
            "url":           self.url,
            "published_at":  _iso(self.published_at),
            "claim":         self.claim,
            "direction":     self.direction,
            "relevance":     round(_clamp01(self.relevance), 4),
            "credibility":   round(_clamp01(self.credibility), 4),
            "recency_score": round(_clamp01(self.recency_score), 4),
            "summary":       self.summary,
            "agent":         self.agent,
        }


@dataclass
class CategoryResearchItem:
    """
    Category-aware research item produced by category_research.py.

    Matches the spec output schema directly. Convert to legacy ResearchItem
    via to_legacy() to feed sentiment.py / edge.py pipelines unchanged.
    """
    query: str                     # the question being researched (usually market.title)
    source_type: str               # official | news | social | market_data
    source_name: str               # publisher / exchange / agency name
    claim: str                     # one specific assertion in one sentence
    supports: str                  # yes | no | neutral | unclear
    credibility: float             # 0.0–1.0
    relevance: float               # 0.0–1.0
    recency: float                 # 0.0–1.0
    risk_flags: list[str] = field(default_factory=list)
    # supplementary, not in spec schema but needed for downstream tooling
    url: str = ""
    published_at: Optional[datetime] = None
    summary: str = ""

    def to_legacy(self) -> "ResearchItem":
        """Convert to legacy ResearchItem for sentiment/edge consumers."""
        direction_map = {
            "yes": "supports_yes",
            "no": "supports_no",
            "neutral": "neutral",
            "unclear": "unclear",
        }
        return ResearchItem(
            source=self.source_name,
            url=self.url,
            published_at=self.published_at,
            claim=self.claim,
            direction=direction_map.get(self.supports, "unclear"),
            relevance=self.relevance,
            credibility=self.credibility,
            recency_score=self.recency,
            summary=self.summary or self.claim,
            agent=f"category:{self.source_type}",
        )

    def to_dict(self) -> dict:
        """Spec-shaped dict for JSON output / logging."""
        return {
            "query": self.query,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "claim": self.claim,
            "supports": self.supports,
            "credibility": round(self.credibility, 3),
            "relevance": round(self.relevance, 3),
            "recency": round(self.recency, 3),
            "risk_flags": self.risk_flags,
        }


@dataclass
class ResearchResult:
    ticker: str
    query: str
    items: list[ResearchItem] = field(default_factory=list)
    failed_reason: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def raw_text(self) -> str:
        """Formatted text for LLM consumption, ranked by composite score."""
        if self.failed_reason:
            return f"[Research failed: {self.failed_reason}]"
        if not self.items:
            return "No research findings available."
        _DIR = {"supports_yes": "YES↑", "supports_no": "NO↑", "neutral": "→", "unclear": "?"}
        ranked = sorted(self.items,
                        key=lambda x: -(x.relevance * x.credibility * x.recency_score))
        parts = []
        for item in ranked:
            tag = _DIR.get(item.direction, "?")
            parts.append(
                f"[{item.source}] [{tag}] cred={item.credibility:.0%} "
                f"recency={item.recency_score:.0%}\n"
                f"Claim: {item.claim}\n"
                f"Summary: {item.summary}"
            )
        return "\n\n---\n\n".join(parts)

    @property
    def sources(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in self.items:
            if item.source not in seen:
                seen.add(item.source)
                out.append(item.source)
        return out

    def to_dict(self) -> dict:
        """JSON-safe dict including items as nested dicts."""
        return {
            "ticker":        self.ticker,
            "query":         self.query,
            "items":         [item.to_dict() for item in self.items],
            "failed_reason": self.failed_reason,
            "timestamp":     _iso(self.timestamp),
        }


@dataclass
class SentimentResult:
    ticker: str
    sentiment_score: float              # -1.0 (strong NO signal) to +1.0 (strong YES signal)
    narrative_direction: str            # supports_yes | supports_no | mixed | neutral
    confidence: float                   # 0.0 to 1.0; capped by source quality
    market_impact_estimate_cents: int   # estimated market price move if signal is right
    major_contradictions: list[str]     # human-readable conflict descriptions
    item_count: int = 0
    contributing_sources: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    # ── Spec-aligned narrative fields (added; default for back-compat) ─────
    source_credibility: float = 0.0     # weighted avg credibility of inputs
    event_relevance: float = 0.0        # weighted avg relevance of inputs
    rumor_risk: str = "low"             # low | medium | high

    def to_spec_dict(self) -> dict:
        """
        Spec-shaped dict for the narrative-analysis schema:
          sentiment_score, narrative_direction, source_credibility,
          event_relevance, market_impact_estimate_cents, confidence,
          contradictions, rumor_risk
        """
        return {
            "sentiment_score":              round(self.sentiment_score, 4),
            "narrative_direction":          self.narrative_direction,
            "source_credibility":           round(self.source_credibility, 4),
            "event_relevance":              round(self.event_relevance, 4),
            "market_impact_estimate_cents": int(self.market_impact_estimate_cents),
            "confidence":                   round(self.confidence, 4),
            "contradictions":               list(self.major_contradictions),
            "rumor_risk":                   self.rumor_risk,
        }

    def to_dict(self) -> dict:
        """Full JSON-safe dict including ticker + timestamp + spec fields."""
        return {
            "ticker":                       self.ticker,
            "sentiment_score":              round(self.sentiment_score, 4),
            "narrative_direction":          self.narrative_direction,
            "confidence":                   round(_clamp01(self.confidence), 4),
            "market_impact_estimate_cents": int(self.market_impact_estimate_cents),
            "major_contradictions":         list(self.major_contradictions),
            "item_count":                   int(self.item_count),
            "contributing_sources":         list(self.contributing_sources),
            "timestamp":                    _iso(self.timestamp),
            "source_credibility":           round(_clamp01(self.source_credibility), 4),
            "event_relevance":              round(_clamp01(self.event_relevance), 4),
            "rumor_risk":                   self.rumor_risk,
        }


@dataclass
class FeatureVector:
    """
    Inputs to the probability model. All fields are deterministic functions
    of upstream module outputs (Market, SentimentResult, WeirdMoveSignal)
    plus optional context (spot price, sibling markets).
    """
    ticker: str
    # ── Market microstructure ────────────────────────────────────────────
    current_yes_bid:           int      # cents 0-100
    current_yes_ask:           int      # cents 0-100
    mid_price:                 float    # cents
    spread:                    int      # cents
    volume:                    int      # 24h
    open_interest:             int
    order_book_depth:          int      # contracts available at best ask
    time_to_resolution_minutes: float
    # ── Price dynamics (from weird_move) ─────────────────────────────────
    price_change_5m:           float    # cents over last 5 min
    price_change_15m:          float    # cents over last 15 min
    volume_spike_ratio:        float    # vol(last 10m) / median vol 10m windows
    # ── Context ──────────────────────────────────────────────────────────
    category:                  str
    related_market_prices:     list[float] = field(default_factory=list)
                                                # implied YES probs (0-1) of siblings
    # ── Narrative (from sentiment) ───────────────────────────────────────
    sentiment_score:           float = 0.0
    source_credibility:        float = 0.0
    event_relevance:           float = 0.0
    # ── Derived from history / external ──────────────────────────────────
    historical_volatility:     float = 0.0     # stdev of yes-mid Δ in cents
    distance_to_threshold:     Optional[float] = None
                                                # |spot - strike| / strike, when both known

    def to_dict(self) -> dict:
        return {
            "ticker":                     self.ticker,
            "current_yes_bid":            self.current_yes_bid,
            "current_yes_ask":            self.current_yes_ask,
            "mid_price":                  round(self.mid_price, 3),
            "spread":                     self.spread,
            "volume":                     self.volume,
            "open_interest":              self.open_interest,
            "order_book_depth":           self.order_book_depth,
            "time_to_resolution_minutes": round(self.time_to_resolution_minutes, 2),
            "price_change_5m":            round(self.price_change_5m, 3),
            "price_change_15m":           round(self.price_change_15m, 3),
            "volume_spike_ratio":         round(self.volume_spike_ratio, 3),
            "category":                   self.category,
            "related_market_prices":      [round(p, 4) for p in self.related_market_prices],
            "sentiment_score":            round(self.sentiment_score, 4),
            "source_credibility":         round(self.source_credibility, 4),
            "event_relevance":            round(self.event_relevance, 4),
            "historical_volatility":      round(self.historical_volatility, 4),
            "distance_to_threshold":      (round(self.distance_to_threshold, 4)
                                           if self.distance_to_threshold is not None
                                           else None),
        }


@dataclass
class ProbabilityEstimate:
    ticker: str
    yes_probability: float   # 0.0 to 1.0
    confidence: str          # low | medium-low | medium | medium-high | high
    reasoning: str
    assumptions: list[str] = field(default_factory=list)
    invalidation_conditions: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def no_probability(self) -> float:
        """Derived NO probability — always 1 - yes_probability."""
        return round(1.0 - _clamp01(self.yes_probability), 6)

    def to_dict(self) -> dict:
        return {
            "ticker":                  self.ticker,
            "yes_probability":         round(_clamp01(self.yes_probability), 6),
            "no_probability":          self.no_probability,
            "confidence":              self.confidence,
            "reasoning":               self.reasoning,
            "assumptions":             _str_list(self.assumptions),
            "invalidation_conditions": _str_list(self.invalidation_conditions),
            "timestamp":               _iso(self.timestamp),
        }


@dataclass
class EdgeResult:
    """
    Result of the edge/EV layer for a single market.

    Unit conventions:
      *_pct        — percentage points (cents on a 100¢ contract).
      expected_value / adjusted_ev / confidence_adjusted_ev — decimal per $1
                     risked (e.g. 0.05 means +$0.05 per $1).
      spread_cents — raw cents.

    For binary YES/NO contracts the edge equals the EV per $1, so
    `adjusted_edge_pct == adjusted_ev * 100` and similarly for the
    confidence-weighted variants. The fields are duplicated for clarity
    and because downstream callers read different naming conventions.
    """
    ticker: str
    side: str                       # YES | NO
    entry_price_cents: int
    implied_yes_prob: float         # from market price
    estimated_yes_prob: float       # from prediction model
    raw_edge_pct: float             # estimated - implied (positive = edge in our favor)
    adjusted_edge_pct: float        # after spread / slippage / fees
    expected_value: float           # raw EV per dollar risked (decimal)
    adjusted_ev: float              # EV after spread / slippage / fees (decimal)
    confidence: str = ""            # from ProbabilityEstimate
    confidence_adjusted_ev: float = 0.0          # adjusted_ev * confidence_weight (decimal)
    confidence_adjusted_edge_pct: float = 0.0    # adjusted_edge_pct * confidence_weight (pp)
    spread_cents: int = 0           # spread of the entry side

    def to_dict(self) -> dict:
        return {
            "ticker":                       self.ticker,
            "side":                         self.side,
            "entry_price_cents":            int(self.entry_price_cents),
            "implied_yes_prob":             round(_clamp01(self.implied_yes_prob), 6),
            "estimated_yes_prob":           round(_clamp01(self.estimated_yes_prob), 6),
            "raw_edge_pct":                 round(self.raw_edge_pct, 4),
            "adjusted_edge_pct":            round(self.adjusted_edge_pct, 4),
            "expected_value":               round(self.expected_value, 6),
            "adjusted_ev":                  round(self.adjusted_ev, 6),
            "confidence":                   self.confidence,
            "confidence_adjusted_ev":       round(self.confidence_adjusted_ev, 6),
            "confidence_adjusted_edge_pct": round(self.confidence_adjusted_edge_pct, 4),
            "spread_cents":                 int(self.spread_cents),
        }


@dataclass
class RiskDecision:
    ticker: str
    approved: bool
    reason: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    ticker: str
    side: str
    approved: bool
    reason: str
    mode: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def rejection_reasons(self) -> list[str]:
        """Blueprint alias — list of failed checks for rejected assessments."""
        return list(self.checks_failed) if not self.approved else []

    def to_dict(self) -> dict:
        return {
            "ticker":          self.ticker,
            "side":            self.side,
            "approved":        bool(self.approved),
            "reason":          self.reason,
            "mode":            self.mode,
            "checks_passed":   _str_list(self.checks_passed),
            "checks_failed":   _str_list(self.checks_failed),
            "timestamp":       _iso(self.timestamp),
        }


@dataclass
class PositionSize:
    ticker: str
    side: str
    dollars: float
    contracts: int
    entry_price_cents: int
    max_loss_dollars: float

    def to_dict(self) -> dict:
        return {
            "ticker":            self.ticker,
            "side":              self.side,
            "dollars":           round(float(self.dollars), 2),
            "contracts":         int(self.contracts),
            "entry_price_cents": int(self.entry_price_cents),
            "max_loss_dollars":  round(float(self.max_loss_dollars), 2),
        }


@dataclass
class TradeRecord:
    id: str
    ticker: str
    side: str
    contracts: int
    entry_price_cents: int
    dollars_at_risk: float
    mode: str                        # dry_run | paper | live
    thesis: str = ""
    estimated_yes_prob: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    exit_price_cents: Optional[int] = None
    exit_timestamp: Optional[datetime] = None
    pnl_dollars: Optional[float] = None
    result: Optional[str] = None     # win | loss | push | open

    def to_dict(self) -> dict:
        return {
            "id":                self.id,
            "ticker":            self.ticker,
            "side":              self.side,
            "contracts":         int(self.contracts),
            "entry_price_cents": int(self.entry_price_cents),
            "dollars_at_risk":   round(float(self.dollars_at_risk), 2),
            "mode":              self.mode,
            "thesis":            self.thesis,
            "estimated_yes_prob": round(_clamp01(self.estimated_yes_prob), 6),
            "timestamp":         _iso(self.timestamp),
            "exit_price_cents":  self.exit_price_cents,
            "exit_timestamp":    _iso(self.exit_timestamp),
            "pnl_dollars":       (round(float(self.pnl_dollars), 2)
                                  if self.pnl_dollars is not None else None),
            "result":            self.result,
        }


# ── Step 18 blueprint: TradeDecision ─────────────────────────────────────────
# This mirrors the dict returned by decision_formatter.format_decision().
# We keep the formatter returning a plain dict (existing contract) and offer
# this dataclass for callers who want a typed handle. from_dict() / to_dict()
# round-trip cleanly with the formatter's output.

@dataclass
class TradeDecision:
    """
    Final trade-decision record. Produced by decision_formatter; consumed by
    logging/storage and downstream review. Does not place orders.

    `action` is one of: BUY_YES | BUY_NO | NO_TRADE | EXIT | HOLD.
    `side`   is one of: "yes" | "no" | None (None for NO_TRADE / HOLD).
    """
    action: str
    ticker: str
    side: Optional[str] = None
    limit_price_cents: int = 0
    contracts: int = 0
    dollar_size: float = 0.0
    thesis: str = ""
    edge_cents: float = 0.0
    confidence: float = 0.0     # numeric 0.0–1.0
    risk_summary: str = ""
    monitoring_plan: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "TradeDecision":
        """Construct from the dict returned by decision_formatter.format_decision()."""
        return cls(
            action=str(data.get("action", "NO_TRADE")),
            ticker=str(data.get("ticker", "")),
            side=(str(data["side"]) if data.get("side") is not None else None),
            limit_price_cents=int(data.get("limit_price_cents", 0) or 0),
            contracts=int(data.get("contracts", 0) or 0),
            dollar_size=float(data.get("dollar_size", 0.0) or 0.0),
            thesis=str(data.get("thesis", "") or ""),
            edge_cents=float(data.get("edge_cents", 0.0) or 0.0),
            confidence=_clamp01(data.get("confidence", 0.0)),
            risk_summary=str(data.get("risk_summary", "") or ""),
            monitoring_plan=_str_list(data.get("monitoring_plan")),
        )

    def to_dict(self) -> dict:
        return {
            "action":            self.action,
            "ticker":            self.ticker,
            "side":              self.side,
            "limit_price_cents": int(self.limit_price_cents),
            "contracts":         int(self.contracts),
            "dollar_size":       round(float(self.dollar_size), 2),
            "thesis":            self.thesis,
            "edge_cents":        round(float(self.edge_cents), 4),
            "confidence":        round(_clamp01(self.confidence), 4),
            "risk_summary":      self.risk_summary,
            "monitoring_plan":   _str_list(self.monitoring_plan),
        }

    @property
    def is_trade(self) -> bool:
        return self.action in ("BUY_YES", "BUY_NO")


# ── Step 18 blueprint: ExecutionReport ───────────────────────────────────────
# trading.py currently returns a TradeRecord; this dataclass is the structured
# equivalent of the per-order log payload that trading.py also emits (mode,
# trade_id, ticker, side, contracts, entry/limit prices, status, fills,
# remaining, order_id, error). It exists so future code can hand back a typed
# execution result without changing trading.py's current contract.

@dataclass
class ExecutionReport:
    ticker: str
    mode: str                                  # dry_run | paper | live
    status: str                                # SUBMITTED | OPEN | FILLED | PARTIAL
                                               # | CANCELLED | FAILED | DRY_RUN
    trade_id: str = ""                         # internal id (matches TradeRecord.id)
    side: str = ""                             # YES | NO
    contracts_requested: int = 0
    contracts_filled: int = 0
    contracts_remaining: int = 0
    limit_price_cents: int = 0
    avg_fill_price_cents: Optional[int] = None
    dollars_at_risk: float = 0.0
    order_id: Optional[str] = None             # broker-assigned id (live only)
    client_order_id: Optional[str] = None
    placed: bool = False                       # True if any order was actually placed
    error: str = ""                            # populated on FAILED
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def succeeded(self) -> bool:
        return self.status in ("FILLED", "PARTIAL", "OPEN", "DRY_RUN") and not self.error

    def to_dict(self) -> dict:
        return {
            "ticker":               self.ticker,
            "mode":                 self.mode,
            "status":               self.status,
            "trade_id":             self.trade_id,
            "side":                 self.side,
            "contracts_requested":  int(self.contracts_requested),
            "contracts_filled":     int(self.contracts_filled),
            "contracts_remaining":  int(self.contracts_remaining),
            "limit_price_cents":    int(self.limit_price_cents),
            "avg_fill_price_cents": self.avg_fill_price_cents,
            "dollars_at_risk":      round(float(self.dollars_at_risk), 2),
            "order_id":             self.order_id,
            "client_order_id":      self.client_order_id,
            "placed":               bool(self.placed),
            "error":                self.error,
            "timestamp":            _iso(self.timestamp),
        }


@dataclass
class Postmortem:
    trade_id: str
    ticker: str
    original_thesis: str
    estimated_yes_prob: float
    market_price_at_entry: int
    actual_result: str
    was_variance: bool
    data_was_stale: bool
    resolution_handled_correctly: bool
    liquidity_hurt: bool
    sizing_appropriate: bool
    analysis: str
    rule_change_proposal: str
    human_approved: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def proposed_rule_change(self) -> str:
        """Blueprint alias — rule changes are proposals only, never auto-applied."""
        return self.rule_change_proposal

    def to_dict(self) -> dict:
        return {
            "trade_id":                    self.trade_id,
            "ticker":                      self.ticker,
            "original_thesis":             self.original_thesis,
            "estimated_yes_prob":          round(_clamp01(self.estimated_yes_prob), 6),
            "market_price_at_entry":       int(self.market_price_at_entry),
            "actual_result":               self.actual_result,
            "was_variance":                bool(self.was_variance),
            "data_was_stale":              bool(self.data_was_stale),
            "resolution_handled_correctly": bool(self.resolution_handled_correctly),
            "liquidity_hurt":              bool(self.liquidity_hurt),
            "sizing_appropriate":          bool(self.sizing_appropriate),
            "analysis":                    self.analysis,
            "rule_change_proposal":        self.rule_change_proposal,
            "human_approved":              bool(self.human_approved),
            "timestamp":                   _iso(self.timestamp),
        }


# Step 18 blueprint alias — the blueprint refers to "PostmortemReport".
# Code in the project uses "Postmortem"; expose both names so either import
# style works without renaming a stable on-disk dataclass.
PostmortemReport = Postmortem
