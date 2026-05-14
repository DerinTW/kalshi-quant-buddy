from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


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
    # last known trade time — used to check orderbook freshness
    last_trade_at: Optional[datetime] = None
    # total contracts available at the best ask/bid — set by enrich_with_orderbook_depth()
    orderbook_depth: int = 0


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


@dataclass
class PositionSize:
    ticker: str
    side: str
    dollars: float
    contracts: int
    entry_price_cents: int
    max_loss_dollars: float


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
