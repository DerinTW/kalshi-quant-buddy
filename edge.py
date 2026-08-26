"""
Edge & expected-value layer.

Given a Market and a ProbabilityEstimate, computes executable EV:

  YES:  edge = p_yes - yes_ask/100
        EV   = p_yes * (1 - price) - (1 - p_yes) * price  ≡  p_yes - price
  NO:   edge = (1 - p_yes) - no_ask/100
        EV   = (1 - p_yes) - no_ask/100

  spread_cost     = entry_side_spread_cents / 200       (half-spread reserve)
  slippage_cost   = effective_slippage_cents / 100
  fee_cost        = cfg.fee_pct / 100                   (FEE_PCT is documented
                                                         as percentage points;
                                                         /100 turns it into a
                                                         decimal cost fraction)
  adjusted_edge   = edge - spread_cost - slippage_cost - fee_cost
  conf_adj_edge   = adjusted_edge * confidence_weight

It also reports a synchronized YES fair mid from both books:

  fair_yes_prob = (YES bid + (100 - NO bid)) / 200

That fair anchor is diagnostic only. Trading EV remains anchored to the entry
ask because that is the executable price. The threshold gate separately checks
half-spread as a percentage of entry cost so cheap contracts with ugly relative
spreads are rejected even when their absolute spread looks acceptable.

All cost terms are computed in DECIMAL units; the `_pct` fields exposed on
EdgeResult are pure unit conversions (× 100). This keeps adjusted_edge_pct
and adjusted_ev exactly equivalent (modulo the × 100 factor) for binary
contracts.

The side with the larger threshold-passing adjusted edge wins, but only if at
least one side is strictly positive. This lets the evaluator route to the
opposite contract when the initially richer side is blocked by one-sided
spread/liquidity friction and the inverse book is actually executable.
`passes_threshold()` applies the blueprint's no-trade rules:

  - adjusted_edge <= 0
  - raw_edge_pct < MIN_EDGE
  - adjusted_edge_pct < MIN_ADJUSTED_EDGE
  - confidence_adjusted_ev (in cents) < MIN_CONFIDENCE_ADJUSTED_EDGE_CENTS
  - confidence_weight < MIN_CONFIDENCE
  - spread_cents > MAX_SPREAD_CENTS_EDGE
  - spread_cost_of_entry_pct > MAX_SPREAD_COST_OF_ENTRY_PCT_EDGE
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import logger
from config import Config
from models import EdgeResult, Market, ProbabilityEstimate

_MODULE = "edge"

# Mirrors prediction_model.CONFIDENCE_WEIGHTS — inlined to avoid a circular import.
_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "low":         0.30,
    "medium-low":  0.50,
    "medium":      0.65,
    "medium-high": 0.80,
    "high":        0.90,
}


# ── Side-specific cost & edge bundle ─────────────────────────────────────────

@dataclass(frozen=True)
class _SideCalc:
    """All numbers needed to pick a side, in canonical decimal units."""
    side:           str   # "YES" or "NO"
    entry_cents:    int
    implied_prob:   float
    fair_prob:      Optional[float]
    est_prob:       float
    raw_edge:       float   # decimal
    model_vs_market_edge: Optional[float]  # decimal
    spread_cost:    float   # decimal
    spread_cost_of_entry_pct: float
    slippage_cost:  float   # decimal
    fee_cost:       float   # decimal
    adjusted_edge:  float   # decimal
    spread_cents:   int


def _calc_side(
    side: str,
    entry_cents: int,
    spread_cents: int,
    implied_prob: float,
    fair_prob: Optional[float],
    est_prob: float,
    slippage_cents: float,
    fee_dec: float,
) -> _SideCalc:
    raw_edge = est_prob - implied_prob
    spread_cost = max(0, spread_cents) / 200.0    # half-spread reserve
    slippage_dec = max(0.0, slippage_cents) / 100.0
    spread_cost_of_entry_pct = (
        (max(0, spread_cents) / 2.0) / entry_cents * 100.0
        if entry_cents > 0
        else 999.0
    )
    adjusted_edge = raw_edge - spread_cost - slippage_dec - fee_dec
    return _SideCalc(
        side=side,
        entry_cents=entry_cents,
        implied_prob=implied_prob,
        fair_prob=fair_prob,
        est_prob=est_prob,
        raw_edge=raw_edge,
        model_vs_market_edge=(est_prob - fair_prob if fair_prob is not None else None),
        spread_cost=spread_cost,
        spread_cost_of_entry_pct=spread_cost_of_entry_pct,
        slippage_cost=slippage_dec,
        fee_cost=fee_dec,
        adjusted_edge=adjusted_edge,
        spread_cents=max(0, spread_cents),
    )


def _effective_slippage_cents(
    market: Market,
    estimate: ProbabilityEstimate,
    cfg: Config,
    *,
    spread_cents: int,
) -> float:
    """
    Scale the slippage reserve down only when conviction and visible depth
    justify it. The base configured value remains the ceiling.
    """
    base = max(0.0, float(cfg.slippage_cents))
    if base <= 0:
        return 0.0

    confidence = str(getattr(estimate, "confidence", "") or "")
    depth = float(getattr(market, "orderbook_depth", 0) or 0)
    liquidity = float(getattr(market, "liquidity_dollars", 0.0) or 0.0)
    min_depth = max(1.0, float(getattr(cfg, "min_orderbook_depth_at_limit", 1) or 1))
    min_liquidity = max(1.0, float(getattr(cfg, "min_liquidity_dollars", 1.0) or 1.0))
    spread_ok = spread_cents <= max(1, int(getattr(cfg, "max_spread_cents_edge", 1) or 1))

    deep_depth = depth >= max(100.0, min_depth * 4.0)
    deep_liquidity = liquidity >= max(1_000.0, min_liquidity * 20.0)
    good_depth = depth >= max(50.0, min_depth * 2.0)
    good_liquidity = liquidity >= max(500.0, min_liquidity * 10.0)

    if confidence == "high" and spread_ok and deep_depth and deep_liquidity:
        return min(base, 0.5)
    if confidence == "medium-high" and spread_ok and good_depth and good_liquidity:
        return min(base, 1.0)
    return base


def _clamp_prob(value: float) -> float:
    return max(0.0, min(1.0, value))


def _synthetic_yes_mid_prob(market: Market) -> tuple[Optional[float], Optional[int]]:
    """
    Return a YES fair midpoint synchronized across YES/NO books.

    The preferred anchor is the midpoint between the executable YES bid and
    synthetic YES ask implied by the NO bid. Fall back to same-book mids only
    when one side of the reciprocal book is unavailable.
    """
    if market.yes_bid > 0 and market.no_bid > 0:
        synthetic_yes_ask = 100 - market.no_bid
        synthetic_spread = 100 - (market.yes_bid + market.no_bid)
        return _clamp_prob((market.yes_bid + synthetic_yes_ask) / 200.0), synthetic_spread

    if market.yes_bid > 0 and market.yes_ask > 0:
        return _clamp_prob((market.yes_bid + market.yes_ask) / 200.0), None

    if market.no_bid > 0 and market.no_ask > 0:
        no_mid = (market.no_bid + market.no_ask) / 200.0
        return _clamp_prob(1.0 - no_mid), None

    return None, None


def _build_result(
    *,
    ticker: str,
    chosen: _SideCalc,
    yes_entry: int,
    yes_prob: float,
    confidence: str,
    conf_weight: float,
    fair_yes_prob: Optional[float],
    synthetic_spread_cents: Optional[int],
) -> EdgeResult:
    raw_edge_pct = chosen.raw_edge * 100.0
    adjusted_edge_pct = chosen.adjusted_edge * 100.0
    expected_value = chosen.raw_edge
    adjusted_ev = chosen.adjusted_edge
    return EdgeResult(
        ticker=ticker,
        side=chosen.side,
        entry_price_cents=chosen.entry_cents,
        implied_yes_prob=yes_entry / 100.0,
        estimated_yes_prob=yes_prob,
        raw_edge_pct=raw_edge_pct,
        adjusted_edge_pct=adjusted_edge_pct,
        expected_value=expected_value,
        adjusted_ev=adjusted_ev,
        confidence=confidence,
        confidence_adjusted_ev=adjusted_ev * conf_weight,
        confidence_adjusted_edge_pct=adjusted_edge_pct * conf_weight,
        spread_cents=chosen.spread_cents,
        slippage_cost_cents=chosen.slippage_cost * 100.0,
        fair_yes_prob=fair_yes_prob,
        fair_side_prob=chosen.fair_prob,
        model_vs_market_edge_pct=(
            chosen.model_vs_market_edge * 100.0
            if chosen.model_vs_market_edge is not None
            else None
        ),
        spread_cost_pct=chosen.spread_cost * 100.0,
        spread_cost_of_entry_pct=chosen.spread_cost_of_entry_pct,
        synthetic_spread_cents=synthetic_spread_cents,
    )


# ── Main entry point ─────────────────────────────────────────────────────────

def calculate(
    market: Market,
    estimate: ProbabilityEstimate,
    cfg: Config,
) -> Optional[EdgeResult]:
    """
    Compute edge, EV, and confidence-adjusted EV for both sides; prefer the
    highest adjusted side that clears threshold gates, or return the best
    positive side so downstream risk checks can explain the rejection.
    """
    yes_prob = max(0.0, min(1.0, estimate.yes_probability))
    no_prob = 1.0 - yes_prob

    yes_entry = market.yes_ask
    no_entry = market.no_ask

    if yes_entry <= 0 or no_entry <= 0:
        logger.debug(_MODULE, "no_valid_prices", ticker=market.ticker,
                     yes_ask=yes_entry, no_ask=no_entry)
        return None

    # Cost components — single source of truth in decimal units
    fee_dec = max(0.0, cfg.fee_pct) / 100.0

    yes_spread = max(0, market.yes_ask - market.yes_bid)
    no_spread  = max(0, market.no_ask - market.no_bid)
    fair_yes_prob, synthetic_spread_cents = _synthetic_yes_mid_prob(market)
    fair_no_prob = 1.0 - fair_yes_prob if fair_yes_prob is not None else None
    yes_slippage = _effective_slippage_cents(
        market, estimate, cfg, spread_cents=yes_spread,
    )
    no_slippage = _effective_slippage_cents(
        market, estimate, cfg, spread_cents=no_spread,
    )

    yes = _calc_side(
        "YES", yes_entry, yes_spread,
        implied_prob=yes_entry / 100.0,
        fair_prob=fair_yes_prob,
        est_prob=yes_prob,
        slippage_cents=yes_slippage, fee_dec=fee_dec,
    )
    no_ = _calc_side(
        "NO",  no_entry,  no_spread,
        implied_prob=no_entry / 100.0,
        fair_prob=fair_no_prob,
        est_prob=no_prob,
        slippage_cents=no_slippage, fee_dec=fee_dec,
    )

    # Need at least one strictly positive adjusted edge to even consider trading.
    if yes.adjusted_edge <= 0 and no_.adjusted_edge <= 0:
        logger.info(_MODULE, "no_edge", ticker=market.ticker,
                    yes_adj=f"{yes.adjusted_edge:+.4f}",
                    no_adj=f"{no_.adjusted_edge:+.4f}")
        return None

    conf_weight = _CONFIDENCE_WEIGHTS.get(estimate.confidence, 0.0)
    ranked_calcs = sorted([yes, no_], key=lambda calc: calc.adjusted_edge, reverse=True)
    candidates = [
        _build_result(
            ticker=market.ticker,
            chosen=side_calc,
            yes_entry=yes_entry,
            yes_prob=yes_prob,
            confidence=estimate.confidence,
            conf_weight=conf_weight,
            fair_yes_prob=fair_yes_prob,
            synthetic_spread_cents=synthetic_spread_cents,
        )
        for side_calc in ranked_calcs
        if side_calc.adjusted_edge > 0
    ]
    passing = [candidate for candidate in candidates if not _threshold_failures(candidate, cfg)]
    result = passing[0] if passing else candidates[0]
    chosen = yes if result.side == "YES" else no_

    if candidates and result.side != candidates[0].side:
        logger.info(
            _MODULE, "alternate_side_selected",
            ticker=market.ticker,
            preferred_side=candidates[0].side,
            routed_side=result.side,
            preferred_failures=_threshold_failures(candidates[0], cfg),
        )

    raw_edge_pct = result.raw_edge_pct
    adjusted_edge_pct = result.adjusted_edge_pct
    expected_value = result.expected_value
    adjusted_ev = result.adjusted_ev
    conf_adj_edge_pct = result.confidence_adjusted_edge_pct

    logger.info(
        _MODULE, "edge_found",
        ticker=market.ticker,
        side=chosen.side,
        entry=f"{chosen.entry_cents}¢",
        est=f"{chosen.est_prob:.1%}",
        implied=f"{chosen.implied_prob:.1%}",
        raw_edge=f"{raw_edge_pct:+.2f}pp",
        fair_edge=(
            f"{result.model_vs_market_edge_pct:+.2f}pp"
            if result.model_vs_market_edge_pct is not None
            else "n/a"
        ),
        adj_edge=f"{adjusted_edge_pct:+.2f}pp",
        ev=f"{expected_value:+.4f}",
        adj_ev=f"{adjusted_ev:+.4f}",
        conf=estimate.confidence,
        conf_adj_edge=f"{conf_adj_edge_pct:+.2f}pp",
        spread_entry=f"{chosen.spread_cost_of_entry_pct:.1f}%",
        spread=f"{chosen.spread_cents}¢",
    )
    return result


# ── Threshold gate ───────────────────────────────────────────────────────────

def _threshold_failures(edge_result: EdgeResult, cfg: Config) -> list[str]:
    conf_weight = _CONFIDENCE_WEIGHTS.get(edge_result.confidence, 0.0)
    conf_adj_cents = edge_result.confidence_adjusted_ev * 100.0

    fails: list[str] = []

    if edge_result.adjusted_edge_pct <= 0:
        fails.append(f"adj_edge={edge_result.adjusted_edge_pct:+.2f}pp <= 0")
    if edge_result.raw_edge_pct < cfg.min_edge_pct:
        fails.append(f"raw_edge={edge_result.raw_edge_pct:.2f}pp < {cfg.min_edge_pct}pp")
    if edge_result.adjusted_edge_pct < cfg.min_adjusted_edge_pct:
        fails.append(
            f"adj_edge={edge_result.adjusted_edge_pct:.2f}pp "
            f"< {cfg.min_adjusted_edge_pct}pp"
        )
    if conf_weight < cfg.min_confidence:
        fails.append(
            f"confidence={edge_result.confidence} ({conf_weight:.2f}) "
            f"< {cfg.min_confidence:.2f}"
        )
    if conf_adj_cents < cfg.min_confidence_adjusted_edge_cents:
        fails.append(
            f"conf_adj_ev={conf_adj_cents:.2f}c "
            f"< {cfg.min_confidence_adjusted_edge_cents:.2f}c"
        )
    if edge_result.spread_cents > cfg.max_spread_cents_edge:
        fails.append(
            f"spread={edge_result.spread_cents}c "
            f"> {cfg.max_spread_cents_edge}c"
        )

    if edge_result.spread_cost_of_entry_pct > cfg.max_spread_cost_of_entry_pct_edge:
        fails.append(
            f"spread_cost_entry={edge_result.spread_cost_of_entry_pct:.1f}% "
            f"> {cfg.max_spread_cost_of_entry_pct_edge:.1f}%"
        )

    return fails


def passes_threshold(edge_result: EdgeResult, cfg: Config) -> bool:
    """
    Return True only when every no-trade condition from the blueprint is clear:

      - adjusted_edge_pct > 0                                  (strict)
      - raw_edge_pct       >= cfg.min_edge_pct
      - adjusted_edge_pct  >= cfg.min_adjusted_edge_pct
      - confidence_weight  >= cfg.min_confidence
      - confidence_adjusted_ev (in cents) >= cfg.min_confidence_adjusted_edge_cents
      - spread_cents       <= cfg.max_spread_cents_edge

    All comparisons are conservative — equality at the boundary is allowed
    only where the blueprint allows it (raw/adjusted edge minimums and
    confidence floor are >=; adjusted_edge_pct > 0 is strict).
    """
    conf_weight = _CONFIDENCE_WEIGHTS.get(edge_result.confidence, 0.0)
    conf_adj_cents = edge_result.confidence_adjusted_ev * 100.0

    fails: list[str] = []

    if edge_result.adjusted_edge_pct <= 0:
        fails.append(f"adj_edge={edge_result.adjusted_edge_pct:+.2f}pp <= 0")
    if edge_result.raw_edge_pct < cfg.min_edge_pct:
        fails.append(f"raw_edge={edge_result.raw_edge_pct:.2f}pp < {cfg.min_edge_pct}pp")
    if edge_result.adjusted_edge_pct < cfg.min_adjusted_edge_pct:
        fails.append(
            f"adj_edge={edge_result.adjusted_edge_pct:.2f}pp "
            f"< {cfg.min_adjusted_edge_pct}pp"
        )
    if conf_weight < cfg.min_confidence:
        fails.append(
            f"confidence={edge_result.confidence} ({conf_weight:.2f}) "
            f"< {cfg.min_confidence:.2f}"
        )
    if conf_adj_cents < cfg.min_confidence_adjusted_edge_cents:
        fails.append(
            f"conf_adj_ev={conf_adj_cents:.2f}¢ "
            f"< {cfg.min_confidence_adjusted_edge_cents:.2f}¢"
        )
    if edge_result.spread_cents > cfg.max_spread_cents_edge:
        fails.append(
            f"spread={edge_result.spread_cents}¢ "
            f"> {cfg.max_spread_cents_edge}¢"
        )

    if edge_result.spread_cost_of_entry_pct > cfg.max_spread_cost_of_entry_pct_edge:
        fails.append(
            f"spread_cost_entry={edge_result.spread_cost_of_entry_pct:.1f}% "
            f"> {cfg.max_spread_cost_of_entry_pct_edge:.1f}%"
        )

    if fails:
        logger.info(_MODULE, "threshold_failed",
                    ticker=edge_result.ticker, reasons=fails)
        return False
    return True
