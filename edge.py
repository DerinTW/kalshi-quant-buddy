"""
Edge & expected-value layer.

Given a Market and a ProbabilityEstimate, computes:

  YES:  edge = p_yes - yes_ask/100
        EV   = p_yes * (1 - price) - (1 - p_yes) * price  ≡  p_yes - price
  NO:   edge = (1 - p_yes) - no_ask/100
        EV   = (1 - p_yes) - no_ask/100

  spread_cost     = entry_side_spread_cents / 200       (half-spread reserve)
  slippage_cost   = cfg.slippage_cents / 100
  fee_cost        = cfg.fee_pct / 100                   (FEE_PCT is documented
                                                         as percentage points;
                                                         /100 turns it into a
                                                         decimal cost fraction)
  adjusted_edge   = edge - spread_cost - slippage_cost - fee_cost
  conf_adj_edge   = adjusted_edge * confidence_weight

All cost terms are computed in DECIMAL units; the `_pct` fields exposed on
EdgeResult are pure unit conversions (× 100). This keeps adjusted_edge_pct
and adjusted_ev exactly equivalent (modulo the × 100 factor) for binary
contracts.

The side with the larger adjusted edge wins, but only if at least one side
is strictly positive. `passes_threshold()` then applies the blueprint's
no-trade rules:

  - adjusted_edge <= 0
  - raw_edge_pct < MIN_EDGE
  - adjusted_edge_pct < MIN_ADJUSTED_EDGE
  - confidence_adjusted_ev (in cents) < MIN_CONFIDENCE_ADJUSTED_EDGE_CENTS
  - confidence_weight < MIN_CONFIDENCE
  - spread_cents > MAX_SPREAD_CENTS_EDGE
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
    est_prob:       float
    raw_edge:       float   # decimal
    spread_cost:    float   # decimal
    slippage_cost:  float   # decimal
    fee_cost:       float   # decimal
    adjusted_edge:  float   # decimal
    spread_cents:   int


def _calc_side(
    side: str,
    entry_cents: int,
    spread_cents: int,
    implied_prob: float,
    est_prob: float,
    slippage_dec: float,
    fee_dec: float,
) -> _SideCalc:
    raw_edge = est_prob - implied_prob
    spread_cost = max(0, spread_cents) / 200.0    # half-spread reserve
    adjusted_edge = raw_edge - spread_cost - slippage_dec - fee_dec
    return _SideCalc(
        side=side,
        entry_cents=entry_cents,
        implied_prob=implied_prob,
        est_prob=est_prob,
        raw_edge=raw_edge,
        spread_cost=spread_cost,
        slippage_cost=slippage_dec,
        fee_cost=fee_dec,
        adjusted_edge=adjusted_edge,
        spread_cents=max(0, spread_cents),
    )


# ── Main entry point ─────────────────────────────────────────────────────────

def calculate(
    market: Market,
    estimate: ProbabilityEstimate,
    cfg: Config,
) -> Optional[EdgeResult]:
    """
    Compute edge, EV, and confidence-adjusted EV for both sides; return the
    side with the larger adjusted edge, or None if neither side is positive.
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
    slippage_dec = max(0.0, cfg.slippage_cents) / 100.0   # 2¢   → 0.02
    fee_dec      = max(0.0, cfg.fee_pct) / 100.0          # 2pp  → 0.02

    yes_spread = max(0, market.yes_ask - market.yes_bid)
    no_spread  = max(0, market.no_ask - market.no_bid)

    yes = _calc_side(
        "YES", yes_entry, yes_spread,
        implied_prob=yes_entry / 100.0,
        est_prob=yes_prob,
        slippage_dec=slippage_dec, fee_dec=fee_dec,
    )
    no_ = _calc_side(
        "NO",  no_entry,  no_spread,
        implied_prob=no_entry / 100.0,
        est_prob=no_prob,
        slippage_dec=slippage_dec, fee_dec=fee_dec,
    )

    # Need at least one strictly positive adjusted edge to even consider trading.
    if yes.adjusted_edge <= 0 and no_.adjusted_edge <= 0:
        logger.info(_MODULE, "no_edge", ticker=market.ticker,
                    yes_adj=f"{yes.adjusted_edge:+.4f}",
                    no_adj=f"{no_.adjusted_edge:+.4f}")
        return None

    chosen = yes if yes.adjusted_edge >= no_.adjusted_edge else no_

    conf_weight = _CONFIDENCE_WEIGHTS.get(estimate.confidence, 0.0)

    # All output fields derive from the chosen side's decimal-unit numbers.
    raw_edge_pct      = chosen.raw_edge * 100.0
    adjusted_edge_pct = chosen.adjusted_edge * 100.0
    expected_value    = chosen.raw_edge                         # decimal per $1
    adjusted_ev       = chosen.adjusted_edge                    # decimal per $1
    conf_adj_ev       = adjusted_ev * conf_weight
    conf_adj_edge_pct = adjusted_edge_pct * conf_weight

    result = EdgeResult(
        ticker=market.ticker,
        side=chosen.side,
        entry_price_cents=chosen.entry_cents,
        implied_yes_prob=yes_entry / 100.0,
        estimated_yes_prob=yes_prob,
        raw_edge_pct=raw_edge_pct,
        adjusted_edge_pct=adjusted_edge_pct,
        expected_value=expected_value,
        adjusted_ev=adjusted_ev,
        confidence=estimate.confidence,
        confidence_adjusted_ev=conf_adj_ev,
        confidence_adjusted_edge_pct=conf_adj_edge_pct,
        spread_cents=chosen.spread_cents,
    )

    logger.info(
        _MODULE, "edge_found",
        ticker=market.ticker,
        side=chosen.side,
        entry=f"{chosen.entry_cents}¢",
        est=f"{chosen.est_prob:.1%}",
        implied=f"{chosen.implied_prob:.1%}",
        raw_edge=f"{raw_edge_pct:+.2f}pp",
        adj_edge=f"{adjusted_edge_pct:+.2f}pp",
        ev=f"{expected_value:+.4f}",
        adj_ev=f"{adjusted_ev:+.4f}",
        conf=estimate.confidence,
        conf_adj_edge=f"{conf_adj_edge_pct:+.2f}pp",
        spread=f"{chosen.spread_cents}¢",
    )
    return result


# ── Threshold gate ───────────────────────────────────────────────────────────

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

    if fails:
        logger.info(_MODULE, "threshold_failed",
                    ticker=edge_result.ticker, reasons=fails)
        return False
    return True
