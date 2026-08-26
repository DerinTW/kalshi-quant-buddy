from __future__ import annotations

import math

import db
import logger
from config import Config
from models import EdgeResult, Market, PositionSize, ProbabilityEstimate

_MODULE = "position_sizing"

# Avoid taking more than this fraction of visible liquidity/depth.
_LIQUIDITY_DEPTH_FRACTION = 0.20
_EXECUTION_RISK_SIZE_MULTIPLIERS: dict[str, float] = {
    "near_resolution": 0.50,
    "weird_move_liquidity_gap_move": 0.50,
}


def compute(
    market: Market,
    edge: EdgeResult,
    estimate: ProbabilityEstimate,
    cfg: Config,
) -> PositionSize:
    """
    Compute a safe, flat-capped position size for the early system.

    Caps applied:
      - max_trade_dollars, or max_live_dollars_per_trade in live mode
      - paper_bankroll * max_position_pct_of_bankroll
      - 20% of liquidity_dollars
      - 20% of visible orderbook depth when available
      - direct execution-risk multiplier for timing/liquidity-gap hazards
      - time-to-resolution haircut for longer capital lockups

    No Kelly sizing, compounding, or model-confidence scaling is used at this
    stage. Risk manager remains the final authority before execution.
    """
    trade_cap = (
        cfg.max_live_dollars_per_trade
        if cfg.trading_mode == "live"
        else cfg.max_trade_dollars
    )

    current_exposure = db.total_open_exposure()
    bankroll_cap = cfg.paper_bankroll * (cfg.max_position_pct_of_bankroll / 100.0)
    available_bankroll_cap = max(0.0, bankroll_cap - current_exposure)

    if trade_cap <= 0 or available_bankroll_cap <= 0:
        logger.warn(
            _MODULE,
            "zero_base",
            ticker=edge.ticker,
            reason="trade or bankroll cap exhausted",
            exposure=f"${current_exposure:.2f}",
        )
        return _zero(edge)

    cost_per_contract = edge.entry_price_cents / 100.0
    if cost_per_contract <= 0:
        logger.warn(
            _MODULE,
            "invalid_entry_price",
            ticker=edge.ticker,
            entry=f"{edge.entry_price_cents}c",
        )
        return _zero(edge)

    liquidity_cap = max(0.0, market.liquidity_dollars) * _LIQUIDITY_DEPTH_FRACTION
    caps = [trade_cap, available_bankroll_cap, liquidity_cap]

    if market.orderbook_depth > 0:
        caps.append(market.orderbook_depth * cost_per_contract * _LIQUIDITY_DEPTH_FRACTION)

    dollar_size = max(0.0, min(caps))
    execution_multiplier = _execution_risk_multiplier(estimate)
    time_multiplier = time_to_resolution_size_multiplier(market, cfg)
    dollar_size *= execution_multiplier * time_multiplier

    if dollar_size <= 0:
        logger.warn(
            _MODULE,
            "zero_size",
            ticker=edge.ticker,
            reason="liquidity/depth cap reduced size to zero",
            liq_cap=f"${liquidity_cap:.2f}",
        )
        return _zero(edge)

    contracts = math.floor(dollar_size / cost_per_contract)

    if contracts < 1:
        logger.info(
            _MODULE,
            "below_min_contracts",
            ticker=edge.ticker,
            dollar_size=f"${dollar_size:.2f}",
            entry=f"{edge.entry_price_cents}c",
        )
        return _zero(edge)

    actual_dollars = contracts * cost_per_contract

    logger.info(
        _MODULE,
        "sized",
        ticker=edge.ticker,
        side=edge.side,
        contracts=contracts,
        dollars=f"${actual_dollars:.2f}",
        entry=f"{edge.entry_price_cents}c",
        trade_cap=f"${trade_cap:.2f}",
        bankroll_cap=f"${available_bankroll_cap:.2f}",
        liq_cap=f"${liquidity_cap:.2f}",
        depth=market.orderbook_depth,
        execution_multiplier=f"{execution_multiplier:.2f}",
        time_multiplier=f"{time_multiplier:.2f}",
    )
    return PositionSize(
        ticker=edge.ticker,
        side=edge.side,
        dollars=actual_dollars,
        contracts=contracts,
        entry_price_cents=edge.entry_price_cents,
        max_loss_dollars=actual_dollars,
    )


def _zero(edge: EdgeResult) -> PositionSize:
    return PositionSize(
        ticker=edge.ticker,
        side=edge.side,
        dollars=0.0,
        contracts=0,
        entry_price_cents=edge.entry_price_cents,
        max_loss_dollars=0.0,
    )


def _execution_risk_multiplier(estimate: ProbabilityEstimate) -> float:
    penalties = (
        estimate.confidence_breakdown.get("execution_risk_penalties", [])
        if isinstance(estimate.confidence_breakdown, dict)
        else []
    )
    if not isinstance(penalties, list):
        return 1.0
    multiplier = 1.0
    for penalty in penalties:
        multiplier = min(
            multiplier,
            _EXECUTION_RISK_SIZE_MULTIPLIERS.get(str(penalty), 1.0),
        )
    return multiplier


def time_to_resolution_size_multiplier(market: Market, cfg: Config) -> float:
    """
    Conservative collateral-velocity haircut.

    Size is multiplied by 1 / sqrt(days_to_settlement), capped at 1.0 so
    intraday markets keep the existing flat-cap behavior instead of being
    boosted. ``TIME_TO_RESOLUTION_SIZE_FLOOR_DAYS`` prevents tiny horizons
    from producing a multiplier above 1.
    """
    floor_days = max(1e-9, float(cfg.time_to_resolution_size_floor_days))
    minutes = float(
        getattr(market, "minutes_to_settlement", None)
        or getattr(market, "minutes_to_close", 0.0)
        or 0.0
    )
    days = max(floor_days, minutes / (24.0 * 60.0))
    return min(1.0, 1.0 / math.sqrt(days))
