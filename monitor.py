from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import db
import logger
import trading
from config import Config
from kalshi_client import KalshiClient
from models import TradeRecord

_MODULE = "monitor"


@dataclass(frozen=True)
class _MarketState:
    yes_ask: int | None = None
    yes_bid: int | None = None
    no_ask: int | None = None
    no_bid: int | None = None
    yes_bid_depth: int | None = None
    no_bid_depth: int | None = None


@dataclass(frozen=True)
class _ExitQuote:
    current_value_cents: int
    exit_yes_price_cents: int
    spread_cents: int | None
    depth: int | None


def check_positions(client: KalshiClient, cfg: Config) -> None:
    """
    Monitor open paper trades and close only when a deterministic, safe paper
    exit price is available. This module never places live orders.
    """
    open_trades = db.get_open_trades()
    if not open_trades:
        logger.info(_MODULE, "no_open_positions", "nothing to monitor")
        return

    logger.info(_MODULE, "checking", f"monitoring {len(open_trades)} open positions")

    for trade in open_trades:
        _check_one(trade, client, cfg)


def _check_one(trade: TradeRecord, client: KalshiClient, cfg: Config) -> None:
    ticker = trade.ticker

    result = client.get_settlement(ticker)
    if result is not None:
        _handle_settlement(trade, result)
        return

    state = _fetch_market_state(client, ticker)
    quote = _exit_quote(trade, state)
    if quote is None:
        logger.warn(_MODULE, "price_fetch_failed", ticker=ticker, trade_id=trade.id)
        return

    move_cents = quote.current_value_cents - trade.entry_price_cents
    liquidity_unsafe = _warn_if_liquidity_deteriorated(trade, quote, cfg)
    minutes_remaining = _minutes_to_close(client, ticker)

    if minutes_remaining is not None and minutes_remaining <= cfg.force_review_last_minutes:
        logger.warn(
            _MODULE,
            "force_review",
            ticker=ticker,
            trade_id=trade.id,
            minutes_remaining=minutes_remaining,
            current_value_cents=quote.current_value_cents,
            pnl_cents=move_cents,
        )
        if _is_thesis_uncertain(trade):
            logger.warn(
                _MODULE,
                "time_exit_uncertain_thesis",
                ticker=ticker,
                trade_id=trade.id,
                minutes_remaining=minutes_remaining,
            )
            _close_at_market(trade, quote.exit_yes_price_cents)
            return

    if _check_thesis_break(trade, client, cfg):
        logger.warn(_MODULE, "thesis_break_exit", ticker=ticker, trade_id=trade.id)
        _close_at_market(trade, quote.exit_yes_price_cents)
        return

    if move_cents <= -cfg.stop_loss_cents:
        logger.warn(
            _MODULE,
            "stop_loss_triggered",
            ticker=ticker,
            trade_id=trade.id,
            entry_cents=trade.entry_price_cents,
            current_value_cents=quote.current_value_cents,
            move_cents=move_cents,
        )
        _close_at_market(trade, quote.exit_yes_price_cents)
        return

    if move_cents >= cfg.take_profit_cents:
        remaining_edge_cents = _remaining_hold_edge_cents(trade, quote)
        if remaining_edge_cents is not None and remaining_edge_cents > cfg.exit_if_edge_below_cents:
            logger.info(
                _MODULE,
                "take_profit_hold_edge_positive",
                ticker=ticker,
                trade_id=trade.id,
                current_value_cents=quote.current_value_cents,
                move_cents=move_cents,
                remaining_edge_cents=remaining_edge_cents,
                exit_threshold_cents=cfg.exit_if_edge_below_cents,
            )
            return
        if liquidity_unsafe:
            logger.warn(
                _MODULE,
                "take_profit_deferred_liquidity",
                ticker=ticker,
                trade_id=trade.id,
                current_value_cents=quote.current_value_cents,
                move_cents=move_cents,
            )
            return
        logger.info(
            _MODULE,
            "take_profit_triggered",
            ticker=ticker,
            trade_id=trade.id,
            entry_cents=trade.entry_price_cents,
            current_value_cents=quote.current_value_cents,
            move_cents=move_cents,
        )
        _close_at_market(trade, quote.exit_yes_price_cents)
        return

    logger.debug(
        _MODULE,
        "position_ok",
        ticker=ticker,
        trade_id=trade.id,
        entry_cents=trade.entry_price_cents,
        current_value_cents=quote.current_value_cents,
        move_cents=move_cents,
        minutes_remaining=minutes_remaining,
    )


def _fetch_market_state(client: KalshiClient, ticker: str) -> _MarketState:
    try:
        raw = client.get_orderbook(ticker, depth=1)
        ob = raw.get("orderbook", raw)
        yes = ob.get("yes", {}) if isinstance(ob, dict) else {}
        no = ob.get("no", {}) if isinstance(ob, dict) else {}
        yes_ask, _ = _best_level(yes.get("ask", []))
        yes_bid, yes_bid_depth = _best_level(yes.get("bid", []))
        no_ask, _ = _best_level(no.get("ask", []))
        no_bid, no_bid_depth = _best_level(no.get("bid", []))
        if _valid_price(no_bid) is None and _valid_price(yes_ask) is not None:
            no_bid = 100 - int(yes_ask)
        return _MarketState(
            yes_ask=_valid_price(yes_ask),
            yes_bid=_valid_price(yes_bid),
            no_ask=_valid_price(no_ask),
            no_bid=_valid_price(no_bid),
            yes_bid_depth=yes_bid_depth,
            no_bid_depth=no_bid_depth,
        )
    except Exception as exc:
        logger.warn(_MODULE, "orderbook_fetch_failed", ticker=ticker, err=str(exc))

    try:
        best_ask, best_bid = client.get_best_prices(ticker)
        yes_ask = _valid_price(best_ask)
        yes_bid = _valid_price(best_bid)
        no_bid = 100 - yes_ask if yes_ask is not None else None
        return _MarketState(yes_ask=yes_ask, yes_bid=yes_bid, no_bid=no_bid)
    except Exception as exc:
        logger.warn(_MODULE, "best_prices_failed", ticker=ticker, err=str(exc))
        return _MarketState()


def _best_level(levels: Any) -> tuple[int | None, int | None]:
    if not levels:
        return None, None
    first = levels[0]
    if isinstance(first, dict):
        price = first.get("price") or first.get("yes_price") or first.get("no_price")
        size = first.get("count") or first.get("size") or first.get("quantity")
    else:
        price = first[0] if len(first) > 0 else None
        size = first[1] if len(first) > 1 else None
    try:
        parsed_price = int(price) if price is not None else None
    except (TypeError, ValueError):
        parsed_price = None
    try:
        parsed_size = int(size) if size is not None else None
    except (TypeError, ValueError):
        parsed_size = None
    return parsed_price, parsed_size


def _exit_quote(trade: TradeRecord, state: _MarketState) -> _ExitQuote | None:
    if trade.side == "YES":
        yes_bid = _valid_price(state.yes_bid)
        if yes_bid is None:
            return None
        spread = _spread(state.yes_ask, state.yes_bid)
        return _ExitQuote(
            current_value_cents=yes_bid,
            exit_yes_price_cents=yes_bid,
            spread_cents=spread,
            depth=state.yes_bid_depth,
        )

    no_bid = _valid_price(state.no_bid)
    if no_bid is None:
        yes_ask = _valid_price(state.yes_ask)
        if yes_ask is None:
            return None
        no_bid = 100 - yes_ask
        exit_yes_price = yes_ask
    else:
        exit_yes_price = 100 - no_bid

    exit_yes_price = _valid_price(exit_yes_price)
    if exit_yes_price is None:
        return None
    spread = _spread(state.no_ask, state.no_bid)
    if spread is None:
        spread = _spread(state.yes_ask, state.yes_bid)
    return _ExitQuote(
        current_value_cents=no_bid,
        exit_yes_price_cents=exit_yes_price,
        spread_cents=spread,
        depth=state.no_bid_depth,
    )


def _warn_if_liquidity_deteriorated(trade: TradeRecord, quote: _ExitQuote, cfg: Config) -> bool:
    unsafe = False
    if quote.spread_cents is not None and quote.spread_cents > cfg.max_spread_cents:
        unsafe = True
        logger.warn(
            _MODULE,
            "wide_spread_exit_risk",
            ticker=trade.ticker,
            trade_id=trade.id,
            spread_cents=quote.spread_cents,
            max_spread_cents=cfg.max_spread_cents,
        )
    if quote.depth is not None and quote.depth < cfg.min_orderbook_depth_at_limit:
        unsafe = True
        logger.warn(
            _MODULE,
            "thin_book_exit_risk",
            ticker=trade.ticker,
            trade_id=trade.id,
            depth=quote.depth,
            min_depth=cfg.min_orderbook_depth_at_limit,
        )
    return unsafe


def _handle_settlement(trade: TradeRecord, result: str) -> None:
    """Close settled trades. Exit price is always expressed as YES cents."""
    normalized = result.lower()
    if normalized not in {"yes", "no"}:
        logger.warn(_MODULE, "unknown_settlement_result", ticker=trade.ticker, result=result)
        return

    exit_yes_price = 100 if normalized == "yes" else 0
    trading.close_paper_trade(trade, exit_yes_price)
    logger.info(
        _MODULE,
        "settled",
        ticker=trade.ticker,
        result=normalized,
        our_side=trade.side,
        trade_id=trade.id,
    )


def _close_at_market(trade: TradeRecord, current_price_cents: int) -> None:
    exit_price = _valid_price(current_price_cents)
    if exit_price is None:
        logger.warn(_MODULE, "invalid_exit_price", ticker=trade.ticker, trade_id=trade.id, price=current_price_cents)
        return
    trading.close_paper_trade(trade, exit_price)


def _minutes_to_close(client: KalshiClient, ticker: str) -> float | None:
    try:
        raw = client.get_market(ticker)
    except Exception as exc:
        logger.warn(_MODULE, "market_fetch_failed", ticker=ticker, err=str(exc))
        return None

    market = raw.get("market", raw) if isinstance(raw, dict) else {}
    for key in ("minutes_to_close", "minutes_to_settlement"):
        value = market.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    for key in ("close_time", "close_ts", "settlement_time"):
        parsed = _parse_datetime(market.get(key))
        if parsed is not None:
            return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds() / 60)
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _valid_price(value: Any) -> int | None:
    try:
        price = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= price <= 100:
        return price
    return None


def _spread(ask: int | None, bid: int | None) -> int | None:
    ask_price = _valid_price(ask)
    bid_price = _valid_price(bid)
    if ask_price is None or bid_price is None or ask_price < bid_price:
        return None
    return ask_price - bid_price


def _is_thesis_uncertain(trade: TradeRecord) -> bool:
    thesis = (trade.thesis or "").lower()
    return "uncertain" in thesis or "unclear" in thesis


def _remaining_hold_edge_cents(trade: TradeRecord, quote: _ExitQuote) -> float | None:
    p_yes = trade.estimated_yes_prob
    if not (0 < p_yes < 1):
        return None
    if trade.side == "YES":
        return (p_yes * 100) - quote.current_value_cents
    return ((1 - p_yes) * 100) - quote.current_value_cents


def _check_thesis_break(trade: TradeRecord, client: KalshiClient, cfg: Config) -> bool:
    # TODO: Wire to deterministic fresh-research contradiction checks once that
    # pipeline exposes a non-LLM, auditable signal for open positions.
    return False


def run_loop(client: KalshiClient, cfg: Config, interval_seconds: int = 300) -> None:
    logger.info(_MODULE, "loop_start", f"monitoring every {interval_seconds}s")
    try:
        while True:
            check_positions(client, cfg)
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        logger.info(_MODULE, "loop_stopped", "monitoring loop interrupted")
