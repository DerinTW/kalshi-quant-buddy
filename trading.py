from __future__ import annotations
import time
import uuid
from typing import Optional

import db
import logger
import postmortem
from config import Config
from kalshi_client import KalshiClient
from models import PositionSize, ProbabilityEstimate, EdgeResult, TradeRecord

_MODULE = "trading"

# Execution modes
DRY_RUN = "dry_run"
PAPER = "paper"
LIVE = "live"

live_buy_guard: set[str] = set()
_REQUIRED_LIVE_PHRASE = "I_UNDERSTAND_THIS_CAN_LOSE_MONEY"
_LIVE_OPEN_STATUSES = {"open", "resting", "pending", "partially_filled"}


def _live_gates_satisfied(cfg: Config) -> tuple[bool, str]:
    """
    All conditions must be true to allow live execution.
    Returns (ok, reason).
    """
    if cfg.kill_switch:
        return False, "KILL_SWITCH=true"
    if cfg.trading_mode != LIVE:
        return False, f"TRADING_MODE={cfg.trading_mode}"
    if cfg.paper_only:
        return False, "PAPER_ONLY=true"
    if not cfg.live_trading_enabled:
        return False, "LIVE_TRADING_ENABLED=false"
    if not cfg.allow_live_orders:
        return False, "ALLOW_LIVE_ORDERS=false"
    if cfg.live_confirmation_phrase != _REQUIRED_LIVE_PHRASE:
        return False, "LIVE_CONFIRMATION_PHRASE mismatch"
    if cfg.max_live_dollars_per_trade <= 0:
        return False, "MAX_LIVE_DOLLARS_PER_TRADE must be positive"

    paper_days = db.count_paper_trading_days()
    if paper_days < cfg.min_paper_days_before_live:
        return False, f"only {paper_days}/{cfg.min_paper_days_before_live} paper days completed"

    completed = db.count_completed_trades()
    if completed < cfg.min_paper_trades_before_live:
        return False, f"only {completed}/{cfg.min_paper_trades_before_live} paper trades completed"

    pnl = db.total_paper_pnl()
    if pnl < cfg.min_paper_pnl_before_live:
        return False, f"paper P&L ${pnl:.2f} < required ${cfg.min_paper_pnl_before_live:.2f}"

    return True, "all gates passed"


def execute(
    sizing: PositionSize,
    estimate: ProbabilityEstimate,
    edge: EdgeResult,
    cfg: Config,
    client: Optional[KalshiClient] = None,
    mode_override: Optional[str] = None,
    risk_approved: bool = False,
) -> Optional[TradeRecord]:
    """
    Execute a trade in the appropriate mode.

    Mode priority:
      1. mode_override (for tests and explicit dry runs)
      2. cfg.trading_mode

    Paper/live execution requires risk_approved=True from the deterministic
    risk manager. Dry runs are logging-only and do not require approval.
    """
    if sizing.contracts <= 0:
        logger.warn(_MODULE, "zero_contracts", ticker=sizing.ticker)
        return None

    # Determine mode
    if mode_override:
        if mode_override not in (DRY_RUN, PAPER, LIVE):
            logger.warn(_MODULE, "invalid_mode_override", mode=mode_override)
            return None
        if mode_override == LIVE:
            live_ok, live_reason = _live_gates_satisfied(cfg)
            if not live_ok or client is None:
                logger.warn(_MODULE, "live_override_rejected",
                            reason=live_reason if not live_ok else "client missing")
                return None
        mode = mode_override
    else:
        if cfg.trading_mode == DRY_RUN:
            mode = DRY_RUN
        elif cfg.trading_mode == PAPER:
            mode = PAPER
        elif cfg.trading_mode == LIVE:
            live_ok, live_reason = _live_gates_satisfied(cfg)
            if not live_ok or client is None:
                logger.warn(_MODULE, "live_rejected",
                            reason=live_reason if not live_ok else "client missing")
                return None
            mode = LIVE
        else:
            logger.warn(_MODULE, "invalid_trading_mode", mode=cfg.trading_mode)
            return None

    if mode in (PAPER, LIVE) and not risk_approved:
        logger.warn(_MODULE, "risk_approval_required",
                    ticker=sizing.ticker, mode=mode)
        return None

    if sizing.entry_price_cents > edge.entry_price_cents:
        logger.warn(
            _MODULE, "entry_price_above_model",
            ticker=sizing.ticker,
            proposed=sizing.entry_price_cents,
            modeled=edge.entry_price_cents,
        )
        return None

    trade_id = str(uuid.uuid4())[:12]
    thesis = estimate.reasoning

    if mode == LIVE and sizing.dollars > cfg.max_live_dollars_per_trade:
        logger.warn(
            _MODULE, "live_size_rejected",
            dollars=f"${sizing.dollars:.2f}",
            cap=f"${cfg.max_live_dollars_per_trade:.2f}",
        )
        return None

    record = TradeRecord(
        id=trade_id,
        ticker=sizing.ticker,
        side=sizing.side,
        contracts=sizing.contracts,
        entry_price_cents=sizing.entry_price_cents,
        dollars_at_risk=sizing.dollars,
        mode=mode,
        thesis=thesis,
        estimated_yes_prob=estimate.yes_probability,
        result="open",
    )

    if mode == DRY_RUN:
        logger.info(
            _MODULE, "dry_run_trade",
            mode=DRY_RUN,
            ticker=sizing.ticker,
            side=sizing.side,
            contracts=sizing.contracts,
            entry_price_cents=sizing.entry_price_cents,
            dollars_at_risk=sizing.dollars,
            client_order_id=trade_id,
        )
        _print_dry_run(record, edge)
        logger.trade({
            "mode": DRY_RUN,
            "trade_id": trade_id,
            "ticker": sizing.ticker,
            "side": sizing.side,
            "contracts": sizing.contracts,
            "entry_cents": sizing.entry_price_cents,
            "dollars": sizing.dollars,
            "status": "DRY_RUN",
        })
        return record

    if mode == PAPER:
        logger.info(
            _MODULE, "paper_trade_insert",
            mode=PAPER,
            ticker=sizing.ticker,
            side=sizing.side,
            contracts=sizing.contracts,
            entry_price_cents=sizing.entry_price_cents,
            dollars_at_risk=sizing.dollars,
            client_order_id=trade_id,
        )
        db.insert_trade(record)
        _print_execution(record, mode)
        logger.trade({
            "mode": PAPER,
            "trade_id": trade_id,
            "ticker": sizing.ticker,
            "side": sizing.side,
            "contracts": sizing.contracts,
            "entry_cents": sizing.entry_price_cents,
            "dollars": sizing.dollars,
            "status": "OPEN",
        })
        return record

    if mode == LIVE:
        return _execute_live(record, client, cfg)  # type: ignore[arg-type]

    return None


def _execute_live(record: TradeRecord, client: KalshiClient, cfg: Config) -> Optional[TradeRecord]:
    """Place a real order on Kalshi. Only reachable after all gates pass."""
    ticker = record.ticker
    side = record.side.lower()
    price = record.entry_price_cents
    guard_key = f"{ticker}:{record.side}"

    if guard_key in live_buy_guard:
        logger.warn(_MODULE, "live_buy_guard_duplicate", ticker=ticker, side=record.side)
        return None

    logger.info(
        _MODULE, "live_order_submit",
        mode=LIVE,
        ticker=ticker,
        side=record.side,
        contracts=record.contracts,
        entry_price_cents=price,
        dollars_at_risk=record.dollars_at_risk,
        client_order_id=record.id,
        order_type="limit",
    )
    logger.trade({
        "mode": LIVE,
        "trade_id": record.id,
        "ticker": ticker,
        "side": record.side,
        "contracts": record.contracts,
        "entry_cents": price,
        "dollars": record.dollars_at_risk,
        "client_order_id": record.id,
        "status": "SUBMITTING",
        "order_type": "limit",
    })

    try:
        resp = client.place_order(
            ticker=ticker,
            side=side,
            action="buy",
            contracts=record.contracts,
            price_cents=price,
            order_type="limit",
            client_order_id=record.id,
        )
        order = resp.get("order", resp)
        status = order.get("status", "unknown")
        order_id = order.get("order_id") or order.get("id") or "?"
        fill_count = order.get("filled_count", order.get("fill_count"))
        remaining_count = order.get("remaining_count")
        if remaining_count is None and fill_count is not None:
            remaining_count = max(0, record.contracts - int(fill_count))
        logger.info(_MODULE, "live_order_placed",
                    ticker=ticker, order_id=order_id, status=status,
                    fill_count=fill_count, remaining_count=remaining_count)
    except Exception as exc:
        logger.error(
            _MODULE, "live_order_failed",
            ticker=ticker,
            side=record.side,
            contracts=record.contracts,
            price=price,
            err_type=type(exc).__name__,
            err=str(exc),
        )
        logger.trade({
            "mode": LIVE,
            "trade_id": record.id,
            "ticker": ticker,
            "side": record.side,
            "contracts": record.contracts,
            "price": price,
            "status": "FAILED",
            "err_type": type(exc).__name__,
            "err": str(exc),
        })
        return None

    live_buy_guard.add(guard_key)
    _cancel_unfilled_if_configured(client, order_id, status, remaining_count, cfg)
    db.insert_trade(record)
    _print_execution(record, LIVE)
    logger.trade({
        "mode": LIVE,
        "trade_id": record.id,
        "ticker": ticker,
        "side": record.side,
        "contracts": record.contracts,
        "entry_cents": record.entry_price_cents,
        "dollars": record.dollars_at_risk,
        "client_order_id": record.id,
        "order_id": order_id,
        "status": status,
        "fill_count": fill_count,
        "remaining_count": remaining_count,
        "order_type": "limit",
    })
    return record


def _cancel_unfilled_if_configured(
    client: KalshiClient,
    order_id: str,
    status: str,
    remaining_count,
    cfg: Config,
) -> None:
    timeout_seconds = getattr(cfg, "live_order_timeout_seconds", 0)
    if not timeout_seconds or order_id == "?":
        return
    if str(status).lower() not in _LIVE_OPEN_STATUSES:
        return
    if remaining_count is not None and int(remaining_count) <= 0:
        return

    time.sleep(float(timeout_seconds))
    try:
        resp = client.cancel_order(order_id)
        logger.info(_MODULE, "live_order_cancel_requested",
                    order_id=order_id, response_status=resp.get("status", "unknown"))
        logger.trade({
            "mode": LIVE,
            "order_id": order_id,
            "status": "CANCEL_REQUESTED",
            "cancel_response_status": resp.get("status", "unknown"),
        })
    except Exception as exc:
        logger.warn(_MODULE, "live_order_cancel_failed",
                    order_id=order_id, err_type=type(exc).__name__, err=str(exc))


def close_paper_trade(trade: TradeRecord, exit_price_cents: int) -> None:
    """Settle a paper trade given its exit price."""
    if trade.side == "YES":
        # Bought YES at entry, now worth exit_price / 100 per contract
        pnl = (exit_price_cents - trade.entry_price_cents) / 100.0 * trade.contracts
    else:
        # Bought NO, priced as (100 - yes_price)
        pnl = ((100 - exit_price_cents) - trade.entry_price_cents) / 100.0 * trade.contracts

    result = "win" if pnl > 0 else "loss" if pnl < 0 else "push"
    db.close_trade(trade.id, exit_price_cents, pnl, result)
    trade.exit_price_cents = exit_price_cents
    trade.pnl_dollars = pnl
    trade.result = result
    logger.info(
        _MODULE, "trade_closed",
        trade_id=trade.id,
        ticker=trade.ticker,
        result=result,
        pnl=f"${pnl:+.2f}",
        exit_price=f"{exit_price_cents}¢",
    )
    if result == "loss":
        try:
            postmortem.run_for_trade(trade)
        except Exception as exc:
            logger.error(_MODULE, "postmortem_failed", trade_id=trade.id, ticker=trade.ticker, err=str(exc))


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_dry_run(record: TradeRecord, edge: EdgeResult) -> None:
    print(f"\n{'='*60}")
    print(f"  DRY RUN — {record.ticker}")
    print(f"  Side:      BUY {record.side}")
    print(f"  Contracts: {record.contracts}")
    print(f"  Entry:     {record.entry_price_cents}¢")
    print(f"  Cost:      ${record.dollars_at_risk:.2f}")
    print(f"  Adj edge:  {edge.adjusted_edge_pct:+.1f}pp")
    print(f"  Adj EV:    {edge.adjusted_ev:+.3f}")
    print(f"  Thesis:    {record.thesis[:120]}")
    print(f"{'='*60}\n")


def _print_execution(record: TradeRecord, mode: str) -> None:
    label = "PAPER TRADE" if mode == PAPER else "LIVE ORDER"
    print(f"\n{'='*60}")
    print(f"  {label} EXECUTED — {record.ticker}")
    print(f"  ID:        {record.id}")
    print(f"  Side:      BUY {record.side}")
    print(f"  Contracts: {record.contracts}")
    print(f"  Entry:     {record.entry_price_cents}¢")
    print(f"  At risk:   ${record.dollars_at_risk:.2f}")
    print(f"{'='*60}\n")
