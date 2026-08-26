"""
Agent H: Risk Manager

Final hard gate before any order is placed.
All checks are deterministic — no LLM, no overrides.
Returns RiskAssessment(approved=True) only when every applicable check passes.

Check order:
  1. Kill switch (short-circuits live immediately; paper/dry-run are allowed)
  2. Valid trading mode
  3. Live-trading enabled when mode == live
  4. Daily loss circuit breaker
  5. Daily trade count circuit breaker
  6. Per-trade size absolute cap
  7. Per-trade size bankroll-pct cap
  8. Category exposure cap
  9. Correlated exposure cap
 10. Spread gate (absolute cents)
 11. Liquidity gate
 12. Duplicate ticker+side guard
 13. Time-to-resolution gate
 14. Edge-after-costs gate
 15. Positive adjusted EV
[Live mode only, 16–21]
 16. ALLOW_LIVE_ORDERS flag
 17. LIVE_CONFIRMATION_PHRASE match
 18. Per-trade live size cap
 19. Paper days completed
 20. Paper trade count completed
 21. Paper P&L non-negative
"""
from __future__ import annotations

import db
import logger
from config import Config
from edge import passes_threshold as _edge_passes_threshold
from models import Market, EdgeResult, PositionSize, TradeRecord, RiskAssessment

_MODULE = "risk_manager"
_REQUIRED_LIVE_PHRASE = "I_UNDERSTAND_THIS_CAN_LOSE_MONEY"
_VALID_MODES = {"dry_run", "paper", "live"}
_SUPPORTED_LONG_HORIZON_CATEGORIES = {"crypto", "economic", "financial", "weather"}
_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "low": 0.30,
    "medium-low": 0.50,
    "medium": 0.65,
    "medium-high": 0.80,
    "high": 0.90,
}


def assess(
    decision: PositionSize,
    market: Market,
    edge: EdgeResult,
    open_positions: list[TradeRecord],
    daily_pnl: float,
    trades_today: int,
    bankroll: float,
    cfg: Config,
    *,
    category_exposure: float = 0.0,
    correlated_exposure: float = 0.0,
    action_type: str = "entry",
    live_buy_guard: set[str] | None = None,
) -> RiskAssessment:
    """
    Run all hard-reject checks then live-mode gates (if applicable).
    category_exposure: caller-computed dollars at risk in markets of the same category.
    correlated_exposure: caller-computed dollars at risk in correlated tickers
                         (same event group or explicitly related markets).
    """
    mode = cfg.trading_mode
    is_entry = action_type == "entry"
    passed: list[str] = []
    failed: list[str] = []

    def ok(label: str) -> None:
        passed.append(label)

    def fail(label: str) -> None:
        failed.append(label)

    # ── Check 1: Kill switch ───────────────────────────────────────────────────
    if cfg.kill_switch and mode == "live":
        return _build(
            decision, mode, approved=False,
            reason="KILL_SWITCH is enabled - live trading halted",
            passed=[], failed=["kill_switch_engaged"],
        )
    if cfg.kill_switch:
        ok("kill_switch_live_only: paper/dry_run execution allowed")

    # ── Check 2: Valid trading mode ────────────────────────────────────────────
    if mode in _VALID_MODES:
        ok(f"trading_mode_valid: {mode}")
    else:
        fail(f"invalid_trading_mode: '{mode}' not in {sorted(_VALID_MODES)}")

    if decision.contracts <= 0:
        fail(f"non_positive_contracts: {decision.contracts}")
    else:
        ok(f"contracts_positive: {decision.contracts}")

    if decision.dollars <= 0:
        fail(f"non_positive_dollars: ${decision.dollars:.2f}")
    else:
        ok(f"dollars_positive: ${decision.dollars:.2f}")

    # ── Check 3: Live not enabled ──────────────────────────────────────────────
    if mode == "live" and not cfg.live_trading_enabled:
        fail("live_mode_not_enabled: LIVE_TRADING_ENABLED=false")
    elif mode != "live":
        ok("live_gate_not_required")
    else:
        ok("live_trading_enabled")

    # ── Check 4: Daily loss circuit breaker ────────────────────────────────────
    if daily_pnl <= -cfg.max_daily_loss_dollars:
        fail(
            f"daily_loss_limit: pnl=${daily_pnl:.2f} <= "
            f"-${cfg.max_daily_loss_dollars:.2f}"
        )
    else:
        ok(f"daily_pnl_ok: ${daily_pnl:+.2f} (limit -${cfg.max_daily_loss_dollars:.2f})")

    # ── Check 5: Daily trade count ─────────────────────────────────────────────
    if trades_today >= cfg.max_trades_per_day:
        fail(f"daily_trade_limit: {trades_today} >= {cfg.max_trades_per_day}")
    else:
        ok(f"trades_today_ok: {trades_today}/{cfg.max_trades_per_day}")

    # ── Check 6: Per-trade absolute size cap ───────────────────────────────────
    trade_cap = cfg.max_live_dollars_per_trade if mode == "live" else cfg.max_trade_dollars
    if decision.dollars > trade_cap:
        fail(
            f"position_too_large: ${decision.dollars:.2f} > "
            f"${trade_cap:.2f} (MAX_DOLLARS_PER_TRADE)"
        )
    else:
        ok(f"size_ok: ${decision.dollars:.2f} <= ${trade_cap:.2f}")

    # ── Check 7: Bankroll percentage cap ──────────────────────────────────────
    bankroll_limit = bankroll * cfg.max_position_pct_of_bankroll / 100.0
    if decision.dollars > bankroll_limit:
        fail(
            f"bankroll_pct_exceeded: ${decision.dollars:.2f} > "
            f"${bankroll_limit:.2f} "
            f"({cfg.max_position_pct_of_bankroll}% of ${bankroll:.2f})"
        )
    else:
        ok(
            f"bankroll_pct_ok: ${decision.dollars:.2f} <= "
            f"${bankroll_limit:.2f} ({cfg.max_position_pct_of_bankroll}%)"
        )

    # ── Check 8: Category exposure cap ────────────────────────────────────────
    if category_exposure > cfg.max_category_exposure_dollars:
        fail(
            f"category_exposure_cap: ${category_exposure:.2f} > "
            f"${cfg.max_category_exposure_dollars:.2f} "
            f"(category={market.category})"
        )
    else:
        ok(
            f"category_exposure_ok: ${category_exposure:.2f} <= "
            f"${cfg.max_category_exposure_dollars:.2f}"
        )

    # ── Check 9: Correlated exposure cap ──────────────────────────────────────
    if correlated_exposure > cfg.max_correlated_exposure_dollars:
        fail(
            f"correlated_exposure_cap: ${correlated_exposure:.2f} > "
            f"${cfg.max_correlated_exposure_dollars:.2f}"
        )
    else:
        ok(
            f"correlated_exposure_ok: ${correlated_exposure:.2f} <= "
            f"${cfg.max_correlated_exposure_dollars:.2f}"
        )

    # ── Check 10: Spread gate (absolute cents) ─────────────────────────────────
    spread_cents = edge.spread_cents
    if spread_cents > cfg.max_spread_cents:
        fail(
            f"spread_too_wide: {spread_cents}¢ > {cfg.max_spread_cents}¢"
        )
    else:
        ok(f"spread_ok: {spread_cents}¢ <= {cfg.max_spread_cents}¢")

    # ── Check 11: Liquidity gate ───────────────────────────────────────────────
    if market.liquidity_dollars < cfg.min_liquidity_dollars:
        fail(
            f"insufficient_liquidity: ${market.liquidity_dollars:.0f} < "
            f"${cfg.min_liquidity_dollars:.0f}"
        )
    else:
        ok(f"liquidity_ok: ${market.liquidity_dollars:.0f}")

    # ── Check 12: Duplicate ticker+side guard ─────────────────────────────────
    dupe = any(
        t.ticker == market.ticker and t.side == decision.side
        for t in open_positions
    )
    if dupe:
        fail(f"duplicate_position: {market.ticker} {decision.side} already open")
    else:
        ok(f"no_duplicate: {market.ticker} {decision.side}")

    opposite = any(
        t.ticker == market.ticker and t.side != decision.side
        for t in open_positions
    )
    if opposite and is_entry:
        fail(f"opposite_side_open: {market.ticker} already has opposing side")
    else:
        ok("no_opposite_side_conflict")

    related_group = market.event_ticker or market.ticker
    related_dupe = any(
        t.ticker != market.ticker
        and related_group
        and (t.ticker == related_group or t.ticker.startswith(f"{related_group}-"))
        for t in open_positions
    )
    if related_dupe and is_entry:
        fail(f"related_group_position_open: event_ticker={related_group}")
    else:
        ok("no_related_group_conflict")

    # No thesis_break field exists yet. Conservatively, same-market re-buys are
    # already rejected above, so averaging down cannot be approved for entries.

    # ── Check 13: Time-to-resolution gate ─────────────────────────────────────
    minutes_left = market.minutes_to_close
    conf_weight = _CONFIDENCE_WEIGHTS.get(edge.confidence, 0.0)
    if minutes_left < 5 and is_entry:
        fail(f"time_entry_rejected: minutes_to_close={minutes_left:.0f} < 5")
    elif minutes_left < 20 and is_entry:
        fail(f"time_entry_rejected: 5 <= minutes_to_close={minutes_left:.0f} < 20")
    elif minutes_left < 60:
        if conf_weight >= 0.80:
            ok(f"near_resolution_confidence_ok: {edge.confidence} ({conf_weight:.2f})")
        else:
            fail(
                f"near_resolution_confidence_low: {edge.confidence} "
                f"({conf_weight:.2f}) < 0.80"
            )
        near_cap = trade_cap * 0.50
        if decision.dollars > near_cap:
            fail(
                f"near_resolution_size_too_large: ${decision.dollars:.2f} > "
                f"${near_cap:.2f} (50% time cap)"
            )
        else:
            ok(f"near_resolution_size_ok: ${decision.dollars:.2f} <= ${near_cap:.2f}")
    elif minutes_left <= 24 * 60:
        ok(f"time_ok: minutes_to_close={minutes_left:.0f} in normal window")
    elif minutes_left > 72 * 60 and market.category not in _SUPPORTED_LONG_HORIZON_CATEGORIES:
        fail(
            f"unsupported_long_horizon: minutes_to_close={minutes_left:.0f}, "
            f"category={market.category}"
        )
    else:
        ok(f"time_ok: minutes_to_close={minutes_left:.0f}")

    # ── Check 14: Edge after costs ─────────────────────────────────────────────
    if edge.adjusted_edge_pct < cfg.min_adjusted_edge_pct:
        fail(
            f"edge_too_small: {edge.adjusted_edge_pct:.1f}pp < "
            f"{cfg.min_adjusted_edge_pct:.1f}pp"
        )
    else:
        ok(f"edge_ok: {edge.adjusted_edge_pct:.1f}pp")

    # ── Check 15: Positive adjusted EV ────────────────────────────────────────
    if edge.adjusted_ev <= 0:
        fail(f"non_positive_ev: {edge.adjusted_ev:+.4f}")
    else:
        ok(f"ev_positive: {edge.adjusted_ev:+.4f}")

    # ── Live mode gates (checks 16–21) ─────────────────────────────────────────
    if _edge_passes_threshold(edge, cfg):
        ok("edge_thresholds_ok")
    else:
        fail("edge_thresholds_failed")

    if mode == "live" and is_entry and live_buy_guard is not None:
        live_key = f"{market.ticker}:{decision.side}"
        if live_key in live_buy_guard:
            fail(f"live_buy_guard_duplicate: {live_key}")
        else:
            ok(f"live_buy_guard_clear: {live_key}")

    if mode == "live" and cfg.live_trading_enabled:
        _run_live_gates(cfg, passed, failed, ok, fail)

    approved = len(failed) == 0
    reason = (
        f"All {len(passed)} checks passed"
        if approved
        else f"Failed {len(failed)} check(s): {'; '.join(failed)}"
    )

    logger.decision(
        _MODULE, market.ticker, "risk_assessment",
        "APPROVED" if approved else "REJECTED",
        mode=mode,
        side=decision.side,
        checks_passed=len(passed),
        checks_failed=len(failed),
        failures=failed if not approved else [],
    )

    return _build(decision, mode, approved, reason, passed, failed)


def _run_live_gates(
    cfg: Config,
    passed: list[str],
    failed: list[str],
    ok,
    fail,
) -> None:
    """Additional checks required to place a real order. All must pass."""
    failure_count_before_live_gates = len(failed)

    if cfg.trading_mode == "live":
        ok("trading_mode_live")
    else:
        fail("TRADING_MODE must be live for live orders")

    if cfg.paper_only:
        fail("PAPER_ONLY must be false for live orders")
    else:
        ok("paper_only_false")

    # 16. ALLOW_LIVE_ORDERS flag
    if cfg.allow_live_orders:
        ok("allow_live_orders=true")
    else:
        fail("ALLOW_LIVE_ORDERS not set to true")

    # 17. Confirmation phrase
    if cfg.live_confirmation_phrase == _REQUIRED_LIVE_PHRASE:
        ok("live_confirmation_phrase_matched")
    else:
        fail(
            f"live_confirmation_phrase_mismatch: "
            f"set LIVE_CONFIRMATION_PHRASE={_REQUIRED_LIVE_PHRASE}"
        )

    if len(failed) > failure_count_before_live_gates:
        return

    # 18. Per-trade live size safety cap
    if cfg.max_trade_dollars <= cfg.max_live_dollars_per_trade:
        ok(
            f"live_size_cap_ok: ${cfg.max_trade_dollars:.2f} <= "
            f"${cfg.max_live_dollars_per_trade:.2f}"
        )
    else:
        fail(
            f"live_size_cap_exceeded: MAX_DOLLARS_PER_TRADE=${cfg.max_trade_dollars:.2f} "
            f"> MAX_LIVE_DOLLARS_PER_TRADE=${cfg.max_live_dollars_per_trade:.2f}"
        )

    # 19. Paper days completed
    paper_days = db.count_paper_trading_days()
    if paper_days >= cfg.min_paper_days_before_live:
        ok(f"paper_days_ok: {paper_days} >= {cfg.min_paper_days_before_live}")
    else:
        fail(
            f"paper_days_insufficient: {paper_days} < "
            f"{cfg.min_paper_days_before_live} required"
        )

    # 20. Paper trade count
    paper_count = db.count_completed_trades()
    if paper_count >= cfg.min_paper_trades_before_live:
        ok(f"paper_count_ok: {paper_count} >= {cfg.min_paper_trades_before_live}")
    else:
        fail(
            f"paper_count_insufficient: {paper_count} < "
            f"{cfg.min_paper_trades_before_live} required"
        )

    # 21. Paper P&L non-negative
    paper_pnl = db.total_paper_pnl()
    if paper_pnl >= cfg.min_paper_pnl_before_live:
        ok(f"paper_pnl_ok: ${paper_pnl:+.2f}")
    else:
        fail(
            f"paper_pnl_insufficient: ${paper_pnl:+.2f} < "
            f"${cfg.min_paper_pnl_before_live:.2f} required"
        )


def _build(
    decision: PositionSize,
    mode: str,
    approved: bool,
    reason: str,
    passed: list[str],
    failed: list[str],
) -> RiskAssessment:
    return RiskAssessment(
        ticker=decision.ticker,
        side=decision.side,
        approved=approved,
        reason=reason,
        mode=mode,
        checks_passed=passed,
        checks_failed=failed,
    )
