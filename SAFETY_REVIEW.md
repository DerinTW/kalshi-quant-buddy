# Black Gibbie — Safety & Reality Review (Step 21)

Audit date: 2026-05-15
Branch: `main`
Test suite: `python -m pytest` → **336 passed, 0 failed**.
Entry points:
- `python main.py --init-only` → initialises log dir + sqlite DB, exits.
- `python main.py` → logs `startup_safe_mode`; **does not** start a loop.
- `python main.py --run-once` → one scan→evaluate→(maybe paper-execute) cycle; **no loop**.
- `python main.py --monitor-only` → **new in Step 21** — one pass of `monitor.check_positions`; **no loop, no live orders**.

Defaults in `config.py` and `.env`/`.env.example` were not changed in this step.

---

## 1. Live trading is default-OFF — every gate is positive

`risk_manager.assess` runs 15 deterministic gates plus 6 live-only gates. The first failure causes a hard reject (the kill switch is a short-circuit). `trading.execute` re-checks every live gate via `_live_gates_satisfied` before any order can be placed.

| Gate | Source | Default | Purpose |
| --- | --- | --- | --- |
| `KILL_SWITCH` | `cfg.kill_switch` | `true` | Hard stop on all trading. First check in `risk_manager.assess`; short-circuits. |
| `TRADING_MODE` | `cfg.trading_mode` | `paper` | Must equal `live` to even consider a real order. |
| `PAPER_ONLY` | `cfg.paper_only` | `true` | Independent of mode; if true, live is blocked. |
| `LIVE_TRADING_ENABLED` | `cfg.live_trading_enabled` | `false` | Must be flipped on AND mode=live. |
| `ALLOW_LIVE_ORDERS` | `cfg.allow_live_orders` | `false` | Last on-by-default safety flag. |
| `LIVE_CONFIRMATION_PHRASE` | `cfg.live_confirmation_phrase` | unset | Must equal exactly `I_UNDERSTAND_THIS_CAN_LOSE_MONEY`. |
| `MAX_LIVE_DOLLARS_PER_TRADE` | `cfg.max_live_dollars_per_trade` | `1` | Per-trade size cap when mode=live. |
| `MIN_PAPER_TRADES_BEFORE_LIVE` | `cfg.min_paper_trades_before_live` | `100` | Paper-volume earn-in. |
| `MIN_PAPER_TRADING_DAYS_BEFORE_LIVE` | `cfg.min_paper_days_before_live` | `7` | Paper-time earn-in (distinct calendar days). |
| `MIN_PAPER_PNL_BEFORE_LIVE` | `cfg.min_paper_pnl_before_live` | `0.0` | Non-negative paper P&L. |

All eleven gates default to "OFF for live". An order in live mode requires **all of them** to be satisfied simultaneously. The same checks run inside `risk_manager.assess` and again inside `trading._live_gates_satisfied` before `client.place_order` is called.

Tests pinning this down: `tests/test_risk_manager.py`, `tests/test_trading_modes.py`, `tests/test_risk_control_review.py`.

## 2. Kill switch

- `KILL_SWITCH=true` is the default; `risk_manager.assess` returns an `approved=False` `RiskAssessment` with reason `KILL_SWITCH is enabled — all trading halted` before any other check runs.
- `trading._live_gates_satisfied` re-checks the flag — live orders are blocked even if a stale `RiskAssessment` were somehow handed in.
- The pipeline's positive-list gate (`pipeline._execution_skip_reason`) additionally refuses to execute in live mode when `kill_switch` is true.
- Toggling the kill switch is environment-only (`.env`); the codebase never sets it.

## 3. Paper / dry-run behavior

- `TRADING_MODE=paper` is the default. `trading.execute` writes to sqlite `paper_trades` via `db.insert_trade` and never calls `client.place_order`.
- `TRADING_MODE=dry_run` writes structured logs to `logs/trades.jsonl` and `logs/agent.jsonl` but does not insert into sqlite — it's a pure observation mode.
- Paper / live execute paths both require `risk_approved=True`. `trading.execute` rejects the call with `risk_approval_required` if the caller forgot to pass it.
- The pipeline only sets `risk_approved=True` when `risk_assessment.approved` is actually `True`. The "positive-list" gate `_execution_skip_reason` enforces this; tests in `tests/test_pipeline.py` sweep across approve/reject permutations.

## 4. Exposure handling — Step 21 fix

Before Step 21, `pipeline.run_once` passed `category_exposure=0.0` and `correlated_exposure=0.0` to `risk_manager.assess`. The risk manager *accepted* them but the values were stubbed, so caps 8 (category) and 9 (correlated) could never bind.

Step 21 adds a deterministic helper:

```python
pipeline.compute_open_exposures(market, open_positions, markets_by_ticker)
    → (category_exposure: float, correlated_exposure: float)
```

- **Correlated exposure**: sums `dollars_at_risk` across open trades whose event group (derived via `market_scanner._derive_event_ticker`) matches the candidate's. Deterministic from ticker shape alone; no scan data required.
- **Category exposure**: `TradeRecord` does not store category. The helper looks up each open trade's ticker in the current scan's market map; if found, uses that market's category. If the open trade's ticker is **not** in the current scan (closed, filtered out, or otherwise absent), we **fail safe** and count the trade toward the candidate's category — i.e. the cap binds *more aggressively*, not less.
- Zero / negative `dollars_at_risk` are skipped.

The real values are passed to `risk_manager.assess` from `pipeline.run_once`. Tests (`tests/test_safety_step21.py`) demonstrate end-to-end rejection when either cap is exceeded.

### Known limitation
Until we add `category` to the `paper_trades` schema, category exposure for trades opened against markets that have since fallen out of the scan is conservatively over-counted. This is documented in the helper's docstring and surfaced in the agent log as a `debug` event. We will tighten this once `TradeRecord` carries category (a separate, scoped change with schema migration).

## 5. Duplicate trade prevention

`risk_manager.assess` enforces three duplicate-guard rules:
1. **Same ticker, same side** open → fail (`duplicate_position`).
2. **Same ticker, opposite side** open → fail when the action is an entry (`opposite_side_open`).
3. **Same event group** open under a different ticker → fail (`related_group_position_open`), using `market.event_ticker or market.ticker` and a prefix match on `f"{group}-"`.

`trading.live_buy_guard` is an in-process set keyed by `ticker:side`; even after risk approval a live order is rejected if the guard already has the key. The pipeline passes this set into `risk_manager.assess` via the `live_buy_guard` keyword.

## 6. Stale / missing data behavior

- **Filters drop, not crash.** `filters.run` returns a `FilterResult` with structured rejection reasons; nothing raises if a field is missing because `market_scanner._validate` marks the row `is_unsafe` instead.
- **Orderbook age.** `_check_orderbook_age` rejects when the most recent trade is older than `MAX_ORDERBOOK_AGE_SECONDS` (default 60s); skipped only when `last_trade_at` is unknown (no enrichment).
- **Orderbook depth.** `_check_orderbook_depth` rejects when top-of-book depth < `MIN_ORDERBOOK_DEPTH_AT_LIMIT`; skipped when the depth is unknown (0).
- **Stale book during anomaly check.** `weird_move.detect` classifies `stale_book_artifact` and applies the maximum confidence step-down (`_WEIRD_MOVE_STEPS=2`) in `prediction_model._step_down`.
- **Research absent.** `pipeline.run_once` substitutes an empty `ResearchResult(failed_reason="research_budget_or_error")` when the research budget is exhausted or the call raised. `sentiment.analyze` produces an honestly weak signal from it; the LLM is never used to invent missing research.
- **No Perplexity key.** `research_agents._search` returns an empty string and `_items_from_raw` short-circuits — we never fall back to the LLM for content. Tests in `tests/test_llm_role.py` pin this contract.
- **No Kalshi credentials.** `main._build_client_if_credentials_present` returns `None`; `pipeline.run_once` logs `no_client_skip_fetch` and returns an empty summary; `--monitor-only` logs `monitor_only_no_client` and exits cleanly.

## 7. Postmortem rule-change safety

- `postmortem.run_for_trade` only fires when `trade.result == "loss"`.
- Output is split into two surfaces:
  - the `postmortems` sqlite table for queryability;
  - proposed rule changes go to `rules/rules_pending_review.json`.
- `rules/base_rules.json` is **never modified** by the postmortem path; the test `tests/test_postmortem.py::test_rule_change_goes_to_pending_review_not_base_rules` asserts this directly.
- Each trade ID is deduped via an in-memory set and `db.postmortem_exists`; restarting the process and seeing the same trade does not re-run the prompt.
- If the LLM call fails, a deterministic fallback report is produced; we never silently drop the loss.

## 8. Monitor entry point — Step 21 addition

`python main.py --monitor-only`:
- Builds a Kalshi client iff credentials are present; otherwise logs `monitor_only_no_client` and exits.
- Calls `monitor.check_positions(client, cfg)` **exactly once** and returns.
- Any exception inside `monitor.check_positions` is logged via `logger.error` and swallowed — the CLI never propagates a stack trace.
- `monitor.check_positions` only ever calls `trading.close_paper_trade` (writes to sqlite) or logs. It never reaches `client.place_order` or `client.cancel_order`.
- Tests in `tests/test_safety_step21.py` cover:
  - exactly one call when client is present,
  - zero calls when client is absent,
  - exceptions inside monitor do not leak,
  - the CLI subprocess exits 0 promptly (no hang).

## 9. Remaining known limitations

These do not block Step 21 but should be tracked before Stage 7 (limited live mode) is opened:

1. **Category metadata not in `paper_trades`.** Exposure helper falls back conservatively, but adding the column closes the over-counting gap. Requires a small `ALTER TABLE` migration in `db._create_tables`.
2. **`monitor.check_positions` is still manual.** There is no scheduler. Acceptable for paper; before live, decide whether a `cron` or systemd timer wraps `--monitor-only`, or whether a separate `--paper-loop` is added (still off by default).
3. **No analytics module** (Stage 8). Calibration, Brier, realised EV, win-rate-by-category, slippage, drawdown, postmortem-cause histogram are unimplemented. The raw inputs are captured in `paper_trades` and `postmortems`.
4. **`datetime.utcnow()` deprecations.** ~1.3k warnings during the suite from `logger.py` and `postmortem.py`. Cosmetic; tracked separately.
5. **RSS feed URLs can rot.** No active health check. Today the `RSSNewsAgent` swallows feed errors and falls back to Perplexity.

---

## Files changed in Step 21

- `pipeline.py` — added `compute_open_exposures` helper; built `markets_by_ticker` in `run_once`; replaced hard-coded `0.0` exposures with computed values.
- `main.py` — added `--monitor-only` CLI path and `_monitor_only_cli` helper. Imports `monitor` lazily.
- `tests/test_safety_step21.py` — new file covering the exposure helper, end-to-end rejection through `risk_manager.assess`, and the `--monitor-only` one-shot guarantees.
- `SAFETY_REVIEW.md` — this document.

## Commands run

- `python -m pytest` → **336 passed, 0 failed**.
- `python main.py --init-only` → exits cleanly.
- `python main.py` → exits cleanly; no loop.
- `python main.py --monitor-only` → one monitor pass, "nothing to monitor", exits cleanly.

No `.env` defaults were changed; no safety check was relaxed. The kill switch remains `true` by default.
