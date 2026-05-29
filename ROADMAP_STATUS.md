# Black Gibbie — Step 20 Roadmap Status

Audit date: 2026-05-15
Branch: `main`
Test suite: `python -m pytest` → **425 passed, 0 failed** (UTC timestamp deprecation warnings resolved).
Entry points:
- `python main.py --init-only` → initialises log dir + sqlite DB, exits cleanly.
- `python main.py` → logs `startup_safe_mode` and exits. No autonomous loop is started, as required.

Legend: **DONE** / **PARTIAL** / **MISSING** / **NEEDS TEST** / **DO NOT BUILD YET**.

The roadmap covers eight stages. We are operating in **paper-only / dry-run mode**; every live-trading gate is still defaulted off (`KILL_SWITCH=true`, `TRADING_MODE=paper`, `LIVE_TRADING_ENABLED=false`, `ALLOW_LIVE_ORDERS=false`, `LIVE_CONFIRMATION_PHRASE` unset, `MAX_LIVE_DOLLARS_PER_TRADE=1`). Nothing in this audit changes that.

---

## Stage 1 — Paper-only scanner and logger — **DONE**

Files involved
- `config.py` — typed dataclass + env loading (kalshi/anthropic/perplexity keys, filter thresholds, risk caps, live gates, log/db paths).
- `kalshi_client.py` — Kalshi REST client (auth, market list, orderbook, trades, settlement).
- `market_scanner.py` — fetch → normalize → `Market` objects, marks unsafe rows instead of dropping.
- `models.py` — `Market`, `ResearchItem`, `ResearchResult`, `SentimentResult`, `ProbabilityEstimate`, `EdgeResult`, `PositionSize`, `RiskAssessment`, `TradeRecord`, `Postmortem`.
- `filters.py` — 11 single-market structural checks + duplicate-event-group dedup pass, structured `FilterResult` with `skip_reason_counts` / `skip_reason_examples`.
- `logger.py` — JSONL writers for `agent.jsonl`, `decisions.jsonl`, `trades.jsonl`, plus console mirror.
- `db.py` — sqlite schema (`paper_trades`, `postmortems`, `research_cache`) and helpers.
- `main.py` — paper-first CLI; `--init-only` and `--run-once` flags; default path never starts a loop.

What appears implemented
- Config-driven filter thresholds (price, spread cents + %, volume, liquidity, expiry window, orderbook age & depth, category allowlist, blocked tickers).
- Per-rejection reason logging plus a normalised summary suitable for "skipped summary".
- JSONL logs are wired through `logger.info/warn/error/decision/trade`.
- No LLM is invoked at scan/filter time; that is enforced by module imports (filters and market_scanner only import config/logger/models).
- No trading yet on this path: scanner→filter is callable in isolation via `pipeline.run_once` and never reaches `trading.execute` unless every upstream gate is positive.

What still needs work
- None for Stage 1.

Tests that cover it
- `tests/test_config_env.py` — env wiring.
- `tests/test_project_structure.py` — required modules/log files exist.
- `tests/test_pipeline.py` — `run_once` correctly aborts when no client/markets, surfaces filter rejection counts, never reaches execution without risk approval.
- `tests/test_schemas.py` — model shapes (incl. JSON-safety) for execution reports.

Tests missing
- Direct unit tests for `filters.run` paths (per-rejection-reason coverage matrix). Today filter behaviour is only exercised end-to-end through `test_pipeline.py`.
- A unit test on `market_scanner._validate` for each "unsafe" reason.

---

## Stage 2 — Research agents — **DONE (with documented X/Reddit caveats)**

Files involved
- `research_agents.py` — five agents (`OfficialSourceAgent`, `NewsWireAgent`, `RSSNewsAgent`, `TwitterXAgent`, `RedditAgent`, `MarketSpecificAgent`); `CREDIBILITY` table; `recency_score`; `deduplicate` (URL + Jaccard ≥ 0.65); `_query_hash`; 60-min `research_cache` read/write.
- `category_research.py` — preferred path: FRED/EIA/BLS/NOAA/NWS/SEC etc. with deterministic source scoring.
- `llm.py` — `extract_research_items` is a strict structured-extraction call (no inventing claims; short-circuits on empty raw text).
- `db.py` — `get_cached_research` / `set_cached_research`.

What appears implemented
- Real-time RSS pull + parse with keyword pre-filtering.
- Per-source credibility from the spec table; recency decay function; deduplication on URL then claim similarity.
- Market-specific query generator per category in `_MARKET_SOURCES` and `MarketSpecificAgent`.
- Perplexity as the only real-time web backend; explicit refusal to fall back to the LLM for content (only used for structured extraction). That satisfies the "do not invent missing data" rule from the blueprint.
- 60-minute cache via sqlite; budget cap (`MAX_RESEARCH_MARKETS_PER_RUN`).
- X / Reddit agents exist but downgrade unverified items to `CREDIBILITY["reddit"]`, matching the blueprint's "skip X/Reddit at first unless APIs are easy" intent — they're present but heavily de-weighted.

What still needs work
- X / Reddit currently rely on Perplexity site queries rather than real APIs. Acceptable per the blueprint ("skip…unless APIs are easy"); document this and consider gating them off behind a flag if they introduce noise.
- RSS feed URLs are best-effort and can rot; a periodic feed-health log would help.

Tests that cover it
- `tests/test_research_agent_prompt.py` — prompt construction.
- `tests/test_research_integration.py` — orchestration & dedup.
- `tests/test_llm_role.py` — ensures the LLM is used for extraction, never invention.

Tests missing
- Direct unit test of `deduplicate` against URL + Jaccard collisions.
- A unit test that confirms `_search` returns empty (not LLM-fabricated text) when `PERPLEXITY_API_KEY` is unset.

---

## Stage 3 — Probability model — **DONE**

Files involved
- `prediction_model.py` — ensemble (`_W_MARKET=0.70`, `_W_RESEARCH=0.25`, `_W_LLM=0.05`); `_SENTIMENT_SHIFT_CAP_PP=0.10`; `_LLM_SHIFT_CAP_PP=0.10`; confidence step-down rules (thin liquidity, near-resolution, sparse history, high volatility, neighbour disagreement, weird-move classifications).
- `sentiment.py` — deterministic R1–R5 narrative rules (no LLM call), produces `SentimentResult`.
- `features.py` — feature vector (liquidity, volatility, neighbours, history depth).
- `weird_move.py` — anomaly classifier feeding the step-down map.
- `edge.py` — side selection, cost adjustments (half-spread, slippage, fees), confidence-weighted EV, `passes_threshold` no-trade rules.
- `llm.py` — capped LLM probability adjustment.

What appears implemented
- Market baseline probability from `yes_ask` mid.
- Sentiment adjustment driven by `market_impact_estimate_cents` and direction, with a ±10pp cap.
- LLM probability adjustment is capped to ±10pp from market price.
- Edge calculator returns `EdgeResult` with both raw and adjusted edge; `passes_threshold` enforces all six no-trade conditions (raw edge, adjusted edge, conf-adj EV, confidence weight, spread gate, EV ≤ 0).

What still needs work
- None blocking. Probably worth a CHANGELOG entry that historical_model_prob's 0.15 weight is folded into market price (already commented in code).

Tests that cover it
- `tests/test_prediction_model.py`, `tests/test_probability_estimator.py` — ensemble math and step-downs.
- `tests/test_sentiment.py` — R1–R5 rules.
- `tests/test_edge.py` — side selection, cost stacking, threshold logic.
- `tests/test_features.py`, `tests/test_weird_move.py`.

Tests missing
- Property-style test that the final probability is always clamped to `[0.01, 0.99]` regardless of input.
- A regression test that the LLM adjustment cannot exceed `_LLM_SHIFT_CAP_PP` even with a malicious LLM payload (today only covered indirectly).

---

## Stage 4 — Risk and position sizing — **DONE**

Files involved
- `position_sizing.py` — flat sizing only; caps are `max_trade_dollars` (or `max_live_dollars_per_trade` in live mode), bankroll-% cap, 20% of liquidity, 20% of orderbook depth. Reads `db.total_open_exposure()`. No Kelly, no confidence scaling.
- `risk_manager.py` — 15 deterministic gates + 6 live-only gates: kill switch, mode validity, daily loss/trade circuit breakers, per-trade cap, bankroll %, category exposure, correlated exposure, spread, liquidity, duplicate ticker+side, time-to-resolution, edge-after-costs, positive adjusted EV; live gates: `ALLOW_LIVE_ORDERS`, `LIVE_CONFIRMATION_PHRASE` exact match, per-trade live cap, paper days, paper trade count, non-negative paper P&L.
- `trading.py` — `live_buy_guard` set deduplicates intra-run live buys; `_live_gates_satisfied` repeats every gate before any live order is placed.

What appears implemented
- Kill switch defaults true and short-circuits.
- Daily limits (`MAX_TRADES_PER_DAY`, `MAX_DAILY_LOSS`) wired through risk manager.
- Exposure limits (total, category, correlated) all present.
- Duplicate guard prevents buying the same ticker+side twice while a position is open.
- Flat sizing only.

What still needs work
- `pipeline.run_once` currently passes `category_exposure=0.0`, `correlated_exposure=0.0` to `risk_manager.assess`. Risk manager *receives* those numbers, so the limit is enforceable but not yet *computed* upstream. Add a helper in `db.py` or `pipeline.py` that derives category/correlated exposure from open trades and feed it in.

Tests that cover it
- `tests/test_position_sizing.py` — caps & zero-size paths.
- `tests/test_risk_manager.py` — every gate, including live gates.
- `tests/test_risk_control_review.py` — invariants across the gate matrix.

Tests missing
- A regression test that `pipeline.run_once` would fail risk if category exposure were already maxed (today the pipeline passes 0.0, so this can't fire). Add once the exposure computation is wired.

---

## Stage 5 — Paper trading loop — **PARTIAL**

Files involved
- `trading.py` — paper & dry-run execution paths; simulated fill at the spec'd limit price; live path is fully gated.
- `db.py` — `insert_trade`, `close_trade`, `get_open_trades`, `total_open_exposure`, `count_completed_trades`, `total_paper_pnl`, `count_paper_trading_days`.
- `monitor.py` — `check_positions(client, cfg)` runs over open trades, computes exit quotes, force-reviews near resolution, handles settlement payouts.
- `pipeline.py` — single-cycle scan→evaluate→(paper-execute) orchestrator.

What appears implemented
- Paper execution with simulated fills.
- Position ledger in sqlite; P&L recorded on close.
- Monitor logic that detects stop/take/edge-collapse exits and forces a review near close.
- Single-cycle pipeline (`run_once`) is fully exercised.

What still needs work
- **There is no scheduler/loop wrapping `pipeline.run_once` + `monitor.check_positions`.** This is intentional per the user's hard rule "do not start an autonomous loop", so it stays as a manual `--run-once` for now. Before promoting to a long-running paper loop we will need: a tick-rate config, a per-tick scan budget, and an explicit `--paper-loop` flag (still off by default).
- `monitor.check_positions` is not called from `pipeline.run_once`; nothing in the default code path ages open paper trades automatically.

Tests that cover it
- `tests/test_trading_modes.py` — paper / dry-run / live mode boundaries.
- `tests/test_monitor.py` — exit-quote logic and settlement handling.
- `tests/test_pipeline.py` — full single-cycle behaviour including risk gating.

Tests missing
- A test that runs `pipeline.run_once` immediately followed by `monitor.check_positions` against the same DB to verify a paper trade's full lifecycle (entry → monitor tick → exit → P&L). Today the two are tested separately.

---

## Stage 6 — Postmortem system — **DONE**

Files involved
- `postmortem.py` — `run_for_trade(trade, …)`; only triggers on `result == "loss"`; dedups via `_processed_trade_ids` and `db.postmortem_exists`; writes proposed rule changes to `rules/rules_pending_review.json`; never edits `rules/base_rules.json` or `.env`.
- `llm.py` — `run_postmortem` prompt (loss detector + structured analysis).
- `db.py` — `postmortems` table + `insert_postmortem` / `postmortem_exists`.
- `logger.py` — postmortem events flow through `agent.jsonl`.
- `rules/base_rules.json`, `rules/rules_pending_review.json`, `rules/blocked_markets.json`.

What appears implemented
- Losing-trade detector and "skip non-losses" guard.
- Postmortem JSONL/SQLite log.
- Pending rule changes file is the *only* surface for proposed changes; postmortems are explicitly "suggestions only".

What still needs work
- Postmortem currently uses sqlite (`postmortems` table) rather than a dedicated `postmortems.jsonl`. The blueprint says "JSONL postmortem logs"; consider whether the sqlite store is sufficient (it is queryable and survives restarts) or whether we want to mirror to JSONL for grep-ability.

Tests that cover it
- `tests/test_postmortem.py` — structured JSON output, dedup, pending-rules file, fallback when LLM is unavailable, rule changes never written to `base_rules.json`.

Tests missing
- A test that confirms `rules/base_rules.json` is unchanged after a postmortem run (today's tests imply it but don't assert the file hash).

---

## Stage 7 — Limited live mode — **DO NOT BUILD YET**

Status: scaffolding is present and **defaulted off**. Do not enable until Stages 5 & 6 have produced paper results.

What is already in place (no action needed before Step 21)
- `cfg.trading_mode` validated; live path in `trading.execute` re-checks every gate via `_live_gates_satisfied`.
- `MAX_LIVE_DOLLARS_PER_TRADE` defaults to `1`.
- Limit-orders-only is the behaviour `trading.execute` uses (`limit_price_cents`); market orders are never sent.
- `LIVE_CONFIRMATION_PHRASE` must equal `I_UNDERSTAND_THIS_CAN_LOSE_MONEY`; `ALLOW_LIVE_ORDERS` must be true; `KILL_SWITCH` must be false; min paper days / paper trades / paper P&L must all be satisfied.
- "No autonomous live trading during first live tests" is naturally enforced because `main.py` does not start a loop.

What still needs work *before* this stage is opened
- Run the suite (`pytest`) green — currently green.
- Accumulate `min_paper_trades_before_live` paper trades and `min_paper_days_before_live` distinct paper days; today both counters are zero.
- Decide on a separate manual-confirmation flow (CLI prompt vs env var). Today the only knob is `LIVE_CONFIRMATION_PHRASE`; that satisfies the spec but a `--confirm-live` CLI flag would be cleaner.

Tests that cover it
- `tests/test_trading_modes.py` — paper default does not place live; live cancel-on-timeout; guard key.
- `tests/test_risk_manager.py` — all 6 live-only gates.

Tests missing
- An end-to-end test that drives `cfg.trading_mode="live"` with **all** gates satisfied except one, asserting the live order is blocked for *each* of the six live gates. (Several individual gates are tested; the per-gate exhaustive matrix isn't.)

---

## Stage 8 — Evaluation and improvement — **MISSING**

Files involved (today)
- `logger.py` — emits structured decision and trade events; raw substrate for calibration math exists.
- `db.py` — closed trades + postmortems queryable.

What appears implemented
- The *data* needed (per-decision `estimated_yes_prob`, `entry_price_cents`, exit price, P&L, category, edge, sentiment confidence, postmortem causes) is being captured.

What still needs work — none of these are blocking Step 20, but they ARE Stage 8 itself
- No calibration / Brier / realized-EV report module.
- No "win rate by category" or "average edge realised" rollup.
- No slippage measurement (entry-vs-fill cents).
- No drawdown / equity-curve report.
- No aggregator that groups postmortem causes.

Suggested follow-up file: a single read-only `analytics.py` that ingests `paper_trades` + `postmortems` and emits the eight metrics. Should not import `trading.py` or anything that mutates state.

Tests that cover it
- None.

Tests missing
- All of them (calibration, Brier, realised-EV, win-rate-by-category, average-edge, slippage, drawdown, postmortem-cause histogram).

---

## Remaining gaps before Step 21

Blocking
1. **Stage 8 evaluation module is absent.** Need `analytics.py` (or equivalent) before we can rationally enable Stage 7 live mode.
2. **Pipeline does not compute category/correlated exposure.** Risk manager *accepts* them but the pipeline passes 0.0. Wire a `db` helper that aggregates open exposure by category and by event-group and pass the real numbers in.
3. **Monitor loop is not called from any orchestration.** Either wire it into `pipeline.run_once` as an optional final step or add an explicit `--monitor-only` CLI flag.

Non-blocking polish
4. Add the missing unit tests called out under each stage above (filter matrix, dedup, LLM-cap regression, full lifecycle, base_rules.json invariant, live-gate matrix).
5. Decide whether postmortems should mirror to `logs/postmortems.jsonl` for parity with the blueprint wording.

Run results
- `python -m pytest` → 425 passed, 0 failed.
- `python main.py --init-only` → exits cleanly, logs `initialized`.
- `python main.py` → exits cleanly, logs `startup_safe_mode`, does **not** start a loop.
- No code changes were required to get the suite green; no safety checks were touched.
