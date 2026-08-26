# Black Gibbie Kalshi Agent

[![tests](https://github.com/DerinTW/black-gibbie/actions/workflows/tests.yml/badge.svg)](https://github.com/DerinTW/black-gibbie/actions/workflows/tests.yml)

This is a paper-first Kalshi prediction-market agent. It scans markets, researches them, estimates probabilities, computes edge, applies deterministic risk gates, sizes conservatively, executes dry-run or paper trades, monitors positions, and writes postmortems for losses.

Live trading is disabled by default. Do not treat this as financial advice or as a production trading system.

## Setup

Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

or bash:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create local environment settings (PowerShell):

```powershell
Copy-Item .env.example .env
```

or bash:

```bash
cp .env.example .env
```

Keep the safe defaults unless you are explicitly testing a narrow path:

```text
KILL_SWITCH=true
TRADING_MODE=paper
LIVE_TRADING_ENABLED=false
ALLOW_LIVE_ORDERS=false
```

Initialize storage and logs:

```bash
python main.py --init-only
```

## Running Safely

The default state is paper-only. The risk manager remains the hard gate before trade execution, and `main.py` does not start an autonomous live trading loop.

Live orders require all explicit live flags, the confirmation phrase, paper-trading requirements, position-size caps, and risk-manager approval. The LLM cannot override risk controls.

Scanner filters are **mode-aware**: `config.py` carries a permissive paper default and a stricter live default for each threshold, so relaxing a gate for paper research can never relax it for live orders. Paper / live respectively: `MIN_VOLUME_24H` 50 / 1000 (alias `MIN_VOLUME`), `MIN_YES_PRICE` 1 / 15, `MAX_SPREAD_CENTS` 10 / 6, `MAX_ORDERBOOK_AGE_SECONDS` 300 / 60, `MIN_ORDERBOOK_DEPTH_AT_LIMIT` 25 / 100. `MAX_TIME_TO_RESOLUTION_HOURS=72` (alias `MAX_MINUTES_TO_EXPIRY=4320`) applies to both. Any threshold can be overridden per mode via a `LIVE_`-prefixed variable — see `.env.example`. The default tradable categories are crypto, finance, economics, commodities, climate, tech & science, and culture; obvious sports event prefixes are skipped early through `BLOCKED_EVENT_PREFIXES`.

Position sizing also applies a capital-velocity haircut for longer settlement windows: size is multiplied by `1 / sqrt(days_to_settlement)`, capped at `1.0` so intraday markets are not boosted. `TIME_TO_RESOLUTION_SIZE_FLOOR_DAYS` defaults to `1.0`.

## Tests

Run the suite:

```bash
python -m pytest
```

## Persistence

SQLite persistence lives in `db.py`. `storage.py` is a thin compatibility wrapper around `db.py`.

Rules files live under `rules/`. Proposed rule changes are written to `rules/rules_pending_review.json` for human review only. The bot must not auto-edit active rules, `.env`, or config defaults.

## License

MIT — see `LICENSE`.
