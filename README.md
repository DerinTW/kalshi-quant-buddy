# Black Gibbie Kalshi Agent

This is a paper-first Kalshi prediction-market agent. It scans markets, researches them, estimates probabilities, computes edge, applies deterministic risk gates, sizes conservatively, executes dry-run or paper trades, monitors positions, and writes postmortems for losses.

Live trading is disabled by default. Do not treat this as financial advice or as a production trading system.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create local environment settings:

```powershell
Copy-Item .env.example .env
```

Keep the safe defaults unless you are explicitly testing a narrow path:

```text
KILL_SWITCH=true
TRADING_MODE=paper
LIVE_TRADING_ENABLED=false
ALLOW_LIVE_ORDERS=false
```

Initialize storage and logs:

```powershell
python main.py --init-only
```

## Running Safely

The default state is paper-only. The risk manager remains the hard gate before trade execution, and `main.py` does not start an autonomous live trading loop.

Live orders require all explicit live flags, the confirmation phrase, paper-trading requirements, position-size caps, and risk-manager approval. The LLM cannot override risk controls.

## Tests

Run the suite:

```powershell
python -m pytest
```

## Persistence

SQLite persistence lives in `db.py`. `storage.py` is a thin compatibility wrapper around `db.py`.

Rules files live under `rules/`. Proposed rule changes are written to `rules/rules_pending_review.json` for human review only. The bot must not auto-edit active rules, `.env`, or config defaults.
