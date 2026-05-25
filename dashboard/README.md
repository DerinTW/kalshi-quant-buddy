# Black Gibbie Paper Dashboard

Local-only observability for paper-mode scanner runs. It reads the SQLite audit
tables and paper ledger; it does not place orders.

## Install

From the repo root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Record A Scan

```powershell
.\.venv\Scripts\python.exe scripts\paper_scanner_run.py --limit 4000 --category crypto
```

To allow approved decisions to enter the paper ledger:

```powershell
.\.venv\Scripts\python.exe scripts\paper_scanner_run.py --limit 4000 --category crypto --execute-paper
```

The dashboard reads the same scan summary written by the paper runner,
including canonical filter settings: `MIN_VOLUME_24H=1000` / `MIN_VOLUME`,
`MAX_TIME_TO_RESOLUTION_HOURS=72` / `MAX_MINUTES_TO_EXPIRY=4320`,
`MIN_YES_PRICE=15`, and `MIN_ORDERBOOK_DEPTH_AT_LIMIT=100`.

## Run The Dashboard

```powershell
.\.venv\Scripts\python.exe -m dashboard.app
```

Open:

```text
http://127.0.0.1:8000/
```

## API

- `GET /api/summary`
- `GET /api/scan-runs`
- `GET /api/audit`
- `GET /api/skip-reasons`
- `GET /api/paper-trades`
- `GET /api/export/audit.csv`
- `GET /api/export/skip-reasons.csv`
- `GET /api/export/latest-scan.json`
