# Black Gibbie Dashboard Prototype

This is a local-only React + Vite prototype for visualizing the summary JSON shape returned by `scripts/paper_scanner_run.py`.

It uses bundled mock data from `src/mockScannerData.js`. It does not call Kalshi, does not read `.env`, does not connect to trading code, and does not insert paper or live trades.

## Run Locally

From the repo root:

```powershell
cd dashboard
npm install
npm run dev
```

Then open the Vite URL shown in the terminal, usually:

```text
http://127.0.0.1:5173/
```

## Prototype Sections

- Safety Banner
- Scan Summary cards
- Skip Reason table
- Trade Decisions table
- Paper Trades summary
- Errors panel

## Data Shape

The mock scanner object includes:

- `raw_markets`
- `normalized_markets`
- `passed_count`
- `rejected_count`
- `pass_rate`
- `skip_reason_counts`
- `skip_reason_examples`
- `candidates_analyzed`
- `execute_paper`
- `dry_run`
- `decisions`
- `paper_trades_inserted`
- `errors`
