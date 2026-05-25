from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db
from config import get_config

STATIC_DIR = Path(__file__).resolve().parent / "static"

cfg = get_config()
db.init(cfg.db_path)

app = FastAPI(title="Black Gibbie Paper Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/summary")
def summary() -> dict:
    return db.get_dashboard_summary()


@app.get("/api/scan-runs")
def scan_runs(limit: int = 20) -> list[dict]:
    return db.get_recent_scan_runs(limit=min(max(1, int(limit)), 200))


@app.get("/api/audit")
def audit(
    scan_run_id: Optional[str] = None,
    outcome: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    ticker: Optional[str] = None,
    skip_reason_key: Optional[str] = None,
    min_liquidity: Optional[float] = None,
    max_spread: Optional[float] = None,
    min_minutes_to_close: Optional[float] = None,
    max_minutes_to_close: Optional[float] = None,
    limit: int = 1000,
) -> list[dict]:
    return db.get_market_audit(
        scan_run_id=scan_run_id,
        outcome=outcome,
        stage=stage,
        ticker=ticker,
        limit=min(max(1, int(limit)), 10000),
        category=category,
        skip_reason_key=skip_reason_key,
        min_liquidity=min_liquidity,
        max_spread=max_spread,
        min_minutes_to_close=min_minutes_to_close,
        max_minutes_to_close=max_minutes_to_close,
    )


@app.get("/api/skip-reasons")
def skip_reasons(scan_run_id: Optional[str] = None) -> list[dict]:
    if scan_run_id is None:
        runs = db.get_recent_scan_runs(limit=1)
        scan_run_id = runs[0]["id"] if runs else None
    if not scan_run_id:
        return []
    rows = db.get_market_audit(
        scan_run_id=scan_run_id,
        outcome="skipped",
        limit=10000,
    )
    counts: dict[str, dict] = {}
    for row in rows:
        key = row.get("skip_reason_key") or "unknown"
        item = counts.setdefault(
            key,
            {"reason": key, "count": 0, "examples": [], "scan_run_id": scan_run_id},
        )
        item["count"] += 1
        if len(item["examples"]) < 3:
            item["examples"].append(row.get("ticker"))
    return sorted(counts.values(), key=lambda item: -item["count"])


@app.get("/api/paper-trades")
def paper_trades(limit: int = 1000) -> list[dict]:
    return db.get_paper_trades(limit=min(max(1, int(limit)), 10000))


@app.get("/api/export/audit.csv")
def export_audit_csv(
    scan_run_id: Optional[str] = None,
    outcome: Optional[str] = None,
    stage: Optional[str] = None,
    category: Optional[str] = None,
    ticker: Optional[str] = None,
    skip_reason_key: Optional[str] = None,
    min_liquidity: Optional[float] = None,
    max_spread: Optional[float] = None,
    min_minutes_to_close: Optional[float] = None,
    max_minutes_to_close: Optional[float] = None,
) -> StreamingResponse:
    rows = audit(
        scan_run_id=scan_run_id,
        outcome=outcome,
        stage=stage,
        category=category,
        ticker=ticker,
        skip_reason_key=skip_reason_key,
        min_liquidity=min_liquidity,
        max_spread=max_spread,
        min_minutes_to_close=min_minutes_to_close,
        max_minutes_to_close=max_minutes_to_close,
        limit=10000,
    )
    return _csv_response(rows, "black_gibbie_audit.csv")


@app.get("/api/export/skip-reasons.csv")
def export_skip_reasons_csv(scan_run_id: Optional[str] = None) -> StreamingResponse:
    return _csv_response(skip_reasons(scan_run_id), "black_gibbie_skip_reasons.csv")


@app.get("/api/export/latest-scan.json")
def export_latest_scan_json() -> JSONResponse:
    runs = db.get_recent_scan_runs(limit=1)
    if not runs:
        return JSONResponse({})
    return JSONResponse(runs[0].get("summary_json") or {})


def _csv_response(rows: list[dict], filename: str) -> StreamingResponse:
    buffer = io.StringIO()
    fieldnames = sorted({key for row in rows for key in row.keys()})
    writer = csv.DictWriter(buffer, fieldnames=fieldnames or ["empty"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=8000, reload=False)
