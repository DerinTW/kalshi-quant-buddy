from __future__ import annotations
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Optional

from models import TradeRecord, Postmortem, ResearchItem

_db_path: str = "./black_gibbie.db"


def init(db_path: str) -> None:
    global _db_path
    _db_path = db_path
    with _conn() as conn:
        _create_tables(conn)


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(_db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            contracts INTEGER NOT NULL,
            entry_price_cents INTEGER NOT NULL,
            dollars_at_risk REAL NOT NULL,
            mode TEXT NOT NULL,
            thesis TEXT DEFAULT '',
            estimated_yes_prob REAL DEFAULT 0,
            timestamp TEXT NOT NULL,
            exit_price_cents INTEGER,
            exit_timestamp TEXT,
            pnl_dollars REAL,
            result TEXT DEFAULT 'open'
        );

        CREATE TABLE IF NOT EXISTS postmortems (
            trade_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            original_thesis TEXT,
            estimated_yes_prob REAL,
            market_price_at_entry INTEGER,
            actual_result TEXT,
            was_variance INTEGER,
            data_was_stale INTEGER,
            resolution_handled_correctly INTEGER,
            liquidity_hurt INTEGER,
            sizing_appropriate INTEGER,
            analysis TEXT,
            rule_change_proposal TEXT,
            human_approved INTEGER DEFAULT 0,
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS research_cache (
            ticker TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            raw_text TEXT,
            sources TEXT,
            timestamp TEXT NOT NULL,
            PRIMARY KEY (ticker, query_hash)
        );

        CREATE TABLE IF NOT EXISTS scan_runs (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            mode TEXT,
            category TEXT,
            limit_requested INTEGER,
            raw_markets INTEGER DEFAULT 0,
            normalized_markets INTEGER DEFAULT 0,
            passed_count INTEGER DEFAULT 0,
            rejected_count INTEGER DEFAULT 0,
            candidates_analyzed INTEGER DEFAULT 0,
            paper_trades_inserted INTEGER DEFAULT 0,
            errors TEXT DEFAULT '[]',
            summary_json TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS market_audit (
            id TEXT PRIMARY KEY,
            scan_run_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            event_ticker TEXT,
            title TEXT,
            category TEXT,
            status TEXT,
            stage TEXT NOT NULL,
            outcome TEXT,
            skip_reason TEXT,
            skip_reason_key TEXT,
            decision_action TEXT,
            risk_summary TEXT,
            edge_cents REAL,
            confidence REAL,
            yes_bid INTEGER,
            yes_ask INTEGER,
            no_bid INTEGER,
            no_ask INTEGER,
            spread_cents INTEGER,
            spread_pct REAL,
            volume INTEGER,
            volume_24h INTEGER,
            liquidity_dollars REAL,
            open_interest INTEGER,
            minutes_to_close REAL,
            minutes_to_settlement REAL,
            orderbook_depth INTEGER,
            orderbook_depth_fetched INTEGER DEFAULT 0,
            estimated_yes_prob REAL,
            side TEXT,
            limit_price_cents INTEGER,
            contracts INTEGER,
            dollar_size REAL,
            thesis TEXT,
            raw_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(scan_run_id) REFERENCES scan_runs(id)
        );

        CREATE INDEX IF NOT EXISTS idx_trades_ticker ON paper_trades(ticker);
        CREATE INDEX IF NOT EXISTS idx_trades_result ON paper_trades(result);
        CREATE INDEX IF NOT EXISTS idx_scan_runs_started ON scan_runs(started_at);
        CREATE INDEX IF NOT EXISTS idx_market_audit_scan ON market_audit(scan_run_id);
        CREATE INDEX IF NOT EXISTS idx_market_audit_ticker ON market_audit(ticker);
        CREATE INDEX IF NOT EXISTS idx_market_audit_stage ON market_audit(stage);
        CREATE INDEX IF NOT EXISTS idx_market_audit_outcome ON market_audit(outcome);
        CREATE INDEX IF NOT EXISTS idx_market_audit_skip_reason ON market_audit(skip_reason_key);
    """)
    _ensure_column(conn, "market_audit", "orderbook_depth_fetched", "INTEGER DEFAULT 0")


_MARKET_AUDIT_COLUMNS = {
    "id", "scan_run_id", "ticker", "event_ticker", "title", "category", "status",
    "stage", "outcome", "skip_reason", "skip_reason_key", "decision_action",
    "risk_summary", "edge_cents", "confidence", "yes_bid", "yes_ask", "no_bid",
    "no_ask", "spread_cents", "spread_pct", "volume", "volume_24h",
    "liquidity_dollars", "open_interest", "minutes_to_close",
    "minutes_to_settlement", "orderbook_depth", "orderbook_depth_fetched",
    "estimated_yes_prob", "side", "limit_price_cents", "contracts",
    "dollar_size", "thesis", "raw_json", "created_at",
}


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except Exception:
        return json.dumps(str(value))


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _decode_json_field(record: dict[str, Any], key: str, default: Any) -> None:
    raw = record.get(key)
    if raw in (None, ""):
        record[key] = default
        return
    try:
        record[key] = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        record[key] = default


# ── Scan/audit operations ────────────────────────────────────────────────────

def create_scan_run(
    *,
    mode: str = "",
    category: Optional[str] = None,
    limit_requested: Optional[int] = None,
) -> str:
    scan_id = str(uuid.uuid4())
    with _conn() as conn:
        conn.execute("""
            INSERT INTO scan_runs
            (id, started_at, mode, category, limit_requested)
            VALUES (?,?,?,?,?)
        """, (scan_id, _utc_now_iso(), mode, category, limit_requested))
    return scan_id


def finish_scan_run(scan_run_id: str, summary: dict[str, Any]) -> None:
    errors = summary.get("errors", [])
    with _conn() as conn:
        conn.execute("""
            UPDATE scan_runs
            SET finished_at=?,
                raw_markets=?,
                normalized_markets=?,
                passed_count=?,
                rejected_count=?,
                candidates_analyzed=?,
                paper_trades_inserted=?,
                errors=?,
                summary_json=?
            WHERE id=?
        """, (
            _utc_now_iso(),
            int(summary.get("raw_markets", summary.get("markets_fetched", 0)) or 0),
            int(summary.get("normalized_markets", 0) or 0),
            int(summary.get("passed_count", summary.get("markets_passed_filters", 0)) or 0),
            int(summary.get("rejected_count", 0) or 0),
            int(summary.get("candidates_analyzed", 0) or 0),
            int(summary.get("paper_trades_inserted", 0) or 0),
            _json_dumps(errors),
            _json_dumps(summary),
            scan_run_id,
        ))


def insert_market_audit(record: dict[str, Any]) -> None:
    insert_many_market_audits([record])


def insert_many_market_audits(records: list[dict[str, Any]]) -> None:
    prepared = [_prepare_market_audit_record(record) for record in records]
    if not prepared:
        return
    columns = [
        "id", "scan_run_id", "ticker", "event_ticker", "title", "category",
        "status", "stage", "outcome", "skip_reason", "skip_reason_key",
        "decision_action", "risk_summary", "edge_cents", "confidence",
        "yes_bid", "yes_ask", "no_bid", "no_ask", "spread_cents",
        "spread_pct", "volume", "volume_24h", "liquidity_dollars",
        "open_interest", "minutes_to_close", "minutes_to_settlement",
        "orderbook_depth", "orderbook_depth_fetched", "estimated_yes_prob",
        "side", "limit_price_cents", "contracts", "dollar_size", "thesis",
        "raw_json", "created_at",
    ]
    placeholders = ",".join("?" for _ in columns)
    with _conn() as conn:
        conn.executemany(
            f"INSERT INTO market_audit ({','.join(columns)}) VALUES ({placeholders})",
            [[record.get(column) for column in columns] for record in prepared],
        )


def _prepare_market_audit_record(record: dict[str, Any]) -> dict[str, Any]:
    prepared = {key: record.get(key) for key in _MARKET_AUDIT_COLUMNS}
    prepared["id"] = prepared.get("id") or str(uuid.uuid4())
    prepared["created_at"] = prepared.get("created_at") or _utc_now_iso()
    prepared["ticker"] = str(prepared.get("ticker") or "")
    prepared["scan_run_id"] = str(prepared.get("scan_run_id") or "")
    prepared["stage"] = str(prepared.get("stage") or "")
    raw_json = prepared.get("raw_json")
    if raw_json is not None and not isinstance(raw_json, str):
        prepared["raw_json"] = _json_dumps(raw_json)
    return prepared


def get_recent_scan_runs(limit: int = 20) -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM scan_runs ORDER BY started_at DESC LIMIT ?",
            (max(0, int(limit)),),
        ).fetchall()
    records = [_row_to_dict(row) for row in rows]
    for record in records:
        _decode_json_field(record, "errors", [])
        _decode_json_field(record, "summary_json", {})
    return records


def get_market_audit(
    scan_run_id: Optional[str] = None,
    outcome: Optional[str] = None,
    stage: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: int = 1000,
    *,
    category: Optional[str] = None,
    skip_reason_key: Optional[str] = None,
    min_liquidity: Optional[float] = None,
    max_spread: Optional[float] = None,
    min_minutes_to_close: Optional[float] = None,
    max_minutes_to_close: Optional[float] = None,
) -> list[dict[str, Any]]:
    where: list[str] = []
    params: list[Any] = []

    def add_exact(column: str, value: Optional[str]) -> None:
        if value:
            where.append(f"{column}=?")
            params.append(value)

    add_exact("scan_run_id", scan_run_id)
    add_exact("outcome", outcome if outcome != "all" else None)
    add_exact("stage", stage if stage != "all" else None)
    add_exact("category", category if category != "all" else None)
    add_exact("skip_reason_key", skip_reason_key if skip_reason_key != "all" else None)
    if ticker:
        where.append("(ticker LIKE ? OR title LIKE ?)")
        like = f"%{ticker}%"
        params.extend([like, like])
    if min_liquidity is not None:
        where.append("liquidity_dollars >= ?")
        params.append(float(min_liquidity))
    if max_spread is not None:
        where.append("spread_cents <= ?")
        params.append(float(max_spread))
    if min_minutes_to_close is not None:
        where.append("minutes_to_close >= ?")
        params.append(float(min_minutes_to_close))
    if max_minutes_to_close is not None:
        where.append("minutes_to_close <= ?")
        params.append(float(max_minutes_to_close))

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(max(0, int(limit)))
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM market_audit {clause} "
            "ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_dashboard_summary() -> dict[str, Any]:
    runs = get_recent_scan_runs(limit=1)
    latest = runs[0] if runs else None
    scan_run_id = latest["id"] if latest else None

    skip_reason_counts: dict[str, int] = {}
    skip_reason_examples: dict[str, list[str]] = {}
    if scan_run_id:
        with _conn() as conn:
            rows = conn.execute("""
                SELECT skip_reason_key, COUNT(*) AS count
                FROM market_audit
                WHERE scan_run_id=? AND outcome='skipped'
                GROUP BY skip_reason_key
                ORDER BY count DESC
            """, (scan_run_id,)).fetchall()
            example_rows = conn.execute("""
                SELECT skip_reason_key, ticker
                FROM market_audit
                WHERE scan_run_id=? AND outcome='skipped'
                ORDER BY created_at ASC
            """, (scan_run_id,)).fetchall()
        skip_reason_counts = {
            str(row["skip_reason_key"] or "unknown"): int(row["count"])
            for row in rows
        }
        for row in example_rows:
            key = str(row["skip_reason_key"] or "unknown")
            skip_reason_examples.setdefault(key, [])
            if len(skip_reason_examples[key]) < 3:
                skip_reason_examples[key].append(row["ticker"])

    with _conn() as conn:
        trade_counts = conn.execute("""
            SELECT
                SUM(CASE WHEN result='open' THEN 1 ELSE 0 END) AS open_count,
                SUM(CASE WHEN result!='open' THEN 1 ELSE 0 END) AS closed_count,
                COALESCE(SUM(CASE WHEN result='open' THEN dollars_at_risk ELSE 0 END),0) AS open_exposure,
                COALESCE(SUM(CASE WHEN pnl_dollars IS NOT NULL THEN pnl_dollars ELSE 0 END),0) AS total_pnl
            FROM paper_trades
        """).fetchone()
    return {
        "latest_scan_run": latest,
        "skip_reason_counts": skip_reason_counts,
        "skip_reason_examples": skip_reason_examples,
        "paper": {
            "open_trades": int(trade_counts["open_count"] or 0) if trade_counts else 0,
            "closed_trades": int(trade_counts["closed_count"] or 0) if trade_counts else 0,
            "open_exposure": float(trade_counts["open_exposure"] or 0.0) if trade_counts else 0.0,
            "total_pnl": float(trade_counts["total_pnl"] or 0.0) if trade_counts else 0.0,
        },
    }


# ── Trade operations ──────────────────────────────────────────────────────────

def insert_trade(trade: TradeRecord) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO paper_trades
            (id, ticker, side, contracts, entry_price_cents, dollars_at_risk, mode,
             thesis, estimated_yes_prob, timestamp, exit_price_cents, exit_timestamp,
             pnl_dollars, result)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            trade.id, trade.ticker, trade.side, trade.contracts,
            trade.entry_price_cents, trade.dollars_at_risk, trade.mode,
            trade.thesis, trade.estimated_yes_prob,
            trade.timestamp.isoformat(),
            trade.exit_price_cents,
            trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
            trade.pnl_dollars, trade.result or "open",
        ))


def close_trade(trade_id: str, exit_price_cents: int, pnl_dollars: float, result: str) -> None:
    with _conn() as conn:
        conn.execute("""
            UPDATE paper_trades
            SET exit_price_cents=?, exit_timestamp=?, pnl_dollars=?, result=?
            WHERE id=?
        """, (exit_price_cents, _utc_now_iso(), pnl_dollars, result, trade_id))


def get_open_trades() -> list[TradeRecord]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE result='open'"
        ).fetchall()
    return [_row_to_trade(r) for r in rows]


def get_closed_trades() -> list[TradeRecord]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades WHERE result != 'open'"
        ).fetchall()
    return [_row_to_trade(r) for r in rows]


def get_paper_trades(limit: int = 1000) -> list[dict[str, Any]]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_trades ORDER BY timestamp DESC LIMIT ?",
            (max(0, int(limit)),),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def count_completed_trades() -> int:
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM paper_trades WHERE result IN ('win','loss','push')"
        ).fetchone()
    return row[0] if row else 0


def total_paper_pnl() -> float:
    with _conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(pnl_dollars),0) FROM paper_trades WHERE pnl_dollars IS NOT NULL"
        ).fetchone()
    return row[0] if row else 0.0


def count_paper_trading_days() -> int:
    """Count distinct calendar days on which at least one paper trade completed."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT date(timestamp)) FROM paper_trades "
            "WHERE result IN ('win','loss','push') AND mode IN ('paper','dry_run')"
        ).fetchone()
    return row[0] if row else 0


def total_open_exposure() -> float:
    with _conn() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(dollars_at_risk),0) FROM paper_trades WHERE result='open'"
        ).fetchone()
    return row[0] if row else 0.0


def _row_to_trade(row: sqlite3.Row) -> TradeRecord:
    return TradeRecord(
        id=row["id"],
        ticker=row["ticker"],
        side=row["side"],
        contracts=row["contracts"],
        entry_price_cents=row["entry_price_cents"],
        dollars_at_risk=row["dollars_at_risk"],
        mode=row["mode"],
        thesis=row["thesis"] or "",
        estimated_yes_prob=row["estimated_yes_prob"] or 0.0,
        timestamp=datetime.fromisoformat(row["timestamp"]),
        exit_price_cents=row["exit_price_cents"],
        exit_timestamp=datetime.fromisoformat(row["exit_timestamp"]) if row["exit_timestamp"] else None,
        pnl_dollars=row["pnl_dollars"],
        result=row["result"],
    )


# ── Postmortem operations ─────────────────────────────────────────────────────

def insert_postmortem(pm: Postmortem) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO postmortems
            (trade_id, ticker, original_thesis, estimated_yes_prob, market_price_at_entry,
             actual_result, was_variance, data_was_stale, resolution_handled_correctly,
             liquidity_hurt, sizing_appropriate, analysis, rule_change_proposal,
             human_approved, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pm.trade_id, pm.ticker, pm.original_thesis, pm.estimated_yes_prob,
            pm.market_price_at_entry, pm.actual_result,
            int(pm.was_variance), int(pm.data_was_stale),
            int(pm.resolution_handled_correctly), int(pm.liquidity_hurt),
            int(pm.sizing_appropriate), pm.analysis, pm.rule_change_proposal,
            int(pm.human_approved), pm.timestamp.isoformat(),
        ))


def postmortem_exists(trade_id: str) -> bool:
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM postmortems WHERE trade_id=? LIMIT 1",
            (trade_id,),
        ).fetchone()
    return row is not None


# ── Research cache ────────────────────────────────────────────────────────────

def get_cached_research(
    ticker: str, query_hash: str, max_age_minutes: int = 60
) -> Optional[list[ResearchItem]]:
    """Return cached ResearchItem list, or None if absent/expired."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT sources, timestamp FROM research_cache WHERE ticker=? AND query_hash=?",
            (ticker, query_hash),
        ).fetchone()
    if not row:
        return None
    cached_at = _as_aware_utc(datetime.fromisoformat(row["timestamp"]))
    if (_utc_now() - cached_at).total_seconds() / 60 > max_age_minutes:
        return None
    return _deserialize_items(row["sources"] or "[]")


def set_cached_research(ticker: str, query_hash: str, items: list[ResearchItem]) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO research_cache (ticker, query_hash, raw_text, sources, timestamp)
            VALUES (?,?,?,?,?)
        """, (
            ticker, query_hash,
            "",                                     # raw_text col kept for schema compat
            _serialize_items(items),
            _utc_now_iso(),
        ))


def _serialize_items(items: list[ResearchItem]) -> str:
    out = []
    for item in items:
        out.append({
            "source": item.source,
            "url": item.url,
            "published_at": item.published_at.isoformat() if item.published_at else None,
            "claim": item.claim,
            "direction": item.direction,
            "relevance": item.relevance,
            "credibility": item.credibility,
            "recency_score": item.recency_score,
            "summary": item.summary,
            "agent": item.agent,
        })
    return json.dumps(out)


def _deserialize_items(items_json: str) -> list[ResearchItem]:
    raw = json.loads(items_json)
    items = []
    for d in raw:
        published_at = None
        if d.get("published_at"):
            try:
                published_at = datetime.fromisoformat(d["published_at"])
            except ValueError:
                pass
        items.append(ResearchItem(
            source=d.get("source", ""),
            url=d.get("url", ""),
            published_at=published_at,
            claim=d.get("claim", ""),
            direction=d.get("direction", "unclear"),
            relevance=float(d.get("relevance", 0.5)),
            credibility=float(d.get("credibility", 0.5)),
            recency_score=float(d.get("recency_score", 0.5)),
            summary=d.get("summary", ""),
            agent=d.get("agent", ""),
        ))
    return items
