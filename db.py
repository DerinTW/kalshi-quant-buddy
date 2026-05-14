from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

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

        CREATE INDEX IF NOT EXISTS idx_trades_ticker ON paper_trades(ticker);
        CREATE INDEX IF NOT EXISTS idx_trades_result ON paper_trades(result);
    """)


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
        """, (exit_price_cents, datetime.utcnow().isoformat(), pnl_dollars, result, trade_id))


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
    cached_at = datetime.fromisoformat(row["timestamp"])
    if (datetime.utcnow() - cached_at).total_seconds() / 60 > max_age_minutes:
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
            datetime.utcnow().isoformat(),
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
