from __future__ import annotations

"""
Compatibility layer for persistence.

db.py is the source of truth for SQLite schema and storage behavior. This
module exists so callers that expect a storage.py facade can use the same
implementation without duplicating database logic.
"""

import db
from models import Postmortem, TradeRecord


def init_storage(db_path: str) -> None:
    db.init(db_path)


def get_open_trades() -> list[TradeRecord]:
    return db.get_open_trades()


def get_closed_trades() -> list[TradeRecord]:
    return db.get_closed_trades()


def insert_trade(trade: TradeRecord) -> None:
    db.insert_trade(trade)


def close_trade(trade_id: str, exit_price_cents: int, pnl_dollars: float, result: str) -> None:
    db.close_trade(trade_id, exit_price_cents, pnl_dollars, result)


def insert_postmortem(pm: Postmortem) -> None:
    db.insert_postmortem(pm)
