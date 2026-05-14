from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


_log_dir: str = "./logs"
_initialized: bool = False


def init(log_dir: str) -> None:
    global _log_dir, _initialized
    _log_dir = log_dir
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _initialized = True


def _ensure_init() -> None:
    if not _initialized:
        Path(_log_dir).mkdir(parents=True, exist_ok=True)


def _entry(level: str, module: str, event: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ts": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "module": module,
        "event": event,
        **(data or {}),
    }


def _write(record: dict[str, Any], filename: str) -> None:
    _ensure_init()
    path = os.path.join(_log_dir, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _console(level: str, module: str, msg: str) -> None:
    ts = datetime.utcnow().strftime("%H:%M:%S")
    prefix = {"INFO": "·", "WARN": "!", "ERROR": "✗", "DEBUG": "·"}.get(level, "·")
    print(f"[{ts}] {prefix} [{module}] {msg}", file=sys.stdout if level != "ERROR" else sys.stderr)


# ── Public API ────────────────────────────────────────────────────────────────

def info(module: str, event: str, msg: str = "", **data: Any) -> None:
    record = _entry("INFO", module, event, {"msg": msg, **data} if msg or data else None)
    _write(record, "agent.jsonl")
    _console("INFO", module, msg or event)


def warn(module: str, event: str, msg: str = "", **data: Any) -> None:
    record = _entry("WARN", module, event, {"msg": msg, **data} if msg or data else None)
    _write(record, "agent.jsonl")
    _console("WARN", module, msg or event)


def error(module: str, event: str, msg: str = "", **data: Any) -> None:
    record = _entry("ERROR", module, event, {"msg": msg, **data} if msg or data else None)
    _write(record, "agent.jsonl")
    _console("ERROR", module, msg or event)


def debug(module: str, event: str, msg: str = "", **data: Any) -> None:
    if os.getenv("DEBUG", "").lower() in ("1", "true"):
        record = _entry("DEBUG", module, event, {"msg": msg, **data} if msg or data else None)
        _write(record, "debug.jsonl")
        _console("DEBUG", module, msg or event)


def decision(module: str, ticker: str, decision_type: str, outcome: str, **data: Any) -> None:
    """Structured decision audit record — written to decisions.jsonl."""
    record = _entry("DECISION", module, decision_type, {
        "ticker": ticker,
        "outcome": outcome,
        **data,
    })
    _write(record, "decisions.jsonl")
    _console("INFO", module, f"{ticker} → {decision_type}: {outcome}")


def trade(record_data: dict[str, Any]) -> None:
    """Append a trade execution record to trades.jsonl."""
    _ensure_init()
    record = {"ts": datetime.utcnow().isoformat() + "Z", "level": "TRADE", **record_data}
    _write(record, "trades.jsonl")
