from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import storage
from models import Postmortem, TradeRecord


ROOT = Path(__file__).resolve().parents[1]


def _env_value(text: str, name: str) -> str | None:
    match = re.search(rf"^{re.escape(name)}=(.*)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def test_expected_project_structure_files_exist():
    for relative in [
        "main.py",
        "config.py",
        "kalshi_client.py",
        "models.py",
        "market_scanner.py",
        "filters.py",
        "weird_move.py",
        "research_agents.py",
        "sentiment.py",
        "prediction_model.py",
        "edge.py",
        "risk_manager.py",
        "position_sizing.py",
        "trading.py",
        "monitor.py",
        "postmortem.py",
        "logger.py",
        "storage.py",
        ".env.example",
        "requirements.txt",
        "README.md",
        "rules/base_rules.json",
        "rules/rules_pending_review.json",
        "rules/blocked_markets.json",
        "data",
        "data/logs",
        "data/snapshots",
        "tests",
    ]:
        assert (ROOT / relative).exists(), f"missing {relative}"


def test_rules_files_are_valid_json_and_safe_defaults():
    base = json.loads((ROOT / "rules" / "base_rules.json").read_text(encoding="utf-8"))
    pending = json.loads((ROOT / "rules" / "rules_pending_review.json").read_text(encoding="utf-8"))
    blocked = json.loads((ROOT / "rules" / "blocked_markets.json").read_text(encoding="utf-8"))

    assert base["kill_switch_default"] is True
    assert base["trading_mode_default"] == "paper"
    assert base["live_trading_enabled_default"] is False
    assert base["max_dollars_per_trade_paper"] == 10
    assert base["max_dollars_per_trade_live_test"] == 1
    assert base["max_bankroll_pct_per_trade"] == 0.5
    assert base["max_daily_loss"] == 20
    assert base["max_trades_per_day"] == 5
    assert base["max_category_exposure"] == 25
    assert base["max_correlated_exposure"] == 15
    assert base["max_spread_cents"] == 6
    assert base["min_liquidity"] == 500
    assert base["no_new_entries_under_minutes"] == 20
    assert base["never_enter_under_minutes"] == 5
    assert base["rules_require_human_approval"] is True
    assert pending == {"pending_rule_changes": []}
    assert blocked["blocked_tickers"] == []
    assert blocked["blocked_categories"] == ["politics"]
    assert blocked["blocked_event_groups"] == []


def test_env_example_contains_safe_defaults():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert _env_value(text, "KILL_SWITCH") == "true"
    assert _env_value(text, "TRADING_MODE") == "paper"
    assert _env_value(text, "LIVE_TRADING_ENABLED") == "false"
    assert _env_value(text, "ALLOW_LIVE_ORDERS") == "false"
    assert _env_value(text, "MAX_DOLLARS_PER_TRADE") == "10"
    assert _env_value(text, "MAX_TRADE_DOLLARS") == "10"
    assert _env_value(text, "MAX_DAILY_LOSS") == "20"
    assert _env_value(text, "MAX_TRADES_PER_DAY") == "5"
    assert _env_value(text, "MAX_SPREAD_CENTS") == "10"
    assert _env_value(text, "MIN_LIQUIDITY") == "25"
    assert _env_value(text, "MIN_LIQUIDITY_DOLLARS") == "25"
    assert "LIVE_MAX_SPREAD_CENTS=6" in text
    assert "LIVE_MIN_LIQUIDITY=500" in text


def test_storage_wrappers_delegate_to_db(monkeypatch):
    calls = []
    trade = TradeRecord(
        id="s1",
        ticker="KXSTORAGE",
        side="YES",
        contracts=1,
        entry_price_cents=40,
        dollars_at_risk=0.40,
        mode="paper",
    )
    pm = Postmortem(
        trade_id="s1",
        ticker="KXSTORAGE",
        original_thesis="test",
        estimated_yes_prob=0.6,
        market_price_at_entry=40,
        actual_result="NO",
        was_variance=False,
        data_was_stale=False,
        resolution_handled_correctly=True,
        liquidity_hurt=False,
        sizing_appropriate=True,
        analysis="test",
        rule_change_proposal="[]",
    )
    monkeypatch.setattr(storage.db, "init", lambda path: calls.append(("init", path)))
    monkeypatch.setattr(storage.db, "get_open_trades", lambda: [trade])
    monkeypatch.setattr(storage.db, "get_closed_trades", lambda: [trade])
    monkeypatch.setattr(storage.db, "insert_trade", lambda t: calls.append(("insert_trade", t.id)))
    monkeypatch.setattr(storage.db, "close_trade", lambda *args: calls.append(("close_trade", args)))
    monkeypatch.setattr(storage.db, "insert_postmortem", lambda p: calls.append(("insert_postmortem", p.trade_id)))

    storage.init_storage("test.db")
    assert storage.get_open_trades() == [trade]
    assert storage.get_closed_trades() == [trade]
    storage.insert_trade(trade)
    storage.close_trade("s1", 0, -0.4, "loss")
    storage.insert_postmortem(pm)

    assert calls == [
        ("init", "test.db"),
        ("insert_trade", "s1"),
        ("close_trade", ("s1", 0, -0.4, "loss")),
        ("insert_postmortem", "s1"),
    ]
