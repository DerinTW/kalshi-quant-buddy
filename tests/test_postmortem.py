from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import db
import postmortem
import trading
from config import Config
from models import TradeRecord


@pytest.fixture(autouse=True)
def _clear_postmortem_state():
    postmortem._clear_processed_for_tests()
    yield
    postmortem._clear_processed_for_tests()


def cfg(**overrides) -> Config:
    base = dict(
        kalshi_api_key="x",
        anthropic_api_key="",
        kalshi_private_key_path="",
        max_trade_dollars=10.0,
    )
    base.update(overrides)
    return Config(**base)


def closed_trade(**overrides) -> TradeRecord:
    base = dict(
        id="pm-trade-1",
        ticker="KXPM-TEST",
        side="YES",
        contracts=10,
        entry_price_cents=40,
        dollars_at_risk=4.0,
        mode="paper",
        thesis="YES because official source confirms the condition",
        estimated_yes_prob=0.62,
        exit_price_cents=0,
        pnl_dollars=-4.0,
        result="loss",
    )
    base.update(overrides)
    return TradeRecord(**base)


def test_postmortem_only_runs_on_loss(monkeypatch, tmp_path):
    inserted = []
    monkeypatch.setattr(postmortem.db, "postmortem_exists", lambda trade_id: False)
    monkeypatch.setattr(postmortem.db, "insert_postmortem", lambda pm: inserted.append(pm))

    result = postmortem.run_for_trade(
        closed_trade(result="loss"),
        cfg=cfg(),
        pending_rules_path=tmp_path / "rules_pending_review.json",
    )

    assert result is not None
    assert inserted


@pytest.mark.parametrize("result", ["win", "push", "open", None])
def test_postmortem_not_run_for_win_or_push(monkeypatch, tmp_path, result):
    inserted = []
    monkeypatch.setattr(postmortem.db, "postmortem_exists", lambda trade_id: False)
    monkeypatch.setattr(postmortem.db, "insert_postmortem", lambda pm: inserted.append(pm))

    result_obj = postmortem.run_for_trade(
        closed_trade(result=result, pnl_dollars=1.0),
        cfg=cfg(),
        pending_rules_path=tmp_path / "rules_pending_review.json",
    )

    assert result_obj is None
    assert inserted == []


def test_postmortem_fallback_when_llm_fails(monkeypatch, tmp_path):
    inserted = []
    monkeypatch.setattr(postmortem.db, "postmortem_exists", lambda trade_id: False)
    monkeypatch.setattr(postmortem.db, "insert_postmortem", lambda pm: inserted.append(pm))
    monkeypatch.setattr(postmortem.llm, "run_postmortem", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("llm down")))

    pm = postmortem.run_for_trade(
        closed_trade(),
        cfg=cfg(anthropic_api_key="x"),
        pending_rules_path=tmp_path / "rules_pending_review.json",
    )

    assert pm is not None
    assert "Deterministic fallback postmortem" in pm.analysis
    assert inserted == [pm]


def test_postmortem_written_to_db(tmp_path):
    old_path = db._db_path
    try:
        db.init(str(tmp_path / "test.db"))
        pm = postmortem.run_for_trade(
            closed_trade(),
            cfg=cfg(),
            pending_rules_path=tmp_path / "rules_pending_review.json",
        )

        assert pm is not None
        with sqlite3.connect(tmp_path / "test.db") as conn:
            row = conn.execute("SELECT trade_id, ticker FROM postmortems WHERE trade_id=?", (pm.trade_id,)).fetchone()
        assert row == ("pm-trade-1", "KXPM-TEST")
    finally:
        db._db_path = old_path


def test_rule_change_goes_to_pending_review_not_base_rules(monkeypatch, tmp_path):
    inserted = []
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    base_rules = rules_dir / "base_rules.json"
    base_rules.write_text('{"active": true}', encoding="utf-8")
    pending = rules_dir / "rules_pending_review.json"
    monkeypatch.setattr(postmortem.db, "postmortem_exists", lambda trade_id: False)
    monkeypatch.setattr(postmortem.db, "insert_postmortem", lambda pm: inserted.append(pm))

    postmortem.run_for_trade(closed_trade(), cfg=cfg(), pending_rules_path=pending)

    assert base_rules.read_text(encoding="utf-8") == '{"active": true}'
    data = json.loads(pending.read_text(encoding="utf-8"))
    pending_changes = data["pending_rule_changes"]
    assert pending_changes[0]["trade_id"] == "pm-trade-1"
    assert pending_changes[0]["requires_human_approval"] is True
    assert all(change["requires_human_approval"] is True for change in pending_changes[0]["rule_changes_proposed"])


def test_postmortem_does_not_crash_trade_close(monkeypatch):
    calls = []
    trade = closed_trade(result=None, exit_price_cents=None, pnl_dollars=None)
    monkeypatch.setattr(trading.db, "close_trade", lambda *args: calls.append(args))
    monkeypatch.setattr(trading.postmortem, "run_for_trade", lambda trade: (_ for _ in ()).throw(RuntimeError("boom")))

    trading.close_paper_trade(trade, 0)

    assert calls == [("pm-trade-1", 0, pytest.approx(-4.0), "loss")]
    assert trade.result == "loss"


def test_no_duplicate_postmortem_for_same_trade(monkeypatch, tmp_path):
    inserted = []
    monkeypatch.setattr(postmortem.db, "postmortem_exists", lambda trade_id: False)
    monkeypatch.setattr(postmortem.db, "insert_postmortem", lambda pm: inserted.append(pm))

    trade = closed_trade()
    first = postmortem.run_for_trade(trade, cfg=cfg(), pending_rules_path=tmp_path / "rules_pending_review.json")
    second = postmortem.run_for_trade(trade, cfg=cfg(), pending_rules_path=tmp_path / "rules_pending_review.json")

    assert first is not None
    assert second is None
    assert len(inserted) == 1
