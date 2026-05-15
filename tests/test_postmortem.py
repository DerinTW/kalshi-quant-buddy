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


def test_losing_trade_uses_structured_postmortem_json(monkeypatch, tmp_path):
    inserted = []
    captured = {}
    monkeypatch.setattr(postmortem.db, "postmortem_exists", lambda trade_id: False)
    monkeypatch.setattr(postmortem.db, "insert_postmortem", lambda pm: inserted.append(pm))

    def fake_run_postmortem(*args, **kwargs):
        captured.update(kwargs)
        return {
            "trade_id": kwargs["trade_id"],
            "good_process_bad_outcome": False,
            "root_causes": ["Ignored contradictory official data"],
            "data_quality_issues": ["Contradictory source timestamps"],
            "reasoning_issues": ["Overweighted social-media move"],
            "risk_issues": ["Edge discipline was weak"],
            "execution_issues": [],
            "market_structure_issues": ["Thin book"],
            "proposed_rule_changes": [
                {
                    "rule": "Require official-source confirmation before this category",
                    "reason": "The loss came from unverified social signal.",
                    "priority": "high",
                    "requires_human_approval": False,
                }
            ],
            "should_update_rules_file": True,
        }

    monkeypatch.setattr(postmortem.llm, "run_postmortem", fake_run_postmortem)

    pm = postmortem.run_for_trade(
        closed_trade(),
        cfg=cfg(anthropic_api_key="x"),
        pending_rules_path=tmp_path / "rules_pending_review.json",
    )

    assert pm is not None
    assert captured["trade_id"] == "pm-trade-1"
    analysis = json.loads(pm.analysis)
    assert analysis["root_causes"] == ["Ignored contradictory official data"]
    proposals = json.loads(pm.rule_change_proposal)
    assert proposals[0]["requires_human_approval"] is True


def test_llm_postmortem_returns_new_json_schema(monkeypatch):
    captured = {}

    def fake_call_json(c, system, user, *, max_tokens=0, temperature=0.0):
        captured["system"] = system
        captured["user"] = user
        captured["max_tokens"] = max_tokens
        captured["temperature"] = temperature
        return {
            "trade_id": "pm-trade-1",
            "good_process_bad_outcome": True,
            "root_causes": ["Correctly sized trade resolved against thesis"],
            "data_quality_issues": [],
            "reasoning_issues": [],
            "risk_issues": [],
            "execution_issues": [],
            "market_structure_issues": [],
            "proposed_rule_changes": [],
            "should_update_rules_file": False,
        }

    monkeypatch.setattr(postmortem.llm, "call_json", fake_call_json)

    out = postmortem.llm.run_postmortem(
        cfg(anthropic_api_key="x"),
        "KXPM-TEST",
        "KXPM-TEST",
        "Good process thesis",
        0.62,
        40,
        "NO",
        trade_id="pm-trade-1",
        side="YES",
        contracts=10,
        exit_price_cents=0,
        pnl_dollars=-4.0,
        result="loss",
        research_summary="Official source was mixed.",
        sentiment_result={"narrative_direction": "mixed"},
        probability_estimate={"confidence": "medium-high"},
        edge_result={"adjusted_edge_pct": 6.0},
        risk_assessment={"approved": True},
        execution_log={"fill": "complete"},
        market_structure_notes={"spread_cents": 2},
        time_to_resolution_at_entry_minutes=120,
        resolution_rules="Official source resolves the market.",
    )

    assert set(out) == {
        "trade_id",
        "good_process_bad_outcome",
        "root_causes",
        "data_quality_issues",
        "reasoning_issues",
        "risk_issues",
        "execution_issues",
        "market_structure_issues",
        "proposed_rule_changes",
        "should_update_rules_file",
    }
    assert out["trade_id"] == "pm-trade-1"
    assert out["good_process_bad_outcome"] is True
    assert out["should_update_rules_file"] is False
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 1800
    assert "Return only valid JSON" in captured["system"]
    assert "Do not edit live rules, config, .env, or execution settings" in captured["system"]
    assert "Postmortem payload JSON:" in captured["user"]


def test_missing_evidence_is_marked_not_invented(monkeypatch):
    monkeypatch.setattr(
        postmortem.llm,
        "call_json",
        lambda *a, **kw: {
            "trade_id": "pm-trade-1",
            "good_process_bad_outcome": False,
            "root_causes": [],
            "data_quality_issues": [],
            "reasoning_issues": [],
            "risk_issues": [],
            "execution_issues": [],
            "market_structure_issues": [],
            "proposed_rule_changes": [],
            "should_update_rules_file": False,
        },
    )

    out = postmortem.llm.run_postmortem(
        cfg(anthropic_api_key="x"),
        "KXPM-TEST",
        "KXPM-TEST",
        "",
        0.0,
        40,
        "UNKNOWN_EXIT",
        trade_id="pm-trade-1",
    )

    assert any(issue.startswith("Missing ") for issue in out["data_quality_issues"])
    assert out["proposed_rule_changes"] == []
    assert out["should_update_rules_file"] is False


def test_good_process_bad_outcome_can_be_true_for_loss(monkeypatch, tmp_path):
    inserted = []
    monkeypatch.setattr(postmortem.db, "postmortem_exists", lambda trade_id: False)
    monkeypatch.setattr(postmortem.db, "insert_postmortem", lambda pm: inserted.append(pm))
    monkeypatch.setattr(
        postmortem.llm,
        "run_postmortem",
        lambda *a, **kw: {
            "trade_id": kw["trade_id"],
            "good_process_bad_outcome": True,
            "root_causes": ["Variance after correctly sized entry"],
            "data_quality_issues": [],
            "reasoning_issues": [],
            "risk_issues": [],
            "execution_issues": [],
            "market_structure_issues": [],
            "proposed_rule_changes": [],
            "should_update_rules_file": False,
        },
    )

    pm = postmortem.run_for_trade(
        closed_trade(result="loss"),
        cfg=cfg(anthropic_api_key="x"),
        pending_rules_path=tmp_path / "rules_pending_review.json",
    )

    assert pm is not None
    analysis = json.loads(pm.analysis)
    assert analysis["good_process_bad_outcome"] is True
    assert pm.was_variance is False
    assert not (tmp_path / "rules_pending_review.json").exists()


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
