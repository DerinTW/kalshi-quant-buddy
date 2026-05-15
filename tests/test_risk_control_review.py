from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import llm
import trading
from config import Config
from models import EdgeResult, PositionSize, ProbabilityEstimate


ROOT = Path(__file__).resolve().parents[1]


def cfg(**overrides) -> Config:
    base = dict(
        kalshi_api_key="x",
        anthropic_api_key="x",
        kalshi_private_key_path="",
        kill_switch=False,
        trading_mode="paper",
        paper_only=True,
        live_trading_enabled=False,
        allow_live_orders=False,
        live_confirmation_phrase="",
        max_live_dollars_per_trade=1.0,
        min_paper_days_before_live=0,
        min_paper_trades_before_live=0,
        min_paper_pnl_before_live=0.0,
    )
    base.update(overrides)
    return Config(**base)


def _llm_approves(max_allowed: float = 999.0) -> dict:
    return {
        "approved": True,
        "rejection_reasons": [],
        "risk_flags": [],
        "max_allowed_dollars": max_allowed,
        "requires_human_confirmation": False,
    }


def _deterministic(approved: bool = True, failed: list[str] | None = None) -> dict:
    return {
        "approved": approved,
        "checks_passed": ["deterministic_ok"] if approved else [],
        "checks_failed": failed or ([] if approved else ["position_too_large"]),
    }


def _review(monkeypatch, raw, *, context: dict | None = None, deterministic=None, cap=5.0):
    def fake_call_json(*args, **kwargs):
        return raw

    monkeypatch.setattr(llm, "call_json", fake_call_json)
    return llm.run_risk_control_review(
        cfg(),
        risk_context=context or {},
        deterministic_assessment=deterministic or _deterministic(True),
        deterministic_allowed_dollars=cap,
    )


def test_risk_control_prompt_is_json_only_and_not_profit_seeking():
    prompt = llm._RISK_CONTROL_REVIEW_SYSTEM

    assert "You do not seek profit" in prompt
    assert "You prevent bad trades" in prompt
    assert "Never override deterministic risk rules" in prompt
    assert "Return only JSON" in prompt
    assert '"approved": true/false' in prompt
    assert '"max_allowed_dollars": 0.0' in prompt


def test_risk_review_passes_only_structured_context(monkeypatch):
    captured = {}

    def fake_call_json(c, system, user, *, max_tokens=0, temperature=0.0):
        captured["system"] = system
        captured["user"] = user
        captured["max_tokens"] = max_tokens
        captured["temperature"] = temperature
        return _llm_approves(3.0)

    monkeypatch.setattr(llm, "call_json", fake_call_json)
    out = llm.run_risk_control_review(
        cfg(),
        risk_context={"market": {"ticker": "KXRISK", "liquidity_dollars": 1000.0}},
        deterministic_assessment=_deterministic(True),
        deterministic_allowed_dollars=4.0,
    )

    assert out["approved"] is True
    assert "Risk-control review payload JSON:" in captured["user"]
    assert "Return JSON only." in captured["user"]
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 700


def test_deterministic_rejection_cannot_be_overridden_by_llm_approval(monkeypatch):
    out = _review(
        monkeypatch,
        _llm_approves(999.0),
        deterministic=_deterministic(False, ["daily_loss_limit"]),
        cap=5.0,
    )

    assert out["approved"] is False
    assert out["max_allowed_dollars"] == 0.0
    assert "daily_loss_limit" in out["rejection_reasons"]
    assert "deterministic_risk_rejected" in out["risk_flags"]


def test_invalid_llm_json_causes_safe_rejection(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("not json")

    monkeypatch.setattr(llm, "call_json", boom)
    out = llm.run_risk_control_review(
        cfg(),
        risk_context={},
        deterministic_assessment=_deterministic(True),
        deterministic_allowed_dollars=5.0,
    )

    assert out["approved"] is False
    assert out["max_allowed_dollars"] == 0.0
    assert out["requires_human_confirmation"] is True
    assert "llm_risk_review_invalid" in out["risk_flags"]


def test_missing_llm_fields_causes_safe_rejection(monkeypatch):
    out = _review(monkeypatch, {"approved": True})

    assert out["approved"] is False
    assert out["max_allowed_dollars"] == 0.0
    assert "missing_risk_control_review_fields" in out["rejection_reasons"]


def test_near_resolution_trade_gets_stricter_flag_and_smaller_cap(monkeypatch):
    out = _review(
        monkeypatch,
        _llm_approves(10.0),
        context={"action_type": "entry", "market": {"minutes_to_close": 45.0}},
        cap=8.0,
    )

    assert "near_resolution_strict_review" in out["risk_flags"]
    assert out["max_allowed_dollars"] <= 4.0


def test_illiquid_market_gets_rejected_even_if_llm_approves(monkeypatch):
    out = _review(
        monkeypatch,
        _llm_approves(5.0),
        context={
            "market": {"liquidity_dollars": 100.0},
            "risk_limits": {"min_liquidity_dollars": 500.0},
        },
        cap=5.0,
    )

    assert out["approved"] is False
    assert "insufficient_liquidity" in out["rejection_reasons"]
    assert "illiquid_market" in out["risk_flags"]


def test_correlated_exposure_near_cap_requires_human_confirmation(monkeypatch):
    out = _review(
        monkeypatch,
        _llm_approves(5.0),
        context={
            "correlated_exposure_dollars": 14.0,
            "risk_limits": {"max_correlated_exposure_dollars": 15.0},
        },
        cap=5.0,
    )

    assert out["approved"] is False
    assert out["requires_human_confirmation"] is True
    assert "correlated_exposure_near_cap" in out["risk_flags"]


def test_max_allowed_dollars_never_exceeds_deterministic_cap(monkeypatch):
    out = _review(monkeypatch, _llm_approves(999.0), cap=4.25)

    assert out["max_allowed_dollars"] == pytest.approx(4.25)


def test_live_confirmation_gate_cannot_be_overridden_by_risk_review(monkeypatch):
    out = _review(
        monkeypatch,
        _llm_approves(1.0),
        deterministic=_deterministic(False, ["live_confirmation_phrase_mismatch"]),
        context={"mode": "live"},
        cap=1.0,
    )

    assert out["approved"] is False
    assert "live_confirmation_phrase_mismatch" in out["rejection_reasons"]


def test_risk_manager_py_does_not_place_orders_or_import_trading():
    source = (ROOT / "risk_manager.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    calls = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)

    assert "trading" not in imports
    assert "kalshi_client" not in imports
    assert "execute" not in calls
    assert "place_order" not in calls


def test_live_execution_still_requires_risk_approved(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.orders = []

        def place_order(self, **kwargs):
            self.orders.append(kwargs)
            return {"order": {"order_id": "o1", "status": "resting"}}

    client = FakeClient()
    monkeypatch.setattr(trading.db, "count_paper_trading_days", lambda: 0)
    monkeypatch.setattr(trading.db, "count_completed_trades", lambda: 0)
    monkeypatch.setattr(trading.db, "total_paper_pnl", lambda: 0.0)
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: None)

    record = trading.execute(
        PositionSize(
            ticker="KXRISK-LIVE",
            side="YES",
            dollars=0.8,
            contracts=2,
            entry_price_cents=40,
            max_loss_dollars=0.8,
        ),
        ProbabilityEstimate(
            ticker="KXRISK-LIVE",
            yes_probability=0.60,
            confidence="high",
            reasoning="test",
        ),
        EdgeResult(
            ticker="KXRISK-LIVE",
            side="YES",
            entry_price_cents=40,
            implied_yes_prob=0.40,
            estimated_yes_prob=0.60,
            raw_edge_pct=20.0,
            adjusted_edge_pct=15.0,
            expected_value=0.20,
            adjusted_ev=0.15,
            confidence="high",
            confidence_adjusted_ev=0.135,
            confidence_adjusted_edge_pct=13.5,
            spread_cents=1,
        ),
        cfg(
            trading_mode="live",
            paper_only=False,
            live_trading_enabled=True,
            allow_live_orders=True,
            live_confirmation_phrase="I_UNDERSTAND_THIS_CAN_LOSE_MONEY",
        ),
        client=client,
        mode_override=trading.LIVE,
        risk_approved=False,
    )

    assert record is None
    assert client.orders == []
