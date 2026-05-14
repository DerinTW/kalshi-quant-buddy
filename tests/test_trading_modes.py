from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import trading
from config import Config
from models import EdgeResult, PositionSize, ProbabilityEstimate


class FakeClient:
    def __init__(self):
        self.orders: list[dict] = []
        self.canceled: list[str] = []
        self.response = {"order": {"order_id": "test-order", "status": "resting"}}

    def place_order(self, **kwargs):
        self.orders.append(kwargs)
        return self.response

    def cancel_order(self, order_id: str):
        self.canceled.append(order_id)
        return {"status": "canceled"}


@pytest.fixture(autouse=True)
def _clear_live_guard():
    trading.live_buy_guard.clear()
    yield
    trading.live_buy_guard.clear()


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
        max_trade_dollars=10.0,
        max_live_dollars_per_trade=1.0,
        min_paper_days_before_live=0,
        min_paper_trades_before_live=0,
        min_paper_pnl_before_live=0.0,
    )
    base.update(overrides)
    return Config(**base)


def sizing(**overrides) -> PositionSize:
    base = dict(
        ticker="KXTRADE-TEST",
        side="YES",
        dollars=0.80,
        contracts=2,
        entry_price_cents=40,
        max_loss_dollars=0.80,
    )
    base.update(overrides)
    return PositionSize(**base)


def estimate() -> ProbabilityEstimate:
    return ProbabilityEstimate(
        ticker="KXTRADE-TEST",
        yes_probability=0.60,
        confidence="high",
        reasoning="test",
    )


def edge_result(**overrides) -> EdgeResult:
    base = dict(
        ticker="KXTRADE-TEST",
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
    )
    base.update(overrides)
    return EdgeResult(**base)


def live_cfg(**overrides) -> Config:
    base = dict(
        kill_switch=False,
        trading_mode="live",
        paper_only=False,
        live_trading_enabled=True,
        allow_live_orders=True,
        live_confirmation_phrase="I_UNDERSTAND_THIS_CAN_LOSE_MONEY",
        max_live_dollars_per_trade=1.0,
    )
    base.update(overrides)
    return cfg(**base)


def test_paper_default_does_not_place_live_order(monkeypatch):
    client = FakeClient()
    inserted = []
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: inserted.append(record))

    record = trading.execute(
        sizing(), estimate(), edge_result(), cfg(), client=client, risk_approved=True
    )

    assert record is not None
    assert record.mode == "paper"
    assert client.orders == []
    assert inserted == [record]


def test_dry_run_override_does_not_place_live_order(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: None)

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        cfg(),
        client=client,
        mode_override=trading.DRY_RUN,
    )

    assert record is not None
    assert record.mode == "dry_run"
    assert client.orders == []


def test_live_override_rejected_when_live_gates_are_not_satisfied(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: None)

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        cfg(trading_mode="paper", live_trading_enabled=False),
        client=client,
        mode_override=trading.LIVE,
    )

    assert record is None
    assert client.orders == []


def test_live_size_cap_blocks_execution(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(trading.db, "count_paper_trading_days", lambda: 0)
    monkeypatch.setattr(trading.db, "count_completed_trades", lambda: 0)
    monkeypatch.setattr(trading.db, "total_paper_pnl", lambda: 0.0)
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: None)

    record = trading.execute(
        sizing(dollars=1.20),
        estimate(),
        edge_result(),
        live_cfg(),
        client=client,
        mode_override=trading.LIVE,
        risk_approved=True,
    )

    assert record is None
    assert client.orders == []


def test_successful_live_buy_adds_guard_key(monkeypatch):
    client = FakeClient()
    trading.live_buy_guard.clear()
    monkeypatch.setattr(trading.db, "count_paper_trading_days", lambda: 0)
    monkeypatch.setattr(trading.db, "count_completed_trades", lambda: 0)
    monkeypatch.setattr(trading.db, "total_paper_pnl", lambda: 0.0)
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: None)

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        live_cfg(),
        client=client,
        mode_override=trading.LIVE,
        risk_approved=True,
    )

    assert record is not None
    assert record.mode == "live"
    assert client.orders
    assert client.orders[0]["order_type"] == "limit"
    assert client.orders[0]["client_order_id"] == record.id
    assert client.orders[0]["price_cents"] == record.entry_price_cents
    assert "KXTRADE-TEST:YES" in trading.live_buy_guard


def test_live_buy_guard_rejects_duplicate_live_buy(monkeypatch):
    client = FakeClient()
    trading.live_buy_guard.clear()
    trading.live_buy_guard.add("KXTRADE-TEST:YES")
    monkeypatch.setattr(trading.db, "count_paper_trading_days", lambda: 0)
    monkeypatch.setattr(trading.db, "count_completed_trades", lambda: 0)
    monkeypatch.setattr(trading.db, "total_paper_pnl", lambda: 0.0)
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: None)

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        live_cfg(),
        client=client,
        mode_override=trading.LIVE,
        risk_approved=True,
    )

    assert record is None
    assert client.orders == []


def test_cfg_dry_run_does_not_insert_trade(monkeypatch):
    inserted = []
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: inserted.append(record))

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        cfg(trading_mode=trading.DRY_RUN),
    )

    assert record is not None
    assert record.mode == "dry_run"
    assert inserted == []


def test_paper_requires_risk_approval(monkeypatch):
    inserted = []
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: inserted.append(record))

    record = trading.execute(sizing(), estimate(), edge_result(), cfg())

    assert record is None
    assert inserted == []


def test_live_without_client_does_not_place_order(monkeypatch):
    monkeypatch.setattr(trading.db, "count_paper_trading_days", lambda: 0)
    monkeypatch.setattr(trading.db, "count_completed_trades", lambda: 0)
    monkeypatch.setattr(trading.db, "total_paper_pnl", lambda: 0.0)

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        live_cfg(),
        client=None,
        risk_approved=True,
    )

    assert record is None


def test_live_gated_when_paper_only_true(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(trading.db, "count_paper_trading_days", lambda: 0)
    monkeypatch.setattr(trading.db, "count_completed_trades", lambda: 0)
    monkeypatch.setattr(trading.db, "total_paper_pnl", lambda: 0.0)

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        live_cfg(paper_only=True),
        client=client,
        risk_approved=True,
    )

    assert record is None
    assert client.orders == []


def test_live_gated_when_live_trading_disabled(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(trading.db, "count_paper_trading_days", lambda: 0)
    monkeypatch.setattr(trading.db, "count_completed_trades", lambda: 0)
    monkeypatch.setattr(trading.db, "total_paper_pnl", lambda: 0.0)

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        live_cfg(live_trading_enabled=False),
        client=client,
        risk_approved=True,
    )

    assert record is None
    assert client.orders == []


def test_unknown_mode_fails_safely(monkeypatch):
    inserted = []
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: inserted.append(record))

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        cfg(trading_mode="mystery"),
        risk_approved=True,
    )

    assert record is None
    assert inserted == []


def test_entry_price_above_modeled_entry_is_rejected(monkeypatch):
    inserted = []
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: inserted.append(record))

    record = trading.execute(
        sizing(entry_price_cents=45),
        estimate(),
        edge_result(entry_price_cents=40),
        cfg(),
        risk_approved=True,
    )

    assert record is None
    assert inserted == []


def test_live_order_failure_returns_none_and_logs(monkeypatch):
    class FailingClient(FakeClient):
        def place_order(self, **kwargs):
            self.orders.append(kwargs)
            raise RuntimeError("simulated exchange failure")

    client = FailingClient()
    trade_logs = []
    monkeypatch.setattr(trading.db, "count_paper_trading_days", lambda: 0)
    monkeypatch.setattr(trading.db, "count_completed_trades", lambda: 0)
    monkeypatch.setattr(trading.db, "total_paper_pnl", lambda: 0.0)
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: (_ for _ in ()).throw(AssertionError("inserted")))
    monkeypatch.setattr(trading.logger, "trade", lambda data: trade_logs.append(data))

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        live_cfg(),
        client=client,
        mode_override=trading.LIVE,
        risk_approved=True,
    )

    assert record is None
    assert client.orders
    assert any(item.get("status") == "FAILED" for item in trade_logs)


def test_live_cancel_unfilled_after_timeout_if_configured(monkeypatch):
    client = FakeClient()
    client.response = {
        "order": {
            "order_id": "test-order",
            "status": "resting",
            "filled_count": 0,
            "remaining_count": 2,
        }
    }
    monkeypatch.setattr(trading.db, "count_paper_trading_days", lambda: 0)
    monkeypatch.setattr(trading.db, "count_completed_trades", lambda: 0)
    monkeypatch.setattr(trading.db, "total_paper_pnl", lambda: 0.0)
    monkeypatch.setattr(trading.db, "insert_trade", lambda record: None)
    monkeypatch.setattr(trading.time, "sleep", lambda seconds: None)
    c = live_cfg()
    c.live_order_timeout_seconds = 1

    record = trading.execute(
        sizing(),
        estimate(),
        edge_result(),
        c,
        client=client,
        mode_override=trading.LIVE,
        risk_approved=True,
    )

    assert record is not None
    assert client.canceled == ["test-order"]


def test_close_paper_trade_yes_pnl(monkeypatch):
    calls = []
    trade = trading.TradeRecord(
        id="t1",
        ticker="KX",
        side="YES",
        contracts=10,
        entry_price_cents=40,
        dollars_at_risk=4.0,
        mode="paper",
    )
    monkeypatch.setattr(trading.db, "close_trade", lambda *args: calls.append(args))
    monkeypatch.setattr(trading.postmortem, "run_for_trade", lambda trade: None)

    trading.close_paper_trade(trade, 70)

    assert calls == [("t1", 70, pytest.approx(3.0), "win")]


def test_close_paper_trade_no_pnl(monkeypatch):
    calls = []
    postmortems = []
    trade = trading.TradeRecord(
        id="t1",
        ticker="KX",
        side="NO",
        contracts=10,
        entry_price_cents=35,
        dollars_at_risk=3.5,
        mode="paper",
    )
    monkeypatch.setattr(trading.db, "close_trade", lambda *args: calls.append(args))
    monkeypatch.setattr(trading.postmortem, "run_for_trade", lambda trade: postmortems.append(trade.id))

    trading.close_paper_trade(trade, 80)

    assert calls == [("t1", 80, pytest.approx(-1.5), "loss")]
    assert postmortems == ["t1"]


def test_trading_py_does_not_import_llm():
    source = Path(trading.__file__).read_text(encoding="utf-8")

    assert "import llm" not in source
    assert "from llm" not in source
