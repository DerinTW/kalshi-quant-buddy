from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import monitor
from config import Config
from models import TradeRecord


class FakeClient:
    def __init__(
        self,
        *,
        settlement: str | None = None,
        orderbook: dict | None = None,
        market: dict | None = None,
    ):
        self.settlement = settlement
        self.orderbook = orderbook or {
            "orderbook": {
                "yes": {"ask": [[51, 100]], "bid": [[50, 100]]},
                "no": {"ask": [[51, 100]], "bid": [[49, 100]]},
            }
        }
        self.market = market or {"market": {"minutes_to_close": 120}}

    def get_settlement(self, ticker: str):
        return self.settlement

    def get_orderbook(self, ticker: str, depth: int = 1):
        return self.orderbook

    def get_best_prices(self, ticker: str):
        yes = self.orderbook.get("orderbook", self.orderbook).get("yes", {})
        ask = yes.get("ask", [[-1]])[0][0] if yes.get("ask") else -1
        bid = yes.get("bid", [[-1]])[0][0] if yes.get("bid") else -1
        return ask, bid

    def get_market(self, ticker: str):
        return self.market


def cfg(**overrides) -> Config:
    base = dict(
        max_spread_cents=6,
        min_orderbook_depth_at_limit=1,
        stop_loss_cents=12,
        take_profit_cents=8,
        force_review_last_minutes=30,
    )
    base.update(overrides)
    return Config(**base)


def trade(**overrides) -> TradeRecord:
    base = dict(
        id="trade-1",
        ticker="KXMONITOR-TEST",
        side="YES",
        contracts=10,
        entry_price_cents=50,
        dollars_at_risk=5.0,
        mode="paper",
        thesis="clear deterministic thesis",
    )
    base.update(overrides)
    return TradeRecord(**base)


def orderbook(*, yes_bid: int | None, yes_ask: int | None, no_bid: int | None = None, no_ask: int | None = None, depth: int = 100):
    def level(price):
        return [[price, depth]] if price is not None else []

    return {
        "orderbook": {
            "yes": {"ask": level(yes_ask), "bid": level(yes_bid)},
            "no": {"ask": level(no_ask), "bid": level(no_bid)},
        }
    }


def test_yes_stop_loss_closes_at_yes_bid(monkeypatch):
    calls = []
    monkeypatch.setattr(monitor.db, "get_open_trades", lambda: [trade(entry_price_cents=50)])
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))

    monitor.check_positions(FakeClient(orderbook=orderbook(yes_bid=38, yes_ask=39)), cfg())

    assert calls == [("trade-1", 38)]


def test_yes_take_profit_closes_at_yes_bid(monkeypatch):
    calls = []
    monkeypatch.setattr(monitor.db, "get_open_trades", lambda: [trade(entry_price_cents=50)])
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))

    monitor.check_positions(FakeClient(orderbook=orderbook(yes_bid=58, yes_ask=59)), cfg())

    assert calls == [("trade-1", 58)]


def test_take_profit_holds_when_remaining_edge_is_still_positive(monkeypatch):
    calls = []
    monkeypatch.setattr(
        monitor.db,
        "get_open_trades",
        lambda: [trade(entry_price_cents=50, estimated_yes_prob=0.70)],
    )
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))

    monitor.check_positions(
        FakeClient(orderbook=orderbook(yes_bid=58, yes_ask=59)),
        cfg(exit_if_edge_below_cents=2),
    )

    assert calls == []


def test_take_profit_closes_when_remaining_edge_is_below_threshold(monkeypatch):
    calls = []
    monkeypatch.setattr(
        monitor.db,
        "get_open_trades",
        lambda: [trade(entry_price_cents=50, estimated_yes_prob=0.59)],
    )
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))

    monitor.check_positions(
        FakeClient(orderbook=orderbook(yes_bid=58, yes_ask=59)),
        cfg(exit_if_edge_below_cents=2),
    )

    assert calls == [("trade-1", 58)]


def test_no_stop_loss_uses_no_bid_value(monkeypatch):
    calls = []
    monkeypatch.setattr(monitor.db, "get_open_trades", lambda: [trade(side="NO", entry_price_cents=40)])
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))

    monitor.check_positions(
        FakeClient(orderbook=orderbook(yes_bid=70, yes_ask=72, no_bid=28, no_ask=30)),
        cfg(),
    )

    assert calls == [("trade-1", 72)]


def test_no_take_profit_uses_no_bid_value(monkeypatch):
    calls = []
    monkeypatch.setattr(monitor.db, "get_open_trades", lambda: [trade(side="NO", entry_price_cents=40)])
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))

    monitor.check_positions(
        FakeClient(orderbook=orderbook(yes_bid=50, yes_ask=52, no_bid=48, no_ask=50)),
        cfg(),
    )

    assert calls == [("trade-1", 52)]


def test_no_value_can_derive_from_yes_ask_when_no_bid_missing(monkeypatch):
    calls = []
    monkeypatch.setattr(monitor.db, "get_open_trades", lambda: [trade(side="NO", entry_price_cents=40)])
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))

    monitor.check_positions(FakeClient(orderbook=orderbook(yes_bid=50, yes_ask=52)), cfg())

    assert calls == [("trade-1", 52)]


@pytest.mark.parametrize(
    ("side", "settlement", "exit_yes_price"),
    [
        ("YES", "yes", 100),
        ("YES", "no", 0),
        ("NO", "no", 0),
        ("NO", "yes", 100),
    ],
)
def test_settlement_closes_yes_and_no_with_yes_exit_price(monkeypatch, side, settlement, exit_yes_price):
    calls = []
    monkeypatch.setattr(monitor.db, "get_open_trades", lambda: [trade(side=side)])
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))

    monitor.check_positions(FakeClient(settlement=settlement), cfg())

    assert calls == [("trade-1", exit_yes_price)]


def test_invalid_price_data_does_not_close(monkeypatch):
    calls = []
    monkeypatch.setattr(monitor.db, "get_open_trades", lambda: [trade()])
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))

    monitor.check_positions(FakeClient(orderbook=orderbook(yes_bid=None, yes_ask=None)), cfg())

    assert calls == []


def test_force_review_logs_when_close_to_resolution(monkeypatch):
    warnings = []
    calls = []
    monkeypatch.setattr(monitor.db, "get_open_trades", lambda: [trade()])
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))
    monkeypatch.setattr(monitor.logger, "warn", lambda module, event, msg="", **data: warnings.append((event, data)))

    monitor.check_positions(
        FakeClient(market={"market": {"minutes_to_close": 25}}),
        cfg(force_review_last_minutes=30),
    )

    assert any(event == "force_review" for event, _ in warnings)
    assert calls == []


def test_uncertain_thesis_time_exit_closes_with_safe_price(monkeypatch):
    calls = []
    monkeypatch.setattr(
        monitor.db,
        "get_open_trades",
        lambda: [trade(thesis="uncertain thesis near resolution", entry_price_cents=50)],
    )
    monkeypatch.setattr(monitor.trading, "close_paper_trade", lambda t, price: calls.append((t.id, price)))

    monitor.check_positions(
        FakeClient(orderbook=orderbook(yes_bid=49, yes_ask=50), market={"market": {"minutes_to_close": 25}}),
        cfg(force_review_last_minutes=30),
    )

    assert calls == [("trade-1", 49)]


def test_config_monitoring_defaults_exist(monkeypatch):
    for name in (
        "STOP_LOSS_CENTS",
        "TAKE_PROFIT_CENTS",
        "EXIT_IF_EDGE_BELOW_CENTS",
        "NO_NEW_ENTRIES_LAST_MINUTES",
        "FORCE_REVIEW_LAST_MINUTES",
    ):
        monkeypatch.delenv(name, raising=False)

    c = Config()

    assert c.stop_loss_cents == 12
    assert c.take_profit_cents == 8
    assert c.exit_if_edge_below_cents == 2
    assert c.no_new_entries_last_minutes == 20
    assert c.force_review_last_minutes == 30


def test_env_example_documents_monitoring_defaults():
    text = Path(".env.example").read_text(encoding="utf-8")

    assert "STOP_LOSS_CENTS=12" in text
    assert "TAKE_PROFIT_CENTS=8" in text
    assert "EXIT_IF_EDGE_BELOW_CENTS=2" in text
    assert "NO_NEW_ENTRIES_LAST_MINUTES=20" in text
    assert "FORCE_REVIEW_LAST_MINUTES=30" in text
