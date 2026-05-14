from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm
from config import Config
from models import Market


def cfg() -> Config:
    return Config(kalshi_api_key="x", anthropic_api_key="x", kalshi_private_key_path="")


def test_analyze_market_structure_uses_json_only_structural_prompt(monkeypatch):
    captured = {}

    def fake_call_json(c, system, user, *, max_tokens=0, temperature=0.0):
        captured["system"] = system
        captured["user"] = user
        captured["max_tokens"] = max_tokens
        captured["temperature"] = temperature
        return {
            "ticker": "KXSTRUCTURE-TEST",
            "analyze": False,
            "reasons": ["Wide spread makes deeper analysis inefficient"],
            "risk_flags": ["wide_spread"],
            "market_structure_score": 0.35,
        }

    monkeypatch.setattr(llm, "call_json", fake_call_json)

    payload = {
        "ticker": "KXSTRUCTURE-TEST",
        "title": "Will test market resolve YES?",
        "rules_primary": "Resolves based on official source.",
        "yes_ask": 55,
        "yes_bid": 45,
        "volume_24h": 100,
        "liquidity_dollars": 75,
        "minutes_to_close": 180,
    }
    result = llm.analyze_market_structure(cfg(), payload)

    assert result == {
        "ticker": "KXSTRUCTURE-TEST",
        "analyze": False,
        "reasons": ["Wide spread makes deeper analysis inefficient"],
        "risk_flags": ["wide_spread"],
        "market_structure_score": 0.35,
    }
    assert set(result) == {"ticker", "analyze", "reasons", "risk_flags", "market_structure_score"}
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 1200
    assert "Respond ONLY with valid JSON" in captured["system"]
    assert "Do NOT recommend a trade" in captured["system"]
    assert "Do NOT estimate YES probability" in captured["system"]
    assert "Do NOT calculate expected value" in captured["system"]
    assert "Do NOT suggest an order" in captured["system"]
    assert "Do NOT invent missing data" in captured["system"]
    assert '"ticker": "..."' in captured["system"]
    assert '"analyze": true' in captured["system"]
    assert '"reasons": []' in captured["system"]
    assert '"risk_flags": []' in captured["system"]
    assert '"market_structure_score": 0.0' in captured["system"]
    assert "must exactly match the input ticker" in captured["system"]
    assert "Market payload JSON:" in captured["user"]
    assert json.loads(captured["user"].split("Market payload JSON:\n", 1)[1].split("\n\n", 1)[0]) == payload


def test_market_analysis_prompt_is_not_trade_probability_or_edge_prompt():
    prompt = llm._MARKET_ANALYSIS_SYSTEM

    assert "estimate YES probability" in prompt
    assert "fair value" in prompt
    assert "edge" in prompt
    assert "order, entry, exit, size, or side" in prompt
    assert "expected value" in prompt
    assert "structural quality" in prompt
    assert "0.7+ = generally analyzable" in prompt


def test_analyze_market_structure_fails_safe_on_bad_ticker_and_low_score(monkeypatch):
    monkeypatch.setattr(
        llm,
        "call_json",
        lambda *a, **kw: {
            "ticker": "WRONG",
            "analyze": True,
            "reasons": ["Looks fine"],
            "risk_flags": [],
            "market_structure_score": 0.2,
            "extra_key": "removed",
        },
    )

    result = llm.analyze_market_structure(cfg(), {"ticker": "KXSTRUCTURE-TEST"})

    assert set(result) == {"ticker", "analyze", "reasons", "risk_flags", "market_structure_score"}
    assert result["ticker"] == "KXSTRUCTURE-TEST"
    assert result["analyze"] is False
    assert result["market_structure_score"] == 0.2
    assert "Score below 0.4 requires skip" in result["reasons"]
    assert "Output ticker did not match input ticker" in result["reasons"]
    assert "LLM ticker mismatch" in result["risk_flags"]


def test_build_market_structure_payload_contains_required_structural_fields():
    now = datetime.now(timezone.utc)
    market = Market(
        ticker="KXSTRUCTURE-TEST",
        title="Will the test condition happen?",
        status="open",
        yes_ask=55,
        yes_bid=50,
        no_ask=50,
        no_bid=45,
        volume=10_000,
        volume_24h=1_200,
        open_interest=2_500,
        close_time=now + timedelta(hours=2),
        settlement_time=now + timedelta(hours=3),
        category="economic",
        rules_primary="Resolves from an official source.",
        spread_pct=9.5,
        minutes_to_close=120,
        minutes_to_settlement=180,
        liquidity_dollars=1_375,
        is_unsafe=True,
        unsafe_reason="test_unsafe_reason",
        event_ticker="KXSTRUCTURE",
        last_trade_at=now - timedelta(minutes=3),
        orderbook_depth=150,
        price_history=[{"yes_price": 51}],
    )
    sibling = Market(
        ticker="KXSTRUCTURE-SIBLING",
        title="Sibling market",
        status="open",
        yes_ask=65,
        yes_bid=60,
        no_ask=40,
        no_bid=35,
        volume=8_000,
        volume_24h=900,
        open_interest=1_200,
        close_time=now + timedelta(hours=2),
        settlement_time=now + timedelta(hours=3),
        category="economic",
        rules_primary="Sibling rules.",
        spread_pct=8.0,
        minutes_to_close=118,
        minutes_to_settlement=178,
        liquidity_dollars=750,
        event_ticker="KXSTRUCTURE",
    )

    payload = llm.build_market_structure_payload(
        market,
        related_markets=[market, sibling],
        filter_rejections=["spread too wide"],
    )

    assert payload["identity"]["ticker"] == "KXSTRUCTURE-TEST"
    assert payload["identity"]["event_ticker"] == "KXSTRUCTURE"
    assert payload["market_text"]["has_title"] is True
    assert payload["market_text"]["has_rules_primary"] is True
    assert payload["prices"]["yes_ask"] == 55
    assert payload["prices"]["yes_bid"] == 50
    assert payload["prices"]["no_ask"] == 50
    assert payload["prices"]["no_bid"] == 45
    assert payload["prices"]["spread_cents"] == 5
    assert payload["prices"]["price_sum"] == 105
    assert payload["liquidity"]["liquidity_dollars"] == 1_375
    assert payload["liquidity"]["orderbook_depth"] == 150
    assert payload["volume"]["volume_24h"] == 1_200
    assert payload["timing"]["minutes_to_close"] == 120
    assert payload["timing"]["minutes_to_settlement"] == 180
    assert payload["freshness"]["has_recent_trade_timestamp"] is True
    assert payload["freshness"]["price_history_count"] == 1
    assert payload["safety"]["is_unsafe"] is True
    assert payload["safety"]["unsafe_reason"] == "test_unsafe_reason"
    assert payload["safety"]["filter_rejections"] == ["spread too wide"]
    assert payload["related_markets"] == [
        {
            "ticker": "KXSTRUCTURE-SIBLING",
            "event_ticker": "KXSTRUCTURE",
            "yes_ask": 65,
            "yes_bid": 60,
            "spread_cents": 5,
            "volume_24h": 900,
            "liquidity_dollars": 750,
            "minutes_to_close": 118,
            "is_same_event": True,
        }
    ]
    assert "deeper research" in payload["deeper_analysis_question"]
