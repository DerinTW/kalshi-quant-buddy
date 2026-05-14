from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import llm
from config import Config


def cfg() -> Config:
    return Config(kalshi_api_key="x", anthropic_api_key="x", kalshi_private_key_path="")


def test_research_agent_evidence_uses_strict_json_only_prompt(monkeypatch):
    captured = {}

    def fake_call_json(c, system, user, *, max_tokens=0, temperature=0.0):
        captured["system"] = system
        captured["user"] = user
        captured["max_tokens"] = max_tokens
        captured["temperature"] = temperature
        return {
            "market_ticker": "KXTEST",
            "research_items": [
                {
                    "source_name": "Reuters",
                    "source_type": "news",
                    "published_at": "2026-05-13T15:00:00Z",
                    "claim": "A reported fact affects the market outcome.",
                    "supports": "yes",
                    "credibility": 0.85,
                    "relevance": 0.75,
                    "recency": 0.9,
                    "url": "https://example.com/story",
                    "risk_flags": [],
                }
            ],
            "summary": "The evidence set contains one relevant news claim.",
            "missing_information": ["official confirmation"],
        }

    monkeypatch.setattr(llm, "call_json", fake_call_json)

    result = llm.research_agent_evidence(
        cfg(),
        ticker="KXTEST",
        title="Will the reported event happen?",
        rules="Resolves from an official source.",
        raw_text="Reuters, 2026-05-13T15:00:00Z: A reported fact affects the market outcome.",
    )

    assert result["market_ticker"] == "KXTEST"
    assert result["research_items"][0]["source_type"] == "news"
    assert result["research_items"][0]["supports"] == "yes"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 2500
    assert "Respond ONLY with valid JSON matching this schema exactly" in captured["system"]
    assert "Do NOT recommend a trade" in captured["system"]
    assert "Do NOT estimate YES probability" in captured["system"]
    assert "Do NOT calculate expected value" in captured["system"]
    assert "Do NOT suggest an order" in captured["system"]
    assert "Do NOT place orders" in captured["system"]
    assert "Do NOT override deterministic filters" in captured["system"]
    assert "Do NOT invent missing facts" in captured["system"]
    assert "official|news|social|market_data|other" in captured["system"]
    assert "yes|no|neutral|unclear" in captured["system"]
    assert "Research extraction payload JSON:" in captured["user"]

    payload = json.loads(captured["user"].split("Research extraction payload JSON:\n", 1)[1].split("\n\n", 1)[0])
    assert payload["market_ticker"] == "KXTEST"
    assert payload["market_title"] == "Will the reported event happen?"
    assert payload["resolution_criteria"] == "Resolves from an official source."
    assert "Reuters" in payload["raw_research_text"]


def test_research_agent_evidence_empty_raw_text_does_not_call_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "call_json", lambda *a, **kw: calls.append("call_json") or {})

    result = llm.research_agent_evidence(cfg(), "KXEMPTY", "Title", "Rules", " \n\t ")

    assert calls == []
    assert result == {
        "market_ticker": "KXEMPTY",
        "research_items": [],
        "summary": "No structured research evidence available.",
        "missing_information": ["No raw research text was available from external sources."],
    }


def test_legacy_research_helper_fails_closed_without_source_text(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "call_json", lambda *a, **kw: calls.append("call_json") or {})

    result = llm.research(cfg(), "KXLEGACY", "Title", "Rules")

    assert calls == []
    assert result["market_ticker"] == "KXLEGACY"
    assert result["research_items"] == []
    assert result["missing_information"] == [
        "No raw research text was available from external sources."
    ]


def test_extract_research_items_adapts_new_schema_to_legacy_shape(monkeypatch):
    def fake_research_agent_evidence(c, ticker, title, rules, raw_text):
        return {
            "market_ticker": ticker,
            "research_items": [
                {
                    "source_name": "FederalReserve.gov",
                    "source_type": "official",
                    "published_at": "2026-05-13T14:00:00Z",
                    "claim": "The official source published a relevant fact.",
                    "supports": "no",
                    "credibility": 0.95,
                    "relevance": 0.9,
                    "recency": 0.8,
                    "url": "https://federalreserve.gov/example",
                    "risk_flags": [],
                }
            ],
            "summary": "One official item.",
            "missing_information": [],
        }

    monkeypatch.setattr(llm, "research_agent_evidence", fake_research_agent_evidence)

    items = llm.extract_research_items(cfg(), "KXTEST", "Title", "Rules", "raw")

    assert items == [
        {
            "source": "FederalReserve.gov",
            "url": "https://federalreserve.gov/example",
            "published_at": "2026-05-13T14:00:00Z",
            "claim": "The official source published a relevant fact.",
            "direction": "supports_no",
            "relevance": 0.9,
            "summary": "The official source published a relevant fact.",
        }
    ]
