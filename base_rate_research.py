from __future__ import annotations

import json
import os
import re
from typing import Any

import logger
from config import Config
from models import BaseRateSignal, Market

_MODULE = "base_rate_research"

_STRUCTURED_PATTERNS = (
    re.compile(r"\btemperature\b", re.I),
    re.compile(r"\b(high|low)\s+temp", re.I),
    re.compile(r"\bbetween\s+\d+", re.I),
    re.compile(r"\babove\s+\d+", re.I),
    re.compile(r"\bbelow\s+\d+", re.I),
    re.compile(r"\b(price|level|threshold|band)\b", re.I),
)


def estimate(market: Market, cfg: Config) -> BaseRateSignal | None:
    """
    Deterministic base-rate hook for structured/range markets.

    This never invents historical probabilities. It only returns a populated
    signal when a precomputed table is explicitly supplied through
    PRECOMPUTED_BASE_RATES_JSON. Otherwise structured markets log an
    unavailable signal and the probability model falls back to its existing
    weights.
    """
    if not _looks_structured(market):
        return None

    table = _load_precomputed_table()
    for key in (market.ticker, market.event_ticker, market.title):
        raw = table.get(key)
        if raw is None:
            continue
        signal = _coerce_signal(raw)
        if signal and signal.available:
            logger.info(
                _MODULE,
                "base_rate_found",
                ticker=market.ticker,
                source=signal.source,
                historical_base_rate=signal.historical_base_rate,
                confidence=signal.confidence,
            )
            logger.audit(
                _MODULE,
                "historical_base_rate_found",
                ticker=market.ticker,
                source=signal.source,
                historical_base_rate=signal.historical_base_rate,
                confidence=signal.confidence,
            )
            return signal

    logger.info(
        _MODULE,
        "historical_base_rate_unavailable",
        ticker=market.ticker,
        event_ticker=market.event_ticker,
        category=market.category,
    )
    logger.audit(
        _MODULE,
        "historical_base_rate_unavailable",
        ticker=market.ticker,
        event_ticker=market.event_ticker,
        category=market.category,
    )
    return BaseRateSignal(
        source="precomputed_base_rate",
        historical_base_rate=None,
        coverage="unavailable",
        confidence=0.0,
        notes="historical_base_rate_unavailable",
        available=False,
    )


def _looks_structured(market: Market) -> bool:
    text = f"{market.title} {market.rules_primary}".strip()
    return any(pattern.search(text) for pattern in _STRUCTURED_PATTERNS)


def _load_precomputed_table() -> dict[str, Any]:
    raw = os.getenv("PRECOMPUTED_BASE_RATES_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warn(_MODULE, "precomputed_base_rates_json_invalid", err=str(exc))
        return {}
    return data if isinstance(data, dict) else {}


def _coerce_signal(raw: Any) -> BaseRateSignal | None:
    if isinstance(raw, (int, float)):
        return BaseRateSignal(
            source="precomputed_base_rate",
            historical_base_rate=float(raw),
            coverage="precomputed",
            confidence=0.5,
            notes="numeric_precomputed_base_rate",
        )
    if not isinstance(raw, dict):
        return None
    try:
        rate = float(raw.get("historical_base_rate"))
    except (TypeError, ValueError):
        return None
    return BaseRateSignal(
        source=str(raw.get("source") or "precomputed_base_rate"),
        historical_base_rate=rate,
        coverage=str(raw.get("coverage") or "precomputed"),
        confidence=float(raw.get("confidence", 0.5) or 0.5),
        notes=str(raw.get("notes") or ""),
        available=True,
    )
