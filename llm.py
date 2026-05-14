"""
LLM gateway and role contract.

This module is the single entry point for every LLM call in the system. It
also documents the rules that govern *what the LLM is allowed to do*. Every
caller in this codebase must respect them; the tests in
``tests/test_llm_role.py`` codify the boundary so future changes don't drift.

The LLM IS used for:
  1. Interpreting resolution rules        → reasoning in estimate_probability
  2. Summarizing research                 → extract_research_items
  3. Identifying contradictions           → sentiment layer surfaces these
                                            into the LLM payload; the LLM
                                            reasons over them
  4. Explaining the thesis                → reasoning / assumptions /
                                            invalidation_conditions fields
  5. Producing structured decision JSON   → every public function in this
                                            module returns JSON-shaped dicts

The LLM is NEVER used for:
  A. Direct order placement
     - trading.py and kalshi_client.py must not import this module.
  B. Overriding the risk manager
     - risk_manager.py must not import this module.
  C. Uncapped probability jumps
     - prediction_model.py clamps the LLM's yes_probability to ±10pp from
       the market mid before blending, and weights it at only 5% of the
       ensemble. This is non-negotiable.
  D. Inventing missing data
     - When a real-time data source (Perplexity, FRED, EIA, NWS, Coinbase,
       etc.) is unavailable or returns nothing, the calling module returns
       an empty result. We do NOT prompt the LLM to "recall" what it knows
       about the topic and treat that as research. Training-cutoff data
       posing as real-time signal is the most dangerous failure mode for a
       trading system; this codebase chooses to fail loud rather than
       fabricate.

If you add a new LLM call, add it via call_json (so output is structured),
keep the system prompt explicit about not inventing facts, and add a row to
tests/test_llm_role.py if a new caller is permitted to import this module.
"""
from __future__ import annotations
import json
from typing import Any

import anthropic

import logger
from config import Config
from models import Market

_MODULE = "llm"

_client: anthropic.Anthropic | None = None


def _get_client(cfg: Config) -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    return _client


# ── Core call ─────────────────────────────────────────────────────────────────

def call(
    cfg: Config,
    system: str,
    user: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """Single-turn LLM call. Returns the text content of the first response block."""
    client = _get_client(cfg)
    response = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = response.content[0].text if response.content else ""
    logger.debug(_MODULE, "llm_call", tokens_used=response.usage.output_tokens)
    return text


def call_json(
    cfg: Config,
    system: str,
    user: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """
    Call the LLM and parse the response as JSON.
    The system prompt must instruct the model to respond with only valid JSON.
    Raises ValueError if the response cannot be parsed.
    """
    raw = call(cfg, system, user, max_tokens=max_tokens, temperature=temperature)
    # Strip markdown code fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.error(_MODULE, "json_parse_failed", raw=raw[:300], err=str(exc))
        raise ValueError(f"LLM did not return valid JSON: {exc}") from exc


# ── Typed wrappers used by downstream modules ─────────────────────────────────

_MARKET_ANALYSIS_SYSTEM = """You are a market-structure triage analyst for a prediction market system.
Your job is ONLY to decide whether a market is structurally worth spending research/probability budget on.

Hard boundaries:
- Do NOT recommend a trade.
- Do NOT estimate YES probability, NO probability, fair value, or edge.
- Do NOT calculate expected value.
- Do NOT suggest an order, entry, exit, size, or side.
- Do NOT override deterministic filters, risk manager, or blocked-market rules.
- Do NOT invent missing data. If a field is missing, mark it as unknown or a skip reason.

Evaluate only structural quality:
- market question and resolution-rule clarity
- liquidity, volume, orderbook depth, bid/ask spread, and stale data
- time to resolution / settlement
- price-data sanity and missing fields
- category and blocked/unsupported-market concerns
- duplicate or highly related market groups
- whether deeper research is likely to be worthwhile

Respond ONLY with valid JSON matching this schema:
{
  "ticker": "...",
  "analyze": true,
  "reasons": [],
  "risk_flags": [],
  "market_structure_score": 0.0
}

Field rules:
- ticker: string, must exactly match the input ticker.
- analyze: boolean. True only if the market is structurally worth deeper research/probability estimation.
- reasons: array of short strings explaining why analyze=true or analyze=false.
- risk_flags: array of short strings describing structural risks.
- market_structure_score: float from 0.0 to 1.0.
  1.0 = excellent structure.
  0.7+ = generally analyzable.
  0.4-0.69 = questionable, usually only analyze if there is a strong reason.
  below 0.4 = skip."""

_RESEARCH_SYSTEM = """You are a research analyst for a prediction market trading system.
Your task is to research a specific real-world event relevant to a Kalshi prediction market.
Provide factual, current information. Separate facts from assumptions.
Respond ONLY with valid JSON matching this schema:
{
  "summary": "2-3 sentence factual summary",
  "key_facts": ["fact1", "fact2", ...],
  "key_uncertainties": ["uncertainty1", ...],
  "sources_consulted": ["description of sources"],
  "data_freshness": "current | somewhat_stale | stale | unknown"
}"""

_RESEARCH_AGENT_SYSTEM = """You are a JSON-only research evidence extractor for a Kalshi prediction-market trading system.
Your job is ONLY to structure evidence from source text that has already been gathered by upstream real-time data/search tools.

Hard boundaries:
- Do NOT recommend a trade.
- Do NOT estimate YES probability, NO probability, fair value, or edge.
- Do NOT calculate expected value.
- Do NOT suggest an order, entry, exit, size, or side.
- Do NOT place orders or discuss position sizing.
- Do NOT override deterministic filters, blocked-market rules, or risk controls.
- Do NOT invent missing facts, sources, URLs, timestamps, scores, or claims.
- Do NOT use training-memory facts as research. Use only the provided source text and market context.
- If a field is unavailable in the provided text, use an empty string, "unclear", a neutral score, or missing_information.

Evidence requirements:
- Extract individual factual claims relevant to whether the market resolves YES or NO.
- Separate source classes using source_type: official, news, social, market_data, or other.
- Include publication time when explicitly available. Use ISO 8601 when possible; otherwise copy the provided timestamp string. Use "" when unavailable.
- Score credibility, relevance, and recency from 0.0 to 1.0 based only on the provided text.
- Use supports = "yes" only when the claim supports the market's YES outcome, "no" only when it supports the NO outcome, "neutral" for context without directional force, and "unclear" when direction cannot be inferred.
- Include risk_flags for evidence-quality issues such as stale_source, unverified_social, anonymous_rumor, missing_timestamp, ambiguous_resolution_link, paywalled_source, or contradicted_by_other_sources.
- The summary must summarize the evidence set, not recommend action.

Respond ONLY with valid JSON matching this schema exactly:
{
  "market_ticker": "...",
  "research_items": [
    {
      "source_name": "...",
      "source_type": "official|news|social|market_data|other",
      "published_at": "...",
      "claim": "...",
      "supports": "yes|no|neutral|unclear",
      "credibility": 0.0,
      "relevance": 0.0,
      "recency": 0.0,
      "url": "...",
      "risk_flags": []
    }
  ],
  "summary": "...",
  "missing_information": []
}"""

_PROBABILITY_SYSTEM = """You are a probability calibration engine for a prediction market.
Using market data and research, estimate the true probability of the YES outcome.
You must be well-calibrated — do not default to 50%. Use all available evidence.
Respond ONLY with valid JSON matching this schema:
{
  "yes_probability": <float 0.0 to 1.0>,
  "confidence": "<low|medium-low|medium|medium-high|high>",
  "reasoning": "<2-4 sentences explaining your estimate>",
  "assumptions": ["assumption1", ...],
  "invalidation_conditions": ["condition that would change your estimate", ...]
}"""

_POSTMORTEM_SYSTEM = """You are a trade reviewer for a prediction market trading system.
Analyze a completed trade that resulted in a loss and produce a structured postmortem.
Be honest and specific. Identify whether the loss was bad luck (variance) or a bad process.
Respond ONLY with valid JSON matching this schema:
{
  "was_variance": <bool>,
  "data_was_stale": <bool>,
  "resolution_handled_correctly": <bool>,
  "liquidity_hurt": <bool>,
  "sizing_appropriate": <bool>,
  "analysis": "<3-5 sentences of honest analysis>",
  "rule_change_proposal": "<specific rule change to consider, or 'none'>"
}"""


_EXTRACT_ITEMS_SYSTEM = """You are a structured data extractor for a prediction market research system.
Given raw research text and market context, extract individual factual claims as structured items.
Each item must be directly relevant to whether this specific market resolves YES or NO.
Do not invent facts — only extract what is present in the provided text.
If the text contains no clear relevant claims, return [].
Return ONLY a valid JSON array where each element matches this schema exactly:
{
  "source": "<publisher name — e.g. Reuters, CoinDesk, FederalReserve.gov>",
  "url": "<url if explicitly mentioned, else empty string>",
  "published_at": "<ISO 8601 timestamp if mentioned, else null>",
  "claim": "<one specific, concrete claim in a single sentence>",
  "direction": "<supports_yes | supports_no | neutral | unclear>",
  "relevance": <float 0.0-1.0, how relevant is this to the YES/NO resolution>,
  "summary": "<2-3 sentences of context around this claim>"
}"""


_SOURCE_TYPES = {"official", "news", "social", "market_data", "other"}
_SUPPORTS = {"yes", "no", "neutral", "unclear"}


def _empty_research_agent_result(ticker: str, reason: str) -> dict[str, Any]:
    return {
        "market_ticker": ticker,
        "research_items": [],
        "summary": "No structured research evidence available.",
        "missing_information": [reason] if reason else [],
    }


def _clamp01(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _normalize_research_agent_result(raw: Any, ticker: str) -> dict[str, Any]:
    """
    Normalize LLM JSON into the strict research-agent schema.

    The raw-list branch preserves compatibility with older tests and fakes
    that returned the previous extractor array shape.
    """
    if isinstance(raw, list):
        raw = {"market_ticker": ticker, "research_items": raw}
    if not isinstance(raw, dict):
        return _empty_research_agent_result(ticker, "LLM response was not a JSON object.")

    normalized_items: list[dict[str, Any]] = []
    for item in raw.get("research_items", []):
        if not isinstance(item, dict):
            continue

        source_name = item.get("source_name", item.get("source", ""))
        source_type = str(item.get("source_type", "other")).strip().lower()
        supports = str(item.get("supports", item.get("direction", "unclear"))).strip().lower()
        supports = {
            "supports_yes": "yes",
            "supports_no": "no",
        }.get(supports, supports)

        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue

        risk_flags = item.get("risk_flags", [])
        if not isinstance(risk_flags, list):
            risk_flags = [str(risk_flags)] if risk_flags else []

        normalized_items.append({
            "source_name": str(source_name or "").strip(),
            "source_type": source_type if source_type in _SOURCE_TYPES else "other",
            "published_at": "" if item.get("published_at") is None else str(item.get("published_at", "")).strip(),
            "claim": claim,
            "supports": supports if supports in _SUPPORTS else "unclear",
            "credibility": _clamp01(item.get("credibility"), default=0.5),
            "relevance": _clamp01(item.get("relevance"), default=0.5),
            "recency": _clamp01(item.get("recency", item.get("recency_score")), default=0.5),
            "url": str(item.get("url", "") or "").strip(),
            "risk_flags": [str(flag) for flag in risk_flags],
        })

    missing = raw.get("missing_information", [])
    if not isinstance(missing, list):
        missing = [str(missing)] if missing else []

    return {
        "market_ticker": str(raw.get("market_ticker") or ticker),
        "research_items": normalized_items,
        "summary": str(raw.get("summary", "") or "").strip(),
        "missing_information": [str(item) for item in missing],
    }


def research_agent_evidence(
    cfg: Config,
    ticker: str,
    title: str,
    rules: str,
    raw_text: str,
) -> dict[str, Any]:
    """
    Structure externally gathered research into strict JSON evidence items.

    This wrapper does not search, recommend trades, estimate probabilities,
    calculate edge, place orders, or override risk controls. It only extracts
    claims present in raw_text and returns the research-agent schema.
    """
    if not raw_text or not raw_text.strip():
        return _empty_research_agent_result(
            ticker,
            "No raw research text was available from external sources.",
        )

    payload = {
        "market_ticker": ticker,
        "market_title": title,
        "resolution_criteria": rules,
        "raw_research_text": raw_text[:6000],
    }
    user = f"""Research extraction payload JSON:
{json.dumps(payload, sort_keys=True, default=str)}

Extract and summarize evidence relevant to this market outcome. Return JSON only."""
    result = call_json(cfg, _RESEARCH_AGENT_SYSTEM, user, max_tokens=2500, temperature=0.0)
    return _normalize_research_agent_result(result, ticker)


def extract_research_items(
    cfg: Config,
    ticker: str,
    title: str,
    rules: str,
    raw_text: str,
) -> list[dict[str, Any]]:
    """
    Back-compat adapter for legacy research callers.

    The canonical research-agent wrapper returns the stricter object schema
    with source_type/supports fields. Older downstream modules still consume
    the previous array shape with source/direction fields, so map it here.
    """
    try:
        result = research_agent_evidence(cfg, ticker, title, rules, raw_text)
    except ValueError:
        return []

    direction_map = {
        "yes": "supports_yes",
        "no": "supports_no",
        "neutral": "neutral",
        "unclear": "unclear",
    }
    items: list[dict[str, Any]] = []
    for item in result.get("research_items", []):
        items.append({
            "source": item.get("source_name", ""),
            "url": item.get("url", ""),
            "published_at": item.get("published_at") or None,
            "claim": item.get("claim", ""),
            "direction": direction_map.get(item.get("supports"), "unclear"),
            "relevance": item.get("relevance", 0.5),
            "summary": item.get("claim", ""),
        })
    return items


def analyze_market_structure(
    cfg: Config,
    market_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Classify whether a market is structurally worth deeper research.

    This wrapper must not recommend trades, estimate probabilities, calculate
    edge, or place orders. It is a JSON-only pre-research triage step.
    """
    user = f"""Market payload JSON:
{json.dumps(market_payload, sort_keys=True, default=str)[:5000]}

Classify only the market's structure and data quality. Return JSON only."""
    raw = call_json(cfg, _MARKET_ANALYSIS_SYSTEM, user, max_tokens=1200, temperature=0.0)
    return _normalize_market_structure_result(raw, _payload_ticker(market_payload))


def build_market_structure_payload(
    market: Market,
    *,
    related_markets: list[Market] | None = None,
    filter_rejections: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build the canonical structural payload for analyze_market_structure().

    It includes the fields needed to assess liquidity, volume, bid/ask spread,
    time to close/settlement, stale data, market ambiguity, related/duplicate
    markets, and suitability for deeper analysis.
    """
    related = related_markets or []
    return {
        "identity": {
            "ticker": market.ticker,
            "event_ticker": market.event_ticker,
            "category": market.category,
            "status": market.status,
        },
        "market_text": {
            "title": market.title,
            "rules_primary": market.rules_primary,
            "has_title": bool(market.title),
            "has_rules_primary": bool(market.rules_primary),
        },
        "prices": {
            "yes_ask": market.yes_ask,
            "yes_bid": market.yes_bid,
            "no_ask": market.no_ask,
            "no_bid": market.no_bid,
            "spread_cents": _spread_cents(market),
            "spread_pct": market.spread_pct,
            "price_sum": _price_sum(market),
        },
        "liquidity": {
            "liquidity_dollars": market.liquidity_dollars,
            "orderbook_depth": market.orderbook_depth,
            "open_interest": market.open_interest,
        },
        "volume": {
            "volume": market.volume,
            "volume_24h": market.volume_24h,
        },
        "timing": {
            "close_time": market.close_time.isoformat() if market.close_time else None,
            "settlement_time": market.settlement_time.isoformat() if market.settlement_time else None,
            "minutes_to_close": market.minutes_to_close,
            "minutes_to_settlement": market.minutes_to_settlement,
        },
        "freshness": {
            "last_trade_at": market.last_trade_at.isoformat() if market.last_trade_at else None,
            "has_recent_trade_timestamp": market.last_trade_at is not None,
            "price_history_count": len(market.price_history),
        },
        "safety": {
            "is_unsafe": market.is_unsafe,
            "unsafe_reason": market.unsafe_reason,
            "filter_rejections": filter_rejections or [],
        },
        "related_markets": [
            {
                "ticker": other.ticker,
                "event_ticker": other.event_ticker,
                "yes_ask": other.yes_ask,
                "yes_bid": other.yes_bid,
                "spread_cents": _spread_cents(other),
                "volume_24h": other.volume_24h,
                "liquidity_dollars": other.liquidity_dollars,
                "minutes_to_close": other.minutes_to_close,
                "is_same_event": bool(market.event_ticker and other.event_ticker == market.event_ticker),
            }
            for other in related
            if other.ticker != market.ticker
        ],
        "deeper_analysis_question": (
            "Based only on structure and data quality, should this market consume "
            "deeper research/probability-estimation budget?"
        ),
    }


def _spread_cents(market: Market) -> int | None:
    if market.yes_ask < 0 or market.yes_bid < 0 or market.yes_ask < market.yes_bid:
        return None
    return market.yes_ask - market.yes_bid


def _price_sum(market: Market) -> int | None:
    if market.yes_ask <= 0 or market.no_ask <= 0:
        return None
    return market.yes_ask + market.no_ask


def _payload_ticker(market_payload: dict[str, Any]) -> str:
    ticker = market_payload.get("ticker")
    if ticker is None and isinstance(market_payload.get("identity"), dict):
        ticker = market_payload["identity"].get("ticker")
    return str(ticker or "")


def _normalize_market_structure_result(raw: dict[str, Any], input_ticker: str) -> dict[str, Any]:
    reasons = _short_string_list(raw.get("reasons"))
    risk_flags = _short_string_list(raw.get("risk_flags"))
    score = _bounded_float(raw.get("market_structure_score"))
    raw_analyze = raw.get("analyze")
    analyze = isinstance(raw_analyze, bool) and raw_analyze and score >= 0.4

    if raw_analyze is True and score < 0.4:
        reasons.append("Score below 0.4 requires skip")
    elif not isinstance(raw_analyze, bool):
        reasons.append("Analyze field was not boolean")

    output_ticker = str(raw.get("ticker") or "")
    if output_ticker != input_ticker:
        analyze = False
        risk_flags.append("LLM ticker mismatch")
        reasons.append("Output ticker did not match input ticker")
        output_ticker = input_ticker

    return {
        "ticker": output_ticker,
        "analyze": analyze,
        "reasons": reasons,
        "risk_flags": risk_flags,
        "market_structure_score": score,
    }


def _short_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text[:160])
    return out


def _bounded_float(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def research(cfg: Config, ticker: str, title: str, rules: str) -> dict[str, Any]:
    """
    Deprecated compatibility helper.

    Direct LLM research would invite training-memory facts to masquerade as
    real-time evidence, so this fails closed. Callers that have external
    source text should use research_agent_evidence().
    """
    return research_agent_evidence(cfg, ticker, title, rules, raw_text="")


def estimate_probability(
    cfg: Config,
    ticker: str,
    title: str,
    rules: str,
    yes_ask: int,
    no_ask: int,
    research_summary: str,
    sentiment_data: dict[str, Any],
) -> dict[str, Any]:
    user = f"""Market: {ticker}
Title: {title}
Resolution criteria: {rules}
Current YES ask: {yes_ask}¢ (market implies ~{yes_ask}% YES probability)
Current NO ask: {no_ask}¢

Research summary:
{research_summary}

Sentiment signal: {sentiment_data.get('sentiment_score', 0.0):+.3f} (confidence: {sentiment_data.get('confidence', 0.0):.2f})
Narrative: {sentiment_data.get('narrative_direction', 'neutral')}
Market impact estimate: {sentiment_data.get('market_impact_cents', 0)}¢
{("Contradictions: " + "; ".join(sentiment_data.get('contradictions', []))) if sentiment_data.get('contradictions') else ""}

Estimate the true probability this resolves YES."""
    return call_json(cfg, _PROBABILITY_SYSTEM, user, max_tokens=1000)


def run_postmortem(
    cfg: Config,
    ticker: str,
    title: str,
    original_thesis: str,
    estimated_yes_prob: float,
    entry_price_cents: int,
    actual_result: str,
) -> dict[str, Any]:
    user = f"""Trade postmortem:
Market: {ticker} — {title}
Original thesis: {original_thesis}
Estimated YES probability at entry: {estimated_yes_prob:.1%}
Entry price (YES): {entry_price_cents}¢ (implied market probability: ~{entry_price_cents}%)
Actual result: {actual_result}

Analyze this trade."""
    return call_json(cfg, _POSTMORTEM_SYSTEM, user, max_tokens=1200)
