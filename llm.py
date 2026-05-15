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
  6. Adding risk-control caution          -> run_risk_control_review may add
                                            flags, rejection reasons, or human
                                            confirmation requirements

The LLM is NEVER used for:
  A. Direct order placement
     - trading.py and kalshi_client.py must not import this module.
  B. Overriding the risk manager
     - risk_manager.py must not import this module.
     - The risk-control reviewer cannot approve any trade rejected by
       deterministic risk checks.
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
from dataclasses import asdict, is_dataclass
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


_PROBABILITY_ESTIMATOR_SYSTEM = """You are a probability calibration engine for a prediction-market system.

Your job:
- Estimate the true probability of the YES outcome.
- Use the market-implied probability as the anchor.
- Adjust only when the provided evidence justifies it.
- Treat official data, reliable market data, and credible news as stronger than social sentiment.
- Treat social-only claims, rumors, missing data, stale data, contradictions, thin liquidity, wide spreads, and unclear resolution rules as uncertainty factors.
- Do not recommend a trade.
- Do not calculate edge.
- Do not suggest side, entry, exit, or position size.
- Do not place orders.
- Do not override risk controls.
- Do not invent missing facts.
- Return only valid JSON.

Respond ONLY with valid JSON matching this schema exactly:
{
  "estimated_yes_probability": 0.0,
  "estimated_no_probability": 0.0,
  "confidence": 0.0,
  "main_factors": [],
  "uncertainty_factors": [],
  "probability_rationale": "...",
  "overconfidence_warning": true
}

Field rules:
- estimated_yes_probability: float from 0.0 to 1.0.
- estimated_no_probability: must equal 1.0 - estimated_yes_probability, rounded safely.
- confidence: float from 0.0 to 1.0.
- main_factors: short strings explaining the strongest evidence-based drivers.
- uncertainty_factors: short strings explaining missing/conflicting/stale/weak evidence.
- probability_rationale: concise explanation of why the estimate differs from, or stays near, the market baseline.
- overconfidence_warning: true when evidence is weak, social-driven, contradictory, stale, sparse, or the estimate moves far from the market baseline."""

_RISK_CONTROL_REVIEW_SYSTEM = """You are a risk-control reviewer. You do not seek profit. You prevent bad trades.

Rules:
- If any hard risk limit is violated, reject the trade.
- Never override deterministic risk rules.
- Be stricter near resolution.
- Be stricter with illiquid markets.
- Be stricter with correlated exposure.
- Return only JSON.

Output:
{
  "approved": true/false,
  "rejection_reasons": [],
  "risk_flags": [],
  "max_allowed_dollars": 0.0,
  "requires_human_confirmation": true/false
}"""

_POSTMORTEM_SYSTEM = """You are a postmortem reviewer for a prediction-market trading system.

Your job is to explain why a losing trade lost and whether the process was flawed.

Rules:
- Do not assume all losses are bad decisions.
- Separate bad process from bad outcome.
- Identify stale data, bad reasoning, bad sizing, execution issues, and market-structure issues.
- Propose rule changes, but mark them as requiring human approval.
- Do not recommend a new trade.
- Do not place orders.
- Do not override the risk manager.
- Do not edit live rules, config, .env, or execution settings.
- If evidence is missing, say so instead of inventing causes.
- Return only valid JSON.

Required JSON output:
{
  "trade_id": "...",
  "good_process_bad_outcome": true,
  "root_causes": [],
  "data_quality_issues": [],
  "reasoning_issues": [],
  "risk_issues": [],
  "execution_issues": [],
  "market_structure_issues": [],
  "proposed_rule_changes": [
    {
      "rule": "...",
      "reason": "...",
      "priority": "low|medium|high",
      "requires_human_approval": true
    }
  ],
  "should_update_rules_file": false
}

Field rules:
- trade_id must exactly match the input trade_id.
- good_process_bad_outcome should be true only when the trade followed the intended process but lost due to variance, unavoidable uncertainty, or a correctly sized risk that resolved against the thesis.
- root_causes should summarize the main causes of the loss.
- data_quality_issues should include stale, missing, contradictory, low-credibility, or poorly timestamped data.
- reasoning_issues should include overconfidence, ignored contradictions, bad probability adjustment, weak resolution-rule interpretation, or social-media overreaction.
- risk_issues should include oversizing, excessive correlated exposure, taking a trade too close to resolution, violating spread/liquidity rules, or weak confidence/edge discipline.
- execution_issues should include bad entry, bad limit price, late fill, partial fill, failure to exit, spread crossing, or slippage.
- market_structure_issues should include thin book, wide spread, stale book, related-market disagreement, liquidity gap, or manipulation/rumor-driven movement.
- proposed_rule_changes must be suggestions only.
- Every proposed rule change must include "requires_human_approval": true.
- should_update_rules_file should be true only when there is a concrete rule-change proposal worth writing to pending review.
- If no useful rule change is justified, proposed_rule_changes should be [] and should_update_rules_file should be false."""


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


_RISK_REVIEW_KEYS = {
    "approved",
    "rejection_reasons",
    "risk_flags",
    "max_allowed_dollars",
    "requires_human_confirmation",
}


def _structured(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _structured(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_structured(v) for v in value]
    if isinstance(value, tuple):
        return [_structured(v) for v in value]
    return value


def _risk_string_list(value: Any, *, max_items: int = 16, max_len: int = 200) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:max_items]:
        text = str(item).strip()
        if text:
            out.append(text[:max_len])
    return out


def _risk_safe_dollars(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return max(0.0, default)


def _risk_review_fallback(reason: str) -> dict[str, Any]:
    return {
        "approved": False,
        "rejection_reasons": [reason or "invalid_risk_control_review_json"],
        "risk_flags": ["llm_risk_review_invalid"],
        "max_allowed_dollars": 0.0,
        "requires_human_confirmation": True,
    }


def _nested_get(payload: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        cur: Any = payload
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if cur is not None:
            return cur
    return None


def _add_unique(items: list[str], text: str) -> None:
    if text and text not in items:
        items.append(text)


def _apply_structural_risk_cautions(
    result: dict[str, Any],
    risk_context: dict[str, Any],
    deterministic_allowed_dollars: float,
) -> None:
    action_type = str(_nested_get(
        risk_context,
        ("action_type",),
        ("trade", "action_type"),
        ("decision", "action_type"),
    ) or "entry").lower()
    is_entry = action_type == "entry"

    minutes = _nested_get(
        risk_context,
        ("minutes_to_resolution",),
        ("minutes_to_close",),
        ("market", "minutes_to_resolution"),
        ("market", "minutes_to_close"),
        ("timing", "minutes_to_resolution"),
        ("timing", "minutes_to_close"),
    )
    try:
        minutes_left = float(minutes)
    except (TypeError, ValueError):
        minutes_left = None

    if minutes_left is not None:
        if minutes_left < 5 and is_entry:
            result["approved"] = False
            _add_unique(result["rejection_reasons"], "time_to_resolution_under_5_min")
            _add_unique(result["risk_flags"], "near_resolution_hard_reject")
            result["max_allowed_dollars"] = 0.0
        elif minutes_left < 20 and is_entry:
            result["approved"] = False
            _add_unique(result["rejection_reasons"], "time_to_resolution_5_to_20_min_entry")
            _add_unique(result["risk_flags"], "near_resolution_entry_rejected")
            result["max_allowed_dollars"] = 0.0
        elif minutes_left < 60:
            _add_unique(result["risk_flags"], "near_resolution_strict_review")
            result["max_allowed_dollars"] = min(
                result["max_allowed_dollars"],
                deterministic_allowed_dollars * 0.50,
            )
        elif minutes_left > 72 * 60:
            _add_unique(result["risk_flags"], "long_horizon_requires_supported_category")

    liquidity = _nested_get(
        risk_context,
        ("liquidity_dollars",),
        ("market", "liquidity_dollars"),
        ("liquidity", "liquidity_dollars"),
    )
    min_liquidity = _nested_get(
        risk_context,
        ("min_liquidity_dollars",),
        ("risk_limits", "min_liquidity_dollars"),
        ("limits", "min_liquidity_dollars"),
    )
    min_liquidity = _risk_safe_dollars(min_liquidity, default=0.0)
    try:
        liquidity_dollars = float(liquidity)
    except (TypeError, ValueError):
        liquidity_dollars = None

    if liquidity_dollars is not None and min_liquidity > 0:
        if liquidity_dollars < min_liquidity:
            result["approved"] = False
            _add_unique(result["rejection_reasons"], "insufficient_liquidity")
            _add_unique(result["risk_flags"], "illiquid_market")
            result["max_allowed_dollars"] = 0.0
        elif liquidity_dollars < min_liquidity * 2:
            _add_unique(result["risk_flags"], "thin_liquidity")
            result["requires_human_confirmation"] = True

    corr_exposure = _nested_get(
        risk_context,
        ("correlated_exposure",),
        ("correlated_exposure_dollars",),
        ("exposure", "correlated_dollars"),
    )
    corr_cap = _nested_get(
        risk_context,
        ("max_correlated_exposure_dollars",),
        ("risk_limits", "max_correlated_exposure_dollars"),
        ("limits", "max_correlated_exposure_dollars"),
    )
    corr_cap = _risk_safe_dollars(corr_cap, default=0.0)
    try:
        correlated = float(corr_exposure)
    except (TypeError, ValueError):
        correlated = None

    if correlated is not None and corr_cap > 0:
        if correlated > corr_cap:
            result["approved"] = False
            _add_unique(result["rejection_reasons"], "correlated_exposure_cap")
            _add_unique(result["risk_flags"], "correlated_exposure_limit_exceeded")
            result["max_allowed_dollars"] = 0.0
        elif correlated >= corr_cap * 0.80:
            _add_unique(result["risk_flags"], "correlated_exposure_near_cap")
            result["requires_human_confirmation"] = True


def _normalize_risk_control_review(
    raw: Any,
    deterministic_assessment: dict[str, Any],
    risk_context: dict[str, Any],
    deterministic_allowed_dollars: float,
) -> dict[str, Any]:
    cap = _risk_safe_dollars(deterministic_allowed_dollars)

    if not isinstance(raw, dict):
        result = _risk_review_fallback("invalid_risk_control_review_json")
    elif not _RISK_REVIEW_KEYS.issubset(raw.keys()):
        result = _risk_review_fallback("missing_risk_control_review_fields")
    elif not isinstance(raw.get("approved"), bool) or not isinstance(raw.get("requires_human_confirmation"), bool):
        result = _risk_review_fallback("invalid_risk_control_review_field_types")
    else:
        result = {
            "approved": bool(raw["approved"]),
            "rejection_reasons": _risk_string_list(raw.get("rejection_reasons")),
            "risk_flags": _risk_string_list(raw.get("risk_flags")),
            "max_allowed_dollars": min(_risk_safe_dollars(raw.get("max_allowed_dollars")), cap),
            "requires_human_confirmation": bool(raw["requires_human_confirmation"]),
        }

    deterministic_approved = deterministic_assessment.get("approved") is True
    if not deterministic_approved:
        result["approved"] = False
        result["max_allowed_dollars"] = 0.0
        _add_unique(result["risk_flags"], "deterministic_risk_rejected")
        failed = deterministic_assessment.get("checks_failed", [])
        if not isinstance(failed, list):
            failed = [str(failed)] if failed else []
        if failed:
            for failure in failed:
                _add_unique(result["rejection_reasons"], str(failure)[:200])
        else:
            _add_unique(result["rejection_reasons"], "deterministic_risk_rejected")
    else:
        _apply_structural_risk_cautions(result, risk_context, cap)
        result["max_allowed_dollars"] = min(_risk_safe_dollars(result["max_allowed_dollars"]), cap)

    if result["requires_human_confirmation"] and result["approved"]:
        result["approved"] = False
        _add_unique(result["rejection_reasons"], "human_confirmation_required")

    return {key: result[key] for key in (
        "approved",
        "rejection_reasons",
        "risk_flags",
        "max_allowed_dollars",
        "requires_human_confirmation",
    )}


def run_risk_control_review(
    cfg: Config,
    *,
    risk_context: dict[str, Any],
    deterministic_assessment: dict[str, Any] | Any,
    deterministic_allowed_dollars: float,
) -> dict[str, Any]:
    """
    Ask the LLM for a JSON-only risk-control review, then fail closed.

    This does not place orders and does not import or call trading.py. The
    deterministic risk assessment remains authoritative: if it rejected the
    trade, the returned review is rejected even if the LLM says approved.
    """
    context = _structured(risk_context)
    if not isinstance(context, dict):
        context = {"risk_context": context}

    assessment = _structured(deterministic_assessment)
    if not isinstance(assessment, dict):
        assessment = {"approved": False, "checks_failed": ["invalid_deterministic_assessment"]}

    cap = _risk_safe_dollars(deterministic_allowed_dollars)
    payload = {
        "risk_context": context,
        "deterministic_risk_assessment": assessment,
        "deterministic_allowed_dollars": cap,
        "review_instruction": (
            "Add only risk-control caution. Do not seek profit, calculate edge, "
            "recommend size above the deterministic cap, or override deterministic rejection."
        ),
    }
    user = (
        "Risk-control review payload JSON:\n"
        f"{json.dumps(payload, sort_keys=True, default=str)}\n\n"
        "Return JSON only."
    )

    try:
        raw = call_json(cfg, _RISK_CONTROL_REVIEW_SYSTEM, user, max_tokens=700, temperature=0.0)
    except Exception as exc:
        logger.error(_MODULE, "risk_control_review_failed", err=str(exc))
        raw = _risk_review_fallback(f"risk_control_review_failed: {exc}")

    return _normalize_risk_control_review(raw, assessment, context, cap)


_PROB_CONFIDENCE_BANDS: list[tuple[float, str]] = [
    (0.85, "high"),
    (0.70, "medium-high"),
    (0.55, "medium"),
    (0.40, "medium-low"),
]


def _confidence_float_to_label(value: float) -> str:
    """Map a 0.0–1.0 confidence float to the internal label scale."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "low"
    v = max(0.0, min(1.0, v))
    for floor, label in _PROB_CONFIDENCE_BANDS:
        if v >= floor:
            return label
    return "low"


def _string_list(value: Any, *, max_items: int = 12, max_len: int = 200) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:max_items]:
        text = str(item).strip()
        if text:
            out.append(text[:max_len])
    return out


def _probability_estimator_fallback(p_market: float, reason: str) -> dict[str, Any]:
    """Safe spec-shaped fallback when the estimator is unavailable or invalid."""
    p_yes = max(0.01, min(0.99, float(p_market)))
    p_no = round(1.0 - p_yes, 6)
    return {
        "estimated_yes_probability": round(p_yes, 6),
        "estimated_no_probability":  p_no,
        "confidence":                0.25,
        "main_factors":              ["Market-implied probability used as fallback"],
        "uncertainty_factors":       [reason or "Probability estimator unavailable or returned invalid JSON"],
        "probability_rationale":     "No reliable estimator output was available, so the system stayed anchored to the market price.",
        "overconfidence_warning":    True,
    }


def _normalize_probability_estimator_result(raw: Any, p_market: float) -> dict[str, Any]:
    """
    Validate and normalize a probability-estimator JSON object.

    Guarantees in the returned dict:
      - all spec keys are present
      - estimated_yes_probability is a float in [0.01, 0.99]
      - estimated_no_probability == 1 - estimated_yes_probability (rounded)
      - confidence is a float in [0.0, 1.0]
      - main_factors / uncertainty_factors are list[str]
      - probability_rationale is a string
      - overconfidence_warning is a bool
    """
    if not isinstance(raw, dict):
        return _probability_estimator_fallback(p_market, "LLM response was not a JSON object")

    try:
        yes_raw = float(raw.get("estimated_yes_probability"))
    except (TypeError, ValueError):
        return _probability_estimator_fallback(p_market, "estimated_yes_probability missing or non-numeric")
    p_yes = max(0.01, min(0.99, yes_raw))
    p_no = round(1.0 - p_yes, 6)

    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(raw.get("probability_rationale", "") or "").strip()
    if not rationale:
        rationale = "No rationale provided."

    warning = raw.get("overconfidence_warning", True)
    if not isinstance(warning, bool):
        warning = bool(warning)

    main_factors = _string_list(raw.get("main_factors"))
    uncertainty_factors = _string_list(raw.get("uncertainty_factors"))

    # Spec rule: any large move from baseline should carry an overconfidence
    # warning even if the LLM forgot to set it.
    if abs(p_yes - max(0.01, min(0.99, float(p_market)))) > 0.10:
        warning = True

    return {
        "estimated_yes_probability": round(p_yes, 6),
        "estimated_no_probability":  p_no,
        "confidence":                round(confidence, 6),
        "main_factors":              main_factors,
        "uncertainty_factors":       uncertainty_factors,
        "probability_rationale":     rationale[:1000],
        "overconfidence_warning":    bool(warning),
    }


def run_probability_estimator(
    cfg: Config,
    *,
    ticker: str,
    title: str,
    rules: str,
    market_implied_yes_probability: float,
    yes_bid: int,
    yes_ask: int,
    no_bid: int,
    no_ask: int,
    spread_cents: int,
    liquidity_dollars: float,
    volume_24h: int,
    minutes_to_resolution: float,
    research_summary: str,
    sentiment_payload: dict[str, Any] | None = None,
    weird_move_payload: dict[str, Any] | None = None,
    related_market_context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Probability calibration wrapper.

    Returns the strict spec-shaped JSON dict (see _PROBABILITY_ESTIMATOR_SYSTEM).
    Never raises — on any failure, returns the market-anchored fallback dict
    with confidence=0.25 and overconfidence_warning=True.

    This wrapper does not recommend a trade, calculate edge, suggest a side,
    place orders, or override risk controls. It only emits a calibrated
    probability estimate and uncertainty notes.
    """
    p_market = max(0.01, min(0.99, float(market_implied_yes_probability)))

    payload = {
        "ticker": ticker,
        "title": title,
        "resolution_rules": rules,
        "market_implied_yes_probability": round(p_market, 4),
        "prices_cents": {
            "yes_bid": yes_bid, "yes_ask": yes_ask,
            "no_bid":  no_bid,  "no_ask":  no_ask,
            "spread":  spread_cents,
        },
        "liquidity_dollars": liquidity_dollars,
        "volume_24h": volume_24h,
        "minutes_to_resolution": minutes_to_resolution,
        "research_summary": (research_summary or "")[:4000],
        "sentiment": sentiment_payload or {},
        "weird_move": weird_move_payload or {},
        "related_market_context": related_market_context or [],
    }

    user = (
        "Probability estimation payload JSON:\n"
        f"{json.dumps(payload, sort_keys=True, default=str)}\n\n"
        "Estimate the true YES probability for this market. Anchor to the "
        "market-implied probability and adjust only when evidence justifies "
        "it. Return JSON only."
    )

    try:
        raw = call_json(cfg, _PROBABILITY_ESTIMATOR_SYSTEM, user, max_tokens=1000)
    except Exception as exc:
        logger.error(_MODULE, "probability_estimator_failed",
                     ticker=ticker, err=str(exc))
        return _probability_estimator_fallback(
            p_market, f"Probability estimator call failed: {exc}"
        )

    return _normalize_probability_estimator_result(raw, p_market)


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
    """
    Legacy adapter — internally routes through run_probability_estimator
    using the new spec prompt, then re-shapes the result into the legacy
    dict that prediction_model._llm_component consumes.

    Returned dict has the legacy keys:
      yes_probability        — float 0–1
      confidence             — label (low | medium-low | medium | medium-high | high)
      reasoning              — probability_rationale
      assumptions            — main_factors
      invalidation_conditions — uncertainty_factors
    """
    # Build a coarse market baseline from yes_ask alone (cents → probability).
    # prediction_model._llm_component re-clamps the result to ±10pp from the
    # true mid before blending, so this approximation is safe.
    p_market = max(0.01, min(0.99, yes_ask / 100.0))
    yes_bid = max(0, 100 - no_ask)
    spec = run_probability_estimator(
        cfg,
        ticker=ticker,
        title=title,
        rules=rules,
        market_implied_yes_probability=p_market,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=max(0, 100 - yes_ask),
        no_ask=no_ask,
        spread_cents=max(0, yes_ask - yes_bid),
        liquidity_dollars=0.0,
        volume_24h=0,
        minutes_to_resolution=0.0,
        research_summary=research_summary,
        sentiment_payload=sentiment_data or {},
    )
    return {
        "yes_probability":         spec["estimated_yes_probability"],
        "confidence":              _confidence_float_to_label(spec["confidence"]),
        "reasoning":               spec["probability_rationale"],
        "assumptions":             list(spec["main_factors"]),
        "invalidation_conditions": list(spec["uncertainty_factors"]),
    }


def _normalize_postmortem_rule_changes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, str):
            rule = item.strip()
            reason = "Legacy rule-change proposal from postmortem reviewer."
            priority = "medium"
        elif isinstance(item, dict):
            rule = str(item.get("rule", "")).strip()
            reason = str(item.get("reason", "")).strip()
            priority = str(item.get("priority", "medium")).strip().lower()
        else:
            continue
        if not rule or rule.lower() == "none":
            continue
        if priority not in {"low", "medium", "high"}:
            priority = "medium"
        out.append({
            "rule": rule[:300],
            "reason": (reason or "No reason provided by reviewer.")[:500],
            "priority": priority,
            "requires_human_approval": True,
        })
    return out


def _postmortem_missing_evidence(payload: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in (
        "research_summary",
        "sentiment_result",
        "probability_estimate",
        "edge_result",
        "risk_assessment",
        "execution_log",
        "market_structure_notes",
        "resolution_rules",
    ):
        value = payload.get(key)
        if value in (None, "", {}, []):
            missing.append(f"Missing {key}")
    return missing


def _normalize_postmortem_result(raw: Any, trade_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    root_causes = _string_list(raw.get("root_causes"), max_items=12, max_len=240)
    if not root_causes:
        analysis = str(raw.get("analysis", "")).strip()
        if analysis:
            root_causes = [analysis[:240]]

    proposed = _normalize_postmortem_rule_changes(raw.get("proposed_rule_changes"))
    if not proposed:
        proposed = _normalize_postmortem_rule_changes(raw.get("rule_changes_proposed"))
    if not proposed:
        proposal = str(raw.get("rule_change_proposal", "")).strip()
        if proposal and proposal.lower() != "none":
            proposed = _normalize_postmortem_rule_changes([{
                "rule": proposal,
                "reason": "Legacy rule_change_proposal field from postmortem reviewer.",
                "priority": "medium",
            }])

    missing = _postmortem_missing_evidence(payload)
    data_quality = _string_list(raw.get("data_quality_issues"), max_items=12, max_len=240)
    for issue in missing:
        _add_unique(data_quality, issue)

    output_trade_id = str(raw.get("trade_id") or trade_id)
    if output_trade_id != trade_id:
        _add_unique(root_causes, "LLM postmortem trade_id did not match input trade_id")
        output_trade_id = trade_id

    should_update = bool(raw.get("should_update_rules_file", False)) and bool(proposed)

    return {
        "trade_id": output_trade_id,
        "good_process_bad_outcome": bool(raw.get("good_process_bad_outcome", raw.get("was_variance", False))),
        "root_causes": root_causes or ["Loss requires human review; evidence was insufficient for a specific cause."],
        "data_quality_issues": data_quality,
        "reasoning_issues": _string_list(raw.get("reasoning_issues"), max_items=12, max_len=240),
        "risk_issues": _string_list(raw.get("risk_issues"), max_items=12, max_len=240),
        "execution_issues": _string_list(raw.get("execution_issues"), max_items=12, max_len=240),
        "market_structure_issues": _string_list(raw.get("market_structure_issues"), max_items=12, max_len=240),
        "proposed_rule_changes": proposed,
        "should_update_rules_file": should_update,
    }


def run_postmortem(
    cfg: Config,
    ticker: str,
    title: str,
    original_thesis: str,
    estimated_yes_prob: float,
    entry_price_cents: int,
    actual_result: str,
    *,
    trade_id: str = "",
    side: str = "",
    contracts: int = 0,
    exit_price_cents: int = 0,
    pnl_dollars: float = 0.0,
    result: str = "loss",
    research_summary: str = "",
    sentiment_result: dict[str, Any] | None = None,
    probability_estimate: dict[str, Any] | None = None,
    edge_result: dict[str, Any] | None = None,
    risk_assessment: dict[str, Any] | None = None,
    execution_log: dict[str, Any] | None = None,
    market_structure_notes: dict[str, Any] | None = None,
    time_to_resolution_at_entry_minutes: float = 0.0,
    resolution_rules: str = "",
) -> dict[str, Any]:
    payload = {
        "trade_id": trade_id,
        "ticker": ticker,
        "side": side,
        "contracts": contracts,
        "entry_price_cents": entry_price_cents,
        "exit_price_cents": exit_price_cents,
        "pnl_dollars": pnl_dollars,
        "result": result,
        "original_thesis": original_thesis,
        "estimated_yes_probability": estimated_yes_prob,
        "market_price_at_entry": entry_price_cents,
        "actual_result": actual_result,
        "research_summary": research_summary,
        "sentiment_result": sentiment_result or {},
        "probability_estimate": probability_estimate or {},
        "edge_result": edge_result or {},
        "risk_assessment": risk_assessment or {},
        "execution_log": execution_log or {},
        "market_structure_notes": market_structure_notes or {},
        "time_to_resolution_at_entry_minutes": time_to_resolution_at_entry_minutes,
        "resolution_rules": resolution_rules,
    }
    user = (
        "Postmortem payload JSON:\n"
        f"{json.dumps(payload, sort_keys=True, default=str)}\n\n"
        "Review only this losing trade. Return JSON only."
    )
    raw = call_json(cfg, _POSTMORTEM_SYSTEM, user, max_tokens=1800, temperature=0.0)
    return _normalize_postmortem_result(raw, trade_id, payload)
