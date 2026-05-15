from __future__ import annotations
from typing import Optional

import features
import llm
import logger
from config import Config
from models import (
    FeatureVector,
    Market,
    ProbabilityEstimate,
    ResearchResult,
    SentimentResult,
    WeirdMoveSignal,
)

_MODULE = "prediction_model"

# Confidence → numeric weight for downstream edge calculation
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "low": 0.3,
    "medium-low": 0.5,
    "medium": 0.65,
    "medium-high": 0.80,
    "high": 0.90,
}

# Ensemble weights (must sum to 1.0).
# historical_model_prob is absent — its 0.15 weight is folded into market price.
_W_MARKET   = 0.70    # market mid-price (0.55 base + 0.15 historical fallback)
_W_RESEARCH = 0.25    # sentiment-adjusted price
_W_LLM      = 0.05    # LLM reasoning, capped ±10pp from market

# Defensive cap on sentiment-driven shift, regardless of upstream caps.
_SENTIMENT_SHIFT_CAP_PP = 0.10     # ±10pp

# LLM cap on adjustment from market price.
_LLM_SHIFT_CAP_PP = 0.10           # ±10pp

# Confidence step-down thresholds
_THIN_LIQUIDITY_USD       = 500.0      # below → step down once
_NEAR_RESOLUTION_MIN      = 30.0       # minutes_to_settlement below → step down once
_SPARSE_HISTORY_TICKS     = 5          # fewer ticks → step down once
_HIGH_VOLATILITY_CENTS    = 4.0        # stdev > this → step down once
_NEIGHBOR_DISAGREE_PROB   = 0.20       # > 20pp from neighbor mean → step down once
_MIN_NEIGHBORS_FOR_CHECK  = 2          # need at least 2 siblings for a sanity check

# Weird-move classifications and the step-down they apply (never a boost).
_WEIRD_MOVE_STEPS: dict[str, int] = {
    "stale_book_artifact":        2,   # bad signal — strong penalty
    "rumor_move":                 1,   # price moved without volume confirmation
    "liquidity_gap_move":         1,   # spread spike / thin book
    "related_market_dislocation": 1,   # siblings disagree
    "news_confirmed_move":        0,   # no penalty — but no automatic boost either
}

_CONF_ORDER = ["low", "medium-low", "medium", "medium-high", "high"]


def _step_down(confidence: str, steps: int = 1) -> str:
    if steps <= 0:
        return confidence
    idx = _CONF_ORDER.index(confidence) if confidence in _CONF_ORDER else 0
    return _CONF_ORDER[max(0, idx - steps)]


def _clamp(v: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, v))


def _signed_sentiment_shift(sentiment: Optional[SentimentResult]) -> float:
    """
    Convert sentiment.market_impact_estimate_cents (an unsigned magnitude)
    into a signed probability shift in (-cap, +cap), respecting the
    narrative direction. Mixed/neutral narratives produce no shift.
    """
    if sentiment is None:
        return 0.0
    magnitude = sentiment.market_impact_estimate_cents / 100.0
    direction = sentiment.narrative_direction
    if direction == "supports_yes":
        signed = +magnitude
    elif direction == "supports_no":
        signed = -magnitude
    else:
        signed = 0.0   # mixed / neutral / unknown
    return max(-_SENTIMENT_SHIFT_CAP_PP, min(_SENTIMENT_SHIFT_CAP_PP, signed))


def estimate(
    market: Market,
    research: Optional[ResearchResult],
    sentiment: Optional[SentimentResult],
    weird_move: Optional[WeirdMoveSignal],
    cfg: Config,
    *,
    spot_price: Optional[float] = None,
    markets_by_event: Optional[dict[str, list[Market]]] = None,
) -> ProbabilityEstimate:
    """
    Estimate the true probability of YES using an ensemble of:
      p_market   (70%) — market mid-price (anchor)
      p_research (25%) — market price shifted by signed sentiment impact,
                         capped to ±10pp
      p_llm      ( 5%) — LLM reasoning, capped to ±10pp from p_market

    Then apply rule-layer confidence step-downs derived from a FeatureVector:
      - thin liquidity / near resolution / sparse history
      - high historical volatility
      - large neighbor-prob disagreement (sanity check, not a price input)
      - weird_move classification (rumor/liquidity_gap/dislocation/stale)

    Missing inputs degrade specific features to neutral defaults; the
    function never raises on missing data and falls back to a low-confidence
    midpoint estimate on any unexpected error.
    """
    try:
        return _estimate_impl(
            market, research, sentiment, weird_move, cfg,
            spot_price=spot_price,
            markets_by_event=markets_by_event,
        )
    except Exception as exc:
        logger.error(_MODULE, "estimate_failed",
                     ticker=market.ticker, err=str(exc))
        return _fallback(market)


def _estimate_impl(
    market: Market,
    research: Optional[ResearchResult],
    sentiment: Optional[SentimentResult],
    weird_move: Optional[WeirdMoveSignal],
    cfg: Config,
    *,
    spot_price: Optional[float],
    markets_by_event: Optional[dict[str, list[Market]]],
) -> ProbabilityEstimate:
    # ── Build the feature vector (fail-soft over None inputs) ────────────
    fv: FeatureVector = features.extract_features(
        market, sentiment, weird_move,
        spot_price=spot_price,
        markets_by_event=markets_by_event,
    )

    # ── Component 1: market mid-price (anchor) ───────────────────────────
    p_market = _clamp(fv.mid_price / 100.0)

    # ── Component 2: sentiment-shifted price (signed, capped) ────────────
    sentiment_shift = _signed_sentiment_shift(sentiment)
    p_research = _clamp(p_market + sentiment_shift)

    # ── Component 3: LLM, capped ±10pp from market ───────────────────────
    p_llm, llm_confidence, reasoning, assumptions, invalidations = _llm_component(
        market, research, sentiment, weird_move, cfg, p_market
    )

    # ── Ensemble blend ───────────────────────────────────────────────────
    p_model = _clamp(
        _W_MARKET   * p_market
        + _W_RESEARCH * p_research
        + _W_LLM      * p_llm
    )

    # ── Rule-layer confidence step-downs ─────────────────────────────────
    confidence = llm_confidence
    confidence = _apply_microstructure_stepdowns(market, confidence)
    confidence = _apply_feature_stepdowns(fv, p_model, market, confidence)
    confidence = _apply_weird_move_stepdowns(weird_move, confidence, market.ticker)
    confidence = _apply_sentiment_research_stepdowns(
        sentiment, research, confidence, market.ticker
    )

    result = ProbabilityEstimate(
        ticker=market.ticker,
        yes_probability=p_model,
        confidence=confidence,
        reasoning=reasoning,
        assumptions=assumptions,
        invalidation_conditions=invalidations,
    )

    logger.info(
        _MODULE, "estimated",
        ticker=market.ticker,
        yes_prob=f"{p_model:.1%}",
        p_market=f"{p_market:.1%}",
        p_research=f"{p_research:.1%}",
        p_llm=f"{p_llm:.1%}",
        market_ask=f"{market.yes_ask}¢",
        confidence=confidence,
        sentiment_shift=f"{sentiment_shift:+.3f}",
        hist_vol=f"{fv.historical_volatility:.2f}",
        siblings=len(fv.related_market_prices),
    )
    return result


# ── LLM component ────────────────────────────────────────────────────────────

def _llm_component(
    market: Market,
    research: Optional[ResearchResult],
    sentiment: Optional[SentimentResult],
    weird_move: Optional[WeirdMoveSignal],
    cfg: Config,
    p_market: float,
) -> tuple[float, str, str, list[str], list[str]]:
    """
    Call the LLM probability estimator and clamp its output to ±10pp from
    p_market. On any error, default to p_market with "low" confidence.
    """
    p_llm = p_market
    llm_confidence = "low"
    reasoning = "LLM estimate unavailable — ensemble uses market price"
    assumptions: list[str] = []
    invalidations: list[str] = []

    move_note = ""
    if weird_move is not None and weird_move.flagged:
        move_note = f"\nNOTE: Unusual market activity detected — {weird_move.description}"

    research_text = research.raw_text if research is not None else "No research available."

    if sentiment is not None:
        sentiment_payload = {
            "sentiment_score":      sentiment.sentiment_score,
            "narrative_direction":  sentiment.narrative_direction,
            "confidence":           sentiment.confidence,
            "contradictions":       sentiment.major_contradictions,
            "market_impact_cents":  sentiment.market_impact_estimate_cents,
        }
    else:
        sentiment_payload = {
            "sentiment_score": 0.0,
            "narrative_direction": "neutral",
            "confidence": 0.0,
            "contradictions": [],
            "market_impact_cents": 0,
        }

    try:
        data = llm.estimate_probability(
            cfg,
            market.ticker,
            market.title,
            market.rules_primary,
            market.yes_ask,
            market.no_ask,
            research_text + move_note,
            sentiment_payload,
        )
        raw_llm = _clamp(float(data.get("yes_probability", p_market)))
        # Defense in depth: cap LLM adjustment to ±10pp from market price
        delta = max(-_LLM_SHIFT_CAP_PP, min(_LLM_SHIFT_CAP_PP, raw_llm - p_market))
        p_llm = _clamp(p_market + delta)
        cand = data.get("confidence", "low")
        llm_confidence = cand if cand in CONFIDENCE_WEIGHTS else "low"
        reasoning = data.get("reasoning", "")
        assumptions = data.get("assumptions", [])
        invalidations = data.get("invalidation_conditions", [])
    except Exception as exc:
        logger.error(_MODULE, "llm_failed", ticker=market.ticker, err=str(exc))

    return p_llm, llm_confidence, reasoning, assumptions, invalidations


# ── Confidence step-down rules ───────────────────────────────────────────────

def _apply_microstructure_stepdowns(market: Market, confidence: str) -> str:
    """Liquidity, time-to-resolution, and history-depth gates."""
    if market.liquidity_dollars < _THIN_LIQUIDITY_USD:
        confidence = _step_down(confidence)
        logger.debug(_MODULE, "thin_liquidity", ticker=market.ticker,
                     liquidity_usd=market.liquidity_dollars)
    if market.minutes_to_settlement < _NEAR_RESOLUTION_MIN:
        confidence = _step_down(confidence)
        logger.debug(_MODULE, "near_resolution", ticker=market.ticker,
                     minutes_to_settlement=market.minutes_to_settlement)
    if len(market.price_history) < _SPARSE_HISTORY_TICKS:
        confidence = _step_down(confidence)
        logger.debug(_MODULE, "sparse_history", ticker=market.ticker,
                     ticks=len(market.price_history))
    return confidence


def _apply_feature_stepdowns(
    fv: FeatureVector,
    p_model: float,
    market: Market,
    confidence: str,
) -> str:
    """Step down on high historical volatility and sibling disagreement."""
    if fv.historical_volatility > _HIGH_VOLATILITY_CENTS:
        confidence = _step_down(confidence)
        logger.debug(_MODULE, "high_volatility", ticker=market.ticker,
                     hist_vol=f"{fv.historical_volatility:.2f}")

    # Related markets: sanity check only — do NOT blend prices into p_model.
    # We use this as a confidence gate when our estimate disagrees strongly
    # with sibling-market consensus.
    if len(fv.related_market_prices) >= _MIN_NEIGHBORS_FOR_CHECK:
        neighbor_mean = sum(fv.related_market_prices) / len(fv.related_market_prices)
        disagreement = abs(p_model - neighbor_mean)
        if disagreement > _NEIGHBOR_DISAGREE_PROB:
            confidence = _step_down(confidence)
            logger.debug(_MODULE, "neighbor_disagreement",
                         ticker=market.ticker,
                         disagreement=f"{disagreement:.2f}",
                         neighbor_mean=f"{neighbor_mean:.2f}",
                         p_model=f"{p_model:.2f}")
    return confidence


def _apply_sentiment_research_stepdowns(
    sentiment: Optional[SentimentResult],
    research: Optional[ResearchResult],
    confidence: str,
    ticker: str,
) -> str:
    """
    Step down confidence based on signals the probability-estimator spec
    treats as uncertainty: rumor risk, contradictions, social-only signal,
    and missing/empty research.
    """
    # Missing or empty research evidence — keyed off the sentiment aggregate
    # (which already summarises the typed research evidence). Sentiment is the
    # canonical evidence carrier; raw ResearchResult.items can be empty in
    # callers that pre-aggregate, so the sentiment view is the right signal.
    if sentiment is None:
        confidence = _step_down(confidence)
        logger.debug(_MODULE, "missing_sentiment_stepdown", ticker=ticker)
        return confidence

    if getattr(sentiment, "item_count", 0) == 0:
        confidence = _step_down(confidence)
        logger.debug(_MODULE, "missing_research_stepdown", ticker=ticker)

    # Rumor risk medium/high — social/unverified signal cannot anchor
    # high confidence (mirrors sentiment.py's R1 cap, applied again here so
    # the LLM cannot relax it).
    if sentiment.rumor_risk in ("medium", "high"):
        confidence = _step_down(confidence)
        logger.debug(_MODULE, "rumor_risk_stepdown",
                     ticker=ticker, rumor_risk=sentiment.rumor_risk)

    # Conflicting evidence
    if sentiment.major_contradictions:
        confidence = _step_down(confidence)
        logger.debug(_MODULE, "contradictions_stepdown",
                     ticker=ticker,
                     contradictions=len(sentiment.major_contradictions))

    return confidence


def _apply_weird_move_stepdowns(
    weird_move: Optional[WeirdMoveSignal],
    confidence: str,
    ticker: str,
) -> str:
    """
    Weird-move signals are *confidence modifiers*, not directional inputs.
    They never boost confidence above what the LLM/microstructure produced.
    """
    if weird_move is None or not weird_move.flagged:
        return confidence
    steps = _WEIRD_MOVE_STEPS.get(weird_move.classification, 0)
    if steps > 0:
        new_conf = _step_down(confidence, steps)
        logger.debug(_MODULE, "weird_move_stepdown",
                     ticker=ticker,
                     classification=weird_move.classification,
                     steps=steps,
                     from_=confidence, to=new_conf)
        return new_conf
    return confidence


# ── Fallback ─────────────────────────────────────────────────────────────────

def _fallback(market: Market) -> ProbabilityEstimate:
    """Low-confidence estimate equal to the market midpoint."""
    if market.yes_bid > 0:
        mid = (market.yes_ask + market.yes_bid) / 2 / 100
    else:
        mid = (market.yes_ask or 50) / 100
    return ProbabilityEstimate(
        ticker=market.ticker,
        yes_probability=_clamp(mid),
        confidence="low",
        reasoning="Estimation failed — using market midpoint as fallback",
        assumptions=["No edge case — do not trade on this"],
        invalidation_conditions=[],
    )


def confidence_weight(confidence: str) -> float:
    return CONFIDENCE_WEIGHTS.get(confidence, 0.3)


def batch_estimate(
    markets: list[Market],
    research_map: dict[str, ResearchResult],
    sentiment_map: dict[str, SentimentResult],
    weird_move_map: dict[str, WeirdMoveSignal],
    cfg: Config,
    *,
    spot_prices: Optional[dict[str, float]] = None,
    markets_by_event: Optional[dict[str, list[Market]]] = None,
) -> dict[str, ProbabilityEstimate]:
    """Estimate probability for a batch of markets. Returns ticker → ProbabilityEstimate."""
    if markets_by_event is None:
        markets_by_event = {}
        for m in markets:
            key = m.event_ticker or m.ticker
            markets_by_event.setdefault(key, []).append(m)

    spot_prices = spot_prices or {}
    results: dict[str, ProbabilityEstimate] = {}

    for market in markets:
        research = research_map.get(market.ticker)
        sent = sentiment_map.get(market.ticker)
        weird = weird_move_map.get(market.ticker)
        if research is None or sent is None or weird is None:
            logger.warn(
                _MODULE, "missing_inputs",
                ticker=market.ticker,
                has_research=research is not None,
                has_sentiment=sent is not None,
                has_weird_move=weird is not None,
            )
            continue
        try:
            results[market.ticker] = estimate(
                market, research, sent, weird, cfg,
                spot_price=spot_prices.get(market.ticker),
                markets_by_event=markets_by_event,
            )
        except Exception as exc:
            logger.error(_MODULE, "batch_estimate_failed",
                         ticker=market.ticker, err=str(exc))
    return results
