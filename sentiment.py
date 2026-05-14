"""
Sentiment & narrative analysis.

Converts a ResearchResult into a SentimentResult that matches the
narrative-analysis spec:

  {
    "sentiment_score": -1.0..1.0,
    "narrative_direction": "supports_yes|supports_no|mixed|neutral",
    "source_credibility": 0.0..1.0,
    "event_relevance":    0.0..1.0,
    "market_impact_estimate_cents": int,
    "confidence":         0.0..1.0,
    "contradictions":     [str],
    "rumor_risk":         "low|medium|high"
  }

Rules enforced:
  R1. Social sentiment cannot create high confidence alone — confidence
      is capped when the signal is driven by social-tier sources.
  R2. Official data overrides social chatter — when official sources have
      meaningful weight and disagree with social, the official direction
      wins.
  R3. Conflicting sources reduce confidence — each detected contradiction
      removes a fixed amount.
  R4. Stale sources reduce recency — already baked into the per-item
      recency_score; reflected in the weighted formula.
  R5. Market impact is capped unless official confirmation exists — the
      cap tightens as rumor_risk rises.

No LLM calls. All scoring is deterministic from the per-item fields
already produced by research_agents / category_research.
"""
from __future__ import annotations

import logger
from models import Market, ResearchItem, ResearchResult, SentimentResult

_MODULE = "sentiment"

_DIRECTION_SCORE: dict[str, float] = {
    "supports_yes":  1.0,
    "supports_no":  -1.0,
    "neutral":       0.0,
    "unclear":       0.0,
}

# ── Credibility tiers ────────────────────────────────────────────────────────
# Aligns with research_agents.CREDIBILITY:
#   official 0.95 / major_wire 0.85 / niche_analyst 0.70 /
#   verified_x 0.60 / reddit 0.35 / anonymous 0.15
_OFFICIAL_FLOOR = 0.80   # official + major_wire
_SOCIAL_CEIL    = 0.60   # verified_x / reddit / anonymous
_STALE_RECENCY  = 0.70   # older than ~72h
_FRESH_RECENCY  = 0.85   # newer than ~6h

# ── Caps under R5 (market_impact_estimate_cents) ─────────────────────────────
_IMPACT_CAP_HIGH_RUMOR    = 3     # cents — almost nothing without confirmation
_IMPACT_CAP_MEDIUM_RUMOR  = 8     # cents — analyst/wire only, no official
_IMPACT_CAP_LOW_RUMOR     = 20    # cents — official confirmation present
_IMPACT_SCALE             = 25    # raw scale before capping

# ── Confidence caps under R1 ─────────────────────────────────────────────────
_CONF_CAP_HIGH_RUMOR   = 0.40
_CONF_CAP_MEDIUM_RUMOR = 0.65
_CONF_PENALTY_PER_CONTRADICTION = 0.08

# Threshold of "meaningful" weight share for mixed/override detection
_MIN_TIER_WEIGHT_SHARE = 0.20


# ── Weighted aggregates ──────────────────────────────────────────────────────

def _weighted(items: list[ResearchItem]) -> tuple[float, float, float, float]:
    """
    Returns (signal, total_weight, avg_credibility, avg_relevance).
    Signal = sum(direction * weight) / sum(weight)
    weight = relevance * credibility * recency_score
    """
    signal_num = 0.0
    cred_num = 0.0
    rel_num = 0.0
    total = 0.0

    for item in items:
        d = _DIRECTION_SCORE.get(item.direction, 0.0)
        w = item.relevance * item.credibility * item.recency_score
        if w <= 0:
            continue
        signal_num += d * w
        cred_num   += item.credibility * w
        rel_num    += item.relevance * w
        total      += w

    if total == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    return signal_num / total, total, cred_num / total, rel_num / total


def _split_by_tier(items: list[ResearchItem]) -> tuple[list[ResearchItem], list[ResearchItem]]:
    official = [i for i in items if i.credibility >= _OFFICIAL_FLOOR]
    social   = [i for i in items if i.credibility <= _SOCIAL_CEIL]
    return official, social


# ── Rumor risk (drives R5 impact cap and R1 confidence cap) ──────────────────

def _rumor_risk(
    items: list[ResearchItem],
    direction: str,
    official_weight: float,
    total_weight: float,
) -> str:
    """
    low    = at least one official-tier item (cred ≥ 0.80) supports the
             signal direction, OR official sources carry ≥20% of total weight
             and don't disagree with the headline.
    medium = no official confirmation but at least some major_wire / analyst
             items (cred ≥ 0.60) in the mix.
    high   = signal is driven purely by social / anonymous sources.
    """
    if total_weight == 0.0:
        return "low"   # nothing to be rumor-y about

    # Aligned official confirmation
    aligned_official = [
        i for i in items
        if i.credibility >= _OFFICIAL_FLOOR
        and i.direction == direction
    ]
    if aligned_official:
        return "low"

    official_share = official_weight / total_weight if total_weight else 0.0
    if official_share >= _MIN_TIER_WEIGHT_SHARE:
        # Official sources are in the mix even if their direction differs.
        return "low"

    has_midtier = any(_SOCIAL_CEIL < i.credibility < _OFFICIAL_FLOOR for i in items)
    if has_midtier:
        return "medium"
    return "high"


# ── Contradiction detection (R3) ─────────────────────────────────────────────

def _detect_contradictions(items: list[ResearchItem]) -> list[str]:
    contradictions: list[str] = []
    if not items:
        return contradictions

    official, social = _split_by_tier(items)

    # 1. Official vs social point in opposite directions
    if official and social:
        off_sig, _, _, _ = _weighted(official)
        soc_sig, _, _, _ = _weighted(social)
        if off_sig > 0.1 and soc_sig < -0.1:
            contradictions.append(
                f"Official sources lean YES ({off_sig:+.2f}) but social leans NO ({soc_sig:+.2f})"
            )
        elif off_sig < -0.1 and soc_sig > 0.1:
            contradictions.append(
                f"Official sources lean NO ({off_sig:+.2f}) but social leans YES ({soc_sig:+.2f})"
            )

    # 2. Stale official contradicted by fresh social
    stale_official = [i for i in official if i.recency_score <= _STALE_RECENCY]
    fresh_social   = [i for i in social   if i.recency_score >= _FRESH_RECENCY]
    if stale_official and fresh_social:
        stale_sig, _, _, _ = _weighted(stale_official)
        fresh_sig, _, _, _ = _weighted(fresh_social)
        if abs(stale_sig - fresh_sig) > 0.3:
            contradictions.append(
                f"Stale official signal ({stale_sig:+.2f}) conflicts with "
                f"fresh social chatter ({fresh_sig:+.2f})"
            )

    # 3. High-credibility minority contradicts the majority
    overall, _, _, _ = _weighted(items)
    if len(items) >= 4:
        if overall > 0.05:
            minority = [i for i in items if i.direction == "supports_no"]
        elif overall < -0.05:
            minority = [i for i in items if i.direction == "supports_yes"]
        else:
            minority = []

        if minority:
            minority_pct = len(minority) / len(items)
            avg_minority_cred = sum(i.credibility for i in minority) / len(minority)
            if minority_pct < 0.35 and avg_minority_cred >= 0.75:
                contradictions.append(
                    f"High-credibility minority ({len(minority)} items, "
                    f"avg cred {avg_minority_cred:.2f}) contradicts majority"
                )

    return contradictions


# ── Mixed-direction detection ────────────────────────────────────────────────

def _is_mixed(items: list[ResearchItem], signal: float) -> bool:
    """
    Returns True when meaningful weight sits on BOTH sides — i.e. the
    aggregate signal is muted because YES and NO each have credible support.
    """
    yes_items = [i for i in items if i.direction == "supports_yes"]
    no_items  = [i for i in items if i.direction == "supports_no"]
    if not yes_items or not no_items:
        return False

    _, yes_w, _, _ = _weighted(yes_items)
    _, no_w,  _, _ = _weighted(no_items)
    total = yes_w + no_w
    if total == 0:
        return False

    yes_share = yes_w / total
    no_share  = no_w / total
    if min(yes_share, no_share) >= _MIN_TIER_WEIGHT_SHARE and abs(signal) < 0.30:
        return True
    return False


# ── Main entry point ─────────────────────────────────────────────────────────

def analyze(market: Market, research: ResearchResult) -> SentimentResult:
    if not research.items:
        logger.warn(_MODULE, "no_items",
                    ticker=market.ticker,
                    reason=research.failed_reason or "empty")
        return _null_result(market.ticker)

    items = research.items
    signal, total_weight, avg_cred, avg_rel = _weighted(items)

    if total_weight == 0.0:
        return _null_result(market.ticker)

    # ── R2: official overrides social ────────────────────────────────────
    # If official sources have meaningful weight AND disagree with the
    # current aggregate direction, recompute the signal from official only.
    official, social = _split_by_tier(items)
    off_sig, off_w, _, _ = _weighted(official)
    if off_w > 0 and (off_w / total_weight) >= _MIN_TIER_WEIGHT_SHARE:
        if (off_sig > 0.1 and signal < -0.05) or (off_sig < -0.1 and signal > 0.05):
            logger.debug(_MODULE, "official_override",
                         ticker=market.ticker,
                         orig_signal=f"{signal:+.2f}",
                         official_signal=f"{off_sig:+.2f}")
            signal = off_sig   # official takes the wheel

    # Direction including 4-way "mixed"
    if _is_mixed(items, signal):
        narrative_direction = "mixed"
    elif signal > 0.05:
        narrative_direction = "supports_yes"
    elif signal < -0.05:
        narrative_direction = "supports_no"
    else:
        narrative_direction = "neutral"

    contradictions = _detect_contradictions(items)
    rumor_risk = _rumor_risk(items, narrative_direction, off_w, total_weight)

    # ── R1 + R3: confidence ──────────────────────────────────────────────
    signal_strength = abs(signal)
    base_conf = avg_cred * (0.5 + 0.5 * signal_strength)
    base_conf -= _CONF_PENALTY_PER_CONTRADICTION * len(contradictions)
    if narrative_direction == "mixed":
        base_conf *= 0.7   # mixed evidence inherently less actionable

    # R1: social sentiment cannot create high confidence alone
    if rumor_risk == "high":
        base_conf = min(base_conf, _CONF_CAP_HIGH_RUMOR)
    elif rumor_risk == "medium":
        base_conf = min(base_conf, _CONF_CAP_MEDIUM_RUMOR)

    confidence = max(0.0, min(1.0, base_conf))

    # ── R5: market impact capped unless official confirmation ────────────
    raw_impact = abs(signal) * confidence * _IMPACT_SCALE
    if rumor_risk == "high":
        cap = _IMPACT_CAP_HIGH_RUMOR
    elif rumor_risk == "medium":
        cap = _IMPACT_CAP_MEDIUM_RUMOR
    else:
        cap = _IMPACT_CAP_LOW_RUMOR
    market_impact = int(round(min(raw_impact, cap)))
    # Mixed: don't claim a direction-pointed impact
    if narrative_direction in ("mixed", "neutral"):
        market_impact = min(market_impact, 2)

    contributing_sources = list(dict.fromkeys(i.source for i in items if i.source))

    result = SentimentResult(
        ticker=market.ticker,
        sentiment_score=round(signal, 4),
        narrative_direction=narrative_direction,
        confidence=round(confidence, 4),
        market_impact_estimate_cents=market_impact,
        major_contradictions=contradictions,
        item_count=len(items),
        contributing_sources=contributing_sources,
        source_credibility=round(avg_cred, 4),
        event_relevance=round(avg_rel, 4),
        rumor_risk=rumor_risk,
    )

    logger.info(
        _MODULE, "analyzed",
        ticker=market.ticker,
        signal=f"{signal:+.3f}",
        narrative=narrative_direction,
        confidence=f"{confidence:.3f}",
        rumor_risk=rumor_risk,
        items=len(items),
        contradictions=len(contradictions),
        impact_cents=market_impact,
    )
    return result


def _null_result(ticker: str) -> SentimentResult:
    return SentimentResult(
        ticker=ticker,
        sentiment_score=0.0,
        narrative_direction="neutral",
        confidence=0.0,
        market_impact_estimate_cents=0,
        major_contradictions=[],
        item_count=0,
        contributing_sources=[],
        source_credibility=0.0,
        event_relevance=0.0,
        rumor_risk="low",
    )


def batch_analyze(
    markets: list[Market],
    research_map: dict[str, ResearchResult],
) -> dict[str, SentimentResult]:
    results: dict[str, SentimentResult] = {}
    for market in markets:
        research = research_map.get(market.ticker)
        if research is None:
            logger.warn(_MODULE, "no_research", ticker=market.ticker)
            continue
        results[market.ticker] = analyze(market, research)
    return results
