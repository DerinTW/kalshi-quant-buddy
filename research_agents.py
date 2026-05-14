"""
Agent D: Research Agents

Five specialized agents gather external evidence for markets that passed filters.
Each agent targets a different source class, applies the appropriate credibility
weight, and returns structured ResearchItems.

Agents run in sequence; all results are merged, deduplicated, and cached.

Search backends (in priority order):
  1. Perplexity API  — real-time web search (requires PERPLEXITY_API_KEY)
  2. LLM fallback    — Claude reasoning from training data (potentially stale)

Source reliability table (from spec):
  official / government / exchange / court / Fed / EIA  →  0.95
  major wire (Reuters, AP, Bloomberg)                   →  0.85
  respected niche analyst/source                        →  0.70
  verified X account with track record                  →  0.60
  Reddit / social chatter                               →  0.35
  anonymous rumor                                       →  0.15
"""
from __future__ import annotations
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import requests

import category_research
import db
import llm
import logger
from config import Config
from models import Market, ResearchItem, ResearchResult

_MODULE = "research_agents"

# ── Source credibility table ──────────────────────────────────────────────────

CREDIBILITY: dict[str, float] = {
    "official":      0.95,   # government, central bank, exchange, court, EIA, FDA
    "major_wire":    0.85,   # Reuters, AP, Bloomberg, WSJ
    "niche_analyst": 0.70,   # respected domain-specific source
    "verified_x":    0.60,   # verified X / Twitter account with track record
    "reddit":        0.35,   # Reddit / social chatter
    "anonymous":     0.15,   # anonymous rumor / unverified claim
}

# Category → market-specific source metadata
_MARKET_SOURCES: dict[str, dict[str, Any]] = {
    "crypto":       {"name": "CoinDesk/CoinTelegraph/Glassnode",  "type": "niche_analyst",
                     "hint": "cryptocurrency blockchain on-chain data price analysis"},
    "economics":    {"name": "Fed/BLS/FRED/Census",               "type": "official",
                     "hint": "economic data GDP unemployment inflation Federal Reserve"},
    "politics":     {"name": "Politico/538/RealClearPolitics",     "type": "niche_analyst",
                     "hint": "polling election forecast political analysis"},
    "sports":       {"name": "ESPN/Sports-Reference/official team","type": "niche_analyst",
                     "hint": "sports statistics player performance team record"},
    "finance":      {"name": "SEC/EDGAR/CME/Bloomberg",           "type": "official",
                     "hint": "SEC filing earnings regulatory announcement market data"},
    "climate":      {"name": "NOAA/WMO/IPCC",                     "type": "official",
                     "hint": "weather climate official forecast temperature anomaly"},
    "entertainment":{"name": "Box Office Mojo/Nielsen",           "type": "niche_analyst",
                     "hint": "ratings viewership box office entertainment industry"},
    "default":      {"name": "General sources",                   "type": "niche_analyst",
                     "hint": "general news and analysis"},
}

# RSS feeds by category (best-effort; URLs may change over time)
_RSS_FEEDS: dict[str, list[str]] = {
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://coindesk.com/arc/outboundfeeds/rss/",
    ],
    "economics": [
        "https://feeds.bloomberg.com/markets/news.rss",
    ],
    "politics": [
        "https://feeds.npr.org/1001/rss.xml",
    ],
    "sports": [
        "https://www.espn.com/espn/rss/news",
    ],
    "default": [
        "https://feeds.reuters.com/reuters/topNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    ],
}


# ── Recency scoring ───────────────────────────────────────────────────────────

def recency_score(published_at: Optional[datetime]) -> float:
    if published_at is None:
        return 0.50   # unknown age → neutral
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600
    if age_h < 1:
        return 1.00
    if age_h < 6:
        return 0.95
    if age_h < 24:
        return 0.85
    if age_h < 72:
        return 0.70
    if age_h < 168:
        return 0.50
    return 0.20


# ── Deduplication ─────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def deduplicate(items: list[ResearchItem]) -> list[ResearchItem]:
    """
    Remove duplicate items:
      1. Same URL → keep the higher-credibility copy.
      2. Claims with Jaccard similarity > 0.65 → keep the higher-credibility copy.
    """
    # Pass 1: URL dedup
    by_url: dict[str, ResearchItem] = {}
    no_url: list[ResearchItem] = []
    for item in items:
        if item.url:
            existing = by_url.get(item.url)
            if existing is None or item.credibility > existing.credibility:
                by_url[item.url] = item
        else:
            no_url.append(item)

    candidates = list(by_url.values()) + no_url

    # Pass 2: claim similarity dedup (O(n²) — acceptable for < 50 items)
    kept: list[ResearchItem] = []
    for item in sorted(candidates, key=lambda x: -x.credibility):
        duplicate = False
        for existing in kept:
            if _jaccard(item.claim, existing.claim) > 0.65:
                duplicate = True
                break
        if not duplicate:
            kept.append(item)

    return kept


# ── Search backends ───────────────────────────────────────────────────────────

def _perplexity_search(query: str, cfg: Config) -> str:
    """Call Perplexity API. Raises RuntimeError if key is absent."""
    if not cfg.perplexity_api_key:
        raise RuntimeError("PERPLEXITY_API_KEY not set")
    resp = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {cfg.perplexity_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": cfg.perplexity_model,
            "messages": [{"role": "user", "content": query}],
            "search_recency_filter": "week",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _search(query: str, source_hint: str, market: Market, cfg: Config) -> str:
    """
    Real-time web search via Perplexity. Returns the raw text payload, or
    an empty string when Perplexity is unavailable.

    We intentionally do NOT fall back to the LLM here. The LLM does not have
    real-time data, so using it as a research substitute would mean inventing
    missing data — exactly what the role contract in llm.py prohibits. A
    missing real-time signal must surface as "no research available" rather
    than hallucinated content with a fake credibility score.
    """
    try:
        return _perplexity_search(query, cfg)
    except RuntimeError:
        logger.debug(_MODULE, "perplexity_unavailable",
                     msg="No Perplexity key — research produces no items")
    except Exception as exc:
        logger.warn(_MODULE, "perplexity_failed", err=str(exc),
                    msg="Perplexity search failed — research produces no items")
    return ""


# ── Item extraction ───────────────────────────────────────────────────────────

def _parse_date(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    raw = str(raw).strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%a, %d %b %Y %H:%M:%S %z",   # RSS format
        "%a, %d %b %Y %H:%M:%S %Z",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _items_from_raw(
    raw_text: str,
    market: Market,
    base_credibility: float,
    agent_name: str,
    cfg: Config,
) -> list[ResearchItem]:
    """
    Use the LLM to extract structured ResearchItems from raw search/RSS text.
    Applies the source-type credibility and recency scoring after extraction.

    LLM role: this is a structured-extraction call (raw text → typed claims).
    It is NOT used to invent claims when raw_text is empty — we short-circuit
    that case to satisfy the "do not invent missing data" contract.
    """
    if not raw_text or not raw_text.strip():
        return []

    raw_items = llm.extract_research_items(
        cfg,
        ticker=market.ticker,
        title=market.title,
        rules=market.rules_primary,
        raw_text=raw_text,
    )

    items: list[ResearchItem] = []
    for d in raw_items:
        if not d.get("claim"):
            continue
        pub = _parse_date(d.get("published_at"))
        r_score = recency_score(pub)
        rel = max(0.0, min(1.0, float(d.get("relevance", 0.5))))
        items.append(ResearchItem(
            source=d.get("source", agent_name),
            url=d.get("url", ""),
            published_at=pub,
            claim=d["claim"],
            direction=d.get("direction", "unclear"),
            relevance=rel,
            credibility=base_credibility,
            recency_score=r_score,
            summary=d.get("summary", ""),
            agent=agent_name,
        ))
    return items


# ── Base agent ────────────────────────────────────────────────────────────────

class ResearchAgent(ABC):
    name: str
    source_type: str

    @property
    def base_credibility(self) -> float:
        return CREDIBILITY.get(self.source_type, 0.50)

    @abstractmethod
    def _build_query(self, market: Market) -> str: ...

    def _source_hint(self) -> str:
        return self.name

    def run(self, market: Market, cfg: Config) -> list[ResearchItem]:
        query = self._build_query(market)
        try:
            raw = _search(query, self._source_hint(), market, cfg)
            items = _items_from_raw(raw, market, self.base_credibility, self.name, cfg)
            logger.debug(_MODULE, "agent_done",
                         agent=self.name, ticker=market.ticker, items=len(items))
            return items
        except Exception as exc:
            logger.warn(_MODULE, "agent_failed",
                        agent=self.name, ticker=market.ticker, err=str(exc))
            return []


# ── The five agents ───────────────────────────────────────────────────────────

class OfficialSourceAgent(ResearchAgent):
    """Government bodies, central banks, regulatory agencies, court records, exchanges."""
    name = "official_source"
    source_type = "official"

    def _build_query(self, market: Market) -> str:
        return (
            f'site:gov OR site:federalreserve.gov OR site:sec.gov OR site:cftc.gov '
            f'OR site:eia.gov OR site:bls.gov OR site:judiciary.gov '
            f'"{market.title}" official announcement statement ruling'
        )

    def _source_hint(self) -> str:
        return "government agencies, central banks, regulatory bodies, court records"


class NewsWireAgent(ResearchAgent):
    """Reuters, AP, Bloomberg, WSJ — high-credibility mainstream news."""
    name = "news_wire"
    source_type = "major_wire"

    def _build_query(self, market: Market) -> str:
        return (
            f'(Reuters OR "Associated Press" OR Bloomberg OR "Wall Street Journal") '
            f'{market.title} {market.category} news'
        )

    def _source_hint(self) -> str:
        return "Reuters, AP, Bloomberg, Wall Street Journal"


class RSSNewsAgent(ResearchAgent):
    """
    Fetches live RSS feeds for the market's category.
    Falls back to LLM search if all feeds fail.
    """
    name = "rss_news"
    source_type = "major_wire"

    def _build_query(self, market: Market) -> str:
        return (
            f'latest news about: {market.title} — '
            f'category: {market.category}'
        )

    def run(self, market: Market, cfg: Config) -> list[ResearchItem]:
        feeds = _RSS_FEEDS.get(market.category.lower(), _RSS_FEEDS["default"])
        rss_items: list[ResearchItem] = []

        for feed_url in feeds:
            try:
                feed_text = _fetch_rss(feed_url)
                candidates = _parse_rss(feed_text, market)
                rss_items.extend(candidates)
            except Exception as exc:
                logger.debug(_MODULE, "rss_feed_failed", url=feed_url, err=str(exc))

        if rss_items:
            # RSS items already structured — score them
            for item in rss_items:
                item.agent = self.name
                item.credibility = self.base_credibility
                item.recency_score = recency_score(item.published_at)
            logger.debug(_MODULE, "rss_done",
                         ticker=market.ticker, items=len(rss_items))
            return rss_items

        # All feeds failed — fall back to LLM search
        logger.debug(_MODULE, "rss_fallback", ticker=market.ticker)
        return super().run(market, cfg)


class TwitterXAgent(ResearchAgent):
    """X / Twitter: verified accounts and public discussion."""
    name = "twitter_x"
    source_type = "verified_x"

    def _build_query(self, market: Market) -> str:
        return (
            f'twitter OR X.com {market.title} {market.category} '
            f'prediction market tweet verified account'
        )

    def _source_hint(self) -> str:
        return "X / Twitter — verified accounts and public discussion threads"

    def run(self, market: Market, cfg: Config) -> list[ResearchItem]:
        items = super().run(market, cfg)
        # Social media items get a credibility penalty unless explicitly from verified source
        for item in items:
            if "verified" not in item.source.lower() and "official" not in item.source.lower():
                item.credibility = CREDIBILITY["reddit"]   # downgrade unverified
        return items


class RedditAgent(ResearchAgent):
    """Reddit subreddits: community discussion and sentiment."""
    name = "reddit"
    source_type = "reddit"

    def _build_query(self, market: Market) -> str:
        return (
            f'site:reddit.com {market.title} {market.category} '
            f'discussion prediction opinion'
        )

    def _source_hint(self) -> str:
        return "Reddit — community discussion, subreddit posts and comments"


class MarketSpecificAgent(ResearchAgent):
    """
    Category-specific domain sources:
    crypto → CoinDesk/Glassnode, economics → Fed/BLS,
    politics → 538/Politico, sports → ESPN, etc.
    """
    name = "market_specific"
    source_type = "niche_analyst"

    def _build_query(self, market: Market) -> str:
        meta = _MARKET_SOURCES.get(market.category.lower(), _MARKET_SOURCES["default"])
        return (
            f'{meta["hint"]} {market.title} latest data analysis '
            f'({meta["name"]})'
        )

    def _source_hint(self) -> str:
        return "domain-specific analysts and data providers for this market category"

    @property
    def base_credibility(self) -> float:
        # Override: use source-type credibility from category mapping
        # (overridden per-item after extraction)
        return CREDIBILITY["niche_analyst"]

    def run(self, market: Market, cfg: Config) -> list[ResearchItem]:
        meta = _MARKET_SOURCES.get(market.category.lower(), _MARKET_SOURCES["default"])
        cred = CREDIBILITY.get(meta["type"], CREDIBILITY["niche_analyst"])
        query = self._build_query(market)
        try:
            raw = _search(query, meta["name"], market, cfg)
            items = _items_from_raw(raw, market, cred, self.name, cfg)
            return items
        except Exception as exc:
            logger.warn(_MODULE, "market_specific_failed",
                        ticker=market.ticker, err=str(exc))
            return []


# ── RSS parsing ───────────────────────────────────────────────────────────────

def _fetch_rss(url: str, timeout: int = 10) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "black-gibbie/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_rss(feed_text: str, market: Market) -> list[ResearchItem]:
    """
    Parse RSS 2.0 XML into ResearchItems.
    Items are pre-filtered by title/description relevance to the market.
    """
    root = ET.fromstring(feed_text)
    items: list[ResearchItem] = []
    keywords = set(market.title.lower().split()) | {market.category.lower()}
    # Keep only words ≥4 chars to avoid stopword noise
    keywords = {w for w in keywords if len(w) >= 4}

    for entry in root.iter("item"):
        title_el  = entry.find("title")
        desc_el   = entry.find("description")
        link_el   = entry.find("link")
        pubdate_el = entry.find("pubDate")

        title_text = (title_el.text or "").strip() if title_el is not None else ""
        desc_text  = (desc_el.text  or "").strip() if desc_el is not None else ""
        link       = (link_el.text  or "").strip() if link_el is not None else ""
        pub_raw    = (pubdate_el.text or "").strip() if pubdate_el is not None else ""

        combined = (title_text + " " + desc_text).lower()
        if not any(kw in combined for kw in keywords):
            continue

        pub = _parse_date(pub_raw)
        # Simple relevance: fraction of keywords present
        rel = sum(1 for kw in keywords if kw in combined) / max(len(keywords), 1)
        rel = min(1.0, rel)

        items.append(ResearchItem(
            source=_extract_domain(link),
            url=link,
            published_at=pub,
            claim=title_text,
            direction="unclear",   # direction assigned by LLM extraction step later
            relevance=rel,
            credibility=0.0,       # set by caller
            recency_score=0.0,     # set by caller
            summary=desc_text[:300] if desc_text else title_text,
            agent="rss_news",
        ))

    return items


def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.netloc.replace("www.", "")
        return host or "rss_feed"
    except Exception:
        return "rss_feed"


# ── Orchestration ─────────────────────────────────────────────────────────────

_AGENTS: list[ResearchAgent] = [
    OfficialSourceAgent(),
    NewsWireAgent(),
    RSSNewsAgent(),
    TwitterXAgent(),
    RedditAgent(),
    MarketSpecificAgent(),
]


def _query_hash(ticker: str, title: str) -> str:
    return hashlib.sha256(f"{ticker}:{title}".encode()).hexdigest()[:16]


def research_market(market: Market, cfg: Config, force: bool = False) -> ResearchResult:
    """
    Run all 5 research agents for a single market.
    Checks the 60-minute cache first.
    Deduplicates results and returns a ResearchResult.
    """
    qhash = _query_hash(market.ticker, market.title)

    if not force:
        cached_items = db.get_cached_research(market.ticker, qhash, max_age_minutes=60)
        if cached_items is not None:
            logger.debug(_MODULE, "cache_hit", ticker=market.ticker,
                         items=len(cached_items))
            return ResearchResult(
                ticker=market.ticker,
                query=market.title,
                items=cached_items,
            )

    logger.info(_MODULE, "researching",
                ticker=market.ticker, title=market.title[:60])

    # ── Pass 1: category-aware research (preferred) ─────────────────────────
    # Try the specialized category-aware fetchers first. If they return any
    # usable items, we cache and return without running the generic agents.
    # Any exception is logged and treated as "no items" so the legacy path
    # always remains a safe fallback.
    cat_legacy: list[ResearchItem] = []
    try:
        cat_items = category_research.research_market_categorical(market, cfg)
        cat_legacy = [ci.to_legacy() for ci in cat_items]
    except Exception as exc:
        logger.warn(_MODULE, "categorical_failed",
                    ticker=market.ticker, err=str(exc))
        cat_legacy = []

    if cat_legacy:
        deduped = deduplicate(cat_legacy)
        deduped.sort(key=lambda x: -(x.relevance * x.credibility * x.recency_score))
        db.set_cached_research(market.ticker, qhash, deduped)
        logger.info(
            _MODULE, "research_done",
            ticker=market.ticker,
            path="categorical",
            raw_items=len(cat_legacy),
            after_dedup=len(deduped),
        )
        return ResearchResult(
            ticker=market.ticker,
            query=market.title,
            items=deduped,
        )

    # ── Pass 2: legacy generic agents (fallback) ────────────────────────────
    all_items: list[ResearchItem] = []
    for agent in _AGENTS:
        items = agent.run(market, cfg)
        all_items.extend(items)

    deduped = deduplicate(all_items)
    # Sort by composite score: relevance × credibility × recency
    deduped.sort(key=lambda x: -(x.relevance * x.credibility * x.recency_score))

    db.set_cached_research(market.ticker, qhash, deduped)

    logger.info(
        _MODULE, "research_done",
        ticker=market.ticker,
        path="legacy",
        raw_items=len(all_items),
        after_dedup=len(deduped),
        agents_ran=len(_AGENTS),
    )
    return ResearchResult(
        ticker=market.ticker,
        query=market.title,
        items=deduped,
    )


def batch_research(
    markets: list[Market],
    cfg: Config,
    force: bool = False,
) -> dict[str, ResearchResult]:
    """Research a list of markets. Returns ticker → ResearchResult."""
    results: dict[str, ResearchResult] = {}
    for i, market in enumerate(markets):
        logger.info(_MODULE, "progress",
                    f"researching {i+1}/{len(markets)}: {market.ticker}")
        try:
            results[market.ticker] = research_market(market, cfg, force=force)
        except Exception as exc:
            logger.error(_MODULE, "research_failed",
                         ticker=market.ticker, err=str(exc))
            results[market.ticker] = ResearchResult(
                ticker=market.ticker,
                query=market.title,
                failed_reason=str(exc),
            )
    return results
