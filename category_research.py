"""
Category-aware research.

Given a Kalshi Market, route to the appropriate set of data sources:

  crypto       → exchange spot prices, ETF flow / Fed / liquidation headlines
  commodities  → EIA, OPEC headlines, futures, geopolitical headlines
  economic     → BLS, BEA, FRED, Census, release calendars
  weather      → NWS, NOAA forecasts

Each fetcher returns CategoryResearchItem objects with the spec output shape:
  {
    "query": ...,
    "source_type": "official|news|social|market_data",
    "source_name": ...,
    "claim": ...,
    "supports": "yes|no|neutral|unclear",
    "credibility": 0.0-1.0,
    "relevance": 0.0-1.0,
    "recency": 0.0-1.0,
    "risk_flags": [...]
  }

Fetchers fail soft: any individual source failure logs and returns []. The
caller always gets a (possibly empty) list back.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Optional

import requests

import llm
import logger
from config import Config
from models import CategoryResearchItem, Market

_MODULE = "category_research"

# ── Credibility per (source_type, source_name) ────────────────────────────────
# These match the existing CREDIBILITY table in research_agents.py but
# are expressed as source-name keys so each fetcher can be explicit.

_CRED_OFFICIAL    = 0.95   # gov / central bank / exchange / EIA / BLS / FRED / NOAA / NWS
_CRED_MARKET_DATA = 0.90   # exchange price feed
_CRED_MAJOR_WIRE  = 0.85   # Reuters / AP / Bloomberg
_CRED_ANALYST     = 0.70   # respected niche analyst
_CRED_SOCIAL      = 0.45   # verified social + caveat
_CRED_UNKNOWN     = 0.50

_HTTP_TIMEOUT = 10


# ── Category detection ────────────────────────────────────────────────────────

_CRYPTO_KEYWORDS = ("bitcoin", "btc", "ethereum", "eth", "solana", "sol",
                    "dogecoin", "doge", "crypto", "cryptocurrency")
_COMMODITY_KEYWORDS = ("oil", "wti", "brent", "crude", "natural gas", "gasoline",
                       "gold", "silver", "copper", "opec", "barrel")
_ECON_KEYWORDS = ("cpi", "ppi", "inflation", "unemployment", "nonfarm", "payroll",
                  "gdp", "fed funds", "interest rate", "fomc", "jobs report",
                  "consumer price", "retail sales", "jobless claims")
_WEATHER_KEYWORDS = ("temperature", "snowfall", "snow", "rain", "hurricane",
                     "weather", "forecast", "high temp", "low temp", "degrees",
                     "precipitation")


def detect_category(market: Market) -> str:
    """
    Map a market to one of: crypto | commodities | economic | weather | other.

    Uses the Kalshi-provided category as a strong prior, then falls back to
    keyword matching on the title for cases where Kalshi's labels are coarse
    (e.g. 'financial' covers both crypto and commodities).
    """
    cat = (market.category or "").lower()
    title = (market.title or "").lower()

    if cat == "crypto" or any(k in title for k in _CRYPTO_KEYWORDS):
        return "crypto"
    if any(k in title for k in _COMMODITY_KEYWORDS):
        return "commodities"
    if cat in ("economic", "economics") or any(k in title for k in _ECON_KEYWORDS):
        return "economic"
    if cat in ("weather", "climate") or any(k in title for k in _WEATHER_KEYWORDS):
        return "weather"
    return "other"


# ── Recency scoring ───────────────────────────────────────────────────────────

def recency_score(published_at: Optional[datetime]) -> float:
    if published_at is None:
        return 0.50
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600
    if age_h < 0.25:  return 1.00
    if age_h < 1:     return 0.95
    if age_h < 6:     return 0.90
    if age_h < 24:    return 0.80
    if age_h < 72:    return 0.65
    if age_h < 168:   return 0.45
    return 0.20


# ── Title parsers (deterministic supports/relevance for price-style markets) ──

_CRYPTO_SYMBOLS = {
    "btc": "BTC", "bitcoin": "BTC",
    "eth": "ETH", "ethereum": "ETH",
    "sol": "SOL", "solana": "SOL",
    "doge": "DOGE", "dogecoin": "DOGE",
}


def _extract_crypto_symbol(title: str) -> Optional[str]:
    t = title.lower()
    for k, v in _CRYPTO_SYMBOLS.items():
        if re.search(rf"\b{k}\b", t):
            return v
    return None


_THRESHOLD_RE = re.compile(
    r"(above|over|under|below|at least|greater than|less than|>=|<=|>|<)\s*"
    r"\$?\s*([0-9][0-9,\.]*)\s*(k|thousand|m|million)?",
    re.IGNORECASE,
)


def _extract_threshold(title: str) -> Optional[tuple[str, float]]:
    """
    Returns (direction, value) where direction ∈ {"above", "below"}.
    e.g. "BTC above $104,000" → ("above", 104000.0)
    """
    m = _THRESHOLD_RE.search(title)
    if not m:
        return None
    word = m.group(1).lower()
    num = m.group(2).replace(",", "")
    suffix = (m.group(3) or "").lower()
    try:
        value = float(num)
    except ValueError:
        return None
    if suffix in ("k", "thousand"):
        value *= 1_000
    elif suffix in ("m", "million"):
        value *= 1_000_000

    if word in ("above", "over", "at least", "greater than", ">=", ">"):
        return ("above", value)
    if word in ("under", "below", "less than", "<=", "<"):
        return ("below", value)
    return None


def _supports_from_threshold(
    current_value: float,
    direction: str,
    threshold: float,
) -> str:
    """Map (current, threshold, direction) to supports = yes|no|neutral|unclear."""
    if direction == "above":
        if current_value >= threshold:           return "yes"
        if current_value < threshold * 0.99:     return "no"
        return "neutral"   # within 1% — too close to call
    if direction == "below":
        if current_value <= threshold:           return "yes"
        if current_value > threshold * 1.01:     return "no"
        return "neutral"
    return "unclear"


# ── Risk-flag helpers ─────────────────────────────────────────────────────────

def _crypto_risk_flags(
    market: Market,
    current: float,
    threshold: Optional[float],
) -> list[str]:
    flags: list[str] = []
    # Close to threshold
    if threshold is not None and threshold > 0:
        pct_away = abs(current - threshold) / threshold
        if pct_away < 0.005:
            flags.append("price within 0.5% of strike")
        elif pct_away < 0.02:
            flags.append("price within 2% of strike")
    # Time pressure
    if 0 < market.minutes_to_close < 30:
        flags.append("<30 min to close")
    elif market.minutes_to_close > 0 and market.minutes_to_close < 120:
        flags.append("<2 hours to close")
    # Thin book
    if 0 < market.orderbook_depth < 200:
        flags.append("thin Kalshi book")
    if market.spread_pct > 8:
        flags.append("wide Kalshi spread")
    return flags


def _market_data_risk_flags(market: Market) -> list[str]:
    flags: list[str] = []
    if 0 < market.orderbook_depth < 200:
        flags.append("thin Kalshi book")
    if market.spread_pct > 8:
        flags.append("wide Kalshi spread")
    return flags


# ── Crypto: exchange spot price (Coinbase, no auth) ──────────────────────────

_COINBASE_PRODUCT_MAP = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "DOGE": "DOGE-USD",
}


def _fetch_coinbase_spot(symbol: str) -> Optional[tuple[float, datetime]]:
    """Returns (price_usd, fetched_at_utc) or None on failure."""
    product = _COINBASE_PRODUCT_MAP.get(symbol)
    if not product:
        return None
    try:
        resp = requests.get(
            f"https://api.coinbase.com/v2/prices/{product}/spot",
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        amount = float(resp.json()["data"]["amount"])
        return (amount, datetime.now(timezone.utc))
    except Exception as exc:
        logger.debug(_MODULE, "coinbase_fetch_failed", symbol=symbol, err=str(exc))
        return None


def _research_crypto_price(market: Market) -> list[CategoryResearchItem]:
    symbol = _extract_crypto_symbol(market.title)
    if not symbol:
        return []

    spot = _fetch_coinbase_spot(symbol)
    if spot is None:
        return []
    price, fetched_at = spot

    threshold_info = _extract_threshold(market.title)
    if threshold_info:
        direction, threshold = threshold_info
        supports = _supports_from_threshold(price, direction, threshold)
        relevance = 0.95   # direct quote of the underlying
    else:
        direction, threshold = "unclear", None
        supports = "unclear"
        relevance = 0.70

    mins = market.minutes_to_close
    timing = f"{mins:.0f} minutes to close" if 0 < mins < 240 else f"{mins/60:.1f} hours to close"
    claim = f"{symbol} is trading at ${price:,.2f} on Coinbase with {timing}."

    flags = _crypto_risk_flags(market, price, threshold)

    return [CategoryResearchItem(
        query=market.title,
        source_type="market_data",
        source_name="Coinbase",
        claim=claim,
        supports=supports,
        credibility=_CRED_MARKET_DATA,
        relevance=relevance,
        recency=recency_score(fetched_at),
        risk_flags=flags,
        url=f"https://www.coinbase.com/price/{symbol.lower()}",
        published_at=fetched_at,
        summary=f"Spot price from Coinbase exchange. Threshold parsed: "
                f"{direction} {threshold}." if threshold else
                "Spot price from Coinbase exchange. No numeric threshold parsed from title.",
    )]


# ── Commodities: EIA weekly petroleum status ─────────────────────────────────

def _fetch_eia_wti_spot(cfg: Config) -> Optional[tuple[float, datetime]]:
    """Pull the latest WTI spot price from EIA."""
    if not cfg.eia_api_key:
        return None
    try:
        # PET.RWTC.D = Cushing OK WTI spot, daily
        resp = requests.get(
            "https://api.eia.gov/v2/seriesid/PET.RWTC.D",
            params={"api_key": cfg.eia_api_key, "length": 1},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json().get("response", {}).get("data") or []
        if not rows:
            return None
        latest = rows[0]
        period = latest.get("period")   # "YYYY-MM-DD"
        value = latest.get("value")
        if value is None or period is None:
            return None
        published = datetime.strptime(period, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (float(value), published)
    except Exception as exc:
        logger.debug(_MODULE, "eia_wti_failed", err=str(exc))
        return None


def _research_commodities(market: Market, cfg: Config) -> list[CategoryResearchItem]:
    items: list[CategoryResearchItem] = []
    title_lower = market.title.lower()

    # WTI / crude markets get an EIA price quote
    if any(k in title_lower for k in ("oil", "wti", "crude", "brent")):
        spot = _fetch_eia_wti_spot(cfg)
        if spot is not None:
            price, published = spot
            threshold_info = _extract_threshold(market.title)
            if threshold_info:
                direction, threshold = threshold_info
                supports = _supports_from_threshold(price, direction, threshold)
                relevance = 0.90
            else:
                supports = "unclear"
                relevance = 0.60

            items.append(CategoryResearchItem(
                query=market.title,
                source_type="official",
                source_name="EIA (US Energy Information Administration)",
                claim=f"WTI crude spot price was ${price:.2f}/barrel "
                      f"as of {published.strftime('%Y-%m-%d')} (EIA).",
                supports=supports,
                credibility=_CRED_OFFICIAL,
                relevance=relevance,
                recency=recency_score(published),
                risk_flags=_market_data_risk_flags(market),
                url="https://www.eia.gov/petroleum/",
                published_at=published,
                summary="EIA Cushing, OK WTI spot price (PET.RWTC.D).",
            ))

    # News-based items (OPEC headlines, geopolitical, futures) via Perplexity
    items.extend(_research_news(
        market,
        cfg,
        query_hint="oil OPEC crude futures geopolitical headlines",
        source_name="Reuters/Bloomberg/AP energy desk",
    ))
    return items


# ── Economic: FRED + BLS + release calendars ─────────────────────────────────

# Title-keyword → (FRED series_id, human label, recency-friendly description)
_FRED_LOOKUPS = [
    (("fed funds", "fomc", "rate cut", "rate hike", "interest rate"),
     "DFF",     "Effective Federal Funds Rate"),
    (("unemployment rate",),
     "UNRATE",  "Unemployment Rate (FRED)"),
    (("cpi", "consumer price", "inflation"),
     "CPIAUCSL", "Consumer Price Index (FRED)"),
    (("10-year", "10y treasury", "10 year treasury"),
     "DGS10",   "10-Year Treasury Constant Maturity"),
    (("gdp",),
     "GDP",     "Gross Domestic Product (FRED)"),
]


def _fetch_fred_latest(series_id: str, cfg: Config) -> Optional[tuple[float, datetime]]:
    if not cfg.fred_api_key:
        return None
    try:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id":   series_id,
                "api_key":     cfg.fred_api_key,
                "file_type":   "json",
                "sort_order":  "desc",
                "limit":       1,
            },
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        obs = resp.json().get("observations") or []
        if not obs:
            return None
        latest = obs[0]
        value_raw = latest.get("value")
        date_raw = latest.get("date")
        if value_raw in (None, ".", "") or not date_raw:
            return None
        published = datetime.strptime(date_raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (float(value_raw), published)
    except Exception as exc:
        logger.debug(_MODULE, "fred_failed", series=series_id, err=str(exc))
        return None


def _fetch_bls_latest(series_id: str, cfg: Config) -> Optional[tuple[float, datetime]]:
    """
    BLS public time-series API. Works without a key (lower daily quota).
    """
    try:
        body: dict = {"seriesid": [series_id], "latest": True}
        if cfg.bls_api_key:
            body["registrationkey"] = cfg.bls_api_key
        resp = requests.post(
            "https://api.bls.gov/publicAPI/v2/timeseries/data/",
            json=body,
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("Results", {}).get("series") or []
        if not results:
            return None
        data = results[0].get("data") or []
        if not data:
            return None
        latest = data[0]
        value = float(latest["value"])
        year = int(latest["year"])
        period = latest.get("period", "M01")
        month = int(period[1:]) if period.startswith("M") else 1
        published = datetime(year, month, 1, tzinfo=timezone.utc)
        return (value, published)
    except Exception as exc:
        logger.debug(_MODULE, "bls_failed", series=series_id, err=str(exc))
        return None


def _research_economic(market: Market, cfg: Config) -> list[CategoryResearchItem]:
    items: list[CategoryResearchItem] = []
    title_lower = market.title.lower()

    for keywords, series_id, label in _FRED_LOOKUPS:
        if any(k in title_lower for k in keywords):
            data = _fetch_fred_latest(series_id, cfg)
            if data is None:
                continue
            value, published = data

            threshold_info = _extract_threshold(market.title)
            if threshold_info:
                direction, threshold = threshold_info
                supports = _supports_from_threshold(value, direction, threshold)
                relevance = 0.85
            else:
                supports = "unclear"
                relevance = 0.65

            items.append(CategoryResearchItem(
                query=market.title,
                source_type="official",
                source_name="FRED (St. Louis Fed)",
                claim=f"{label} latest observation: {value} "
                      f"(as of {published.strftime('%Y-%m-%d')}).",
                supports=supports,
                credibility=_CRED_OFFICIAL,
                relevance=relevance,
                recency=recency_score(published),
                risk_flags=_econ_release_flags(market, published),
                url=f"https://fred.stlouisfed.org/series/{series_id}",
                published_at=published,
                summary=f"FRED series {series_id} ({label}).",
            ))

    # BLS for unemployment / payrolls if not already covered by FRED
    if "nonfarm" in title_lower or "payroll" in title_lower:
        data = _fetch_bls_latest("CES0000000001", cfg)
        if data is not None:
            value, published = data
            items.append(CategoryResearchItem(
                query=market.title,
                source_type="official",
                source_name="BLS",
                claim=f"Total nonfarm payrolls latest reading: {value:,.0f} thousand "
                      f"(reference month {published.strftime('%Y-%m')}).",
                supports="unclear",   # change vs prior isn't computed here
                credibility=_CRED_OFFICIAL,
                relevance=0.80,
                recency=recency_score(published),
                risk_flags=_econ_release_flags(market, published),
                url="https://www.bls.gov/ces/",
                published_at=published,
                summary="BLS Current Employment Statistics — total nonfarm employment.",
            ))

    items.extend(_research_news(
        market,
        cfg,
        query_hint="Federal Reserve FOMC BLS BEA Census economic data release",
        source_name="Reuters/Bloomberg/AP economics desk",
    ))
    return items


def _econ_release_flags(market: Market, published_at: datetime) -> list[str]:
    flags: list[str] = []
    age_days = (datetime.now(timezone.utc) - published_at).days
    if age_days > 35:
        flags.append("data older than 1 month")
    if 0 < market.minutes_to_close < 60:
        flags.append("<1 hour to close — release timing risk")
    if 0 < market.orderbook_depth < 200:
        flags.append("thin Kalshi book")
    return flags


# ── Weather: NWS forecast (no key, needs UA) ─────────────────────────────────

# Small static map of common Kalshi weather-market city codes → lat/lon.
# Kalshi uses airport-style codes in HIGHNY/HIGHLA/HIGHCH series.
_CITY_COORDS = {
    "NY":  (40.7128, -74.0060),  "NYC": (40.7128, -74.0060),
    "LA":  (34.0522, -118.2437),
    "CH":  (41.8781, -87.6298),  "CHI": (41.8781, -87.6298),
    "MI":  (25.7617, -80.1918),  "MIA": (25.7617, -80.1918),
    "AU":  (30.2672, -97.7431),  "AUS": (30.2672, -97.7431),
    "DC":  (38.9072, -77.0369),
    "PHX": (33.4484, -112.0740),
    "DEN": (39.7392, -104.9903),
    "BOS": (42.3601, -71.0589),
    "SEA": (47.6062, -122.3321),
}

_CITY_NAME_LOOKUPS = (
    ("new york",     "NYC"),
    ("los angeles",  "LA"),
    ("chicago",      "CHI"),
    ("miami",        "MIA"),
    ("austin",       "AUS"),
    ("washington",   "DC"),
    ("phoenix",      "PHX"),
    ("denver",       "DEN"),
    ("boston",       "BOS"),
    ("seattle",      "SEA"),
)


def _detect_city(market: Market) -> Optional[tuple[str, tuple[float, float]]]:
    t = market.title.lower()
    for name, code in _CITY_NAME_LOOKUPS:
        if name in t:
            return (code, _CITY_COORDS[code])
    # Try ticker pattern: HIGHNY25NOV13 → NY
    m = re.match(r"^(HIGH|LOW|TEMP|RAIN|SNOW)([A-Z]{2,3})", market.ticker.upper())
    if m:
        code = m.group(2)
        if code in _CITY_COORDS:
            return (code, _CITY_COORDS[code])
    return None


def _fetch_nws_forecast(
    lat: float,
    lon: float,
    cfg: Config,
) -> Optional[dict]:
    """Returns the next forecast period (a dict) or None."""
    headers = {
        "User-Agent": cfg.nws_user_agent or "black-gibbie (contact: unset)",
        "Accept":     "application/geo+json",
    }
    try:
        points = requests.get(
            f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}",
            headers=headers,
            timeout=_HTTP_TIMEOUT,
        )
        points.raise_for_status()
        forecast_url = points.json()["properties"]["forecast"]

        fc = requests.get(forecast_url, headers=headers, timeout=_HTTP_TIMEOUT)
        fc.raise_for_status()
        periods = fc.json()["properties"]["periods"]
        return periods[0] if periods else None
    except Exception as exc:
        logger.debug(_MODULE, "nws_failed", lat=lat, lon=lon, err=str(exc))
        return None


def _research_weather(market: Market, cfg: Config) -> list[CategoryResearchItem]:
    city = _detect_city(market)
    if city is None:
        return _research_news(
            market, cfg,
            query_hint="NWS NOAA weather forecast temperature",
            source_name="NWS/NOAA via news desks",
        )

    code, (lat, lon) = city
    period = _fetch_nws_forecast(lat, lon, cfg)
    if period is None:
        return []

    temp = period.get("temperature")
    temp_unit = period.get("temperatureUnit", "F")
    label = period.get("name", "next period")
    short = period.get("shortForecast", "")
    start = period.get("startTime")
    try:
        published = datetime.fromisoformat(start) if start else datetime.now(timezone.utc)
    except ValueError:
        published = datetime.now(timezone.utc)

    threshold_info = _extract_threshold(market.title)
    if threshold_info and isinstance(temp, (int, float)):
        direction, threshold = threshold_info
        supports = _supports_from_threshold(float(temp), direction, threshold)
        relevance = 0.85
    else:
        supports = "unclear"
        relevance = 0.65

    flags: list[str] = []
    if market.minutes_to_close > 60 * 24 * 3:
        flags.append("forecast >3 days out — high uncertainty")
    if 0 < market.orderbook_depth < 200:
        flags.append("thin Kalshi book")

    return [CategoryResearchItem(
        query=market.title,
        source_type="official",
        source_name=f"NWS ({code})",
        claim=f"NWS {label} forecast for {code}: {temp}°{temp_unit}, {short}.",
        supports=supports,
        credibility=_CRED_OFFICIAL,
        relevance=relevance,
        recency=recency_score(published),
        risk_flags=flags,
        url="https://api.weather.gov/",
        published_at=published,
        summary=f"National Weather Service forecast period '{label}' for {code} "
                f"({lat:.2f},{lon:.2f}).",
    )]


# ── News fetching via Perplexity (shared by all categories) ──────────────────

def _research_news(
    market: Market,
    cfg: Config,
    *,
    query_hint: str,
    source_name: str,
) -> list[CategoryResearchItem]:
    """
    Hit Perplexity for category-relevant news, then ask the LLM to extract
    structured claims. Returns [] if Perplexity is not configured or fails.
    """
    if not cfg.perplexity_api_key:
        return []

    query = f"{market.title} — {query_hint} (last 24 hours, factual reporting only)"
    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {cfg.perplexity_api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model": cfg.perplexity_model,
                "messages": [{"role": "user", "content": query}],
                "search_recency_filter": "day",
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw_text = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.debug(_MODULE, "perplexity_news_failed",
                     ticker=market.ticker, err=str(exc))
        return []

    raw_items = llm.extract_research_items(
        cfg,
        ticker=market.ticker,
        title=market.title,
        rules=market.rules_primary,
        raw_text=raw_text,
    )

    direction_map = {
        "supports_yes": "yes",
        "supports_no":  "no",
        "neutral":      "neutral",
        "unclear":      "unclear",
    }

    items: list[CategoryResearchItem] = []
    for d in raw_items:
        if not d.get("claim"):
            continue
        pub = _try_parse_iso(d.get("published_at"))
        publisher = d.get("source", source_name)
        items.append(CategoryResearchItem(
            query=market.title,
            source_type="news",
            source_name=publisher,
            claim=d["claim"],
            supports=direction_map.get(d.get("direction"), "unclear"),
            credibility=_credibility_for_publisher(publisher),
            relevance=float(d.get("relevance", 0.5)),
            recency=recency_score(pub),
            risk_flags=[],
            url=d.get("url", ""),
            published_at=pub,
            summary=d.get("summary", ""),
        ))
    return items


_WIRE_HOSTS = ("reuters", "apnews", "bloomberg", "wsj", "ft.com")
_ANALYST_HOSTS = ("coindesk", "cointelegraph", "glassnode", "kitco",
                  "marketwatch", "cnbc")


def _credibility_for_publisher(name: str) -> float:
    n = name.lower()
    if any(h in n for h in _WIRE_HOSTS):
        return _CRED_MAJOR_WIRE
    if any(h in n for h in _ANALYST_HOSTS):
        return _CRED_ANALYST
    if "reddit" in n or "twitter" in n or "x.com" in n:
        return _CRED_SOCIAL
    return _CRED_UNKNOWN


def _try_parse_iso(raw) -> Optional[datetime]:
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(str(raw), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ── Crypto: combine spot + news ───────────────────────────────────────────────

def _research_crypto(market: Market, cfg: Config) -> list[CategoryResearchItem]:
    items: list[CategoryResearchItem] = []
    items.extend(_research_crypto_price(market))
    items.extend(_research_news(
        market, cfg,
        query_hint="ETF flows Fed rates liquidation volatility crypto macro",
        source_name="Reuters/Bloomberg/CoinDesk",
    ))
    return items


# ── Top-level entry point ─────────────────────────────────────────────────────

def research_market_categorical(
    market: Market,
    cfg: Config,
) -> list[CategoryResearchItem]:
    """
    Route a market through the category-aware research pipeline.
    Returns spec-shaped CategoryResearchItems. Always returns a list
    (possibly empty); failures of individual sources are logged.
    """
    category = detect_category(market)
    logger.debug(_MODULE, "routing", ticker=market.ticker,
                 category=category, title=market.title[:60])

    if category == "crypto":
        items = _research_crypto(market, cfg)
    elif category == "commodities":
        items = _research_commodities(market, cfg)
    elif category == "economic":
        items = _research_economic(market, cfg)
    elif category == "weather":
        items = _research_weather(market, cfg)
    else:
        items = _research_news(
            market, cfg,
            query_hint="latest news developments",
            source_name="general news",
        )

    # Sort by composite score
    items.sort(key=lambda x: -(x.relevance * x.credibility * x.recency))

    logger.info(_MODULE, "research_done",
                ticker=market.ticker, category=category,
                items=len(items))
    return items
