from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logger
import market_scanner
from config import Config


class FakeScannerClient:
    def __init__(self, responses: dict[str | None, list[dict]]):
        self.responses = responses
        self.calls: list[tuple[str, str | None]] = []

    def get_all_markets(
        self,
        status: str = "open",
        category: str | None = None,
        max_markets: int | None = None,
    ) -> list[dict]:
        self.calls.append((status, category, max_markets))
        response = list(self.responses.get(category, []))
        return response[:max_markets] if max_markets is not None else response


def _cfg(tmp_path: Path, categories: list[str]) -> Config:
    logger.init(str(tmp_path / "logs"))
    return Config(
        category_allowlist=categories,
        log_dir=str(tmp_path / "logs"),
        db_path=str(tmp_path / "black_gibbie.sqlite"),
    )


def _raw_market(ticker: str = "KXTEST-26MAY15", category: str = "crypto") -> dict:
    close = datetime.now(timezone.utc) + timedelta(hours=2)
    return {
        "ticker": ticker,
        "title": "Will the test market resolve yes?",
        "status": "open",
        "yes_ask": 45,
        "yes_bid": 42,
        "no_ask": 57,
        "no_bid": 55,
        "volume": 5000,
        "volume_24h": 2500,
        "open_interest": 5000,
        "close_time": close.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "settlement_time": close.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "category": category,
        "rules_primary": "Test rules.",
    }


def test_scan_fetches_single_allowlisted_category(tmp_path):
    client = FakeScannerClient({"crypto": [_raw_market()]})

    markets = market_scanner.scan(client, _cfg(tmp_path, ["crypto"]))

    assert client.calls == [("open", "crypto", None)]
    assert [market.ticker for market in markets] == ["KXTEST-26MAY15"]


def test_scan_fetches_once_per_allowlisted_category(tmp_path):
    client = FakeScannerClient(
        {
            "crypto": [_raw_market("KXCRYPTO-26MAY15", "crypto")],
            "financial": [_raw_market("KXFIN-26MAY15", "financial")],
        }
    )

    markets = market_scanner.scan(client, _cfg(tmp_path, ["crypto", "financial"]))

    assert client.calls == [("open", "crypto", None), ("open", "financial", None)]
    assert {market.ticker for market in markets} == {
        "KXCRYPTO-26MAY15",
        "KXFIN-26MAY15",
    }


def test_scan_fetches_all_once_when_allowlist_empty(tmp_path):
    client = FakeScannerClient({None: [_raw_market()]})

    markets = market_scanner.scan(client, _cfg(tmp_path, []))

    assert client.calls == [("open", None, None)]
    assert [market.ticker for market in markets] == ["KXTEST-26MAY15"]


def test_scan_dedupes_duplicate_tickers_across_category_responses(tmp_path):
    duplicate = _raw_market("KXDUP-26MAY15", "crypto")
    client = FakeScannerClient(
        {
            "crypto": [duplicate],
            "financial": [dict(duplicate, category="financial")],
        }
    )

    markets = market_scanner.scan(client, _cfg(tmp_path, ["crypto", "financial"]))

    assert client.calls == [("open", "crypto", None), ("open", "financial", None)]
    assert [market.ticker for market in markets] == ["KXDUP-26MAY15"]


def test_scan_falls_back_to_all_markets_when_category_fetches_are_empty(tmp_path):
    client = FakeScannerClient(
        {
            "crypto": [],
            "financial": [],
            None: [_raw_market("KXFALLBACK-26MAY15", "crypto")],
        }
    )

    markets = market_scanner.scan(client, _cfg(tmp_path, ["crypto", "financial"]))

    assert client.calls == [
        ("open", "crypto", None),
        ("open", "financial", None),
        ("open", None, None),
    ]
    assert [market.ticker for market in markets] == ["KXFALLBACK-26MAY15"]


def test_scan_honors_raw_market_fetch_cap_across_categories(tmp_path):
    client = FakeScannerClient(
        {
            "crypto": [
                _raw_market("KXCRYPTO1-26MAY15", "crypto"),
                _raw_market("KXCRYPTO2-26MAY15", "crypto"),
            ],
            "financial": [_raw_market("KXFIN-26MAY15", "financial")],
        }
    )
    c = _cfg(tmp_path, ["crypto", "financial"])
    c.max_raw_markets_per_scan = 2

    markets = market_scanner.scan(client, c)

    assert client.calls == [("open", "crypto", 2)]
    assert [market.ticker for market in markets] == [
        "KXCRYPTO1-26MAY15",
        "KXCRYPTO2-26MAY15",
    ]


def test_scan_prefilters_blocked_sports_prefix_before_normalize(monkeypatch, tmp_path):
    sports = dict(
        _raw_market("KXATP-26MAY15-DJOKER", "sports"),
        event_ticker="KXATP-26MAY15",
        title="Will a tennis player win this ATP match?",
    )
    allowed = _raw_market("KXWEATHER-26MAY15-RAIN", "weather")
    client = FakeScannerClient({"weather": [sports, allowed]})
    c = _cfg(tmp_path, ["weather"])
    c.blocked_event_prefixes = ["KXATP", "KXWNBA"]
    normalized: list[str] = []
    real_normalize = market_scanner.normalize

    def spy_normalize(raw):
        normalized.append(raw["ticker"])
        return real_normalize(raw)

    monkeypatch.setattr(market_scanner, "normalize", spy_normalize)

    markets = market_scanner.scan(client, c)

    assert [market.ticker for market in markets] == ["KXWEATHER-26MAY15-RAIN"]
    assert normalized == ["KXWEATHER-26MAY15-RAIN"]


def test_prefilter_allows_default_tradable_categories(tmp_path):
    c = _cfg(
        tmp_path,
        [
            "crypto",
            "economic",
            "financial",
            "commodities",
            "weather",
            "science and technology",
            "culture",
        ],
    )
    c.blocked_event_prefixes = ["KXATP", "KXWNBA"]

    for category in (
        "crypto",
        "economic",
        "financial",
        "commodities",
        "weather",
        "science and technology",
        "culture",
    ):
        raw = _raw_market(f"KX{category.upper()}-26MAY15", category)
        assert market_scanner.prefilter_raw_market(raw, c) is None


def test_prefilter_allows_requested_user_facing_category_names(tmp_path):
    c = _cfg(
        tmp_path,
        ["crypto", "finance", "economics", "commodities", "climate", "tech & science", "culture"],
    )
    c.blocked_event_prefixes = []

    cases = [
        ("crypto", "Crypto"),
        ("finance", "Financials"),
        ("economics", "Economics"),
        ("commodities", "Commodities"),
        ("climate", "Climate and Weather"),
        ("tech", "Science and Technology"),
        ("culture", "Culture"),
    ]
    for ticker_part, raw_category in cases:
        raw = _raw_market(f"KX{ticker_part.upper()}-26MAY15", raw_category)
        assert market_scanner.prefilter_raw_market(raw, c) is None


def test_prefilter_prefix_does_not_block_unrelated_ticker(tmp_path):
    c = _cfg(tmp_path, ["science and technology"])
    c.blocked_event_prefixes = ["KXATP"]
    raw = _raw_market("KXATOMIC-26MAY15", "science and technology")

    assert market_scanner.prefilter_raw_market(raw, c) is None
