from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_client import KalshiClient


def _private_key_file(tmp_path: Path) -> tuple[Path, rsa.RSAPrivateKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    path = tmp_path / "kalshi_test_private.pem"
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return path, key


def test_signing_path_includes_trade_api_base_path_without_query(tmp_path):
    key_path, _ = _private_key_file(tmp_path)
    client = KalshiClient(
        api_key="test-api-key",
        private_key_path=str(key_path),
        base_url="https://trading-api.kalshi.com/trade-api/v2",
    )

    assert client._base_path == "/trade-api/v2"
    assert client._signing_path("/markets") == "/trade-api/v2/markets"
    assert (
        client._signing_path("/markets?status=open&limit=200")
        == "/trade-api/v2/markets"
    )
    assert (
        client._signing_path("/trade-api/v2/markets")
        == "/trade-api/v2/markets"
    )


def test_signature_uses_rsa_pss_over_signing_path(tmp_path):
    key_path, key = _private_key_file(tmp_path)
    client = KalshiClient(
        api_key="test-api-key",
        private_key_path=str(key_path),
        base_url="https://trading-api.kalshi.com/trade-api/v2",
    )
    signing_path = client._signing_path("/markets?status=open")
    signature = base64.b64decode(client._sign("123", "GET", signing_path))
    message = b"123GET/trade-api/v2/markets"

    key.public_key().verify(
        signature,
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_headers_sign_normalized_path_without_query(tmp_path, monkeypatch):
    key_path, _ = _private_key_file(tmp_path)
    client = KalshiClient(
        api_key="test-api-key",
        private_key_path=str(key_path),
        base_url="https://trading-api.kalshi.com/trade-api/v2",
    )
    signed = {}

    def fake_sign(timestamp: str, method: str, path: str) -> str:
        signed["timestamp"] = timestamp
        signed["method"] = method
        signed["path"] = path
        return "signature"

    monkeypatch.setattr(client, "_sign", fake_sign)

    headers = client._headers("GET", "/markets?status=open&limit=200")

    assert headers["KALSHI-ACCESS-KEY"] == "test-api-key"
    assert headers["KALSHI-ACCESS-SIGNATURE"] == "signature"
    assert signed["method"] == "GET"
    assert signed["path"] == "/trade-api/v2/markets"


def test_get_markets_passes_category_when_present(monkeypatch):
    client = object.__new__(KalshiClient)
    calls = []

    def fake_get(path, params=None):
        calls.append((path, params))
        return {"markets": []}

    monkeypatch.setattr(client, "_get", fake_get)

    client.get_markets(status="open", limit=200, category="crypto")

    assert calls == [
        (
            "/markets",
            {"status": "open", "limit": 200, "category": "crypto"},
        )
    ]


def test_get_markets_omits_category_when_none(monkeypatch):
    client = object.__new__(KalshiClient)
    calls = []

    def fake_get(path, params=None):
        calls.append((path, params))
        return {"markets": []}

    monkeypatch.setattr(client, "_get", fake_get)

    client.get_markets(status="open", limit=200, category=None)

    assert calls == [("/markets", {"status": "open", "limit": 200})]


def test_get_all_markets_passes_category_through_pagination(monkeypatch):
    client = object.__new__(KalshiClient)
    calls = []
    responses = [
        {"markets": [{"ticker": "A"}], "cursor": "next-page"},
        {"markets": [{"ticker": "B"}], "cursor": ""},
    ]

    def fake_get_markets(status="open", limit=200, cursor=None, category=None):
        calls.append(
            {
                "status": status,
                "limit": limit,
                "cursor": cursor,
                "category": category,
            }
        )
        return responses.pop(0)

    monkeypatch.setattr(client, "get_markets", fake_get_markets)

    markets = client.get_all_markets(status="open", category="crypto")

    assert markets == [{"ticker": "A"}, {"ticker": "B"}]
    assert calls == [
        {"status": "open", "limit": 200, "cursor": None, "category": "crypto"},
        {"status": "open", "limit": 200, "cursor": "next-page", "category": "crypto"},
    ]


def test_get_all_markets_honors_max_markets(monkeypatch):
    client = object.__new__(KalshiClient)
    calls = []
    responses = [
        {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": "next-page"},
        {"markets": [{"ticker": "C"}], "cursor": ""},
    ]

    def fake_get_markets(status="open", limit=200, cursor=None, category=None):
        calls.append(
            {
                "status": status,
                "limit": limit,
                "cursor": cursor,
                "category": category,
            }
        )
        return responses.pop(0)

    monkeypatch.setattr(client, "get_markets", fake_get_markets)

    markets = client.get_all_markets(
        status="open",
        category="crypto",
        max_markets=2,
    )

    assert markets == [{"ticker": "A"}, {"ticker": "B"}]
    assert calls == [
        {"status": "open", "limit": 2, "cursor": None, "category": "crypto"},
    ]


def test_get_orderbooks_passes_tickers_to_bulk_endpoint(monkeypatch):
    client = object.__new__(KalshiClient)
    calls = []

    def fake_get(path, params=None):
        calls.append((path, params))
        return {"orderbooks": []}

    monkeypatch.setattr(client, "_get", fake_get)

    assert client.get_orderbooks(["KXA", "KXB"]) == {"orderbooks": []}
    assert calls == [
        ("/markets/orderbooks", {"tickers": ["KXA", "KXB"]}),
    ]


def test_get_best_prices_parses_current_orderbook_fp_shape(monkeypatch):
    client = object.__new__(KalshiClient)

    monkeypatch.setattr(
        client,
        "get_orderbook",
        lambda ticker, depth=1: {
            "orderbook_fp": {
                "yes_dollars": [["0.0100", "200.00"], ["0.4200", "13.00"]],
                "no_dollars": [["0.0100", "100.00"], ["0.5600", "117.00"]],
            }
        },
    )

    assert client.get_best_prices("KXTEST") == (44, 42)


def test_get_best_prices_parses_legacy_bid_only_shape(monkeypatch):
    client = object.__new__(KalshiClient)

    monkeypatch.setattr(
        client,
        "get_orderbook",
        lambda ticker, depth=1: {
            "orderbook": {
                "yes": [[1, 200], [42, 13]],
                "no": [[1, 100], [56, 117]],
            }
        },
    )

    assert client.get_best_prices("KXTEST") == (44, 42)
