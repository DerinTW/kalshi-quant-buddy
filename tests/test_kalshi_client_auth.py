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
