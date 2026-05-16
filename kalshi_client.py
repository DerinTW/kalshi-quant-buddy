from __future__ import annotations
import base64
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import logger

_MODULE = "kalshi_client"


class KalshiClient:
    def __init__(self, api_key: str, private_key_path: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._base_path = urlparse(self.base_url).path.rstrip("/")
        with open(private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str) -> str:
        message = (timestamp + method.upper() + path).encode("utf-8")
        sig = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("utf-8")

    def _signing_path(self, path: str) -> str:
        clean_path = path.split("?")[0]
        if self._base_path and not clean_path.startswith(self._base_path):
            return self._base_path + clean_path
        return clean_path

    def _headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(ts, method, self._signing_path(path)),
        }

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        headers = self._headers("GET", path)
        resp = requests.get(self.base_url + path, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> Any:
        import json as _json
        headers = self._headers("POST", path)
        resp = requests.post(self.base_url + path, headers=headers, json=body, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> Any:
        headers = self._headers("DELETE", path)
        resp = requests.delete(self.base_url + path, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Markets ───────────────────────────────────────────────────────────────

    def get_markets(
        self,
        status: str = "open",
        limit: int = 200,
        cursor: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        mve_filter: Optional[str] = None,
        tickers: Optional[str] = None,
        category: Optional[str] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if mve_filter:
            params["mve_filter"] = mve_filter
        if tickers:
            params["tickers"] = tickers
        return self._get("/markets", params=params)

    def get_series_list(self) -> list[dict[str, Any]]:
        data = self._get("/series")
        series = data.get("series", [])
        return series if isinstance(series, list) else []

    def get_series(self, series_ticker: str) -> dict[str, Any]:
        data = self._get(f"/series/{series_ticker}")
        series = data.get("series", data)
        return series if isinstance(series, dict) else {}

    def get_all_markets(self, status: str = "open") -> list[dict[str, Any]]:
        """Paginate through all markets, returning the raw list."""
        markets: list[dict[str, Any]] = []
        cursor: Optional[str] = None
        page = 0
        while True:
            page += 1
            logger.debug(_MODULE, "paginating", f"page {page}", cursor=cursor)
            data = self.get_markets(status=status, limit=200, cursor=cursor)
            batch = data.get("markets", [])
            markets.extend(batch)
            cursor = data.get("cursor")
            if not cursor or not batch:
                break
        logger.info(_MODULE, "markets_fetched", f"fetched {len(markets)} markets")
        return markets

    def get_market(self, ticker: str) -> dict[str, Any]:
        return self._get(f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        return self._get(f"/markets/{ticker}/orderbook", params={"depth": depth})

    def get_market_history(self, ticker: str, limit: int = 100) -> dict[str, Any]:
        """Recent trade history for a market (used for anomaly detection)."""
        return self._get(f"/markets/{ticker}/history", params={"limit": limit})

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def get_balance(self) -> dict[str, Any]:
        return self._get("/portfolio/balance")

    def get_positions(self, ticker: Optional[str] = None) -> dict[str, Any]:
        params = {}
        if ticker:
            params["ticker"] = ticker
        return self._get("/portfolio/positions", params=params)

    def get_orders(self, ticker: Optional[str] = None, status: Optional[str] = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        return self._get("/portfolio/orders", params=params)

    # ── Order placement ───────────────────────────────────────────────────────

    def place_order(
        self,
        ticker: str,
        side: str,
        action: str,
        contracts: int,
        price_cents: int,
        order_type: str = "limit",
        client_order_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        side: 'yes' or 'no'
        action: 'buy' or 'sell'
        price_cents: 1-99
        """
        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side.lower(),
            "action": action.lower(),
            "count": contracts,
            "type": order_type,
        }
        if order_type == "limit":
            body["yes_price"] = price_cents if side.lower() == "yes" else (100 - price_cents)
        if client_order_id:
            body["client_order_id"] = client_order_id
        logger.info(_MODULE, "place_order", f"{action} {contracts}x {ticker} {side} @ {price_cents}¢")
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        logger.info(_MODULE, "cancel_order", order_id)
        return self._delete(f"/portfolio/orders/{order_id}")

    # ── Settlement check ──────────────────────────────────────────────────────

    def get_settlement(self, ticker: str) -> Optional[str]:
        """Return 'yes', 'no', or None if not yet settled."""
        try:
            data = self.get_market(ticker)
            market = data.get("market", data)
            result = market.get("result", "")
            if result in ("yes", "no"):
                return result
        except Exception as exc:
            logger.warn(_MODULE, "settlement_check_failed", ticker=ticker, err=str(exc))
        return None

    # ── Convenience: current best prices ─────────────────────────────────────

    def get_best_prices(self, ticker: str) -> tuple[int, int]:
        """Return (best_yes_ask_cents, best_yes_bid_cents). Returns (-1,-1) on failure."""
        try:
            ob_data = self.get_orderbook(ticker, depth=1)
            ob = ob_data.get("orderbook", ob_data)
            asks = ob.get("yes", {}).get("ask", [])
            bids = ob.get("yes", {}).get("bid", [])
            best_ask = asks[0][0] if asks else -1
            best_bid = bids[0][0] if bids else -1
            return best_ask, best_bid
        except Exception as exc:
            logger.warn(_MODULE, "best_prices_failed", ticker=ticker, err=str(exc))
            return -1, -1
