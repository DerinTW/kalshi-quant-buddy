from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() == "true"


def _env_float(names: tuple[str, ...], default: str) -> float:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return float(value)
    return float(default)


def _env_int(names: tuple[str, ...], default: str) -> int:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return int(value)
    return int(default)


def _mode_prefix() -> str:
    mode = os.getenv("TRADING_MODE", "paper").strip().lower()
    if mode == "live":
        return "LIVE"
    if mode == "paper":
        return "PAPER"
    return ""


def _mode_names(names: tuple[str, ...]) -> tuple[str, ...]:
    prefix = _mode_prefix()
    if prefix == "LIVE":
        return tuple(f"{prefix}_{name}" for name in names)
    if not prefix:
        return names
    return tuple(f"{prefix}_{name}" for name in names) + names


def _env_float_mode(names: tuple[str, ...], paper_default: str, live_default: str) -> float:
    default = live_default if _mode_prefix() == "LIVE" else paper_default
    return _env_float(_mode_names(names), default)


def _env_int_mode(names: tuple[str, ...], paper_default: str, live_default: str) -> int:
    default = live_default if _mode_prefix() == "LIVE" else paper_default
    return _env_int(_mode_names(names), default)


def _max_minutes_to_expiry_default() -> int:
    hours = os.getenv("MAX_TIME_TO_RESOLUTION_HOURS")
    if hours is not None:
        return int(float(hours) * 60)
    return _env_int(("MAX_MINUTES_TO_EXPIRY",), "4320")


def _env_csv(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


@dataclass
class Config:
    # Kalshi API
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_private_key_path: str = field(default_factory=lambda: os.getenv("KALSHI_PRIVATE_KEY_PATH", ""))
    kalshi_base_url: str = field(default_factory=lambda: os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"))

    # Anthropic
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))

    # Perplexity (optional — used by research agents for real-time web search)
    # If absent, research agents fall back to LLM-only research (training data only)
    perplexity_api_key: str = field(default_factory=lambda: os.getenv("PERPLEXITY_API_KEY", ""))
    perplexity_model: str = field(default_factory=lambda: os.getenv("PERPLEXITY_MODEL", "sonar"))

    # ── Category-aware research sources (all optional; absent = source skipped) ──
    # FRED (St. Louis Fed) — free key at https://fred.stlouisfed.org/docs/api/api_key.html
    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))
    # EIA (US Energy Info Administration) — free key at https://www.eia.gov/opendata/register.php
    eia_api_key: str = field(default_factory=lambda: os.getenv("EIA_API_KEY", ""))
    # BLS — optional key at https://data.bls.gov/registrationEngine/ (raises daily quota)
    bls_api_key: str = field(default_factory=lambda: os.getenv("BLS_API_KEY", ""))
    # NOAA Climate Data Online — free token at https://www.ncdc.noaa.gov/cdo-web/token
    noaa_token: str = field(default_factory=lambda: os.getenv("NOAA_TOKEN", ""))
    # NWS api.weather.gov requires a contact User-Agent (no key); provide an email/URL
    nws_user_agent: str = field(default_factory=lambda: os.getenv("NWS_USER_AGENT", "black-gibbie (contact: unset)"))

    # Kill switch — set to true to halt all trading immediately
    kill_switch: bool = field(default_factory=lambda: _env_bool("KILL_SWITCH", "true"))

    # Explicit trading mode: dry_run | paper | live
    trading_mode: str = field(default_factory=lambda: os.getenv("TRADING_MODE", "paper"))

    # Trading mode (legacy flags kept for backward compat)
    paper_only: bool = field(default_factory=lambda: os.getenv("PAPER_ONLY", "true").lower() != "false")
    live_trading_enabled: bool = field(default_factory=lambda: _env_bool("LIVE_TRADING_ENABLED", "false"))
    dry_run_log_only: bool = field(default_factory=lambda: _env_bool("DRY_RUN_LOG_ONLY", "true"))
    paper_bankroll: float = field(default_factory=lambda: float(os.getenv("PAPER_BANKROLL", "1000.0")))

    # Market filters
    min_liquidity_dollars: float = field(default_factory=lambda: _env_float_mode(("MIN_LIQUIDITY", "MIN_LIQUIDITY_DOLLARS"), "25", "500"))
    max_spread_pct: float = field(default_factory=lambda: _env_float_mode(("MAX_SPREAD_PCT",), "50", "20"))
    min_minutes_to_expiry: int = field(default_factory=lambda: _env_int(("MIN_MINUTES_TO_EXPIRY", "MIN_TIME_TO_RESOLUTION_MINUTES"), "20"))
    max_minutes_to_expiry: int = field(default_factory=_max_minutes_to_expiry_default)
    min_volume_24h: int = field(default_factory=lambda: _env_int_mode(("MIN_VOLUME_24H", "MIN_VOLUME"), "50", "1000"))
    min_yes_price: int = field(default_factory=lambda: _env_int_mode(("MIN_YES_PRICE",), "1", "15"))
    max_yes_price: int = field(default_factory=lambda: _env_int_mode(("MAX_YES_PRICE",), "99", "85"))
    # Absolute spread cap in cents (yes_ask - yes_bid); separate from the pct check
    max_spread_cents: int = field(default_factory=lambda: _env_int_mode(("MAX_SPREAD_CENTS",), "10", "6"))
    # How old our fetched market/orderbook snapshot can be before it is considered stale
    max_orderbook_age_seconds: int = field(default_factory=lambda: _env_int_mode(("MAX_ORDERBOOK_AGE_SECONDS",), "300", "60"))
    # Minimum contracts available at the best ask/bid before the market is considered too thin
    min_orderbook_depth_at_limit: int = field(default_factory=lambda: _env_int_mode(("MIN_ORDERBOOK_DEPTH_AT_LIMIT",), "25", "100"))
    # Comma-separated list of allowed categories; empty = all categories allowed
    # politics is intentionally omitted from the default (too noisy / regulatory risk)
    category_allowlist: list[str] = field(
        default_factory=lambda: [
            c.strip()
            for c in os.getenv(
                "CATEGORY_ALLOWLIST",
                "crypto,financial,economic,commodities,weather,science and technology,culture",
            ).split(",")
            if c.strip()
        ]
    )
    # Optional fetch cap for smoke tests/manual runs. None means full scan.
    max_raw_markets_per_scan: int | None = field(
        default_factory=lambda: (
            int(os.getenv("MAX_RAW_MARKETS_PER_SCAN"))
            if os.getenv("MAX_RAW_MARKETS_PER_SCAN")
            else None
        )
    )
    # Comma-separated list of tickers that are permanently blocked
    blocked_tickers: set[str] = field(
        default_factory=lambda: {
            t.strip() for t in os.getenv("BLOCKED_TICKERS", "").split(",") if t.strip()
        }
    )
    blocked_event_prefixes: list[str] = field(
        default_factory=lambda: _env_csv(
            "BLOCKED_EVENT_PREFIXES",
            "KXATP,KXWTA,KXWNBA,KXNBA,KXNFL,KXNHL,KXMLB,KXUFC,KXPGA,KXNCAAF,KXNCAAB,KXMLS",
        )
    )

    # Anomaly detection
    price_jump_threshold_pct: float = field(default_factory=lambda: float(os.getenv("PRICE_JUMP_THRESHOLD_PCT", "10")))
    volume_spike_multiplier: float = field(default_factory=lambda: float(os.getenv("VOLUME_SPIKE_MULTIPLIER", "3.0")))
    stale_book_minutes: int = field(default_factory=lambda: int(os.getenv("STALE_BOOK_MINUTES", "60")))

    # Edge thresholds
    min_edge_pct: float = field(default_factory=lambda: _env_float_mode(("MIN_EDGE_PCT",), "4", "7"))
    min_adjusted_edge_pct: float = field(default_factory=lambda: _env_float_mode(("MIN_ADJUSTED_EDGE_PCT",), "2", "5"))
    slippage_cents: int = field(default_factory=lambda: int(os.getenv("SLIPPAGE_CENTS", "2")))
    fee_pct: float = field(default_factory=lambda: float(os.getenv("FEE_PCT", "0.0")))
    min_confidence: float = field(default_factory=lambda: float(os.getenv("MIN_CONFIDENCE", "0.65")))
    min_confidence_adjusted_edge_cents: float = field(default_factory=lambda: _env_float_mode(("MIN_CONFIDENCE_ADJUSTED_EDGE_CENTS",), "1.5", "4.0"))
    # Spread gate applied at the edge layer (stricter than the filter's max_spread_cents)
    max_spread_cents_edge: int = field(default_factory=lambda: _env_int_mode(("MAX_SPREAD_CENTS_EDGE",), "10", "6"))

    # Risk limits
    max_trade_dollars: float = field(default_factory=lambda: _env_float(("MAX_DOLLARS_PER_TRADE", "MAX_TRADE_DOLLARS"), "10"))
    max_total_exposure_dollars: float = field(default_factory=lambda: float(os.getenv("MAX_TOTAL_EXPOSURE_DOLLARS", "200")))
    max_position_pct_of_bankroll: float = field(default_factory=lambda: _env_float(("MAX_BANKROLL_PCT_PER_TRADE", "MAX_POSITION_PCT_OF_BANKROLL"), "0.5"))
    max_correlated_exposure_dollars: float = field(default_factory=lambda: _env_float(("MAX_CORRELATED_EXPOSURE", "MAX_CORRELATED_EXPOSURE_DOLLARS"), "15"))
    max_category_exposure_dollars: float = field(default_factory=lambda: _env_float(("MAX_CATEGORY_EXPOSURE", "MAX_CATEGORY_EXPOSURE_DOLLARS"), "25"))
    # Daily circuit breakers
    max_daily_loss_dollars: float = field(default_factory=lambda: _env_float(("MAX_DAILY_LOSS", "MAX_DAILY_LOSS_DOLLARS"), "20"))
    max_trades_per_day: int = field(default_factory=lambda: _env_int(("MAX_TRADES_PER_DAY",), "5"))

    # Position monitoring / exit controls
    stop_loss_cents: int = field(default_factory=lambda: _env_int(("STOP_LOSS_CENTS",), "12"))
    take_profit_cents: int = field(default_factory=lambda: _env_int(("TAKE_PROFIT_CENTS",), "8"))
    exit_if_edge_below_cents: int = field(default_factory=lambda: _env_int(("EXIT_IF_EDGE_BELOW_CENTS",), "2"))
    no_new_entries_last_minutes: int = field(default_factory=lambda: _env_int(("NO_NEW_ENTRIES_LAST_MINUTES",), "20"))
    force_review_last_minutes: int = field(default_factory=lambda: _env_int(("FORCE_REVIEW_LAST_MINUTES",), "30"))

    # Live trading gates
    min_paper_trades_before_live: int = field(default_factory=lambda: _env_int(("MIN_PAPER_TRADES_BEFORE_LIVE",), "100"))
    min_paper_pnl_before_live: float = field(default_factory=lambda: float(os.getenv("MIN_PAPER_PNL_BEFORE_LIVE", "0.0")))
    min_paper_days_before_live: int = field(default_factory=lambda: _env_int(("MIN_PAPER_TRADING_DAYS_BEFORE_LIVE", "MIN_PAPER_DAYS_BEFORE_LIVE"), "7"))
    # Safety cap: max_trade_dollars must be ≤ this when mode is live
    max_live_dollars_per_trade: float = field(default_factory=lambda: _env_float(("MAX_LIVE_TRADE_DOLLARS", "MAX_LIVE_DOLLARS_PER_TRADE"), "1"))
    # Must be set to exactly "I_UNDERSTAND_THIS_CAN_LOSE_MONEY" to enable live orders
    live_confirmation_phrase: str = field(default_factory=lambda: os.getenv("LIVE_CONFIRMATION_PHRASE", ""))
    # Must be true (in addition to live_trading_enabled) to allow actual order placement
    allow_live_orders: bool = field(default_factory=lambda: _env_bool("ALLOW_LIVE_ORDERS", "false"))

    # Storage
    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "./logs"))
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "./black_gibbie.db"))

    @property
    def max_dollars_per_trade(self) -> float:
        return self.max_trade_dollars

    @property
    def max_bankroll_pct_per_trade(self) -> float:
        return self.max_position_pct_of_bankroll

    @property
    def max_daily_loss(self) -> float:
        return self.max_daily_loss_dollars

    @property
    def max_category_exposure(self) -> float:
        return self.max_category_exposure_dollars

    @property
    def max_correlated_exposure(self) -> float:
        return self.max_correlated_exposure_dollars

    @property
    def min_liquidity(self) -> float:
        return self.min_liquidity_dollars

    @property
    def max_live_trade_dollars(self) -> float:
        return self.max_live_dollars_per_trade

    def validate(self) -> None:
        errors: list[str] = []
        if not self.kalshi_api_key:
            errors.append("KALSHI_API_KEY is required")
        if not self.kalshi_private_key_path:
            errors.append("KALSHI_PRIVATE_KEY_PATH is required")
        elif not os.path.exists(self.kalshi_private_key_path):
            errors.append(f"Private key file not found: {self.kalshi_private_key_path}")
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
