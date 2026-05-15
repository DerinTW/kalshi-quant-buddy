from __future__ import annotations

import argparse
import json
from typing import Optional

import db
import logger
from config import get_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-first Kalshi prediction-market agent"
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="initialize configured storage/log directories and exit",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help=(
            "Run exactly ONE scan→evaluate→(maybe execute) pipeline cycle "
            "and exit. This is not an autonomous loop. Execution still "
            "requires every existing safety gate (trading mode, kill switch, "
            "live gates, risk manager) to be satisfied."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Override MAX_CANDIDATES_PER_RUN for this single run.",
    )
    args = parser.parse_args()

    cfg = get_config()
    logger.init(cfg.log_dir)
    db.init(cfg.db_path)

    if args.init_only:
        logger.info("main", "initialized",
                    log_dir=cfg.log_dir, db_path=cfg.db_path)
        return

    if args.run_once:
        _run_once_cli(cfg, args)
        return

    # Default path: NO autonomous trading loop is started.
    logger.info(
        "main",
        "startup_safe_mode",
        "no autonomous trading loop is started by main.py",
        trading_mode=cfg.trading_mode,
        kill_switch=cfg.kill_switch,
        live_trading_enabled=cfg.live_trading_enabled,
    )


def _run_once_cli(cfg, args) -> None:
    """
    Run a single pipeline cycle. Imports pipeline lazily so the default
    main path stays cheap and so `--init-only` doesn't pull in trading deps.
    """
    import pipeline  # lazy import — keeps default startup minimal

    client = _build_client_if_credentials_present(cfg)
    if client is None:
        logger.warn(
            "main", "run_once_no_client",
            "Kalshi credentials missing — running with no market source. "
            "Use a paper environment or set credentials to fetch real markets.",
        )

    summary = pipeline.run_once(
        cfg,
        client=client,
        max_candidates=args.max_candidates,
    )

    logger.info(
        "main", "run_once_summary",
        markets_fetched=summary["markets_fetched"],
        markets_passed_filters=summary["markets_passed_filters"],
        candidates_analyzed=summary["candidates_analyzed"],
        executions_attempted=summary["executions_attempted"],
        errors=len(summary["errors"]),
    )
    # Surface a JSON-safe summary to stdout for operators.
    print(json.dumps(_jsonable(summary), indent=2, default=str))


def _build_client_if_credentials_present(cfg) -> Optional[object]:
    """
    Construct a KalshiClient only when both keys are present. Returns None
    in dev/test environments — pipeline.run_once handles None safely.
    """
    if not cfg.kalshi_api_key or not cfg.kalshi_private_key_path:
        return None
    try:
        from kalshi_client import KalshiClient
        return KalshiClient(
            api_key=cfg.kalshi_api_key,
            private_key_path=cfg.kalshi_private_key_path,
            base_url=getattr(cfg, "kalshi_base_url", "https://api.elections.kalshi.com"),
        )
    except Exception as exc:
        logger.error("main", "kalshi_client_init_failed", err=str(exc))
        return None


def _jsonable(value):
    """Best-effort JSON-safe coercion for summary printing."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":
    main()
