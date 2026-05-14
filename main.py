from __future__ import annotations

import argparse

import db
import logger
from config import get_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-first Kalshi prediction-market agent")
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="initialize configured storage/log directories and exit",
    )
    args = parser.parse_args()

    cfg = get_config()
    logger.init(cfg.log_dir)
    db.init(cfg.db_path)

    if args.init_only:
        logger.info("main", "initialized", log_dir=cfg.log_dir, db_path=cfg.db_path)
        return

    logger.info(
        "main",
        "startup_safe_mode",
        "no autonomous trading loop is started by main.py",
        trading_mode=cfg.trading_mode,
        kill_switch=cfg.kill_switch,
        live_trading_enabled=cfg.live_trading_enabled,
    )


if __name__ == "__main__":
    main()
