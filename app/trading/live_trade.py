# app/trading/live_trade.py
from __future__ import annotations

import argparse

from app.trading.config import load_trading_settings
from app.trading.runtime import run_trading


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/live_trade.yml")
    args = parser.parse_args()

    settings = load_trading_settings(args.config)
    run_trading(settings)


if __name__ == "__main__":
    main()
