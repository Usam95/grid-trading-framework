# app/grid_research.py
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

from core.research.grid_search import GridResearchRunner, GridTrainRunResult, GridResearchSummary
from core.results.repository import save_backtest_result
from infra.config_loader import load_run_config
from infra.logging_setup import init_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grid parameter research (train + forward) for the simple grid strategy."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/grid_run.yml",
        help="Path to the YAML/JSON run config (default: config/grid_run.yml).",
    )
    return parser.parse_args()


def _write_leaderboard_csv(
    leaderboard: List[GridTrainRunResult],
    path: Path,
) -> None:
    """
    Write param_leaderboard.csv sorted by objective descending.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(leaderboard, key=lambda r: r.score, reverse=True)

    fieldnames = [
        "timeframe",
        "score",
        "run_id",
        "base_order_size",
        "n_levels",
        "lower_pct",
        "upper_pct",
        "lower_price",
        "upper_price",
        "range_mode",
        "spacing",
        "net_pnl",
        "total_return_pct",
        "max_drawdown_pct",
        "n_trades",
        "win_rate_pct",
        "profit_factor",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in sorted_results:
            p = r.params
            m = r.metrics
            row = {
                "timeframe": r.timeframe,
                "score": r.score,
                "run_id": r.run_id,
                "base_order_size": p["base_order_size"],
                "n_levels": p["n_levels"],
                "lower_pct": p["lower_pct"],
                "upper_pct": p["upper_pct"],
                "lower_price": p["lower_price"],
                "upper_price": p["upper_price"],
                "range_mode": getattr(p["range_mode"], "value", str(p["range_mode"])),
                "spacing": getattr(p["spacing"], "value", str(p["spacing"])),
                "net_pnl": m.get("net_pnl", 0.0),
                "total_return_pct": m.get("total_return_pct", 0.0),
                "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
                "n_trades": m.get("n_trades", 0.0),
                "win_rate_pct": m.get("win_rate_pct", 0.0),
                "profit_factor": m.get("profit_factor", 0.0),
            }
            writer.writerow(row)


def _write_summary_json(
    summary: GridResearchSummary,
    symbol: str,
    path: Path,
    train_dir: Path,
    forward_dir: Path,
) -> None:
    """
    Save summary.json containing best params + train/forward metrics.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    best = summary.best_train_result
    fwd = summary.best_forward_result

    payload = {
        "symbol": symbol,
        "best": {
            "timeframe": summary.best_timeframe,
            "params": {
                k: getattr(v, "value", v)  # enums -> their value
                for k, v in summary.best_params.items()
                if k != "timeframe"
            },
            "train": {
                "run_id": best.run_id,
                "initial_balance": best.initial_balance,
                "final_equity": best.final_equity,
                "metrics": best.metrics,
                "result_dir": str(train_dir),
            },
            "forward": {
                "run_id": fwd.run_id,
                "initial_balance": fwd.initial_balance,
                "final_equity": fwd.final_equity,
                "metrics": fwd.metrics,
                "result_dir": str(forward_dir),
            },
        },
        "meta": {
            "total_combinations": len(summary.leaderboard),
            "primary_metric": "total_return_pct",   # explicit
        },
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()

    # 1) Load config
    run_cfg = load_run_config(args.config)

    # 2) Initialize logging
    logfile = init_logging(
        run_name=run_cfg.logging.name,
        level_name=run_cfg.logging.level,
        log_dir=run_cfg.logging.log_dir,
    )
    logger = get_logger(run_cfg.logging.name)
    logger.info("=== Starting grid research run: %s ===", run_cfg.name)
    logger.info("Description: %s", run_cfg.description or "(none)")
    logger.info("Log file for this run: %s", logfile)

    # 3) Run research
    runner = GridResearchRunner(run_cfg)
    summary = runner.run()

    symbol = run_cfg.data.symbol  # type: ignore
    results_root = Path("output") / symbol
    results_root.mkdir(parents=True, exist_ok=True)

    # 4) Save best train & forward runs as full backtest results
    train_dir = save_backtest_result(
        summary.best_train_result,
        base_dir=results_root / "best_train",
    )
    forward_dir = save_backtest_result(
        summary.best_forward_result,
        base_dir=results_root / "best_forward",
    )

    logger.info("Best train run saved under: %s", train_dir)
    logger.info("Best forward run saved under: %s", forward_dir)

    # 5) Write leaderboard + summary
    leaderboard_path = results_root / "param_leaderboard.csv"
    summary_path = results_root / "summary.json"

    _write_leaderboard_csv(summary.leaderboard, leaderboard_path)
    _write_summary_json(summary, symbol, summary_path, train_dir, forward_dir)

    logger.info("Leaderboard written to: %s", leaderboard_path)
    logger.info("Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
