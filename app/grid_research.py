# app/grid_research.py
from __future__ import annotations

import argparse
import csv
import json
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.research.grid_search import GridResearchRunner, GridTrainRunResult, GridResearchSummary
from core.results.repository import save_backtest_result
from infra.config_loader import load_run_config
from infra.logging_setup import init_logging, get_logger


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grid parameter research (train + forward) for the grid strategy."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/grid_run.yml",
        help="Path to the YAML/JSON run config (default: config/grid_run.yml).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Formatting helpers (CSV + JSON)
# --------------------------------------------------------------------------------------

def _csv_cell(v: Any, decimals: int = 3) -> str:
    """Format cells for CSV: empty for None, ints as ints, floats with fixed decimals."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, Number):
        return f"{float(v):.{decimals}f}"
    return str(v)


def _round_numbers(obj: Any, decimals: int = 3) -> Any:
    """Recursively round floats (and numpy floats) to N decimals; keep ints unchanged."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, Number):
        f = float(obj)
        # Keep integers (e.g. numpy int-like) as int if possible
        if float(int(f)) == f:
            return int(f)
        return round(f, decimals)
    if isinstance(obj, dict):
        return {k: _round_numbers(v, decimals) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_numbers(v, decimals) for v in obj]
    # datetime-like
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    return obj


def _enum_to_value(x: Any) -> Any:
    return getattr(x, "value", x)


def _extract_data_range(result: Any) -> Dict[str, Any]:
    """
    Prefer engine-provided `result.extra["data_range"]` if present.
    Otherwise derive from equity curve timestamps (best-effort).
    """
    try:
        dr = result.extra.get("data_range")
        if isinstance(dr, dict) and dr:
            return dict(dr)
    except Exception:
        pass

    data_start = None
    data_end = None
    n_bars = None

    try:
        eq = getattr(result, "equity_curve", None) or []
        if eq:
            data_start = getattr(eq[0], "timestamp", None)
            data_end = getattr(eq[-1], "timestamp", None)
            n_bars = len(eq)
    except Exception:
        pass

    return {
        "data_start": data_start,
        "data_end": data_end,
        "n_bars": n_bars,
    }


# --------------------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------------------

def _write_leaderboard_csv(
    leaderboard: List[GridTrainRunResult],
    path: Path,
    *,
    decimals: int = 3,
    run_id_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Write param_leaderboard.csv sorted by objective descending.

    - Floats formatted to 3 decimals
    - run_id replaced by incrementing number (via run_id_map if provided)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_results = sorted(leaderboard, key=lambda r: float(r.score), reverse=True)

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
            p = r.params or {}
            m = r.metrics or {}

            rid = r.run_id
            if run_id_map is not None:
                rid = run_id_map.get(r.run_id, r.run_id)

            row = {
                "timeframe": r.timeframe,
                "score": _csv_cell(r.score, decimals),
                "run_id": rid,

                "base_order_size": _csv_cell(p.get("base_order_size"), decimals),
                "n_levels": str(int(p.get("n_levels") or 0)),

                "lower_pct": _csv_cell(p.get("lower_pct"), decimals),
                "upper_pct": _csv_cell(p.get("upper_pct"), decimals),

                "lower_price": _csv_cell(p.get("lower_price"), decimals),
                "upper_price": _csv_cell(p.get("upper_price"), decimals),

                "range_mode": str(_enum_to_value(p.get("range_mode"))),
                "spacing": str(_enum_to_value(p.get("spacing"))),

                "net_pnl": _csv_cell(m.get("net_pnl", 0.0), decimals),
                "total_return_pct": _csv_cell(m.get("total_return_pct", 0.0), decimals),
                "max_drawdown_pct": _csv_cell(m.get("max_drawdown_pct", 0.0), decimals),

                "n_trades": str(int(round(float(m.get("n_trades", 0.0) or 0.0)))),
                "win_rate_pct": _csv_cell(m.get("win_rate_pct", 0.0), decimals),
                "profit_factor": _csv_cell(m.get("profit_factor", 0.0), decimals),
            }
            writer.writerow(row)


def _write_summary_json(
    summary: GridResearchSummary,
    symbol: str,
    path: Path,
    train_dir: Path,
    forward_dir: Optional[Path],
    *,
    primary_metric: str = "total_return_pct",
    topk: Optional[List[Dict[str, Any]]] = None,
    run_id_map: Optional[Dict[str, str]] = None,
    configured_start: Any = None,
    configured_end: Any = None,
    decimals: int = 3,
) -> None:
    """
    Write the overall research summary.json.

    - Floats rounded to 3 decimals
    - Uses incrementing run_id in output (via run_id_map) but also keeps orig_run_id
    - Adds backtest data range (data_start/data_end/n_bars) for best train/forward
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    best_train = summary.best_train_result
    best_forward = summary.best_forward_result

    def _map_run_id(orig: str) -> str:
        if run_id_map is None:
            return orig
        return run_id_map.get(orig, orig)

    best_train_range = _extract_data_range(best_train)
    best_forward_range = _extract_data_range(best_forward) if best_forward is not None else None

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "primary_metric": primary_metric,

        "run": {
            "started_at": summary.started_at,
            "finished_at": summary.finished_at,
            "total_combinations": summary.total_combinations,
            "total_timeframes": summary.total_timeframes,
            "successful_runs": summary.successful_runs,
            "failed_runs": summary.failed_runs,
        },

        # what data range the user configured (optional; research may split internally)
        "configured_data": {
            "start": configured_start,
            "end": configured_end,
        },

        "best": {
            "timeframe": summary.best_timeframe,
            "score": summary.best_score,
            "params": {k: _enum_to_value(v) for k, v in (summary.best_params or {}).items()},
            "train": {
                "run_id": _map_run_id(best_train.run_id),
                "orig_run_id": best_train.run_id,
                "result_dir": str(train_dir),
                "metrics": dict(best_train.metrics or {}),
                "data_range": best_train_range,
            },
            "forward": None,
        },

        "topk": topk or [],
        "meta": {
            "leaderboard_rows": len(summary.leaderboard or []),
        },
    }

    if best_forward is not None:
        payload["best"]["forward"] = {
            "run_id": _map_run_id(best_forward.run_id),
            "orig_run_id": best_forward.run_id,
            "result_dir": str(forward_dir) if forward_dir is not None else None,
            "metrics": dict(best_forward.metrics or {}),
            "data_range": best_forward_range,
        }

    payload = _round_numbers(payload, decimals=decimals)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

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

    symbol = run_cfg.data.symbol  # type: ignore[attr-defined]
    results_root = Path("output") / symbol
    results_root.mkdir(parents=True, exist_ok=True)

    research_cfg = getattr(run_cfg, "research", None)
    save_cfg = getattr(research_cfg, "save", None) if research_cfg is not None else None

    save_mode = getattr(save_cfg, "mode", "best") if save_cfg else "best"      # "best" | "topk"
    top_k = int(getattr(save_cfg, "top_k", 10) or 10) if save_cfg else 10
    write_extra = bool(getattr(save_cfg, "write_extra", True)) if save_cfg else True
    primary_metric = getattr(research_cfg, "primary_metric", "total_return_pct") if research_cfg else "total_return_pct"

    # 4) Save best train & (optional) forward as full backtest results
    train_dir = save_backtest_result(
        summary.best_train_result,
        base_dir=results_root / "best_train",
        write_extra=write_extra,
    )

    forward_dir: Optional[Path] = None
    if summary.best_forward_result is not None:
        forward_dir = save_backtest_result(
            summary.best_forward_result,
            base_dir=results_root / "best_forward",
            write_extra=write_extra,
        )

    logger.info("Best train run saved under: %s", train_dir)
    logger.info("Best forward run saved under: %s", forward_dir)

    # 5) Build run_id -> incrementing number map (leaderboard rank)
    sorted_lb = sorted(summary.leaderboard, key=lambda r: float(r.score), reverse=True)
    run_id_map: Dict[str, str] = {r.run_id: f"{i + 1:06d}" for i, r in enumerate(sorted_lb)}

    # 6) Optionally save top-k train runs
    topk_saved: List[Dict[str, Any]] = []
    if save_mode == "topk":
        topk_entries = getattr(summary, "topk_train", []) or []
        topk_results = getattr(summary, "topk_train_results", []) or []

        if topk_entries and topk_results:
            topk_root = results_root / "topk"
            topk_root.mkdir(parents=True, exist_ok=True)

            for rank, (entry, result) in enumerate(
                zip(topk_entries[:top_k], topk_results[:top_k]),
                start=1,
            ):
                rank_dir = topk_root / f"rank_{rank:03d}"
                saved_dir = save_backtest_result(
                    result,
                    base_dir=rank_dir,
                    write_extra=write_extra,
                )

                topk_saved.append(
                    {
                        "rank": rank,
                        "timeframe": entry.timeframe,
                        "score": entry.score,
                        "run_id": run_id_map.get(entry.run_id, entry.run_id),
                        "orig_run_id": entry.run_id,
                        "params": {k: _enum_to_value(v) for k, v in (entry.params or {}).items()},
                        "result_dir": str(saved_dir),
                    }
                )

            logger.info("Top-%d train runs saved under: %s", len(topk_saved), topk_root)
        else:
            logger.warning("save.mode='topk' but summary.topk_* lists are empty -> nothing saved.")

    # 7) Write leaderboard + overall summary
    leaderboard_path = results_root / "param_leaderboard.csv"
    summary_path = results_root / "summary.json"

    _write_leaderboard_csv(
        summary.leaderboard,
        leaderboard_path,
        decimals=3,
        run_id_map=run_id_map,
    )

    _write_summary_json(
        summary=summary,
        symbol=symbol,
        path=summary_path,
        train_dir=train_dir,
        forward_dir=forward_dir,
        primary_metric=primary_metric,
        topk=topk_saved if save_mode == "topk" else None,
        run_id_map=run_id_map,
        configured_start=getattr(run_cfg.data, "start", None),
        configured_end=getattr(run_cfg.data, "end", None),
        decimals=3,
    )

    logger.info("Leaderboard written to: %s", leaderboard_path)
    logger.info("Summary written to: %s", summary_path)


if __name__ == "__main__":
    main()
