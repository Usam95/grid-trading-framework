# core/results/metrics.py
from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .models import BacktestResult, EquityPoint, Trade

MetricFunc = Callable[[BacktestResult], float]


class MetricRegistry:
    """
    Name -> metric function mapping.

    Each metric function:
      (BacktestResult) -> float
    """

    def __init__(self) -> None:
        self._metrics: Dict[str, MetricFunc] = {}

    def register(self, name: str, func: MetricFunc) -> None:
        if name in self._metrics:
            # you can choose to raise error instead
            raise ValueError(f"Metric '{name}' already registered")
        self._metrics[name] = func

    def compute(self, name: str, result: BacktestResult) -> float:
        return self._metrics[name](result)

    def compute_all(
        self,
        result: BacktestResult,
        names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        metric_names = names or list(self._metrics.keys())
        return {n: self._metrics[n](result) for n in metric_names}

    def list_metrics(self) -> List[str]:
        return list(self._metrics.keys())


# ----------------------------------------------------------------------
# Concrete metric implementations
# ----------------------------------------------------------------------


def m_net_pnl(res: BacktestResult) -> float:
    # initial_balance is the field in your BacktestResult
    return res.final_equity - res.initial_balance


def m_total_return_pct(res: BacktestResult) -> float:
    # same here: use initial_balance
    if res.initial_balance <= 0:
        return 0.0
    return (res.final_equity / res.initial_balance - 1.0) * 100.0


def _max_drawdown_values(equity_curve: List[EquityPoint]) -> tuple[float, float]:
    if not equity_curve:
        return 0.0, 0.0

    peak = equity_curve[0].equity
    max_dd = 0.0

    for point in equity_curve:
        if point.equity > peak:
            peak = point.equity
        dd = peak - point.equity
        if dd > max_dd:
            max_dd = dd

    return max_dd, peak


def m_max_drawdown(res: BacktestResult) -> float:
    max_dd, _ = _max_drawdown_values(res.equity_curve)
    return max_dd


def m_max_drawdown_pct(res: BacktestResult) -> float:
    max_dd, peak = _max_drawdown_values(res.equity_curve)
    if peak <= 0:
        return 0.0
    return max_dd / peak * 100.0


def m_n_trades(res: BacktestResult) -> float:
    return float(len(res.trades))


def m_win_rate_pct(res: BacktestResult) -> float:
    trades: List[Trade] = res.trades
    if not trades:
        return 0.0

    # Trade has net_pnl (not realized_pnl)
    wins = [t for t in trades if t.net_pnl > 0]
    return len(wins) / len(trades) * 100.0


def m_avg_trade_pnl(res: BacktestResult) -> float:
    trades: List[Trade] = res.trades
    if not trades:
        return 0.0

    # Again, initial_balance instead of initial_equity
    return (res.final_equity - res.initial_balance) / len(trades)


def m_profit_factor(res: BacktestResult) -> float:
    trades: List[Trade] = res.trades
    if not trades:
        return 0.0

    # use net_pnl as realized result per trade
    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = sum(t.net_pnl for t in trades if t.net_pnl < 0)  # negative

    if gross_loss >= 0:
        return float("inf")
    return gross_profit / abs(gross_loss)


# ----------------------------------------------------------------------
# Helper to build default registry
# ----------------------------------------------------------------------


def create_default_metric_registry() -> MetricRegistry:
    reg = MetricRegistry()
    reg.register("net_pnl", m_net_pnl)
    reg.register("total_return_pct", m_total_return_pct)
    reg.register("max_drawdown", m_max_drawdown)
    reg.register("max_drawdown_pct", m_max_drawdown_pct)
    reg.register("n_trades", m_n_trades)
    reg.register("win_rate_pct", m_win_rate_pct)
    reg.register("avg_trade_pnl", m_avg_trade_pnl)
    reg.register("profit_factor", m_profit_factor)
    return reg
