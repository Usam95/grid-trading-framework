# app/trading/strategy_builder.py
from __future__ import annotations

from core.strategy.base import BaseStrategy
from core.strategy.grid_strategy_simple import GridConfig, SimpleGridStrategy
from core.strategy.grid_strategy_dynamic import DynamicGridConfig, DynamicGridStrategy

from infra.config.strategy_grid import GridStrategyConfig, DynamicGridStrategyConfig, StrategyConfig


def build_dynamic_grid_strategy(strat_cfg: DynamicGridStrategyConfig) -> BaseStrategy:
    """
    Convert Pydantic config -> core DynamicGridConfig -> Strategy.
    Keep this in live_trade.py as you requested (single strategy).
    """
    cfg = DynamicGridConfig(
        symbol=strat_cfg.symbol,
        base_order_size=float(strat_cfg.base_order_size),
        n_levels=int(strat_cfg.n_levels),

        range_mode=strat_cfg.range_mode,
        spacing=strat_cfg.spacing,

        lower_pct=float(strat_cfg.lower_pct) if strat_cfg.lower_pct is not None else None,
        upper_pct=float(strat_cfg.upper_pct) if strat_cfg.upper_pct is not None else None,
        lower_price=float(strat_cfg.lower_price) if strat_cfg.lower_price is not None else None,
        upper_price=float(strat_cfg.upper_price) if strat_cfg.upper_price is not None else None,

        spacing_mode=strat_cfg.spacing_mode,
        spacing_pct=float(strat_cfg.spacing_pct),
        spacing_atr_mult=float(strat_cfg.spacing_atr_mult),

        range_atr_period=int(strat_cfg.range_atr_period),
        range_atr_lower_mult=float(strat_cfg.range_atr_lower_mult),
        range_atr_upper_mult=float(strat_cfg.range_atr_upper_mult),

        use_stop_loss=bool(strat_cfg.use_stop_loss),
        stop_loss_type=strat_cfg.stop_loss_type,
        stop_loss_pct=float(strat_cfg.stop_loss_pct),
        stop_loss_atr_mult=float(strat_cfg.stop_loss_atr_mult),

        use_take_profit=bool(strat_cfg.use_take_profit),
        take_profit_type=strat_cfg.take_profit_type,
        take_profit_pct=float(strat_cfg.take_profit_pct),
        take_profit_atr_mult=float(strat_cfg.take_profit_atr_mult),

        sltp_atr_period=int(strat_cfg.sltp_atr_period),

        floating_grid=strat_cfg.floating_grid,  # pydantic model, used by strategy config (you already defined it)

        use_rsi_filter=bool(strat_cfg.use_rsi_filter),
        rsi_period=int(strat_cfg.rsi_period),
        rsi_min=float(strat_cfg.rsi_min),
        rsi_max=float(strat_cfg.rsi_max),

        use_trend_filter=bool(strat_cfg.use_trend_filter),
        ema_period=int(strat_cfg.ema_period),
        max_deviation_pct=float(strat_cfg.max_deviation_pct),
    )
    return DynamicGridStrategy(cfg)
