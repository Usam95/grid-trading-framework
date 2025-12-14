# backtest/engine.py
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd

from core.models import (
    Candle,
    Order,
    OrderType,
    Side,
    Position,
    PositionSide,
    AccountState,
    OrderFilledEvent,
)

from core.engine_actions import EngineAction, EngineActionType
from core.strategy.base import IStrategy

from infra.data_source import LocalFileDataSource, DatasetConfig
from infra.config import BacktestEngineConfig, LocalDataConfig
from infra.logging_setup import get_logger
from infra.indicators import enrich_indicators

from core.results.models import BacktestResult, EquityPoint
from core.results.trade_builder import TradeBuilder
from core.results.metrics import MetricRegistry, create_default_metric_registry


class BacktestEngine:
    """
    Simple backtest engine for one symbol & one strategy.

    Assumptions (for MVP):
      - Spot-like behaviour (LONG only in practice).
      - One position per filled BUY order (FIFO close on SELL).
      - Simple fill model based on candle OHLC.
      - Fees charged on notional of each fill.

    IMPORTANT (cleaned architecture):
      - Strategies return EngineAction objects (PLACE_ORDER / GRID_EXIT).
      - The engine is the single authority for assigning Order.id.
      - Order IDs are assigned as SYMBOL-N (e.g. "XRPUSDT-1").
      - client_tag is propagated into OrderFilledEvent so strategies can map fills.
    """

    def __init__(
        self,
        engine_cfg: BacktestEngineConfig,
        data_cfg: LocalDataConfig,
        strategy: IStrategy,
        data_source: Optional[LocalFileDataSource] = None,
        metric_registry: Optional[MetricRegistry] = None,
    ):
        self.engine_cfg = engine_cfg
        self.data_cfg = data_cfg
        self.strategy = strategy
        self.data_source = data_source or LocalFileDataSource()
        self.log = get_logger("backtest.engine")

        # Metrics
        self.metric_registry = metric_registry or create_default_metric_registry()

        # Engine state – use AccountState fields
        self.account = AccountState(
            cash_balance=self.engine_cfg.initial_balance,
            total_value=self.engine_cfg.initial_balance,
            invested_cost=0.0,
        )

        # Open orders/positions for the run
        self._open_orders: Dict[str, Order] = {}          # order_id -> Order
        self._open_positions: Dict[str, Position] = {}    # position_id -> Position
        self._equity_curve: List[EquityPoint] = []
        self._all_orders: List[Order] = []

        # Trade builder (logical trades from fills)
        self._trade_builder = TradeBuilder(symbol=self.data_cfg.symbol)

        # Order ID sequencing (engine-owned):
        # Each symbol has its own counter, producing IDs like: XRPUSDT-1, XRPUSDT-2, ...
        self._order_seq_by_symbol: Dict[str, int] = {}

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def run(self) -> BacktestResult:
        """
        Run the full backtest and return BacktestResult.
        """
        started_at = datetime.now()

        self.log.info(
            "Backtest started: symbol=%s, start=%s, end=%s, initial_balance=%.2f, fee_pct=%.5f",
            self.data_cfg.symbol,
            self.data_cfg.start,
            self.data_cfg.end,
            self.engine_cfg.initial_balance,
            self.engine_cfg.trading_fee_pct,
        )

        df = self._load_raw_data()
        candles = self._to_candles(df)
        self.log.info(
            "Converted dataframe to %d candles for symbol=%s",
            len(candles),
            self.data_cfg.symbol,
        )

        grid_exit_reason: Optional[str] = None

        # ----------------------------------------------------------------
        # Main backtest loop
        # ----------------------------------------------------------------
        for idx, candle in enumerate(candles):
            # Optional: support max_candles in engine_cfg
            if self.engine_cfg.max_candles is not None and idx >= self.engine_cfg.max_candles:
                self.log.info("Stopping early due to max_candles=%d", self.engine_cfg.max_candles)
                break

            # Strategy decides on new actions based on latest candle + account state
            actions = self.strategy.on_candle(candle, self.account) or []

            if actions:
                self.log.debug(
                    "on_candle: ts=%s close=%.6f -> strategy produced %d actions",
                    candle.timestamp,
                    candle.close,
                    len(actions),
                )

            # Handle actions (GRID_EXIT and PLACE_ORDER)
            for action in actions:
                # -------------------------------
                # GRID EXIT (global SL/TP)
                # -------------------------------
                if action.type == EngineActionType.GRID_EXIT:
                    symbol = action.symbol or self.data_cfg.symbol
                    reason = (action.exit_reason or "unknown").lower()

                    self.log.info(
                        "Received GRID_EXIT action: symbol=%s reason=%s at %s",
                        symbol,
                        reason,
                        candle.timestamp,
                    )

                    self._force_flatten_and_cancel(symbol, candle, reason)
                    grid_exit_reason = reason

                    # Stop processing further actions on this candle
                    break

                # -------------------------------
                # PLACE ORDER
                # -------------------------------
                if action.type == EngineActionType.PLACE_ORDER:
                    order = action.order
                    if order is None:
                        self.log.warning(
                            "PLACE_ORDER action without order at %s – ignored",
                            candle.timestamp,
                        )
                        continue

                    # Engine is single authority for order IDs:
                    # always assign/overwrite to ensure deterministic ID scheme.
                    assigned_id = self._assign_order_id(order)

                    self._open_orders[order.id] = order
                    self._all_orders.append(order)

                    self.log.debug(
                        "New order accepted: id=%s side=%s price=%.6f qty=%.4f type=%s tag=%s",
                        assigned_id,
                        order.side.value,
                        order.price,
                        order.qty,
                        order.type.value,
                        order.client_tag,
                    )
                    continue

                # -------------------------------
                # Unknown action
                # -------------------------------
                self.log.warning(
                    "Unknown EngineAction type=%s at %s",
                    action.type,
                    candle.timestamp,
                )

            # Simulate fills for all open orders against this candle
            filled_events = self._simulate_fills_for_candle(candle)

            # Apply fills to positions & account state and notify strategy
            for event in filled_events:
                fee = event.price * event.qty * self.engine_cfg.trading_fee_pct

                self.log.debug(
                    "Order fill: order_id=%s side=%s price=%.6f qty=%.4f ts=%s fee=%.6f tag=%s",
                    event.order_id,
                    event.side.value,
                    event.price,
                    event.qty,
                    event.filled_at,
                    fee,
                    event.client_tag,
                )

                # Update positions & account (pass fee explicitly)
                self._apply_fill(event, fee)

                # Feed TradeBuilder so we get closed trades (pass same fee)
                self._trade_builder.on_fill(event, fee)

                # Strategy callback
                self.strategy.on_order_filled(event)

            # Recompute equity based on open positions at candle close
            self._update_equity(candle)

            # Stop the backtest after a global grid exit
            if grid_exit_reason is not None:
                self.log.info(
                    "Stopping backtest early due to grid_exit_reason=%s at %s",
                    grid_exit_reason,
                    candle.timestamp,
                )
                break

        # ---- After loop: build BacktestResult and compute metrics ----

        final_equity = self._equity_curve[-1].equity if self._equity_curve else self.account.total_value

        result = BacktestResult(
            run_id=str(uuid4()),
            run_name=f"{self.data_cfg.symbol}_backtest",
            symbol=self.data_cfg.symbol,
            timeframe=getattr(self.data_cfg, "timeframe", "unknown"),
            started_at=started_at,
            finished_at=datetime.now(),
            initial_balance=self.engine_cfg.initial_balance,
            final_equity=final_equity,
            trades=self._trade_builder.get_trades(),
            equity_curve=self._equity_curve,
        )

        # If we stopped due to a global grid SL/TP, persist the reason
        if grid_exit_reason is not None:
            result.extra["grid_exit_reason"] = grid_exit_reason

        # Keep positions & raw orders in extra so they’re still accessible
        result.extra["positions"] = list(self._open_positions.values())
        result.extra["raw_orders"] = self._all_orders

        # Compute metrics via registry
        metric_names = self.engine_cfg.metrics or None  # None = all registered
        result.metrics = self.metric_registry.compute_all(result, metric_names)

        # Log summary using metrics
        self.log.info("=== Backtest completed for symbol=%s ===", self.data_cfg.symbol)
        self.log.info(
            "Final total_value=%.2f (start=%.2f) -> total_return=%.2f%%, max_drawdown=%.2f%%, trades=%d, win_rate=%.2f%%",
            result.final_equity,
            result.initial_balance,
            result.metrics.get("total_return_pct", 0.0),
            result.metrics.get("max_drawdown_pct", 0.0),
            int(result.metrics.get("n_trades", 0.0)),
            result.metrics.get("win_rate_pct", 0.0),
        )

        return result

    # ----------------------------------------------------------------
    # Order ID assignment
    # ----------------------------------------------------------------
    def _assign_order_id(self, order: Order) -> str:
        """
        Assign engine-owned order ID as SYMBOL-N (e.g. XRPUSDT-1).

        We always assign/overwrite to ensure:
          - deterministic IDs
          - strategies never rely on IDs (they should use client_tag)
        """
        symbol = order.symbol
        next_n = self._order_seq_by_symbol.get(symbol, 0) + 1
        self._order_seq_by_symbol[symbol] = next_n

        new_id = f"{symbol}-{next_n}"

        if order.id and order.id != new_id:
            self.log.warning(
                "Order already has id=%s but engine overwrites with id=%s (engine is the single authority).",
                order.id,
                new_id,
            )

        order.id = new_id
        return new_id

    # ----------------------------------------------------------------
    # Forced grid exit (flatten + cancel)
    # ----------------------------------------------------------------
    def _force_flatten_and_cancel(self, symbol: str, candle: Candle, reason: str) -> None:
        """
        Force-flatten all open positions and cancel all open orders
        for a symbol.

        Used when a grid strategy sends a GRID_EXIT action
        (global stop-loss / take-profit).
        """
        # 1) Cancel open orders for this symbol
        remaining: Dict[str, Order] = {}
        for order_id, order in self._open_orders.items():
            if order.symbol != symbol:
                remaining[order_id] = order
            else:
                self.log.info(
                    "Grid exit %s: cancelling open order id=%s side=%s price=%.6f qty=%.4f tag=%s",
                    reason,
                    order_id,
                    order.side.value,
                    order.price,
                    order.qty,
                    order.client_tag,
                )
        self._open_orders = remaining

        # 2) Close all open positions for this symbol (LONG-only MVP)
        for pos_id, pos in list(self._open_positions.items()):
            if pos.symbol != symbol or not pos.is_open:
                continue

            event = OrderFilledEvent(
                order_id=f"GRID_EXIT_{reason.upper()}_{pos_id}",
                symbol=pos.symbol,
                side=Side.SELL,  # flatten LONG position
                price=candle.close,
                qty=pos.size,
                filled_at=candle.timestamp,
                client_tag=None,
                position_id=pos.id,
            )
            fee = event.price * event.qty * self.engine_cfg.trading_fee_pct

            self.log.info(
                "Grid exit %s: closing position %s at %.6f qty=%.4f fee=%.6f",
                reason,
                pos_id,
                event.price,
                event.qty,
                fee,
            )

            # Same flow as normal fills
            self._apply_fill(event, fee)
            self._trade_builder.on_fill(event, fee)
            self.strategy.on_order_filled(event)

    # ----------------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------------
    def _load_raw_data(self) -> pd.DataFrame:
        """
        Load raw OHLCV data for the configured symbol/timeframe, then
        optionally enrich it with indicators (ATR/EMA/RSI) according to
        self.data_cfg.indicators.
        """
        cfg = DatasetConfig(
            symbol=self.data_cfg.symbol,
            start=self.data_cfg.start,
            end=self.data_cfg.end,
            timeframe=getattr(self.data_cfg, "timeframe", "1m"),
        )
        df = self.data_source.load(cfg)

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Data for {self.data_cfg.symbol} is missing columns: {missing}")

        # Log some basic info about the loaded data
        first_ts = df.index[0]
        last_ts = df.index[-1]
        n_rows = len(df)
        self.log.info(
            "Data loaded for %s (timeframe=%s): %d rows from %s to %s",
            self.data_cfg.symbol,
            getattr(self.data_cfg, "timeframe", "unknown"),
            n_rows,
            first_ts,
            last_ts,
        )

        if n_rows >= 2:
            approx_delta = df.index[1] - df.index[0]
            self.log.info("Approximate candle interval for %s: %s", self.data_cfg.symbol, approx_delta)

        # ------------------------------------------------------------------
        # Enrich with technical indicators (ATR/EMA/RSI)
        # ------------------------------------------------------------------
        if hasattr(self.data_cfg, "indicators"):
            before_cols = set(df.columns)
            df = enrich_indicators(df, self.data_cfg.indicators)
            after_cols = set(df.columns)
            added_cols = sorted(after_cols - before_cols)
            if added_cols:
                self.log.info(
                    "Indicator columns added for %s: %s",
                    self.data_cfg.symbol,
                    ", ".join(added_cols),
                )
            else:
                self.log.info(
                    "No indicator columns added for %s (config may be empty).",
                    self.data_cfg.symbol,
                )

        return df

    def _to_candles(self, df: pd.DataFrame) -> List[Candle]:
        """
        Convert OHLCV(+indicator) DataFrame (datetime index + columns
        ['Open', 'High', 'Low', 'Close', 'Volume', ...indicators...])
        to a list of Candle objects.

        - Base OHLCV go into the dedicated fields.
        - All additional numeric columns go into Candle.extra.
        """
        candles: List[Candle] = []
        append = candles.append

        base_cols = {"Open", "High", "Low", "Close", "Volume"}
        extra_cols = [c for c in df.columns if c not in base_cols]

        for row in df.itertuples(index=True, name="Row"):
            extras = {col: getattr(row, col) for col in extra_cols}
            append(
                Candle(
                    timestamp=row.Index,
                    open=row.Open,
                    high=row.High,
                    low=row.Low,
                    close=row.Close,
                    volume=row.Volume,
                    extra=extras,
                )
            )

        return candles

    # ----------------------------------------------------------------
    # Fill simulation
    # ----------------------------------------------------------------
    def _simulate_fills_for_candle(self, candle: Candle) -> List[OrderFilledEvent]:
        """
        Very simple fill model:

        - LIMIT BUY fills if candle.low <= price
        - LIMIT SELL fills if candle.high >= price
        - MARKET orders fill at candle.close immediately

        No partial fills for MVP.
        """
        filled: List[OrderFilledEvent] = []
        remaining_orders: Dict[str, Order] = {}

        for order_id, order in self._open_orders.items():
            fill_price: Optional[float] = None

            if order.type == OrderType.MARKET:
                fill_price = candle.close

            elif order.type == OrderType.LIMIT:
                if order.side == Side.BUY and candle.low <= order.price:
                    fill_price = order.price
                elif order.side == Side.SELL and candle.high >= order.price:
                    fill_price = order.price
                else:
                    fill_price = None
            else:
                # STOP / STOP_LIMIT not handled yet in MVP
                fill_price = None

            if fill_price is not None:
                fill = OrderFilledEvent(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    price=fill_price,
                    qty=order.qty,
                    filled_at=candle.timestamp,
                    client_tag=order.client_tag,   # IMPORTANT for grid mapping
                    position_id=None,
                )
                filled.append(fill)
            else:
                remaining_orders[order_id] = order

        self._open_orders = remaining_orders
        return filled

    # ----------------------------------------------------------------
    # Position/account updates
    # ----------------------------------------------------------------
    def _apply_fill(self, event: OrderFilledEvent, fee: float) -> None:
        """
        Update positions & account state given a single fill event.
        For MVP:
          - BUY opens a new LONG position (one per fill).
          - SELL closes the oldest still-open LONG position (FIFO) of same symbol.
        """
        if event.side == Side.BUY:
            pos_id = str(uuid4())
            pos = Position(
                id=pos_id,
                symbol=event.symbol,
                side=PositionSide.LONG,
                entry_price=event.price,
                size=event.qty,
                opened_at=event.filled_at,
                closed_at=None,
                realized_pnl=0.0,
                fees_paid=fee,
            )
            self._open_positions[pos_id] = pos

            notional = event.price * event.qty
            old_cash = self.account.cash_balance
            self.account.cash_balance -= notional + fee

            self.log.debug(
                "BUY fill -> opened position %s: size=%.4f entry=%.6f, fee=%.6f; cash %.2f -> %.2f",
                pos_id,
                event.qty,
                event.price,
                fee,
                old_cash,
                self.account.cash_balance,
            )

        elif event.side == Side.SELL:
            open_positions = [
                p
                for p in self._open_positions.values()
                if p.symbol == event.symbol and p.side == PositionSide.LONG and p.is_open
            ]
            if not open_positions:
                self.log.debug(
                    "SELL fill with no open LONG position for symbol=%s. Ignoring.",
                    event.symbol,
                )
                return

            pos = sorted(open_positions, key=lambda p: p.opened_at)[0]

            notional = event.price * event.qty
            gross_pnl = (event.price - pos.entry_price) * event.qty
            total_fee = fee + pos.fees_paid

            pos.closed_at = event.filled_at
            pos.realized_pnl = gross_pnl - total_fee

            old_cash = self.account.cash_balance
            self.account.cash_balance += notional - fee

            self.log.debug(
                "SELL fill -> closed position %s: size=%.4f entry=%.6f exit=%.6f gross_pnl=%.6f total_fee=%.6f realized_pnl=%.6f; cash %.2f -> %.2f",
                pos.id,
                pos.size,
                pos.entry_price,
                event.price,
                gross_pnl,
                total_fee,
                pos.realized_pnl,
                old_cash,
                self.account.cash_balance,
            )

        # Recompute total_value & invested_cost from open positions
        self._recompute_equity_unrealized(last_price=event.price, symbol=event.symbol)

    def _recompute_equity_unrealized(self, last_price: float, symbol: str) -> None:
        """
        Recompute:
          - invested_cost: sum(entry_price * size) for open positions
          - total_value: cash_balance + market value of open positions
        """
        invested_cost = 0.0
        market_value = 0.0

        for pos in self._open_positions.values():
            if pos.symbol != symbol or not pos.is_open:
                continue

            if pos.side == PositionSide.LONG:
                invested_cost += pos.entry_price * pos.size
                market_value += last_price * pos.size
            elif pos.side == PositionSide.SHORT:
                # LONG-only MVP. Placeholder for later.
                invested_cost += pos.entry_price * pos.size
                market_value += (2 * pos.entry_price - last_price) * pos.size  # placeholder

        self.account.invested_cost = invested_cost
        self.account.total_value = self.account.cash_balance + market_value

        self.log.debug(
            "Recompute equity (unrealized) -> cash=%.2f, invested_cost=%.2f, market_value=%.2f, total_value=%.2f",
            self.account.cash_balance,
            self.account.invested_cost,
            market_value,
            self.account.total_value,
        )

    def _update_equity(self, candle: Candle) -> None:
        """
        Recompute equity (total_value) using candle.close for all open positions.
        Also appends to equity_curve.
        """
        invested_cost = 0.0
        market_value = 0.0

        for pos in self._open_positions.values():
            if not pos.is_open:
                continue
            if pos.symbol != self.data_cfg.symbol:
                continue  # single-symbol engine for now

            if pos.side == PositionSide.LONG:
                invested_cost += pos.entry_price * pos.size
                market_value += candle.close * pos.size
            elif pos.side == PositionSide.SHORT:
                invested_cost += pos.entry_price * pos.size
                market_value += (2 * pos.entry_price - candle.close) * pos.size  # placeholder

        self.account.invested_cost = invested_cost
        self.account.total_value = self.account.cash_balance + market_value

        self._equity_curve.append(EquityPoint(timestamp=candle.timestamp, equity=self.account.total_value))

        self.log.debug(
            "Equity update @ %s -> cash=%.2f, invested_cost=%.2f, market_value=%.2f, total_value=%.2f",
            candle.timestamp,
            self.account.cash_balance,
            self.account.invested_cost,
            market_value,
            self.account.total_value,
        )
