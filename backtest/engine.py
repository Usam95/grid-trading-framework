# backtest/engine.py
from __future__ import annotations

from dataclasses import dataclass
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
from core.strategy.base import IStrategy
from infra.data_source import LocalFileDataSource, DatasetConfig
from backtest.config import BacktestConfig
from infra.logging_setup import get_logger

# --------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------


@dataclass
class TradeRecord:
    position_id: str
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    realized_pnl: float
    total_fees: float


@dataclass
class BacktestResult:
    trades: List[TradeRecord]
    equity_curve: List[tuple[datetime, float]]
    positions: List[Position]
    raw_orders: List[Order]


# --------------------------------------------------------------------
# Backtest Engine
# --------------------------------------------------------------------


class BacktestEngine:
    """
    Simple backtest engine for one symbol & one strategy.

    Assumptions (for MVP):
      - Spot-like behaviour (LONG only in practice).
      - One position per filled BUY order (FIFO close on SELL).
      - Simple fill model based on candle OHLC.
      - Fees charged on notional of each fill.
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_source: Optional[LocalFileDataSource],
        strategy: IStrategy,
    ):
        self.config = config
        self.data_source = data_source or LocalFileDataSource()
        self.strategy = strategy
        self.log = get_logger("backtest.engine")

        # Engine state – use new AccountState fields
        self.account = AccountState(
            cash_balance=config.initial_balance,
            total_value=config.initial_balance,
            invested_cost=0.0,
        )

        self._open_orders: Dict[str, Order] = {}
        self._open_positions: Dict[str, Position] = {}  # id -> Position
        self._trade_records: List[TradeRecord] = []
        self._equity_curve: List[tuple[datetime, float]] = []
        self._all_orders: List[Order] = []

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def run(self) -> BacktestResult:
        """
        Run the full backtest and return BacktestResult.
        """
        self.log.info(
            "Backtest started: symbol=%s, start=%s, end=%s, initial_balance=%.2f, fee_pct=%.5f",
            self.config.symbol,
            self.config.start,
            self.config.end,
            self.config.initial_balance,
            self.config.trading_fee_pct,
        )

        df = self._load_raw_data()
        candles = self._to_candles(df)
        self.log.info(
            "Converted dataframe to %d candles for symbol=%s",
            len(candles),
            self.config.symbol,
        )

        # Main backtest loop
        for candle in candles:
            # Strategy decides on new orders based on the latest candle + account state
            new_orders = self.strategy.on_candle(candle, self.account) or []

            if new_orders:
                self.log.info(
                    "on_candle: ts=%s close=%.6f -> strategy produced %d new orders",
                    candle.timestamp,
                    candle.close,
                    len(new_orders),
                )

            # Ensure IDs on orders; store as open orders
            for order in new_orders:
                if not order.id:
                    order.id = str(uuid4())
                self._open_orders[order.id] = order
                self._all_orders.append(order)

                self.log.info(
                    "New order: id=%s side=%s price=%.6f qty=%.4f type=%s",
                    order.id,
                    order.side.value,
                    order.price,
                    order.qty,
                    order.type.value,
                )

            # Simulate fills for all open orders against this candle
            filled_events = self._simulate_fills_for_candle(candle)

            # Apply fills to positions & account state and notify strategy
            for event in filled_events:
                self.log.info(
                    "Order fill: order_id=%s side=%s price=%.6f qty=%.4f ts=%s",
                    event.order_id,
                    event.side.value,
                    event.price,
                    event.qty,
                    event.filled_at,
                )
                self._apply_fill(event)
                self.strategy.on_order_filled(event)

            # Recompute equity based on open positions at candle close
            self._update_equity(candle)

        # After loop: compute simple stats & log summary
        stats = self._compute_simple_stats()
        self.log.info("=== Backtest completed for symbol=%s ===", self.config.symbol)
        self.log.info(
            "Final total_value=%.2f (start=%.2f) -> total_return=%.2f%%, max_drawdown=%.2f%%, trades=%d, win_rate=%.2f%%",
            stats["final_value"],
            stats["start_value"],
            stats["total_return_pct"],
            stats["max_drawdown_pct"],
            stats["num_trades"],
            stats["win_rate_pct"],
        )

        return BacktestResult(
            trades=self._trade_records,
            equity_curve=self._equity_curve,
            positions=list(self._open_positions.values()),
            raw_orders=self._all_orders,
        )

    # ----------------------------------------------------------------
    # Data loading
    # ----------------------------------------------------------------
    def _load_raw_data(self) -> pd.DataFrame:
        cfg = DatasetConfig(
            symbol=self.config.symbol,
            start=self.config.start,
            end=self.config.end,
        )
        df = self.data_source.load(cfg)

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Data for {self.config.symbol} is missing columns: {missing}")

        # Log some basic info about the loaded data
        first_ts = df.index[0]
        last_ts = df.index[-1]
        n_rows = len(df)
        self.log.info(
            "Data loaded for %s: %d rows from %s to %s",
            self.config.symbol,
            n_rows,
            first_ts,
            last_ts,
        )

        if n_rows >= 2:
            approx_delta = df.index[1] - df.index[0]
            self.log.info("Approximate candle interval for %s: %s", self.config.symbol, approx_delta)

        return df

    def _to_candles(self, df: pd.DataFrame) -> List[Candle]:
        candles: List[Candle] = []
        for ts, row in df.iterrows():
            candles.append(
                Candle(
                    timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
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
                    position_id=None,  # set in _apply_fill
                )
                filled.append(fill)
            else:
                remaining_orders[order_id] = order

        self._open_orders = remaining_orders
        return filled

    # ----------------------------------------------------------------
    # Position/account updates
    # ----------------------------------------------------------------
    def _apply_fill(self, event: OrderFilledEvent) -> None:
        """
        Update positions & account state given a single fill event.
        For MVP:
          - BUY opens a new LONG position (one per fill).
          - SELL closes the oldest still-open LONG position (FIFO) of same symbol.
        """
        fee = event.price * event.qty * self.config.trading_fee_pct

        if event.side == Side.BUY:
            # Create a new LONG position
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

            # Deduct cost + fee from cash
            notional = event.price * event.qty
            old_cash = self.account.cash_balance
            self.account.cash_balance -= notional + fee

            self.log.info(
                "BUY fill -> opened position %s: size=%.4f entry=%.6f, fee=%.6f; cash %.2f -> %.2f",
                pos_id,
                event.qty,
                event.price,
                fee,
                old_cash,
                self.account.cash_balance,
            )

        elif event.side == Side.SELL:
            # Close the oldest open LONG position for this symbol (naive FIFO)
            open_positions = [
                p
                for p in self._open_positions.values()
                if p.symbol == event.symbol and p.side == PositionSide.LONG and p.is_open
            ]
            if not open_positions:
                # Nothing to close – in grid spot trading this shouldn't happen
                self.log.warning(
                    "SELL fill with no open LONG position for symbol=%s. Ignoring.",
                    event.symbol,
                )
                return

            # FIFO: earliest opened
            pos = sorted(open_positions, key=lambda p: p.opened_at)[0]

            # For MVP assume qty == pos.size (no partials)
            notional = event.price * event.qty
            gross_pnl = (event.price - pos.entry_price) * event.qty
            total_fee = fee + pos.fees_paid

            pos.closed_at = event.filled_at
            pos.realized_pnl = gross_pnl - total_fee

            # Credit cash minus fee
            old_cash = self.account.cash_balance
            self.account.cash_balance += notional - fee

            self._trade_records.append(
                TradeRecord(
                    position_id=pos.id,
                    symbol=pos.symbol,
                    side=pos.side,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    exit_price=event.price,
                    entry_time=pos.opened_at,
                    exit_time=pos.closed_at,
                    realized_pnl=pos.realized_pnl,
                    total_fees=total_fee,
                )
            )

            self.log.info(
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
                # For shorts you might define cost/market differently;
                # keeping LONG-only for MVP.
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
            if pos.symbol != self.config.symbol:
                continue  # single-symbol engine for now

            if pos.side == PositionSide.LONG:
                invested_cost += pos.entry_price * pos.size
                market_value += candle.close * pos.size
            elif pos.side == PositionSide.SHORT:
                invested_cost += pos.entry_price * pos.size
                market_value += (2 * pos.entry_price - candle.close) * pos.size  # placeholder

        self.account.invested_cost = invested_cost
        self.account.total_value = self.account.cash_balance + market_value

        self._equity_curve.append((candle.timestamp, self.account.total_value))

        self.log.debug(
            "Equity update @ %s -> cash=%.2f, invested_cost=%.2f, market_value=%.2f, total_value=%.2f",
            candle.timestamp,
            self.account.cash_balance,
            self.account.invested_cost,
            market_value,
            self.account.total_value,
        )

    # ----------------------------------------------------------------
    # Simple stats / summary
    # ----------------------------------------------------------------
    def _compute_simple_stats(self) -> Dict[str, float]:
        if not self._equity_curve:
            return {
                "start_value": self.config.initial_balance,
                "final_value": self.account.total_value,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "num_trades": len(self._trade_records),
                "win_rate_pct": 0.0,
            }

        values = [v for _, v in self._equity_curve]
        start_value = values[0]
        final_value = values[-1]
        total_return_pct = (final_value / start_value - 1.0) * 100.0

        # Max drawdown
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = v / peak - 1.0
            if dd < max_dd:
                max_dd = dd
        max_drawdown_pct = max_dd * 100.0

        # Win-rate
        num_trades = len(self._trade_records)
        wins = sum(1 for tr in self._trade_records if tr.realized_pnl > 0)
        win_rate_pct = (wins / num_trades * 100.0) if num_trades > 0 else 0.0

        return {
            "start_value": start_value,
            "final_value": final_value,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "num_trades": num_trades,
            "win_rate_pct": win_rate_pct,
        }
