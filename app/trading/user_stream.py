# app/trading/user_stream.py
from __future__ import annotations

from typing import Dict, Optional

from infra.marketdata.binance_user_stream import BinanceUserDataStream, UserStreamConfig


def build_execution_report_handler(ctx):
    """
    Thread-safe handler: enqueue raw execution reports ONLY.
    Main thread is responsible for:
      - repo persistence
      - order_mgr updates
      - strategy fill processing
    """
    def _on_exec_report(evt: Dict[str, object]) -> None:
        # IMPORTANT: no strategy calls, no order_mgr calls, no repo I/O here
        ctx.exec_reports.put(evt)

    return _on_exec_report


def start_user_stream(ctx) -> Optional[BinanceUserDataStream]:
    """
    Start Binance user data stream if enabled and creds exist.
    """
    if not ctx.has_private_creds:
        return None

    us_cfg = ctx.trading.user_stream
    if not bool(getattr(us_cfg, "enabled", True)):
        ctx.repo.append_event("userstream.started", {"ok": False, "reason": "disabled"})
        return None

    keepalive = int(getattr(us_cfg, "keepalive_sec", 1800))
    backoff = int(getattr(us_cfg, "reconnect_backoff_sec", 5))

    handler = build_execution_report_handler(ctx)

    stream = BinanceUserDataStream(
        exchange=ctx.exchange,
        logger=ctx.us_log,
        cfg=UserStreamConfig(
            keepalive_interval_sec=keepalive,
            reconnect_backoff_sec=backoff,
        ),
        on_execution_report=handler,
    )
    stream.start()

    if not stream.wait_until_connected(timeout_sec=10.0):
        ctx.us_log.warning("User stream did not connect within timeout.")
        ctx.repo.append_event("userstream.started", {"ok": False, "reason": "timeout"})
        return None

    ctx.us_log.info("User stream started.")
    ctx.repo.append_event("userstream.started", {"ok": True})
    return stream
