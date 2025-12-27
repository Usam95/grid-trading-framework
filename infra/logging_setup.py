# infra/logging_setup.py
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

# default logs/ directly under project root
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)


class FsyncFileHandler(logging.FileHandler):
    """
    FileHandler that periodically fsync()'s so SMB-backed mounts (Azure Files)
    reflect non-zero Content Length while the process is still running.

    Why: normal FileHandler flushes Python buffers, but may not force the OS/SMB
    layer to commit the bytes in a way Azure Files reports immediately.
    """

    def __init__(self, filename: Path, *, encoding: str = "utf-8", fsync_every_sec: float = 1.0):
        super().__init__(filename, encoding=encoding)
        self._fsync_every_sec = max(0.0, float(fsync_every_sec))
        self._last_fsync = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)

        if self._fsync_every_sec <= 0:
            return

        now = time.monotonic()
        if (now - self._last_fsync) >= self._fsync_every_sec:
            try:
                self.flush()
                # force the OS/SMB layer to commit content
                os.fsync(self.stream.fileno())
            except Exception:
                # Never let logging crash the app
                pass
            self._last_fsync = now


def init_logging(
    run_name: str = "gridbt",
    level_name: str = "INFO",
    log_dir: Path | str = DEFAULT_LOG_DIR,
) -> Path:
    """
    Initialize root logging with:
      - one file handler (logs/<run_name>_<timestamp>.log)
      - one console handler.

    Returns the path to the log file.
    """
    # Allow runtime override (useful for containers)
    log_dir = Path(os.getenv("LOG_DIR", str(log_dir)))
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logfile = log_dir / f"{run_name}_{ts}.log"

    # Map "INFO" / "DEBUG" / ... to logging level
    level = getattr(logging, level_name.upper(), logging.INFO)

    root = logging.getLogger()

    # Remove old handlers if any (avoid duplicates in REPL/tests)
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(level)

    fmt = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    # File handler with periodic fsync (default 1s; override via env)
    fsync_every = float(os.getenv("LOG_FSYNC_EVERY_SEC", "1.0"))
    fh = FsyncFileHandler(logfile, encoding="utf-8", fsync_every_sec=fsync_every)
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)

    root.addHandler(fh)
    root.addHandler(ch)

    root.info("Logging initialized. Log file: %s", logfile)
    return logfile


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that uses the global handlers configured by init_logging().
    """
    return logging.getLogger(name)
