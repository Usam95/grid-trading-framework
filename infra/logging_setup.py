# infra/logging_setup.py
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

# default logs/ directly under project root
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)


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
    log_dir = Path(log_dir)
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

    # File handler
    fh = logging.FileHandler(logfile, encoding="utf-8")
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
