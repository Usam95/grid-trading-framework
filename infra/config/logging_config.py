from __future__ import annotations

from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """
    Controls basic logging behaviour.

    Interacts with infra.logging_setup.get_logger().
    """

    name: str = Field(
        "gridbt",
        description="Logger name; used as root for module loggers.",
    )
    level: str = Field(
        "INFO",
        description="Log level: DEBUG, INFO, WARNING, ERROR.",
    )
    log_dir: str = Field(
        "logs",
        description="Directory for log files.",
    )
    to_console: bool = True
    to_file: bool = True
