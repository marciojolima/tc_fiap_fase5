from __future__ import annotations

import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(name)s | %(message)s"


def _resolve_log_level() -> int:
    """Resolve o nível de log a partir da variável de ambiente."""

    log_level_name = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, log_level_name, logging.INFO)


def _configure_logging() -> None:
    """Configura um handler único com Rich para todo o processo."""

    if getattr(_configure_logging, "_configured", False):
        return

    console = Console(stderr=False)
    handler = RichHandler(
        console=console,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
        omit_repeated_times=False,
    )
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(_resolve_log_level())
    root_logger.addHandler(handler)

    # Mantém warnings e logs de libs visíveis em stdout de forma consistente.
    logging.captureWarnings(True)
    sys.excepthook = sys.__excepthook__
    _configure_logging._configured = True


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger do projeto com formatação rica no terminal."""

    _configure_logging()
    logger = logging.getLogger(name)
    logger.setLevel(_resolve_log_level())
    logger.propagate = True
    return logger
