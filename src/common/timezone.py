"""Utilitários centralizados de timezone do projeto."""

from __future__ import annotations

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

DEFAULT_PROJECT_TIMEZONE = "America/Sao_Paulo"


def get_project_timezone_name() -> str:
    """Resolve o timezone do projeto a partir do ambiente."""

    return os.getenv("PROJECT_TIMEZONE") or os.getenv("TZ") or DEFAULT_PROJECT_TIMEZONE


def get_project_timezone() -> ZoneInfo:
    """Retorna o timezone configurado para o projeto."""

    return ZoneInfo(get_project_timezone_name())


def configure_process_timezone() -> None:
    """Alinha o timezone efetivo do processo para logs e timestamps locais."""

    timezone_name = get_project_timezone_name()
    if os.getenv("TZ") != timezone_name:
        os.environ["TZ"] = timezone_name
    if hasattr(time, "tzset"):
        time.tzset()


def now() -> datetime:
    """Retorna o horário atual no timezone configurado para o projeto."""

    return datetime.now(get_project_timezone())


def now_isoformat() -> str:
    """Retorna o horário atual serializado em ISO-8601 com offset."""

    return now().isoformat()
