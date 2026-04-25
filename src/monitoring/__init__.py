"""Compatibilidade entre imports `monitoring.*` e `src.monitoring.*`."""

from __future__ import annotations

import sys

if "monitoring.metrics" in sys.modules:
    sys.modules.setdefault("src.monitoring.metrics", sys.modules["monitoring.metrics"])
