"""Basic PII detection/redaction helpers used by guardrails."""

from __future__ import annotations

import re

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}-?\d{4}\b")
CPF_PATTERN = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")


def redact_pii(text: str) -> str:
    """Mask simple PII patterns in generated outputs."""

    sanitized = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
    sanitized = PHONE_PATTERN.sub("[PHONE_REDACTED]", sanitized)
    sanitized = CPF_PATTERN.sub("[CPF_REDACTED]", sanitized)
    return sanitized
