"""Input and output guardrails for LLM interactions."""

from __future__ import annotations

import re

from security.pii_detection import redact_pii


class InputGuardrail:
    """Validate user input before sending it to the LLM."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+a",
        r"developer:\s*",
        r"assistant:\s*",
        r"system:\s*",
        r"<\|im_start\|>",
        r"\[INST\]",
        r"forget\s+(everything|all|your\s+instructions)",
    ]

    def __init__(self, max_input_chars: int = 4096):
        self.max_input_chars = max_input_chars
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS
        ]

    def validate(self, user_input: str) -> tuple[bool, str]:
        sanitized_input = "".join(ch for ch in user_input if ch.isprintable()).strip()
        for pattern in self._compiled_patterns:
            if pattern.search(sanitized_input):
                return False, "Input bloqueado: padrão suspeito detectado."
        if len(sanitized_input) > self.max_input_chars:
            return False, (
                "Input bloqueado: excede tamanho máximo "
                f"({self.max_input_chars} chars)."
            )
        return True, "OK"


class OutputGuardrail:
    """Sanitize model output before returning it to users."""

    @staticmethod
    def sanitize(llm_output: str) -> str:
        return redact_pii(llm_output)


class FinancialGuardrail:
    """Legacy guardrail hook preserved for compatibility."""

    @staticmethod
    def validate_output(response_text: str) -> str:
        return OutputGuardrail().sanitize(response_text)
