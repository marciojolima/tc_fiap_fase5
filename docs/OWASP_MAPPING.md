# Mapeamento OWASP para LLM

Mapeamento de ameacas OWASP para a trilha LLM/agente do projeto, com
foco em evidências já existentes no repositório.

| Ameaça (OWASP LLM) | Risco no projeto | Mitigação aplicada | Gap residual | Evidência |
| --- | --- | --- | --- | --- |
| Prompt Injection | Usuário tenta sobrescrever instruções do agente para desviar comportamento | `InputGuardrail` bloqueia padrões de injeção por regex e interrompe request com HTTP 400 | Regex é cobertura básica (não semântica) | `src/security/guardrails.py`, `src/serving/llm_routes.py`, `tests/test_guardrails.py` |
| Sensitive Information Disclosure | Resposta do LLM pode expor PII (email, telefone, CPF) | `OutputGuardrail` aplica `redact_pii` antes da resposta ao cliente | Redação cobre padrões comuns, não todo tipo de PII | `src/security/pii_detection.py`, `src/security/guardrails.py`, `tests/test_guardrails.py` |
| Insecure Output Handling | Conteúdo inseguro retornar direto ao cliente sem sanitização | Sanitização obrigatória na saída do agente via `output_guardrail.sanitize(...)` | Não há classificador avançado de conteúdo tóxico | `src/serving/llm_routes.py` |
| Excessive Agency / Tool Misuse | LLM tentar chamar ferramenta inválida ou fora do escopo | ReAct valida ferramenta por allowlist (`tool_by_name`) e rejeita ação inválida com observação controlada | Não há política de autorização por usuário/tenant | `src/agent/react_agent.py`, `tests/test_llm_routes.py` |
| Denial of Service (Input Abuse) | Entradas muito longas degradam latência/custo do endpoint | `InputGuardrail` impõe limite de tamanho (`max_input_chars`) e bloqueia inputs acima do limite | Limite é estático e local (não adaptativo) | `src/security/guardrails.py`, `tests/test_guardrails_adversarial.py` |
