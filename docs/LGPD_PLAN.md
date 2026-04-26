# Plano de Conformidade LGPD — Datathon Fase 05

## 1. Mapeamento de Dados e Sensibilidade
Identificamos a presença de dados pessoais (`CustomerId`, `Surname`) e dados
financeiros (`Balance`, `EstimatedSalary`) no dataset original.

- **Ação aplicada:** colunas de identificação direta (`Surname`, `CustomerId`) são
  removidas da trilha de treino tabular.

## 2. Bases Legais Utilizadas
- **Legítimo Interesse:** uso de dados para predição de churn (`Exited`) e ações
  de retenção.
- **Execução de Contrato:** consultas operacionais para atendimento e
  relacionamento no contexto bancário.

## 3. Técnicas de Proteção (Privacy by Design)
- **Minimização:** exclusão de identificadores diretos da trilha de treino e
  limitação do payload de inferência ao necessário para predição.
- **Sanitização de saída:** mascaramento de PII no retorno do componente LLM
  (`email`, `telefone`, `CPF`) por `OutputGuardrail`.
- **Guardrails de input:** bloqueio de padrões de prompt injection e limite de
  tamanho de entrada no endpoint `/llm/chat`.

## 4. Gestão de Viés e Discriminação (Fairness)
- O dataset possui atributos sensíveis (`Gender`, `Geography`) que exigem
  monitoramento de viés.
- Nesta fase, a gestão de fairness está documentada e com análise qualitativa
  no `MODEL_CARD`, sem gate automatizado de fairness no pipeline.

## 5. Retenção e Descarte
- Artefatos e logs técnicos são mantidos para rastreabilidade de treino,
  serving, drift e auditoria operacional.
- A política formal de retenção com prazos por tipo de dado permanece como
  evolução posterior ao escopo mínimo da banca.

## 6. Evidências Implementadas no Repositório

- Minimização e seleção de features no pipeline tabular:
  `src/feature_engineering/feature_engineering.py`.
- Sanitização de PII:
  `src/security/pii_detection.py` e `src/security/guardrails.py`.
- Aplicação de guardrails no serving LLM:
  `src/serving/llm_routes.py`.
- Cenários adversariais e evidências de bloqueio/sanitização:
  `tests/test_guardrails.py`,
  `tests/test_guardrails_adversarial.py`,
  `docs/RED_TEAM_REPORT.md`.

## 7. Fora de Escopo nesta Fase

- DSR automatizado (acesso, correção, exclusão) ponta a ponta.
- Catálogo formal de consentimento por finalidade.
- Geração automática de relatório DPIA completo.
