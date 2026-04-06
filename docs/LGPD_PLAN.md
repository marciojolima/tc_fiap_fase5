# Plano de Conformidade LGPD — Datathon Fase 05

## 1. Mapeamento de Dados e Sensibilidade
Identificamos a presença de Dados Pessoais (`Surname`, `CustomerId`) e Dados Financeiros (`Balance`, `EstimatedSalary`). 
- **Ação:** A coluna `Surname` será excluída de todos os artefatos de treinamento e logs de inferência.

## 2. Bases Legais Utilizadas
- **Execução de Contrato:** Para ferramentas do Agente que consultam saldo e produtos ativos.
- **Legítimo Interesse:** Para o modelo de predição de Churn (`Exited`) visando retenção de clientes.

## 3. Técnicas de Proteção (Privacy by Design)
- **Minimização:** O Agente ReAct só acessa o `CustomerId` anonimizado.
- **Anonimização:** Em ambiente de Staging, os salários são perturbados (adição de ruído) para evitar identificação por engenharia reversa.
- **Sanitização de Saída:** Implementação de filtros de PII (Microsoft Presidio) para garantir que nomes ou identificadores não apareçam nos logs do Langfuse/TruLens.

## 4. Gestão de Viés e Discriminação (Fairness)
Dado que o dataset contém `Gender` e `Geography`, realizamos testes de impacto para garantir que o modelo não penalize grupos específicos na predição de Churn ou oferta de `Card Type`.

## 5. Retenção e Descarte
Os logs de interação do Agente serão armazenados por 90 dias para auditoria de segurança e depois anonimizados permanentemente.
