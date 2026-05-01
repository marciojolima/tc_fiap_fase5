# Artefatos Relevantes

Este documento reúne os principais artefatos do projeto e o papel de cada um
na trilha de dados, treino, serving, monitoramento e avaliação.

| Artefato | Papel no projeto |
|---|---|
| `data/interim/cleaned.parquet` | Base saneada da camada `interim`, com remoção de identificadores diretos, deduplicação, tratamento de nulos e validação de schema, antes da conversão para o formato final de modelagem. |
| `data/processed/train.parquet` | Base final de treino da camada `processed`, com split, criação de features derivadas, remoção de leakage, encoding e scaling, pronta para consumo pelos algoritmos. |
| `data/processed/test.parquet` | Base final de teste da camada `processed`, gerada com o mesmo pipeline do treino e mantida separada para validação sem vazamento. |
| `data/processed/feature_columns.json` | Registra a ordem e os nomes finais das features, ajudando a manter consistência entre treino e inferência. |
| `data/processed/schema_report.json` | Evidência da validação estrutural dos dados processados, reforçando a etapa de qualidade de dados. |
| `artifacts/models/feature_pipeline.joblib` | Pipeline de transformação persistido para reutilização no serving, evitando divergência entre treino e produção. |
| `artifacts/models/current.pkl` | Modelo champion mantido como versão principal para inferência. |
| `artifacts/models/current_metadata.json` | Metadados do champion, incluindo informações de versão, configuração e métricas relevantes. |
| `artifacts/models/challengers/` | Diretório reservado para challengers gerados em ciclos de retreino e comparados antes de eventual promoção. |
| `artifacts/logs/inference/predictions.jsonl` | Log de inferências usado como base para monitoramento posterior. O contrato registra principalmente as features transformadas efetivamente servidas ao modelo, com metadados mínimos de predição e origem. |
| `artifacts/evaluation/model/drift/drift_report.html` | Relatório HTML oficial do projeto para drift, coerente com `drift_metrics.json` e com a decisão operacional baseada em PSI. |
| `artifacts/evaluation/model/drift/drift_report_evidently.html` | Relatório auxiliar do Evidently, mantido para diagnóstico visual complementar das distribuições e widgets estatísticos. |
| `artifacts/evaluation/model/drift/drift_metrics.json` | Consolidação das métricas de drift, incluindo PSI por feature e resumo para automação de decisão. |
| `artifacts/evaluation/model/drift/drift_status.json` | Estado mais recente do monitoramento de drift, com classificação para apoio ao gatilho de retreino. |
| `artifacts/evaluation/model/drift/drift_runs.jsonl` | Histórico de execuções do monitoramento, útil para trilha de auditoria e acompanhamento temporal. |
| `artifacts/evaluation/model/retraining/retrain_request.json` | Registro do pedido de retreino, com motivação e contexto do disparo do processo. |
| `artifacts/evaluation/model/retraining/retrain_run.json` | Resultado consolidado da execução do retreino, incluindo status, motivo, métricas e decisão final. |
| `artifacts/evaluation/model/retraining/promotion_decision.json` | Decisão champion-challenger com regra de promoção explícita e deltas de métricas entre os modelos comparados. |
| `artifacts/evaluation/model/retraining/generated_configs/` | Configurações geradas automaticamente para retreinos auditáveis e reproduzíveis. |
| `configs/scenario_experiments/inference_cases.yaml` | Suíte versionada de cenários de inferência usada para validar comportamento do modelo em casos de negócio. |
| `artifacts/evaluation/model/scenario_experiments/drift/*.jsonl` | Lotes sintéticos construídos para simular diferentes perfis de drift e testar o fluxo de monitoramento. |
| `artifacts/evaluation/model/scenario_experiments/drift/*_report.html` | Relatórios HTML dos cenários sintéticos, usados para demonstração e validação do processo de drift. |
| [EVALUATION.md](EVALUATION.md) | Índice principal das avaliações do projeto: modelo tabular, cenários, drift, retreino e RAG/LLM. |
