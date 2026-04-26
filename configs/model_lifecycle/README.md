# Training Configs

Esta pasta separa a configuracao global do projeto das configuracoes
de treino de modelos.

- `model_current.yaml`: experimento principal aprovado para treino operacional.
- `experiments/*.yaml`: experimentos candidatos independentes.

O executor de treino consome um unico experimento por vez. Novos
experimentos devem ser adicionados em `experiments/` seguindo o mesmo
contrato YAML.

Convencao de governanca com Feast:
- cada YAML de treino deve declarar `feast.feature_service_name`
- esse nome representa o contrato de features consumido pela versao do modelo
- o serving usa esse mesmo contrato ao consultar a online store

Convencao recomendada para artefatos:
- `model_current.yaml` deve apontar para um caminho estavel, como `artifacts/models/model_current.pkl`
- `experiments/*.yaml` podem usar caminhos versionados, como `artifacts/models/<experiment.name>.pkl`
- o treino operacional gera um sidecar de rastreabilidade em `artifacts/models/model_current_metadata.json`
