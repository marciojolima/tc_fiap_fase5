# Training Configs

Esta pasta separa a configuracao global do projeto das configuracoes
de treino de modelos.

- `current.json`: experimento principal aprovado para treino operacional.
- `experiments/*.json`: experimentos candidatos independentes.

O executor de treino consome um unico experimento por vez. Novos
experimentos devem ser adicionados em `experiments/` seguindo o mesmo
contrato YAML.

Convencao de governanca com Feast:
- cada config de treino deve declarar `feast.feature_service_name`
- esse nome representa o contrato de features consumido pela versao do modelo
- o serving usa esse mesmo contrato ao consultar a online store

Convencao recomendada para artefatos:
- `current.json` deve apontar para um caminho estavel, como `artifacts/models/current.pkl`
- `experiments/*.json` podem usar caminhos versionados, como `artifacts/models/<experiment.name>.pkl`
- o treino operacional gera um sidecar de rastreabilidade em `artifacts/models/current_metadata.json`
