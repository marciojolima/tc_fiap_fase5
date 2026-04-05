# Training Configs

Esta pasta separa a configuracao global do projeto das configuracoes
de treino de modelos.

- `model_current.yaml`: experimento principal aprovado para treino operacional.
- `experiments/*.yaml`: experimentos candidatos independentes.

Nesta fase da refatoracao, os arquivos ja existem para estabelecer o
contrato e a organizacao futura. O treino atual ainda depende
temporariamente da secao `models` em
`configs/pipeline_global_config.yaml`.
