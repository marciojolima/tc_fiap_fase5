from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from common.config_loader import load_global_config

_GLOBAL_CONFIG = load_global_config()
_CATEGORICAL_FEATURES = _GLOBAL_CONFIG["features"]["categorical_features"]
_GEOGRAPHY_CATEGORIES = _CATEGORICAL_FEATURES["one_hot"]["Geography"]
_GENDER_CATEGORIES = _CATEGORICAL_FEATURES["ordinal"]["Gender"]
_CARD_TYPE_CATEGORIES = _CATEGORICAL_FEATURES["ordinal"]["Card Type"]


class ChurnPredictionRequest(BaseModel):
    """Payload de entrada para inferência de churn de um único cliente."""

    CreditScore: int = Field(
        600,
        ge=0,
        le=850,
        description="Pontuação de crédito: Confiabilidade financeira (300-850)",
    )
    Geography: str = Field(
        _GEOGRAPHY_CATEGORIES[0],
        description="País de residência do cliente",
    )
    Gender: str = Field(
        _GENDER_CATEGORIES[0],
        description="Gênero do cliente",
    )
    Age: int = Field(
        40,
        ge=18,
        le=100,
        description="Idade do cliente em anos",
    )
    Tenure: int = Field(
        5,
        ge=0,
        le=10,
        description="Anos que o cliente possui conta no banco",
    )
    Balance: float = Field(
        0.0,
        ge=0,
        description="Saldo disponível em conta",
    )
    NumOfProducts: int = Field(
        1,
        ge=1,
        le=4,
        description="Quantidade de produtos bancários ativos",
    )
    HasCrCard: int = Field(
        1,
        ge=0,
        le=1,
        description="Possui cartão de crédito? (1=Sim, 0=Não)",
    )
    IsActiveMember: int = Field(
        1,
        ge=0,
        le=1,
        description="Cliente movimenta a conta com frequência? (1=Sim, 0=Não)",
    )
    EstimatedSalary: float = Field(
        50000.0,
        gt=0,
        description="Rendimento anual estimado do cliente",
    )
    card_type: str = Field(
        _CARD_TYPE_CATEGORIES[0],
        alias="Card Type",
        description="Categoria do cartão do cliente",
    )
    point_earned: int = Field(
        100,
        alias="Point Earned",
        ge=0,
        description="Pontos de fidelidade acumulados",
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "CreditScore": 600,
                "Geography": _GEOGRAPHY_CATEGORIES[1],
                "Gender": _GENDER_CATEGORIES[0],
                "Age": 40,
                "Tenure": 3,
                "Balance": 60000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0,
                "Card Type": _CARD_TYPE_CATEGORIES[-1],
                "Point Earned": 450,
            }
        },
    }

    @field_validator("Geography")
    @classmethod
    def validate_geography(cls, value: str) -> str:
        if value not in _GEOGRAPHY_CATEGORIES:
            raise ValueError(f"Geography deve ser um de {_GEOGRAPHY_CATEGORIES}")
        return value

    @field_validator("Gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        if value not in _GENDER_CATEGORIES:
            raise ValueError(f"Gender deve ser um de {_GENDER_CATEGORIES}")
        return value

    @field_validator("card_type")
    @classmethod
    def validate_card_type(cls, value: str) -> str:
        if value not in _CARD_TYPE_CATEGORIES:
            raise ValueError(f"Card Type deve ser um de {_CARD_TYPE_CATEGORIES}")
        return value


class ChurnCustomerLookupRequest(BaseModel):
    """Payload enxuto para predição usando a online store do Feast."""

    customer_id: int = Field(
        ...,
        gt=0,
        description="Identificador técnico do cliente na Feature Store.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": 15634602,
            }
        },
    }


class ChurnPredictionResponse(BaseModel):
    """Resposta de inferência com probabilidade e decisão final."""

    churn_probability: float
    churn_prediction: int
    model_name: str
    threshold: float
    feature_source: str
    customer_id: int | None = None


SUPPORTED_TRAINING_ALGORITHMS = (
    "gradient_boosting",
    "logistic_regression",
    "random_forest",
    "xgboost",
)
SUPPORTED_MODEL_FLAVORS = ("sklearn",)


class TrainExperimentConfig(BaseModel):
    """Bloco de identificação do experimento de treino."""

    name: str = Field(..., min_length=1, description="Nome lógico do experimento.")
    run_name: str = Field(..., min_length=1, description="Nome da run no MLflow.")
    version: str = Field(..., min_length=1, description="Versão semântica do modelo.")
    algorithm: str = Field(
        ...,
        description=(
            "Algoritmo de treino. "
            f"Suportados: {SUPPORTED_TRAINING_ALGORITHMS}."
        ),
    )
    flavor: str = Field(
        default="sklearn",
        description=f"Stack de serialização. Suportados: {SUPPORTED_MODEL_FLAVORS}.",
    )

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, value: str) -> str:
        if value not in SUPPORTED_TRAINING_ALGORITHMS:
            raise ValueError(
                f"algorithm deve ser um de {SUPPORTED_TRAINING_ALGORITHMS}"
            )
        return value

    @field_validator("flavor")
    @classmethod
    def validate_flavor(cls, value: str) -> str:
        if value not in SUPPORTED_MODEL_FLAVORS:
            raise ValueError(f"flavor deve ser um de {SUPPORTED_MODEL_FLAVORS}")
        return value


class TrainDatasetConfig(BaseModel):
    """Bloco de dados e contrato de features do treino."""

    target_col: str = Field(
        "Exited",
        min_length=1,
        description="Nome da coluna alvo no dataset processado.",
    )
    feature_set: str = Field(
        ...,
        min_length=1,
        description="Identificador lógico do conjunto de features.",
    )


class TrainParamsConfig(BaseModel):
    """Bloco de hiperparâmetros do modelo."""

    params: dict[str, Any] = Field(
        ...,
        description=(
            "Hiperparâmetros do estimador, no mesmo formato do config de treino."
        ),
    )

    @field_validator("params")
    @classmethod
    def validate_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("training.params não pode ser vazio")
        return value


class TrainInferenceConfig(BaseModel):
    """Bloco de parâmetros usados na etapa de avaliação do treino."""

    threshold: float = Field(
        ...,
        gt=0,
        lt=1,
        description="Threshold de classificação usado para métricas e metadata.",
    )


class TrainFeastConfig(BaseModel):
    """Bloco do contrato Feast associado ao modelo treinado."""

    feature_service_name: str = Field(
        ...,
        min_length=1,
        description="FeatureService que representa o contrato online do modelo.",
    )


class TrainArtifactsConfig(BaseModel):
    """Bloco de artefatos gerados pelo treino."""

    model_path: str = Field(
        ...,
        min_length=1,
        description=(
            "Caminho de saída do modelo serializado. "
            "Não pode apontar para o modelo champion ativo do serving."
        ),
    )

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, value: str) -> str:
        if Path(value).suffix != ".pkl":
            raise ValueError("artifacts.model_path deve terminar com .pkl")
        return value


class TrainMlflowConfig(BaseModel):
    """Bloco de metadados MLflow do experimento."""

    experiment_name: str = Field(
        ...,
        min_length=1,
        description="Nome do experimento no MLflow.",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="Tags adicionais do experimento a serem registradas no MLflow.",
    )


class TrainRegistryConfig(BaseModel):
    """Bloco de registry lógico do modelo."""

    enabled: bool = Field(
        False,
        description=(
            "Mantido por compatibilidade de contrato; não promove modelo "
            "no endpoint."
        ),
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Nome lógico do modelo no registry.",
    )
    alias: str | None = Field(
        default=None,
        description="Alias opcional do modelo no registry.",
    )


class TrainGovernanceConfig(BaseModel):
    """Bloco de governança do experimento."""

    risk_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Classificação de risco do modelo.",
    )
    fairness_checked: bool = Field(
        default=False,
        description="Indica se houve checagem formal de fairness.",
    )


class TrainModelRequest(BaseModel):
    """Payload HTTP para treino síncrono de um experimento individual."""

    experiment: TrainExperimentConfig
    dataset: TrainDatasetConfig
    training: TrainParamsConfig
    inference: TrainInferenceConfig
    feast: TrainFeastConfig
    artifacts: TrainArtifactsConfig
    mlflow: TrainMlflowConfig
    registry: TrainRegistryConfig
    governance: TrainGovernanceConfig = Field(
        default_factory=TrainGovernanceConfig,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "experiment": {
                    "name": "random_forest_candidate_api",
                    "run_name": "random_forest_candidate_api",
                    "version": "0.2.1",
                    "algorithm": "random_forest",
                    "flavor": "sklearn",
                },
                "dataset": {
                    "target_col": "Exited",
                    "feature_set": "processed_v1",
                },
                "training": {
                    "params": {
                        "n_estimators": 200,
                        "max_depth": 15,
                        "random_state": 42,
                        "class_weight": "balanced",
                        "min_samples_leaf": 5,
                    }
                },
                "inference": {
                    "threshold": 0.5,
                },
                "feast": {
                    "feature_service_name": "customer_churn_rf_v2",
                },
                "artifacts": {
                    "model_path": (
                        "artifacts/models/candidates/"
                        "random_forest_candidate_api.pkl"
                    ),
                },
                "mlflow": {
                    "experiment_name": "datathon-churn-baseline",
                    "tags": {
                        "owner": "datathon-grupo",
                        "phase": "datathon-fase05",
                        "dataset_name": "bank-customer-churn",
                        "candidate_type": "api_candidate",
                    },
                },
                "registry": {
                    "enabled": False,
                    "model_name": "churn-classifier",
                    "alias": None,
                },
                "governance": {
                    "risk_level": "high",
                    "fairness_checked": False,
                },
            }
        },
    }


class TrainModelResponse(BaseModel):
    """Resposta do endpoint síncrono de treino."""

    status: Literal["completed"]
    experiment_name: str
    run_name: str
    model_version: str
    model_path: str
    metadata_path: str
    metrics: dict[str, float]
    promoted_to_serving: bool
    message: str


class LLMChatRequest(BaseModel):
    """Request payload for LLM chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        description=(
            "Pergunta do usuário para o agente ReAct. "
            "Para smoke test, use perguntas alinhadas ao golden set do projeto."
        ),
    )
    include_trace: bool = Field(
        default=True,
        description=(
            "Quando `true`, retorna a trilha ReAct com parse, tools usadas, "
            "observações e metadados úteis para debug."
        ),
    )
    answer_style: Literal["short", "medium", "long"] = Field(
        default="medium",
        description=(
            "Controla o tamanho da resposta final: `short` para respostas "
            "curtas, `medium` para respostas objetivas e `long` para respostas "
            "mais detalhadas."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": (
                    "Cite pelo menos três ferramentas do agente ReAct "
                    "ligadas ao domínio do datathon."
                ),
                "include_trace": True,
                "answer_style": "short",
            }
        }
    }


class LLMChatResponse(BaseModel):
    """Response payload for LLM chat endpoint."""

    answer: str
    used_tools: list[str]
    trace: list[dict]
