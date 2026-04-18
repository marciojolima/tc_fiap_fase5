from __future__ import annotations

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


class ChurnPredictionResponse(BaseModel):
    """Resposta de inferência com probabilidade e decisão final."""

    churn_probability: float
    churn_prediction: int
    model_name: str
    threshold: float


class LLMChatRequest(BaseModel):
    """Request payload for LLM chat endpoint."""

    message: str = Field(..., min_length=1, description="Pergunta do usuário.")
    include_trace: bool = Field(
        default=False,
        description="Retorna trilha ReAct com ferramentas usadas.",
    )


class LLMChatResponse(BaseModel):
    """Response payload for LLM chat endpoint."""

    answer: str
    used_tools: list[str]
    trace: list[dict]
