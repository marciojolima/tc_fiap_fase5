from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChurnPredictionRequest(BaseModel):
    """Payload de entrada para inferência de churn de um único cliente."""

    CreditScore: int = Field(
        600,
        ge=0,
        le=850,
        description="Pontuação de crédito: Confiabilidade financeira (300-850)",
    )
    Geography: Literal["Germany", "France", "Spain"] = Field(
        "Germany",
        description="País de residência do cliente",
    )
    Gender: Literal["Female", "Male"] = Field(
        "Female",
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
        le=20,
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
        le=9,
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
    card_type: Literal["SILVER", "GOLD", "PLATINUM", "DIAMOND"] = Field(
        "SILVER",
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
                "Geography": "Germany",
                "Gender": "Female",
                "Age": 40,
                "Tenure": 3,
                "Balance": 60000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.0,
                "Card Type": "DIAMOND",
                "Point Earned": 450,
            }
        },
    }


class ChurnPredictionResponse(BaseModel):
    """Resposta de inferência com probabilidade e decisão final."""

    churn_probability: float
    churn_prediction: int
    model_name: str
    threshold: float
