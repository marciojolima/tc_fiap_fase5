from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Literal

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field

from common.config_loader import load_config
from features.feature_engineering import create_features

app = FastAPI(
    title="TC FIAP Fase 5 API - Datathon",
    version="0.1.0",
    description="API de serving para predição de churn bancário.",
)


class ServingConfig(NamedTuple):
    """Configuração necessária para inferência em produção."""

    target_col: str
    leakage_columns: list[str]
    model_path: Path
    preprocessor_path: Path
    threshold: float

# 1. Defina a tabela em uma constante para reutilizar
DATA_DICT_TABLE = """
### 📋 Dicionário de Dados de Entrada

| Campo | Tradução | Tipo | Restrições | Explicação |
|---|---|---|---|---|
| **CreditScore** | Pontuação de Crédito | int | 300–850 | Confiabilidade financeira |
| **Geography** | País | str | Germany, France, Spain | País de residência |
| **Gender** | Gênero | str | Male, Female | Gênero do cliente |
| **Age** | Idade | int | 18–100 | Idade em anos |
| **Tenure** | Tempo de Casa | int | 0–10 | Anos como cliente |
| **Balance** | Saldo | float | ≥ 0 | Valor em conta |
| **NumOfProducts** | Nº de Produtos | int | 1–4 | Serviços ativos |
| **HasCrCard** | Possui Cartão | int | 0 ou 1 | 1 = Sim |
| **IsActiveMember** | Membro Ativo | int | 0 ou 1 | 1 = Movimenta a conta |
| **EstimatedSalary** | Salário Estimado | float | > 0 | Rendimento anual |
| **Card Type** | Tipo de Cartão | str | DIAMOND, GOLD, etc | Categoria do cartão |
| **Point Earned** | Pontos | int | ≥ 0 | Pontos acumulados |
"""

class ChurnPredictionRequest(BaseModel):
    """Realiza a inferência de churn para um cliente individual."""

    # 2. Adicionamos 'description' em cada campo para aparecer no formulário
    CreditScore: int = Field(
        600, ge=0, le=850, 
        description="Pontuação de crédito: Confiabilidade financeira (300-850)"
    )
    Geography: Literal["Germany", "France", "Spain"] = Field(
        "Germany", 
        description="País de residência do cliente"
    )
    Gender: Literal["Female", "Male"] = Field(
        "Female", 
        description="Gênero do cliente"
    )
    Age: int = Field(
        40, ge=18, le=100, 
        description="Idade do cliente em anos"
    )
    Tenure: int = Field(
        5, ge=0, le=10, 
        description="Anos que o cliente possui conta no banco"
    )
    Balance: float = Field(
        0.0, ge=0, 
        description="Saldo disponível em conta"
    )
    NumOfProducts: int = Field(
        1, ge=1, le=4, 
        description="Quantidade de produtos bancários ativos"
    )
    HasCrCard: int = Field(
        1, ge=0, le=1, 
        description="Possui cartão de crédito? (1=Sim, 0=Não)"
    )
    IsActiveMember: int = Field(
        1, ge=0, le=1, 
        description="Cliente movimenta a conta com frequência? (1=Sim, 0=Não)"
    )
    EstimatedSalary: float = Field(
        50000.0, gt=0, 
        description="Rendimento anual estimado do cliente"
    )
    card_type: Literal["SILVER", "GOLD", "PLATINUM", "DIAMOND"] = Field(
        "SILVER", 
        alias="Card Type", 
        description="Categoria do cartão do cliente"
    )
    point_earned: int = Field(
        100, 
        alias="Point Earned", 
        ge=0, 
        description="Pontos de fidelidade acumulados"
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
                "Point Earned": 450
            }
        }
    }


class ChurnPredictionResponse(BaseModel):
    """Resposta de inferência com probabilidade e decisão final."""

    churn_probability: float
    churn_prediction: int
    model_name: str
    threshold: float


def load_serving_config() -> ServingConfig:
    """Carrega a configuração usada pela API de inferência."""

    config = load_config()
    return ServingConfig(
        target_col=config["data"]["target_col"],
        leakage_columns=config["features"]["leakage_columns"],
        model_path=Path("artifacts/challenger_model.pkl"),
        preprocessor_path=Path("artifacts/preprocessor.joblib"),
        threshold=0.5,
    )


@lru_cache
def load_prediction_model():
    """Carrega o modelo challenger persistido para serving."""

    return load(load_serving_config().model_path)


@lru_cache
def load_preprocessor():
    """Carrega o pré-processador persistido para serving."""

    return load(load_serving_config().preprocessor_path)


def prepare_inference_dataframe(
    payload: ChurnPredictionRequest,
    cfg: ServingConfig,
) -> pd.DataFrame:
    """Prepara um único registro bruto para inferência no modelo."""

    df = pd.DataFrame([payload.model_dump(by_alias=True)])
    df_feat = create_features(df)

    leakage_in_features = [c for c in cfg.leakage_columns if c != cfg.target_col]
    existing_leakage = [c for c in leakage_in_features if c in df_feat.columns]
    if existing_leakage:
        df_feat = df_feat.drop(columns=existing_leakage)

    return df_feat


def predict_from_dataframe(df_feat: pd.DataFrame) -> tuple[float, int]:
    """Aplica pré-processamento e modelo para obter a predição final."""

    preprocessor = load_preprocessor()
    model = load_prediction_model()

    X = preprocessor.transform(df_feat)
    probability = float(model.predict_proba(X)[0][1])
    prediction = int(probability >= load_serving_config().threshold)
    return probability, prediction


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    description=f"""
    Realiza a inferência de churn para um cliente individual.
    {DATA_DICT_TABLE}
    """
)
def predict_churn(payload: ChurnPredictionRequest) -> ChurnPredictionResponse:

    cfg = load_serving_config()
    df_feat = prepare_inference_dataframe(payload, cfg)
    probability, prediction = predict_from_dataframe(df_feat)

    return ChurnPredictionResponse(
        churn_probability=probability,
        churn_prediction=prediction,
        model_name="challenger_model",
        threshold=cfg.threshold,
    )
