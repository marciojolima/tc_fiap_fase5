"""Fixtures compartilhadas do pytest."""

import pandas as pd
import pytest


@pytest.fixture
def churn_dataframe() -> pd.DataFrame:
    """Dataset pequeno, mas representativo, para testes unitários de churn."""

    return pd.DataFrame(
        {
            "CreditScore": [600, 650, 700, 720, 680, 710, 690, 640, 730, 660],
            "Geography": [
                "France",
                "Germany",
                "Spain",
                "France",
                "Germany",
                "Spain",
                "France",
                "Germany",
                "Spain",
                "France",
            ],
            "Gender": [
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
            ],
            "Age": [30, 40, 35, 50, 28, 45, 33, 41, 38, 29],
            "Tenure": [2, 5, 3, 7, 1, 8, 4, 6, 9, 2],
            "Balance": [
                1000.0,
                2000.0,
                1500.0,
                3000.0,
                0.0,
                2500.0,
                1800.0,
                2200.0,
                2700.0,
                1300.0,
            ],
            "NumOfProducts": [1, 2, 1, 3, 2, 1, 2, 1, 3, 2],
            "HasCrCard": [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
            "IsActiveMember": [1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
            "EstimatedSalary": [
                50000.0,
                60000.0,
                55000.0,
                70000.0,
                52000.0,
                68000.0,
                58000.0,
                61000.0,
                73000.0,
                54000.0,
            ],
            "Exited": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "Complain": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            "Satisfaction Score": [4, 2, 5, 1, 4, 2, 5, 3, 1, 4],
            "Card Type": [
                "SILVER",
                "GOLD",
                "PLATINUM",
                "DIAMOND",
                "SILVER",
                "GOLD",
                "PLATINUM",
                "DIAMOND",
                "SILVER",
                "GOLD",
            ],
            "Point Earned": [100, 200, 150, 250, 120, 230, 170, 210, 260, 140],
        }
    )
