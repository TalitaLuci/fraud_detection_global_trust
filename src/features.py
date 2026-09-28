"""
Engenharia de variáveis para o modelo de detecção de fraude.

Todas as transformações aqui são feature-wise e não aprendem parâmetros
a partir dos dados (diferente de scalers), então podem ser aplicadas
antes ou depois do split sem risco de vazamento -- mas os SCALERS
(ver train_model.py) são sempre fitados só no treino.
"""
import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona:
    - Hour: hora do dia (0-23), derivada de Time (segundos desde a 1a
      transação do dataset). Time bruto não é diretamente interpretável
      como horário real, mas o padrão cíclico de 24h é o que importa.
    - Amount_log: log1p(Amount), para reduzir a cauda longa da distribuição
      de valores (Amount vai de 0 a ~25.691, com forte concentração em
      valores baixos).
    """
    df = df.copy()
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["Amount_log"] = np.log1p(df["Amount"])
    return df


FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount_log", "Hour"]
TARGET_COLUMN = "Class"


def get_feature_matrix(df: pd.DataFrame):
    """Retorna (X, y) prontos para split, já com as features engenheiradas."""
    df = add_engineered_features(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
