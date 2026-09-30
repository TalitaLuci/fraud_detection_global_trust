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
    Adiciona variáveis derivadas ao dataframe de transações.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dataframe contendo, no mínimo, as colunas `Time` e `Amount`.

    Retorna
    -------
    pd.DataFrame
        Cópia do dataframe original com as colunas adicionais:

        - `Hour`: hora do dia (0-23), derivada de `Time` (segundos desde
          a 1a transação do dataset). `Time` bruto não é diretamente
          interpretável como horário real, mas o padrão cíclico de 24h
          é o que importa para o negócio.
        - `Hour_sin`, `Hour_cos`: codificação cíclica de `Hour` via seno
          e cosseno. Necessária porque, numericamente, 23 e 0 são
          "vizinhos" no relógio (23h -> 0h), mas um `Hour` inteiro trata
          essas duas horas como as mais distantes possíveis da escala.
          Isso afeta principalmente modelos lineares (ex. Regressão
          Logística); modelos de árvore são menos sensíveis a essa
          descontinuidade, mas a feature cíclica não prejudica nenhum
          dos dois casos.
        - `Amount_log`: log1p(Amount), para reduzir a cauda longa da
          distribuição de valores (Amount vai de 0 a ~25.691, com forte
          concentração em valores baixos).
    """
    df = df.copy()
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Amount_log"] = np.log1p(df["Amount"])
    return df


FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + [
    "Amount_log", "Hour", "Hour_sin", "Hour_cos"
]
TARGET_COLUMN = "Class"


def get_feature_matrix(df: pd.DataFrame):
    """
    Monta a matriz de features (X) e o vetor-alvo (y) prontos para split.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dataframe limpo (ver `data_prep.clean_data`), contendo as colunas
        originais V1..V28, Time, Amount e Class.

    Retorna
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) -- X contém as colunas em `FEATURE_COLUMNS` (já com as
        variáveis engenheiradas aplicadas); y é a coluna `Class`.
    """
    df = add_engineered_features(df)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
