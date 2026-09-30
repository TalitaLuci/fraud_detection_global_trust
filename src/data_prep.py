"""
Funções de carga e limpeza dos dados de transações de cartão de crédito.

Mantém a lógica de preparação separada da modelagem para permitir reuso
tanto no notebook de EDA quanto no pipeline de treino (src/train_model.py)
e no dashboard (dashboard/app.py), evitando duplicação de código.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "creditcard.csv"


def load_raw_data(path: Path = RAW_CSV) -> pd.DataFrame:
    """
    Carrega o CSV bruto de transações e normaliza o tipo da coluna Class.

    Parâmetros
    ----------
    path : Path
        Caminho para o CSV bruto. Default: data/raw/creditcard.csv.

    Retorna
    -------
    pd.DataFrame
        Dataframe com as colunas originais (Time, V1..V28, Amount, Class),
        com `Class` já convertida para inteiro (o arquivo bruto traz essa
        coluna como string "0"/"1").
    """
    df = pd.read_csv(path)
    df["Class"] = df["Class"].astype(int)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicatas exatas e valida ausência de nulos.

    Parâmetros
    ----------
    df : pd.DataFrame
        Dataframe bruto (ver `load_raw_data`).

    Retorna
    -------
    pd.DataFrame
        Dataframe sem duplicatas, com índice resetado.

    Levanta
    -------
    ValueError
        Se restarem valores nulos após a limpeza (não esperado nesta
        base, mas validado defensivamente).

    Notas
    -----
    Armadilha estrutural conhecida desta base: ~1.081 linhas
    completamente duplicadas (idênticas em TODAS as colunas, incluindo
    `Time` e `Class` -- portanto não podem ser transações legítimas
    diferentes ocorrendo no mesmo segundo, e sim registros repetidos).
    Removê-las antes do split evita que a mesma observação apareça em
    treino e teste simultaneamente (vazamento de dados).

    Esta função não faz nenhuma transformação sensível a distribuição
    (scaling, reamostragem) -- isso só pode acontecer depois do split,
    para evitar vazamento de dados (data leakage).
    """
    n_antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_depois = len(df)
    print(f"Duplicatas removidas: {n_antes - n_depois} ({n_antes} -> {n_depois})")

    n_nulos = df.isnull().sum().sum()
    if n_nulos > 0:
        raise ValueError(f"Encontrados {n_nulos} valores nulos - tratar antes de prosseguir.")

    return df


def load_clean_data(path: Path = RAW_CSV) -> pd.DataFrame:
    """Atalho: `load_raw_data` + `clean_data` em uma chamada."""
    return clean_data(load_raw_data(path))


if __name__ == "__main__":
    data = load_clean_data()
    print(data.shape)
    print(data["Class"].value_counts(normalize=True) * 100)
