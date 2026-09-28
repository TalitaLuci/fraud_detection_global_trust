"""
Carrega o dataset bruto de transações de cartão de crédito para um banco
SQLite local, servindo de camada de dados para as consultas SQL do projeto.

Uso:
    python sql/load_data.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "creditcard.csv"
DB_PATH = ROOT / "data" / "processed" / "fraud_detection.db"


def load_csv_to_sqlite(csv_path: Path = RAW_CSV, db_path: Path = DB_PATH) -> None:
    """Lê o CSV bruto e grava na tabela `transactions` de um banco SQLite."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    # Class vem como string "0"/"1" no arquivo bruto -> normaliza para inteiro
    df["Class"] = df["Class"].astype(int)

    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_class ON transactions(Class);"
        )
        conn.commit()
    finally:
        conn.close()

    print(f"Carregado {len(df):,} registros em {db_path}")


if __name__ == "__main__":
    load_csv_to_sqlite()
