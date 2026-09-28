-- Schema da tabela `transactions`, gerada automaticamente por
-- sql/load_data.py a partir de data/raw/creditcard.csv.
-- Documentado aqui para referência (a criação real é feita via pandas.to_sql).

CREATE TABLE IF NOT EXISTS transactions (
    "Time"   REAL,     -- segundos desde a primeira transação do dataset
    "V1"     REAL,     -- V1..V28: componentes principais (PCA) de variáveis originais sigilosas
    "V2"     REAL,
    "V3"     REAL,
    "V4"     REAL,
    "V5"     REAL,
    "V6"     REAL,
    "V7"     REAL,
    "V8"     REAL,
    "V9"     REAL,
    "V10"    REAL,
    "V11"    REAL,
    "V12"    REAL,
    "V13"    REAL,
    "V14"    REAL,
    "V15"    REAL,
    "V16"    REAL,
    "V17"    REAL,
    "V18"    REAL,
    "V19"    REAL,
    "V20"    REAL,
    "V21"    REAL,
    "V22"    REAL,
    "V23"    REAL,
    "V24"    REAL,
    "V25"    REAL,
    "V26"    REAL,
    "V27"    REAL,
    "V28"    REAL,
    "Amount" REAL,     -- valor da transação
    "Class"  INTEGER   -- 0 = legítima, 1 = fraude
);

CREATE INDEX IF NOT EXISTS idx_transactions_class ON transactions("Class");
