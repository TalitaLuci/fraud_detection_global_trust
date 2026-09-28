-- =====================================================================
-- Queries de Análise Exploratória e de Negócio
-- Base: transactions (SQLite, gerada por sql/load_data.py)
-- =====================================================================

-- 1) Distribuição geral da variável-alvo (desbalanceamento)
SELECT
    Class,
    COUNT(*) AS total_transacoes,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 4) AS percentual
FROM transactions
GROUP BY Class;

-- 2) Duplicatas exatas na base (armadilha estrutural nº1)
SELECT
    COUNT(*) AS total_linhas,
    COUNT(*) - COUNT(DISTINCT Time || '-' || V1 || '-' || V2 || '-' || Amount || '-' || Class)
        AS duplicatas_aproximadas
FROM transactions;

-- 3) Estatísticas de Amount por classe (legítima vs. fraude)
WITH ranked AS (
    SELECT
        Class,
        Amount,
        ROW_NUMBER() OVER (PARTITION BY Class ORDER BY Amount) AS rn,
        COUNT(*) OVER (PARTITION BY Class) AS n
    FROM transactions
)
SELECT
    t.Class,
    COUNT(*)                       AS n,
    ROUND(AVG(t.Amount), 2)        AS media_valor,
    ROUND(MIN(t.Amount), 2)        AS valor_min,
    ROUND(MAX(t.Amount), 2)        AS valor_max,
    ROUND((SELECT AVG(Amount) FROM ranked r
           WHERE r.Class = t.Class AND r.rn IN ((r.n + 1) / 2, (r.n + 2) / 2)), 2)
                                    AS mediana_valor
FROM transactions t
GROUP BY t.Class;

-- 4) Distribuição de fraudes por hora do dia (Time em segundos desde a 1a transacao)
SELECT
    CAST((Time / 3600) AS INT) % 24 AS hora_do_dia,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraudes,
    COUNT(*) AS total_transacoes,
    ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS taxa_fraude_pct
FROM transactions
GROUP BY hora_do_dia
ORDER BY hora_do_dia;

-- 5) Faixas de valor (buckets) x taxa de fraude
SELECT
    CASE
        WHEN Amount = 0                THEN '00 - Zero'
        WHEN Amount <= 10               THEN '01 - Ate 10'
        WHEN Amount <= 50               THEN '02 - 10 a 50'
        WHEN Amount <= 100              THEN '03 - 50 a 100'
        WHEN Amount <= 500              THEN '04 - 100 a 500'
        WHEN Amount <= 2000             THEN '05 - 500 a 2000'
        ELSE '06 - Acima de 2000'
    END AS faixa_valor,
    COUNT(*) AS total_transacoes,
    SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraudes,
    ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS taxa_fraude_pct
FROM transactions
GROUP BY faixa_valor
ORDER BY faixa_valor;

-- 6) Top 10 maiores transações fraudulentas (para storytelling de negócio)
SELECT Time, Amount, Class
FROM transactions
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 10;

-- 7) Prejuízo potencial: soma de Amount em fraudes vs. volume total transacionado
SELECT
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS valor_total_fraudes,
    ROUND(SUM(Amount), 2) AS valor_total_transacionado,
    ROUND(
        100.0 * SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END) / SUM(Amount), 4
    ) AS percentual_prejuizo_sobre_volume
FROM transactions;
