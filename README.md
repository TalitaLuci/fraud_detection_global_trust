# 🏦 Detecção de Fraude em Transações Bancárias — Global Trust Bank

Projeto final do curso **EBAC — Profissão: Cientista de Dados** (Módulo 43).
Simula o atendimento a um gestor de risco de um grande banco que precisa de
um modelo capaz de identificar fraudes em transações de cartão de crédito
o mais rápido possível, minimizando prejuízo financeiro e preservando a
experiência de clientes legítimos.

> Projeto conduzido de ponta a ponta: definição do problema, EDA estatística,
> consultas SQL, engenharia de features, modelagem comparativa, avaliação
> orientada a negócio e um dashboard interativo para exploração dos
> resultados.

## 📌 O problema de negócio

| Item | Definição |
|---|---|
| Problema de negócio | Identificar transações fraudulentas em tempo hábil, minimizando prejuízo financeiro e fricção com clientes legítimos |
| Tarefa de ML | Classificação binária supervisionada |
| Variável-alvo | `Class` (0 = legítima, 1 = fraude) |
| Volume analisado | 284.807 transações em ~2 dias · 492 fraudes (0,172%) |
| Desafio central | Dataset extremamente desbalanceado (~578 transações legítimas para cada fraude) |

## 🗂️ Estrutura do repositório

```
fraud-detection-global-trust/
├── data/
│   ├── raw/            # CSV bruto (não versionado — ver seção "Dados")
│   └── processed/      # banco SQLite gerado a partir do CSV
├── sql/
│   ├── create_table.sql   # schema documentado da tabela transactions
│   ├── load_data.py       # carrega o CSV para o SQLite
│   └── eda_queries.sql    # consultas de EDA e análise de negócio
├── src/
│   ├── data_prep.py       # carga e limpeza (duplicatas, nulos)
│   ├── features.py        # engenharia de variáveis (Hour, Amount_log)
│   └── train_model.py     # pipeline de treino/avaliação de todos os modelos
├── notebooks/
│   └── 01_eda_e_modelagem.ipynb   # EDA, estatística, SQL e storytelling completo
├── dashboard/
│   └── app.py              # dashboard interativo em Streamlit
├── models/                 # artefatos treinados (.joblib, gerados localmente)
├── outputs/
│   ├── metrics.json               # métricas de todos os modelos comparados
│   └── test_predictions.parquet   # predições do modelo principal no teste
├── reports/figures/         # gráficos exportados do notebook
├── requirements.txt
└── README.md
```

## 🧠 Por que essa arquitetura

O projeto foi organizado como um pipeline de dados real, não como um único
notebook monolítico:

- **`src/`** concentra a lógica reutilizável (limpeza, features, treino),
  compartilhada entre o notebook de análise e o dashboard — evita duplicar
  código e mantém a preparação de dados consistente em todos os pontos de
  uso.
- **`sql/`** demonstra a camada de consulta direta ao dado bruto, simulando
  como um analista de risco consultaria a base sem depender de Python.
- **`notebooks/`** é onde a exploração, os testes estatísticos e o
  storytelling final acontecem — consome os artefatos gerados por `src/`
  em vez de retreinar tudo a cada execução.
- **`dashboard/`** expõe os resultados para um público não-técnico
  (o "stakeholder"), incluindo um simulador interativo de threshold de
  decisão.

## 📥 Dados

A base usada é a clássica **Credit Card Fraud Detection** (transações
europeias, setembro/2013), disponibilizada pela EBAC para este módulo com
o nome `Base_M43_Pratique_CREDIT_CARD_FRAUD.csv`. O arquivo tem ~150 MB e
**não é versionado no Git** (ver `.gitignore`). Para reproduzir o projeto:

1. Baixe o CSV e salve como `data/raw/creditcard.csv`.
2. Rode `python sql/load_data.py` para gerar o banco SQLite.
3. Siga os passos de reprodução abaixo.

## ▶️ Como reproduzir

```bash
# 1. Criar ambiente e instalar dependências
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Carregar os dados no SQLite
python sql/load_data.py

# 3. Treinar e avaliar todos os modelos (gera outputs/metrics.json)
python src/train_model.py

# 4. Rodar o notebook de EDA + storytelling
jupyter notebook notebooks/01_eda_e_modelagem.ipynb

# 5. Rodar o dashboard interativo
streamlit run dashboard/app.py
```

## 🔍 Principais insights da EDA

- O desbalanceamento é extremo (0,172% de fraude) — **acurácia não é uma
  métrica válida** neste problema.
- A base tinha **1.081 linhas duplicadas**, removidas antes do split para
  evitar vazamento de dados.
- Fraudes têm **mediana de valor menor, mas média maior** que transações
  legítimas — indício de mistura entre "testes de cartão" (valores baixos)
  e golpes pontuais de alto valor.
- Componentes PCA como `V14`, `V4`, `V10` e `V12` mostram separação visual
  e **estatisticamente significativa** (Mann-Whitney U, p < 0,05) entre
  classes, mesmo sendo variáveis anônimas.
- A taxa de fraude é proporcionalmente mais alta na madrugada, quando o
  volume de transações legítimas é menor.
- Transações de **valor zero** (comumente usadas para validar se um
  cartão roubado está ativo) têm a maior taxa de fraude entre todas as
  faixas de valor analisadas (1,48%).

Detalhamento completo, com os 7 métodos de EDA e os testes estatísticos,
está no notebook `notebooks/01_eda_e_modelagem.ipynb`.

## 🤖 Modelagem

Modelos comparados (todos com split estratificado 70/30 e scaler ajustado
apenas no treino, sem vazamento de dados):

| Modelo | Recall | Precision | PR-AUC | Falsos Negativos |
|---|---|---|---|---|
| Dummy (baseline) | 0,0% | — | 0,002 | 142/142 |
| Regressão Logística | 88,7% | 5,1% | 0,691 | 16 |
| Random Forest | 74,7% | 88,3% | 0,808 | 36 |
| **XGBoost (scale_pos_weight)** | 76,1% | 90,0% | **0,815** | 34 |
| XGBoost (threshold otimizado p/ F2) | 78,2% | 84,1% | 0,815 | 31 |
| XGBoost + SMOTE | 76,1% | 78,3% | 0,794 | 34 |
| Isolation Forest (não-supervisionado) | 23,2% | 23,4% | 0,138 | 109 |

> Validação cruzada estratificada (5 folds) no XGBoost principal:
> PR-AUC médio de **0,843** (± 0,038).

**Modelo recomendado:** XGBoost com threshold de decisão otimizado para
F2-Score (≈0,147), por apresentar o menor custo financeiro simulado entre
todas as opções testadas — ver detalhamento na seção de storytelling do
notebook e no dashboard (aba "Simulador de Threshold").

> Os parâmetros de custo (R$500 por fraude não detectada, R$5 por falso
> alarme) são hipotéticos, documentados para transparência, e podem ser
> ajustados interativamente no dashboard.

## 📊 Dashboard

O dashboard (`streamlit run dashboard/app.py`) tem 4 seções:

1. **Visão Geral** — KPIs do período e modelo recomendado.
2. **Exploração dos Dados** — gráficos interativos e consultas SQL ao vivo.
3. **Comparação de Modelos** — métricas e importância de variáveis.
4. **Simulador de Threshold** — ajuste o limiar de decisão e os custos de
   FN/FP em tempo real e veja o impacto na matriz de confusão.

## ⚠️ Limitações conhecidas

- As variáveis `V1`-`V28` são componentes PCA anônimas — não é possível
  atribuir significado de negócio direto a elas, apenas à magnitude da
  sua importância no modelo.
- O período de coleta é de apenas 2 dias; um re-treino periódico seria
  necessário em produção para acompanhar mudanças de padrão de fraude.
- Os parâmetros de custo financeiro são estimativas hipotéticas, não
  valores reais fornecidos pelo banco.

## 🛠️ Stack utilizada

Python (pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, SciPy) ·
SQL (SQLite) · Streamlit + Plotly · Jupyter Notebook.

---

**Autora:** Yu · [GitHub](https://github.com/TalitaLuci)
