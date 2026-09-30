# Detecção de Fraude em Transações Bancárias — Global Trust Bank

Projeto final do curso **EBAC — Profissão: Cientista de Dados** (Módulo 43).
Simula o atendimento a um gestor de risco de um grande banco que precisa de
um modelo capaz de identificar fraudes em transações de cartão de crédito
o mais rápido possível, minimizando prejuízo financeiro e preservando a
experiência de clientes legítimos.

> Projeto conduzido de ponta a ponta: definição do problema, EDA estatística,
> consultas SQL, engenharia de features, modelagem comparativa, avaliação
> orientada a negócio e um dashboard interativo para exploração dos
> resultados.

## O problema de negócio

| Item | Definição |
|---|---|
| Problema de negócio | Identificar transações fraudulentas em tempo hábil, minimizando prejuízo financeiro e fricção com clientes legítimos |
| Tarefa de ML | Classificação binária supervisionada |
| Variável-alvo | `Class` (0 = legítima, 1 = fraude) |
| Volume analisado | 284.807 transações em ~2 dias · 492 fraudes (0,172%) |
| Desafio central | Dataset extremamente desbalanceado (~578 transações legítimas para cada fraude) |

## Estrutura do repositório

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
│   ├── xgboost_model.joblib        # modelo principal (scores brutos)
│   └── xgboost_calibrated.joblib   # mesmo modelo, com calibração isotônica
├── outputs/
│   ├── metrics.json               # métricas de todos os modelos comparados
│   ├── test_predictions.parquet   # predições do modelo principal no teste
│   └── shap_sample.parquet        # amostra escalada do teste, para SHAP no notebook
├── reports/figures/         # gráficos exportados do notebook
├── requirements.txt
└── README.md
```

## Por que essa arquitetura

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

## Dados

A base usada é a clássica **Credit Card Fraud Detection** (transações
europeias, setembro/2013), disponibilizada pela EBAC para este módulo com
o nome `Base_M43_Pratique_CREDIT_CARD_FRAUD.csv`. O arquivo tem ~150 MB e
**não é versionado no Git** (ver `.gitignore`). Para reproduzir o projeto:

1. Baixe o CSV e salve como `data/raw/creditcard.csv`.
2. Rode `python sql/load_data.py` para gerar o banco SQLite.
3. Siga os passos de reprodução abaixo.

## Como reproduzir

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

## Principais insights da EDA

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

## Modelagem

Split estratificado em **3 partes** — 60% treino / 15% validação / 25%
teste — com scaler ajustado apenas no treino, sem vazamento de dados. O
conjunto de **validação** existe especificamente para escolher o
threshold de decisão sem tocar no teste (ver caixa de metodologia
abaixo).

| Modelo | Recall | Precision | PR-AUC | Falsos Negativos | Falsos Positivos | Custo simulado |
|---|---|---|---|---|---|---|
| Dummy (baseline) | 0,0% | — | 0,002 | 118/118 | 0 | R$ 59.000 |
| Regressão Logística | 90,7% | 5,3% | 0,733 | 11 | 1.903 | R$ 15.015 |
| Random Forest | 78,0% | 86,8% | 0,815 | 26 | 14 | R$ 13.070 |
| XGBoost (parâmetros padrão) | 81,4% | 93,2% | 0,838 | 22 | 7 | R$ 11.035 |
| XGBoost tunado (RandomizedSearchCV, threshold 0,5) | 81,4% | 92,3% | **0,867** | 22 | 8 | R$ 11.040 |
| **XGBoost tunado + threshold ótimo por custo (RECOMENDADO)** | **89,0%** | 45,9% | 0,867 | 13 | 124 | **R$ 7.120** |
| ↳ mesma configuração, com probabilidade **calibrada** | 89,0% | 45,9% | 0,855 | 13 | 124 | R$ 7.120 |
| XGBoost + SMOTE | 81,4% | 88,1% | 0,842 | 22 | 13 | R$ 11.065 |
| Isolation Forest (não-supervisionado) | 29,7% | 29,2% | 0,212 | 83 | 85 | R$ 41.925 |

> Validação cruzada estratificada (5 folds): PR-AUC médio de **0,846**
> (± 0,027). Hiperparâmetros tunados via `RandomizedSearchCV`
> (scoring=PR-AUC, 3-fold, 10 combinações): `max_depth=5`,
> `learning_rate≈0,215`, `n_estimators≈271`, `subsample≈0,80`,
> `colsample_bytree≈0,64`, `min_child_weight=7`. Ver
> `outputs/metrics.json` para os valores exatos.

### Metodologia do threshold: por que um conjunto de validação separado

Em versões anteriores deste projeto, o threshold de decisão era
escolhido observando o desempenho no próprio conjunto de teste — uma
forma sutil de vazamento de informação, já que o número reportado como
"resultado no teste" já teria sido otimizado *para* aquele teste.

A versão atual corrige isso: o threshold é escolhido no conjunto de
**validação** (15% dos dados, nunca visto pelo teste), **minimizando
diretamente a fórmula de custo financeiro** (não um proxy estatístico
como F2-Score), e só então aplicado, intocado, ao teste.

**Resultado prático:** como o custo de deixar passar uma fraude (R$500)
é **100x maior** que o de um falso alarme (R$5), o threshold
matematicamente ótimo é bem mais agressivo (≈0,011 em vez de 0,5) — o
modelo passa a sinalizar muito mais transações como suspeitas (124
falsos positivos, vs. 7-8 nas outras configurações), mas em troca captura
quase 89% das fraudes e reduz o custo total simulado de ~R$11.000 para
**R$ 7.120**.

> ⚠️ Essa recomendação só é válida *se* a proporção de custo 100:1 entre
> FN e FP refletir a realidade do banco. Um volume de alertas 15x maior
> tem custo operacional real que não está sendo capturado pela fórmula
> simplificada — validar com o time de risco antes de adotar em produção.
> O dashboard permite testar outras proporções de custo interativamente.

### Calibração de probabilidade

O XGBoost gera *scores* que discriminam bem entre classes, mas não são
necessariamente probabilidades bem calibradas (comum em modelos de
árvore virem "superconfiantes"). Isso importa porque um threshold só é
comunicável ao stakeholder como "X% de chance de fraude" se a
probabilidade for real — e é exatamente o que o simulador de threshold
do dashboard assume.

Calibramos com **regressão isotônica** (`CalibratedClassifierCV` +
`FrozenEstimator`, para não re-treinar o XGBoost), ajustada no conjunto
de **validação**. Por ser uma transformação monotônica, a calibração
**não muda o ranking das transações** — recall, precisão e custo no
threshold ótimo permanecem idênticos (ver tabela acima); só o *valor* do
threshold muda, de **0,0112** (score bruto, sem significado direto) para
**0,0625** (6,25% de chance real de fraude, interpretável).

- **Brier Score no teste:** 0,000383 → 0,000367 (melhora após calibrar).
- O notebook (`notebooks/01_eda_e_modelagem.ipynb`) traz o *reliability
  diagram* comparando antes/depois, e o dashboard usa a probabilidade
  calibrada no simulador de threshold por padrão.
- Nota técnica: o PR-AUC muda marginalmente entre bruto e calibrado
  (0,867 → 0,855) porque a isotônica é monotônica *não-decrescente*, não
  *estritamente* monotônica — ela empata vários scores brutos num mesmo
  valor calibrado, o que afeta o cálculo do PR-AUC sem alterar a decisão
  no threshold escolhido.

### Explicabilidade (SHAP)

Além da importância por *gain* do XGBoost, o notebook inclui uma análise
via **SHAP** (`TreeExplainer`), mostrando tanto o impacto agregado de
cada variável quanto uma explicação pontual para uma fraude específica —
essencial para justificar decisões de bloqueio ao time de risco, mesmo
com as features sendo componentes PCA anônimas.

### Curva de Ganho Acumulado

Complementando a matriz de confusão, o notebook e o dashboard mostram
quantas fraudes seriam capturadas revisando-se apenas uma fração das
transações com maior score — uma visão mais acionável para dimensionar
a capacidade operacional do time de revisão manual.


## Dashboard

O dashboard (`streamlit run dashboard/app.py`) tem 4 seções:

1. **Visão Geral** — KPIs do período e modelo recomendado.
2. **Exploração dos Dados** — gráficos interativos e consultas SQL ao vivo.
3. **Comparação de Modelos** — métricas, importância de variáveis e curva de ganho.
4. **Simulador de Threshold** — ajuste o limiar de decisão e os custos de
   FN/FP em tempo real e veja o impacto na matriz de confusão.

## Limitações conhecidas

- **O threshold recomendado assume a proporção de custo 100:1 (FN:FP).**
  Ela é hipotética; se o custo operacional de revisar um alerta for
  maior na prática, ou se o time não tiver capacidade para o volume de
  alertas gerado, o threshold precisa ser recalibrado com números reais.
- As variáveis `V1`-`V28` são componentes PCA anônimas — não é possível
  atribuir significado de negócio direto a elas, apenas à magnitude da
  sua importância no modelo.
- O período de coleta é de apenas 2 dias; um re-treino periódico seria
  necessário em produção para acompanhar mudanças de padrão de fraude.
- A busca de hiperparâmetros foi limitada a 10 combinações (3-fold) por
  restrição de ambiente (CPU única); com mais capacidade computacional,
  uma busca mais ampla (ou bayesiana, via Optuna) tende a melhorar ainda
  mais o resultado.

## Stack utilizada

Python (pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, SciPy, SHAP) ·
SQL (SQLite) · Streamlit + Plotly · Jupyter Notebook.

---

**Autora:** Talita Luci · [GitHub](https://github.com/TalitaLuci)
