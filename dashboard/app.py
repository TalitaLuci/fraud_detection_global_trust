"""
Dashboard interativo — Detecção de Fraude em Transações Bancárias
Global Trust Bank | EBAC Projeto Final (Módulo 43)

Executar com:
    streamlit run dashboard/app.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

st.set_page_config(
    page_title="Detecção de Fraude — Global Trust Bank",
    page_icon="🏦",
    layout="wide",
)


@st.cache_data
def load_metrics():
    with open(ROOT / "outputs" / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    return pd.read_parquet(ROOT / "outputs" / "test_predictions.parquet")


@st.cache_data
def load_raw_for_eda():
    df = pd.read_csv(ROOT / "data" / "raw" / "creditcard.csv")
    df["Class"] = df["Class"].astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)
    return df


@st.cache_data
def query_sql(query: str) -> pd.DataFrame:
    conn = sqlite3.connect(ROOT / "data" / "processed" / "fraud_detection.db")
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


metrics = load_metrics()
predictions = load_predictions()
df = load_raw_for_eda()

st.title("🏦 Detecção de Fraude em Transações Bancárias")
st.caption("Global Trust Bank · Projeto Final EBAC — Profissão: Cientista de Dados")

tab_overview, tab_eda, tab_models, tab_simulator = st.tabs(
    ["📊 Visão Geral", "🔍 Exploração dos Dados", "🤖 Comparação de Modelos", "🎚️ Simulador de Threshold"]
)

# =====================================================================
with tab_overview:
    split_info = metrics["split_info"]
    total_transacoes = split_info["n_treino"] + split_info["n_teste"]
    total_fraudes = split_info["fraudes_treino"] + split_info["fraudes_teste"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transações analisadas", f"{total_transacoes:,}".replace(",", "."))
    col2.metric("Fraudes identificadas", f"{total_fraudes}")
    col3.metric("Taxa de fraude", f"{100*total_fraudes/total_transacoes:.3f}%")

    prejuizo = query_sql(
        "SELECT ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS v FROM transactions"
    ).iloc[0, 0]
    col4.metric("Prejuízo direto no período", f"R$ {prejuizo:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    st.divider()

    best_model_row = pd.DataFrame(metrics["resultados_modelos"])
    best_model_row = best_model_row.sort_values("custo_financeiro_simulado_R$").iloc[0]

    st.subheader("Modelo recomendado")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelo", best_model_row["model"])
    c2.metric("Recall (fraudes capturadas)", f"{best_model_row['recall']*100:.1f}%")
    c3.metric("Precision", f"{best_model_row['precision']*100:.1f}%")
    c4.metric("Custo simulado (teste)", f"R$ {best_model_row['custo_financeiro_simulado_R$']:,.2f}".replace(",", "."))

    st.info(
        "💡 **Leitura de negócio:** o modelo recomendado captura a maior parte das fraudes "
        "com o menor custo financeiro simulado (falsos negativos custam muito mais que falsos "
        "positivos neste cenário). Veja a aba **Simulador de Threshold** para ajustar esse "
        "trade-off interativamente."
    )

# =====================================================================
with tab_eda:
    st.subheader("Distribuição da variável-alvo")
    class_counts = df["Class"].value_counts().rename({0: "Legítima", 1: "Fraude"})
    fig_class = px.bar(
        x=class_counts.index, y=class_counts.values, log_y=True,
        labels={"x": "Classe", "y": "Nº de transações (log)"},
        color=class_counts.index, color_discrete_sequence=["#2E86AB", "#C73E1D"],
    )
    st.plotly_chart(fig_class, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Valor da transação por classe")
        fig_amount = px.box(
            df, x="Class", y="Amount", color="Class",
            labels={"Class": "Classe (0=legítima, 1=fraude)"},
            color_discrete_sequence=["#2E86AB", "#C73E1D"],
        )
        fig_amount.update_yaxes(range=[0, 500])  # foco na parte densa da distribuição
        st.plotly_chart(fig_amount, use_container_width=True)

    with col2:
        st.subheader("Taxa de fraude por hora do dia")
        hourly = query_sql("""
            SELECT
                CAST((Time / 3600) AS INT) % 24 AS hora_do_dia,
                SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraudes,
                COUNT(*) AS total_transacoes,
                ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS taxa_fraude_pct
            FROM transactions
            GROUP BY hora_do_dia
            ORDER BY hora_do_dia
        """)
        fig_hour = px.line(hourly, x="hora_do_dia", y="taxa_fraude_pct", markers=True,
                            labels={"hora_do_dia": "Hora do dia", "taxa_fraude_pct": "Taxa de fraude (%)"})
        fig_hour.update_traces(line_color="#C73E1D")
        st.plotly_chart(fig_hour, use_container_width=True)

    st.subheader("Faixas de valor x taxa de fraude (consulta SQL ao vivo)")
    faixas = query_sql("""
        SELECT
            CASE
                WHEN Amount = 0 THEN '00 - Zero'
                WHEN Amount <= 10 THEN '01 - Ate 10'
                WHEN Amount <= 50 THEN '02 - 10 a 50'
                WHEN Amount <= 100 THEN '03 - 50 a 100'
                WHEN Amount <= 500 THEN '04 - 100 a 500'
                WHEN Amount <= 2000 THEN '05 - 500 a 2000'
                ELSE '06 - Acima de 2000'
            END AS faixa_valor,
            COUNT(*) AS total_transacoes,
            SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) AS fraudes,
            ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS taxa_fraude_pct
        FROM transactions
        GROUP BY faixa_valor
        ORDER BY faixa_valor
    """)
    st.dataframe(faixas, use_container_width=True, hide_index=True)

# =====================================================================
with tab_models:
    st.subheader("Comparação entre os modelos treinados")
    models_df = pd.DataFrame(metrics["resultados_modelos"])
    display_cols = ["model", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc",
                     "false_negative", "false_positive", "custo_financeiro_simulado_R$"]
    st.dataframe(models_df[display_cols], use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_pr = px.bar(models_df, x="model", y="pr_auc", color="model",
                         title="PR-AUC por modelo (métrica principal)")
        fig_pr.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_pr, use_container_width=True)
    with col2:
        fig_cost = px.bar(models_df.sort_values("custo_financeiro_simulado_R$"),
                           x="model", y="custo_financeiro_simulado_R$", color="model",
                           title="Custo financeiro simulado por modelo (menor = melhor)")
        fig_cost.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_cost, use_container_width=True)

    st.subheader("Importância das variáveis — XGBoost")
    top_features = pd.Series(metrics["top_10_features_xgboost"]).sort_values()
    fig_imp = px.bar(x=top_features.values, y=top_features.index, orientation="h",
                      labels={"x": "Importância (gain)", "y": ""})
    st.plotly_chart(fig_imp, use_container_width=True)

    cv = metrics["validacao_cruzada_xgboost"]
    st.caption(
        f"Validação cruzada estratificada (5 folds) — PR-AUC médio: **{cv['pr_auc_mean']}** "
        f"(± {cv['pr_auc_std']}) · folds individuais: {cv['folds']}"
    )

# =====================================================================
with tab_simulator:
    st.subheader("Simulador de threshold de decisão")
    st.write(
        "Ajuste o limiar de probabilidade a partir do qual uma transação é sinalizada "
        "como fraude, e veja o impacto em tempo real nas métricas e no custo financeiro."
    )

    custo_fn = st.number_input("Custo de um Falso Negativo (fraude não detectada) — R$",
                                min_value=0.0, value=500.0, step=50.0)
    custo_fp = st.number_input("Custo de um Falso Positivo (cliente legítimo sinalizado) — R$",
                                min_value=0.0, value=5.0, step=1.0)
    threshold = st.slider("Threshold de decisão", min_value=0.0, max_value=1.0, value=0.15, step=0.01)

    y_true = predictions["Class_real"]
    y_proba = predictions["proba_fraude_xgb"]
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    custo_total = fn * custo_fn + fp * custo_fp

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recall", f"{recall*100:.1f}%")
    c2.metric("Precision", f"{precision*100:.1f}%")
    c3.metric("Falsos Negativos", fn)
    c4.metric("Falsos Positivos", fp)
    c5.metric("Custo total simulado", f"R$ {custo_total:,.2f}".replace(",", "."))

    fig_conf = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Previsto: Legítima", "Previsto: Fraude"],
        y=["Real: Legítima", "Real: Fraude"],
        text=[[tn, fp], [fn, tp]],
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
    ))
    fig_conf.update_layout(title="Matriz de Confusão no threshold selecionado")
    st.plotly_chart(fig_conf, use_container_width=True)

    st.caption(
        "💡 Reduza o threshold para capturar mais fraudes (maior recall), à custa de mais "
        "falsos alarmes. Aumente o threshold para reduzir a carga operacional de revisão, "
        "à custa de deixar mais fraudes passarem."
    )
