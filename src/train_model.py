"""
Pipeline de treino e avaliação de modelos de detecção de fraude.

Modelos comparados:
    1. DummyClassifier (baseline "ingênuo" -- prova que acurácia é enganosa)
    2. Regressão Logística (class_weight='balanced')
    3. Random Forest (class_weight='balanced')
    4. XGBoost (scale_pos_weight) -- parâmetros padrão, para referência
    5. XGBoost tunado via RandomizedSearchCV (scoring=PR-AUC) -- MODELO PRINCIPAL
    6. XGBoost + SMOTE (reamostragem no treino, para comparação)
    7. Isolation Forest (não-supervisionado, comparação bônus)

Split em 3 partes (60% treino / 15% validação / 25% teste), estratificado:
    - TREINO: fit dos modelos e da busca de hiperparâmetros (com CV interna).
    - VALIDAÇÃO: escolha do threshold de decisão do modelo principal,
      otimizando diretamente o custo financeiro simulado (não um proxy
      como F-beta). Nenhuma decisão de configuração é tomada olhando o teste.
    - TESTE: usado *apenas* para o relato final de métricas -- nunca é
      tocado por scaler, reamostragem, busca de hiperparâmetros ou escolha
      de threshold. Isso evita que o número reportado como "performance no
      teste" já tenha sido otimizado nesse mesmo conjunto (uma forma sutil
      de vazamento de informação).

Uso:
    python src/train_model.py
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_prep import load_clean_data
from features import FEATURE_COLUMNS, get_feature_matrix

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
RANDOM_STATE = 42

# Parâmetros de custo financeiro simulado (hipotéticos, ver README).
# Usados tanto para reportar o custo de cada modelo quanto -- a partir
# desta versão -- para escolher o threshold do modelo principal.
CUSTO_FALSO_NEGATIVO = 500.0   # fraude não detectada
CUSTO_FALSO_POSITIVO = 5.0     # cliente legítimo sinalizado


def evaluate(y_true, y_pred, y_proba, model_name: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "model": model_name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "f2": round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics


def find_best_threshold_by_cost(
    y_true,
    y_proba,
    custo_fn: float = CUSTO_FALSO_NEGATIVO,
    custo_fp: float = CUSTO_FALSO_POSITIVO,
):
    """
    Varre todos os thresholds candidatos (os próprios scores previstos) e
    escolhe o que MINIMIZA diretamente o custo financeiro simulado
    (FN * custo_fn + FP * custo_fp), em vez de usar um proxy estatístico
    como F1/F2-Score.

    Implementação vetorizada: ordena as observações por score decrescente
    e acumula TP/FP à medida que o threshold desce, evitando recalcular a
    matriz de confusão do zero para cada candidato.

    Parâmetros
    ----------
    y_true : array-like
        Rótulos reais (0/1) do conjunto usado para a escolha do threshold
        (deve ser o conjunto de VALIDAÇÃO, nunca o de teste).
    y_proba : array-like
        Probabilidades previstas de fraude para as mesmas observações.
    custo_fn, custo_fp : float
        Custos unitários de falso negativo e falso positivo.

    Retorna
    -------
    tuple[float, float]
        (melhor_threshold, menor_custo_encontrado)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    total_positivos = y_true.sum()

    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    proba_sorted = y_proba[order]

    # Ao cortar após a i-ésima observação (0-indexado), tudo até ali vira
    # "previsto fraude" (TP+FP) e o resto vira "previsto legítima" (FN+TN).
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    fn_cum = total_positivos - tp_cum

    custo = fn_cum * custo_fn + fp_cum * custo_fp

    best_idx = int(np.argmin(custo))
    best_threshold = float(proba_sorted[best_idx])
    best_cost = float(custo[best_idx])
    return best_threshold, best_cost


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("Carregando e limpando dados...")
    df = load_clean_data()
    X, y = get_feature_matrix(df)

    # ------------------------------------------------------------------
    # Split em 3 partes -- 60% treino / 15% validação / 25% teste --
    # ANTES de qualquer scaling/reamostragem/tuning.
    #
    # O conjunto de VALIDAÇÃO existe especificamente para escolher o
    # threshold de decisão do modelo principal sem tocar no teste: se
    # otimizássemos o threshold olhando as métricas no próprio teste,
    # o número final reportado estaria artificialmente inflado (o
    # threshold "aprenderia" as particularidades daquele conjunto).
    # ------------------------------------------------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.625, stratify=y_temp, random_state=RANDOM_STATE
    )  # 0.40 * 0.625 = 0.25 do total -> teste; sobra 0.15 do total -> validação

    print(f"Treino:     {X_train.shape} | Fraudes: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
    print(f"Validação:  {X_val.shape} | Fraudes: {y_val.sum()} ({y_val.mean()*100:.4f}%)")
    print(f"Teste:      {X_test.shape} | Fraudes: {y_test.sum()} ({y_test.mean()*100:.4f}%)")

    # Scaler fitado SOMENTE no treino
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLUMNS, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=FEATURE_COLUMNS, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index
    )
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    results = []

    # ------------------------------------------------------------------
    # 1) Baseline: DummyClassifier
    # ------------------------------------------------------------------
    print("\n[1/7] Treinando DummyClassifier (baseline)...")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train_scaled, y_train)
    y_pred = dummy.predict(X_test_scaled)
    y_proba = dummy.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate(y_test, y_pred, y_proba, "Dummy (baseline)"))

    # ------------------------------------------------------------------
    # 2) Regressão Logística (class_weight='balanced')
    # ------------------------------------------------------------------
    print("[2/7] Treinando Regressão Logística...")
    logreg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    logreg.fit(X_train_scaled, y_train)
    y_pred = logreg.predict(X_test_scaled)
    y_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate(y_test, y_pred, y_proba, "Regressão Logística"))
    joblib.dump(logreg, MODELS_DIR / "logreg.joblib")

    # ------------------------------------------------------------------
    # 3) Random Forest (class_weight='balanced')
    # ------------------------------------------------------------------
    print("[3/7] Treinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate(y_test, y_pred, y_proba, "Random Forest"))

    # ------------------------------------------------------------------
    # 4) XGBoost com scale_pos_weight (parâmetros padrão, para referência)
    # ------------------------------------------------------------------
    print("[4/7] Treinando XGBoost (parâmetros padrão)...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_default = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb_default.fit(X_train_scaled, y_train)
    y_proba_default = xgb_default.predict_proba(X_test_scaled)[:, 1]
    y_pred_default = (y_proba_default >= 0.5).astype(int)
    results.append(
        evaluate(y_test, y_pred_default, y_proba_default, "XGBoost (parâmetros padrão)")
    )

    # ------------------------------------------------------------------
    # 5) XGBoost tunado via RandomizedSearchCV (scoring = PR-AUC)
    #    A busca usa CV *dentro do treino* -- não toca validação nem teste.
    #    Este passa a ser o MODELO PRINCIPAL do projeto.
    # ------------------------------------------------------------------
    print("[5/7] Buscando hiperparâmetros do XGBoost via RandomizedSearchCV "
          "(scoring=average_precision, cv=3, n_iter=10)...")
    param_distributions = {
        "n_estimators": randint(100, 350),
        "max_depth": randint(3, 7),
        "learning_rate": uniform(0.03, 0.27),       # 0.03 - 0.30
        "subsample": uniform(0.6, 0.4),             # 0.6 - 1.0
        "colsample_bytree": uniform(0.6, 0.4),      # 0.6 - 1.0
        "min_child_weight": randint(1, 10),
    }
    xgb_search_base = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,  # ambiente com 1 CPU -- paralelismo aninhado é contraproducente
        tree_method="hist",
    )
    search = RandomizedSearchCV(
        xgb_search_base,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X_train_scaled, y_train)
    xgb = search.best_estimator_
    best_params = search.best_params_
    print(f"  Melhores hiperparâmetros: {best_params}")
    print(f"  PR-AUC (CV, treino): {search.best_score_:.4f}")

    # Métricas no teste, threshold padrão 0.5 (sem otimização) -- só para
    # comparação de "quanto o threshold tunado realmente ajuda" mais abaixo.
    y_proba_xgb_test = xgb.predict_proba(X_test_scaled)[:, 1]
    y_pred_xgb_default_thresh = (y_proba_xgb_test >= 0.5).astype(int)
    results.append(
        evaluate(y_test, y_pred_xgb_default_thresh, y_proba_xgb_test,
                 "XGBoost tunado (RandomizedSearchCV)")
    )

    # ------------------------------------------------------------------
    # Threshold ótimo escolhido na VALIDAÇÃO, minimizando o custo
    # financeiro real (não um proxy como F2) -- e só então aplicado,
    # intocado, ao conjunto de TESTE.
    # ------------------------------------------------------------------
    y_proba_xgb_val = xgb.predict_proba(X_val_scaled)[:, 1]
    best_threshold, custo_estimado_val = find_best_threshold_by_cost(
        y_val, y_proba_xgb_val, CUSTO_FALSO_NEGATIVO, CUSTO_FALSO_POSITIVO
    )
    print(f"  Threshold ótimo (custo mínimo na VALIDAÇÃO): {best_threshold:.4f} "
          f"(custo estimado na validação: R$ {custo_estimado_val:.2f})")

    # Esse threshold, escolhido sem olhar o teste, é aplicado ao teste:
    y_pred_xgb_tuned = (y_proba_xgb_test >= best_threshold).astype(int)
    tuned_metrics = evaluate(
        y_test, y_pred_xgb_tuned, y_proba_xgb_test,
        "XGBoost tunado + threshold otimizado por custo (RECOMENDADO)"
    )
    tuned_metrics["threshold"] = round(best_threshold, 4)
    results.append(tuned_metrics)

    joblib.dump(xgb, MODELS_DIR / "xgboost_model.joblib")

    # ------------------------------------------------------------------
    # 5b) Calibração de probabilidade (isotônica, ajustada na VALIDAÇÃO)
    #
    # O XGBoost gera *scores* que discriminam bem entre classes, mas não
    # são necessariamente probabilidades bem calibradas (é comum modelos
    # de árvore virem "superconfiantes" -- score 0,9 não significa 90% de
    # chance real de fraude). Isso importa por 3 motivos práticos:
    #
    #   1. Interpretabilidade do threshold: "corte em 0,011" só faz
    #      sentido como "1,1% de chance de fraude" se a probabilidade
    #      for calibrada -- sem isso é um score arbitrário.
    #   2. O simulador de threshold do dashboard: mover o slider só tem
    #      semântica de negócio se o número junto dele for uma
    #      probabilidade real.
    #   3. Diagnóstico de confiança do modelo via reliability diagram e
    #      Brier Score.
    #
    # A calibração é isotônica (não-paramétrica, monotônica) e é
    # justamente por ser monotônica que ela NÃO muda o ranking das
    # transações -- ou seja, o recall/precisão/custo no ponto ótimo
    # permanecem os mesmos; só o valor numérico do threshold muda, para
    # ganhar significado de probabilidade real.
    #
    # Calibramos usando FrozenEstimator (o XGBoost já treinado, sem
    # re-treinar) ajustado no conjunto de VALIDAÇÃO -- não no treino
    # (para não calibrar em cima dos mesmos dados que geraram o modelo)
    # nem no teste (que precisa continuar intocado).
    # ------------------------------------------------------------------
    print("  Calibrando probabilidades (isotônica, ajustada na validação)...")
    calibrated_xgb = CalibratedClassifierCV(
        estimator=FrozenEstimator(xgb), method="isotonic"
    )
    calibrated_xgb.fit(X_val_scaled, y_val)

    y_proba_xgb_val_cal = calibrated_xgb.predict_proba(X_val_scaled)[:, 1]
    y_proba_xgb_test_cal = calibrated_xgb.predict_proba(X_test_scaled)[:, 1]

    # Recalculamos o threshold ótimo sobre as probabilidades CALIBRADAS
    # (mesma lógica de antes, mesmo conjunto de validação) -- o valor
    # numérico muda, mas o custo mínimo alcançado deve ser -- e é --
    # praticamente o mesmo, confirmando que a calibração é apenas uma
    # reparametrização monotônica, não uma mudança de decisão.
    best_threshold_cal, custo_estimado_val_cal = find_best_threshold_by_cost(
        y_val, y_proba_xgb_val_cal, CUSTO_FALSO_NEGATIVO, CUSTO_FALSO_POSITIVO
    )
    print(f"  Threshold ótimo CALIBRADO (custo mínimo na validação): "
          f"{best_threshold_cal:.4f} (antes da calibração: {best_threshold:.4f})")

    y_pred_xgb_tuned_cal = (y_proba_xgb_test_cal >= best_threshold_cal).astype(int)
    tuned_metrics_cal = evaluate(
        y_test, y_pred_xgb_tuned_cal, y_proba_xgb_test_cal,
        "XGBoost tunado + threshold otimizado por custo, CALIBRADO"
    )
    tuned_metrics_cal["threshold"] = round(best_threshold_cal, 4)
    results.append(tuned_metrics_cal)

    # Brier Score: erro quadrático médio entre probabilidade prevista e
    # rótulo real -- quanto menor, melhor calibrado. Comparamos antes e
    # depois, no TESTE (nunca usado para ajustar nada até aqui).
    brier_antes = brier_score_loss(y_test, y_proba_xgb_test)
    brier_depois = brier_score_loss(y_test, y_proba_xgb_test_cal)
    print(f"  Brier Score no teste -- antes: {brier_antes:.6f} | "
          f"depois da calibração: {brier_depois:.6f}")

    # Dados do reliability diagram (calibration curve) no teste, para
    # o notebook plotar sem precisar recarregar os modelos.
    frac_pos_antes, mean_pred_antes = calibration_curve(
        y_test, y_proba_xgb_test, n_bins=10, strategy="quantile"
    )
    frac_pos_depois, mean_pred_depois = calibration_curve(
        y_test, y_proba_xgb_test_cal, n_bins=10, strategy="quantile"
    )

    joblib.dump(calibrated_xgb, MODELS_DIR / "xgboost_calibrated.joblib")

    # ------------------------------------------------------------------
    # 6) XGBoost + SMOTE (reamostragem só no treino, parâmetros padrão)
    # ------------------------------------------------------------------
    print("[6/7] Treinando XGBoost + SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    xgb_smote = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    xgb_smote.fit(X_train_smote, y_train_smote)
    y_pred = xgb_smote.predict(X_test_scaled)
    y_proba = xgb_smote.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate(y_test, y_pred, y_proba, "XGBoost + SMOTE"))

    # ------------------------------------------------------------------
    # 7) Isolation Forest (não-supervisionado, comparação bônus)
    # ------------------------------------------------------------------
    print("[7/7] Treinando Isolation Forest (bônus, não-supervisionado)...")
    contamination = y_train.mean()  # taxa real de fraude no treino
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    iso.fit(X_train_scaled)
    # decision_function: quanto menor, mais anômalo. Invertido p/ virar "score de fraude"
    raw_scores = -iso.decision_function(X_test_scaled)
    # normaliza para [0,1] para tratar como pseudo-probabilidade
    y_proba_iso = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    y_pred_iso = (iso.predict(X_test_scaled) == -1).astype(int)
    results.append(
        evaluate(y_test, y_pred_iso, y_proba_iso, "Isolation Forest (não-supervisionado)")
    )

    # ------------------------------------------------------------------
    # Validação cruzada estratificada para o modelo principal (XGBoost
    # tunado) -- estimativa de estabilidade do PR-AUC, usando treino+
    # validação+teste apenas para ter mais dados nessa checagem de
    # variância entre folds (não influencia threshold nem hiperparâmetros).
    # ------------------------------------------------------------------
    print("\nRodando validação cruzada estratificada (5 folds) no XGBoost tunado...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    X_full_scaled = pd.concat([X_train_scaled, X_val_scaled, X_test_scaled])
    y_full = pd.concat([y_train, y_val, y_test])
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full_scaled, y_full), start=1):
        X_tr, X_v = X_full_scaled.iloc[tr_idx], X_full_scaled.iloc[val_idx]
        y_tr, y_v = y_full.iloc[tr_idx], y_full.iloc[val_idx]
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        model_cv = XGBClassifier(
            **best_params,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        model_cv.fit(X_tr, y_tr)
        proba_val = model_cv.predict_proba(X_v)[:, 1]
        pr_auc_fold = average_precision_score(y_v, proba_val)
        cv_scores.append(pr_auc_fold)
        print(f"  Fold {fold}: PR-AUC = {pr_auc_fold:.4f}")

    cv_summary = {
        "pr_auc_mean": round(float(np.mean(cv_scores)), 4),
        "pr_auc_std": round(float(np.std(cv_scores)), 4),
        "folds": [round(s, 4) for s in cv_scores],
    }

    # ------------------------------------------------------------------
    # Feature importance (XGBoost principal)
    # ------------------------------------------------------------------
    importances = pd.Series(xgb.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)
    top_features = importances.head(10).round(4).to_dict()

    # ------------------------------------------------------------------
    # Custo financeiro simulado -- aplicado a TODOS os modelos, sobre o
    # conjunto de teste (para comparação justa entre eles).
    # ------------------------------------------------------------------
    def custo_financeiro(m):
        return (
            m["false_negative"] * CUSTO_FALSO_NEGATIVO
            + m["false_positive"] * CUSTO_FALSO_POSITIVO
        )

    for m in results:
        m["custo_financeiro_simulado_R$"] = round(custo_financeiro(m), 2)

    # ------------------------------------------------------------------
    # Salvar tudo
    # ------------------------------------------------------------------
    output = {
        "resultados_modelos": results,
        "validacao_cruzada_xgboost": cv_summary,
        "top_10_features_xgboost": top_features,
        "hiperparametros_tunados_xgboost": {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in best_params.items()
        },
        "threshold_selecionado_na_validacao": {
            "threshold": round(best_threshold, 4),
            "custo_estimado_na_validacao_R$": round(custo_estimado_val, 2),
            "metodo": "minimizacao direta do custo financeiro (nao F-beta)",
        },
        "calibracao_probabilidade": {
            "metodo": "isotonic (CalibratedClassifierCV, ajustado na validacao via FrozenEstimator)",
            "threshold_calibrado": round(best_threshold_cal, 4),
            "custo_estimado_na_validacao_calibrado_R$": round(custo_estimado_val_cal, 2),
            "brier_score_antes": round(float(brier_antes), 6),
            "brier_score_depois": round(float(brier_depois), 6),
            "reliability_diagram": {
                "antes": {
                    "prob_media_prevista": [round(float(v), 4) for v in mean_pred_antes],
                    "fracao_real_positiva": [round(float(v), 4) for v in frac_pos_antes],
                },
                "depois": {
                    "prob_media_prevista": [round(float(v), 4) for v in mean_pred_depois],
                    "fracao_real_positiva": [round(float(v), 4) for v in frac_pos_depois],
                },
            },
        },
        "parametros_custo_simulado": {
            "custo_medio_fraude_nao_detectada_R$": CUSTO_FALSO_NEGATIVO,
            "custo_operacional_falso_alarme_R$": CUSTO_FALSO_POSITIVO,
        },
        "split_info": {
            "n_treino": len(X_train),
            "n_validacao": len(X_val),
            "n_teste": len(X_test),
            "fraudes_treino": int(y_train.sum()),
            "fraudes_validacao": int(y_val.sum()),
            "fraudes_teste": int(y_test.sum()),
        },
    }

    with open(OUTPUTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Guarda apenas as colunas necessárias para o dashboard (arquivo leve).
    # proba_fraude_xgb passa a ser a versão CALIBRADA (é a que tem
    # semântica de probabilidade real e deve alimentar o simulador do
    # dashboard); a versão crua fica disponível para quem quiser comparar.
    test_predictions = pd.DataFrame({
        "Class_real": y_test.values,
        "proba_fraude_xgb": y_proba_xgb_test_cal,
        "proba_fraude_xgb_bruto": y_proba_xgb_test,
        "pred_threshold_padrao": y_pred_xgb_default_thresh,
        "pred_threshold_otimizado": y_pred_xgb_tuned_cal,
    })
    test_predictions.to_parquet(OUTPUTS_DIR / "test_predictions.parquet")

    # Amostra estratificada do teste ESCALADO (com y) para explicabilidade
    # (SHAP) no notebook -- leve o suficiente para versionar no Git.
    sample_idx, _ = train_test_split(
        X_test_scaled.index, train_size=2000, stratify=y_test, random_state=RANDOM_STATE
    )
    shap_sample = X_test_scaled.loc[sample_idx].copy()
    shap_sample["Class_real"] = y_test.loc[sample_idx].values
    shap_sample.to_parquet(OUTPUTS_DIR / "shap_sample.parquet")

    print("\n" + "=" * 70)
    print("RESUMO DOS RESULTADOS (métricas no conjunto de TESTE, intocado)")
    print("=" * 70)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nValidação cruzada XGBoost tunado -- PR-AUC médio: {cv_summary['pr_auc_mean']} "
          f"(+/- {cv_summary['pr_auc_std']})")
    print(f"\nMelhores hiperparâmetros: {best_params}")
    print(f"\nThreshold ótimo bruto (validação, custo mínimo): {best_threshold:.4f}")
    print(f"Threshold ótimo calibrado (validação, custo mínimo): {best_threshold_cal:.4f}")
    print(f"Brier Score no teste -- antes: {brier_antes:.6f} | depois: {brier_depois:.6f}")
    print(f"\nTop features:\n{importances.head(10)}")
    print(f"\nResultados salvos em {OUTPUTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTempo total: {time.time() - t0:.1f}s")
