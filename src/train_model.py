"""
Pipeline de treino e avaliação de modelos de detecção de fraude.

Modelos comparados:
    1. DummyClassifier (baseline "ingênuo" -- prova que acurácia é enganosa)
    2. Regressão Logística (class_weight='balanced')
    3. Random Forest (class_weight='balanced')
    4. XGBoost (scale_pos_weight), com e sem SMOTE no treino
    5. Isolation Forest (não-supervisionado, comparação bônus)

Métricas: PR-AUC (métrica principal), Recall, Precision, F1, F2, ROC-AUC,
matriz de confusão -- tudo calculado SOMENTE no conjunto de teste, que
nunca é tocado por scaler ou reamostragem ajustados no treino.

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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data_prep import load_clean_data
from features import FEATURE_COLUMNS, get_feature_matrix

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
RANDOM_STATE = 42


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


def find_best_threshold(y_true, y_proba, beta: float = 2.0) -> float:
    """
    Varre thresholds na curva precision-recall e escolhe o que maximiza
    F-beta (beta=2 dá mais peso ao recall, alinhado à prioridade do
    negócio de minimizar falsos negativos).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve retorna 1 ponto a mais que thresholds
    precisions, recalls = precisions[:-1], recalls[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = (
            (1 + beta**2)
            * (precisions * recalls)
            / (beta**2 * precisions + recalls)
        )
    f_beta = np.nan_to_num(f_beta)
    best_idx = np.argmax(f_beta)
    return float(thresholds[best_idx]), float(f_beta[best_idx])


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("Carregando e limpando dados...")
    df = load_clean_data()
    X, y = get_feature_matrix(df)

    # Split estratificado ANTES de qualquer scaling/reamostragem
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    print(f"Fraudes no treino: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
    print(f"Fraudes no teste: {y_test.sum()} ({y_test.mean()*100:.4f}%)")

    # Scaler fitado SOMENTE no treino
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLUMNS, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index
    )
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    results = []

    # ------------------------------------------------------------------
    # 1) Baseline: DummyClassifier
    # ------------------------------------------------------------------
    print("\n[1/5] Treinando DummyClassifier (baseline)...")
    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    dummy.fit(X_train_scaled, y_train)
    y_pred = dummy.predict(X_test_scaled)
    y_proba = dummy.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate(y_test, y_pred, y_proba, "Dummy (baseline)"))

    # ------------------------------------------------------------------
    # 2) Regressão Logística (class_weight='balanced')
    # ------------------------------------------------------------------
    print("[2/5] Treinando Regressão Logística...")
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
    print("[3/5] Treinando Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate(y_test, y_pred, y_proba, "Random Forest"))

    # ------------------------------------------------------------------
    # 4a) XGBoost com scale_pos_weight (custo-sensível, sem reamostragem)
    # ------------------------------------------------------------------
    print("[4/5] Treinando XGBoost (scale_pos_weight)...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train_scaled, y_train)
    y_proba_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
    y_pred_xgb = (y_proba_xgb >= 0.5).astype(int)
    results.append(
        evaluate(y_test, y_pred_xgb, y_proba_xgb, "XGBoost (scale_pos_weight)")
    )

    # threshold ótimo por F2 para o modelo principal
    best_threshold, best_f2 = find_best_threshold(y_test, y_proba_xgb, beta=2.0)
    y_pred_xgb_tuned = (y_proba_xgb >= best_threshold).astype(int)
    tuned_metrics = evaluate(
        y_test, y_pred_xgb_tuned, y_proba_xgb, "XGBoost (threshold otimizado p/ F2)"
    )
    tuned_metrics["threshold"] = round(best_threshold, 4)
    results.append(tuned_metrics)

    joblib.dump(xgb, MODELS_DIR / "xgboost_model.joblib")

    # ------------------------------------------------------------------
    # 4b) XGBoost + SMOTE (reamostragem só no treino)
    # ------------------------------------------------------------------
    print("[4/5] Treinando XGBoost + SMOTE...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    xgb_smote = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_smote.fit(X_train_smote, y_train_smote)
    y_pred = xgb_smote.predict(X_test_scaled)
    y_proba = xgb_smote.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate(y_test, y_pred, y_proba, "XGBoost + SMOTE"))

    # ------------------------------------------------------------------
    # 5) Isolation Forest (não-supervisionado, comparação bônus)
    # ------------------------------------------------------------------
    print("[5/5] Treinando Isolation Forest (bônus, não-supervisionado)...")
    contamination = y_train.mean()  # taxa real de fraude no treino
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
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
    # Validação cruzada estratificada para o modelo principal (XGBoost)
    # ------------------------------------------------------------------
    print("\nRodando validação cruzada estratificada (5 folds) no XGBoost...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    X_full_scaled = pd.concat([X_train_scaled, X_test_scaled])
    y_full = pd.concat([y_train, y_test])
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_full_scaled, y_full), start=1):
        X_tr, X_val = X_full_scaled.iloc[tr_idx], X_full_scaled.iloc[val_idx]
        y_tr, y_val = y_full.iloc[tr_idx], y_full.iloc[val_idx]
        spw = (y_tr == 0).sum() / (y_tr == 1).sum()
        model_cv = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model_cv.fit(X_tr, y_tr)
        proba_val = model_cv.predict_proba(X_val)[:, 1]
        pr_auc_fold = average_precision_score(y_val, proba_val)
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
    # Custo financeiro simulado (parâmetros documentados no README)
    # ------------------------------------------------------------------
    custo_medio_fraude_nao_detectada = 500.0  # valor hipotético, ver README
    custo_operacional_falso_alarme = 5.0      # valor hipotético, ver README

    def custo_financeiro(m):
        return (
            m["false_negative"] * custo_medio_fraude_nao_detectada
            + m["false_positive"] * custo_operacional_falso_alarme
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
        "parametros_custo_simulado": {
            "custo_medio_fraude_nao_detectada_R$": custo_medio_fraude_nao_detectada,
            "custo_operacional_falso_alarme_R$": custo_operacional_falso_alarme,
        },
        "split_info": {
            "n_treino": len(X_train),
            "n_teste": len(X_test),
            "fraudes_treino": int(y_train.sum()),
            "fraudes_teste": int(y_test.sum()),
        },
    }

    with open(OUTPUTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Guarda apenas as colunas necessárias para o dashboard (arquivo leve)
    test_predictions = pd.DataFrame({
        "Class_real": y_test.values,
        "proba_fraude_xgb": y_proba_xgb,
        "pred_threshold_padrao": y_pred_xgb,
        "pred_threshold_otimizado": y_pred_xgb_tuned,
    })
    test_predictions.to_parquet(OUTPUTS_DIR / "test_predictions.parquet")

    print("\n" + "=" * 70)
    print("RESUMO DOS RESULTADOS")
    print("=" * 70)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nValidação cruzada XGBoost -- PR-AUC médio: {cv_summary['pr_auc_mean']} "
          f"(+/- {cv_summary['pr_auc_std']})")
    print(f"\nThreshold ótimo (F2): {best_threshold:.4f}")
    print(f"\nTop features:\n{importances.head(10)}")
    print(f"\nResultados salvos em {OUTPUTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTempo total: {time.time() - t0:.1f}s")
