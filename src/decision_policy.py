"""
Política de decisão em dois thresholds para o modelo de detecção de fraude.

Um único threshold trata a decisão como binária (bloquear ou liberar), mas
isso ignora que, na prática, o time de risco tem uma terceira opção:
mandar a transação para revisão manual quando o modelo está "em dúvida".

Este módulo generaliza a otimização de threshold (ver
find_best_threshold_by_cost em train_model.py) para DUAS variáveis de
decisão:

    score < t_low             -> LIBERA automaticamente
    t_low <= score < t_high   -> REVISÃO MANUAL (assumida perfeita: o
                                  humano acerta o rótulo real, ao custo
                                  fixo de custo_revisao por transação)
    score >= t_high           -> BLOQUEIA automaticamente

Isso só é interessante quando o custo de bloquear automaticamente um
cliente legítimo (custo_bloqueio) é MAIOR que o custo de revisar
manualmente (custo_revisao) -- caso contrário, é sempre mais barato
bloquear direto do que pagar por uma revisão, e a zona de revisão
colapsa (t_low = t_high), reduzindo a política a um único threshold.
Esse colapso não é um bug: é uma conclusão legítima da otimização sob as
premissas de custo fornecidas, e vale reportar como tal.
"""
import numpy as np


def otimizar_politica_dois_thresholds(
    y_true,
    y_proba,
    custo_fn: float,
    custo_bloqueio: float,
    custo_revisao: float,
    n_candidatos: int = 250,
):
    """
    Busca (t_low, t_high) que minimizam o custo total esperado de uma
    política de 3 zonas (libera / revisa / bloqueia).

    Parâmetros
    ----------
    y_true : array-like
        Rótulos reais (0/1) do conjunto usado para a otimização (deve
        ser o conjunto de VALIDAÇÃO, nunca o de teste).
    y_proba : array-like
        Probabilidades previstas (idealmente calibradas) de fraude.
    custo_fn : float
        Custo de uma fraude liberada automaticamente sem revisão.
    custo_bloqueio : float
        Custo de bloquear automaticamente uma transação legítima
        (sem revisão humana). Fraudes bloqueadas automaticamente têm
        custo 0 (foram corretamente barradas).
    custo_revisao : float
        Custo fixo de uma revisão manual, cobrado por transação na
        zona cinzenta, independentemente do rótulo real (assume-se que
        a revisão humana sempre acerta, então não há custo residual de
        erro nessa zona).
    n_candidatos : int
        Número de thresholds candidatos (quantis da distribuição de
        scores) testados para cada uma das duas variáveis de decisão.
        250 candidatos já cobre bem o espaço sem custo computacional
        proibitivo (a busca é O(n_candidatos^2)).

    Retorna
    -------
    dict com t_low, t_high, custo_total, e a contagem de transações em
    cada zona (liberadas / revisão / bloqueadas).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)

    order = np.argsort(y_proba)  # ascendente
    y_sorted = y_true[order]
    proba_sorted = y_proba[order]

    # Cumulativo de fraudes (para custo de liberar) e de legítimas (para
    # custo de bloquear), na ordem crescente de score.
    fraude_cum = np.cumsum(y_sorted)          # fraude_cum[i] = fraudes em [0, i]
    legit_cum = np.cumsum(1 - y_sorted)       # legit_cum[i] = legítimas em [0, i]
    total_fraudes = fraude_cum[-1] if n > 0 else 0
    total_legit = legit_cum[-1] if n > 0 else 0

    # Candidatos: quantis da distribuição de score, incluindo os extremos
    # 0 (ninguém liberado / todo mundo revisado ou bloqueado) e n
    # (ninguém bloqueado).
    idx_candidatos = np.unique(
        np.linspace(0, n, n_candidatos, dtype=int).clip(0, n)
    )

    def custo_ate(i: int) -> float:
        """Custo de liberar automaticamente tudo em [0, i) (índices ordenados)."""
        if i <= 0:
            return 0.0
        return float(fraude_cum[i - 1]) * custo_fn

    def custo_apartir(j: int) -> float:
        """Custo de bloquear automaticamente tudo em [j, n) (índices ordenados)."""
        if j >= n:
            return 0.0
        legit_no_bloqueio = total_legit - (legit_cum[j - 1] if j > 0 else 0)
        return float(legit_no_bloqueio) * custo_bloqueio

    melhor = None
    for i in idx_candidatos:
        for j in idx_candidatos:
            if j < i:
                continue
            custo_liberado = custo_ate(i)
            custo_bloqueado = custo_apartir(j)
            custo_revisado = (j - i) * custo_revisao
            custo_total = custo_liberado + custo_revisado + custo_bloqueado
            if melhor is None or custo_total < melhor["custo_total"]:
                melhor = {
                    "i": int(i),
                    "j": int(j),
                    "custo_total": float(custo_total),
                }

    i, j = melhor["i"], melhor["j"]
    t_low = float(proba_sorted[i]) if i < n else 1.0
    t_high = float(proba_sorted[j]) if j < n else 1.0

    return {
        "t_low": round(t_low, 4),
        "t_high": round(t_high, 4),
        "custo_total_R$": round(melhor["custo_total"], 2),
        "n_liberados": int(i),
        "n_revisados": int(j - i),
        "n_bloqueados": int(n - j),
        "pct_liberados": round(100 * i / n, 2),
        "pct_revisados": round(100 * (j - i) / n, 2),
        "pct_bloqueados": round(100 * (n - j) / n, 2),
        "zona_de_revisao_colapsou": bool(i == j),
    }


def aplicar_politica(y_true, y_proba, t_low: float, t_high: float,
                      custo_fn: float, custo_bloqueio: float, custo_revisao: float):
    """
    Aplica uma política já definida (t_low, t_high) a um conjunto de dados
    (tipicamente o TESTE, intocado durante a otimização) e retorna as
    métricas resultantes: custo total e composição das 3 zonas, incluindo
    quantas fraudes/legítimas caem em cada uma.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    zona_libera = y_proba < t_low
    zona_bloqueia = y_proba >= t_high
    zona_revisa = ~zona_libera & ~zona_bloqueia

    fn_liberados = int((zona_libera & (y_true == 1)).sum())
    fp_bloqueados = int((zona_bloqueia & (y_true == 0)).sum())
    tp_bloqueados = int((zona_bloqueia & (y_true == 1)).sum())
    tn_liberados = int((zona_libera & (y_true == 0)).sum())
    fraudes_revisao = int((zona_revisa & (y_true == 1)).sum())
    legit_revisao = int((zona_revisa & (y_true == 0)).sum())

    custo_total = (
        fn_liberados * custo_fn
        + fp_bloqueados * custo_bloqueio
        + int(zona_revisa.sum()) * custo_revisao
    )

    # Fraudes "capturadas" pela política = bloqueadas automaticamente OU
    # identificadas na revisão manual (que assumimos sempre resolve
    # corretamente). Só as liberadas automaticamente (zona_libera) são
    # de fato perdidas.
    fraudes_capturadas_total = tp_bloqueados + fraudes_revisao
    total_fraudes = fraudes_capturadas_total + fn_liberados

    return {
        "t_low": round(t_low, 4),
        "t_high": round(t_high, 4),
        "n_total": int(len(y_true)),
        "liberados": {"n": int(zona_libera.sum()), "fraudes_perdidas": fn_liberados, "legitimas": tn_liberados},
        "revisao_manual": {"n": int(zona_revisa.sum()), "fraudes": fraudes_revisao, "legitimas": legit_revisao},
        "bloqueados": {"n": int(zona_bloqueia.sum()), "fraudes_capturadas": tp_bloqueados, "legitimas_bloqueadas": fp_bloqueados},
        "custo_total_R$": round(float(custo_total), 2),
        "recall_liquido": round(fraudes_capturadas_total / total_fraudes, 4) if total_fraudes > 0 else 0.0,
    }
