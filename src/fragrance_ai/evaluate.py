"""Métriques d'évaluation et choix des seuils de décision."""

import numpy as np  # Tableaux numériques.
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score  # Métriques.

from fragrance_ai import config  # Noms des labels.


def tune_thresholds(Y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Cherche, pour chaque famille, le seuil de probabilité qui maximise le F1 sur la validation.

    Pourquoi : avec des classes rares, le seuil 0,5 est rarement optimal.
    Les seuils sont choisis sur la VALIDATION puis figés : le test reste totalement inédit.
    """
    grid = np.arange(0.05, 0.96, 0.01)  # Seuils candidats de 0,05 à 0,95 par pas de 0,01.
    thresholds = np.full(Y_true.shape[1], 0.5)  # Valeur par défaut : 0,5 pour chaque label.
    for j in range(Y_true.shape[1]):  # Boucle sur les 12 familles.
        scores = [f1_score(Y_true[:, j], proba[:, j] >= t, zero_division=0) for t in grid]  # F1 pour chaque seuil.
        thresholds[j] = grid[int(np.argmax(scores))]  # Garde le seuil qui donne le meilleur F1.
    return thresholds  # Un seuil par famille.


def evaluate(Y_true: np.ndarray, proba: np.ndarray, thresholds: np.ndarray) -> dict:
    """Calcule les métriques globales et le détail par famille olfactive."""
    Y_pred = (proba >= thresholds).astype(int)  # Applique les seuils : 1 si la proba dépasse le seuil du label.
    per_label = {}  # Diagnostic détaillé, famille par famille.
    for j, label in enumerate(config.LABEL_COLUMNS):  # Pour chacune des 12 familles…
        truth, pred, score = Y_true[:, j], Y_pred[:, j], proba[:, j]  # …vérité, décision et probabilité.
        both_classes = len(np.unique(truth)) == 2  # AUROC et PR-AUC n'ont de sens qu'avec des positifs et des négatifs.
        per_label[label] = {
            "f1": float(f1_score(truth, pred, zero_division=0)),  # Équilibre précision/rappel.
            "precision": float(precision_score(truth, pred, zero_division=0)),  # Part des prédictions positives justes.
            "recall": float(recall_score(truth, pred, zero_division=0)),  # Part des vrais positifs retrouvés.
            "pr_auc": float(average_precision_score(truth, score)) if both_classes else None,  # Qualité sur classe rare.
            "auroc": float(roc_auc_score(truth, score)) if both_classes else None,  # Qualité du classement.
            "threshold": float(thresholds[j]),  # Seuil appliqué.
            "support": int(truth.sum()),  # Nombre de vrais positifs dans ce jeu.
        }

    def macro(key: str) -> float:  # Petite fonction : moyenne d'une métrique sur les 12 familles.
        return float(np.nanmean([m[key] if m[key] is not None else np.nan for m in per_label.values()]))  # Ignore les None.

    return {  # Dictionnaire de résultats, facile à sauvegarder en JSON ou à logger dans MLflow.
        "macro_f1": macro("f1"),  # Chaque famille compte autant : métrique de l'article.
        "macro_precision": macro("precision"),  # Précision moyenne.
        "macro_recall": macro("recall"),  # Rappel moyen.
        "macro_pr_auc": macro("pr_auc"),  # PR-AUC moyenne, plus informative que l'AUROC sur classes rares.
        "macro_auroc": macro("auroc"),  # AUROC moyenne, indépendante du seuil.
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),  # F1 global.
        "macro_f1_at_0.5": float(f1_score(Y_true, proba >= 0.5, average="macro", zero_division=0)),  # Sans réglage de seuil.
        "per_label": per_label,  # Détail complet.
    }
