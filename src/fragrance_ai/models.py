"""Définition des modèles et d'un wrapper multi-label qui sait ignorer les labels manquants."""

import numpy as np  # Tableaux numériques.
from sklearn.base import clone  # Copie « vierge » d'un modèle, pour en entraîner un par famille olfactive.
from sklearn.ensemble import RandomForestClassifier  # Forêt aléatoire : modèle classique robuste.
from sklearn.impute import SimpleImputer  # Remplace les valeurs manquantes des descripteurs.
from sklearn.pipeline import make_pipeline  # Enchaîne prétraitement et modèle dans un seul objet.
from xgboost import XGBClassifier  # Gradient boosting : souvent le meilleur modèle sur données tabulaires.

from fragrance_ai import config  # Graine aléatoire.

MODEL_NAMES = ["random_forest", "xgboost"]  # Modèles « classiques » ; les réseaux de neurones sont dans neural.py.
TREE_MODELS = {"random_forest", "xgboost"}  # Modèles compatibles avec l'explication rapide SHAP TreeExplainer.


def build_estimator(name: str, pos_ratio: float):
    """Crée un classifieur binaire non entraîné.

    pos_ratio = proportion de positifs pour ce label ; sert à compenser le déséquilibre des classes.
    """
    if name == "random_forest":  # Forêt aléatoire.
        return make_pipeline(  # Pipeline = imputation puis modèle, appliqués dans le même ordre au test.
            SimpleImputer(strategy="median"),  # Remplace chaque NaN par la médiane de la colonne.
            RandomForestClassifier(  # La forêt elle-même.
                n_estimators=300,  # Nombre d'arbres : plus il y en a, plus la prédiction est stable.
                max_features="sqrt",  # Chaque nœud ne regarde que √p variables : arbres plus variés et entraînement rapide.
                min_samples_leaf=2,  # Une feuille doit contenir au moins 2 molécules : limite le sur-apprentissage.
                class_weight="balanced_subsample",  # Donne plus de poids à la classe rare (les positifs).
                n_jobs=-1,  # Utilise tous les cœurs du processeur.
                random_state=config.RANDOM_STATE,  # Résultats reproductibles.
            ),
        )
    if name == "xgboost":  # Gradient boosting.
        return XGBClassifier(  # XGBoost gère nativement les NaN, pas besoin d'imputation.
            n_estimators=400,  # Nombre d'arbres ajoutés successivement.
            learning_rate=0.05,  # Pas d'apprentissage : petit = plus lent mais plus précis.
            max_depth=6,  # Profondeur maximale de chaque arbre : contrôle la complexité.
            subsample=0.8,  # Chaque arbre voit 80 % des molécules : réduit le sur-apprentissage.
            colsample_bytree=0.3,  # Chaque arbre voit 30 % des variables : accélère et régularise.
            scale_pos_weight=(1 - pos_ratio) / max(pos_ratio, 1e-6),  # Rééquilibre positifs et négatifs.
            tree_method="hist",  # Algorithme par histogrammes : beaucoup plus rapide.
            eval_metric="logloss",  # Fonction suivie pendant l'entraînement.
            n_jobs=-1,  # Tous les cœurs.
            random_state=config.RANDOM_STATE,  # Reproductibilité.
        )
    raise ValueError(f"Modèle inconnu : {name}")  # Protège contre une faute de frappe.


class MultiLabelModel:
    """Un classifieur binaire par famille olfactive (approche « binary relevance »).

    Avantage clé : pour chaque famille, on n'entraîne que sur les molécules dont le label est connu,
    ce qui permet d'appliquer proprement la stratégie « drop ».
    """

    def __init__(self, model_name: str):
        self.model_name = model_name  # Nom du type de modèle (random_forest ou xgboost).
        self.estimators_ = []  # Liste des 12 classifieurs entraînés, un par label.

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiLabelModel":
        """Entraîne un classifieur par colonne de Y ; les NaN de Y sont ignorés."""
        self.estimators_ = []  # Réinitialise si on ré-entraîne le même objet.
        for j in range(Y.shape[1]):  # Boucle sur les 12 familles olfactives.
            known = ~np.isnan(Y[:, j])  # Lignes où le label j est connu (0 ou 1).
            y = Y[known, j].astype(int)  # Cible binaire restreinte à ces lignes.
            est = build_estimator(self.model_name, pos_ratio=y.mean())  # Nouveau modèle adapté au déséquilibre du label.
            est.fit(X[known], y)  # Entraînement sur les molécules dont le label est connu.
            self.estimators_.append(est)  # Mémorise le modèle entraîné.
        return self  # Convention scikit-learn : fit renvoie l'objet lui-même.

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilité de chaque famille pour chaque molécule : matrice (n_molécules, 12)."""
        return np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators_])  # Colonne 1 = proba d'être positif.

    def final_estimator(self, j: int):
        """Renvoie le modèle « nu » du label j (sans l'imputation), utile pour SHAP."""
        est = self.estimators_[j]  # Classifieur complet du label j.
        return est.steps[-1][1] if hasattr(est, "steps") else est  # Dernière étape du pipeline, ou le modèle lui-même.

    def transform_for_explainer(self, j: int, X: np.ndarray) -> np.ndarray:
        """Applique au label j les mêmes prétraitements que pendant l'entraînement."""
        est = self.estimators_[j]  # Classifieur complet du label j.
        if hasattr(est, "steps"):  # S'il s'agit d'un pipeline…
            for _, step in est.steps[:-1]:  # …on applique chaque étape sauf le modèle final.
                X = step.transform(X)  # Ex. : imputation des NaN par la médiane.
        return X  # Données prêtes à être expliquées.
