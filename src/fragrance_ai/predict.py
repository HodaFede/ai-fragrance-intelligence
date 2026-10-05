"""Service de prédiction : charge le modèle une fois et prédit le profil olfactif d'un SMILES."""

import json  # Lecture des métadonnées.
from functools import lru_cache  # Garde le prédicteur en mémoire : le modèle n'est chargé qu'une seule fois.

import joblib  # Chargement du modèle sauvegardé.
import numpy as np  # Tableaux numériques.

from fragrance_ai import config  # Chemins et libellés.
from fragrance_ai.domain import REFERENCE_PATH, ApplicabilityDomain  # Fiabilité propre à chaque molécule.
from fragrance_ai.explain import explain  # Explications SHAP.
from fragrance_ai.features import canonicalize, featurize_one  # Vectorisation d'une molécule.


class InvalidSmilesError(ValueError):
    """Erreur levée quand le SMILES fourni ne décrit pas une molécule valide."""


class Predictor:
    """Encapsule le modèle, ses seuils et la logique de prédiction."""

    def __init__(self):
        if not config.MODEL_PATH.exists():  # Si le modèle n'a pas encore été entraîné…
            raise FileNotFoundError("Modèle absent : lancez d'abord `python -m fragrance_ai.train`.")  # …message clair.
        self.model = joblib.load(config.MODEL_PATH)  # Charge les 12 classifieurs.
        self.metadata = json.loads(config.METADATA_PATH.read_text())  # Charge seuils et scores.
        self.thresholds = np.array([self.metadata["thresholds"][l] for l in config.LABEL_COLUMNS])  # Seuils dans l'ordre des labels.
        self.domain = ApplicabilityDomain() if REFERENCE_PATH.exists() else None  # Référence d'entraînement, si disponible.

    def predict(self, smiles: str, explain_top: int = 3) -> dict:
        """Prédit les familles olfactives et explique les plus probables."""
        x = featurize_one(smiles)  # Vecteur de variables de la molécule.
        if x is None:  # SMILES illisible…
            raise InvalidSmilesError(f"SMILES invalide : {smiles!r}")  # …erreur explicite pour l'utilisateur.
        proba = self.model.predict_proba(x.reshape(1, -1))[0]  # Probabilité pour chacune des 12 familles.
        families = [  # Une entrée par famille, triée ensuite par probabilité.
            {
                "label": label,  # Nom technique.
                "label_fr": config.LABELS_FR[label],  # Nom affiché.
                "probability": float(p),  # Probabilité prédite.
                "threshold": float(t),  # Seuil de décision de cette famille.
                "predicted": bool(p >= t),  # Vrai si la famille est retenue.
            }
            for label, p, t in zip(config.LABEL_COLUMNS, proba, self.thresholds)  # Parcourt les 12 familles.
        ]
        families.sort(key=lambda f: f["probability"] / f["threshold"], reverse=True)  # Classe par marge au-dessus du seuil.
        explanations = {}  # Explications des familles les plus probables.
        for fam in families[:explain_top]:  # Pour les explain_top premières familles…
            j = config.LABEL_COLUMNS.index(fam["label"])  # …on retrouve leur position…
            explanations[fam["label"]] = explain(self.model, x, smiles, j)  # …et on calcule leurs contributions SHAP.
        return {  # Réponse complète, sérialisable en JSON.
            "smiles": smiles,  # SMILES reçu.
            "canonical_smiles": canonicalize(smiles),  # Forme standard.
            "families": families,  # Profil olfactif.
            "explanations": explanations,  # Raisons des prédictions principales.
            "applicability": self.domain.nearest(smiles) if self.domain else None,  # Ressemblance avec les données apprises.
            "model": self.metadata["model"],  # Modèle utilisé, pour la transparence.
        }


@lru_cache(maxsize=1)  # Mémorise le résultat : un seul Predictor par processus.
def get_predictor() -> Predictor:
    """Point d'accès unique au prédicteur, partagé par l'API et l'application."""
    return Predictor()  # Créé au premier appel, réutilisé ensuite.
