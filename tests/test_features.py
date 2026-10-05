"""Tests unitaires de la vectorisation : rapides, sans modèle entraîné."""

import numpy as np  # Tableaux numériques.
import pandas as pd  # Petits tableaux de labels.

from fragrance_ai.data import apply_policy  # Stratégies de labels.
from fragrance_ai.features import FEATURE_NAMES, canonicalize, featurize, featurize_one  # Fonctions testées.
from fragrance_ai import config  # Noms des labels.


def test_vector_has_expected_length():
    """Un SMILES valide donne un vecteur de la bonne taille."""
    vec = featurize_one("O=Cc1ccc(O)c(OC)c1")  # Vanilline.
    assert vec is not None and vec.shape == (len(FEATURE_NAMES),)  # Taille = descripteurs + 2048 bits.


def test_invalid_smiles_returns_none():
    """Un SMILES invalide est rejeté proprement, sans planter."""
    assert featurize_one("ceci n'est pas une molécule") is None  # Texte quelconque.
    assert featurize_one("") is None  # Chaîne vide.


def test_featurize_masks_invalid_rows():
    """La vectorisation par lot signale les molécules invalides sans décaler les autres."""
    X, valid = featurize(["CCO", "???", "CCC"])  # Une molécule invalide au milieu.
    assert valid.tolist() == [True, False, True]  # Le masque repère la mauvaise ligne.
    assert X.shape[0] == 2  # Seules les deux valides sont vectorisées.


def test_canonical_smiles_is_unique():
    """Deux écritures de l'éthanol donnent le même SMILES canonique."""
    assert canonicalize("OCC") == canonicalize("CCO")  # Même molécule, même écriture.


def test_policies():
    """Les trois stratégies transforment correctement une cellule vide."""
    labels = pd.DataFrame([[np.nan] + [0.0] * 11], columns=config.LABEL_COLUMNS)  # Un label manquant.
    assert np.isnan(apply_policy(labels, "drop")[0, 0])  # drop : reste manquant.
    assert apply_policy(labels, "union")[0, 0] == 1.0  # union : devient positif.
    assert apply_policy(labels, "intersection")[0, 0] == 0.0  # intersection : devient négatif.
