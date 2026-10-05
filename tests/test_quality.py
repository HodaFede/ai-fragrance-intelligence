"""Tests des contrôles qualité, des métriques et des réseaux de neurones."""

import numpy as np  # Tableaux numériques.
import pandas as pd  # Petits tableaux de test.
import pytest  # Framework de tests.

from fragrance_ai import config  # Noms des labels.
from fragrance_ai.data import check_leakage, validate  # Contrôles testés.
from fragrance_ai.evaluate import evaluate, tune_thresholds  # Métriques testées.
from fragrance_ai.neural import ATOM_FEATURES, NeuralModel, masked_bce, mol_to_graph  # Réseaux testés.


def frame(smiles: list[str]) -> pd.DataFrame:
    """Petit DataFrame au format OdorNet."""
    df = pd.DataFrame({config.SMILES_COL: smiles})  # Colonne SMILES.
    for label in config.LABEL_COLUMNS:  # Ajoute les 12 colonnes de labels…
        df[label] = 0.0  # …remplies de zéros.
    return df  # Tableau prêt.


def test_leakage_detects_stereoisomers():
    """Deux stéréoisomères d'une même molécule dans deux splits = fuite détectée."""
    splits = {"train": frame(["C[C@H](O)CC"]), "val": frame(["C[C@@H](O)CC"]), "test": frame(["CCO"])}  # Même squelette.
    with pytest.raises(ValueError, match="Fuite"):  # Le contrôle doit lever une erreur…
        check_leakage(splits)  # …sur cette configuration.


def test_validate_rejects_wrong_size():
    """Un split qui n'a pas la taille de la version publiée est refusé."""
    with pytest.raises(ValueError):  # Erreur attendue…
        validate(frame(["CCO"]), "test")  # …car 1 ligne au lieu de 890.


def test_perfect_predictions_give_f1_of_one():
    """Des probabilités parfaites donnent un macro-F1 de 1."""
    Y = np.tile([[1, 0], [0, 1]], (5, 6))  # 10 molécules, 12 labels alternés.
    result = evaluate(Y, Y.astype(float), tune_thresholds(Y, Y.astype(float)))  # Prédiction = vérité.
    assert result["macro_f1"] == pytest.approx(1.0)  # Score parfait.


def test_masked_loss_ignores_unknown_labels():
    """Changer une cible non observée ne change pas la perte."""
    logits = np.zeros((1, 2), dtype=np.float32)  # Une molécule, deux labels.
    observed = np.array([[1.0, 0.0]], dtype=np.float32)  # Le second label est inconnu.
    a = masked_bce(logits, np.array([[1.0, 0.0]]), observed, 1.0)  # Cible inconnue = 0…
    b = masked_bce(logits, np.array([[1.0, 1.0]]), observed, 1.0)  # …ou = 1.
    assert float(a) == pytest.approx(float(b))  # Même perte : la cellule inconnue est bien ignorée.


def test_graph_shape():
    """L'éthanol donne un graphe de 3 atomes décrits par ATOM_FEATURES nombres."""
    nodes, adj = mol_to_graph("CCO")  # Construit le graphe.
    assert nodes.shape == (3, ATOM_FEATURES) and adj.shape == (3, 3)  # Dimensions attendues.
    assert np.allclose(adj.sum(1), 1.0)  # Adjacence normalisée : chaque ligne somme à 1.


@pytest.mark.parametrize("kind", ["mlp", "gnn"])  # Même test pour les deux architectures.
def test_neural_models_train_and_predict(kind):
    """Les réseaux s'entraînent sur un mini-jeu (avec labels manquants) et renvoient des probabilités."""
    smiles = ["CCO", "CCCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC"] * 4  # 24 petites molécules.
    X = np.random.default_rng(0).normal(size=(24, 10)).astype(np.float32)  # Variables factices pour le MLP.
    Y = np.random.default_rng(1).integers(0, 2, size=(24, 12)).astype(float)  # Labels aléatoires.
    Y[0, 0] = np.nan  # Un label manquant, pour exercer la perte masquée.
    model = NeuralModel(kind, max_epochs=2, batch_size=8).fit(X, Y, smiles, X[:8], np.nan_to_num(Y[:8]), smiles[:8])
    proba = model.predict_proba(X, smiles)  # Prédiction.
    assert proba.shape == (24, 12) and np.all((proba >= 0) & (proba <= 1))  # 12 probabilités valides par molécule.


def test_applicability_domain_recognises_known_molecule():
    """Une molécule d'entraînement est reconnue comme « connue », un isomère proche non."""
    from fragrance_ai.domain import REFERENCE_PATH, ApplicabilityDomain  # Import local : nécessite la référence.

    if not REFERENCE_PATH.exists():  # Sans modèle entraîné, le test n'a pas d'objet.
        pytest.skip("Référence d'entraînement absente")
    domain = ApplicabilityDomain()  # Charge la référence.
    assert domain.nearest("O=Cc1ccc(O)c(OC)c1")["level"] == "connue"  # Vanilline : présente à l'entraînement.
    other = domain.nearest("COc1cc(O)cc(C=O)c1")  # Isomère de la vanilline, absent de l'entraînement.
    assert other["level"] != "connue" and 0 < other["similarity"] < 1  # Proche mais pas identique.


def test_catalog_formulas_are_correct():
    """Les formules brutes du catalogue correspondent aux valeurs connues des chimistes."""
    from fragrance_ai.catalog import CATALOG_BY_NAME, identity  # Catalogue testé.

    expected = {"Vanilline": "C₈H₈O₃", "Coumarine": "C₉H₆O₂", "Hédione": "C₁₃H₂₂O₃", "Linalol": "C₁₀H₁₈O"}  # Références.
    for name, formula in expected.items():  # Vérifie chaque molécule de référence.
        assert identity(CATALOG_BY_NAME[name]["smiles"])["formula"] == formula  # Même formule brute.
    assert all(identity(m["smiles"])["molar_mass"] > 0 for m in CATALOG_BY_NAME.values())  # Toutes les structures sont lisibles.
