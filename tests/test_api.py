"""Tests de bout en bout de l'API : nécessitent un modèle entraîné."""

import pytest  # Framework de tests.

from fragrance_ai import config  # Chemin du modèle.

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Modèle non entraîné")  # Ignore si pas de modèle.


@pytest.fixture(scope="module")  # Un seul client pour tous les tests du fichier.
def client():
    from fastapi.testclient import TestClient  # Client HTTP de test fourni par FastAPI.

    from api.main import app  # Application à tester.

    return TestClient(app)  # Simule des requêtes sans lancer de serveur.


def test_health(client):
    """Le service répond."""
    assert client.get("/health").json() == {"status": "ok"}  # Réponse attendue.


def test_predict_returns_twelve_families(client):
    """Une prédiction renvoie bien les 12 familles avec des probabilités valides."""
    body = client.post("/predict", json={"smiles": "O=Cc1ccc(O)c(OC)c1"}).json()  # Vanilline.
    assert len(body["families"]) == len(config.LABEL_COLUMNS)  # 12 familles.
    assert all(0.0 <= f["probability"] <= 1.0 for f in body["families"])  # Probabilités entre 0 et 1.
    assert body["explanations"]  # Au moins une explication fournie.


def test_invalid_smiles_gives_422(client):
    """Un SMILES invalide renvoie une erreur 422, pas un crash serveur."""
    assert client.post("/predict", json={"smiles": "xyz123"}).status_code == 422  # Code attendu.
