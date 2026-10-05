"""API REST : expose le modèle à n'importe quelle application (web, mobile, script).

Lancement : uvicorn api.main:app --reload
Documentation interactive générée automatiquement : http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException  # FastAPI crée l'API ; HTTPException renvoie une erreur HTTP propre.
from pydantic import BaseModel, Field  # Pydantic valide automatiquement le format des requêtes.

from fragrance_ai.predict import InvalidSmilesError, get_predictor  # Logique de prédiction partagée.

app = FastAPI(  # Création de l'application web.
    title="AI Fragrance Intelligence API",  # Titre affiché dans la documentation /docs.
    description="Prédiction du profil olfactif d'une molécule à partir de son SMILES.",  # Description de l'API.
    version="1.0.0",  # Version, à incrémenter à chaque changement de modèle.
)


class PredictRequest(BaseModel):
    """Format attendu du corps de la requête POST /predict."""

    smiles: str = Field(..., min_length=1, max_length=500, examples=["O=Cc1ccc(O)c(OC)c1"])  # Obligatoire, longueur bornée.


@app.get("/health")  # Route GET /health.
def health() -> dict:
    """Vérifie que le service répond : utilisé par Docker et les hébergeurs."""
    return {"status": "ok"}  # Réponse minimale.


@app.get("/model")  # Route GET /model.
def model_info() -> dict:
    """Informations sur le modèle déployé : type, seuils, performances."""
    return get_predictor().metadata  # Renvoie le contenu de metadata.json.


@app.post("/predict")  # Route POST /predict.
def predict(request: PredictRequest) -> dict:
    """Prédit le profil olfactif d'une molécule."""
    try:  # On tente la prédiction…
        return get_predictor().predict(request.smiles)  # …et on renvoie le résultat en JSON.
    except InvalidSmilesError as exc:  # Si le SMILES est invalide…
        raise HTTPException(status_code=422, detail=str(exc)) from exc  # …erreur 422 « données non traitables ».
