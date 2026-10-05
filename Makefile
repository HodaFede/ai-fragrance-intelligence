# Raccourcis : « make install », « make train », etc.
install:   ## Installe les dépendances de développement et le package.
	pip install -r requirements-dev.txt && pip install -e .
train:     ## Entraîne et compare tous les modèles.
	python -m fragrance_ai.train
test:      ## Lance les tests.
	pytest -q
api:       ## Démarre l'API en local.
	uvicorn api.main:app --reload
app:       ## Démarre l'application en local.
	streamlit run app/streamlit_app.py
mlflow:    ## Ouvre l'interface MLflow pour explorer les expériences.
	mlflow ui --backend-store-uri sqlite:///mlflow.db
docker:    ## Lance API + application avec Docker.
	docker compose up --build
