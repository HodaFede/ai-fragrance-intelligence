# Image Docker unique, utilisée à la fois pour l'API et pour l'application Streamlit.

# Python 3.12 en version allégée : image plus petite et plus rapide à déployer.
FROM python:3.12-slim

# Bibliothèques système nécessaires au dessin des molécules par RDKit.
RUN apt-get update && apt-get install -y --no-install-recommends libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Évite les fichiers .pyc et affiche les logs immédiatement.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Dossier de travail dans le conteneur.
WORKDIR /app

# Copie d'abord les dépendances seules : Docker les met en cache tant qu'elles ne changent pas.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie ensuite le code, le modèle entraîné et les résultats.
COPY pyproject.toml .
COPY src ./src
COPY api ./api
COPY app ./app
COPY models ./models
COPY reports ./reports
COPY .streamlit ./.streamlit

# Installe le package fragrance_ai.
RUN pip install --no-cache-dir --no-deps -e .

# Ports : 8000 pour l'API, 8501 pour Streamlit.
EXPOSE 8000 8501

# Commande par défaut : l'application web (docker-compose la remplace pour l'API).
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
