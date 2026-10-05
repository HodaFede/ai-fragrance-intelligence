"""Configuration centrale : tous les chemins, constantes et hyperparamètres au même endroit."""

from pathlib import Path  # Path manipule les chemins de fichiers de façon portable (Windows, Mac, Linux).

# --- Chemins du projet -------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]  # Remonte de config.py jusqu'à la racine du dépôt (src/fragrance_ai -> racine).
DATA_DIR = ROOT_DIR / "data"  # Dossier où l'on met en cache les CSV téléchargés depuis GitHub.
MODELS_DIR = ROOT_DIR / "models"  # Dossier où l'on sauvegarde le modèle entraîné et ses métadonnées.
REPORTS_DIR = ROOT_DIR / "reports"  # Dossier des résultats chiffrés (tableaux de comparaison, figures).
MLFLOW_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"  # Base SQLite locale où MLflow enregistre chaque expérience.

# --- Source des données ------------------------------------------------------

ODORNET_BASE_URL = (  # URL de base des fichiers bruts du dépôt officiel OdorNet (Nankai University).
    "https://raw.githubusercontent.com/NKU-DOIE/OdorNet/main/data/processed"  # Données sous licence CC BY 4.0.
)
SPLIT_FILES = {  # Associe chaque nom de split au fichier CSV officiel correspondant.
    "train": "dataset_train_aligned.csv",  # 70 % des molécules ; contient des labels non résolus (cellules vides).
    "val": "dataset_val_aligned.csv",  # 20 % ; tous les labels sont explicites (0 ou 1).
    "test": "dataset_test_aligned.csv",  # 10 % ; jeu de test final, jamais utilisé pour choisir un modèle.
}

SMILES_COL = "SMILES"  # Nom de la colonne qui contient la structure moléculaire au format SMILES.
LABEL_COLUMNS = [  # Les 12 familles olfactives d'OdorNet, dans l'ordre du fichier source.
    "animalic&ambery",  # Animal, ambré.
    "sweety&gourmand",  # Sucré, gourmand.
    "floral",  # Floral.
    "fruity&vegetable",  # Fruité, végétal.
    "pungent&disagreeable",  # Piquant, désagréable.
    "green&herbal",  # Vert, herbacé.
    "nutty",  # Noisette.
    "woody&mossy",  # Boisé, mousse.
    "resinous&balsamic",  # Résineux, balsamique.
    "cooked",  # Cuit, grillé.
    "odorless",  # Inodore.
    "spice",  # Épicé.
]
LABELS_FR = {  # Traductions affichées dans l'application, pour un public francophone.
    "animalic&ambery": "Animal / ambré",  # Libellé lisible de la famille animalic&ambery.
    "sweety&gourmand": "Sucré / gourmand",  # Libellé lisible de sweety&gourmand.
    "floral": "Floral",  # Libellé lisible de floral.
    "fruity&vegetable": "Fruité / végétal",  # Libellé lisible de fruity&vegetable.
    "pungent&disagreeable": "Piquant / désagréable",  # Libellé lisible de pungent&disagreeable.
    "green&herbal": "Vert / herbacé",  # Libellé lisible de green&herbal.
    "nutty": "Noisette",  # Libellé lisible de nutty.
    "woody&mossy": "Boisé / mousse",  # Libellé lisible de woody&mossy.
    "resinous&balsamic": "Résineux / balsamique",  # Libellé lisible de resinous&balsamic.
    "cooked": "Cuit / grillé",  # Libellé lisible de cooked.
    "odorless": "Inodore",  # Libellé lisible de odorless.
    "spice": "Épicé",  # Libellé lisible de spice.
}

# --- Gestion des labels manquants -------------------------------------------

POLICIES = ["drop", "union", "intersection"]  # Les trois stratégies proposées par les auteurs d'OdorNet.
# drop         : on ignore les cellules vides (la molécule n'entraîne pas ce label-là).
# union        : une cellule vide est considérée comme positive (1).
# intersection : une cellule vide est considérée comme négative (0).

# --- Représentation des molécules -------------------------------------------

MORGAN_RADIUS = 2  # Rayon des fingerprints de Morgan : chaque bit décrit un atome et ses voisins jusqu'à 2 liaisons (≈ ECFP4).
MORGAN_BITS = 2048  # Longueur du vecteur binaire ; 2048 limite les collisions entre sous-structures différentes.

# --- Reproductibilité --------------------------------------------------------

RANDOM_STATE = 42  # Graine aléatoire fixe : relancer l'entraînement donne exactement les mêmes résultats.

# --- Artefacts sauvegardés ---------------------------------------------------

MODEL_PATH = MODELS_DIR / "model.joblib"  # Fichier contenant le modèle déployé (un classifieur par famille).
METADATA_PATH = MODELS_DIR / "metadata.json"  # Seuils de décision, scores et informations sur le modèle déployé.
RESULTS_PATH = REPORTS_DIR / "results.csv"  # Tableau comparatif de toutes les expériences.

PAPER_BASELINE_MACRO_F1 = 0.42  # Macro-F1 de référence publié par Yang et al., Scientific Data (2026).
