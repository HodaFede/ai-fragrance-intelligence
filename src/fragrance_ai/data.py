"""Téléchargement, contrôle qualité et préparation des données OdorNet."""

import hashlib  # Calcule l'empreinte SHA-256 d'un fichier : prouve quelle version exacte a été utilisée.
import json  # Écrit le manifeste de provenance.
import urllib.request  # Module standard pour télécharger un fichier, sans dépendance externe.

import numpy as np  # Calcul numérique : tableaux et valeurs manquantes (np.nan).
import pandas as pd  # Manipulation des tableaux de données (DataFrame).
from rdkit import Chem, RDLogger  # Lecture des molécules, pour le contrôle anti-fuite.

from fragrance_ai import config  # Constantes du projet (chemins, colonnes, URL).

RDLogger.DisableLog("rdApp.*")  # Coupe les avertissements RDKit pendant les contrôles.

EXPECTED_ROWS = {"train": 6224, "val": 1778, "test": 890}  # Tailles de la version publiée : détecte un changement en amont.
MANIFEST_PATH = config.DATA_DIR / "manifest.json"  # Fichier qui trace l'URL et l'empreinte de chaque CSV.


def download_odornet(force: bool = False) -> None:
    """Télécharge les trois CSV officiels dans data/ et écrit le manifeste de provenance."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)  # Crée le dossier data/ s'il n'existe pas encore.
    manifest = {}  # Contiendra, pour chaque split, l'URL source, la taille et l'empreinte.
    for split, filename in config.SPLIT_FILES.items():  # Parcourt les fichiers train, val et test.
        target = config.DATA_DIR / filename  # Chemin local où le fichier sera enregistré.
        url = f"{config.ODORNET_BASE_URL}/{filename}"  # URL complète du fichier sur GitHub.
        if force or not target.exists():  # On ne télécharge que si nécessaire.
            print(f"Téléchargement de {url}")  # Informe l'utilisateur.
            with urllib.request.urlopen(url, timeout=90) as response:  # Ouvre la connexion (90 s maximum).
                payload = response.read()  # Récupère le contenu brut.
            if not payload.startswith(b"SMILES,"):  # Si l'on reçoit une page d'erreur HTML au lieu d'un CSV…
                raise ValueError(f"Contenu inattendu pour {filename} : ce n'est pas le CSV OdorNet.")  # …on refuse.
            target.write_bytes(payload)  # Écrit le fichier seulement s'il est valide.
        content = target.read_bytes()  # Relit le fichier présent sur le disque.
        manifest[split] = {  # Trace de provenance pour ce split.
            "url": url,  # D'où vient le fichier.
            "bytes": len(content),  # Sa taille.
            "sha256": hashlib.sha256(content).hexdigest(),  # Son empreinte unique.
        }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))  # Sauvegarde le manifeste.


def validate(df: pd.DataFrame, split: str) -> None:
    """Vérifie qu'un split a exactement la structure attendue, sinon s'arrête."""
    expected_cols = [config.SMILES_COL, *config.LABEL_COLUMNS]  # SMILES + 12 labels, dans cet ordre.
    if list(df.columns) != expected_cols:  # Colonnes absentes, renommées ou déplacées…
        raise ValueError(f"[{split}] colonnes inattendues : {list(df.columns)}")  # …le modèle serait faux.
    if len(df) != EXPECTED_ROWS[split]:  # Nombre de lignes différent de la version publiée…
        raise ValueError(f"[{split}] {len(df)} lignes au lieu de {EXPECTED_ROWS[split]} : version amont modifiée ?")
    if df[config.SMILES_COL].isna().any():  # Une molécule sans structure ne peut pas être vectorisée.
        raise ValueError(f"[{split}] SMILES manquant")  # Erreur explicite.
    for label in config.LABEL_COLUMNS:  # Chaque label doit valoir 0, 1 ou être vide.
        values = set(df[label].dropna().unique())  # Valeurs présentes, en ignorant les vides.
        if not values <= {0.0, 1.0}:  # Toute autre valeur trahit une corruption.
            raise ValueError(f"[{split}] valeurs inattendues pour {label} : {values}")
    if split != "train" and df[config.LABEL_COLUMNS].isna().any().any():  # Validation et test doivent être complets.
        raise ValueError(f"[{split}] labels manquants alors que ce split doit être complet")


def load_split(split: str) -> pd.DataFrame:
    """Charge un split ('train', 'val' ou 'test') et contrôle sa structure."""
    if split not in config.SPLIT_FILES:  # Refuse un nom de split inconnu…
        raise ValueError(f"Split inconnu : {split}")  # …avec un message d'erreur explicite.
    download_odornet()  # Garantit que les fichiers sont disponibles localement.
    df = pd.read_csv(config.DATA_DIR / config.SPLIT_FILES[split])  # Lit le CSV en DataFrame.
    validate(df, split)  # Contrôle qualité avant toute utilisation.
    return df.reset_index(drop=True)  # Index propre de 0 à n-1.


def connectivity_key(smiles: str) -> str:
    """Clé chimique sans stéréochimie : deux écritures d'une même molécule donnent la même clé."""
    mol = Chem.MolFromSmiles(smiles)  # Lit la molécule.
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else smiles  # SMILES canonique sans stéréo.


def check_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Vérifie qu'aucune molécule (ni son stéréoisomère) n'apparaît dans deux splits.

    Une fuite gonflerait artificiellement le score de test : le modèle « reconnaîtrait » des molécules.
    """
    keys = {name: set(df[config.SMILES_COL].map(connectivity_key)) for name, df in splits.items()}  # Clés par split.
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:  # Toutes les paires de splits.
        overlap = keys[a] & keys[b]  # Molécules communes aux deux.
        if overlap:  # S'il y en a…
            raise ValueError(f"Fuite chimique {a}/{b} : {len(overlap)} molécules en commun")  # …le protocole est invalide.


def apply_policy(labels: pd.DataFrame, policy: str) -> np.ndarray:
    """Transforme les labels selon la stratégie choisie pour les cellules vides.

    Retourne une matrice (n_molécules, 12) contenant 0, 1, ou NaN (NaN uniquement avec 'drop').
    """
    values = labels[config.LABEL_COLUMNS].to_numpy(dtype=float)  # Convertit les 12 colonnes en matrice de nombres.
    if policy == "drop":  # Stratégie « drop » :
        return values  # on garde les NaN ; ces cellules seront ignorées pendant l'apprentissage.
    if policy == "union":  # Stratégie « union » :
        return np.where(np.isnan(values), 1.0, values)  # une cellule vide devient un label positif.
    if policy == "intersection":  # Stratégie « intersection » :
        return np.where(np.isnan(values), 0.0, values)  # une cellule vide devient un label négatif.
    raise ValueError(f"Stratégie inconnue : {policy}")  # Toute autre valeur est une erreur de programmation.
