"""Transformation d'une molécule (SMILES) en vecteur numérique exploitable par un modèle."""

import numpy as np  # Tableaux numériques.
from rdkit import Chem, RDLogger  # Chem lit et manipule les molécules ; RDLogger contrôle les messages de RDKit.
from rdkit.Chem import Descriptors, rdFingerprintGenerator  # Descripteurs physico-chimiques et générateur de fingerprints.

from fragrance_ai import config  # Paramètres des fingerprints (rayon, nombre de bits).

RDLogger.DisableLog("rdApp.*")  # Coupe les avertissements RDKit, très bavards sur les SMILES atypiques.

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]  # Liste des ~200 descripteurs RDKit (masse, logP, TPSA…).
MORGAN_NAMES = [f"morgan_{i}" for i in range(config.MORGAN_BITS)]  # Un nom lisible pour chacun des 2048 bits.
FEATURE_NAMES = DESCRIPTOR_NAMES + MORGAN_NAMES  # Ordre final des colonnes : descripteurs puis bits de Morgan.

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(  # Crée une seule fois le générateur, pour gagner du temps.
    radius=config.MORGAN_RADIUS,  # Taille du voisinage pris en compte autour de chaque atome.
    fpSize=config.MORGAN_BITS,  # Longueur du vecteur produit.
)


def smiles_to_mol(smiles: str):
    """Convertit un SMILES en objet molécule RDKit ; renvoie None si le SMILES est invalide."""
    if not isinstance(smiles, str) or not smiles.strip():  # Rejette les valeurs vides ou qui ne sont pas du texte.
        return None  # None signale « molécule illisible » au reste du code.
    return Chem.MolFromSmiles(smiles.strip())  # RDKit renvoie lui-même None si la syntaxe est incorrecte.


def canonicalize(smiles: str) -> str | None:
    """Renvoie l'écriture canonique d'un SMILES (une molécule = une seule écriture)."""
    mol = smiles_to_mol(smiles)  # Lit la molécule.
    return Chem.MolToSmiles(mol) if mol is not None else None  # Réécrit le SMILES de façon standard, ou None.


def descriptor_vector(mol) -> np.ndarray:
    """Calcule les descripteurs physico-chimiques d'une molécule."""
    values = Descriptors.CalcMolDescriptors(mol)  # Dictionnaire {nom du descripteur: valeur}.
    vector = np.array([values.get(n, np.nan) for n in DESCRIPTOR_NAMES], dtype=float)  # Remet les valeurs dans l'ordre fixe.
    vector[~np.isfinite(vector)] = np.nan  # Remplace les infinis éventuels par NaN (traités ensuite par imputation).
    return np.clip(vector, -1e6, 1e6)  # Borne les valeurs extrêmes qui déstabiliseraient certains modèles.


def morgan_vector(mol) -> np.ndarray:
    """Calcule le fingerprint de Morgan : 1 si une sous-structure est présente, 0 sinon."""
    return _MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.float32)  # Vecteur binaire de 2048 positions.


def featurize_one(smiles: str) -> np.ndarray | None:
    """Vecteur complet (descripteurs + Morgan) pour une molécule, ou None si invalide."""
    mol = smiles_to_mol(smiles)  # Lit la molécule.
    if mol is None:  # SMILES illisible…
        return None  # …on ne peut rien calculer.
    return np.concatenate([descriptor_vector(mol), morgan_vector(mol)])  # Assemble les deux familles de variables.


def featurize(smiles_list) -> tuple[np.ndarray, np.ndarray]:
    """Vectorise une liste de SMILES.

    Retourne la matrice des variables et un masque booléen indiquant les molécules valides.
    """
    rows, valid = [], []  # rows accumule les vecteurs ; valid mémorise quelles molécules ont pu être lues.
    for smi in smiles_list:  # Parcourt chaque molécule.
        vec = featurize_one(smi)  # Calcule son vecteur.
        valid.append(vec is not None)  # Note si la lecture a réussi.
        if vec is not None:  # Seules les molécules valides sont ajoutées à la matrice.
            rows.append(vec)  # Ajoute le vecteur à la liste.
    X = np.vstack(rows) if rows else np.empty((0, len(FEATURE_NAMES)))  # Empile les vecteurs en une matrice 2D.
    return X.astype(np.float32), np.array(valid, dtype=bool)  # float32 divise la mémoire par deux sans perte utile.
