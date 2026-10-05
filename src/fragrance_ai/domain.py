"""Domaine d'applicabilité : la molécule analysée ressemble-t-elle à ce que le modèle a appris ?

Principe : on cherche la molécule d'entraînement la plus proche (similarité de Tanimoto sur les
fingerprints de Morgan). Plus elle est proche, plus la prédiction repose sur des exemples comparables.
"""

import numpy as np  # Calcul vectoriel des similarités.
import pandas as pd  # Lecture des labels de référence.

from fragrance_ai import config  # Chemins, labels et paramètres des fingerprints.
from fragrance_ai.data import connectivity_key  # Clé chimique sans stéréochimie, pour reconnaître une molécule identique.
from fragrance_ai.features import morgan_vector, smiles_to_mol  # Mêmes fingerprints que le modèle.

REFERENCE_PATH = config.MODELS_DIR / "train_reference.npz"  # Fingerprints et labels des molécules d'entraînement.

HIGH, MEDIUM = 0.6, 0.4  # Seuils indicatifs de similarité de Tanimoto, usuels en chimie computationnelle.


def save_reference(df_train: pd.DataFrame) -> None:
    """Sauvegarde les fingerprints et les labels d'entraînement, pour les comparer plus tard."""
    smiles = df_train[config.SMILES_COL].tolist()  # SMILES d'entraînement.
    bits = np.vstack([morgan_vector(smiles_to_mol(s)) for s in smiles]).astype(bool)  # Matrice binaire (n, 2048).
    labels = df_train[config.LABEL_COLUMNS].to_numpy(dtype=float)  # Labels connus (avec NaN si inconnus).
    np.savez_compressed(  # Fichier compact (~1 Mo) : bits compressés 8 par octet.
        REFERENCE_PATH, bits=np.packbits(bits, axis=1), smiles=np.array(smiles), labels=labels,
        keys=np.array([connectivity_key(s) for s in smiles]),  # Clés exactes des molécules d'entraînement.
    )


class ApplicabilityDomain:
    """Compare une molécule à toutes les molécules d'entraînement."""

    def __init__(self):
        data = np.load(REFERENCE_PATH)  # Charge la référence sauvegardée à l'entraînement.
        self.bits = np.unpackbits(data["bits"], axis=1)[:, : config.MORGAN_BITS].astype(np.float32)  # Bits décompressés.
        self.counts = self.bits.sum(axis=1)  # Nombre de bits allumés par molécule de référence.
        self.smiles = data["smiles"]  # SMILES de référence.
        self.labels = data["labels"]  # Labels de référence.
        self.keys = set(data["keys"].tolist())  # Clés exactes, pour une recherche instantanée.

    def nearest(self, smiles: str) -> dict:
        """Molécule d'entraînement la plus proche, sa similarité et un niveau de confiance."""
        query = morgan_vector(smiles_to_mol(smiles)).astype(np.float32)  # Fingerprint de la molécule analysée.
        common = self.bits @ query  # Bits allumés en commun avec chaque molécule de référence.
        union = self.counts + query.sum() - common  # Bits allumés dans l'une ou l'autre.
        similarity = np.divide(common, union, out=np.zeros_like(common), where=union > 0)  # Tanimoto = commun / union.
        best = int(np.argmax(similarity))  # Indice de la molécule la plus proche.
        score = float(similarity[best])  # Sa similarité, entre 0 et 1.
        if connectivity_key(smiles) in self.keys:  # La molécule elle-même (ou un stéréoisomère) est dans l'entraînement.
            level = "connue"  # La démonstration ne prouve alors pas la capacité à généraliser.
        elif score >= HIGH:  # Très proche de molécules connues (1,0 possible sans être identique : le fingerprint a ses limites).
            level = "élevée"
        elif score >= MEDIUM:  # Ressemblance partielle.
            level = "moyenne"
        else:  # Structure éloignée de tout ce que le modèle a vu.
            level = "faible"
        known = [  # Familles olfactives connues de la molécule voisine (labels égaux à 1).
            config.LABELS_FR[label] for label, value in zip(config.LABEL_COLUMNS, self.labels[best]) if value == 1
        ]
        return {  # Résultat sérialisable en JSON pour l'API.
            "similarity": round(score, 3),  # Similarité de Tanimoto.
            "level": level,  # Niveau de confiance lisible.
            "nearest_smiles": str(self.smiles[best]),  # Voisine la plus proche.
            "nearest_families": known,  # Ce que l'on sait de cette voisine.
            "n_similar": int((similarity >= HIGH).sum()),  # Nombre de molécules d'entraînement très proches.
        }
