"""Explication des prédictions avec SHAP, traduites en langage chimique."""

import numpy as np  # Tableaux numériques.
import shap  # SHAP : mesure la contribution de chaque variable à une prédiction.
from rdkit import Chem  # Manipulation des molécules.
from rdkit.Chem import rdFingerprintGenerator  # Pour retrouver quels atomes ont activé un bit de Morgan.

from fragrance_ai import config  # Paramètres des fingerprints.
from fragrance_ai.features import DESCRIPTOR_NAMES, FEATURE_NAMES  # Noms des variables.

N_DESCRIPTORS = len(DESCRIPTOR_NAMES)  # Les premières colonnes sont des descripteurs, les suivantes des bits de Morgan.


def morgan_bit_fragments(mol) -> dict[int, str]:
    """Associe chaque bit de Morgan actif à la sous-structure chimique qui l'a allumé."""
    generator = rdFingerprintGenerator.GetMorganGenerator(  # Même générateur que pour l'entraînement…
        radius=config.MORGAN_RADIUS, fpSize=config.MORGAN_BITS  # …avec exactement les mêmes paramètres.
    )
    output = rdFingerprintGenerator.AdditionalOutput()  # Objet qui recevra les informations supplémentaires.
    output.AllocateBitInfoMap()  # Demande à RDKit de noter quel atome a produit quel bit.
    generator.GetFingerprint(mol, additionalOutput=output)  # Calcule le fingerprint en remplissant cette table.
    fragments = {}  # Résultat : {numéro de bit: fragment SMILES}.
    for bit, origins in output.GetBitInfoMap().items():  # Pour chaque bit actif et ses origines (atome, rayon)…
        atom_idx, radius = origins[0]  # …on prend la première origine (une suffit pour illustrer).
        if radius == 0:  # Rayon 0 : le bit décrit un atome seul…
            fragments[bit] = mol.GetAtomWithIdx(atom_idx).GetSymbol()  # …on affiche son symbole (C, O, S…).
            continue  # Passe au bit suivant.
        bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)  # Liaisons dans le voisinage de l'atome.
        atoms = {atom_idx}  # Ensemble des atomes du fragment, en commençant par l'atome central.
        for b in bonds:  # Pour chaque liaison du voisinage…
            bond = mol.GetBondWithIdx(b)  # …on récupère la liaison…
            atoms.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])  # …et ses deux atomes.
        fragments[bit] = Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), bondsToUse=list(bonds))  # Écrit le fragment.
    return fragments  # Dictionnaire utilisé pour rendre l'explication lisible.


def readable_name(feature_idx: int, fragments: dict[int, str]) -> str:
    """Nom compréhensible d'une variable : nom du descripteur, ou fragment chimique pour un bit de Morgan."""
    if feature_idx < N_DESCRIPTORS:  # Variable physico-chimique (ex. MolLogP)…
        return DESCRIPTOR_NAMES[feature_idx]  # …on garde son nom RDKit.
    bit = feature_idx - N_DESCRIPTORS  # Numéro du bit de Morgan.
    if bit in fragments:  # Bit présent dans la molécule…
        return f"présence de {fragments[bit]}"  # …on montre la sous-structure correspondante.
    return f"absence du motif {FEATURE_NAMES[feature_idx]}"  # Bit absent : son absence a pesé dans la décision.


def explain(model, x: np.ndarray, smiles: str, label_idx: int, top_k: int = 6) -> list[dict]:
    """Les variables qui ont le plus influencé la prédiction d'une famille olfactive pour une molécule."""
    estimator = model.final_estimator(label_idx)  # Modèle à arbres du label demandé.
    x_ready = model.transform_for_explainer(label_idx, x.reshape(1, -1))  # Même prétraitement qu'à l'entraînement.
    explainer = shap.TreeExplainer(estimator)  # Explicateur exact et rapide pour les modèles à arbres.
    values = np.asarray(explainer.shap_values(x_ready))  # Contribution de chaque variable à la prédiction.
    if values.ndim == 3:  # La forêt aléatoire renvoie une contribution par classe…
        values = values[..., 1]  # …on garde celle de la classe positive.
    contributions = values.reshape(-1)  # Vecteur à une dimension : une valeur par variable.
    mol = Chem.MolFromSmiles(smiles)  # Molécule, pour traduire les bits en fragments.
    fragments = morgan_bit_fragments(mol) if mol is not None else {}  # Table bit -> fragment.
    order = np.argsort(-np.abs(contributions))[:top_k]  # Indices des variables les plus influentes, en valeur absolue.
    return [  # Liste prête à afficher.
        {
            "feature": readable_name(int(i), fragments),  # Nom lisible.
            "contribution": float(contributions[i]),  # Positif : pousse vers la famille ; négatif : en éloigne.
        }
        for i in order  # Pour chacune des variables retenues.
    ]
