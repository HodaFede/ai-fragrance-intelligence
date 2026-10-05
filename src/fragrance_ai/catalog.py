"""Molécules célèbres de la parfumerie et « carte d'identité » chimique d'une molécule."""

import numpy as np  # Tirage aléatoire des molécules surprises.
import pandas as pd  # Lecture des exemples de test.
from rdkit.Chem import Descriptors  # Masse molaire.
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # Formule brute.

from fragrance_ai import config  # Chemins et libellés.
from fragrance_ai.features import smiles_to_mol  # Lecture des SMILES.

TEST_EXAMPLES_PATH = config.MODELS_DIR / "test_examples.csv"  # Molécules du jeu de test, jamais vues par le modèle.

CATALOG = [  # Chaque entrée : nom, structure, odeur en mots simples, place dans la parfumerie.
    {"name": "Vanilline", "smiles": "O=Cc1ccc(O)c(OC)c1",
     "odor": "Vanille douce et gourmande.",
     "story": "Principale molécule odorante de la gousse de vanille. Omniprésente dans les parfums gourmands "
              "et orientaux, et dans l'alimentation."},
    {"name": "Éthylvanilline", "smiles": "CCOc1cc(C=O)ccc1O",
     "odor": "Vanille plus intense, crémeuse.",
     "story": "Cousine de synthèse de la vanilline, deux à quatre fois plus puissante. Shalimar de Guerlain (1925) "
              "lui doit sa célèbre note vanillée."},
    {"name": "Coumarine", "smiles": "O=c1ccc2ccccc2o1",
     "odor": "Foin coupé, fève tonka, amande.",
     "story": "Fougère Royale de Houbigant (1882), l'un des premiers parfums à utiliser une molécule de synthèse, "
              "a donné son nom à toute la famille des parfums « fougère »."},
    {"name": "Hédione", "smiles": "CCCCCC1C(CCC1=O)CC(=O)OC",
     "odor": "Jasmin frais, lumineux, presque transparent.",
     "story": "Découverte dans le jasmin, popularisée par Eau Sauvage de Dior (1966). On la retrouve aujourd'hui "
              "dans une grande partie des parfums."},
    {"name": "Dodécanal (aldéhyde C12)", "smiles": "CCCCCCCCCCCC=O",
     "odor": "Cireux, savonneux, évoque la peau propre.",
     "story": "Les aldéhydes de cette famille ont fait la célébrité de Chanel N°5 (1921), premier grand parfum "
              "à les utiliser massivement."},
    {"name": "Ambroxide (Ambrox)", "smiles": "CC1(C)CCCC2(C)C1CCC1(C)OCCC21",
     "odor": "Ambré, boisé, chaud et minéral.",
     "story": "Remplaçant de synthèse de l'ambre gris, une matière d'origine animale. Très utilisée aujourd'hui, "
              "notamment dans Sauvage de Dior (2015), souvent cité parmi les parfums les plus vendus au monde."},
    {"name": "Calone", "smiles": "Cc1ccc2OCC(=O)COc2c1",
     "odor": "Marin, melon, brise océanique.",
     "story": "À l'origine de la vague des parfums « marins » des années 1990, dont L'Eau d'Issey "
              "d'Issey Miyake (1992)."},
    {"name": "Galaxolide", "smiles": "CC1COCc2cc3c(cc12)C(C)(C)C(C)C3(C)C",
     "odor": "Musc blanc, linge propre.",
     "story": "L'un des muscs de synthèse les plus utilisés au monde, en parfumerie comme dans les lessives."},
    {"name": "Muscone", "smiles": "CC1CCCCCCCCCCCCC(=O)C1",
     "odor": "Musc animal, chaud et sensuel.",
     "story": "Principe odorant du musc naturel du chevrotain porte-musc, aujourd'hui remplacé par des muscs "
              "de synthèse pour protéger l'espèce."},
    {"name": "Linalol", "smiles": "CC(C)=CCCC(C)(O)C=C",
     "odor": "Floral frais, légèrement boisé.",
     "story": "Présent dans la lavande, la bergamote et le bois de rose. Une des molécules les plus courantes "
              "en parfumerie."},
    {"name": "Phényléthanol", "smiles": "OCCc1ccccc1",
     "odor": "Rose douce, miellée.",
     "story": "Un des principaux constituants de l'odeur de la rose, utilisé dans d'innombrables parfums floraux."},
    {"name": "Géraniol", "smiles": "CC(C)=CCC/C(C)=C/CO",
     "odor": "Rose, légèrement citronnée.",
     "story": "Présent dans l'essence de rose et de géranium. Même formule brute que le linalol, mais une "
              "structure différente."},
    {"name": "Limonène", "smiles": "CC1=CCC(CC1)C(=C)C",
     "odor": "Orange, zeste d'agrumes.",
     "story": "Constituant majoritaire de l'huile essentielle d'orange, au cœur des eaux de Cologne."},
    {"name": "Eugénol", "smiles": "COc1cc(CC=C)ccc1O",
     "odor": "Clou de girofle, épicé.",
     "story": "Molécule du clou de girofle, utilisée dans les parfums épicés et aussi en dentisterie."},
    {"name": "Cinnamaldéhyde", "smiles": "O=C/C=C/c1ccccc1",
     "odor": "Cannelle.",
     "story": "Molécule responsable de l'odeur de la cannelle, surtout utilisée dans les arômes alimentaires."},
    {"name": "cis-3-Hexénol", "smiles": "CC/C=C\\CCO",
     "odor": "Herbe fraîchement coupée.",
     "story": "Libérée par les feuilles quand on les coupe. Apporte des notes vertes et naturelles."},
    {"name": "Acétylpyrazine", "smiles": "CC(=O)c1cnccn1",
     "odor": "Pop-corn, grillé.",
     "story": "Molécule des notes grillées, plus présente dans les arômes alimentaires que dans les parfums."},
]
CATALOG_BY_NAME = {m["name"]: m for m in CATALOG}  # Accès direct par nom.

SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")  # Chiffres en indice, pour écrire C₈H₈O₃.


def identity(smiles: str) -> dict:
    """Formule brute et masse molaire d'une molécule."""
    mol = smiles_to_mol(smiles)  # Lit la molécule.
    return {
        "formula": CalcMolFormula(mol).translate(SUBSCRIPTS),  # Formule brute avec indices (ex. C₈H₈O₃).
        "molar_mass": round(Descriptors.MolWt(mol), 2),  # Masse molaire en g/mol.
    }


def save_test_examples(df_test: pd.DataFrame) -> None:
    """Sauvegarde le jeu de test (SMILES et vrais labels), pour les molécules surprises de l'application."""
    df_test[[config.SMILES_COL, *config.LABEL_COLUMNS]].to_csv(TEST_EXAMPLES_PATH, index=False)  # Petit CSV (~60 Ko).


def random_test_molecule(rng: np.random.Generator) -> dict:
    """Une molécule du jeu de test, avec ses vraies familles olfactives."""
    df = pd.read_csv(TEST_EXAMPLES_PATH)  # Charge les 890 molécules de test.
    df = df[df[config.LABEL_COLUMNS].sum(axis=1) > 0]  # Garde celles qui ont au moins une famille connue.
    row = df.iloc[int(rng.integers(len(df)))]  # Tire une ligne au hasard.
    return {
        "smiles": row[config.SMILES_COL],  # Structure.
        "true_families": [config.LABELS_FR[l] for l in config.LABEL_COLUMNS if row[l] == 1],  # Vrai profil.
    }
