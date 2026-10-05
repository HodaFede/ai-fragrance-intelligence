"""Interface web : saisir une molécule et voir son profil olfactif prédit, expliqué.

Lancement : streamlit run app/streamlit_app.py
"""

import json  # Lecture/écriture JSON pour dialoguer avec l'API.
import os  # Lecture des variables d'environnement (adresse de l'API).
import sys  # Permet d'ajouter le dossier src/ au chemin d'import.
import urllib.request  # Appel HTTP vers l'API, sans dépendance supplémentaire.
from pathlib import Path  # Chemins de fichiers.

import altair as alt  # Graphiques déclaratifs (installé avec Streamlit).
import numpy as np  # Tirage aléatoire des molécules surprises.
import pandas as pd  # Tableaux de données.
import streamlit as st  # Framework de l'interface web.
from rdkit import Chem  # Lecture de la molécule pour la dessiner.
from rdkit.Chem import Draw  # Dessin 2D de la molécule.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # Rend le package fragrance_ai importable sans installation.

from fragrance_ai import config  # noqa: E402  Chemins et libellés (import après la ligne précédente, volontairement).
from fragrance_ai.catalog import CATALOG_BY_NAME, identity, random_test_molecule  # noqa: E402  Molécules célèbres et identité chimique.

API_URL = os.getenv("API_URL")  # Si défini (Docker Compose), l'app interroge l'API ; sinon elle prédit en local.

ANALYSIS_PATH = config.REPORTS_DIR / "analysis.json"  # Résultats du notebook d'analyse.
ANALYSIS = json.loads(ANALYSIS_PATH.read_text()) if ANALYSIS_PATH.exists() else None  # Absent si le notebook n'a pas tourné.
SURPRISE = "Molécule surprise (jamais vue par le modèle)"  # Option : molécule du jeu de test.
CUSTOM = "Saisir ma propre molécule (SMILES)"  # Option : saisie libre.

MODEL_FR = {  # Noms affichés des modèles.
    "xgboost": "XGBoost",  # Gradient boosting.
    "random_forest": "Forêt aléatoire",  # Random Forest.
    "mlp": "Réseau dense (MLP)",  # Réseau de neurones classique.
    "gnn": "Réseau sur graphe (GNN)",  # Réseau de neurones sur la structure moléculaire.
}
POLICY_FR = {"drop": "Ignorés", "intersection": "Comptés négatifs", "union": "Comptés positifs"}  # Stratégies en clair.

ACCENT, INK, MUTED = "#A0446B", "#2E1F2B", "#D8C0CC"  # Palette : prune (familles retenues), encre prune (seuils), rose poudré (non retenues).

st.set_page_config(page_title="AI Fragrance Intelligence", page_icon="🧪", layout="wide")  # Titre de l'onglet et mise en page large.

st.markdown(  # Petite feuille de style pour la typographie et les espacements.
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=Manrope:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    h1, h2, h3 { font-family: 'Fraunces', Georgia, serif; color: #2E1F2B; font-weight: 500; }
    .block-container { padding-top: 2rem; max-width: 1100px; }
    </style>
    """,
    unsafe_allow_html=True,  # Autorise l'injection de ce CSS.
)


@st.cache_resource  # Charge le modèle une seule fois pour toute la durée de vie de l'application.
def local_predictor():
    """Prédicteur local, utilisé quand aucune API n'est configurée."""
    from fragrance_ai.predict import get_predictor  # Import tardif : inutile si l'on passe par l'API.

    return get_predictor()  # Renvoie l'instance partagée.


def run_prediction(smiles: str) -> dict:
    """Obtient la prédiction, via l'API si elle est configurée, sinon en local."""
    if API_URL:  # Mode « microservices » (Docker Compose).
        request = urllib.request.Request(  # Prépare la requête HTTP.
            f"{API_URL}/predict",  # Route de prédiction de l'API.
            data=json.dumps({"smiles": smiles}).encode(),  # Corps JSON encodé en octets.
            headers={"Content-Type": "application/json"},  # Indique que l'on envoie du JSON.
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # Envoie la requête…
            return json.loads(response.read())  # …et décode la réponse.
    return local_predictor().predict(smiles)  # Mode autonome (Streamlit Cloud, Hugging Face Spaces).


def profile_chart(families: list[dict]) -> alt.Chart:
    """Graphique du profil olfactif : une barre par famille, avec le seuil de décision."""
    df = pd.DataFrame(families)  # Tableau : une ligne par famille.
    df["Statut"] = df["predicted"].map({True: "Retenue", False: "Non retenue"})  # Libellé de la couleur.
    order = df["label_fr"].tolist()  # Conserve l'ordre de pertinence calculé par le prédicteur.
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=16).encode(  # Barres horizontales.
        x=alt.X("probability:Q", title="Probabilité prédite", scale=alt.Scale(domain=[0, 1])),  # Axe des probabilités.
        y=alt.Y("label_fr:N", sort=order, title=None),  # Une ligne par famille, dans l'ordre choisi.
        color=alt.Color("Statut:N", scale=alt.Scale(domain=["Retenue", "Non retenue"], range=[ACCENT, MUTED])),  # Ambre = retenue.
        tooltip=[  # Infobulle au survol.
            alt.Tooltip("label_fr:N", title="Famille"),  # Nom de la famille.
            alt.Tooltip("probability:Q", title="Probabilité", format=".2f"),  # Probabilité.
            alt.Tooltip("threshold:Q", title="Seuil", format=".2f"),  # Seuil de décision.
        ],
    )
    ticks = alt.Chart(df).mark_tick(color=INK, thickness=2, size=20).encode(  # Petit trait vertical = seuil.
        x="threshold:Q",  # Position du seuil.
        y=alt.Y("label_fr:N", sort=order),  # Même ligne que la barre.
    )
    return (bars + ticks).properties(height=36 * len(df))  # Superpose barres et seuils ; hauteur adaptée.


def render_molecule(smiles: str, info: dict | None, true_families: list | None, surprise: bool) -> None:
    """Affiche l'analyse complète d'une molécule : identité, profil prédit et réel, fiabilité, explications."""
    try:  # On tente la prédiction.
        result = run_prediction(smiles)  # Appelle le modèle.
    except Exception as exc:  # En cas de SMILES invalide ou de modèle absent…
        st.error(f"Impossible d'analyser cette molécule : {exc}. Vérifiez la syntaxe du SMILES.")  # …message actionnable.
        return  # Arrête l'affichage de cette molécule, sans bloquer les autres onglets.

    ident = identity(smiles)  # Formule brute et masse molaire.
    domain = result.get("applicability")  # Ressemblance avec les molécules d'entraînement.
    if true_families is None and domain and domain["level"] == "connue":  # Molécule présente dans OdorNet…
        true_families = domain["nearest_families"]  # …on connaît son vrai profil.
    left, right = st.columns([1, 2])  # Mise en page : carte d'identité à gauche, odeur à droite.
    with left:  # Colonne de gauche : la molécule.
        st.image(Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(360, 300)))  # Dessin 2D de la molécule.
        st.markdown(  # Carte d'identité chimique.
            f"**{info['name'] if info else 'Molécule analysée'}**  \n"  # Nom (ou intitulé générique).
            f"Formule brute : **{ident['formula']}**  \n"  # Ex. C₈H₈O₃.
            f"Masse molaire : {str(ident['molar_mass']).replace('.', ',')} g/mol"  # En grammes par mole, virgule à la française.
        )
        st.caption(f"Structure (SMILES) : `{result['canonical_smiles']}`")  # Écriture complète de la structure.
    with right:  # Colonne de droite : l'odeur.
        if info:  # Molécule du catalogue : on raconte son histoire.
            st.markdown(f"**Odeur décrite :** {info['odor']}  \n**En parfumerie :** {info['story']}")
        if surprise:  # Molécule tirée du jeu de test.
            st.markdown("Molécule tirée au hasard parmi les 890 molécules d'évaluation : "
                        "**le modèle ne l'a jamais vue** pendant son apprentissage.")
        predicted = [f["label_fr"] for f in result["families"] if f["predicted"]]  # Familles retenues par le modèle.
        p1, p2 = st.columns(2)  # Prédiction et réalité côte à côte.
        with p1:  # Ce que prédit le modèle.
            st.markdown("**Profil prédit par le modèle**")
            st.write(", ".join(predicted) if predicted else "Aucune famille au-dessus de son seuil.")
        with p2:  # Ce que disent les données.
            st.markdown("**Profil réel (données OdorNet)**")
            if true_families is None:  # Molécule absente d'OdorNet.
                st.write("Inconnu : cette molécule ne figure pas dans les données.")
            else:  # Molécule présente : on affiche son profil, en signalant les familles bien trouvées.
                st.write(", ".join(f"{f} ✓" if f in predicted else f for f in true_families) or "Aucune famille renseignée.")
        if true_families:  # Petit bilan pour le lecteur.
            found = len(set(true_families) & set(predicted))  # Familles réelles retrouvées par le modèle.
            st.caption(f"✓ = famille réelle retrouvée par le modèle ({found} sur {len(true_families)}).")
        st.altair_chart(profile_chart(result["families"]), width="stretch")  # Graphique du profil.
        st.caption("Probabilité estimée pour chaque famille ; le trait foncé marque le seuil à partir duquel la famille est retenue.")
    st.caption(  # Pourquoi on ne saisit pas une formule brute.
        f"Pourquoi pas la formule brute seule ? Elle ne suffit pas à identifier une molécule : le linalol et le géraniol "
        f"partagent la même ({identity('CC(C)=CCCC(C)(O)C=C')['formula']}) mais n'ont pas la même structure. "
        "Le modèle a besoin de la structure complète, décrite par le SMILES."
    )

    if domain:  # Affiché seulement si la référence d'entraînement est disponible.
        st.subheader("Fiabilité de cette prédiction", anchor=False)  # Section domaine d'applicabilité.
        messages = {  # Message adapté au niveau de ressemblance.
            "connue": ("Cette molécule fait partie des données d'entraînement : le modèle l'a déjà vue. "
                       "La prédiction illustre ce qu'il a appris, mais ne prouve pas sa capacité à généraliser."),
            "élevée": "Structure très proche de molécules connues du modèle : la prédiction repose sur des exemples comparables.",
            "moyenne": "Ressemblance partielle avec les molécules connues : prédiction à interpréter avec prudence.",
            "faible": ("Structure éloignée des molécules connues du modèle : la prédiction s'appuie sur peu "
                       "d'exemples comparables. À interpréter avec prudence."),
        }
        notify = {"connue": st.info, "élevée": st.success, "moyenne": st.warning, "faible": st.error}  # Couleur du message.
        notify[domain["level"]](messages[domain["level"]])  # Affiche le message dans la bonne couleur.
        d1, d2 = st.columns([1, 2])  # Similarité à gauche, molécule voisine à droite.
        d1.metric("Similarité avec la plus proche", f"{domain['similarity']:.2f}",  # Tanimoto entre 0 et 1.
                  help="Similarité de Tanimoto sur les fingerprints de Morgan : 1 = sous-structures identiques, 0 = aucune en commun.")
        if ANALYSIS:  # Ce que l'analyse du jeu de test dit de cet indicateur.
            bins = ANALYSIS["by_similarity"]  # Scores par niveau de similarité.
            st.caption(f"Repère indicatif : sur les molécules d'évaluation, le score F1 passe de {bins[0]['Macro-F1']:.2f} "
                       f"pour les moins ressemblantes à {bins[-1]['Macro-F1']:.2f} pour les plus ressemblantes.")  # Chiffres issus du notebook d'analyse.
        families = ", ".join(domain["nearest_families"]) or "aucune famille renseignée"  # Profil connu de la voisine.
        d2.markdown(f"Molécule d'entraînement la plus proche : `{domain['nearest_smiles']}`  \nProfil connu : {families}")  # Voisine.

    st.subheader("Pourquoi ces prédictions ?", anchor=False)  # Section explicabilité.
    st.caption(  # Mode de lecture, pour un lecteur non spécialiste.
        "Pour chaque famille, les éléments de la molécule qui ont le plus pesé : en prune, ceux qui rapprochent "
        "de la famille ; en foncé, ceux qui en éloignent. « présence de … » désigne un fragment de la molécule."
    )
    for label, contributions in result["explanations"].items():  # Pour chaque famille expliquée…
        with st.expander(config.LABELS_FR[label], expanded=True):  # …un bloc dépliable.
            df = pd.DataFrame(contributions)  # Tableau des contributions SHAP.
            df["Effet"] = df["contribution"].apply(lambda v: "Rapproche" if v > 0 else "Éloigne")  # Sens de l'effet.
            chart = alt.Chart(df).mark_bar(height=14).encode(  # Barres des contributions.
                x=alt.X("contribution:Q", title="Influence sur la prédiction (valeur SHAP)"),  # Taille de l'effet.
                y=alt.Y("feature:N", sort=None, title=None),  # Variable, dans l'ordre d'importance.
                color=alt.Color("Effet:N", scale=alt.Scale(domain=["Rapproche", "Éloigne"], range=[ACCENT, INK])),  # Couleur du sens.
            ).properties(height=30 * len(df))  # Hauteur adaptée.
            st.altair_chart(chart, width="stretch")  # Affiche le graphique.

st.title("AI Fragrance Intelligence", anchor=False)  # Titre principal ; anchor=False retire l'icône de lien inutile.
st.write(  # Phrase d'introduction : ce que fait l'outil, en clair.
    "Choisissez une molécule célèbre de la parfumerie, tirez une molécule que le modèle n'a jamais vue, "
    "ou saisissez la vôtre : le modèle prédit ses familles olfactives et montre ce qui a pesé dans sa décision."
)

tab_predict, tab_perf, tab_about = st.tabs(["Analyser une molécule", "Performances", "Méthode"])  # Trois onglets.

with tab_predict:  # Onglet 1 : analyse d'une molécule.
    choice = st.selectbox(  # Liste déroulante des molécules.
        "Choisir une molécule",  # Libellé.
        [*CATALOG_BY_NAME, SURPRISE, CUSTOM],  # Molécules célèbres, puis molécule surprise, puis saisie libre.
        help="Les molécules célèbres font partie des données d'entraînement ; la molécule surprise, jamais.",
    )
    info, true_families, surprise = None, None, choice == SURPRISE  # Valeurs par défaut.
    if choice == CUSTOM:  # Saisie libre.
        smiles = st.text_input("Structure au format SMILES", value="O=Cc1ccc(O)c(OC)c1",  # Champ de saisie.
                               help="Le SMILES décrit la structure d'une molécule sur une ligne. On le trouve sur PubChem ou Wikipédia.")
    elif surprise:  # Molécule du jeu de test.
        new_draw = st.button("Tirer une autre molécule")  # Bouton pour changer de molécule.
        if new_draw or "surprise" not in st.session_state:  # Premier affichage ou clic…
            st.session_state.surprise = random_test_molecule(np.random.default_rng())  # …nouveau tirage.
        smiles = st.session_state.surprise["smiles"]  # Structure tirée.
        true_families = st.session_state.surprise["true_families"]  # Son vrai profil.
    else:  # Molécule du catalogue.
        info = CATALOG_BY_NAME[choice]  # Fiche de la molécule.
        smiles = info["smiles"]  # Sa structure.
    render_molecule(smiles, info, true_families, surprise)  # Affichage complet.

with tab_perf:  # Onglet 2 : performances.
    if config.RESULTS_PATH.exists():  # Si l'entraînement a produit le tableau…
        results = pd.read_csv(config.RESULTS_PATH)  # …on le charge.
        meta = json.loads(config.METADATA_PATH.read_text())  # Métadonnées du modèle déployé.
        st.caption("Performance globale du modèle, mesurée une fois sur 890 molécules jamais vues : ces chiffres ne dépendent pas de la molécule analysée.")  # Évite la confusion.
        a, b, c = st.columns(3)  # Trois indicateurs côte à côte.
        a.metric("Score F1 (test)", f"{meta['test_metrics']['macro_f1']:.3f}",  # Score du modèle déployé.
                 help="Entre 0 et 1 : récompense le fait de trouver les bonnes familles sans en inventer, chaque famille comptant autant.")
        b.metric("Référence publiée", f"{meta['paper_baseline_macro_f1']:.2f}",  # Score de l'article.
                 help="Score F1 du modèle proposé par les auteurs d'OdorNet dans leur article.")
        c.metric("AUROC (test)", f"{meta['test_metrics']['macro_auroc']:.3f}",  # Qualité de classement.
                 help="Entre 0,5 (hasard) et 1 (parfait) : capacité à donner une probabilité plus forte aux bonnes familles qu'aux mauvaises.")
        st.write(f"Modèle déployé : **{meta['model']}**, stratégie de labels **{meta['policy']}**.")  # Rappel du choix.
        if ANALYSIS:  # Marge d'erreur calculée par bootstrap dans le notebook d'analyse.
            lo, hi = ANALYSIS["macro_f1_ci95"]  # Intervalle de confiance à 95 %.
            st.markdown(  # Phrase claire pour un non-spécialiste.
                f"**Marge d'erreur :** avec 95 % de confiance, le score F1 se situe entre **{lo:.3f}** et **{hi:.3f}**, "
                "entièrement au-dessus de la référence publiée. En revanche, XGBoost et la forêt aléatoire sont "
                "statistiquement à égalité : leur écart est plus petit que la marge d'erreur."
            )
        st.subheader("Comparaison des modèles", anchor=False)  # Titre du tableau.
        best_model, best_policy = meta["model"], meta["policy"]  # Combinaison déployée, pour la repérer.
        table = pd.DataFrame({  # Tableau lisible : noms en français, colonnes essentielles seulement.
            "Modèle": [  # Nom clair du modèle, avec une mention pour celui qui est déployé.
                MODEL_FR[m] + (" · déployé" if (m, p) == (best_model, best_policy) else "")
                for m, p in zip(results["model"], results["policy"])  # Parcourt chaque expérience.
            ],
            "Labels manquants": results["policy"].map(POLICY_FR),  # Stratégie, en clair.
            "F1 (test)": results["test_macro_f1"],  # Métrique principale.
            "Précision": results["test_macro_precision"],  # Part des familles prédites qui sont justes.
            "Rappel": results["test_macro_recall"],  # Part des vraies familles retrouvées.
            "PR-AUC": results["test_macro_pr_auc"],  # Qualité sur les familles rares.
        })
        st.dataframe(  # Affichage du tableau.
            table,  # Données à afficher.
            hide_index=True,  # Masque la numérotation des lignes, inutile ici.
            width="stretch",  # Occupe toute la largeur disponible.
            column_config={  # Mise en forme colonne par colonne.
                "F1 (test)": st.column_config.ProgressColumn(  # Barre de progression : la comparaison se lit d'un coup d'œil.
                    "F1 (test)", format="%.3f", min_value=0.0, max_value=1.0),  # Échelle de 0 à 1, trois décimales.
                "Précision": st.column_config.NumberColumn(format="%.3f"),  # Trois décimales.
                "Rappel": st.column_config.NumberColumn(format="%.3f"),  # Trois décimales.
                "PR-AUC": st.column_config.NumberColumn(format="%.3f"),  # Trois décimales.
            },
        )
        st.caption(  # Légende pour un lecteur non spécialiste.
            "Scores mesurés sur 890 molécules jamais vues. Les modèles sont classés selon leur score de validation, "
            "et le modèle déployé est le meilleur d'entre eux. Le F1 combine précision et rappel ; "
            "la PR-AUC mesure la qualité sur les familles rares."
        )
        per_label = pd.DataFrame(  # Détail par famille du modèle déployé (jeu de test).
            [{"Famille": config.LABELS_FR[k], "F1": m["f1"], "Précision": m["precision"], "Rappel": m["recall"],
              "PR-AUC": m["pr_auc"], "Positifs (test)": m["support"]}  # Une ligne par famille.
             for k, m in meta["test_metrics"]["per_label"].items()]  # Parcourt les 12 familles.
        ).sort_values("F1", ascending=False)  # Des familles les mieux prédites aux plus difficiles.
        st.subheader("Détail par famille", anchor=False)  # Titre de la section.
        st.altair_chart(  # Graphique des F1 par famille : montre où le modèle est fort ou faible.
            alt.Chart(per_label).mark_bar(color=ACCENT).encode(x="F1:Q", y=alt.Y("Famille:N", sort="-x", title=None)),
            width="stretch",
        )
        st.dataframe(per_label.round(3), width="stretch", hide_index=True)  # Tableau détaillé.
        st.subheader("Limites", anchor=False)  # Ce que le modèle ne sait pas faire.
        for limit in meta.get("limitations", []):  # Chaque limite déclarée à l'entraînement…
            st.write(f"- {limit}")  # …affichée en clair.
    else:  # Sinon…
        st.info("Lancez `python -m fragrance_ai.train` pour générer les résultats.")  # …on indique quoi faire.

with tab_about:  # Onglet 3 : méthode, rédigée pour un lecteur qui ne connaît ni la parfumerie ni la data science.
    st.subheader("Le problème", anchor=False)  # Pourquoi ce projet.
    st.markdown(  # Texte explicatif.
        "L'odeur d'une molécule dépend de sa structure chimique, mais la relation est difficile à prévoir : "
        "deux molécules presque identiques peuvent sentir très différemment. Aujourd'hui, la seule façon fiable "
        "de connaître une odeur est de la faire sentir par des experts, ce qui est long et coûteux. "
        "Ce projet cherche à **prédire les familles olfactives d'une molécule à partir de sa seule structure**, "
        "pour aider à trier des molécules candidates avant de les tester."
    )

    st.subheader("Les données", anchor=False)  # Source et contenu.
    st.markdown(  # Texte explicatif avec liens cliquables.
        "Le modèle a appris sur **OdorNet**, un jeu de données scientifique publié en octobre 2026 dans la revue "
        "*Scientific Data* (groupe Nature) par une équipe de l'Université de Nankai. Il réunit **8 892 molécules**, "
        "chacune décrite par **12 familles olfactives** (floral, boisé, épicé, gourmand…) établies à partir de "
        "descriptions de parfumeurs et d'aromaticiens.\n\n"
        "- [Lire l'article scientifique](https://www.nature.com/articles/s41597-026-08429-z)\n"
        "- [Voir les données et le code des auteurs](https://github.com/NKU-DOIE/OdorNet)\n\n"
        "Les molécules sont réparties en trois groupes fixés par les auteurs : **6 224 pour apprendre**, "
        "**1 778 pour régler le modèle** et **890 pour l'évaluer** à la toute fin. Deux variantes d'une même "
        "molécule (par exemple deux formes « miroir ») sont toujours placées dans le même groupe, et l'application "
        "le vérifie à chaque entraînement : sinon, le modèle serait évalué sur des molécules qu'il a déjà vues."
    )

    st.subheader("Comment une molécule devient des nombres", anchor=False)  # Variables.
    st.markdown(  # Texte explicatif.
        "Un modèle ne lit pas une formule chimique : il lui faut des nombres. Chaque molécule est donc décrite de deux façons. "
        "D'abord par **217 propriétés physico-chimiques** calculées avec la bibliothèque RDKit : masse, polarité, "
        "nombre de cycles, solubilité estimée… Ensuite par une **empreinte moléculaire** (fingerprint de Morgan) : "
        "une liste de 2 048 cases qui valent 1 si un petit motif chimique est présent (un groupe alcool, un cycle "
        "aromatique, un enchaînement d'atomes…) et 0 sinon."
    )

    st.subheader("Les modèles comparés", anchor=False)  # Approches.
    st.markdown(  # Texte explicatif.
        "Quatre approches ont été entraînées puis comparées sur les mêmes données :\n\n"
        "- **Forêt aléatoire (Random Forest)** : des centaines d'arbres de décision qui votent ensemble.\n"
        "- **XGBoost** : des arbres de décision construits les uns après les autres, chacun corrigeant les erreurs du précédent. "
        "C'est le modèle retenu, car le plus performant.\n"
        "- **Réseau de neurones dense (MLP)** : un modèle d'apprentissage profond qui lit les mêmes 2 265 nombres.\n"
        "- **Réseau de neurones sur graphe (GNN)** : il lit directement la molécule comme un réseau d'atomes reliés "
        "par des liaisons, sans passer par l'empreinte.\n\n"
        "Une difficulté propre à ces données : pour environ une case sur cinq, on ne sait pas si une molécule appartient ou non "
        "à une famille, car l'information manquait dans les sources. Plutôt que de deviner, les modèles **ignorent ces "
        "cases inconnues** pendant l'apprentissage. Cette stratégie s'est révélée la meilleure parmi les trois testées."
    )

    st.subheader("Comment on mesure la qualité sans se tromper soi-même", anchor=False)  # Protocole.
    st.markdown(  # Texte explicatif.
        "Les réglages (par exemple le seuil de probabilité à partir duquel une famille est retenue) et le choix du "
        "meilleur modèle se font uniquement sur le groupe de réglage. Le groupe d'évaluation n'est utilisé qu'une "
        "seule fois, à la fin, comme un examen dont on ne connaît pas les questions à l'avance. "
        "La note principale est le **macro-F1** : un score entre 0 et 1 qui récompense à la fois le fait de trouver "
        "les bonnes familles et le fait de ne pas en inventer, en donnant le même poids aux familles rares et fréquentes. "
        "Le modèle obtient **0,468**, contre **0,42** pour le modèle de référence des auteurs de l'article."
    )

    st.subheader("Pourquoi le modèle prédit ce qu'il prédit", anchor=False)  # Explicabilité.
    st.markdown(  # Texte explicatif.
        "Pour chaque prédiction, l'application montre les éléments qui ont le plus pesé dans la décision, grâce à une "
        "méthode appelée **SHAP**. Quand il s'agit d'un motif chimique, elle affiche le fragment de la molécule concerné. "
        "Pour la vanilline, par exemple, le modèle s'appuie sur sa fonction aldéhyde et son motif méthoxyphénol pour "
        "prédire une note gourmande : le même raisonnement qu'un chimiste."
    )

    st.subheader("L'analyse détaillée", anchor=False)  # Renvoi vers le notebook.
    st.markdown(  # Texte explicatif.
        "Le notebook `notebooks/analyse.ipynb` du dépôt explore les données (familles fréquentes et rares, informations "
        "manquantes, familles qui vont ensemble), analyse les erreurs du modèle et calcule la marge d'erreur du score. "
        "Il montre notamment que l'écart avec la référence publiée est solide, mais que XGBoost et la forêt aléatoire "
        "sont à égalité, et que la famille « vert / herbacé », très fréquente, est la principale source de confusion."
    )

    st.subheader("Ce que le modèle ne sait pas faire", anchor=False)  # Limites.
    st.markdown(  # Texte explicatif.
        "- Les familles olfactives viennent de descriptions humaines, par nature subjectives : deux experts ne "
        "décrivent pas toujours une odeur de la même façon.\n"
        "- Le modèle se trompe encore souvent sur certaines familles, comme « épicé » ou « gourmand » : l'onglet "
        "Performances détaille les scores famille par famille.\n"
        "- Il est moins à l'aise avec les très petites molécules (moins de 10 atomes hors hydrogène) et avec les molécules "
        "éloignées de celles qu'il a apprises ; la section « Fiabilité de cette prédiction » donne ce repère pour chaque molécule.\n"
        "- C'est un outil d'exploration : il ne remplace ni un panel d'experts, ni une évaluation de sécurité."
    )
