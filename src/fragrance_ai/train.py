"""Entraîne et compare tous les modèles, puis sauvegarde le modèle déployé.

Lancement : python -m fragrance_ai.train
"""

import argparse  # Lit les options passées en ligne de commande.
import json  # Écrit les métadonnées du modèle au format JSON.
import time  # Mesure la durée de chaque entraînement.

import joblib  # Sauvegarde et recharge efficacement des objets Python (ici, le modèle).
import numpy as np  # Tableaux numériques.
import pandas as pd  # Tableau de résultats.

from fragrance_ai import config  # Chemins et constantes.
from fragrance_ai.data import apply_policy, check_leakage, load_split  # Données, contrôles et stratégies de labels.
from fragrance_ai.catalog import save_test_examples  # Molécules surprises de l'application.
from fragrance_ai.domain import save_reference  # Référence pour le domaine d'applicabilité.
from fragrance_ai.evaluate import evaluate, tune_thresholds  # Métriques et seuils.
from fragrance_ai.features import featurize  # Vectorisation des molécules.
from fragrance_ai.models import MODEL_NAMES, TREE_MODELS, MultiLabelModel  # Modèles à arbres.
from fragrance_ai.neural import NeuralModel  # Réseaux de neurones (MLP, GNN).

ALL_MODELS = [*MODEL_NAMES, "mlp", "gnn"]  # Les quatre approches comparées.
DEFAULT_RUNS = [(m, p) for m in ["random_forest", "xgboost", "mlp"] for p in config.POLICIES] + [("gnn", "drop")]
# Le GNN n'est entraîné qu'avec « drop » : c'est la stratégie gagnante des autres modèles, et le GNN est le plus coûteux.


def load_features(split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Charge un split et calcule ses variables, avec un cache sur disque pour ne pas tout recalculer."""
    df = load_split(split)  # Lit et contrôle le CSV officiel.
    cache = config.DATA_DIR / f"features_{split}.npz"  # Fichier de cache des variables de ce split.
    if cache.exists():  # Si les variables ont déjà été calculées…
        saved = np.load(cache)  # …on les relit (quelques millisecondes au lieu d'une minute).
        X, valid = saved["X"], saved["valid"]  # Matrice des variables et masque des molécules valides.
    else:  # Sinon…
        X, valid = featurize(df[config.SMILES_COL])  # …on calcule les variables avec RDKit…
        np.savez_compressed(cache, X=X, valid=valid)  # …et on les met en cache, compressées.
    if (~valid).any():  # Si certaines molécules n'ont pas pu être lues…
        print(f"[{split}] {(~valid).sum()} SMILES invalides ignorés")  # …on le signale (transparence).
    return X, df[valid].reset_index(drop=True)  # Variables et lignes correspondantes, alignées.


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement AI Fragrance Intelligence")  # Aide en ligne de commande.
    parser.add_argument("--runs", nargs="+", help="Combinaisons modèle:stratégie, ex. xgboost:drop gnn:drop")  # Optionnel.
    parser.add_argument("--no-mlflow", action="store_true")  # Permet de désactiver MLflow si besoin.
    args = parser.parse_args()  # Lit les options réellement fournies.
    runs = [tuple(r.split(":")) for r in args.runs] if args.runs else DEFAULT_RUNS  # Expériences à lancer.

    use_mlflow = not args.no_mlflow  # Vrai si l'on veut tracer les expériences.
    if use_mlflow:  # Configuration de MLflow, uniquement si utilisé.
        import mlflow  # Import local : MLflow n'est pas nécessaire pour faire tourner l'application.

        mlflow.set_tracking_uri(config.MLFLOW_URI)  # Les expériences sont stockées dans mlflow.db.
        mlflow.set_experiment("fragrance-ai")  # Regroupe toutes les exécutions sous un même nom.

    X_train, df_train = load_features("train")  # Variables et labels d'entraînement.
    X_val, df_val = load_features("val")  # Validation : sert à choisir les seuils et le modèle.
    X_test, df_test = load_features("test")  # Test : utilisé une seule fois par modèle, pour le score final.
    check_leakage({"train": df_train, "val": df_val, "test": df_test})  # Aucune molécule partagée entre splits.
    print("Contrôle anti-fuite : aucune molécule commune entre train, val et test.")  # Confirmation.
    smiles = {k: d[config.SMILES_COL].tolist() for k, d in [("train", df_train), ("val", df_val), ("test", df_test)]}
    Y_val = df_val[config.LABEL_COLUMNS].to_numpy(dtype=int)  # Labels de validation (aucun NaN).
    Y_test = df_test[config.LABEL_COLUMNS].to_numpy(dtype=int)  # Labels de test (aucun NaN).

    rows, fitted = [], {}  # rows : lignes du tableau de résultats ; fitted : modèles à arbres gardés en mémoire.
    for model_name, policy in runs:  # Boucle sur les expériences.
        start = time.time()  # Démarre le chronomètre.
        Y_train = apply_policy(df_train, policy)  # Construit la cible selon la stratégie.
        if model_name in TREE_MODELS:  # Modèles à arbres : un classifieur par famille.
            model = MultiLabelModel(model_name).fit(X_train, Y_train)  # Entraîne les 12 classifieurs.
            proba_val, proba_test = model.predict_proba(X_val), model.predict_proba(X_test)  # Probabilités.
        else:  # Réseaux de neurones : 12 sorties à la fois, perte masquée.
            model = NeuralModel(model_name).fit(X_train, Y_train, smiles["train"], X_val, Y_val, smiles["val"])
            proba_val = model.predict_proba(X_val, smiles["val"])  # Probabilités de validation.
            proba_test = model.predict_proba(X_test, smiles["test"])  # Probabilités de test.
        thresholds = tune_thresholds(Y_val, proba_val)  # Règle les seuils sur la validation.
        val = evaluate(Y_val, proba_val, thresholds)  # Scores de validation.
        test = evaluate(Y_test, proba_test, thresholds)  # Scores de test, seuils figés.
        duration = time.time() - start  # Durée totale de l'expérience.
        print(  # Affiche un résumé lisible en direct.
            f"{model_name:14s} {policy:12s} val F1={val['macro_f1']:.3f} "
            f"test F1={test['macro_f1']:.3f} PR-AUC={test['macro_pr_auc']:.3f} ({duration:.0f}s)", flush=True
        )
        rows.append({  # Ajoute une ligne au tableau comparatif.
            "model": model_name,  # Type de modèle.
            "policy": policy,  # Stratégie de labels.
            "val_macro_f1": val["macro_f1"],  # Critère de sélection du modèle.
            "test_macro_f1": test["macro_f1"],  # Score final, comparable au 0,42 de l'article.
            "test_macro_precision": test["macro_precision"],  # Précision moyenne.
            "test_macro_recall": test["macro_recall"],  # Rappel moyen.
            "test_macro_pr_auc": test["macro_pr_auc"],  # PR-AUC moyenne.
            "test_macro_auroc": test["macro_auroc"],  # Qualité du classement.
            "test_macro_f1_at_0.5": test["macro_f1_at_0.5"],  # Score sans réglage des seuils.
            "train_seconds": round(duration, 1),  # Coût de calcul.
        })
        if model_name in TREE_MODELS:  # Seuls les modèles à arbres sont candidats au déploiement…
            fitted[(model_name, policy)] = (model, thresholds, test)  # …on les garde en mémoire.
        if use_mlflow:  # Trace l'expérience dans MLflow.
            with mlflow.start_run(run_name=f"{model_name}-{policy}"):  # Une « run » par combinaison.
                mlflow.log_params({"model": model_name, "policy": policy})  # Paramètres de l'expérience.
                mlflow.log_metrics({k: v for k, v in rows[-1].items() if isinstance(v, float)})  # Toutes les métriques.

    results = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)  # Classe par score de VALIDATION.
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)  # Crée reports/ si besoin.
    results.to_csv(config.RESULTS_PATH, index=False)  # Sauvegarde le tableau comparatif.
    print("\n", results.to_string(index=False))  # Affiche le classement complet.

    # Choix du modèle déployé : le meilleur modèle à arbres sur la VALIDATION.
    # On impose un modèle à arbres car l'application doit expliquer chaque prédiction avec SHAP.
    trees = results[results["model"].isin(TREE_MODELS)]  # Ne garde que Random Forest et XGBoost.
    if trees.empty:  # Si l'on n'a lancé que des réseaux de neurones…
        print("Aucun modèle à arbres entraîné : modèle déployé inchangé.")  # …on ne touche pas au modèle existant.
        return
    best = trees.iloc[0]  # Le premier est le meilleur (tableau déjà trié).
    model, thresholds, test = fitted[(best["model"], best["policy"])]  # Récupère l'objet entraîné.

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)  # Crée models/ si besoin.
    joblib.dump(model, config.MODEL_PATH, compress=3)  # Sauvegarde compressée du modèle.
    save_reference(df_train)  # Sauvegarde les fingerprints d'entraînement pour juger la fiabilité de chaque prédiction.
    save_test_examples(df_test)  # Sauvegarde le jeu de test pour les « molécules surprises » de l'application.
    metadata = {  # Informations dont l'API et l'application ont besoin.
        "model": best["model"],  # Type du modèle déployé.
        "policy": best["policy"],  # Stratégie de labels retenue.
        "best_overall_on_validation": f"{results.iloc[0]['model']} / {results.iloc[0]['policy']}",  # Transparence.
        "labels": config.LABEL_COLUMNS,  # Ordre des sorties du modèle.
        "thresholds": dict(zip(config.LABEL_COLUMNS, map(float, thresholds))),  # Seuil de décision par famille.
        "test_metrics": test,  # Scores de test du modèle déployé, détaillés par famille.
        "paper_baseline_macro_f1": config.PAPER_BASELINE_MACRO_F1,  # Référence publiée, pour comparaison.
        "data_source": "OdorNet, Yang et al., Scientific Data (2026), doi:10.1038/s41597-026-08429-z",  # Traçabilité.
        "data_manifest": json.loads((config.DATA_DIR / "manifest.json").read_text()),  # URL et SHA-256 des CSV.
        "limitations": [  # Limites à garder en tête, affichées avec le modèle, en langage clair.
            "Les probabilités affichées servent à comparer les familles entre elles ; ce ne sont pas des certitudes.",
            "Les familles olfactives viennent de descriptions humaines, par nature subjectives.",
            "Une molécule pour laquelle aucune famille n'est retenue n'est pas forcément inodore.",
            "C'est un outil d'exploration : il ne remplace ni un panel d'experts, ni une évaluation de sécurité.",
        ],
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))  # Écrit le JSON lisible.
    print(f"\nModèle déployé : {best['model']} / {best['policy']} — test macro-F1 = {test['macro_f1']:.3f}")  # Bilan.


if __name__ == "__main__":  # Exécuté seulement quand le fichier est lancé directement…
    main()  # …et non quand il est importé par un autre module.
