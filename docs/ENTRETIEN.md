# Présenter ce projet en entretien Data Scientist

## Le pitch en trente secondes

J'ai construit un système qui prédit les familles olfactives d'une molécule à partir de sa structure, sur OdorNet, un jeu publié dans *Scientific Data* en octobre 2026. J'ai comparé quatre approches, du gradient boosting au réseau de neurones sur graphe, avec un protocole sans fuite de données. Mon meilleur modèle atteint un macro-F1 de 0,468 sur le test officiel, contre 0,42 pour la référence publiée, et chaque prédiction est expliquée par les sous-structures chimiques qui l'ont motivée. Le tout est déployé avec une API FastAPI, une interface Streamlit et Docker.

## Les questions probables, et quoi répondre

**Pourquoi le macro-F1 et pas l'accuracy ?** Avec douze familles dont certaines rares, une accuracy élevée peut cacher un modèle qui ne prédit jamais les petites classes. Le macro-F1 donne le même poids à chaque famille. Je regarde aussi la PR-AUC, plus informative que l'AUROC quand les positifs sont rares.

**Pourquoi ne pas remplacer les labels manquants par zéro ?** « Non renseigné » ne veut pas dire « absent ». J'ai testé les trois stratégies : ignorer les cellules vides donne le meilleur résultat, et les compter comme positives fait perdre environ 4,5 points de F1. Pour les réseaux de neurones, j'ai implémenté une perte masquée qui ignore ces cellules pendant l'apprentissage.

**Comment avez-vous évité la fuite de données ?** J'utilise le découpage officiel, qui regroupe les stéréoisomères d'une même molécule, et mon pipeline le vérifie à chaque entraînement : il calcule une clé chimique sans stéréochimie et refuse de continuer si une molécule apparaît dans deux ensembles. Les seuils et le choix du modèle se font sur la validation ; le test n'est utilisé qu'à la fin.

**Pourquoi le GNN ne gagne-t-il pas ?** Avec 6 000 molécules, le gradient boosting sur fingerprints de Morgan et descripteurs RDKit reste meilleur (0,468 contre 0,406). Les fingerprints encodent déjà les voisinages atomiques, et le GNN doit apprendre cette représentation à partir de peu de données. C'est un résultat classique en chimie computationnelle sur petits jeux ; un GNN pré-entraîné sur de grandes bases de molécules serait la prochaine piste.

**Votre score de validation est de 0,565 et celui de test de 0,468 : pourquoi cet écart ?** Les seuils de décision sont optimisés sur la validation, donc le score de validation est optimiste. C'est pour cela que je ne communique que le score de test.

**Votre écart avec l'article est-il significatif ?** Oui pour ce qui dépend du jeu de test : un bootstrap de 2 000 rééchantillonnages donne un intervalle de confiance à 95 % de 0,438 à 0,495, entièrement au-dessus de 0,42.

**XGBoost est-il vraiment meilleur que la forêt aléatoire ?** Non, pas significativement : sur des tirages bootstrap appariés, l'écart varie entre −0,011 et +0,019. J'ai gardé XGBoost parce qu'il était meilleur sur la validation et que ses explications SHAP sont rapides, mais je ne le présente pas comme nettement supérieur.

**Où le modèle se trompe-t-il ?** Sur les familles rares, sur les très petites molécules (F1 de 0,29 sous 10 atomes lourds, car elles allument peu de motifs dans l'empreinte), et il ajoute trop souvent « vert / herbacé », la famille la plus fréquente, qui accompagne beaucoup d'autres familles.

**Pouvez-vous affirmer que vous battez l'article ?** Prudemment. Le découpage est le même, mais le réglage des seuils et le traitement des labels ne sont pas identiques à leur protocole. C'est un ordre de grandeur favorable, pas un classement officiel.

**Comment l'utiliser en entreprise ?** Comme aide au tri de molécules candidates avant les tests sensoriels, pour prioriser ce qu'un parfumeur ou un aromaticien va évaluer. Jamais comme substitut à un panel ni à une évaluation réglementaire.

## Ce que je ferais avec plus de temps

Un GNN pré-entraîné (de type MolFormer ou Chemprop sur de grandes bases), une optimisation des hyperparamètres avec Optuna, la calibration des probabilités, et une évaluation sur un découpage par squelette moléculaire, encore plus exigeant.

## Règle d'or

Ne citer sur le CV que les chiffres réellement obtenus, avec le jeu de données et le protocole.
