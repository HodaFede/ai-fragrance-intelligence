"""Deep learning : un MLP sur fingerprints et un GNN sur le graphe moléculaire.

Les deux utilisent une perte « masquée » : une cellule de label vide ne contribue pas à l'erreur.
C'est la façon correcte d'appliquer la stratégie « drop » à un réseau qui prédit les 12 familles à la fois.
Écrit avec JAX (calcul différentiable de Google) et Optax (optimiseurs), légers et rapides sur CPU.
"""

import jax  # Différentiation automatique et compilation du calcul.
import jax.numpy as jnp  # Équivalent de NumPy, mais différentiable et compilable.
import numpy as np  # Préparation des données côté CPU.
import optax  # Optimiseurs (Adam) et fonctions utilitaires.
from rdkit import Chem  # Lecture des molécules pour construire les graphes.

from fragrance_ai import config  # Graine aléatoire.

# --- Représentation d'une molécule en graphe ---------------------------------

ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]  # Éléments courants ; les autres vont dans « autre ».
HYBRIDIZATIONS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]  # États d'hybridation.
ATOM_FEATURES = (len(ELEMENTS) + 1) + 6 + 5 + (len(HYBRIDIZATIONS) + 1) + 1 + 3  # Taille du vecteur décrivant un atome (29).


def one_hot(value, choices: list) -> list[float]:
    """Encodage « un parmi n » ; la dernière case signifie « autre »."""
    vec = [0.0] * (len(choices) + 1)  # Une case par choix, plus une case « autre ».
    vec[choices.index(value) if value in choices else len(choices)] = 1.0  # Allume la bonne case.
    return vec  # Liste de 0 et un seul 1.


def atom_vector(atom) -> list[float]:
    """Décrit un atome par 29 nombres : élément, voisinage, hydrogènes, hybridation, charge, cycle."""
    return (
        one_hot(atom.GetSymbol(), ELEMENTS)  # Quel élément (10 cases).
        + one_hot(min(atom.GetDegree(), 5), list(range(5)))  # Nombre de voisins, plafonné à 5 (6 cases).
        + one_hot(min(atom.GetTotalNumHs(), 3), list(range(4)))  # Nombre d'hydrogènes portés, plafonné à 3 (5 cases).
        + one_hot(atom.GetHybridization(), HYBRIDIZATIONS)  # sp, sp2, sp3 ou autre (4 cases).
        + [float(atom.GetIsAromatic())]  # 1 si l'atome est aromatique.
        + [float(atom.IsInRing()), float(atom.GetFormalCharge()), atom.GetMass() / 100]  # Cycle, charge, masse réduite.
    )


def mol_to_graph(smiles: str) -> tuple[np.ndarray, np.ndarray]:
    """Convertit un SMILES en (caractéristiques des atomes, matrice d'adjacence normalisée)."""
    mol = Chem.MolFromSmiles(smiles)  # Lit la molécule.
    nodes = np.array([atom_vector(a) for a in mol.GetAtoms()], dtype=np.float32)  # Une ligne par atome.
    n = len(nodes)  # Nombre d'atomes.
    adj = np.eye(n, dtype=np.float32)  # Chaque atome est relié à lui-même (il garde sa propre information).
    for bond in mol.GetBonds():  # Pour chaque liaison chimique…
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()  # …ses deux atomes…
        adj[i, j] = adj[j, i] = 1.0  # …sont reliés dans les deux sens.
    adj /= adj.sum(axis=1, keepdims=True)  # Normalisation : chaque atome fait la MOYENNE de ses voisins.
    return nodes, adj  # Graphe prêt pour le réseau.


def pad_batch(graphs: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empile des graphes de tailles différentes dans des tableaux de taille fixe (avec masque)."""
    size = int(np.ceil(max(len(g[0]) for g in graphs) / 16) * 16)  # Taille arrondie à 16 : limite les recompilations.
    nodes = np.zeros((len(graphs), size, ATOM_FEATURES), dtype=np.float32)  # Atomes, complétés par des zéros.
    adj = np.zeros((len(graphs), size, size), dtype=np.float32)  # Adjacences, complétées par des zéros.
    mask = np.zeros((len(graphs), size), dtype=np.float32)  # 1 pour un vrai atome, 0 pour du remplissage.
    for k, (x, a) in enumerate(graphs):  # Recopie chaque graphe dans le coin supérieur gauche.
        n = len(x)  # Taille réelle.
        nodes[k, :n], adj[k, :n, :n], mask[k, :n] = x, a, 1.0  # Copie et marquage.
    return nodes, adj, mask  # Trois tableaux alignés.


# --- Architectures -----------------------------------------------------------


def dense_init(key, n_in: int, n_out: int) -> dict:
    """Initialise une couche dense (initialisation de He, adaptée à ReLU)."""
    return {"w": jax.random.normal(key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in), "b": jnp.zeros(n_out)}  # Poids et biais.


def dense(p: dict, x):
    """Applique une couche dense : x·W + b."""
    return x @ p["w"] + p["b"]  # Produit matriciel plus biais.


def init_mlp(key, n_in: int) -> dict:
    """MLP : entrée -> 512 -> 256 -> 12 sorties."""
    k1, k2, k3 = jax.random.split(key, 3)  # Une clé aléatoire différente par couche.
    return {"l1": dense_init(k1, n_in, 512), "l2": dense_init(k2, 512, 256),  # Deux couches cachées.
            "out": dense_init(k3, 256, len(config.LABEL_COLUMNS))}  # Une sortie par famille.


def mlp_forward(params, x, key, train: bool):
    """Passe avant du MLP, avec dropout pendant l'entraînement."""
    for name in ["l1", "l2"]:  # Pour chaque couche cachée…
        x = jax.nn.relu(dense(params[name], x))  # …transformation linéaire puis activation ReLU…
        if train:  # …et, à l'entraînement seulement…
            key, sub = jax.random.split(key)  # …une nouvelle clé aléatoire…
            x = x * jax.random.bernoulli(sub, 0.8, x.shape) / 0.8  # …dropout : éteint 20 % des neurones au hasard.
    return dense(params["out"], x)  # Logits (scores avant sigmoïde), un par famille.


def init_gnn(key, hidden: int = 128, layers: int = 3) -> dict:
    """GNN : projection des atomes, 3 couches de passage de messages, lecture, tête de classification."""
    keys = jax.random.split(key, layers + 3)  # Clés aléatoires pour chaque couche.
    return {
        "embed": dense_init(keys[0], ATOM_FEATURES, hidden),  # Projette les 29 caractéristiques d'atome.
        "convs": [dense_init(keys[1 + i], hidden, hidden) for i in range(layers)],  # Couches de passage de messages.
        "head": dense_init(keys[-2], 2 * hidden, hidden),  # Couche après la lecture (moyenne + maximum).
        "out": dense_init(keys[-1], hidden, len(config.LABEL_COLUMNS)),  # Une sortie par famille.
    }


def gnn_forward(params, nodes, adj, mask, key, train: bool):
    """Passe avant du GNN : chaque atome agrège progressivement l'information de ses voisins."""
    m = mask[..., None]  # Masque élargi pour multiplier les vecteurs d'atomes.
    h = jax.nn.relu(dense(params["embed"], nodes)) * m  # Représentation initiale de chaque atome.
    for conv in params["convs"]:  # À chaque couche, le « rayon de vision » d'un atome grandit d'une liaison.
        h = (h + jax.nn.relu(dense(conv, adj @ h))) * m  # Moyenne des voisins, transformation, connexion résiduelle.
    mean = h.sum(1) / jnp.maximum(mask.sum(1, keepdims=True), 1.0)  # Lecture 1 : moyenne sur les atomes réels.
    maxi = jnp.max(jnp.where(m > 0, h, -1e9), axis=1)  # Lecture 2 : maximum, sensible à un groupe fonctionnel précis.
    z = jax.nn.relu(dense(params["head"], jnp.concatenate([mean, maxi], axis=-1)))  # Vecteur de la molécule entière.
    if train:  # Dropout à l'entraînement seulement.
        z = z * jax.random.bernoulli(key, 0.8, z.shape) / 0.8  # Éteint 20 % des neurones.
    return dense(params["out"], z)  # Logits, un par famille.


# --- Perte masquée -----------------------------------------------------------


def masked_bce(logits, targets, observed, pos_weight):
    """Entropie croisée binaire moyennée uniquement sur les labels observés.

    observed vaut 0 pour une cellule vide : elle n'influence pas l'apprentissage (stratégie « drop »).
    pos_weight renforce l'erreur sur les positifs, rares pour la plupart des familles.
    """
    log_p, log_not_p = jax.nn.log_sigmoid(logits), jax.nn.log_sigmoid(-logits)  # log(p) et log(1-p), stables numériquement.
    loss = -(pos_weight * targets * log_p + (1 - targets) * log_not_p)  # Erreur par molécule et par famille.
    return (loss * observed).sum() / jnp.maximum(observed.sum(), 1.0)  # Moyenne sur les seules cellules connues.


# --- Modèle entraînable ------------------------------------------------------


class NeuralModel:
    """Interface commune pour le MLP et le GNN : fit(...) puis predict_proba(...)."""

    def __init__(self, kind: str, max_epochs: int = 60, patience: int = 8, batch_size: int = 128, lr: float = 1e-3):
        if kind not in {"mlp", "gnn"}:  # Seules deux architectures existent.
            raise ValueError(f"Architecture inconnue : {kind}")
        self.kind, self.max_epochs, self.patience = kind, max_epochs, patience  # Type et arrêt anticipé.
        self.batch_size, self.lr = batch_size, lr  # Taille des lots et pas d'apprentissage.

    # Préparation des entrées : vecteurs standardisés (MLP) ou graphes (GNN).
    def _prepare(self, X, smiles, fit_stats: bool = False):
        if self.kind == "gnn":  # Le GNN ne lit que la structure.
            return [mol_to_graph(s) for s in smiles]  # Liste de graphes.
        X = np.asarray(X, dtype=np.float32)  # Copie de travail.
        if fit_stats:  # Les statistiques sont calculées sur l'entraînement uniquement (pas de fuite).
            self.median_ = np.nanmedian(X, axis=0)  # Médiane par variable, pour combler les NaN.
            filled = np.where(np.isnan(X), self.median_, X)  # Données sans NaN.
            self.mean_, self.std_ = filled.mean(0), filled.std(0) + 1e-6  # Moyenne et écart-type par variable.
        X = np.where(np.isnan(X), self.median_, X)  # Comble les NaN.
        return np.clip((X - self.mean_) / self.std_, -10, 10)  # Standardise et borne les valeurs extrêmes.

    def _batch(self, inputs, idx):
        """Extrait un lot d'entrées prêt pour le réseau."""
        if self.kind == "gnn":  # Graphes : on les empile avec remplissage.
            return pad_batch([inputs[i] for i in idx])  # (atomes, adjacence, masque).
        return (inputs[idx],)  # Vecteurs : simple sélection de lignes.

    def _forward(self, params, batch, key, train):
        """Appelle la bonne architecture."""
        if self.kind == "gnn":  # GNN.
            return gnn_forward(params, *batch, key, train)  # Trois entrées : atomes, adjacence, masque.
        return mlp_forward(params, batch[0], key, train)  # MLP : une entrée.

    def fit(self, X, Y, smiles, X_val, Y_val, smiles_val) -> "NeuralModel":
        """Entraîne avec Adam et arrêt anticipé sur la perte de validation."""
        train_in = self._prepare(X, smiles, fit_stats=True)  # Entrées d'entraînement.
        val_in = self._prepare(X_val, smiles_val)  # Entrées de validation (mêmes statistiques).
        observed = (~np.isnan(Y)).astype(np.float32)  # 1 = label connu, 0 = cellule vide.
        targets = np.nan_to_num(Y, nan=0.0).astype(np.float32)  # Valeur sans importance là où observed = 0.
        pos_rate = (targets * observed).sum(0) / observed.sum(0)  # Proportion de positifs par famille.
        pos_weight = jnp.asarray(np.clip((1 - pos_rate) / pos_rate, 1.0, 10.0))  # Poids des positifs, plafonné à 10.

        key = jax.random.PRNGKey(config.RANDOM_STATE)  # Graine : résultats reproductibles.
        key, init_key = jax.random.split(key)  # Clé dédiée à l'initialisation.
        params = init_gnn(init_key) if self.kind == "gnn" else init_mlp(init_key, train_in.shape[1])  # Poids initiaux.
        optimizer = optax.adamw(self.lr, weight_decay=1e-4)  # Adam avec régularisation des poids.
        opt_state = optimizer.init(params)  # État interne de l'optimiseur.

        def loss_fn(p, batch, t, o, k, train):  # Perte d'un lot.
            return masked_bce(self._forward(p, batch, k, train), t, o, pos_weight)  # Logits puis perte masquée.

        @jax.jit  # Compile l'étape d'entraînement : beaucoup plus rapide.
        def step(p, s, batch, t, o, k):
            loss, grads = jax.value_and_grad(loss_fn)(p, batch, t, o, k, True)  # Perte et gradients.
            updates, s = optimizer.update(grads, s, p)  # Calcule la mise à jour des poids.
            return optax.apply_updates(p, updates), s, loss  # Nouveaux poids.

        eval_loss = jax.jit(lambda p, batch, t, o: loss_fn(p, batch, t, o, key, False))  # Perte sans dropout.
        Y_val_obs = np.ones_like(Y_val, dtype=np.float32)  # La validation est entièrement labellisée.
        rng = np.random.default_rng(config.RANDOM_STATE)  # Générateur pour mélanger les molécules.
        best_loss, best_params, waited = np.inf, params, 0  # Suivi du meilleur état.
        n = len(targets)  # Nombre de molécules d'entraînement.
        for epoch in range(self.max_epochs):  # Une époque = un passage sur toutes les molécules.
            order = rng.permutation(n)  # Nouvel ordre aléatoire à chaque époque.
            for start in range(0, n, self.batch_size):  # Parcours par lots.
                idx = order[start:start + self.batch_size]  # Indices du lot.
                key, sub = jax.random.split(key)  # Clé pour le dropout de ce lot.
                params, opt_state, _ = step(params, opt_state, self._batch(train_in, idx),
                                            targets[idx], observed[idx], sub)  # Une mise à jour des poids.
            val_loss = float(np.mean([  # Perte moyenne sur la validation…
                eval_loss(params, self._batch(val_in, idx), Y_val[idx].astype(np.float32), Y_val_obs[idx])
                for idx in np.array_split(np.arange(len(Y_val)), max(1, len(Y_val) // self.batch_size))  # …par lots.
            ]))
            if val_loss < best_loss - 1e-4:  # Amélioration réelle…
                best_loss, best_params, waited = val_loss, params, 0  # …on mémorise ces poids.
            else:  # Pas d'amélioration…
                waited += 1  # …on compte les époques perdues…
                if waited >= self.patience:  # …et on s'arrête après « patience » époques.
                    break
        self.params_, self.epochs_ = best_params, epoch + 1  # Garde les meilleurs poids et le nombre d'époques.
        return self  # Convention : fit renvoie l'objet.

    def predict_proba(self, X, smiles) -> np.ndarray:
        """Probabilités des 12 familles pour chaque molécule."""
        inputs = self._prepare(X, smiles)  # Mêmes prétraitements qu'à l'entraînement.
        n = len(smiles) if self.kind == "gnn" else len(inputs)  # Nombre de molécules.
        key = jax.random.PRNGKey(0)  # Inutilisée sans dropout, mais requise par la signature.
        chunks = [  # Prédiction par lots pour limiter la mémoire.
            jax.nn.sigmoid(self._forward(self.params_, self._batch(inputs, idx), key, False))  # Sigmoïde : logits -> probabilités.
            for idx in np.array_split(np.arange(n), max(1, n // self.batch_size))  # Découpage en lots.
        ]
        return np.asarray(jnp.concatenate(chunks))  # Matrice (n, 12) en NumPy.
