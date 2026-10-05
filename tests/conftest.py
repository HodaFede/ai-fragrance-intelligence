"""Configuration commune des tests : rend le package importable."""

import sys  # Accès au chemin d'import de Python.
from pathlib import Path  # Chemins de fichiers.

ROOT = Path(__file__).resolve().parents[1]  # Racine du dépôt.
sys.path.insert(0, str(ROOT / "src"))  # Permet « import fragrance_ai » sans installation.
sys.path.insert(0, str(ROOT))  # Permet « import api.main ».
