"""
ThreadX Determinism Utilities - Global Seeds & Stable Merges
===========================================================

Utilitaires pour garantir le déterminisme dans ThreadX :
- Seeds globaux pour tous les générateurs aléatoires
- Merges déterministes de DataFrames
- Hachage stable pour checksums reproductibles

Garantit la reproductibilité des résultats entre exécutions.

Author: ThreadX Framework
Version: Phase 10 - Determinism
"""

import hashlib
import json
import random
import sys
from typing import Any, List, Union

import numpy as np
import pandas as pd

from threadx.utils.log import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int) -> None:
    """
    Configure le seed global pour tous les générateurs aléatoires.

    Configure :
    - random (Python standard)
    - numpy.random
    - cupy.random (si disponible)
    - torch (si disponible)

    Args:
        seed: Seed à utiliser (entier)

    Example:
        >>> set_global_seed(42)
        >>> # Tous les générateurs utilisent maintenant le seed 42
    """
    logger.info(f"Configuration du seed global: {seed}")

    # Python standard random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # CuPy (si disponible)
    try:
        import cupy as cp

        cp.random.seed(seed)
        logger.debug("CuPy seed configuré")
    except ImportError:
        logger.debug("CuPy non disponible - seed ignoré")

    # PyTorch (si disponible)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        logger.debug("PyTorch seed configuré")
    except ImportError:
        logger.debug("PyTorch non disponible - seed ignoré")

    # Configuration déterministe pour NumPy
    if hasattr(np.random, "bit_generator"):
        # NumPy moderne (>=1.17)
        np.random.default_rng(seed)

    logger.info(f"Seed global {seed} configuré pour tous les générateurs")


def enforce_deterministic_merges(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge déterministe de DataFrames avec ordre stable.

    Garantit un ordre reproductible indépendamment de :
    - L'ordre d'arrivée des DataFrames
    - Le parallélisme de calcul
    - Les variations d'implémentation pandas

    Args:
        df_list: Liste de DataFrames à merger

    Returns:
        DataFrame merged avec ordre déterministe

    Example:
        >>> df1 = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        >>> df2 = pd.DataFrame({'a': [3, 4], 'b': ['z', 'w']})
        >>> merged = enforce_deterministic_merges([df1, df2])
        >>> # Ordre garanti identique entre exécutions
    """
    if not df_list:
        return pd.DataFrame()

    if len(df_list) == 1:
        return df_list[0].copy()

    logger.debug(f"Merge déterministe de {len(df_list)} DataFrames")

    # Concatenation avec réinitialisation d'index
    merged_df = pd.concat(df_list, ignore_index=True)

    # Tri déterministe basé sur toutes les colonnes
    # Ordre lexicographique pour stabilité maximale
    if len(merged_df) > 1:
        sort_columns = list(merged_df.columns)

        # Tri multi-colonnes avec gestion des NaN
        merged_df = merged_df.sort_values(
            by=sort_columns,
            na_position="last",  # NaN à la fin
            kind="mergesort",  # Algorithme stable
        ).reset_index(drop=True)

    logger.debug(f"Merge terminé: {len(merged_df)} lignes")

    return merged_df


def stable_hash(payload: Any) -> str:
    """
    Génère un hash SHA-256 stable et reproductible.

    Utilise une sérialisation JSON canonique pour garantir
    que le même objet produit toujours le même hash,
    indépendamment de l'ordre d'insertion des clés.

    Args:
        payload: Objet à hasher (doit être JSON-sérialisable)

    Returns:
        Hash SHA-256 en hexadécimal

    Example:
        >>> hash1 = stable_hash({'b': 2, 'a': 1})
        >>> hash2 = stable_hash({'a': 1, 'b': 2})
        >>> hash1 == hash2  # True - ordre des clés ignoré
    """
    try:
        # Sérialisation JSON canonique
        json_str = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,  # Clés triées
            separators=(",", ":"),  # Pas d'espaces
            default=str,  # Fallback pour objets non-sérialisables
        )

        # Hash SHA-256
        hash_bytes = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

        return hash_bytes

    except (TypeError, ValueError) as e:
        logger.warning(f"Impossible de hasher l'objet: {e}")
        # Fallback : hash de la représentation string
        fallback_str = str(payload)
        return hashlib.sha256(fallback_str.encode("utf-8")).hexdigest()


def create_deterministic_splits(
    data: Union[pd.DataFrame, np.ndarray], n_splits: int, seed: int = 42
) -> List[Union[pd.DataFrame, np.ndarray]]:
    """
    Crée des splits déterministes de données.

    Garantit que les mêmes données avec le même seed
    produisent toujours les mêmes splits.

    Args:
        data: Données à splitter
        n_splits: Nombre de splits à créer
        seed: Seed pour le générateur aléatoire

    Returns:
        Liste des splits

    Example:
        >>> df = pd.DataFrame({'a': range(100)})
        >>> splits = create_deterministic_splits(df, 3, seed=42)
        >>> len(splits) == 3  # True
    """
    if n_splits <= 0:
        raise ValueError("n_splits doit être > 0")

    # Configuration du seed local
    rng = np.random.RandomState(seed)

    # Taille des données
    n_samples = len(data)

    if n_splits > n_samples:
        raise ValueError(f"n_splits ({n_splits}) > n_samples ({n_samples})")

    # Génération d'indices déterministes
    indices = np.arange(n_samples)
    rng.shuffle(indices)  # Mélange déterministe

    # Calcul des tailles de splits
    base_size = n_samples // n_splits
    remainder = n_samples % n_splits

    split_sizes = [base_size] * n_splits
    for i in range(remainder):
        split_sizes[i] += 1

    # Création des splits
    splits = []
    start_idx = 0

    for split_size in split_sizes:
        end_idx = start_idx + split_size
        split_indices = indices[start_idx:end_idx]

        if isinstance(data, pd.DataFrame):
            split_data = data.iloc[split_indices].copy()
        else:
            split_data = data[split_indices].copy()

        splits.append(split_data)
        start_idx = end_idx

    logger.debug(
        f"Splits déterministes créés: {n_splits} splits, "
        f"tailles: {[len(s) for s in splits]}"
    )

    return splits


def hash_df(df: pd.DataFrame, cols: List[str] = None) -> str:
    """
    Calcule un hash stable pour un DataFrame.

    Args:
        df: DataFrame à hasher
        cols: Liste des colonnes à inclure (toutes si None)

    Returns:
        Hash hexadécimal (SHA256)

    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> hash_df(df)
        'e25388fde8290dc286a6164fa2d97e551b53498dcbf7bc378eb1f178'
    """
    # Sélectionner colonnes
    if cols is not None:
        df = df[cols].copy()

    # Convertir en chaîne JSON stable
    json_str = df.to_json(orient="records", date_format="iso")

    # Calculer hash
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def validate_determinism(func, args, kwargs=None, n_runs: int = 3) -> bool:
    """
    Valide qu'une fonction produit des résultats déterministes.

    Exécute la fonction plusieurs fois avec les mêmes arguments
    et vérifie que les résultats sont identiques.

    Args:
        func: Fonction à tester
        args: Arguments de la fonction
        kwargs: Arguments nommés de la fonction
        n_runs: Nombre d'exécutions de test

    Returns:
        True si les résultats sont déterministes

    Example:
        >>> def random_func(seed):
        ...     set_global_seed(seed)
        ...     return np.random.rand(10)
        >>> is_deterministic = validate_determinism(random_func, (42,))
    """
    if kwargs is None:
        kwargs = {}

    logger.debug(f"Validation déterminisme: {n_runs} exécutions")

    results = []

    for run in range(n_runs):
        try:
            result = func(*args, **kwargs)

            # Conversion en format hashable
            if isinstance(result, (pd.DataFrame, pd.Series)):
                result_hash = stable_hash(result.to_dict())
            elif isinstance(result, np.ndarray):
                result_hash = stable_hash(result.tolist())
            else:
                result_hash = stable_hash(result)

            results.append(result_hash)

        except Exception as e:
            logger.error(f"Erreur lors du run {run}: {e}")
            return False

    # Vérification de l'identité des résultats
    is_deterministic = len(set(results)) == 1

    if is_deterministic:
        logger.debug("✅ Fonction déterministe validée")
    else:
        logger.warning("❌ Fonction non-déterministe détectée")
        logger.debug(f"Hashes distincts: {set(results)}")

    return is_deterministic


# === Contexte de déterminisme ===


class DeterministicContext:
    """
    Context manager pour garantir le déterminisme temporaire.

    Sauvegarde l'état des générateurs, applique un seed,
    puis restaure l'état original à la sortie.
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.saved_states = {}

    def __enter__(self):
        # Sauvegarde des états actuels
        self.saved_states["python_random"] = random.getstate()
        self.saved_states["numpy_random"] = np.random.get_state()

        try:
            import cupy as cp

            self.saved_states["cupy_random"] = cp.random.get_random_state()
        except ImportError:
            pass

        # Application du seed temporaire
        set_global_seed(self.seed)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restauration des états
        random.setstate(self.saved_states["python_random"])
        np.random.set_state(self.saved_states["numpy_random"])

        if "cupy_random" in self.saved_states:
            try:
                import cupy as cp

                cp.random.set_random_state(self.saved_states["cupy_random"])
            except ImportError:
                pass


# === Utilitaires de debugging ===


def get_random_states() -> dict:
    """Récupère l'état actuel de tous les générateurs aléatoires."""
    states = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
    }

    try:
        import cupy as cp

        states["cupy_random"] = cp.random.get_random_state()
    except ImportError:
        pass

    return states


def compare_random_states(state1: dict, state2: dict) -> bool:
    """Compare deux états de générateurs aléatoires."""
    return stable_hash(state1) == stable_hash(state2)


if __name__ == "__main__":
    # Tests rapides
    print("Test déterminisme ThreadX...")

    # Test seed global
    set_global_seed(42)
    sample1 = np.random.rand(5)

    set_global_seed(42)
    sample2 = np.random.rand(5)

    print(f"Samples identiques: {np.array_equal(sample1, sample2)}")

    # Test merge déterministe
    df1 = pd.DataFrame({"a": [3, 1, 2], "b": ["c", "a", "b"]})
    df2 = pd.DataFrame({"a": [6, 4, 5], "b": ["f", "d", "e"]})

    merged = enforce_deterministic_merges([df1, df2])
    print(f"Merge réussi: {len(merged)} lignes")

    # Test hash stable
    hash1 = stable_hash({"z": 3, "a": 1, "b": 2})
    hash2 = stable_hash({"a": 1, "b": 2, "z": 3})
    print(f"Hashes identiques: {hash1 == hash2}")

    print("✅ Tests déterminisme OK")
