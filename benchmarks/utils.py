"""
ThreadX Benchmark Utils - Outils de benchmarking et mesures
==========================================================

Utilitaires pour les benchmarks CPU/GPU et KPI gates:
- Gestion du temps: perf_ns(), gpu_timer()
- Helpers IO: write_csv(), write_md()
- Formatage: now_tag() pour horodatage uniforme
- Hashing stable: stable_hash(), hash_series() pour déterminisme
- Informations environnement: env_snapshot()

Utilisé par tools/benchmarks_cpu_gpu.py et tests/test_kpi_gates.py.
"""

import time
import os
import json
import hashlib
import datetime
import platform
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, ContextManager
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd

# Import ThreadX modules
from threadx.utils.log import get_logger
from threadx.utils.xp import CUPY_AVAILABLE

logger = get_logger(__name__)

# Si CuPy est disponible, l'importer
if CUPY_AVAILABLE:
    import cupy as cp

# Constants for KPIs
KPI_SPEEDUP_THRESHOLD = 3.0  # GPU doit être 3x plus rapide que CPU
KPI_CACHE_HIT_THRESHOLD = 0.8  # 80% hit rate minimum
KPI_PARETO_TOLERANCE = 0.05  # ±5% tolérance pour Pareto


def now_tag() -> str:
    """
    Génère un tag d'horodatage au format YYYYMMDD_HHMM.

    Returns:
        str: Timestamp formaté (ex: "20251005_1422")

    Example:
        >>> now_tag()
        '20251005_1422'
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")


def perf_ns() -> Callable[[], int]:
    """
    Crée un wrapper autour de time.perf_counter_ns() pour mesure précise.

    Returns:
        Callable[[], int]: Fonction qui retourne le temps actuel en nanoseconds

    Example:
        >>> timer = perf_ns()
        >>> start = timer()
        >>> # code à mesurer
        >>> end = timer()
        >>> elapsed_ns = end - start
    """
    return time.perf_counter_ns


@contextmanager
def gpu_timer() -> ContextManager:
    """
    Context manager pour chronométrer les opérations GPU avec CuPy Events.
    Fallback sur perf_counter_ns() si CuPy n'est pas disponible.

    Yields:
        None: Utilisé comme context manager

    Example:
        >>> with gpu_timer() as elapsed_ms:
        >>>     # Opération GPU
        >>> print(f"Opération GPU: {elapsed_ms:.3f} ms")
    """
    if CUPY_AVAILABLE:
        # Utiliser CuPy Events pour timing précis des GPU kernels
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        start_event.record()

        try:
            yield lambda: end_event.elapsed_time(start_event)
            end_event.record()
            end_event.synchronize()
        except Exception as e:
            # Fallback en cas d'erreur
            logger.error(f"Erreur lors du timer GPU: {e}")
            yield lambda: 0.0
    else:
        # Fallback CPU si CuPy n'est pas disponible
        start = time.perf_counter()
        yield lambda: (time.perf_counter() - start) * 1000  # ms


def stable_hash(obj: Any) -> str:
    """
    Calcule un hash stable pour un objet (dict, list, etc.).
    Utile pour vérifier le déterminisme des résultats.

    Args:
        obj: Objet à hasher (doit être serializable)

    Returns:
        str: Hash hexadécimal SHA1

    Example:
        >>> stable_hash({'b': 2, 'a': 1})
        '5f8c3ec3b38fb373bd51f49bb3c7f6f94226f65d'
    """
    if isinstance(obj, dict):
        # Trier les clés pour garantir la stabilité
        return stable_hash([(k, obj[k]) for k in sorted(obj.keys())])
    elif isinstance(obj, (list, tuple)):
        # Hasher récursivement les éléments
        return hashlib.sha1(
            json.dumps([stable_hash(item) for item in obj]).encode()
        ).hexdigest()
    else:
        # Convertir en JSON et hasher
        return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def hash_series(arr: np.ndarray) -> str:
    """
    Calcule un hash SHA256 d'un numpy array pour vérifier le déterminisme.

    Args:
        arr: Array numpy à hasher

    Returns:
        str: Hash hexadécimal SHA256

    Example:
        >>> hash_series(np.array([1.0, 2.0, 3.0]))
        '4b45e1e17afd325f8e420a33cf62b211...'
    """
    if arr is None:
        return "none"

    # Contourner les problèmes d'ordre des bytes avec np.ascontiguousarray
    arr_c = np.ascontiguousarray(arr)

    # Hasher les bytes directement pour éviter les problèmes de représentation str
    return hashlib.sha256(arr_c.tobytes()).hexdigest()


def write_csv(
    path: Union[str, Path], rows: List[Dict[str, Any]], mode: str = "w"
) -> None:
    """
    Écrit des données dans un fichier CSV de façon atomique.

    Args:
        path: Chemin du fichier CSV
        rows: Liste de dictionnaires à écrire (chaque dict = une ligne)
        mode: Mode d'écriture ('w' pour écraser, 'a' pour append)

    Example:
        >>> write_csv(
        ...     'benchmarks/results/bench_cpu_gpu_20251005_1422.csv',
        ...     [{'indicator': 'bollinger', 'N': 10000, 'device': 'cpu', 'mean_ms': 15.2}]
        ... )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Écrire d'abord dans un fichier temporaire
    temp_path = path.with_suffix(".tmp")

    df = pd.DataFrame(rows)
    df.to_csv(temp_path, index=False, mode=mode)

    # Renommer de façon atomique
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)

    logger.info(f"CSV écrit: {path}")


def write_md(path: Union[str, Path], text: str, mode: str = "w") -> None:
    """
    Écrit du texte dans un fichier Markdown de façon atomique.

    Args:
        path: Chemin du fichier Markdown
        text: Texte à écrire
        mode: Mode d'écriture ('w' pour écraser, 'a' pour append)

    Example:
        >>> write_md(
        ...     'benchmarks/reports/REPORT_20251005_1422.md',
        ...     '# Rapport de Benchmark\n\nCPU vs GPU...'
        ... )
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Écrire d'abord dans un fichier temporaire
    temp_path = path.with_suffix(".tmp")

    with open(temp_path, mode, encoding="utf-8") as f:
        f.write(text)

    # Renommer de façon atomique
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)

    logger.info(f"Markdown écrit: {path}")


def env_snapshot() -> Dict[str, str]:
    """
    Capture un snapshot de l'environnement d'exécution.

    Returns:
        Dict[str, str]: Informations sur l'environnement

    Example:
        >>> env_snapshot()
        {
            'python_version': '3.10.8',
            'numpy_version': '1.24.3',
            'cupy_version': '12.2.0',
            'os': 'Windows-11-10.0.22631-SP0',
            'cpu': 'Intel64 Family 6 Model 186 Stepping 2',
            'gpu': 'NVIDIA RTX 5090',
            'devices': ['cpu', 'RTX 5090', 'RTX 2060']
        }
    """
    env = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cupy_version": cp.__version__ if CUPY_AVAILABLE else "N/A",
        "os": platform.platform(),
        "cpu": platform.processor(),
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "hostname": platform.node(),
    }

    # Ajouter des informations GPU si disponible
    if CUPY_AVAILABLE:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()
            devices = []

            for i in range(device_count):
                cp.cuda.runtime.setDevice(i)
                device_props = cp.cuda.runtime.getDeviceProperties(i)
                devices.append(f"{device_props['name'].decode('utf-8')}")

            if devices:
                env["gpu"] = devices[0]  # Premier GPU comme principal
                env["devices"] = ["cpu"] + devices
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos GPU: {e}")
            env["gpu"] = "Error"
            env["devices"] = ["cpu"]
    else:
        env["gpu"] = "N/A"
        env["devices"] = ["cpu"]

    return env
