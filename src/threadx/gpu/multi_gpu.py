"""
ThreadX Multi-GPU Manager - Distribution de Charge
==================================================

Orchestration multi-GPU avec auto-balancing et synchronisation NCCL.

Architecture:
- Split proportionnel des données selon ratios configurables
- Exécution parallèle avec device pinning
- Synchronisation NCCL optionnelle
- Merge déterministe des résultats
- Auto-profiling pour optimisation automatique des ratios
- Fallback CPU transparent

Flux principal:
    Split → Compute → Sync → Merge

Usage:
    >>> manager = MultiGPUManager()  # Balance par défaut 5090:75%, 2060:25%
    >>> result = manager.distribute_workload(data, vectorized_func)
    >>> # Auto-optimisation
    >>> new_ratios = manager.profile_auto_balance()
    >>> manager.set_balance(new_ratios)
"""

import time
import threading
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import numpy as np
import pandas as pd

from threadx.utils.log import get_logger
from threadx.config.settings import S
from .device_manager import (
    is_available,
    list_devices,
    get_device_by_name,
    get_device_by_id,
    check_nccl_support,
    xp,
    DeviceInfo,
    CUPY_AVAILABLE,
)
from .profile_persistence import (
    get_multigpu_ratios,
    update_multigpu_ratios,
    is_profile_valid,
    PROFILES_DIR,
    MULTIGPU_RATIOS_FILE,
)

if CUPY_AVAILABLE:
    import cupy as cp

logger = get_logger(__name__)


# === Exceptions Spécialisées ===


class MultiGPUError(RuntimeError):
    """Erreur base pour operations multi-GPU."""

    pass


class DeviceUnavailableError(MultiGPUError):
    """Device GPU demandé indisponible."""

    pass


class GPUMemoryError(MultiGPUError):
    """Erreur mémoire GPU (OOM)."""

    pass


class ShapeMismatchError(MultiGPUError):
    """Shapes/dtypes incohérents entre chunks."""

    pass


class NonVectorizableFunctionError(MultiGPUError):
    """Fonction non vectorisable ou incompatible."""

    pass


# === Classes utilitaires ===


@dataclass
class WorkloadChunk:
    """
    Chunk de données pour un device spécifique.

    Attributes:
        device_name: Nom du device ("5090", "2060", "cpu")
        data_slice: Slice des données pour ce chunk
        start_idx: Index de début dans les données originales
        end_idx: Index de fin (exclusif)
        expected_size: Taille attendue du résultat
    """

    device_name: str
    data_slice: slice
    start_idx: int
    end_idx: int
    expected_size: int

    def __len__(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class ComputeResult:
    """
    Résultat de calcul d'un chunk.

    Attributes:
        chunk: Chunk original
        result: Résultat du calcul
        compute_time: Temps de calcul en secondes
        device_memory_used: Mémoire utilisée (si disponible)
        error: Exception si erreur
    """

    chunk: WorkloadChunk
    result: Optional[Union[np.ndarray, pd.DataFrame]]
    compute_time: float
    device_memory_used: Optional[float] = None
    error: Optional[Exception] = None

    @property
    def success(self) -> bool:
        return self.error is None


# === Multi-GPU Manager Principal ===


class MultiGPUManager:
    """
    Gestionnaire multi-GPU avec distribution automatique et auto-balancing.

    Gère la répartition proportionnelle de workloads entre GPUs disponibles,
    avec fallback CPU transparent et optimisation automatique des ratios.

    Attributes:
        device_balance: Ratios de charge par device {"device_name": ratio}
        use_streams: Utilisation des CUDA streams par device
        nccl_enabled: Support synchronisation NCCL

    Example:
        >>> # Configuration par défaut (5090: 75%, 2060: 25%)
        >>> manager = MultiGPUManager()
        >>>
        >>> # Distribution d'un workload
        >>> data = np.random.randn(100000, 10)
        >>> result = manager.distribute_workload(data, lambda x: x.sum(axis=1))
        >>>
        >>> # Auto-optimisation
        >>> optimal_ratios = manager.profile_auto_balance(sample_size=50000)
        >>> manager.set_balance(optimal_ratios)
    """

    def __init__(
        self,
        device_balance: Optional[Dict[str, float]] = None,
        use_streams: bool = True,
        enable_nccl: bool = True,
    ):
        """
        Initialise le gestionnaire multi-GPU.

        Args:
            device_balance: Ratios personnalisés {"device": ratio}
                           Si None, utilise balance par défaut (5090:75%, 2060:25%)
            use_streams: Active les CUDA streams pour parallélisme
            enable_nccl: Active synchronisation NCCL si disponible
        """
        self.use_streams = use_streams
        self.nccl_enabled = enable_nccl and check_nccl_support()

        # Détection devices disponibles
        self.available_devices = list_devices()
        self._gpu_devices = [d for d in self.available_devices if d.device_id != -1]
        self._cpu_device = next(
            (d for d in self.available_devices if d.device_id == -1), None
        )

        logger.info(
            f"Multi-GPU Manager initialisé: {len(self._gpu_devices)} GPU(s), "
            f"NCCL={'activé' if self.nccl_enabled else 'désactivé'}"
        )

        # Configuration balance
        if device_balance is None:
            self.device_balance = self._get_default_balance()
        else:
            self.device_balance = device_balance.copy()

        self._validate_balance()

        # Streams GPU (créés à la demande)
        self._device_streams: Dict[str, Any] = {}
        self._lock = threading.Lock()

        logger.info(f"Balance configurée: {self._format_balance()}")

    def _get_default_balance(self) -> Dict[str, float]:
        """Calcule la balance par défaut selon devices disponibles."""
        if not self._gpu_devices:
            return {"cpu": 1.0}

        balance = {}

        # Recherche des devices cibles
        gpu_5090 = get_device_by_name("5090")
        gpu_2060 = get_device_by_name("2060")

        if gpu_5090 and gpu_2060:
            # Configuration idéale RTX 5090 + RTX 2060
            balance["5090"] = 0.75
            balance["2060"] = 0.25
        elif gpu_5090:
            # Seulement RTX 5090
            balance["5090"] = 1.0
        elif gpu_2060:
            # Seulement RTX 2060
            balance["2060"] = 1.0
        else:
            # Autres GPUs: répartition uniforme
            gpu_count = len(self._gpu_devices)
            for device in self._gpu_devices:
                balance[device.name] = 1.0 / gpu_count

        return balance

    def _validate_balance(self) -> None:
        """Valide et normalise la balance."""
        if not self.device_balance:
            raise ValueError("Balance vide")

        # Vérification ratios positifs
        for device, ratio in self.device_balance.items():
            if ratio <= 0:
                raise ValueError(
                    f"Ratio invalide pour {device}: {ratio} (doit être > 0)"
                )

        # Normalisation (somme = 1.0)
        total = sum(self.device_balance.values())
        if abs(total - 1.0) > 1e-6:
            logger.info(f"Normalisation balance: somme {total:.6f} → 1.0")
            for device in self.device_balance:
                self.device_balance[device] /= total

        # Vérification devices disponibles
        for device in self.device_balance:
            if device != "cpu" and not get_device_by_name(device):
                logger.warning(f"Device '{device}' dans balance mais indisponible")

    def _format_balance(self) -> str:
        """Formate la balance pour logging."""
        parts = [f"{dev}:{ratio:.1%}" for dev, ratio in self.device_balance.items()]
        return ", ".join(parts)

    def set_balance(self, new_balance: Dict[str, float]) -> None:
        """
        Met à jour la balance des devices.

        Args:
            new_balance: Nouveaux ratios {"device": ratio}
                        Sera automatiquement normalisé (somme = 1.0)

        Raises:
            ValueError: Si balance invalide

        Example:
            >>> manager.set_balance({"5090": 0.8, "2060": 0.2})
        """
        old_balance = self.device_balance.copy()
        self.device_balance = new_balance.copy()

        try:
            self._validate_balance()
            logger.info(f"Balance mise à jour: {self._format_balance()}")
        except Exception as e:
            self.device_balance = old_balance
            raise ValueError(f"Balance invalide: {e}")

    def _split_workload(
        self, data_size: int, batch_axis: int = 0
    ) -> List[WorkloadChunk]:
        """
        Split proportionnel des données selon balance.

        Args:
            data_size: Taille totale des données
            batch_axis: Axe de split (généralement 0)

        Returns:
            Liste des chunks avec indices corrects
        """
        if data_size == 0:
            return []

        chunks = []
        current_idx = 0

        # Calcul tailles théoriques
        device_names = list(self.device_balance.keys())
        theoretical_sizes = []

        for device_name in device_names[:-1]:  # Tous sauf le dernier
            ratio = self.device_balance[device_name]
            size = int(data_size * ratio)
            theoretical_sizes.append(size)

        # Le dernier récupère le résidu
        remaining = data_size - sum(theoretical_sizes)
        theoretical_sizes.append(remaining)

        # Création des chunks
        for device_name, chunk_size in zip(device_names, theoretical_sizes):
            if chunk_size > 0:
                end_idx = current_idx + chunk_size

                chunk = WorkloadChunk(
                    device_name=device_name,
                    data_slice=slice(current_idx, end_idx),
                    start_idx=current_idx,
                    end_idx=end_idx,
                    expected_size=chunk_size,
                )

                chunks.append(chunk)
                current_idx = end_idx

        # Validation
        total_processed = sum(len(chunk) for chunk in chunks)
        if total_processed != data_size:
            raise RuntimeError(f"Split invalide: {total_processed} != {data_size}")

        logger.debug(
            f"Split workload: {data_size} → {[len(c) for c in chunks]} "
            f"pour {[c.device_name for c in chunks]}"
        )

        return chunks

    def _get_device_stream(self, device_name: str) -> Optional[Any]:
        """Récupère ou crée un stream pour le device."""
        if not self.use_streams or device_name == "cpu" or not CUPY_AVAILABLE:
            return None

        with self._lock:
            if device_name not in self._device_streams:
                device = get_device_by_name(device_name)
                if device and device.device_id != -1:
                    try:
                        with cp.cuda.Device(device.device_id):
                            stream = cp.cuda.Stream()
                            self._device_streams[device_name] = stream
                            logger.debug(f"Stream créé pour {device_name}")
                    except Exception as e:
                        logger.warning(f"Erreur création stream {device_name}: {e}")
                        return None

            return self._device_streams.get(device_name)

    def _compute_chunk(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        chunk: WorkloadChunk,
        func: Callable,
        seed: int,
    ) -> ComputeResult:
        """
        Calcule un chunk sur son device assigné.

        Args:
            data: Données complètes
            chunk: Chunk à traiter
            func: Fonction vectorielle à appliquer
            seed: Seed pour reproductibilité

        Returns:
            ComputeResult avec résultat ou erreur
        """
        start_time = time.time()
        device_memory_used = None

        try:
            # Extraction du chunk
            if isinstance(data, pd.DataFrame):
                chunk_data = data.iloc[chunk.data_slice]
            else:
                chunk_data = data[chunk.data_slice]

            if len(chunk_data) == 0:
                return ComputeResult(chunk, None, 0.0, error=ValueError("Chunk vide"))

            # Configuration device et seed
            if chunk.device_name == "cpu":
                np.random.seed(seed + chunk.start_idx)  # Seed unique par chunk
                xp_module = np
                device_data = chunk_data
            else:
                device = get_device_by_name(chunk.device_name)
                if not device or not CUPY_AVAILABLE:
                    raise DeviceUnavailableError(
                        f"Device {chunk.device_name} indisponible"
                    )

                with cp.cuda.Device(device.device_id):
                    # Seed GPU
                    cp.random.seed(seed + chunk.start_idx)

                    # Stream optionnel
                    stream = self._get_device_stream(chunk.device_name)

                    with cp.cuda.Stream.null if stream is None else stream:
                        # Transfert vers GPU
                        if isinstance(chunk_data, pd.DataFrame):
                            # DataFrame: conversion via numpy
                            numpy_data = chunk_data.values
                            device_data = cp.asarray(numpy_data)
                        else:
                            device_data = cp.asarray(chunk_data)

                        # Mémoire utilisée
                        try:
                            mem_info = cp.cuda.runtime.memGetInfo()
                            device_memory_used = (mem_info[1] - mem_info[0]) / (1024**3)
                        except:
                            pass

                        # Calcul
                        try:
                            result = func(device_data)
                        except Exception as e:
                            raise NonVectorizableFunctionError(
                                f"Fonction échouée sur {chunk.device_name}: {e}"
                            )

                        # Synchronisation si stream
                        if stream is not None:
                            stream.synchronize()

                        # Transfert retour CPU
                        if hasattr(result, "get"):  # CuPy array
                            result = result.get()

                        # Reconstruction DataFrame si nécessaire
                        if isinstance(chunk_data, pd.DataFrame):
                            if result.ndim == 1:
                                # Résultat 1D → Series avec index original
                                result = pd.Series(result, index=chunk_data.index)
                            else:
                                # Résultat 2D → DataFrame avec index original
                                result = pd.DataFrame(result, index=chunk_data.index)

            # Validation résultat
            expected_len = len(chunk_data)
            if hasattr(result, "__len__") and len(result) != expected_len:
                raise ShapeMismatchError(
                    f"Longueur résultat {len(result)} != attendue {expected_len}"
                )

            compute_time = time.time() - start_time

            logger.debug(
                f"Chunk {chunk.device_name}[{chunk.start_idx}:{chunk.end_idx}] "
                f"traité en {compute_time:.3f}s"
            )

            return ComputeResult(
                chunk=chunk,
                result=result,
                compute_time=compute_time,
                device_memory_used=device_memory_used,
            )

        except Exception as e:
            return ComputeResult(
                chunk=chunk, result=None, compute_time=time.time() - start_time, error=e
            )

    def _merge_results(
        self,
        results: List[ComputeResult],
        original_data: Union[np.ndarray, pd.DataFrame],
        batch_axis: int = 0,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Merge déterministe des résultats par ordre des chunks.

        Args:
            results: Résultats des chunks (ordre important)
            original_data: Données originales pour type de référence
            batch_axis: Axe de concaténation

        Returns:
            Résultat merged du même type que original_data
        """
        if not results:
            if isinstance(original_data, pd.DataFrame):
                return pd.DataFrame()
            else:
                return np.array([])

        # Tri par start_idx pour ordre déterministe
        results.sort(key=lambda r: r.chunk.start_idx)

        # Extraction des résultats valides
        valid_results = []
        for result in results:
            if not result.success:
                raise RuntimeError(
                    f"Erreur chunk {result.chunk.device_name}: {result.error}"
                )
            if result.result is not None:
                valid_results.append(result.result)

        if not valid_results:
            if isinstance(original_data, pd.DataFrame):
                return pd.DataFrame()
            else:
                return np.array([])

        # Merge selon type
        if isinstance(original_data, pd.DataFrame):
            if isinstance(valid_results[0], pd.Series):
                # Concaténation de Series
                merged = pd.concat(valid_results, axis=0)
            else:
                # Concaténation de DataFrames
                merged = pd.concat(valid_results, axis=batch_axis)
        else:
            # Concaténation numpy
            merged = np.concatenate(valid_results, axis=batch_axis)

        return merged

    def distribute_workload(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        func: Callable,
        *,
        stream_per_gpu: bool = False,
        batch_axis: int = 0,
        seed: int = 42,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Distribue un workload sur les devices selon balance configurée.

        Architecture: Split → Compute → Sync → Merge

        Args:
            data: Données à traiter (numpy array ou pandas DataFrame)
            func: Fonction vectorielle pure (même input/output shape)
            stream_per_gpu: Force un stream par GPU (défaut: auto)
            batch_axis: Axe de split/merge (défaut: 0)
            seed: Seed pour reproductibilité déterministe

        Returns:
            Résultat merged du même type que data

        Raises:
            DeviceUnavailableError: Device requis indisponible
            GPUMemoryError: Mémoire GPU insuffisante
            ShapeMismatchError: Résultats incohérents
            NonVectorizableFunctionError: Fonction non compatible

        Example:
            >>> data = np.random.randn(100000, 50)
            >>> result = manager.distribute_workload(
            ...     data,
            ...     lambda x: x.sum(axis=1),
            ...     seed=42
            ... )
            >>> assert result.shape == (100000,)
        """
        if stream_per_gpu:
            self.use_streams = True

        start_time = time.time()
        data_size = len(data)

        logger.info(
            f"Distribution workload: {data_size} échantillons, "
            f"fonction {func.__name__ if hasattr(func, '__name__') else 'lambda'}, "
            f"seed={seed}"
        )

        # Cas trivial
        if data_size == 0:
            return data.copy() if hasattr(data, "copy") else data

        # Split en chunks
        chunks = self._split_workload(data_size, batch_axis)

        if len(chunks) == 1:
            # Un seul chunk: exécution directe
            result = self._compute_chunk(data, chunks[0], func, seed)
            if not result.success:
                raise RuntimeError(f"Erreur compute: {result.error}")
            return result.result

        # Exécution parallèle multi-device
        results = []
        max_workers = min(len(chunks), 8)  # Limite raisonnable

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="MultiGPU"
        ) as executor:

            # Soumission des tâches
            future_to_chunk = {}
            for chunk in chunks:
                future = executor.submit(self._compute_chunk, data, chunk, func, seed)
                future_to_chunk[future] = chunk

            # Collecte des résultats
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result(timeout=300)  # 5min timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Erreur chunk {chunk.device_name}: {e}")
                    results.append(ComputeResult(chunk, None, 0.0, error=e))

        # Synchronisation NCCL optionnelle
        if self.nccl_enabled and len(self._gpu_devices) > 1:
            self.synchronize("nccl")

        # Merge des résultats
        merged_result = self._merge_results(results, data, batch_axis)

        total_time = time.time() - start_time
        compute_times = [r.compute_time for r in results if r.success]
        avg_compute_time = np.mean(compute_times) if compute_times else 0.0

        logger.info(
            f"Workload terminé: {total_time:.3f}s total, "
            f"{avg_compute_time:.3f}s compute moyen, "
            f"{len(results)} chunks traités"
        )

        return merged_result

    def synchronize(self, method: str = "nccl") -> None:
        """
        Synchronise tous les devices GPU.

        Args:
            method: Méthode de sync ("nccl", "cuda", "auto")
                   "nccl" : NCCL all-reduce si disponible
                   "cuda" : Synchronisation CUDA basique
                   "auto" : NCCL si dispo, sinon CUDA
        """
        if not self._gpu_devices or not CUPY_AVAILABLE:
            logger.debug("Sync ignorée: pas de GPU")
            return

        if method == "nccl" and not self.nccl_enabled:
            logger.debug("NCCL sync demandée mais indisponible")
            method = "cuda"
        elif method == "auto":
            method = "nccl" if self.nccl_enabled else "cuda"

        try:
            if method == "nccl":
                # Synchronisation NCCL (placeholder - implémentation complexe)
                logger.debug("Sync NCCL multi-GPU")
                # TODO: Implémentation NCCL complète avec communicator
                for device in self._gpu_devices:
                    with cp.cuda.Device(device.device_id):
                        cp.cuda.Device().synchronize()
            else:
                # Synchronisation CUDA basique
                logger.debug("Sync CUDA multi-GPU")
                for device in self._gpu_devices:
                    with cp.cuda.Device(device.device_id):
                        cp.cuda.Device().synchronize()

        except Exception as e:
            logger.warning(f"Erreur synchronisation {method}: {e}")

    def profile_auto_balance(
        self, sample_size: int = 200_000, runs: int = 3
    ) -> Dict[str, float]:
        """
        Profile automatique pour optimiser la balance des devices.

        Exécute des benchmarks sur chaque device disponible et calcule
        les ratios optimaux basés sur le throughput mesuré.

        Args:
            sample_size: Taille des échantillons de test
            warmup: Nombre de runs de warmup (non comptés)
            runs: Nombre de runs de mesure

        Returns:
            Nouveaux ratios normalisés {"device": ratio}

        Example:
            >>> ratios = manager.profile_auto_balance(sample_size=50000)
            >>> print(ratios)  # {'5090': 0.78, '2060': 0.22}
            >>> manager.set_balance(ratios)
        """
        logger.info(
            f"Profiling auto-balance: {sample_size} échantillons, " f"{runs} runs"
        )

        # Données de test: opération vectorielle représentative
        test_data = np.random.randn(sample_size, 10).astype(np.float32)

        def benchmark_func(x):
            """Fonction benchmark: somme + produit matriciel simple."""
            return x.sum(axis=1) + (x * x).mean(axis=1)

        device_throughputs = {}

        # Test de chaque device individuellement
        for device in self.available_devices:
            if device.name in ["cpu"] and len(self._gpu_devices) > 0:
                continue  # Skip CPU si GPU disponibles

            logger.info(f"Profiling device {device.name}...")

            # Configuration balance temporaire (100% sur ce device)
            temp_balance = {device.name: 1.0}
            old_balance = self.device_balance
            self.device_balance = temp_balance

            try:
                # Warmup
                for _ in range(warmup):
                    _ = self.distribute_workload(
                        test_data[:1000], benchmark_func, seed=42
                    )

                # Mesures
                times = []
                for run in range(runs):
                    start_time = time.time()
                    _ = self.distribute_workload(
                        test_data, benchmark_func, seed=42 + run
                    )
                    elapsed = time.time() - start_time
                    times.append(elapsed)

                # Calcul throughput (échantillons/seconde)
                avg_time = np.mean(times)
                throughput = sample_size / avg_time
                device_throughputs[device.name] = throughput

                logger.info(
                    f"Device {device.name}: {throughput:.0f} échantillons/s "
                    f"(avg {avg_time:.3f}s)"
                )

            except Exception as e:
                logger.warning(f"Erreur profiling {device.name}: {e}")
                device_throughputs[device.name] = 0.0

            finally:
                # Restauration balance
                self.device_balance = old_balance

        # Calcul ratios optimaux basés sur throughput
        if not device_throughputs or all(t == 0 for t in device_throughputs.values()):
            logger.warning("Profiling échoué, conservation balance actuelle")
            return self.device_balance.copy()

        # Normalisation des throughputs en ratios
        total_throughput = sum(device_throughputs.values())
        optimal_ratios = {
            device: throughput / total_throughput
            for device, throughput in device_throughputs.items()
            if throughput > 0
        }

        # Log des résultats
        logger.info("Profiling terminé:")
        for device, ratio in optimal_ratios.items():
            old_ratio = self.device_balance.get(device, 0.0)
            throughput = device_throughputs[device]
            logger.info(
                f"  {device}: {ratio:.1%} (était {old_ratio:.1%}), "
                f"{throughput:.0f} échantillons/s"
            )

        return optimal_ratios

    def get_device_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Récupère les statistiques des devices.

        Returns:
            Stats par device: mémoire, balance, etc.
        """
        stats = {}

        for device in self.available_devices:
            device_stats = {
                "device_id": device.device_id,
                "available": device.is_available,
                "memory_total_gb": device.memory_total_gb,
                "memory_free_gb": device.memory_free_gb,
                "memory_used_pct": device.memory_used_pct,
                "compute_capability": device.compute_capability,
                "current_balance": self.device_balance.get(device.name, 0.0),
                "has_stream": device.name in self._device_streams,
            }
            stats[device.name] = device_stats

        return stats

    def __del__(self):
        """Nettoyage des streams GPU."""
        try:
            with self._lock:
                for stream in self._device_streams.values():
                    if hasattr(stream, "close"):
                        stream.close()
                self._device_streams.clear()
        except:
            pass


# === Gestionnaire Global ===

_default_manager: Optional[MultiGPUManager] = None


def get_default_manager() -> MultiGPUManager:
    """
    Récupère le gestionnaire multi-GPU par défaut (singleton).

    Returns:
        Instance MultiGPUManager configurée

    Example:
        >>> manager = get_default_manager()
        >>> result = manager.distribute_workload(data, func)
    """
    global _default_manager

    if _default_manager is None:
        _default_manager = MultiGPUManager()
        logger.info("Gestionnaire multi-GPU par défaut créé")

    return _default_manager
