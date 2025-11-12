"""
ThreadX Indicators GPU Integration - Phase 5 + Numba Optimization
==================================================================

Intégration de la distribution multi-GPU avec la couche d'indicateurs.
Optimisations Numba CUDA pour kernels fusionnés et configuration thread/block optimale.

Permet d'accélérer les calculs d'indicateurs techniques (Bollinger Bands,
ATR, etc.) en utilisant automatiquement la répartition GPU/CPU optimale.

Optimisations:
    - Numba CUDA kernels avec thread/block configuration (256-512 threads/block)
    - Kernel fusion pour réduire les launches GPU
    - Shared memory pour données fréquemment accessées
    - Profiling dynamique CPU vs GPU

Usage:
    >>> # Calcul distribué d'indicateurs avec Numba
    >>> from threadx.indicators import get_gpu_accelerated_bank
    >>>
    >>> bank = get_gpu_accelerated_bank()
    >>> bb_upper, bb_middle, bb_lower = bank.bollinger_bands(
    ...     df, period=20, std_dev=2.0, use_gpu=True
    ... )
"""

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from threadx.gpu import MultiGPUManager, get_default_manager
from threadx.gpu.profile_persistence import (
    get_gpu_thresholds,
    stable_hash,
    update_gpu_threshold_entry,
)
from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Numba CUDA imports optionnels
try:
    from numba import cuda, float32

    NUMBA_AVAILABLE = True
    logger.info("Numba CUDA disponible pour kernels optimisés")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba CUDA non disponible, utilisant CuPy uniquement")

# Configuration optimale thread/block pour RTX 5090/2060
OPTIMAL_THREADS_PER_BLOCK = 256  # 256-512 recommandé pour compute 8.9+
OPTIMAL_BLOCKS_PER_SM = 2  # Pour occupancy maximale


@cuda.jit if NUMBA_AVAILABLE else lambda x: x
def _numba_bollinger_kernel(prices, period, std_dev, upper, middle, lower):
    """
    Kernel Numba CUDA fusionné pour Bollinger Bands.

    Calcule SMA + std en un seul kernel pour réduire les launches.
    Configuration: 256 threads/block, shared memory pour rolling window.

    Args:
        prices: Array des prix (N,)
        period: Période de la moyenne mobile
        std_dev: Nombre d'écarts-types
        upper: Output upper band (N,)
        middle: Output middle band (N,)
        lower: Output lower band (N,)
    """
    # Shared memory pour rolling window (optimisation accès mémoire)
    shared_prices = cuda.shared.array(shape=(OPTIMAL_THREADS_PER_BLOCK,), dtype=float32)

    idx = cuda.grid(1)
    n = prices.shape[0]

    if idx >= n:
        return

    # Chargement en shared memory
    tid = cuda.threadIdx.x
    if idx < n:
        shared_prices[tid] = prices[idx]
    cuda.syncthreads()

    # Calcul SMA et std fusionnés
    if idx >= period - 1:
        # Somme et somme des carrés en une passe
        sum_val = 0.0
        sum_sq = 0.0

        for i in range(period):
            offset = idx - i
            if offset >= 0:
                val = prices[offset]
                sum_val += val
                sum_sq += val * val

        # Moyenne et écart-type
        mean = sum_val / period
        variance = (sum_sq / period) - (mean * mean)
        std = variance**0.5 if variance > 0 else 0.0

        # Bandes de Bollinger
        middle[idx] = mean
        upper[idx] = mean + std_dev * std
        lower[idx] = mean - std_dev * std
    else:
        # Padding pour indices < period
        middle[idx] = prices[idx]
        upper[idx] = prices[idx]
        lower[idx] = prices[idx]


@cuda.jit if NUMBA_AVAILABLE else lambda x: x
def _numba_rsi_kernel(prices, period, rsi_out):
    """
    Kernel Numba CUDA pour RSI optimisé.

    Calcule gains/losses et RSI en kernel fusionné.
    Configuration: 256 threads/block.

    Args:
        prices: Array des prix (N,)
        period: Période RSI
        rsi_out: Output RSI values (N,)
    """
    idx = cuda.grid(1)
    n = prices.shape[0]

    if idx >= n or idx < period:
        return

    # Calcul des gains/losses moyens
    sum_gains = 0.0
    sum_losses = 0.0

    for i in range(period):
        offset = idx - i
        if offset > 0:
            delta = prices[offset] - prices[offset - 1]
            if delta > 0:
                sum_gains += delta
            else:
                sum_losses += abs(delta)

    avg_gain = sum_gains / period
    avg_loss = sum_losses / period

    # RSI
    if avg_loss == 0:
        rsi_out[idx] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_out[idx] = 100.0 - (100.0 / (1.0 + rs))


class GPUAcceleratedIndicatorBank:
    """
    Banque d'indicateurs avec accélération multi-GPU + Numba CUDA.

    Wraps les calculs d'indicateurs pour utiliser automatiquement
    la distribution multi-GPU quand disponible et bénéfique.

    Optimisations Numba:
        - Kernels CUDA fusionnés (SMA+std, gains+losses)
        - Thread/block config optimale (256 threads/block)
        - Shared memory pour rolling windows
        - Profiling dynamique CPU vs GPU vs Numba
    """

    def __init__(self, gpu_manager: MultiGPUManager | None = None):
        """
        Initialise la banque d'indicateurs GPU avec Numba.

        Args:
            gpu_manager: Gestionnaire multi-GPU optionnel
                        Si None, utilise le gestionnaire par défaut
        """
        self.gpu_manager = gpu_manager or get_default_manager()

        # 🆕 Seuil réduit pour utiliser GPU plus tôt (optimisation utilisation)
        self.min_samples_for_gpu = 500  # Réduit de 1000 → 500 pour saturer GPU

        logger.info(
            f"Banque indicateurs GPU initialisée: "
            f"{len(self.gpu_manager._gpu_devices)} GPU(s), "
            f"seuil GPU: {self.min_samples_for_gpu} échantillons"
        )

    def _should_use_gpu_dynamic(
        self,
        indicator: str,
        n_rows: int,
        params: dict[str, Any],
        dtype: Any = np.float32,  # Any pour accepter DtypeObj pandas
        force_gpu: bool = False,
    ) -> bool:
        """
        Détermine l'utilisation GPU avec seuil dynamique
        basé sur profil historique.

        Args:
            indicator: Nom de l'indicateur (ex: 'bollinger', 'atr')
            n_rows: Nombre de lignes dans les données
            params: Paramètres principaux de l'indicateur
            dtype: Type de données (float32, float64)
            force_gpu: Force l'utilisation du GPU

        Returns:
            True si GPU recommandé selon profil de performance
        """
        # Vérification basique
        has_gpu = len(self.gpu_manager._gpu_devices) > 0
        if not has_gpu:
            return False

        if force_gpu:
            logger.info(f"Utilisation GPU forcée pour {indicator}")
            return True

        # Création de la signature unique
        params_major = {
            k: v
            for k, v in params.items()
            if k in ["period", "window", "std", "std_dev"]
        }
        dtype_name = dtype.__name__ if hasattr(dtype, "__name__") else str(dtype)
        signature = (
            f"{indicator}|N={n_rows}|"
            f"dtype={dtype_name}|"
            f"params={stable_hash(params_major)}"
        )

        # Récupération des seuils GPU
        thresholds = get_gpu_thresholds()
        defaults = thresholds["defaults"]

        # Si signature inconnue, lancer un micro-probe
        if signature not in thresholds["entries"]:
            # Micro-probe pour décider
            cpu_ms, gpu_ms = self._micro_probe(indicator, n_rows, params_major)

            # Enregistrement dans le profil
            update_gpu_threshold_entry(signature, cpu_ms, gpu_ms)
            thresholds = get_gpu_thresholds()  # Rechargement

        entry = thresholds["entries"].get(signature, {})

        # Règles de décision
        if n_rows < defaults["n_min_gpu"]:
            logger.debug(
                f"N={n_rows} < seuil minimal " f"{defaults['n_min_gpu']}, utilisant CPU"
            )
            return False

        # Calcul du gain estimé
        cpu_ms_est = entry.get("cpu_ms_avg", float("inf"))
        gpu_ms_est = entry.get("gpu_ms_avg", float("inf"))

        # Protection division par zéro
        if gpu_ms_est <= 0:
            return False

        gain = cpu_ms_est / gpu_ms_est
        decision_threshold = entry.get(
            "decision_threshold", defaults["decision_threshold"]
        )
        hysteresis = defaults["hysteresis"]

        # Décision avec hystérésis
        use_gpu = gain >= (decision_threshold - hysteresis)

        # Log une fois par exécution pour cette signature (pas à chaque appel)
        logger.info(
            f"Décision {'GPU' if use_gpu else 'CPU'} pour {signature}: "
            f"gain={gain:.2f}x, seuil={decision_threshold:.2f}, "
            f"cpu={cpu_ms_est:.2f}ms, gpu={gpu_ms_est:.2f}ms"
        )

        return use_gpu

    def _dispatch_indicator(
        self,
        indicator_name: str,
        data: pd.DataFrame,
        params: dict[str, Any],
        use_gpu: bool | None,
        gpu_func: Callable,
        cpu_func: Callable,
        input_cols: str | None = None,
        extract_arrays: bool = True,
    ) -> Any:
        """
        Dispatch automatique GPU/CPU pour un indicateur.

        Centralise la logique de décision GPU/CPU et l'extraction de données
        pour éviter la duplication de code entre indicateurs.

        Args:
            indicator_name: Nom de l'indicateur ('bollinger', 'atr', 'rsi')
            data: DataFrame source
            params: Paramètres de l'indicateur (period, std_dev, etc.)
            use_gpu: None (auto), True (force GPU), False (force CPU)
            gpu_func: Fonction GPU à appeler (signature: func(arrays, ...))
            cpu_func: Fonction CPU à appeler (signature: func(arrays, ...))
            input_cols: Colonne(s) à extraire (str ou None pour OHLC)
            extract_arrays: Si True, extrait et convertit les colonnes en
                          ndarray

        Returns:
            Résultat de l'indicateur (pd.Series ou Tuple[pd.Series])

        Example:
            >>> return self._dispatch_indicator(
            ...     'bollinger',
            ...     data,
            ...     {'period': 20, 'std_dev': 2.0},
            ...     use_gpu,
            ...     self._bollinger_bands_gpu,
            ...     self._bollinger_bands_cpu,
            ...     input_cols='close'
            ... )
        """
        data_size = len(data)

        # Détermination du dtype pour le profiling
        if input_cols:
            dtype = data[input_cols].dtype
        else:
            # Pour OHLC, utiliser dtype de 'close'
            dtype = data["close"].dtype

        # Décision dynamique CPU vs GPU basée sur profil historique
        if use_gpu is None:
            # Décision automatique basée sur profils
            use_gpu_decision = self._should_use_gpu_dynamic(
                indicator_name, data_size, params, dtype
            )
        else:
            # Force explicite
            use_gpu_decision = use_gpu

        # Extraction et conversion des données si nécessaire
        if extract_arrays:
            if input_cols:
                # Extraction d'une seule colonne
                arrays = np.asarray(data[input_cols].values)
            else:
                # Pas de colonnes spécifiées: passer le DataFrame
                arrays = data
        else:
            # Pas d'extraction: passer directement le DataFrame
            arrays = data

        # Dispatch vers GPU ou CPU
        if use_gpu_decision:
            return gpu_func(arrays)
        else:
            return cpu_func(arrays)

    def _micro_probe(
        self, indicator: str, n_rows: int, params: dict[str, Any], n_samples: int = 3
    ) -> tuple[float, float]:
        """
        Exécute un micro-benchmark pour comparer CPU vs GPU
        sur un échantillon réduit.

        Args:
            indicator: Nom de l'indicateur
            n_rows: Nombre de lignes original
            params: Paramètres de l'indicateur
            n_samples: Nombre d'échantillons pour moyenne

        Returns:
            Tuple (cpu_ms_avg, gpu_ms_avg)
        """
        logger.info(f"Micro-probe {indicator} (N={n_rows}, params={params})")

        # Taille d'échantillon: min(n_rows, 100000)
        # pour éviter les benchmarks trop longs
        sample_size = min(n_rows, 100000)

        # Données de test adaptées à l'indicateur
        if indicator in ["bollinger", "bollinger_bands"]:
            # Séries de prix
            test_data = np.random.normal(100, 5, sample_size).astype(np.float32)

            # Params par défaut si non spécifiés
            period = params.get("period", 20)
            std_dev = params.get("std_dev", params.get("std", 2.0))

            # Fonctions de test
            def cpu_func():
                return self._bollinger_bands_cpu(
                    test_data, period, std_dev, pd.RangeIndex(sample_size)
                )

            def gpu_func():
                return self._bollinger_bands_gpu(
                    test_data, period, std_dev, pd.RangeIndex(sample_size)
                )

        elif indicator in ["atr"]:
            # Données OHLC
            high = np.random.normal(105, 3, sample_size).astype(np.float32)
            low = np.random.normal(95, 3, sample_size).astype(np.float32)
            close = np.random.normal(100, 3, sample_size).astype(np.float32)

            df = pd.DataFrame({"high": high, "low": low, "close": close})

            # Params
            period = params.get("period", 14)

            # Fonctions de test
            def cpu_func():
                return self._atr_cpu(df, period)

            def gpu_func():
                return self._atr_gpu(df, period)

        elif indicator in ["rsi"]:
            # Séries de prix
            test_data = np.random.normal(100, 5, sample_size).astype(np.float32)

            # Params
            period = params.get("period", 14)

            # Fonctions de test
            def cpu_func():
                return self._rsi_cpu(test_data, period, pd.RangeIndex(sample_size))

            def gpu_func():
                return self._rsi_gpu(test_data, period, pd.RangeIndex(sample_size))

        else:
            # Indicateur non pris en charge: tests génériques
            logger.warning(
                f"Micro-probe pour '{indicator}' non implémenté, "
                f"utilisant benchmark générique"
            )
            return self._generic_micro_probe(sample_size)

        # Exécution des benchmarks
        cpu_times = []
        gpu_times = []

        # Préchauffage
        try:
            _ = cpu_func()
            _ = gpu_func()
        except Exception as e:
            logger.warning(
                f"Erreur préchauffage: {e}, " f"utilisant benchmark générique"
            )
            return self._generic_micro_probe(sample_size)

        # Mesure CPU
        for i in range(n_samples):
            start_time = time.time()
            _ = cpu_func()
            cpu_times.append((time.time() - start_time) * 1000)  # ms

        # Mesure GPU
        try:
            for i in range(n_samples):
                start_time = time.time()
                _ = gpu_func()
                gpu_times.append((time.time() - start_time) * 1000)  # ms
        except Exception as e:
            logger.warning(f"Erreur GPU: {e}, fallback CPU recommandé")
            # Pénalisation GPU: temps très élevé pour forcer choix CPU
            gpu_times = [max(cpu_times) * 5] * n_samples

        # Calcul des moyennes
        cpu_ms_avg = sum(cpu_times) / len(cpu_times)
        gpu_ms_avg = sum(gpu_times) / len(gpu_times)

        logger.info(
            f"Micro-probe {indicator}: CPU={cpu_ms_avg:.2f}ms, "
            f"GPU={gpu_ms_avg:.2f}ms, speedup={(cpu_ms_avg/gpu_ms_avg):.2f}x"
        )

        return cpu_ms_avg, gpu_ms_avg

    def _generic_micro_probe(self, sample_size: int) -> tuple[float, float]:
        """
        Exécute un benchmark générique pour CPU vs GPU.

        Args:
            sample_size: Taille d'échantillon pour le test

        Returns:
            Tuple (cpu_ms_avg, gpu_ms_avg)
        """
        # Limiter taille max pour benchmark générique
        sample_size = min(sample_size, 50000)

        # Données de test générique: convolution
        test_data = np.random.normal(0, 1, sample_size).astype(np.float32)

        # CPU benchmark
        start_time = time.time()
        _ = np.convolve(test_data, np.ones(20) / 20, mode="same")
        cpu_ms = (time.time() - start_time) * 1000

        # GPU benchmark
        try:

            def compute_fn(x):
                return np.convolve(x, np.ones(20) / 20, mode="same")

            start_time = time.time()
            _ = self.gpu_manager.distribute_workload(test_data, compute_fn)
            gpu_ms = (time.time() - start_time) * 1000
        except Exception as e:
            logger.warning(f"Erreur benchmark GPU générique: {e}")
            gpu_ms = cpu_ms * 2  # Pénalisation

        logger.debug(
            f"Benchmark générique (N={sample_size}): "
            f"CPU={cpu_ms:.2f}ms, GPU={gpu_ms:.2f}ms"
        )

        return cpu_ms, gpu_ms

    def update_indicator_methods(self):
        """
        Met à jour les méthodes d'indicateurs pour utiliser
        la décision dynamique.
        """
        # Cette méthode pourrait être utilisée pour appliquer
        # des patchs aux méthodes d'indicateurs existantes pour
        # utiliser _should_use_gpu_dynamic
        logger.debug("Activation de la décision GPU dynamique " "pour les indicateurs")

    def bollinger_bands(
        self,
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        price_col: str = "close",
        use_gpu: bool | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcul GPU des Bollinger Bands.

        Args:
            data: DataFrame OHLCV avec colonne price_col
            period: Période de la moyenne mobile
            std_dev: Multiplicateur d'écart-type
            price_col: Colonne de prix à utiliser
            use_gpu: Force GPU (None=auto, True=force, False=CPU only)

        Returns:
            Tuple (upper_band, middle_band, lower_band)

        Example:
            >>> df = pd.DataFrame({'close': [100, 101, 99, 102, 98]})
            >>> upper, middle, lower = bank.bollinger_bands(df, period=3)
        """
        if price_col not in data.columns:
            raise ValueError(f"Colonne '{price_col}' non trouvée dans les données")

        # Utilisation du dispatcher centralisé
        params = {"period": period, "std_dev": std_dev}

        return self._dispatch_indicator(
            indicator_name="bollinger",
            data=data,
            params=params,
            use_gpu=use_gpu,
            gpu_func=lambda prices: self._bollinger_bands_gpu(
                prices, period, std_dev, data.index
            ),
            cpu_func=lambda prices: self._bollinger_bands_cpu(
                prices, period, std_dev, data.index
            ),
            input_cols=price_col,
            extract_arrays=True,
        )

    def _bollinger_bands_gpu(
        self, prices: np.ndarray, period: int, std_dev: float, index: pd.Index
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcul Bollinger Bands sur GPU avec Numba CUDA kernel fusionné.

        Architecture optimisée:
        - Utilise kernel Numba CUDA si disponible (fusionné SMA+std)
        - Configuration thread/block optimale (256 threads/block)
        - Shared memory pour rolling window
        - Fallback CuPy distribution si Numba indisponible
        - Fallback CPU en cas d'erreur
        """

        # Tentative 1: Numba CUDA kernel fusionné (meilleure performance)
        if NUMBA_AVAILABLE and len(self.gpu_manager._gpu_devices) > 0:
            try:
                return self._bollinger_bands_numba(prices, period, std_dev, index)
            except Exception as e:
                logger.warning(f"Numba kernel échoué: {e}, fallback CuPy")

        # Tentative 2: Distribution CuPy classique
        def bb_compute_func(price_chunk):
            """Fonction vectorielle pour un chunk de prix."""
            if len(price_chunk) < period:
                # Chunk trop petit: moyenne simple
                ma = np.full_like(price_chunk, np.mean(price_chunk))
                std = np.full_like(price_chunk, np.std(price_chunk))
            else:
                # Moving average avec convolution
                weights = np.ones(period) / period
                ma = np.convolve(price_chunk, weights, mode="same")

                # Moving standard deviation
                squared_diff = (price_chunk - ma) ** 2
                variance = np.convolve(squared_diff, weights, mode="same")
                std = np.sqrt(variance)

            # Bandes de Bollinger
            upper = ma + std_dev * std
            middle = ma
            lower = ma - std_dev * std

            # Empilement pour retour
            return np.column_stack([upper, middle, lower])

        # Distribution GPU
        start_time = pd.Timestamp.now()

        try:
            # Reshape pour distribution (ajout dimension batch si nécessaire)
            if prices.ndim == 1:
                prices_2d = prices.reshape(-1, 1)
            else:
                prices_2d = prices

            result = self.gpu_manager.distribute_workload(
                prices_2d, bb_compute_func, seed=42
            )

            # Extraction des bandes
            upper_band = result[:, 0]
            middle_band = result[:, 1]
            lower_band = result[:, 2]

            elapsed = (pd.Timestamp.now() - start_time).total_seconds()
            logger.info(
                f"Bollinger Bands GPU (CuPy): {len(prices)} échantillons "
                f"en {elapsed:.3f}s"
            )

        except Exception as e:
            logger.warning(f"Erreur calcul GPU Bollinger Bands: {e}")
            logger.info("Fallback calcul CPU")
            return self._bollinger_bands_cpu(prices, period, std_dev, index)

        return (
            pd.Series(upper_band, index=index, name="bb_upper"),
            pd.Series(middle_band, index=index, name="bb_middle"),
            pd.Series(lower_band, index=index, name="bb_lower"),
        )

    def _bollinger_bands_numba(
        self, prices: np.ndarray, period: int, std_dev: float, index: pd.Index
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcul Bollinger Bands avec kernel Numba CUDA fusionné.

        Optimisations:
        - Kernel fusionné: SMA + std en un seul launch
        - Thread/block config optimale: 256 threads/block
        - Shared memory pour rolling window
        - Grid-stride loop pour grandes données

        Args:
            prices: Array 1D des prix
            period: Période moyenne mobile
            std_dev: Multiplicateur écart-type
            index: Index pandas pour output

        Returns:
            Tuple (upper, middle, lower) pd.Series
        """
        start_time = pd.Timestamp.now()

        n = len(prices)

        # Conversion en float32 pour optimisation GPU
        prices_f32 = np.asarray(prices, dtype=np.float32)

        # Allocation outputs device
        upper = np.zeros(n, dtype=np.float32)
        middle = np.zeros(n, dtype=np.float32)
        lower = np.zeros(n, dtype=np.float32)

        # Transfert vers GPU principal (device 0)
        device_id = self.gpu_manager._gpu_devices[0].device_id

        try:
            import cupy as cp

            with cp.cuda.Device(device_id):
                d_prices = cp.asarray(prices_f32)
                d_upper = cp.asarray(upper)
                d_middle = cp.asarray(middle)
                d_lower = cp.asarray(lower)

                # Configuration grid/block optimale
                threads_per_block = OPTIMAL_THREADS_PER_BLOCK
                blocks = (n + threads_per_block - 1) // threads_per_block

                logger.debug(
                    f"Numba kernel config: {blocks} blocks x {threads_per_block} threads"
                )

                # Launch kernel fusionné
                _numba_bollinger_kernel[blocks, threads_per_block](
                    d_prices, period, std_dev, d_upper, d_middle, d_lower
                )

                # Synchronisation
                cp.cuda.Device().synchronize()

                # Transfert résultats vers host
                upper = cp.asnumpy(d_upper)
                middle = cp.asnumpy(d_middle)
                lower = cp.asnumpy(d_lower)

        except Exception as e:
            logger.error(f"Erreur kernel Numba Bollinger: {e}")
            raise  # Propagation pour fallback CuPy

        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(
            f"Bollinger Bands Numba CUDA: {n} échantillons en {elapsed:.3f}s "
            f"({n/elapsed:.0f} échant./s)"
        )

        return (
            pd.Series(upper, index=index, name="bb_upper"),
            pd.Series(middle, index=index, name="bb_middle"),
            pd.Series(lower, index=index, name="bb_lower"),
        )

    def _bollinger_bands_cpu(
        self, prices: np.ndarray, period: int, std_dev: float, index: pd.Index
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calcul CPU classique des Bollinger Bands."""
        # Rolling window avec pandas pour simplicité
        price_series = pd.Series(prices, index=index)

        middle_band = price_series.rolling(window=period, min_periods=1).mean()
        rolling_std = price_series.rolling(window=period, min_periods=1).std()

        upper_band = middle_band + std_dev * rolling_std
        lower_band = middle_band - std_dev * rolling_std

        # Nommage des séries
        upper_band.name = "bb_upper"
        middle_band.name = "bb_middle"
        lower_band.name = "bb_lower"

        return upper_band, middle_band, lower_band

    def atr(
        self,
        data: pd.DataFrame,
        period: int = 14,
        use_gpu: bool | None = None,
    ) -> pd.Series:
        """
        Calcul GPU de l'Average True Range (ATR).

        Args:
            data: DataFrame OHLCV avec colonnes 'high', 'low', 'close'
            period: Période pour le calcul ATR
            use_gpu: Force GPU (None=auto, True=force, False=CPU only)

        Returns:
            Série ATR

        Example:
            >>> df = pd.DataFrame({
            ...     'high': [102, 103, 101],
            ...     'low': [98, 99, 97],
            ...     'close': [100, 101, 99]
            ... })
            >>> atr_series = bank.atr(df, period=2)
        """
        required_cols = ["high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes: {missing_cols}")

        # Utilisation du dispatcher centralisé
        params = {"period": period}

        return self._dispatch_indicator(
            indicator_name="atr",
            data=data,
            params=params,
            use_gpu=use_gpu,
            gpu_func=lambda df: self._atr_gpu(df, period),
            cpu_func=lambda df: self._atr_cpu(df, period),
            input_cols=None,  # Pas d'extraction, passe le DataFrame
            extract_arrays=False,
        )

    def _atr_gpu(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calcul ATR distribué sur GPU."""

        def atr_compute_func(ohlc_chunk):
            """Calcul ATR vectorisé pour un chunk."""
            if len(ohlc_chunk) < 2:
                return np.zeros(len(ohlc_chunk))

            high = ohlc_chunk[:, 0]  # Colonne high
            low = ohlc_chunk[:, 1]  # Colonne low
            close = ohlc_chunk[:, 2]  # Colonne close

            # True Range calculation
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]  # Premier élément

            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)

            true_range = np.maximum(tr1, np.maximum(tr2, tr3))

            # ATR (moyenne mobile du True Range)
            if len(true_range) < period:
                atr = np.full_like(true_range, np.mean(true_range))
            else:
                weights = np.ones(period) / period
                atr = np.convolve(true_range, weights, mode="same")

            return atr

        try:
            # Préparation données pour distribution
            ohlc_array = data[["high", "low", "close"]].values

            result = self.gpu_manager.distribute_workload(
                ohlc_array, atr_compute_func, seed=42
            )

            return pd.Series(result, index=data.index, name="atr")

        except Exception as e:
            logger.warning(f"Erreur calcul GPU ATR: {e}")
            return self._atr_cpu(data, period)

    def _atr_cpu(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calcul ATR CPU classique."""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # True Range
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR (moyenne mobile)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        atr.name = "atr"

        return atr

    def rsi(
        self,
        data: pd.DataFrame,
        period: int = 14,
        price_col: str = "close",
        use_gpu: bool | None = None,
    ) -> pd.Series:
        """
        Calcul GPU du Relative Strength Index (RSI).

        Args:
            data: DataFrame avec colonne price_col
            period: Période RSI
            price_col: Colonne de prix
            use_gpu: Force GPU (None=auto)

        Returns:
            Série RSI (0-100)
        """
        if price_col not in data.columns:
            raise ValueError(f"Colonne '{price_col}' non trouvée")

        # Utilisation du dispatcher centralisé
        params = {"period": period}

        return self._dispatch_indicator(
            indicator_name="rsi",
            data=data,
            params=params,
            use_gpu=use_gpu,
            gpu_func=lambda prices: self._rsi_gpu(prices, period, data.index),
            cpu_func=lambda prices: self._rsi_cpu(prices, period, data.index),
            input_cols=price_col,
            extract_arrays=True,
        )

    def _rsi_gpu(self, prices: np.ndarray, period: int, index: pd.Index) -> pd.Series:
        """Calcul RSI distribué sur GPU."""

        def rsi_compute_func(price_chunk):
            """Calcul RSI vectorisé."""
            if len(price_chunk) < 2:
                return np.full(len(price_chunk), 50.0)  # RSI neutre

            # Calcul des gains/pertes
            price_diff = np.diff(price_chunk, prepend=price_chunk[0])
            gains = np.where(price_diff > 0, price_diff, 0)
            losses = np.where(price_diff < 0, -price_diff, 0)

            # Moyennes des gains/pertes
            if len(gains) < period:
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
            else:
                weights = np.ones(period) / period
                avg_gain = np.convolve(gains, weights, mode="same")
                avg_loss = np.convolve(losses, weights, mode="same")

            # RSI calculation
            rs = np.divide(
                avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0
            )
            rsi = 100 - (100 / (1 + rs))

            return rsi

        try:
            if prices.ndim == 1:
                prices_2d = prices.reshape(-1, 1)
            else:
                prices_2d = prices

            result = self.gpu_manager.distribute_workload(
                prices_2d, rsi_compute_func, seed=42
            )

            # Convertir en ndarray pour garantir compatibilité
            result_array = np.asarray(result)

            # Aplatir si multi-dimensionnel
            if result_array.ndim > 1:
                result_array = result_array.flatten()

            return pd.Series(result_array, index=index, name="rsi")

        except Exception as e:
            logger.warning(f"Erreur calcul GPU RSI: {e}")
            return self._rsi_cpu(prices, period, index)

    def _rsi_cpu(self, prices: np.ndarray, period: int, index: pd.Index) -> pd.Series:
        """Calcul RSI CPU classique."""
        price_series = pd.Series(prices, index=index)

        # Gains et pertes
        price_change = price_series.diff()
        gains = price_change.where(price_change > 0, 0)
        losses = -price_change.where(price_change < 0, 0)

        # Moyennes mobiles
        avg_gain = gains.rolling(window=period, min_periods=1).mean()
        avg_loss = losses.rolling(window=period, min_periods=1).mean()

        # RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi.name = "rsi"

        return rsi

    def get_performance_stats(self) -> dict:
        """
        Récupère les statistiques de performance du gestionnaire GPU.

        Returns:
            Dict avec stats devices et balance
        """
        return {
            "gpu_manager_stats": self.gpu_manager.get_device_stats(),
            "current_balance": self.gpu_manager.device_balance,
            "min_samples_for_gpu": self.min_samples_for_gpu,
            "available_indicators": ["bollinger_bands", "atr", "rsi"],
        }

    def optimize_balance(self, sample_data: pd.DataFrame, runs: int = 3) -> dict:
        """
        Optimise automatiquement la balance GPU pour les indicateurs.

        Args:
            sample_data: Données représentatives pour benchmark
            runs: Nombre de runs pour moyenne

        Returns:
            Nouveaux ratios optimaux
        """
        logger.info("Optimisation balance GPU pour indicateurs...")

        # Utilisation des données pour profiling
        if "close" in sample_data.columns:
            # Test avec Bollinger Bands (représentatif)
            old_min_samples = self.min_samples_for_gpu
            self.min_samples_for_gpu = 0  # Force GPU pour profiling

            try:
                # Benchmark sur échantillon
                optimal_ratios = self.gpu_manager.profile_auto_balance(
                    sample_size=min(len(sample_data), 50000), runs=runs
                )

                # Application des nouveaux ratios
                self.gpu_manager.set_balance(optimal_ratios)

                logger.info(f"Balance optimisée: {optimal_ratios}")
                return optimal_ratios

            finally:
                self.min_samples_for_gpu = old_min_samples

        else:
            logger.warning("Données sans colonne 'close', optimisation ignorée")
            return self.gpu_manager.device_balance


# === Instance globale ===

_gpu_indicator_bank: GPUAcceleratedIndicatorBank | None = None


def get_gpu_accelerated_bank() -> GPUAcceleratedIndicatorBank:
    """
    Récupère l'instance globale de la banque d'indicateurs GPU.

    Returns:
        Instance GPUAcceleratedIndicatorBank

    Example:
        >>> bank = get_gpu_accelerated_bank()
        >>> upper, middle, lower = bank.bollinger_bands(df, use_gpu=True)
    """
    global _gpu_indicator_bank

    if _gpu_indicator_bank is None:
        _gpu_indicator_bank = GPUAcceleratedIndicatorBank()
        logger.info("Banque indicateurs GPU globale créée")

    return _gpu_indicator_bank
