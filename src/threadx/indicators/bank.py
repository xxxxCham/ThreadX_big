#!/usr/bin/env python3
"""
ThreadX Indicator Bank - Cache centralisé d'indicateurs
=======================================================

Gestion centralisée du cache d'indicateurs techniques avec:
- Cache disque intelligent avec TTL et checksums
- Batch processing automatique (seuil: 100 paramètres)
- Registry automatique mise à jour
- Support GPU multi-carte transparent
- Validation et recompute forcé

Fonctions principales:
- ensure(): Vérifie existence/validité → recalcule si nécessaire
- force_recompute(): Recalcule obligatoirement
- batch_ensure(): Traitement batch avec parallélisation

Spécifications cache:
- TTL: 3600 secondes (1 heure)
- Clés triées alphabétiquement
- Checksums MD5 pour validation intégrité
- Registry Parquet mis à jour automatiquement

Exemple d'usage:
    ```python
    from threadx.indicators.bank import IndicatorBank, ensure_indicator

    # Initialisation
    bank = IndicatorBank(cache_dir="indicators_cache")

    # Simple ensure
    bb_result = ensure_indicator(
        'bollinger',
        {'period': 20, 'std': 2.0},
        close_data,
        symbol='BTCUSDC',
        timeframe='15m'
    )

    # Batch ensure
    params_list = [
        {'period': 20, 'std': 2.0},
        {'period': 50, 'std': 1.5}
    ]
    results = bank.batch_ensure('bollinger', params_list, close_data)
    ```
"""

import hashlib
import json
import logging
import os
import pickle
import time
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# FIX B1: File locking cross-platform
import platform

if platform.system() == "Windows":
    import msvcrt
else:
    import fcntl

# Import des calculateurs
from .bollinger import BollingerBands, BollingerSettings
from .xatr import ATR, ATRSettings

# Import Phase 2 Data
try:
    from ..data.registry import quick_inventory
    from ..data.io import write_frame, read_frame

    HAS_THREADX_DATA = True
except ImportError:
    HAS_THREADX_DATA = False

# Configuration logging
logger = logging.getLogger(__name__)


@dataclass
class IndicatorSettings:
    """Configuration globale pour IndicatorBank"""

    cache_dir: str = "indicators_cache"
    ttl_seconds: int = 3600  # 1 heure
    batch_threshold: int = 100  # Seuil pour parallélisation
    max_workers: int = 8  # Workers pour batch processing
    use_gpu: bool = True
    auto_registry_update: bool = True
    checksum_validation: bool = True
    compression_level: int = 6  # Pour cache Parquet

    # GPU settings (hérités des modules indicateurs)
    gpu_split_ratio: Tuple[float, float] = (0.75, 0.25)
    gpu_batch_size: int = 1000

    def __post_init__(self):
        """Validation et création du répertoire cache"""
        self.cache_path = Path(self.cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Sous-répertoires par type d'indicateur
        (self.cache_path / "bollinger").mkdir(exist_ok=True)
        (self.cache_path / "atr").mkdir(exist_ok=True)
        (self.cache_path / "registry").mkdir(exist_ok=True)

        if self.max_workers < 1:
            raise ValueError(f"max_workers doit être >= 1, reçu: {self.max_workers}")


class CacheManager:
    """Gestionnaire de cache avec TTL et checksums"""

    def __init__(self, settings: IndicatorSettings):
        self.settings = settings
        self.cache_path = Path(settings.cache_dir)

    def _read_metadata(self, meta_file: Path) -> Optional[Dict]:
        """Helper: Lit metadata JSON (élimine duplication)"""
        try:
            with open(meta_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read metadata {meta_file}: {e}")
            return None

    def _generate_cache_key(
        self,
        indicator_type: str,
        params: Dict[str, Any],
        symbol: str = "",
        timeframe: str = "",
        data_hash: str = "",
    ) -> str:
        """
        Génération clé de cache triée alphabétiquement

        Format: {indicator_type}_{symbol}_{timeframe}_{params_hash}_{data_hash[:8]}
        """
        # Tri alphabétique des paramètres pour cohérence
        sorted_params = dict(sorted(params.items()))
        params_str = json.dumps(sorted_params, sort_keys=True, separators=(",", ":"))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:16]

        # Clé finale
        key_parts = [
            indicator_type,
            symbol or "nosymbol",
            timeframe or "notf",
            params_hash,
            data_hash[:8] if data_hash else "nodata",
        ]

        return "_".join(key_parts)

    def _compute_data_hash(
        self, data: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> str:
        """Calcul hash des données pour cache key"""
        if isinstance(data, pd.DataFrame):
            # Pour OHLCV, hash sur close uniquement pour optimisation
            if "close" in data.columns:
                data_bytes = data["close"].values.astype(np.float64).tobytes()
            else:
                data_bytes = data.values.astype(np.float64).tobytes()
        elif isinstance(data, pd.Series):
            data_bytes = data.values.astype(np.float64).tobytes()
        else:  # numpy array
            data_bytes = np.asarray(data, dtype=np.float64).tobytes()

        return hashlib.md5(data_bytes).hexdigest()

    def _get_cache_filepath(self, cache_key: str, indicator_type: str) -> Path:
        """Chemin fichier cache"""
        return self.cache_path / indicator_type / f"{cache_key}.parquet"

    def _get_metadata_filepath(self, cache_key: str, indicator_type: str) -> Path:
        """Chemin fichier métadonnées"""
        return self.cache_path / indicator_type / f"{cache_key}.meta"

    def is_cache_valid(self, cache_key: str, indicator_type: str) -> bool:
        """Vérification validité cache (TTL + intégrité)"""
        cache_file = self._get_cache_filepath(cache_key, indicator_type)
        meta_file = self._get_metadata_filepath(cache_key, indicator_type)

        if not (cache_file.exists() and meta_file.exists()):
            return False

        try:
            # Lecture métadonnées
            metadata = self._read_metadata(meta_file)
            if not metadata:
                return False

            # Vérification TTL
            created_at = metadata.get("created_at", 0)
            if time.time() - created_at > self.settings.ttl_seconds:
                logger.debug(f"🕒 Cache expiré (TTL): {cache_key}")
                return False

            # Vérification checksum si activée
            if self.settings.checksum_validation and "checksum" in metadata:
                current_checksum = self._compute_file_checksum(cache_file)
                if current_checksum != metadata["checksum"]:
                    logger.warning(f"⚠️ Checksum invalide: {cache_key}")
                    return False

            return True

        except Exception as e:
            logger.warning(f"⚠️ Erreur validation cache {cache_key}: {e}")
            return False

    def _compute_file_checksum(self, filepath: Path) -> str:
        """Calcul checksum fichier"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def load_from_cache(self, cache_key: str, indicator_type: str) -> Optional[Any]:
        """Chargement depuis cache"""
        if not self.is_cache_valid(cache_key, indicator_type):
            return None

        cache_file = self._get_cache_filepath(cache_key, indicator_type)

        try:
            # Lecture Parquet
            df = pd.read_parquet(cache_file)

            # Conversion selon format original
            if len(df.columns) == 1:
                # Single array (ex: ATR)
                return df.iloc[:, 0].values
            else:
                # Multiple arrays (ex: Bollinger Bands)
                return tuple(df[col].values for col in df.columns)

        except Exception as e:
            logger.error(f"❌ Erreur chargement cache {cache_key}: {e}")
            return None

    def save_to_cache(
        self,
        cache_key: str,
        indicator_type: str,
        result: Union[np.ndarray, Tuple[np.ndarray, ...]],
        params: Dict[str, Any],
        symbol: str = "",
        timeframe: str = "",
    ) -> bool:
        """Sauvegarde en cache avec métadonnées

        FIX B1: File locking pour éviter race conditions en écriture.
        """
        cache_file = self._get_cache_filepath(cache_key, indicator_type)
        meta_file = self._get_metadata_filepath(cache_key, indicator_type)
        lock_file = cache_file.with_suffix(".lock")

        try:
            # FIX B1: Acquire file lock (cross-platform)
            with open(lock_file, "w") as lockf:
                if platform.system() == "Windows":
                    msvcrt.locking(lockf.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Préparation DataFrame
                if isinstance(result, tuple):
                    # Multiples arrays (ex: Bollinger upper, middle, lower)
                    df_data = {}
                    for i, arr in enumerate(result):
                        df_data[f"result_{i}"] = arr
                    df = pd.DataFrame(df_data)
                else:
                    # Single array (ex: ATR)
                    df = pd.DataFrame({"result_0": result})

                # Sauvegarde Parquet avec compression
                df.to_parquet(cache_file, compression="snappy", index=False)

                # Métadonnées
                metadata = {
                    "cache_key": cache_key,
                    "indicator_type": indicator_type,
                    "params": params,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "created_at": time.time(),
                    "data_shape": df.shape,
                    "columns": list(df.columns),
                }

                # Checksum si activé
                if self.settings.checksum_validation:
                    metadata["checksum"] = self._compute_file_checksum(cache_file)

                # Sauvegarde métadonnées
                with open(meta_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.debug(f"💾 Cache sauvé: {cache_key}")

                # Lock released automatically on file close
                return True

        except (IOError, OSError) as e:
            # Lock conflict ou erreur I/O
            logger.warning(f"⚠️ Cache write conflict {cache_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde cache {cache_key}: {e}")
            return False
        finally:
            # Cleanup lock file
            if lock_file.exists():
                try:
                    lock_file.unlink()
                except Exception:  # FIX: Permission denied ou autres
                    pass

    def cleanup_expired(self) -> int:
        """Nettoyage cache expiré"""
        cleaned_count = 0

        for indicator_dir in self.cache_path.iterdir():
            if not indicator_dir.is_dir():
                continue

            for meta_file in indicator_dir.glob("*.meta"):
                try:
                    metadata = self._read_metadata(meta_file)
                    if not metadata:
                        continue

                    created_at = metadata.get("created_at", 0)
                    if time.time() - created_at > self.settings.ttl_seconds:
                        # Suppression fichiers cache et meta
                        cache_key = meta_file.stem
                        cache_file = indicator_dir / f"{cache_key}.parquet"

                        if cache_file.exists():
                            cache_file.unlink()
                        meta_file.unlink()

                        cleaned_count += 1

                except Exception as e:
                    logger.warning(f"⚠️ Erreur nettoyage {meta_file}: {e}")

        if cleaned_count > 0:
            logger.info(f"🧹 Cache nettoyé: {cleaned_count} fichiers supprimés")

        return cleaned_count


class IndicatorBank:
    """Banque d'indicateurs avec cache intelligent et batch processing"""

    def __init__(self, settings: Optional[IndicatorSettings] = None):
        self.settings = settings or IndicatorSettings()
        self.cache_manager = CacheManager(self.settings)

        # Calculateurs d'indicateurs
        self.calculators = {
            "bollinger": BollingerBands(
                BollingerSettings(use_gpu=self.settings.use_gpu)
            ),
            "atr": ATR(ATRSettings(use_gpu=self.settings.use_gpu)),
        }

        # Stats
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "computations": 0,
            "batch_operations": 0,
        }

        logger.info(f"🏦 IndicatorBank initialisé - Cache: {self.settings.cache_dir}")

    def ensure(
        self,
        indicator_type: str,
        params: Dict[str, Any],
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        symbol: str = "",
        timeframe: str = "",
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Ensure indicateur: vérifie cache → calcule si nécessaire

        Args:
            indicator_type: 'bollinger' ou 'atr'
            params: Paramètres indicateur
            data: Données OHLCV (DataFrame) ou prix (array)
            symbol: Symbole (optionnel)
            timeframe: Timeframe (optionnel)

        Returns:
            Résultat indicateur ou None si erreur

        Exemple:
            ```python
            # Bollinger Bands
            bb_result = bank.ensure(
                'bollinger',
                {'period': 20, 'std': 2.0},
                ohlcv_data,
                symbol='BTCUSDC',
                timeframe='15m'
            )
            # bb_result = (upper, middle, lower)

            # ATR
            atr_result = bank.ensure(
                'atr',
                {'period': 14, 'method': 'ema'},
                ohlcv_data
            )
            # atr_result = atr_values array
            ```
        """
        start_time = time.time()

        # Validation type indicateur
        if indicator_type not in self.calculators:
            raise ValueError(f"Type indicateur non supporté: {indicator_type}")

        # Génération clé cache
        data_hash = self.cache_manager._compute_data_hash(data)
        cache_key = self.cache_manager._generate_cache_key(
            indicator_type, params, symbol, timeframe, data_hash
        )

        # Tentative chargement cache
        cached_result = self.cache_manager.load_from_cache(cache_key, indicator_type)
        if cached_result is not None:
            self.stats["cache_hits"] += 1
            elapsed = time.time() - start_time
            logger.debug(
                f"🎯 Cache HIT {indicator_type}: {cache_key[:20]}... ({elapsed:.3f}s)"
            )
            return cached_result

        # Cache MISS → calcul
        self.stats["cache_misses"] += 1
        logger.debug(f"❌ Cache MISS {indicator_type}: {cache_key[:20]}...")

        try:
            result = self._compute_indicator(indicator_type, params, data)
            if result is None:
                raise RuntimeError(
                    f"_compute_indicator returned None for {indicator_type} with params {params}"
                )

            # Sauvegarde en cache
            self.cache_manager.save_to_cache(
                cache_key, indicator_type, result, params, symbol, timeframe
            )
            self.stats["computations"] += 1

            # Mise à jour registry si activée
            if self.settings.auto_registry_update:
                self._update_registry(
                    indicator_type, cache_key, params, symbol, timeframe
                )

            elapsed = time.time() - start_time
            logger.debug(f"✅ Compute {indicator_type}: {elapsed:.3f}s")
            return result

        except Exception as e:
            logger.error(f"❌ Erreur calcul {indicator_type} {params}: {e}")
            raise RuntimeError(
                f"Failed to compute {indicator_type} with params {params}"
            ) from e

    def _compute_indicator(
        self,
        indicator_type: str,
        params: Dict[str, Any],
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Calcul effectif d'un indicateur selon son type"""

        calculator = self.calculators[indicator_type]

        if indicator_type == "bollinger":
            # Extraction prix close
            if isinstance(data, pd.DataFrame):
                if "close" not in data.columns:
                    raise ValueError(
                        "DataFrame doit contenir colonne 'close' pour Bollinger"
                    )
                close = data["close"]
            else:
                close = data

            # Calcul Bollinger Bands
            return calculator.compute(
                close, period=params.get("period", 20), std=params.get("std", 2.0)
            )

        elif indicator_type == "atr":
            # Extraction prix OHLC
            if isinstance(data, pd.DataFrame):
                required_cols = ["high", "low", "close"]
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    raise ValueError(
                        f"DataFrame doit contenir colonnes {missing_cols} pour ATR"
                    )

                high = data["high"]
                low = data["low"]
                close = data["close"]
            else:
                raise ValueError("ATR nécessite DataFrame OHLC, pas array simple")

            # Calcul ATR
            return calculator.compute(
                high,
                low,
                close,
                period=params.get("period", 14),
                method=params.get("method", "ema"),
            )

        else:
            raise ValueError(f"Type indicateur non implémenté: {indicator_type}")

    def batch_ensure(
        self,
        indicator_type: str,
        params_list: List[Dict[str, Any]],
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        symbol: str = "",
        timeframe: str = "",
    ) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        """
        Calcul batch d'indicateurs avec mutualisation des intermédiaires.

        Optimisations :
        - Concat des paramètres en dim-0 pour calcul groupé
        - Calcul SMA/TR une seule fois si partagé entre paramètres
        - Re-débatch stable avec préservation de l'ordre
        - Cache TTL + checksums avec clés JSON canonisées

        Args:
            indicator_type: Type d'indicateur ('bollinger', 'atr', etc.)
            params_list: Liste des paramètres à calculer
            data: Données source
            symbol: Symbole pour le cache
            timeframe: Timeframe pour le cache

        Returns:
            Dict[params_key, result] des résultats par paramètre

        Example:
            >>> bank = IndicatorBank()
            >>> params_list = [
            ...     {'period': 20, 'std': 2.0},
            ...     {'period': 20, 'std': 2.5},
            ...     {'period': 50, 'std': 2.0}
            ... ]
            >>> results = bank.batch_ensure('bollinger', params_list, close_data)
            >>> len(results) == 3
        """
        if not params_list:
            return {}

        logger.info(f"Batch ensure: {indicator_type}, {len(params_list)} paramètres")

        batch_results = {}
        cache_hits = 0
        cache_misses = 0

        # Phase 1: Vérification du cache existant
        uncached_params = []
        for params in params_list:
            params_key = self._params_to_cache_key(params)

            # Vérification cache individuel
            cache_key = self.cache_manager._generate_cache_key(
                indicator_type,
                params,
                symbol,
                timeframe,
                self.cache_manager._compute_data_hash(data),
            )

            if self.cache_manager.is_cache_valid(cache_key, indicator_type):
                cached_result = self.cache_manager.load_from_cache(
                    cache_key, indicator_type
                )
                if cached_result is not None:
                    batch_results[params_key] = cached_result
                    cache_hits += 1
                    continue

            uncached_params.append(params)
            cache_misses += 1

        logger.debug(f"Cache: {cache_hits} hits, {cache_misses} misses")

        # Phase 2: Calcul batch des paramètres manquants
        if uncached_params:
            computed_results = self._compute_batch_with_intermediates(
                indicator_type, uncached_params, data
            )

            # Phase 3: Sauvegarde en cache et ajout aux résultats
            for params, result in zip(uncached_params, computed_results):
                params_key = self._params_to_cache_key(params)
                batch_results[params_key] = result

                # Sauvegarde cache
                cache_key = self.cache_manager._generate_cache_key(
                    indicator_type,
                    params,
                    symbol,
                    timeframe,
                    self.cache_manager._compute_data_hash(data),
                )

                self.cache_manager.save_to_cache(
                    cache_key, indicator_type, result, params, symbol, timeframe
                )

        # Phase 4: Mise à jour du registry
        self._update_registry_batch(indicator_type, params_list, symbol, timeframe)

        logger.info(
            f"Batch terminé: {len(batch_results)} résultats "
            f"(cache hit rate: {cache_hits/(cache_hits+cache_misses):.1%})"
        )

        return batch_results

    def compute_batch(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        indicators: List[str],
        symbol: str = "",
        timeframe: str = "",
    ) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        """
        Calcule plusieurs indicateurs en batch (API simplifiée).

        Wrapper convivial sur batch_ensure() qui accepte une liste
        d'indicateurs au format "type_param" (ex: "rsi_14", "bb_20_2.0").

        Args:
            data: Données OHLCV (DataFrame ou array)
            indicators: Liste d'indicateurs au format "type_param"
                Exemples:
                - "rsi_14" → RSI période 14
                - "bb_20" → Bollinger Bands période 20 (std=2.0 par défaut)
                - "bb_20_2.5" → Bollinger Bands période 20, std 2.5
                - "atr_14" → ATR période 14
                - "sma_50" → SMA période 50
            symbol: Symbole pour le cache (optionnel)
            timeframe: Timeframe pour le cache (optionnel)

        Returns:
            Dict[indicator_name, result] des résultats par indicateur

        Raises:
            ValueError: Si format d'indicateur invalide

        Example:
            >>> bank = IndicatorBank()
            >>> df = pd.DataFrame({'close': [100, 101, 102, 103, 104]})
            >>> results = bank.compute_batch(
            ...     data=df,
            ...     indicators=["rsi_14", "bb_20", "sma_50"],
            ...     symbol="BTCUSDT",
            ...     timeframe="1h"
            ... )
            >>> 'rsi_14' in results
            True
            >>> 'bb_20' in results
            True
        """
        if not indicators:
            return {}

        logger.info(f"compute_batch: {len(indicators)} indicateurs pour {symbol}")

        # Grouper les indicateurs par type
        grouped_indicators = {}
        for indicator_str in indicators:
            indicator_type, params = self._parse_indicator_string(indicator_str)
            if indicator_type not in grouped_indicators:
                grouped_indicators[indicator_type] = []
            grouped_indicators[indicator_type].append((indicator_str, params))

        # Calculer chaque groupe avec batch_ensure
        all_results = {}
        for indicator_type, indicator_list in grouped_indicators.items():
            params_list = [params for _, params in indicator_list]

            # Appel à batch_ensure
            batch_results = self.batch_ensure(
                indicator_type=indicator_type,
                params_list=params_list,
                data=data,
                symbol=symbol,
                timeframe=timeframe,
            )

            # Mapper les résultats avec les noms d'indicateurs originaux
            for indicator_str, params in indicator_list:
                params_key = self._params_to_cache_key(params)
                if params_key in batch_results:
                    all_results[indicator_str] = batch_results[params_key]
                else:
                    logger.warning(f"Résultat manquant pour {indicator_str}")

        logger.info(f"compute_batch: {len(all_results)} résultats calculés")
        return all_results

    def _parse_indicator_string(self, indicator_str: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse une chaîne d'indicateur au format "type_param1_param2".

        Args:
            indicator_str: Chaîne au format "type_param" (ex: "rsi_14")

        Returns:
            Tuple (type, params_dict)

        Raises:
            ValueError: Si format invalide

        Example:
            >>> bank._parse_indicator_string("rsi_14")
            ('rsi', {'period': 14})
            >>> bank._parse_indicator_string("bb_20_2.5")
            ('bollinger', {'period': 20, 'std': 2.5})
        """
        parts = indicator_str.split("_")
        if len(parts) < 2:
            raise ValueError(
                f"Format indicateur invalide: '{indicator_str}'. "
                f"Format attendu: 'type_param' (ex: 'rsi_14')"
            )

        indicator_type = parts[0].lower()

        # Mapper les raccourcis vers les noms complets
        type_mapping = {
            "bb": "bollinger",
            "sma": "sma",
            "ema": "ema",
            "rsi": "rsi",
            "atr": "atr",
            "macd": "macd",
        }

        if indicator_type not in type_mapping:
            raise ValueError(
                f"Type indicateur inconnu: '{indicator_type}'. "
                f"Types supportés: {list(type_mapping.keys())}"
            )

        indicator_type = type_mapping[indicator_type]

        # Parser les paramètres selon le type
        if indicator_type == "bollinger":
            # Format: bb_period ou bb_period_std
            period = int(parts[1])
            std = float(parts[2]) if len(parts) > 2 else 2.0
            return indicator_type, {"period": period, "std": std}

        elif indicator_type in ["rsi", "atr", "sma", "ema"]:
            # Format: type_period
            period = int(parts[1])
            return indicator_type, {"period": period}

        elif indicator_type == "macd":
            # Format: macd_fast_slow_signal
            fast = int(parts[1]) if len(parts) > 1 else 12
            slow = int(parts[2]) if len(parts) > 2 else 26
            signal = int(parts[3]) if len(parts) > 3 else 9
            return indicator_type, {
                "fast": fast,
                "slow": slow,
                "signal": signal,
            }

        else:
            raise ValueError(f"Parser non implémenté pour type '{indicator_type}'")

    def _compute_batch_with_intermediates(
        self,
        indicator_type: str,
        params_list: List[Dict[str, Any]],
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> List[Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        """Calcule en batch avec mutualisation des intermédiaires."""

        if indicator_type == "bollinger":
            return self._batch_bollinger_with_sma_sharing(params_list, data)
        elif indicator_type == "atr":
            return self._batch_atr_with_tr_sharing(params_list, data)
        else:
            # Fallback: calcul séquentiel
            results = []
            for params in params_list:
                result = self._compute_indicator(indicator_type, params, data)
                results.append(result)
            return results

    def _batch_bollinger_with_sma_sharing(
        self,
        params_list: List[Dict[str, Any]],
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Calcul batch Bollinger avec partage des SMA."""

        # Extraction des périodes uniques
        unique_periods = list(set(params["period"] for params in params_list))

        # Calcul des SMA partagées
        sma_cache = {}
        for period in unique_periods:
            if isinstance(data, pd.Series):
                sma = data.rolling(window=period).mean().values
            elif isinstance(data, np.ndarray):
                sma = pd.Series(data).rolling(window=period).mean().values
            else:
                # DataFrame: utilise la colonne 'close' par défaut
                close_col = data.get("close", data.iloc[:, 0])
                sma = close_col.rolling(window=period).mean().values

            sma_cache[period] = sma

        # Calcul des écarts-types et bandes
        results = []
        for params in params_list:
            period = params["period"]
            std_multiplier = params.get("std", 2.0)

            sma = sma_cache[period]

            # Calcul de l'écart-type mobile
            if isinstance(data, pd.Series):
                rolling_std = data.rolling(window=period).std().values
            elif isinstance(data, np.ndarray):
                rolling_std = pd.Series(data).rolling(window=period).std().values
            else:
                close_col = data.get("close", data.iloc[:, 0])
                rolling_std = close_col.rolling(window=period).std().values

            # Bandes de Bollinger
            upper_band = sma + (std_multiplier * rolling_std)
            lower_band = sma - (std_multiplier * rolling_std)

            results.append((upper_band, sma, lower_band))

        return results

    def _batch_atr_with_tr_sharing(
        self,
        params_list: List[Dict[str, Any]],
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> List[np.ndarray]:
        """Calcul batch ATR avec partage du True Range."""

        # Calcul du True Range une seule fois
        if isinstance(data, pd.DataFrame) and all(
            col in data.columns for col in ["high", "low", "close"]
        ):
            high = data["high"].values
            low = data["low"].values
            close = data["close"].values

            # True Range calculation
            tr1 = high - low
            tr2 = np.abs(high - np.roll(close, 1))
            tr3 = np.abs(low - np.roll(close, 1))

            true_range = np.maximum.reduce([tr1, tr2, tr3])
            true_range[0] = tr1[0]  # Premier élément
        else:
            # Fallback pour données simplifiées
            if isinstance(data, (pd.Series, np.ndarray)):
                prices = data.values if isinstance(data, pd.Series) else data
                true_range = np.abs(np.diff(prices, prepend=prices[0]))
            else:
                raise ValueError("Format de données non supporté pour ATR batch")

        # Calcul des ATR pour chaque période
        results = []
        for params in params_list:
            period = params.get("period", 14)
            method = params.get("method", "ema")

            if method == "ema":
                # EMA du True Range
                alpha = 2.0 / (period + 1)
                atr = np.zeros_like(true_range)
                atr[0] = true_range[0]

                for i in range(1, len(true_range)):
                    atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i - 1]
            else:
                # SMA du True Range
                atr = pd.Series(true_range).rolling(window=period).mean().values

            results.append(atr)

        return results

    def _params_to_cache_key(self, params: Dict[str, Any]) -> str:
        """Convertit les paramètres en clé de cache stable."""
        import json

        return json.dumps(params, sort_keys=True, separators=(",", ":"))

    def _update_registry_batch(
        self,
        indicator_type: str,
        params_list: List[Dict[str, Any]],
        symbol: str,
        timeframe: str,
    ) -> None:
        """Met à jour le registry pour un batch de calculs."""
        # TODO: Implémentation du registry batch si nécessaire
        pass

    def force_recompute(
        self,
        indicator_type: str,
        params: Dict[str, Any],
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        symbol: str = "",
        timeframe: str = "",
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Recompute forcé (ignore cache)

        Args:
            Mêmes que ensure()

        Returns:
            Résultat indicateur recalculé
        """
        logger.info(f"🔄 Force recompute {indicator_type}: {params}")

        # Suppression cache existant
        data_hash = self.cache_manager._compute_data_hash(data)
        cache_key = self.cache_manager._generate_cache_key(
            indicator_type, params, symbol, timeframe, data_hash
        )

        cache_file = self.cache_manager._get_cache_filepath(cache_key, indicator_type)
        meta_file = self.cache_manager._get_metadata_filepath(cache_key, indicator_type)

        if cache_file.exists():
            cache_file.unlink()
        if meta_file.exists():
            meta_file.unlink()

        # Calcul forcé
        try:
            result = self._compute_indicator(indicator_type, params, data)

            # Nouvelle sauvegarde
            self.cache_manager.save_to_cache(
                cache_key, indicator_type, result, params, symbol, timeframe
            )
            self.stats["computations"] += 1

            # Mise à jour registry
            if self.settings.auto_registry_update:
                self._update_registry(
                    indicator_type, cache_key, params, symbol, timeframe
                )

            return result

        except Exception as e:
            logger.error(f"❌ Erreur force recompute {indicator_type}: {e}")
            raise RuntimeError(
                f"Failed to force recompute {indicator_type} with params {params}"
            ) from e

    def _batch_ensure_parallel(
        self,
        indicator_type: str,
        params_list: List[Dict[str, Any]],
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        symbol: str,
        timeframe: str,
    ) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]:
        """Batch ensure parallèle avec ThreadPoolExecutor"""

        logger.info(
            f"🚀 Batch parallel {indicator_type}: {len(params_list)} paramètres, {self.settings.max_workers} workers"
        )

        results = {}

        with ThreadPoolExecutor(max_workers=self.settings.max_workers) as executor:
            # Soumission des tâches
            future_to_params = {}
            for params in params_list:
                future = executor.submit(
                    self.ensure, indicator_type, params, data, symbol, timeframe
                )
                future_to_params[future] = params

            # Collecte des résultats
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                key = self._params_to_key(params)

                try:
                    result = future.result(timeout=30.0)  # 30s timeout
                    results[key] = result
                except Exception as e:
                    logger.error(f"❌ Erreur batch {key}: {e}")
                    results[key] = None

        return results

    def _params_to_key(self, params: Dict[str, Any]) -> str:
        """Conversion paramètres en clé string"""
        sorted_params = dict(sorted(params.items()))
        key_parts = []
        for k, v in sorted_params.items():
            if isinstance(v, float):
                key_parts.append(f"{k}={v:.3f}")
            else:
                key_parts.append(f"{k}={v}")
        return "_".join(key_parts)

    def _update_registry(
        self,
        indicator_type: str,
        cache_key: str,
        params: Dict[str, Any],
        symbol: str,
        timeframe: str,
    ):
        """Mise à jour du registry avec nouvel indicateur"""
        registry_file = (
            self.settings.cache_path / "registry" / f"{indicator_type}_registry.parquet"
        )

        # Nouvelle entrée
        new_entry = {
            "cache_key": cache_key,
            "indicator_type": indicator_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": json.dumps(params, sort_keys=True),
            "created_at": pd.Timestamp.now(),
            "last_accessed": pd.Timestamp.now(),
        }

        try:
            # Chargement registry existant
            if registry_file.exists():
                registry_df = pd.read_parquet(registry_file)
                # Mise à jour ou ajout
                mask = registry_df["cache_key"] == cache_key
                if mask.any():
                    registry_df.loc[mask, "last_accessed"] = new_entry["last_accessed"]
                else:
                    registry_df = pd.concat(
                        [registry_df, pd.DataFrame([new_entry])], ignore_index=True
                    )
            else:
                # Nouveau registry
                registry_df = pd.DataFrame([new_entry])

            # Sauvegarde
            registry_df.to_parquet(registry_file, index=False)
            logger.debug(f"📋 Registry mis à jour: {indicator_type}")

        except Exception as e:
            logger.warning(f"⚠️ Erreur mise à jour registry: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Statistiques d'utilisation"""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            (self.stats["cache_hits"] / total_requests * 100)
            if total_requests > 0
            else 0
        )

        return {
            **self.stats,
            "cache_hit_rate_pct": cache_hit_rate,
            "total_requests": total_requests,
        }

    def cleanup_cache(self) -> int:
        """Nettoyage cache expiré"""
        return self.cache_manager.cleanup_expired()


# ========================================
# FONCTIONS PUBLIQUES (API simplifiée)
# ========================================

# Instance globale pour API simplifiée
_global_bank: Optional[IndicatorBank] = None


def _get_global_bank(cache_dir: str = "indicators_cache") -> IndicatorBank:
    """Récupération instance globale IndicatorBank"""
    global _global_bank
    if _global_bank is None or _global_bank.settings.cache_dir != cache_dir:
        settings = IndicatorSettings(cache_dir=cache_dir)
        _global_bank = IndicatorBank(settings)
    return _global_bank


def ensure_indicator(
    indicator_type: str,
    params: Dict[str, Any],
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    symbol: str = "",
    timeframe: str = "",
    cache_dir: str = "indicators_cache",
) -> Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]]:
    """
    API simple pour ensure indicateur

    Args:
        indicator_type: 'bollinger' ou 'atr'
        params: Paramètres indicateur
        data: Données OHLCV
        symbol, timeframe: Métadonnées (optionnel)
        cache_dir: Répertoire cache (défaut: "indicators_cache")

    Returns:
        Résultat indicateur

    Exemple:
        ```python
        from threadx.indicators.bank import ensure_indicator
        import pandas as pd

        # Données OHLCV
        ohlcv = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [101, 102, 103],
            'volume': [1000, 1500, 1200]
        })

        # Bollinger Bands
        bb_result = ensure_indicator(
            'bollinger',
            {'period': 20, 'std': 2.0},
            ohlcv,
            symbol='BTCUSDC',
            timeframe='15m'
        )
        if bb_result:
            upper, middle, lower = bb_result
            print(f"BB dernier: upper={upper[-1]:.2f}")

        # ATR
        atr_result = ensure_indicator(
            'atr',
            {'period': 14, 'method': 'ema'},
            ohlcv
        )
        if atr_result:
            print(f"ATR dernier: {atr_result[-1]:.4f}")
        ```
    """
    bank = _get_global_bank(cache_dir)
    return bank.ensure(indicator_type, params, data, symbol, timeframe)


def force_recompute_indicator(
    indicator_type: str,
    params: Dict[str, Any],
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    symbol: str = "",
    timeframe: str = "",
    cache_dir: str = "indicators_cache",
) -> Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]]:
    """
    API simple pour force recompute indicateur

    Args:
        Mêmes que ensure_indicator()

    Returns:
        Résultat indicateur recalculé

    Exemple:
        ```python
        # Force recalcul même si cache valide
        bb_result = force_recompute_indicator(
            'bollinger',
            {'period': 20, 'std': 2.0},
            ohlcv_data
        )
        ```
    """
    bank = _get_global_bank(cache_dir)
    return bank.force_recompute(indicator_type, params, data, symbol, timeframe)


def batch_ensure_indicators(
    indicator_type: str,
    params_list: List[Dict[str, Any]],
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    symbol: str = "",
    timeframe: str = "",
    cache_dir: str = "indicators_cache",
) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, ...]]]:
    """
    API simple pour batch ensure

    Args:
        indicator_type: Type d'indicateur
        params_list: Liste paramètres
        data: Données communes
        symbol, timeframe: Métadonnées
        cache_dir: Répertoire cache

    Returns:
        Dict[param_key] = résultat

    Exemple:
        ```python
        # Batch Bollinger avec différents paramètres
        params_list = [
            {'period': 20, 'std': 2.0},
            {'period': 50, 'std': 1.5},
            {'period': 10, 'std': 2.5}
        ]

        results = batch_ensure_indicators('bollinger', params_list, ohlcv_data)

        for key, result in results.items():
            if result:
                upper, middle, lower = result
                print(f"{key}: dernier upper={upper[-1]:.2f}")
        ```
    """
    bank = _get_global_bank(cache_dir)
    return bank.batch_ensure(indicator_type, params_list, data, symbol, timeframe)


def get_bank_stats(cache_dir: str = "indicators_cache") -> Dict[str, Any]:
    """Statistiques de la banque d'indicateurs"""
    bank = _get_global_bank(cache_dir)
    return bank.get_stats()


def cleanup_indicators_cache(cache_dir: str = "indicators_cache") -> int:
    """Nettoyage du cache d'indicateurs"""
    bank = _get_global_bank(cache_dir)
    return bank.cleanup_cache()


# ========================================
# UTILITAIRES ET VALIDATION
# ========================================


def validate_bank_integrity(cache_dir: str = "indicators_cache") -> Dict[str, Any]:
    """
    Validation intégrité complète de la banque

    Returns:
        Dict avec résultats validation
    """
    cache_path = Path(cache_dir)
    results = {
        "total_indicators": 0,
        "valid_cache": 0,
        "invalid_cache": 0,
        "orphaned_files": 0,
        "corrupted_files": 0,
        "expired_files": 0,
        "details": {},
    }

    if not cache_path.exists():
        results["error"] = f"Cache directory does not exist: {cache_dir}"
        return results

    settings = IndicatorSettings(cache_dir=cache_dir)
    cache_manager = CacheManager(settings)

    # Scan par type d'indicateur
    for indicator_type in ["bollinger", "atr"]:
        indicator_dir = cache_path / indicator_type
        if not indicator_dir.exists():
            continue

        type_stats = {"valid": 0, "invalid": 0, "expired": 0, "corrupted": 0}

        for meta_file in indicator_dir.glob("*.meta"):
            results["total_indicators"] += 1
            cache_key = meta_file.stem

            try:
                # Validation cache
                if cache_manager.is_cache_valid(cache_key, indicator_type):
                    type_stats["valid"] += 1
                    results["valid_cache"] += 1
                else:
                    type_stats["invalid"] += 1
                    results["invalid_cache"] += 1

                    # Détail de l'invalidité
                    metadata = cache_manager._read_metadata(meta_file)
                    if not metadata:
                        type_stats["corrupted"] += 1
                        results["corrupted_files"] += 1
                        continue

                    created_at = metadata.get("created_at", 0)
                    if time.time() - created_at > settings.ttl_seconds:
                        type_stats["expired"] += 1
                        results["expired_files"] += 1

            except Exception as e:
                type_stats["corrupted"] += 1
                results["corrupted_files"] += 1
                logger.warning(f"⚠️ Fichier corrompu {meta_file}: {e}")

        results["details"][indicator_type] = type_stats

    # Fichiers orphelins (parquet sans meta)
    for indicator_dir in cache_path.iterdir():
        if not indicator_dir.is_dir() or indicator_dir.name not in ["bollinger", "atr"]:
            continue

        parquet_files = set(f.stem for f in indicator_dir.glob("*.parquet"))
        meta_files = set(f.stem for f in indicator_dir.glob("*.meta"))

        orphaned = parquet_files - meta_files
        results["orphaned_files"] += len(orphaned)

    return results


def benchmark_bank_performance(
    cache_dir: str = "indicators_cache", n_indicators: int = 100, data_size: int = 1000
) -> Dict[str, Any]:
    """
    Benchmark performance de la banque d'indicateurs

    Args:
        cache_dir: Répertoire cache
        n_indicators: Nombre d'indicateurs à tester
        data_size: Taille des données test

    Returns:
        Dict avec métriques performance
    """
    logger.info(f"🏁 Benchmark IndicatorBank: {n_indicators} indicateurs")

    # Données test
    np.random.seed(42)
    ohlcv = pd.DataFrame(
        {
            "open": np.random.randn(data_size) * 5 + 100,
            "high": np.random.randn(data_size) * 5 + 105,
            "low": np.random.randn(data_size) * 5 + 95,
            "close": np.random.randn(data_size) * 5 + 100,
            "volume": np.random.randint(1000, 10000, data_size),
        }
    )

    # Paramètres test
    bb_params = [
        {"period": p, "std": s} for p in range(10, 30, 2) for s in [1.5, 2.0, 2.5]
    ][: n_indicators // 2]

    atr_params = [
        {"period": p, "method": m} for p in range(10, 30, 2) for m in ["ema", "sma"]
    ][: n_indicators // 2]

    bank = IndicatorBank(IndicatorSettings(cache_dir=cache_dir))

    results = {
        "setup": {
            "n_indicators": len(bb_params) + len(atr_params),
            "data_size": data_size,
            "cache_dir": cache_dir,
        },
        "timings": {},
        "cache_performance": {},
    }

    # Test 1: Calculs initiaux (cache cold)
    start_time = time.time()

    for params in bb_params:
        bank.ensure("bollinger", params, ohlcv)

    for params in atr_params:
        bank.ensure("atr", params, ohlcv)

    cold_time = time.time() - start_time
    results["timings"]["cold_cache"] = cold_time

    # Test 2: Rechargement depuis cache (cache warm)
    start_time = time.time()

    for params in bb_params:
        bank.ensure("bollinger", params, ohlcv)

    for params in atr_params:
        bank.ensure("atr", params, ohlcv)

    warm_time = time.time() - start_time
    results["timings"]["warm_cache"] = warm_time

    # Test 3: Batch processing
    start_time = time.time()
    bank.batch_ensure("bollinger", bb_params, ohlcv)
    bank.batch_ensure("atr", atr_params, ohlcv)
    batch_time = time.time() - start_time
    results["timings"]["batch_processing"] = batch_time

    # Stats cache
    stats = bank.get_stats()
    results["cache_performance"] = {
        "hit_rate_pct": stats["cache_hit_rate_pct"],
        "total_requests": stats["total_requests"],
        "cache_hits": stats["cache_hits"],
        "cache_misses": stats["cache_misses"],
        "computations": stats["computations"],
        "speedup_warm": cold_time / warm_time if warm_time > 0 else 0,
    }

    logger.info("✅ Benchmark terminé")
    logger.info(f"   Cold cache: {cold_time:.2f}s")
    logger.info(f"   Warm cache: {warm_time:.2f}s")
    logger.info(f"   Batch: {batch_time:.2f}s")
    logger.info(f"   Cache hit rate: {stats['cache_hit_rate_pct']:.1f}%")
    logger.info(
        f"   Speedup warm: {cold_time/warm_time:.1f}x"
        if warm_time > 0
        else "   Speedup: N/A"
    )

    return results


if __name__ == "__main__":
    # Test rapide
    print("🏦 ThreadX IndicatorBank - Test rapide")

    # Données test OHLCV
    np.random.seed(42)
    n = 1000
    ohlcv = pd.DataFrame(
        {
            "open": np.random.randn(n) * 5 + 100,
            "high": np.random.randn(n) * 5 + 105,
            "low": np.random.randn(n) * 5 + 95,
            "close": np.random.randn(n) * 5 + 100,
            "volume": np.random.randint(1000, 10000, n),
        }
    )

    # Test ensure simple
    print("\n📊 Test ensure simple...")
    bb_result = ensure_indicator(
        "bollinger",
        {"period": 20, "std": 2.0},
        ohlcv,
        symbol="TESTBTC",
        timeframe="15m",
        cache_dir="test_cache",
    )

    if bb_result:
        upper, middle, lower = bb_result
        print(f"✅ Bollinger: {len(upper)} points")
        print(f"   Upper[-1]: {upper[-1]:.2f}")
        print(f"   Middle[-1]: {middle[-1]:.2f}")
        print(f"   Lower[-1]: {lower[-1]:.2f}")

    # Test cache hit
    print("\n🎯 Test cache hit...")
    start = time.time()
    bb_result2 = ensure_indicator(
        "bollinger",
        {"period": 20, "std": 2.0},
        ohlcv,
        symbol="TESTBTC",
        timeframe="15m",
        cache_dir="test_cache",
    )
    cache_time = time.time() - start
    print(f"✅ Cache hit: {cache_time:.4f}s")

    # Test ATR
    print("\n📈 Test ATR...")
    atr_result = ensure_indicator(
        "atr", {"period": 14, "method": "ema"}, ohlcv, cache_dir="test_cache"
    )

    if atr_result is not None:
        print(f"✅ ATR: {len(atr_result)} points")
        print(f"   ATR[-1]: {atr_result[-1]:.4f}")
        print(f"   ATR moyen: {np.nanmean(atr_result):.4f}")

    # Stats
    print("\n📊 Stats bancaire...")
    stats = get_bank_stats("test_cache")
    print(f"   Hit rate: {stats['cache_hit_rate_pct']:.1f}%")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Computations: {stats['computations']}")

    print("\n✅ Tests terminés!")
