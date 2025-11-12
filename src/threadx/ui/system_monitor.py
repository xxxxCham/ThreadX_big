"""
ThreadX System Monitor - Monitoring Temps Réel CPU/GPU
======================================================

Module de monitoring système pour visualiser l'utilisation des ressources
pendant les backtests et optimisations.

Features:
- Monitoring CPU (utilisation, mémoire)
- Monitoring GPU 1 (RTX 5090) - utilisation, mémoire
- Monitoring GPU 2 (RTX 2060) - utilisation, mémoire
- Thread-safe data collection
- Streamlit-ready visualizations

Author: ThreadX Framework
Version: 1.0
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import pandas as pd
import psutil

logger = logging.getLogger(__name__)

# Import GPU monitoring
try:
    import cupy as cp  # noqa: F401

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from threadx.gpu.device_manager import (  # noqa: F401
        get_device_by_name,
        list_devices,
    )

    GPU_MANAGER_AVAILABLE = True
except ImportError:
    GPU_MANAGER_AVAILABLE = False

# Import NVIDIA Management Library pour vraies métriques GPU
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class SystemSnapshot:
    """
    Snapshot instantané des ressources système.

    Attributes:
        timestamp: Temps du snapshot (epoch)
        cpu_percent: Utilisation CPU (0-100)
        memory_percent: Utilisation mémoire RAM (0-100)
        gpu1_percent: Utilisation GPU 1 (0-100)
        gpu1_memory_percent: Mémoire GPU 1 (0-100)
        gpu1_temperature: Température GPU 1 (°C)
        gpu1_power_usage: Consommation GPU 1 (W)
        gpu2_percent: Utilisation GPU 2 (0-100)
        gpu2_memory_percent: Mémoire GPU 2 (0-100)
        gpu2_temperature: Température GPU 2 (°C)
        gpu2_power_usage: Consommation GPU 2 (W)
    """

    timestamp: float
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu1_percent: float = 0.0
    gpu1_memory_percent: float = 0.0
    gpu1_temperature: float = 0.0
    gpu1_power_usage: float = 0.0
    gpu2_percent: float = 0.0
    gpu2_memory_percent: float = 0.0
    gpu2_temperature: float = 0.0
    gpu2_power_usage: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convertit en dictionnaire."""
        return {
            "timestamp": self.timestamp,
            "cpu": self.cpu_percent,
            "memory": self.memory_percent,
            "gpu1": self.gpu1_percent,
            "gpu1_mem": self.gpu1_memory_percent,
            "gpu1_temp": self.gpu1_temperature,
            "gpu1_power": self.gpu1_power_usage,
            "gpu2": self.gpu2_percent,
            "gpu2_mem": self.gpu2_memory_percent,
            "gpu2_temp": self.gpu2_temperature,
            "gpu2_power": self.gpu2_power_usage,
        }


class SystemMonitor:
    """
    Moniteur système temps réel avec collecte thread-safe.

    Collecte les métriques CPU/RAM/GPU dans un thread séparé
    et fournit des méthodes pour visualisation Streamlit.

    Example:
        >>> monitor = SystemMonitor(interval=0.5, max_history=120)
        >>> monitor.start()
        >>> # ... exécution de code ...
        >>> snapshot = monitor.get_latest_snapshot()
        >>> monitor.stop()
    """

    def __init__(self, interval: float = 0.5, max_history: int = 120):
        """
        Initialise le moniteur système.

        Args:
            interval: Intervalle de collecte en secondes (défaut: 0.5s)
            max_history: Nombre max de snapshots gardés en mémoire (défaut: 120 = 1min à 0.5s)
        """
        self.interval = interval
        self.max_history = max_history

        # Thread de collecte
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Historique des snapshots
        self._history: deque = deque(maxlen=max_history)

        # Détection GPU via pynvml
        self._gpu1_handle = None
        self._gpu2_handle = None
        self._gpu_available = False

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()

                # Identifier GPUs par nom
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")

                    if "5090" in name or "5080" in name:  # RTX 5090/5080
                        self._gpu1_handle = handle
                        logger.info(f"GPU 1 détecté: {name} (index {i})")
                    elif "2060" in name:  # RTX 2060
                        self._gpu2_handle = handle
                        logger.info(f"GPU 2 détecté: {name} (index {i})")

                self._gpu_available = (self._gpu1_handle is not None) or (
                    self._gpu2_handle is not None
                )

            except Exception as e:
                logger.warning(f"Erreur initialisation pynvml: {e}", exc_info=True)
                self._gpu_available = False
        else:
            logger.info("pynvml non disponible - monitoring GPU désactivé")

        logger.info(
            f"SystemMonitor initialisé: interval={interval}s, max_history={max_history}"
        )

    def _collect_cpu_metrics(self) -> dict[str, float]:
        """Collecte les métriques CPU/RAM."""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
            }
        except Exception as e:
            logger.warning(f"Erreur collecte CPU: {e}", exc_info=True)
            return {"cpu_percent": 0.0, "memory_percent": 0.0}

    def _collect_gpu_metrics(self) -> dict[str, float]:
        """Collecte les métriques GPU via pynvml."""
        metrics = {
            "gpu1_percent": 0.0,
            "gpu1_memory_percent": 0.0,
            "gpu1_temperature": 0.0,
            "gpu1_power_usage": 0.0,
            "gpu2_percent": 0.0,
            "gpu2_memory_percent": 0.0,
            "gpu2_temperature": 0.0,
            "gpu2_power_usage": 0.0,
        }

        if not self._gpu_available or not PYNVML_AVAILABLE:
            return metrics

        try:
            # GPU 1 (RTX 5090/5080)
            if self._gpu1_handle:
                # Utilisation GPU (%)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu1_handle)
                metrics["gpu1_percent"] = float(util.gpu)

                # Mémoire VRAM (%)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu1_handle)
                metrics["gpu1_memory_percent"] = (
                    (mem_info.used / mem_info.total * 100)
                    if mem_info.total > 0
                    else 0.0
                )

                # Température (°C)
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._gpu1_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                metrics["gpu1_temperature"] = float(temp)

                # Consommation (W)
                power = (
                    pynvml.nvmlDeviceGetPowerUsage(self._gpu1_handle) / 1000.0
                )  # mW → W
                metrics["gpu1_power_usage"] = float(power)

            # GPU 2 (RTX 2060)
            if self._gpu2_handle:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu2_handle)
                metrics["gpu2_percent"] = float(util.gpu)

                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu2_handle)
                metrics["gpu2_memory_percent"] = (
                    (mem_info.used / mem_info.total * 100)
                    if mem_info.total > 0
                    else 0.0
                )

                temp = pynvml.nvmlDeviceGetTemperature(
                    self._gpu2_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                metrics["gpu2_temperature"] = float(temp)

                power = pynvml.nvmlDeviceGetPowerUsage(self._gpu2_handle) / 1000.0
                metrics["gpu2_power_usage"] = float(power)

        except Exception as e:
            logger.warning(f"Erreur collecte GPU: {e}", exc_info=True)

        return metrics

    def _collect_snapshot(self) -> SystemSnapshot:
        """Collecte un snapshot complet du système."""
        timestamp = time.time()

        # Collecte CPU/RAM
        cpu_metrics = self._collect_cpu_metrics()

        # Collecte GPU
        gpu_metrics = self._collect_gpu_metrics()

        return SystemSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_metrics["cpu_percent"],
            memory_percent=cpu_metrics["memory_percent"],
            gpu1_percent=gpu_metrics["gpu1_percent"],
            gpu1_memory_percent=gpu_metrics["gpu1_memory_percent"],
            gpu1_temperature=gpu_metrics["gpu1_temperature"],
            gpu1_power_usage=gpu_metrics["gpu1_power_usage"],
            gpu2_percent=gpu_metrics["gpu2_percent"],
            gpu2_memory_percent=gpu_metrics["gpu2_memory_percent"],
            gpu2_temperature=gpu_metrics["gpu2_temperature"],
            gpu2_power_usage=gpu_metrics["gpu2_power_usage"],
        )

    def _monitoring_loop(self):
        """Boucle de monitoring (exécutée dans thread séparé)."""
        logger.info("Thread monitoring démarré")

        while self._running:
            try:
                snapshot = self._collect_snapshot()

                with self._lock:
                    self._history.append(snapshot)

                time.sleep(self.interval)

            except Exception as e:
                logger.error(f"Erreur dans monitoring loop: {e}", exc_info=True)
                time.sleep(self.interval)

        logger.info("Thread monitoring arrêté")

    def start(self) -> None:
        """Démarre la collecte de métriques."""
        if self._running:
            logger.warning("Monitoring déjà démarré")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="SystemMonitor"
        )
        self._thread.start()

        logger.info("Monitoring démarré")

    def stop(self) -> None:
        """Arrête la collecte de métriques."""
        if not self._running:
            return

        self._running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("Monitoring arrêté")

    def get_latest_snapshot(self) -> SystemSnapshot | None:
        """
        Récupère le snapshot le plus récent.

        Returns:
            SystemSnapshot ou None si aucune donnée
        """
        with self._lock:
            if not self._history:
                return None
            return self._history[-1]

    def get_history(self, n_last: int | None = None) -> list[SystemSnapshot]:
        """
        Récupère l'historique des snapshots.

        Args:
            n_last: Nombre de derniers snapshots (None = tous)

        Returns:
            Liste des snapshots
        """
        with self._lock:
            history = list(self._history)

            if n_last is not None and n_last > 0:
                return history[-n_last:]

            return history

    def get_history_df(self, n_last: int | None = None) -> pd.DataFrame:
        """
        Récupère l'historique sous forme de DataFrame.

        Args:
            n_last: Nombre de derniers snapshots

        Returns:
            DataFrame avec colonnes [timestamp, cpu, memory, gpu1, gpu1_mem, gpu1_temp, gpu1_power, gpu2, gpu2_mem, gpu2_temp, gpu2_power]
        """
        history = self.get_history(n_last)

        if not history:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "cpu",
                    "memory",
                    "gpu1",
                    "gpu1_mem",
                    "gpu1_temp",
                    "gpu1_power",
                    "gpu2",
                    "gpu2_mem",
                    "gpu2_temp",
                    "gpu2_power",
                ]
            )

        data = [snap.to_dict() for snap in history]
        df = pd.DataFrame(data)

        # Convertir timestamp en datetime relatif (secondes depuis début)
        if not df.empty:
            df["time"] = df["timestamp"] - df["timestamp"].iloc[0]

        return df

    def get_history_dataframe(self, n_last: int | None = None) -> pd.DataFrame:
        """Alias de get_history_df() pour compatibilité."""
        return self.get_history_df(n_last)

    def clear_history(self) -> None:
        """Vide l'historique des snapshots."""
        with self._lock:
            self._history.clear()

        logger.info("Historique monitoring vidé")

    def get_stats_summary(self) -> dict[str, Any]:
        """
        Calcule les statistiques résumées sur l'historique.

        Returns:
            Dict avec moyennes, max, min pour chaque métrique
        """
        df = self.get_history_df()

        if df.empty:
            return {}

        summary = {
            "cpu_mean": df["cpu"].mean(),
            "cpu_max": df["cpu"].max(),
            "memory_mean": df["memory"].mean(),
            "memory_max": df["memory"].max(),
            "gpu1_mean": df["gpu1"].mean(),
            "gpu1_max": df["gpu1"].max(),
            "gpu1_mem_mean": df["gpu1_mem"].mean(),
            "gpu1_mem_max": df["gpu1_mem"].max(),
            "gpu2_mean": df["gpu2"].mean(),
            "gpu2_max": df["gpu2"].max(),
            "gpu2_mem_mean": df["gpu2_mem"].mean(),
            "gpu2_mem_max": df["gpu2_mem"].max(),
            "duration_seconds": df["time"].iloc[-1] if len(df) > 0 else 0.0,
            "n_samples": len(df),
        }

        return summary

    def is_running(self) -> bool:
        """Vérifie si le monitoring est actif."""
        return self._running

    def __enter__(self):
        """Context manager: démarre le monitoring."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: arrête le monitoring."""
        self.stop()


# Singleton global pour partage entre composants
_global_monitor: SystemMonitor | None = None


def get_global_monitor() -> SystemMonitor:
    """
    Récupère le moniteur système global (singleton).

    Returns:
        Instance SystemMonitor partagée

    Example:
        >>> monitor = get_global_monitor()
        >>> monitor.start()
        >>> # ... utilisation ...
        >>> monitor.stop()
    """
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = SystemMonitor(interval=0.5, max_history=120)

    return _global_monitor
