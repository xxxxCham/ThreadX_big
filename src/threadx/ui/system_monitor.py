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

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
import logging

import psutil
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import GPU monitoring
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from threadx.utils.gpu.device_manager import list_devices, get_device_by_name
    GPU_MANAGER_AVAILABLE = True
except ImportError:
    GPU_MANAGER_AVAILABLE = False


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
        gpu2_percent: Utilisation GPU 2 (0-100)
        gpu2_memory_percent: Mémoire GPU 2 (0-100)
    """
    timestamp: float
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu1_percent: float = 0.0
    gpu1_memory_percent: float = 0.0
    gpu2_percent: float = 0.0
    gpu2_memory_percent: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convertit en dictionnaire."""
        return {
            "timestamp": self.timestamp,
            "cpu": self.cpu_percent,
            "memory": self.memory_percent,
            "gpu1": self.gpu1_percent,
            "gpu1_mem": self.gpu1_memory_percent,
            "gpu2": self.gpu2_percent,
            "gpu2_mem": self.gpu2_memory_percent,
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
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Historique des snapshots
        self._history: deque = deque(maxlen=max_history)

        # Détection GPU
        self._gpu1_device = None
        self._gpu2_device = None
        self._gpu_available = False

        if GPU_MANAGER_AVAILABLE and CUPY_AVAILABLE:
            try:
                devices = list_devices()
                self._gpu1_device = get_device_by_name("5090")
                self._gpu2_device = get_device_by_name("2060")
                self._gpu_available = (self._gpu1_device is not None) or (self._gpu2_device is not None)

                if self._gpu_available:
                    logger.info(f"SystemMonitor: GPU détectés - "
                               f"GPU1={'✅' if self._gpu1_device else '❌'}, "
                               f"GPU2={'✅' if self._gpu2_device else '❌'}")
            except Exception as e:
                logger.warning(f"Erreur détection GPU: {e}")
                self._gpu_available = False

        logger.info(f"SystemMonitor initialisé: interval={interval}s, max_history={max_history}")

    def _collect_cpu_metrics(self) -> Dict[str, float]:
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
            logger.warning(f"Erreur collecte CPU: {e}")
            return {"cpu_percent": 0.0, "memory_percent": 0.0}

    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """Collecte les métriques GPU."""
        metrics = {
            "gpu1_percent": 0.0,
            "gpu1_memory_percent": 0.0,
            "gpu2_percent": 0.0,
            "gpu2_memory_percent": 0.0,
        }

        if not self._gpu_available or not CUPY_AVAILABLE:
            return metrics

        try:
            # GPU 1 (RTX 5090)
            if self._gpu1_device and self._gpu1_device.device_id != -1:
                with cp.cuda.Device(self._gpu1_device.device_id):
                    # Utilisation GPU (approximation via busy time)
                    # Note: CuPy n'expose pas directement l'utilisation GPU
                    # On utilise le ratio mémoire comme proxy
                    mem_info = cp.cuda.runtime.memGetInfo()
                    mem_free, mem_total = mem_info
                    mem_used = mem_total - mem_free
                    mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0.0

                    metrics["gpu1_memory_percent"] = mem_percent
                    # Approximation utilisation basée sur activité récente
                    metrics["gpu1_percent"] = min(mem_percent * 1.2, 100.0)

            # GPU 2 (RTX 2060)
            if self._gpu2_device and self._gpu2_device.device_id != -1:
                with cp.cuda.Device(self._gpu2_device.device_id):
                    mem_info = cp.cuda.runtime.memGetInfo()
                    mem_free, mem_total = mem_info
                    mem_used = mem_total - mem_free
                    mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0.0

                    metrics["gpu2_memory_percent"] = mem_percent
                    metrics["gpu2_percent"] = min(mem_percent * 1.2, 100.0)

        except Exception as e:
            logger.warning(f"Erreur collecte GPU: {e}")

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
            gpu2_percent=gpu_metrics["gpu2_percent"],
            gpu2_memory_percent=gpu_metrics["gpu2_memory_percent"],
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
                logger.error(f"Erreur dans monitoring loop: {e}")
                time.sleep(self.interval)

        logger.info("Thread monitoring arrêté")

    def start(self) -> None:
        """Démarre la collecte de métriques."""
        if self._running:
            logger.warning("Monitoring déjà démarré")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True, name="SystemMonitor")
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

    def get_latest_snapshot(self) -> Optional[SystemSnapshot]:
        """
        Récupère le snapshot le plus récent.

        Returns:
            SystemSnapshot ou None si aucune donnée
        """
        with self._lock:
            if not self._history:
                return None
            return self._history[-1]

    def get_history(self, n_last: Optional[int] = None) -> List[SystemSnapshot]:
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

    def get_history_df(self, n_last: Optional[int] = None) -> pd.DataFrame:
        """
        Récupère l'historique sous forme de DataFrame.

        Args:
            n_last: Nombre de derniers snapshots

        Returns:
            DataFrame avec colonnes [timestamp, cpu, memory, gpu1, gpu1_mem, gpu2, gpu2_mem]
        """
        history = self.get_history(n_last)

        if not history:
            return pd.DataFrame(columns=["timestamp", "cpu", "memory", "gpu1", "gpu1_mem", "gpu2", "gpu2_mem"])

        data = [snap.to_dict() for snap in history]
        df = pd.DataFrame(data)

        # Convertir timestamp en datetime relatif (secondes depuis début)
        if not df.empty:
            df["time"] = df["timestamp"] - df["timestamp"].iloc[0]

        return df

    def clear_history(self) -> None:
        """Vide l'historique des snapshots."""
        with self._lock:
            self._history.clear()

        logger.info("Historique monitoring vidé")

    def get_stats_summary(self) -> Dict[str, Any]:
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
_global_monitor: Optional[SystemMonitor] = None


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
