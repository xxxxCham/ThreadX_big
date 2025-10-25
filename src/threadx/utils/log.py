"""
ThreadX Centralized Logging System - 
==============================================

Provides logger configuration with rotation, Windows-compatible file handling,
and Settings/TOML integration for production-ready applications.

Author: ThreadX Framework
Version: Phase 7 - Sweep & Logging
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import os
from threading import Lock

# Global setup lock to prevent double initialization
_setup_lock = Lock()
_setup_done = False


def configure_logging(
    level: str = "INFO", log_file: Optional[str] = None, console: bool = True
) -> None:
    """
    Configure logging avec niveau et options spécifiques.

    Args:
        level: Niveau de log ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        log_file: Chemin du fichier de log (optionnel)
        console: Activer logs console
    """
    # Convertir string niveau en constante
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Niveau de log invalide: {level}")

    # Configuration de base
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Supprimer handlers existants
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Format standard
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler console
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Handler fichier
    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logging.error(f"Impossible de configurer log file {log_file}: {e}")


def setup_logging_once() -> None:
    """
    Configure logging system once globally.

    Idempotent function that sets up console and file handlers with rotation.
    Prevents duplicate handlers during multiple calls.

    Notes
    -----
    Thread-safe initialization using locks.
    Creates logs directory if missing.
    Windows-compatible file handling.
    """
    global _setup_done

    with _setup_lock:
        if _setup_done:
            return

        try:
            # Import Settings here to avoid circular dependencies
            try:
                from threadx.utils.settings import Settings

                log_dir = Settings.LOG_DIR
                log_level = Settings.LOG_LEVEL
            except ImportError:
                # Valeurs par défaut si Settings n'est pas disponible
                log_dir = Path("logs")
                log_level = "INFO"
        except (ImportError, AttributeError):
            # Fallback if Settings not available
            log_dir = Path("logs")
            log_level = "INFO"

        # Ensure log directory exists
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)

        # File handler with rotation (Windows-safe)
        log_file = log_dir / "threadx.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)

        _setup_done = True


def get_logger(
    name: str,
    *,
    log_dir: Optional[Path] = None,
    level: Optional[str] = None,
    rotate_max_bytes: int = 10 * 1024 * 1024,
    rotate_backups: int = 5,
) -> logging.Logger:
    """
    Crée ou récupère un logger ThreadX.

    Args:
        name: Nom du logger (généralement __name__)
        level: Niveau de log optionnel ('DEBUG', 'INFO', 'WARNING', 'ERROR')

    Returns:
        Logger configuré pour ThreadX
    """
    logger = logging.getLogger(name)

    # Configuration uniquement si pas déjà configuré
    if not logger.handlers:
        # Handler console
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Niveau par défaut
        log_level = level or "INFO"
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Éviter propagation aux loggers parents
        logger.propagate = False

    return logger


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO") -> None:
    """
    Configuration globale du logging ThreadX.

    Args:
        log_file: Fichier de log optionnel
        level: Niveau de log global
    """
    # Configuration root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Nettoyer handlers existants
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler si spécifié
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


# Logger par défaut pour ce module
logger = get_logger(__name__)
