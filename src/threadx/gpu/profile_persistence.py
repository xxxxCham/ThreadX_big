"""
ThreadX Utils - Persistance des Profils GPU
===========================================

Fonctions utilitaires pour la persistance et la lecture sécurisée
des profils GPU et multi-GPU dans des fichiers JSON.

Features:
- Lecture/écriture atomiques avec gestion des erreurs
- Backup automatique des fichiers corrompus
- Validation des structures de données
- Merge intelligent des profils

Author: ThreadX Team
Version: Phase B - GPU Dynamic & Multi-GPU
"""

import os
import json
import time
import shutil
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import numpy as np

from threadx.utils.log import get_logger
from threadx.config.settings import S

logger = get_logger(__name__)

# Constantes pour les chemins de profils
PROFILES_DIR = Path("artifacts") / "profiles"
GPU_THRESHOLDS_FILE = PROFILES_DIR / "gpu_thresholds.json"
MULTIGPU_RATIOS_FILE = PROFILES_DIR / "multigpu_ratios.json"

# Structure par défaut pour les profils
DEFAULT_GPU_THRESHOLDS = {
    "version": 1,
    "updated_at": datetime.now().isoformat(),
    "entries": {},
    "defaults": {
        "decision_threshold": 1.75,
        "min_samples": 3,
        "n_min_gpu": 10000,
        "hysteresis": 0.10,
    },
}

DEFAULT_MULTIGPU_RATIOS = {
    "version": 1,
    "updated_at": datetime.now().isoformat(),
    "devices": [],
    "ratios": {},
    "sample_size": 3,
    "workload_tag": "indicators/batch_default",
    "ttl_days": 14,
}


def ensure_profiles_dir() -> None:
    """Assure que le dossier des profils existe."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def safe_read_json(file_path: Path) -> Tuple[Dict[str, Any], bool]:
    """
    Lit un fichier JSON de façon sécurisée avec gestion d'erreurs.

    Args:
        file_path: Chemin vers le fichier JSON

    Returns:
        Tuple contenant (data, success)
        - data: Contenu JSON ou structure par défaut
        - success: True si lecture réussie, False sinon
    """
    ensure_profiles_dir()

    if not file_path.exists():
        return ({}, False)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data, True)
    except (json.JSONDecodeError, IOError) as e:
        # Créer un backup du fichier corrompu avec timestamp
        if file_path.exists():
            timestamp = int(time.time())
            backup_path = file_path.with_suffix(f".bak.{timestamp}")
            try:
                shutil.copy2(file_path, backup_path)
                logger.warning(f"Fichier JSON corrompu, backup créé: {backup_path}")
            except Exception as backup_err:
                logger.error(f"Impossible de créer un backup: {backup_err}")

        logger.error(f"Erreur de lecture JSON {file_path}: {e}")
        return ({}, False)


def safe_write_json(file_path: Path, data: Dict[str, Any]) -> bool:
    """
    Écrit un fichier JSON de façon atomique et sécurisée.

    Args:
        file_path: Chemin vers le fichier JSON
        data: Données à écrire (dictionnaire)

    Returns:
        True si écriture réussie, False sinon
    """
    ensure_profiles_dir()

    # Créer un fichier temporaire
    temp_path = file_path.with_suffix(".tmp")

    try:
        # Écrire d'abord dans un fichier temporaire
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Puis remplacer le fichier original (opération atomique)
        if os.name == "nt":  # Windows
            if file_path.exists():
                file_path.unlink()
            os.rename(temp_path, file_path)
        else:  # Unix/Linux/Mac
            os.replace(temp_path, file_path)

        return True

    except Exception as e:
        logger.error(f"Erreur d'écriture JSON {file_path}: {e}")
        # Nettoyer le fichier temporaire si nécessaire
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        return False


def stable_hash(params: Dict[str, Any]) -> str:
    """
    Génère un hash stable pour des paramètres.

    Args:
        params: Dictionnaire de paramètres

    Returns:
        Hash SHA1 hexadécimal des paramètres triés
    """
    # Conversion en format canonique (tri des clés)
    if not params:
        return "empty"

    # Convertir les types numpy en types Python standards
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(i) for i in obj]
        else:
            return obj

    params_clean = convert_numpy(params)

    # Trier les clés et convertir en JSON pour un format canonique
    canonical = json.dumps(params_clean, sort_keys=True, ensure_ascii=True)

    # Calculer le hash SHA1
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def get_gpu_thresholds() -> Dict[str, Any]:
    """
    Récupère les seuils GPU, crée le fichier s'il n'existe pas.

    Returns:
        Dictionnaire contenant les seuils GPU
    """
    data, success = safe_read_json(GPU_THRESHOLDS_FILE)

    if not success or "version" not in data:
        # Initialiser avec les valeurs par défaut
        data = DEFAULT_GPU_THRESHOLDS.copy()
        safe_write_json(GPU_THRESHOLDS_FILE, data)

    return data


def update_gpu_threshold_entry(
    signature: str, cpu_ms: float, gpu_ms: float, n_samples: int = 1
) -> Dict[str, Any]:
    """
    Met à jour une entrée de seuil GPU dans le fichier de profils.

    Args:
        signature: Signature unique de la fonction
        cpu_ms: Temps d'exécution CPU en ms
        gpu_ms: Temps d'exécution GPU en ms
        n_samples: Nombre d'échantillons pour cette mesure

    Returns:
        Le profil mis à jour
    """
    data = get_gpu_thresholds()

    now = datetime.now().isoformat()
    data["updated_at"] = now

    # Mettre à jour l'entrée existante ou en créer une nouvelle
    if signature in data["entries"]:
        entry = data["entries"][signature]

        # Calculer les nouvelles moyennes et variances (EMA)
        alpha = 1 / min(entry["n_samples"] + n_samples, 10)  # Facteur de lissage

        # Mise à jour moyenne mobile exponentielle (EMA)
        entry["cpu_ms_avg"] = (1 - alpha) * entry["cpu_ms_avg"] + alpha * cpu_ms
        entry["gpu_ms_avg"] = (1 - alpha) * entry["gpu_ms_avg"] + alpha * gpu_ms

        # Mise à jour variance (approximation)
        entry["cpu_ms_var"] = (1 - alpha) * entry["cpu_ms_var"] + alpha * (
            cpu_ms - entry["cpu_ms_avg"]
        ) ** 2
        entry["gpu_ms_var"] = (1 - alpha) * entry["gpu_ms_var"] + alpha * (
            gpu_ms - entry["gpu_ms_avg"]
        ) ** 2

        # Mise à jour compteur et date
        entry["n_samples"] += n_samples
        entry["last_seen"] = now

    else:
        # Nouvelle entrée
        entry = {
            "cpu_ms_avg": cpu_ms,
            "cpu_ms_var": 0.0,  # Variance initiale à zéro
            "gpu_ms_avg": gpu_ms,
            "gpu_ms_var": 0.0,  # Variance initiale à zéro
            "n_samples": n_samples,
            "last_seen": now,
            "decision_threshold": data["defaults"]["decision_threshold"],
        }
        data["entries"][signature] = entry

    # Écrire les modifications
    safe_write_json(GPU_THRESHOLDS_FILE, data)
    return data


def get_multigpu_ratios() -> Dict[str, Any]:
    """
    Récupère les ratios multi-GPU, crée le fichier s'il n'existe pas.

    Returns:
        Dictionnaire contenant les ratios multi-GPU
    """
    data, success = safe_read_json(MULTIGPU_RATIOS_FILE)

    if not success or "version" not in data:
        # Initialiser avec les valeurs par défaut
        data = DEFAULT_MULTIGPU_RATIOS.copy()
        safe_write_json(MULTIGPU_RATIOS_FILE, data)

    return data


def update_multigpu_ratios(
    devices: List[Dict[str, Any]],
    ratios: Dict[str, float],
    sample_size: int = 3,
    workload_tag: str = "indicators/batch_default",
    ttl_days: int = 14,
) -> Dict[str, Any]:
    """
    Met à jour les ratios multi-GPU dans le fichier de profils.

    Args:
        devices: Liste des infos de périphériques
        ratios: Dictionnaire des ratios par device_id
        sample_size: Taille de l'échantillon utilisé
        workload_tag: Tag du workload de référence
        ttl_days: Durée de validité en jours

    Returns:
        Le profil mis à jour
    """
    data = {
        "version": 1,
        "updated_at": datetime.now().isoformat(),
        "devices": devices,
        "ratios": ratios,
        "sample_size": sample_size,
        "workload_tag": workload_tag,
        "ttl_days": ttl_days,
    }

    # Écrire les nouvelles valeurs
    safe_write_json(MULTIGPU_RATIOS_FILE, data)
    return data


def is_profile_valid(profile: Dict[str, Any], ttl_days: int = None) -> bool:
    """
    Vérifie si un profil est valide selon sa date.

    Args:
        profile: Dictionnaire du profil
        ttl_days: Durée de validité en jours (si None, utilise la valeur du profil)

    Returns:
        True si le profil est valide, False sinon
    """
    if not profile or "updated_at" not in profile:
        return False

    try:
        # Utiliser le TTL du profil si non spécifié
        if ttl_days is None:
            ttl_days = profile.get("ttl_days", 14)

        # Convertir la date ISO en datetime
        updated_at = datetime.fromisoformat(profile["updated_at"])

        # Calculer l'âge du profil
        age = datetime.now() - updated_at

        # Vérifier si le profil est plus récent que le TTL
        return age < timedelta(days=ttl_days)
    except:
        return False
