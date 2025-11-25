"""
Ollama Manager - Gestion automatique d'Ollama pour éviter les blocages
========================================================================

Ce module gère :
1. Auto-démarrage d'Ollama avant chaque run
2. Nettoyage du cache LLM entre les runs
3. Déchargement des modèles après utilisation
"""

import subprocess
import time
import platform
import requests
from typing import Tuple
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def ensure_ollama_running() -> Tuple[bool, str]:
    """
    S'assure qu'Ollama est démarré et fonctionnel.
    
    Returns:
        tuple[bool, str]: (succès, message)
    """
    # 1. Vérifier si Ollama répond
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if response.status_code == 200:
            logger.info("✅ Ollama déjà actif")
            return True, "✅ Ollama actif"
    except Exception as e:
        logger.debug(f"Ollama check failed (ignoré, démarrage): {e}")
        pass  # Ollama pas actif, on va le démarrer
    
    # 2. Démarrer Ollama
    logger.info("🚀 Démarrage d'Ollama...")
    try:
        is_windows = platform.system() == "Windows"
        
        if is_windows:
            # Windows : Lancer directement avec Popen et flags de création console
            kwargs = {}
            # CREATE_NEW_CONSOLE = 0x00000010
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
            
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **kwargs
            )
        else:
            # Linux/Mac
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        # 3. Attendre qu'Ollama soit prêt (max 10s)
        for i in range(10):
            time.sleep(1)
            try:
                response = requests.get("http://127.0.0.1:11434/api/tags", timeout=1)
                if response.status_code == 200:
                    logger.info(f"✅ Ollama démarré avec succès (après {i+1}s)")
                    return True, f"✅ Ollama démarré ({i+1}s)"
            except Exception as e:
                logger.debug(f"Ollama startup check attempt {i+1}/10 failed (retry): {e}")
                continue
        
        return False, "⏱️ Timeout - Ollama n'a pas démarré en 10s"
        
    except FileNotFoundError:
        return False, "❌ Ollama non trouvé (vérifiez l'installation)"
    except Exception as e:
        return False, f"❌ Erreur: {str(e)}"


def unload_model(model_name: str) -> bool:
    """
    Décharge un modèle Ollama de la mémoire GPU/RAM.
    
    Args:
        model_name: Nom du modèle (ex: "deepseek-r1:32b")
    
    Returns:
        bool: True si succès
    """
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": model_name,
                "keep_alive": 0,  # 0 = décharger immédiatement
                "prompt": "",
            },
            timeout=5
        )
        success = response.status_code == 200
        if success:
            logger.info(f"💾 Modèle {model_name} déchargé de la mémoire")
        return success
    except Exception as e:
        logger.warning(f"⚠️ Impossible de décharger {model_name}: {e}")
        return False


def cleanup_all_models() -> int:
    """
    Décharge TOUS les modèles Ollama de la mémoire.
    
    Returns:
        int: Nombre de modèles déchargés
    """
    try:
        # Lister les modèles chargés
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code != 200:
            return 0
        
        models = response.json().get("models", [])
        count = 0
        
        for model in models:
            model_name = model.get("name", "")
            if model_name and unload_model(model_name):
                count += 1
        
        if count > 0:
            logger.info(f"🧹 {count} modèle(s) déchargé(s) de la mémoire")
        
        return count
        
    except Exception as e:
        logger.warning(f"⚠️ Erreur cleanup_all_models: {e}")
        return 0


def prepare_for_llm_run() -> Tuple[bool, str]:
    """
    Prépare l'environnement pour un run LLM.
    
    Actions:
    1. S'assure qu'Ollama est actif
    2. Nettoie les modèles précédents en mémoire
    
    Returns:
        tuple[bool, str]: (succès, message détaillé)
    """
    messages = []
    
    # 1. Nettoyer les modèles précédents
    cleaned = cleanup_all_models()
    if cleaned > 0:
        messages.append(f"🧹 {cleaned} modèle(s) déchargé(s)")
    
    # 2. S'assurer qu'Ollama est actif
    success, msg = ensure_ollama_running()
    messages.append(msg)
    
    if success:
        time.sleep(1)  # Petite pause pour stabilité
        return True, " | ".join(messages)
    else:
        return False, " | ".join(messages)


