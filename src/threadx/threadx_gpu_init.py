"""
ThreadX GPU Configuration - Multi-GPU Detection & Setup
=======================================================

Ce module DOIT être importé en PREMIER dans streamlit_app.py
pour garantir la détection et configuration correcte des GPUs.

Configuration:
- Expose tous les GPUs NVIDIA via CUDA_VISIBLE_DEVICES
- Ordre physique PCI_BUS_ID pour mapping stable
- Diagnostic complet au démarrage
- Suppression warnings PyTorch non pertinents

Usage:
    # En haut de streamlit_app.py, AVANT tout autre import:
    import threadx_gpu_init  # Init GPU automatique
"""

import os
import sys

# ===================================================================
# CONFIGURATION CRITIQUE: GPUs NVIDIA uniquement
# ===================================================================

# 1. Exposer les GPUs NVIDIA dans l'ordre souhaité (ignore AMD Radeon)
#    - Si l'utilisateur a déjà défini CUDA_VISIBLE_DEVICES, on le respecte.
#    - Sinon, on tente de détecter le GPU le plus puissant (VRAM) via nvidia-smi
#      et on place ce GPU en premier (ex: 5080 avant 2060).
def _detect_cuda_order() -> str:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return "0,1"

        # Parse lines: "0, NVIDIA GeForce RTX 5080, 16384 MiB"
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx = parts[0]
            name = parts[1].upper()
            mem_str = parts[2].split()[0]  # "16384"
            try:
                mem_mb = int(mem_str)
            except ValueError:
                mem_mb = 0
            gpus.append((mem_mb, idx, name))

        if not gpus:
            return "0,1"

        # Trier par mémoire décroissante, garder l'ordre des indices en string
        gpus_sorted = sorted(gpus, key=lambda x: (-x[0], x[1]))
        return ",".join(g[1] for g in gpus_sorted)
    except Exception:
        return "0,1"


os.environ['CUDA_VISIBLE_DEVICES'] = _detect_cuda_order()

# 2. Ordre physique des bus PCI (important pour multi-GPU stable)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# 3. Optimisations CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Mode asynchrone (performance)

# ===================================================================
# SUPPRESSION WARNINGS PYTORCH (ThreadX utilise CuPy, pas PyTorch)
# ===================================================================

import warnings

# ThreadX utilise CuPy pour calculs GPU. Les warnings PyTorch sur
# compute capability sont non pertinents et créent de la confusion.

# Pattern large pour capturer tous les warnings PyTorch CUDA
warnings.filterwarnings(
    "ignore",
    message=".*CUDA capability.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*PyTorch.*CUDA.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*not compatible.*PyTorch.*",
    category=UserWarning,
)

# Filtre spécifique pour le module torch.cuda
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torch.cuda",
)

# ===================================================================
# CONFIGURATION DEVICES PAR DÉFAUT
# ===================================================================

# 4. PyTorch : device par défaut (si installé)
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # GPU primaire
except ImportError:
    pass  # PyTorch non requis

# 5. CuPy : device par défaut (obligatoire pour ThreadX)
try:
    import cupy as cp
    cp.cuda.Device(0).use()
except ImportError:
    print("⚠️  CuPy non disponible - Mode CPU uniquement")

# ===================================================================
# DIAGNOSTIC & INITIALISATION MULTI-GPU
# ===================================================================

# Ajout du chemin src/ pour imports ThreadX
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from threadx.gpu.startup_check import (
        check_and_report_gpus,
        initialize_multi_gpu_manager,
    )

    # Diagnostic complet
    gpu_info = check_and_report_gpus()

    # Init MultiGPUManager si 2+ GPUs (sinon lazy loading)
    if gpu_info.get("gpu_count", 0) >= 2:
        initialize_multi_gpu_manager()

except Exception as e:
    print(f"⚠️  Erreur diagnostic GPU: {e}")
    print("   (ThreadX continuera en mode dégradé)")
