"""
Installation rapide de Numba pour optimisations GPU
===================================================

Script pour installer Numba et vérifier la configuration CUDA.
"""

import subprocess
import sys


def check_numba_available():
    """Vérifie si Numba est déjà installé."""
    try:
        import numba

        print(f"✅ Numba {numba.__version__} déjà installé")
        return True
    except ImportError:
        print("❌ Numba non installé")
        return False


def check_cuda_available():
    """Vérifie si CUDA est disponible pour Numba."""
    try:
        from numba import cuda

        available = cuda.is_available()

        if available:
            print(f"✅ CUDA disponible pour Numba")
            print(f"   Devices: {len(cuda.gpus)} GPU(s)")

            for i, gpu in enumerate(cuda.gpus):
                print(f"   - GPU {i}: {gpu.name}")
                print(f"     Compute Capability: {gpu.compute_capability}")

            return True
        else:
            print("❌ CUDA non disponible pour Numba")
            return False

    except Exception as e:
        print(f"❌ Erreur vérification CUDA: {e}")
        return False


def install_numba():
    """Installe Numba via pip."""
    print("\n🔧 Installation de Numba...")

    try:
        # Installation via pip
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "numba", "--upgrade"]
        )

        print("✅ Numba installé avec succès")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur installation: {e}")
        return False


def main():
    """Lance l'installation et la vérification."""
    print("=" * 60)
    print(" 🚀 INSTALLATION NUMBA POUR THREADX GPU OPTIMISATIONS")
    print("=" * 60)

    # Vérification pré-installation
    if check_numba_available():
        print("\n✅ Numba déjà installé, vérification CUDA...")
        check_cuda_available()
        return 0

    # Installation
    print("\n📦 Numba non trouvé, installation...")
    if not install_numba():
        print("\n❌ Installation échouée")
        return 1

    # Vérification post-installation
    print("\n🔍 Vérification installation...")
    if not check_numba_available():
        print("\n❌ Numba non détecté après installation")
        return 1

    print("\n🔍 Vérification CUDA...")
    cuda_ok = check_cuda_available()

    if cuda_ok:
        print("\n" + "=" * 60)
        print(" 🎉 INSTALLATION RÉUSSIE - NUMBA CUDA OPÉRATIONNEL")
        print("=" * 60)
        print("\n💡 Prochaines étapes:")
        print("   1. Exécuter: python test_gpu_optimizations.py")
        print("   2. Les kernels Numba seront utilisés automatiquement")
        print("   3. Speedup attendu: 2-5x sur calculs indicateurs")
    else:
        print("\n" + "=" * 60)
        print(" ⚠️  NUMBA INSTALLÉ MAIS CUDA NON DISPONIBLE")
        print("=" * 60)
        print("\n💡 Solutions possibles:")
        print("   1. Vérifier que CUDA Toolkit est installé (11.8+ ou 12.x)")
        print("   2. Vérifier PATH contient bin CUDA")
        print("   3. Redémarrer le terminal/IDE")
        print("   4. ThreadX utilisera CuPy en fallback (toujours performant)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
