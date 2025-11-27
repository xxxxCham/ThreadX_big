"""
Test démontrant le bug de cache de module dans Critic._validate_syntax().

Scénario:
1. Créer stratégie V1 avec bug
2. Valider avec Critic → devrait échouer
3. Corriger le bug (V2)
4. Revalider avec Critic → BUG: utilise toujours V1 en cache !
"""

import sys
from pathlib import Path

# Ajouter src/ au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.llm.agents.critic import Critic


def test_module_cache_bug():
    """Démontre que le cache module cause des faux négatifs."""
    print("\n" + "=" * 60)
    print("🐛 TEST BUG: Cache Module dans _validate_syntax()")
    print("=" * 60)

    test_file = Path("src/threadx/strategy/experimental/ai_cache_test.py")

    # Étape 1: Créer stratégie V1 avec syntaxe INVALIDE
    print("\n📝 Étape 1: Créer stratégie V1 (syntaxe INVALIDE)")

    code_v1_invalid = '''"""AI Cache Test V1 - SYNTAXE INVALIDE."""

from dataclasses import dataclass

@dataclass
class AICacheTestParams:
    period: int = 20

class AICacheTestStrategy
    # ❌ BUG SYNTAXE: manque ':'
    def run(self, data, indicators, params):
        return {"trades": []}
'''

    test_file.write_text(code_v1_invalid, encoding="utf-8")
    print(f"   Fichier créé: {test_file.name}")

    # Étape 2: Valider V1 → devrait ÉCHOUER
    print("\n🧪 Étape 2: Valider V1 avec Critic")

    critic = Critic()
    result_v1 = critic._validate_syntax(test_file)

    print(f"   ✅ V1 validation: passed={result_v1['passed']}")

    if result_v1["passed"]:
        print("   ❌ ERREUR: V1 devrait échouer (syntaxe invalide)")
    else:
        print(f"   ✅ V1 correctement rejetée: {result_v1['error'][:60]}...")

    # Étape 3: Corriger le bug → V2 VALIDE
    print("\n📝 Étape 3: Corriger syntaxe → V2 (VALIDE)")

    code_v2_valid = '''"""AI Cache Test V2 - SYNTAXE VALIDE."""

from dataclasses import dataclass
from typing import Any
import pandas as pd

@dataclass
class AICacheTestParams:
    period: int = 20

class AICacheTestStrategy:
    """Fixed strategy."""

    def run(self, data: pd.DataFrame, indicators: dict[str, Any], params):
        from threadx.strategy.model import RunStatsDict
        return RunStatsDict(trades=[], total_trades=0)
'''

    test_file.write_text(code_v2_valid, encoding="utf-8")
    print(f"   Fichier mis à jour: {test_file.name}")

    # Étape 4: Revalider V2 → BUG: utilise cache V1 !
    print("\n🧪 Étape 4: Revalider V2 avec Critic")

    result_v2 = critic._validate_syntax(test_file)

    print(f"   Result: passed={result_v2['passed']}")

    # ❌ BUG: Si result_v2["passed"] == False, c'est parce que sys.modules
    # contient toujours l'ancien module V1 avec syntaxe invalide !

    if result_v2["passed"]:
        print("   ✅ V2 correctement validée")
        print("\n✅ PAS DE BUG: Le module a été rechargé correctement")
    else:
        print(f"   ❌ V2 rejetée: {result_v2['error'][:60]}...")
        print("\n🐛 BUG DÉTECTÉ: Cache module cause faux négatif !")
        print("   → sys.modules contient toujours V1 invalide")
        print("   → _validate_syntax() ne recharge pas le module")

    # Cleanup
    test_file.unlink()

    # Vérifier sys.modules
    module_name = "threadx.strategy.experimental.ai_cache_test"
    if module_name in sys.modules:
        print(f"\n⚠️  Module toujours dans sys.modules: {module_name}")
        print("   → Pollution de l'environnement Python")

    return not result_v2["passed"]  # True si bug détecté


def main():
    """Run bug test."""
    print("\n" + "=" * 60)
    print("🧪 TEST BUG CRITIC - Module Cache")
    print("=" * 60)

    bug_detected = test_module_cache_bug()

    print("\n" + "=" * 60)
    if bug_detected:
        print("🐛 BUG CONFIRMÉ: Cache module cause problème")
        print("\nFIX SUGGÉRÉ:")
        print("  1. Reload module si déjà dans sys.modules")
        print("  2. Ou utiliser importlib.reload()")
        print("  3. Ou supprimer de sys.modules après validation")
    else:
        print("✅ PAS DE BUG: Module rechargé correctement")
    print("=" * 60)


if __name__ == "__main__":
    main()
