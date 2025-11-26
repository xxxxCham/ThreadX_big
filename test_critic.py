"""
Test basique de Critic agent.

Vérifie:
- Initialisation correcte
- Validation syntaxe (py_compile + import)
- Critères quantitatifs
- Décision de promotion
"""

import sys
from pathlib import Path

# Ajouter src/ au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.llm.agents.critic import Critic, ValidationCriteria


def test_critic_init():
    """Test initialisation de Critic."""
    print("\n" + "=" * 60)
    print("TEST 1: Initialisation Critic")
    print("=" * 60)

    criteria = ValidationCriteria(
        min_sharpe=0.5,
        max_drawdown_pct=-30.0,
        min_trades=10,
        min_win_rate_pct=35.0,
    )

    critic = Critic(criteria=criteria, debug=True)

    assert critic.name == "Critic"
    assert critic.model is None  # Critic V1 ne nécessite pas de modèle
    assert critic.use_llm is False  # Désactivé explicitement
    assert critic.client is None  # Pas de LLMClient
    assert critic.timeout == 60.0
    assert critic.criteria.min_sharpe == 0.5
    assert critic.experimental_dir.exists()

    print("✅ Initialisation OK")
    print(f"   - Name: {critic.name}")
    print(f"   - Model: {critic.model} (use_llm={critic.use_llm})")
    print(f"   - Critères: Sharpe≥{criteria.min_sharpe}, DD≥{criteria.max_drawdown_pct}%")


def test_validate_syntax_valid():
    """Test validation syntaxe avec fichier valide."""
    print("\n" + "=" * 60)
    print("TEST 2: Validation Syntaxe (Fichier Valide)")
    print("=" * 60)

    # Créer fichier de test temporaire
    test_file = Path("src/threadx/strategy/experimental/test_valid_strategy.py")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    test_code = '''"""Test strategy."""

from dataclasses import dataclass
from typing import Any
import pandas as pd

@dataclass
class TestValidStrategyParams:
    period: int = 20

class TestValidStrategy:
    """Test strategy."""

    def run(self, data: pd.DataFrame, indicators: dict[str, Any], params):
        from threadx.strategy.model import RunStatsDict
        return RunStatsDict(trades=[], total_trades=0)
'''

    test_file.write_text(test_code, encoding="utf-8")

    try:
        critic = Critic()
        result = critic._validate_syntax(test_file)

        assert result["passed"] is True
        assert result["error"] is None
        assert result["strategy_class"] is not None
        assert result["params_class"] is not None
        assert result["strategy_class"].__name__ == "TestValidStrategy"

        print("✅ Validation syntaxe OK")
        print(f"   - Module: {result['module_name']}")
        print(f"   - Strategy: {result['strategy_class'].__name__}")
        print(f"   - Params: {result['params_class'].__name__}")

    finally:
        # Cleanup
        test_file.unlink()


def test_validate_syntax_invalid():
    """Test validation syntaxe avec fichier invalide."""
    print("\n" + "=" * 60)
    print("TEST 3: Validation Syntaxe (Fichier Invalide)")
    print("=" * 60)

    test_file = Path("src/threadx/strategy/experimental/test_invalid_strategy.py")

    # Code avec erreur de syntaxe
    invalid_code = '''"""Test strategy with syntax error."""

class TestInvalidStrategy
    # Syntaxe invalide: manque ':'
    def run(self):
        pass
'''

    test_file.write_text(invalid_code, encoding="utf-8")

    try:
        critic = Critic()
        result = critic._validate_syntax(test_file)

        assert result["passed"] is False
        assert result["error"] is not None
        assert "SyntaxError" in result["error"]

        print("✅ Détection erreur syntaxe OK")
        print(f"   - Error: {result['error'][:60]}...")

    finally:
        # Cleanup
        test_file.unlink()


def test_check_quantitative_criteria():
    """Test vérification critères quantitatifs."""
    print("\n" + "=" * 60)
    print("TEST 4: Critères Quantitatifs")
    print("=" * 60)

    criteria = ValidationCriteria(
        min_sharpe=0.5,
        max_drawdown_pct=-30.0,
        min_trades=10,
        min_win_rate_pct=35.0,
    )

    critic = Critic(criteria=criteria)

    # Scénario 1: Résultats OK
    good_results = [
        {
            "sharpe": 0.8,
            "max_drawdown_pct": -15.0,
            "total_trades": 25,
            "win_rate_pct": 45.0,
            "error": None,
        },
        {
            "sharpe": 0.6,
            "max_drawdown_pct": -20.0,
            "total_trades": 30,
            "win_rate_pct": 40.0,
            "error": None,
        },
    ]

    result_good = critic._check_quantitative_criteria(good_results)

    assert result_good["passed"] is True
    assert len(result_good["failures"]) == 0
    assert result_good["best_sharpe"] == 0.8

    print("✅ Critères satisfaits OK")
    print(f"   - Best Sharpe: {result_good['best_sharpe']}")
    print(f"   - Worst DD: {result_good['worst_drawdown']}%")

    # Scénario 2: Résultats mauvais
    bad_results = [
        {
            "sharpe": 0.2,  # Trop bas
            "max_drawdown_pct": -45.0,  # Trop profond
            "total_trades": 5,  # Trop peu
            "win_rate_pct": 25.0,  # Trop bas
            "error": None,
        },
    ]

    result_bad = critic._check_quantitative_criteria(bad_results)

    assert result_bad["passed"] is False
    assert len(result_bad["failures"]) > 0

    print("✅ Détection critères non satisfaits OK")
    print(f"   - Failures: {result_bad['failures']}")


def test_full_validation_flow():
    """Test flux complet de validation."""
    print("\n" + "=" * 60)
    print("TEST 5: Flux Complet de Validation")
    print("=" * 60)

    # Créer stratégie valide
    test_file = Path("src/threadx/strategy/experimental/ai_test_v1.py")

    valid_strategy = '''"""AI Test Strategy V1."""

from dataclasses import dataclass
from typing import Any
import pandas as pd

@dataclass
class AITestV1Params:
    period: int = 20
    stop_loss_pct: float = 2.0

class AITestV1Strategy:
    """Test AI strategy."""

    def run(self, data: pd.DataFrame, indicators: dict[str, Any], params):
        from threadx.strategy.model import RunStatsDict
        return RunStatsDict(
            trades=[],
            total_trades=0,
        )
'''

    test_file.write_text(valid_strategy, encoding="utf-8")

    try:
        criteria = ValidationCriteria(
            min_sharpe=0.5,
            max_drawdown_pct=-30.0,
            min_trades=10,
            min_win_rate_pct=35.0,
        )

        critic = Critic(criteria=criteria, debug=True)

        result = critic.run(
            strategy_file="ai_test_v1.py",
            backtest_scenarios=[
                {
                    "symbol": "BTCUSDC",
                    "interval": "15m",
                    "start": "2023-07-01",
                    "end": "2023-12-31",
                    "description": "BTC 15m test",
                }
            ],
        )

        # Avec les données mock, devrait passer
        assert result["status"] in ["approved", "rejected"]
        assert "test_syntax" in result
        assert "test_backtest" in result
        assert "test_quantitative" in result

        print(f"✅ Validation complète OK")
        print(f"   - Status: {result['status']}")
        print(f"   - Recommendation: {result['recommendation']}")

        if result["status"] == "approved":
            print(f"   - Best Sharpe: {result['test_quantitative'].get('best_sharpe', 'N/A')}")
        else:
            print(f"   - Errors: {result.get('errors', [])}")

    finally:
        # Cleanup
        test_file.unlink()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 TESTS CRITIC AGENT")
    print("=" * 60)

    try:
        test_critic_init()
        test_validate_syntax_valid()
        test_validate_syntax_invalid()
        test_check_quantitative_criteria()
        test_full_validation_flow()

        print("\n" + "=" * 60)
        print("✅ TOUS LES TESTS PASSÉS")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST ÉCHOUÉ: {e}")
        raise

    except Exception as e:
        print(f"\n❌ ERREUR INATTENDUE: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
