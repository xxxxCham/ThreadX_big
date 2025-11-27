"""
Test basique de CodeWriter agent.

Vérifie:
- Initialisation correcte
- Construction du contexte
- Extraction code Python
- Sauvegarde fichier
"""

import sys
from pathlib import Path

# Ajouter src/ au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.llm.agents.codewriter import CodeWriter


def test_codewriter_init():
    """Test initialisation de CodeWriter."""
    print("\n" + "=" * 60)
    print("TEST 1: Initialisation CodeWriter")
    print("=" * 60)

    codewriter = CodeWriter(debug=True)

    assert codewriter.name == "CodeWriter"
    assert codewriter.model == "deepseek-r1:32b"
    assert codewriter.timeout == 120.0
    assert codewriter.output_dir.exists()

    print("✅ Initialisation OK")
    print(f"   - Name: {codewriter.name}")
    print(f"   - Model: {codewriter.model}")
    print(f"   - Output dir: {codewriter.output_dir}")


def test_extract_python_code():
    """Test extraction code Python depuis markdown."""
    print("\n" + "=" * 60)
    print("TEST 2: Extraction Code Python")
    print("=" * 60)

    codewriter = CodeWriter()

    # Test 1: Code avec bloc markdown
    code_markdown = """
Voici le code:

```python
def hello():
    print("Hello World")

class MyStrategy:
    pass
```

Fin du code.
"""

    extracted = codewriter._extract_python_code(code_markdown)

    assert "def hello():" in extracted
    assert "class MyStrategy:" in extracted
    assert "```" not in extracted  # Pas de markdown dans résultat

    print("✅ Extraction depuis markdown OK")
    print(f"   Longueur: {len(extracted)} chars")

    # Test 2: Code brut
    code_raw = """
def hello():
    print("Hello World")
"""

    extracted_raw = codewriter._extract_python_code(code_raw)

    assert "def hello():" in extracted_raw

    print("✅ Extraction code brut OK")


def test_save_code():
    """Test sauvegarde code dans experimental/."""
    print("\n" + "=" * 60)
    print("TEST 3: Sauvegarde Code")
    print("=" * 60)

    codewriter = CodeWriter()

    test_code = '''
"""Test strategy."""

class TestStrategy:
    def run(self, data, indicators, params):
        return {"trades": [], "total_trades": 0}
'''

    filepath = codewriter._save_code(
        filename="test_strategy_v1.py",
        code=test_code,
        metadata={
            "task": "test",
            "base_strategy": "Bollinger_Dual",
            "class_name": "TestStrategyV1",
            "explanation": "Test generation",
        }
    )

    assert filepath.exists()
    assert filepath.name == "test_strategy_v1.py"

    # Vérifier contenu
    content = filepath.read_text(encoding="utf-8")

    assert "AI-Generated Strategy" in content
    assert "TestStrategyV1" in content
    assert "class TestStrategy:" in content

    print("✅ Sauvegarde OK")
    print(f"   Filepath: {filepath}")
    print(f"   Size: {filepath.stat().st_size} bytes")

    # Cleanup
    filepath.unlink()
    print("   (fichier test supprimé)")


def test_build_context():
    """Test construction contexte pour prompt."""
    print("\n" + "=" * 60)
    print("TEST 4: Construction Contexte")
    print("=" * 60)

    codewriter = CodeWriter()

    # Mock data
    analysis = {
        "analysis": {
            "patterns": [
                "Sharpe augmente avec window > 25",
                "Drawdown minimal avec std = 2.5",
            ],
            "recommendations": [
                "Augmenter période Bollinger",
                "Ajouter filtre volume",
            ]
        }
    }

    proposals = {
        "proposals": [
            {
                "name": "Conservative",
                "rationale": "Réduire risque",
                "params": {"bb_period": 30, "bb_std": 2.5}
            },
            {
                "name": "Aggressive",
                "rationale": "Maximiser Sharpe",
                "params": {"bb_period": 20, "bb_std": 3.0}
            }
        ]
    }

    failed_metrics = {
        "current_sharpe": 0.5,
        "target_sharpe": 0.8,
    }

    ideas = [
        "Ajouter trailing stop-loss",
        "Implémenter position sizing adaptatif",
    ]

    context = codewriter._build_context(
        task="improve_sharpe",
        base_strategy="Bollinger_Dual",
        analysis=analysis,
        proposals=proposals,
        failed_metrics=failed_metrics,
        ideas=ideas,
    )

    # Vérifications
    assert "current_sharpe" in context
    assert "Sharpe augmente avec window > 25" in context
    assert "Conservative" in context
    assert "Aggressive" in context
    assert "trailing stop-loss" in context

    print("✅ Construction contexte OK")
    print(f"   Longueur: {len(context)} chars")
    print("\nPrévisualisation:")
    print("-" * 60)
    print(context[:500] + "...")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 TESTS CODEWRITER AGENT")
    print("=" * 60)

    try:
        test_codewriter_init()
        test_extract_python_code()
        test_save_code()
        test_build_context()

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
