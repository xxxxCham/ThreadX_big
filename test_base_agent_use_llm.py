"""
Test du paramètre use_llm dans BaseAgent.

Vérifie:
- Agent avec use_llm=True nécessite un modèle
- Agent avec use_llm=False ne peut pas appeler LLM
- LLMNotConfiguredError levée correctement
"""

import sys
from pathlib import Path

# Ajouter src/ au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.llm.agents.critic import Critic
from threadx.llm.client import LLMNotConfiguredError


def test_use_llm_false_no_model_required():
    """Test: Agent avec use_llm=False n'a pas besoin de modèle."""
    print("\n" + "=" * 60)
    print("TEST 1: use_llm=False - Pas de modèle requis")
    print("=" * 60)

    # Critic V1 utilise use_llm=False par défaut
    critic = Critic(debug=True)

    assert critic.use_llm is False
    assert critic.model is None
    assert critic.client is None

    print("✅ Agent créé sans modèle (use_llm=False)")
    print(f"   - Name: {critic.name}")
    print(f"   - Model: {critic.model}")
    print(f"   - Client: {critic.client}")
    print(f"   - use_llm: {critic.use_llm}")


def test_use_llm_true_model_required():
    """Test: Agent avec use_llm=True nécessite un modèle."""
    print("\n" + "=" * 60)
    print("TEST 2: use_llm=True - Modèle requis")
    print("=" * 60)

    from threadx.llm.agents.base_agent import BaseAgent

    # Tentative de créer agent avec use_llm=True mais sans modèle
    try:
        class TestAgent(BaseAgent):
            def analyze(self, *args, **kwargs):
                return {}

        agent = TestAgent(name="Test", model=None, use_llm=True)
        print("❌ ÉCHEC: ValueError devrait être levée")
        assert False, "ValueError attendue"

    except ValueError as e:
        print(f"✅ ValueError levée correctement: {str(e)[:80]}...")
        assert "model requis si use_llm=True" in str(e)


def test_llm_call_blocked_when_use_llm_false():
    """Test: Appel LLM bloqué si use_llm=False."""
    print("\n" + "=" * 60)
    print("TEST 3: Appel LLM bloqué si use_llm=False")
    print("=" * 60)

    from threadx.llm.agents.base_agent import BaseAgent

    class TestAgentNoLLM(BaseAgent):
        def analyze(self, *args, **kwargs):
            return {}

        def try_call_llm(self):
            """Méthode test qui tente d'appeler LLM."""
            return self._call_llm("Test prompt")

    agent = TestAgentNoLLM(name="TestNoLLM", model=None, use_llm=False)

    try:
        agent.try_call_llm()
        print("❌ ÉCHEC: LLMNotConfiguredError devrait être levée")
        assert False, "LLMNotConfiguredError attendue"

    except LLMNotConfiguredError as e:
        print(f"✅ LLMNotConfiguredError levée correctement")
        print(f"   Message: {str(e)[:80]}...")
        assert "configuré sans LLM" in str(e)


def test_llm_structured_call_blocked_when_use_llm_false():
    """Test: Appel LLM structuré bloqué si use_llm=False."""
    print("\n" + "=" * 60)
    print("TEST 4: Appel LLM structuré bloqué si use_llm=False")
    print("=" * 60)

    from threadx.llm.agents.base_agent import BaseAgent

    class TestAgentNoLLM(BaseAgent):
        def analyze(self, *args, **kwargs):
            return {}

        def try_call_llm_structured(self):
            """Méthode test qui tente d'appeler LLM structuré."""
            return self._call_llm_structured(
                "Test prompt", expected_schema={"key": str}
            )

    agent = TestAgentNoLLM(name="TestNoLLM", model=None, use_llm=False)

    try:
        agent.try_call_llm_structured()
        print("❌ ÉCHEC: LLMNotConfiguredError devrait être levée")
        assert False, "LLMNotConfiguredError attendue"

    except LLMNotConfiguredError as e:
        print(f"✅ LLMNotConfiguredError levée correctement")
        print(f"   Message: {str(e)[:80]}...")
        assert "configuré sans LLM" in str(e)


def test_agent_with_llm_still_works():
    """Test: Agent avec use_llm=True fonctionne toujours (rétrocompatibilité)."""
    print("\n" + "=" * 60)
    print("TEST 5: Rétrocompatibilité - Agent avec LLM")
    print("=" * 60)

    from threadx.llm.agents.codewriter import CodeWriter

    # CodeWriter utilise use_llm=True par défaut
    codewriter = CodeWriter(debug=True)

    assert codewriter.use_llm is True
    assert codewriter.model == "deepseek-r1:32b"
    assert codewriter.client is not None

    print("✅ Agent avec LLM créé correctement")
    print(f"   - Name: {codewriter.name}")
    print(f"   - Model: {codewriter.model}")
    print(f"   - Client: {codewriter.client}")
    print(f"   - use_llm: {codewriter.use_llm}")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 TESTS BaseAgent - use_llm Parameter")
    print("=" * 60)

    try:
        test_use_llm_false_no_model_required()
        test_use_llm_true_model_required()
        test_llm_call_blocked_when_use_llm_false()
        test_llm_structured_call_blocked_when_use_llm_false()
        test_agent_with_llm_still_works()

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
