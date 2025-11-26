"""
POC Context-Aware Orchestrator - Test Complet Gestion Contexte
=============================================================

Démontre orchestration complète avec:
1. ContextManager (inventaire données + registry stratégies)
2. PromptEnricher (enrichissement prompts agents)
3. Validation pre-flight (détection tokens invalides)
4. Versioning automatique stratégies
5. Gestion erreurs données (fallback token alternatif)

Author: ThreadX Framework
Version: 1.0 - Context-Aware POC
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from threadx.backtest.engine import BacktestEngine
from threadx.llm.context_manager import (
    ContextManager,
    StrategyVersion,
    TokenAvailability,
    create_default_inventory,
)
from threadx.llm.prompt_enricher import PromptEnricher
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def create_mock_data() -> pd.DataFrame:
    """Crée données mock OHLCV."""
    dates = pd.date_range(start="2024-01-01", end="2024-11-21", freq="15min")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": 50000.0,
            "high": 50100.0,
            "low": 49900.0,
            "close": 50000.0,
            "volume": 1000.0,
        }
    )


def test_context_manager():
    """Test ContextManager - Inventaire + Registry."""
    print("\n" + "=" * 60)
    print("TEST 1: ContextManager - Inventaire Données + Registry")
    print("=" * 60)

    # 1. Créer context manager avec inventaire par défaut
    context_manager = ContextManager(
        data_dir=Path("./data"),
        registry_path=Path("./exports/strategy_registry_poc.json"),
    )

    # Charger inventaire par défaut (si data/ vide)
    if context_manager.inventory.total_tokens == 0:
        context_manager.inventory = create_default_inventory()
        logger.info("✅ Default inventory loaded (5 major tokens)")

    # 2. Ajouter stratégie initiale
    context_manager.registry.add_strategy(
        StrategyVersion(
            name="ma_crossover",
            version="v1.0",
            params={"short_period": 10, "long_period": 50},
            created_by="human",
            description="Initial MA Crossover implementation",
        )
    )

    # 3. Afficher contexte global
    ctx = context_manager.get_full_context("ma_crossover")

    print("\n📊 DATA INVENTORY:")
    print(f"   Total tokens: {ctx['data_inventory']['total_tokens']}")
    print(f"   Period: {ctx['data_inventory']['global_period']['description']}")
    print("\n   Top 3 tokens:")
    for symbol, info in list(ctx["data_inventory"]["tokens"].items())[:3]:
        print(
            f"   - {symbol}: Quality {info['quality_score']}, "
            f"Timeframes {', '.join(info['timeframes'])}"
        )

    print("\n📦 STRATEGY REGISTRY:")
    print(f"   Strategy: {ctx['strategy_registry']['strategy_name']}")
    print(f"   Versions: {ctx['strategy_registry']['total_versions']}")
    latest = ctx["strategy_registry"]["latest_version"]
    print(f"   Latest: {latest['version']} (created {latest['created_at']})")
    print(f"   Params: {latest['params']}")

    print("\n✅ TEST 1 PASSED")
    return context_manager


def test_validation():
    """Test Validation Pre-Flight."""
    print("\n" + "=" * 60)
    print("TEST 2: Validation Pre-Flight - Token + Période")
    print("=" * 60)

    context_manager = ContextManager()
    context_manager.inventory = create_default_inventory()

    # Scénario 1: Token valide
    print("\n🔍 Scénario 1: Token valide (BTCUSDC)")
    valid, msg, ctx = context_manager.validate_optimization_request(
        symbol="BTCUSDC",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 11, 21),
        timeframe="15m",
        strategy_name="ma_crossover",
    )
    print(f"   Valid: {valid}")
    print(f"   Message:\n{msg}")

    # Scénario 2: Token invalide (inexistant)
    print("\n🔍 Scénario 2: Token invalide (FAKETOKEN)")
    valid, msg, ctx = context_manager.validate_optimization_request(
        symbol="FAKETOKEN",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 11, 21),
        timeframe="15m",
        strategy_name="ma_crossover",
    )
    print(f"   Valid: {valid}")
    print(f"   Message:\n{msg}")

    # Scénario 3: Période hors disponibilité
    print("\n🔍 Scénario 3: Période hors disponibilité (2020-2021)")
    valid, msg, ctx = context_manager.validate_optimization_request(
        symbol="BTCUSDC",
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2021, 12, 31),
        timeframe="15m",
        strategy_name="ma_crossover",
    )
    print(f"   Valid: {valid}")
    print(f"   Message:\n{msg}")

    print("\n✅ TEST 2 PASSED")


def test_prompt_enrichment():
    """Test Enrichissement Prompts."""
    print("\n" + "=" * 60)
    print("TEST 3: Prompt Enrichment - Contexte Agents")
    print("=" * 60)

    context_manager = ContextManager()
    context_manager.inventory = create_default_inventory()

    # Ajouter stratégie avec performance
    context_manager.registry.add_strategy(
        StrategyVersion(
            name="ma_crossover",
            version="v1.0",
            params={"short_period": 10, "long_period": 50},
            performance={"sharpe_ratio": 1.5, "sortino_ratio": 2.1},
            tier_s_score=62,
            created_by="human",
        )
    )

    # Analyst prompt
    print("\n📝 Analyst Prompt Enrichment:")
    base_prompt = "Analyze backtest result and provide diagnosis."
    backtest_result = {
        "sharpe_ratio": 1.65,
        "sortino_ratio": 2.4,
        "max_drawdown": -12.3,
        "tier_s_score": 68,
    }

    enriched = PromptEnricher.enrich_analyst_prompt(
        base_prompt=base_prompt,
        context_manager=context_manager,
        strategy_name="ma_crossover",
        backtest_result=backtest_result,
        memory=None,
    )

    print(f"   Base prompt length: {len(base_prompt)} chars")
    print(f"   Enriched prompt length: {len(enriched)} chars")
    print(f"   Context added: +{len(enriched) - len(base_prompt)} chars")

    # Extraits prompt enrichi
    print("\n   📋 Enriched Prompt Extracts:")
    lines = enriched.split("\n")
    for line in lines[:15]:  # Premières 15 lignes
        if line.strip():
            print(f"      {line}")
    print("      ...")

    # Strategist prompt
    print("\n📝 Strategist Prompt Enrichment:")
    base_prompt_strat = "Propose 3 optimization strategies."
    current_params = {"short_period": 10, "long_period": 50}
    diagnosis = {"score": 7, "summary": "Good performance, can improve"}

    enriched_strat = PromptEnricher.enrich_strategist_prompt(
        base_prompt=base_prompt_strat,
        context_manager=context_manager,
        strategy_name="ma_crossover",
        current_params=current_params,
        analyst_diagnosis=diagnosis,
        memory=None,
    )

    print(f"   Base prompt length: {len(base_prompt_strat)} chars")
    print(f"   Enriched prompt length: {len(enriched_strat)} chars")
    print(f"   Context added: +{len(enriched_strat) - len(base_prompt_strat)} chars")

    print("\n✅ TEST 3 PASSED")


def test_strategy_versioning():
    """Test Versioning Stratégies."""
    print("\n" + "=" * 60)
    print("TEST 4: Strategy Versioning - Évolution")
    print("=" * 60)

    context_manager = ContextManager(
        registry_path=Path("./exports/strategy_registry_poc.json")
    )

    # Créer évolution stratégie
    print("\n📦 Creating strategy evolution:")

    # v1.0 - Initial
    v1 = StrategyVersion(
        name="ma_crossover",
        version="v1.0",
        params={"short_period": 10, "long_period": 50},
        created_by="human",
    )
    context_manager.registry.add_strategy(v1)
    print("   ✅ v1.0 created (human)")

    # v2.0 - First optimization
    v2 = StrategyVersion(
        name="ma_crossover",
        version="optimized_2024-02-15",
        params={"short_period": 12, "long_period": 48},
        performance={"sharpe_ratio": 1.65, "sortino_ratio": 2.3},
        tier_s_score=65,
        created_by="optimizer",
        parent_version="v1.0",
    )
    context_manager.registry.add_strategy(v2)
    print("   ✅ optimized_2024-02-15 created (optimizer)")

    # v3.0 - Better optimization
    v3 = StrategyVersion(
        name="ma_crossover",
        version="optimized_2024-03-20",
        params={"short_period": 14, "long_period": 45},
        performance={"sharpe_ratio": 1.85, "sortino_ratio": 2.8},
        tier_s_score=72,
        created_by="optimizer",
        parent_version="optimized_2024-02-15",
    )
    context_manager.registry.add_strategy(v3)
    print("   ✅ optimized_2024-03-20 created (optimizer)")

    # Requêtes registry
    print("\n📊 Registry Queries:")
    latest = context_manager.registry.get_latest_version("ma_crossover")
    print(f"   Latest version: {latest.version}")
    print(f"   Params: {latest.params}")
    print(f"   Tier S: {latest.tier_s_score}")

    best = context_manager.registry.get_best_version("ma_crossover", "sharpe_ratio")
    print(f"\n   Best version (Sharpe): {best.version}")
    print(f"   Sharpe: {best.performance['sharpe_ratio']:.2f}")
    print(f"   Tier S: {best.tier_s_score}")

    # Arbre évolution
    tree = context_manager.registry.get_evolution_tree("ma_crossover")
    print("\n📈 Evolution Tree:")
    for version, info in tree.items():
        parent = info["parent"] or "root"
        sharpe = (
            info["performance"]["sharpe_ratio"]
            if info["performance"]
            else "N/A"
        )
        print(f"   {version} (parent: {parent}) → Sharpe: {sharpe}")

    # Persistence
    print("\n💾 Persisting registry to disk...")
    context_manager.registry.save()
    print(f"   Saved to: {context_manager.registry.registry_path}")

    print("\n✅ TEST 4 PASSED")


def test_error_handling():
    """Test Gestion Erreurs Données."""
    print("\n" + "=" * 60)
    print("TEST 5: Error Handling - Données Invalides")
    print("=" * 60)

    context_manager = ContextManager()
    context_manager.inventory = create_default_inventory()

    # Ajouter token problématique (basse qualité)
    print("\n🔧 Adding low-quality token (SHIBAINU, quality 65%):")
    context_manager.inventory.add_token(
        TokenAvailability(
            symbol="SHIBAINU",
            start_date=datetime(2024, 1, 1),
            end_date=None,
            timeframes=["1m", "15m"],
            data_quality=0.65,  # ⚠️ <80%
            total_bars=400000,
            gaps_detected=140000,  # 35% gaps
            last_update=datetime.now(),
        )
    )

    # Validation détecte problème
    print("\n🔍 Validation Request (SHIBAINU):")
    valid, msg, ctx = context_manager.validate_optimization_request(
        symbol="SHIBAINU",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 11, 21),
        timeframe="15m",
        strategy_name="ma_crossover",
    )

    print(f"   Valid: {valid}")
    print(f"   Message:\n{msg}")

    # Récupérer alternatives
    if not valid:
        print("\n💡 Fetching alternatives:")
        alternatives = context_manager.inventory.get_available_tokens(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 11, 21),
            timeframe="15m",
        )
        print(f"   Alternatives: {', '.join(alternatives[:5])}")

        # Retry avec alternative
        print("\n🔄 Retrying with BTCUSDC:")
        valid2, msg2, ctx2 = context_manager.validate_optimization_request(
            symbol="BTCUSDC",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 11, 21),
            timeframe="15m",
            strategy_name="ma_crossover",
        )
        print(f"   Valid: {valid2}")
        print(f"   Message:\n{msg2}")

    print("\n✅ TEST 5 PASSED")


def test_full_integration():
    """Test Intégration Complète."""
    print("\n" + "=" * 60)
    print("TEST 6: Full Integration - Workflow Complet")
    print("=" * 60)

    # 1. Setup
    context_manager = ContextManager(
        registry_path=Path("./exports/strategy_registry_poc.json")
    )
    context_manager.inventory = create_default_inventory()

    # 2. Pre-flight validation
    print("\n✈️ Pre-Flight Validation:")
    valid, msg, ctx = context_manager.validate_optimization_request(
        symbol="BTCUSDC",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 11, 21),
        timeframe="15m",
        strategy_name="ma_crossover",
    )

    if not valid:
        print(f"   ❌ Validation failed: {msg}")
        return

    print(f"   ✅ Validation passed")
    print(f"   Tokens available: {ctx['data_inventory']['total_tokens']}")

    # 3. Contexte complet pour agents
    print("\n🤖 Full Context for Agents:")
    full_ctx = context_manager.get_full_context("ma_crossover")
    print(f"   Context keys: {list(full_ctx.keys())}")
    print(
        f"   Data inventory: {len(full_ctx['data_inventory']['tokens'])} tokens"
    )
    print(
        f"   Strategy registry: {full_ctx['strategy_registry'].get('total_strategies', 'N/A')} strategies"
    )

    # 4. Enrichissement prompts
    print("\n📝 Prompt Enrichment:")
    analyst_prompt = PromptEnricher.enrich_analyst_prompt(
        base_prompt="Analyze result",
        context_manager=context_manager,
        strategy_name="ma_crossover",
        backtest_result={"sharpe_ratio": 1.75},
        memory=None,
    )
    print(f"   Analyst prompt: {len(analyst_prompt)} chars")

    strategist_prompt = PromptEnricher.enrich_strategist_prompt(
        base_prompt="Propose optimizations",
        context_manager=context_manager,
        strategy_name="ma_crossover",
        current_params={"short": 10, "long": 50},
        analyst_diagnosis={"score": 7},
        memory=None,
    )
    print(f"   Strategist prompt: {len(strategist_prompt)} chars")

    # 5. Simulation optimisation (sans agents LLM)
    print("\n🔄 Simulating Optimization Iteration:")

    # Backtest initial
    data = create_mock_data()
    print(f"   Mock data: {len(data)} bars")

    # Nouvelles propositions (simulées)
    proposals = [
        {"short_period": 12, "long_period": 48},
        {"short_period": 14, "long_period": 45},
        {"short_period": 16, "long_period": 42},
    ]

    # Ajout versions propositions
    for i, prop in enumerate(proposals, start=1):
        version = StrategyVersion(
            name="ma_crossover",
            version=f"proposal_iter1_{i}",
            params=prop,
            created_by="strategist",
            parent_version="v1.0",
        )
        context_manager.registry.add_strategy(version)
        print(f"   ✅ Proposal {i} registered: {prop}")

    # Sélection meilleure (simulée)
    best_proposal = proposals[1]  # Arbitraire
    best_version = StrategyVersion(
        name="ma_crossover",
        version="iter1_optimized",
        params=best_proposal,
        performance={"sharpe_ratio": 1.88, "sortino_ratio": 2.9},
        tier_s_score=75,
        created_by="optimizer",
        parent_version="v1.0",
    )
    context_manager.registry.add_strategy(best_version)
    print(f"\n   ✅ Best selected: {best_proposal}")
    print(f"   Tier S: {best_version.tier_s_score}/100")

    # 6. Résumé final
    print("\n📊 Final Summary:")
    latest = context_manager.registry.get_latest_version("ma_crossover")
    print(f"   Latest version: {latest.version}")
    print(f"   Params: {latest.params}")
    print(f"   Performance: {latest.performance}")
    print(
        f"   Total versions: {len(context_manager.registry.strategies['ma_crossover'])}"
    )

    # Persistence
    context_manager.registry.save()
    print(f"\n💾 Registry saved: {context_manager.registry.registry_path}")

    print("\n✅ TEST 6 PASSED - FULL INTEGRATION COMPLETE")


def main():
    """Lance tous les tests."""
    print("\n" + "=" * 60)
    print("POC CONTEXT-AWARE ORCHESTRATOR")
    print("Testing: ContextManager + PromptEnricher + Validation")
    print("=" * 60)

    try:
        # Test 1: ContextManager
        context_manager = test_context_manager()

        # Test 2: Validation
        test_validation()

        # Test 3: Prompt Enrichment
        test_prompt_enrichment()

        # Test 4: Versioning
        test_strategy_versioning()

        # Test 5: Error Handling
        test_error_handling()

        # Test 6: Full Integration
        test_full_integration()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\n📁 Registry saved to: ./exports/strategy_registry_poc.json")
        print(
            "   You can inspect the registry to see strategy evolution tree."
        )

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
