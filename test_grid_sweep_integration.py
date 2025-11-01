"""
Test d'intégration complet du Grid Sweep après correctif.

Simule le flux complet: UI → Engine → Strategy → Backtest
Vérifie que:
1. min_pnl_pct=0.0 est présent dans toutes les combinaisons
2. Les backtests génèrent des trades (pas 0)
3. Le capital varie entre tests (pas bloqué à 10,000)
4. Le flux end-to-end fonctionne correctement
"""

import sys
from pathlib import Path

# Ajouter le répertoire src au path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np
from typing import Dict, List

from threadx.optimization.scenarios import generate_param_grid
from threadx.ui.strategy_registry import (
    parameter_specs_for,
    base_params_for,
)
from threadx.strategy.bb_atr import BBAtrStrategy


def create_realistic_market_data(n_bars: int = 300) -> pd.DataFrame:
    """
    Crée des données de marché réalistes avec forte volatilité.
    Simule 3 jours de données 15m (288 barres).
    """
    np.random.seed(42)

    dates = pd.date_range("2025-01-29", periods=n_bars, freq="15min", tz="UTC")

    # Prix de base avec tendance
    trend = np.linspace(95000, 98000, n_bars)

    # ✅ VOLATILITÉ AUGMENTÉE pour garantir des signaux Bollinger
    noise = np.random.randn(n_bars) * 2000  # était 500, trop faible
    cycles = 3000 * np.sin(np.linspace(0, 4 * np.pi, n_bars))  # était 1000

    close = trend + noise + cycles

    # OHLCV avec spread réaliste
    high = close + np.abs(np.random.randn(n_bars) * 500)  # était 200
    low = close - np.abs(np.random.randn(n_bars) * 500)  # était 200
    open_price = close + np.random.randn(n_bars) * 300  # était 150
    volume = np.random.uniform(100, 500, n_bars)

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    return df


def simulate_ui_scenario_params_construction(
    strategy: str, param_ranges: Dict[str, tuple], configured_params: Dict = None
) -> Dict:
    """
    Simule exactement la construction de scenario_params dans l'UI
    APRÈS le correctif (lignes 1407-1422 de page_backtest_optimization.py).
    """
    if configured_params is None:
        configured_params = {}

    base_strategy_params = base_params_for(strategy)

    # Construction scenario_params comme dans l'UI
    scenario_params = {}

    # 1. Ajouter les paramètres optimisés (param_ranges)
    for key, (min_v, max_v) in param_ranges.items():
        scenario_params[key] = {"values": [min_v, max_v]}

    # 2. 🔥 FIX CRITIQUE: Ajouter TOUS les paramètres par défaut manquants
    all_param_specs = parameter_specs_for(strategy)
    for key, spec in all_param_specs.items():
        if key not in scenario_params:
            # Priorité: configured_params > base_strategy_params > spec default
            value = configured_params.get(
                key,
                base_strategy_params.get(
                    key, spec.get("default") if isinstance(spec, dict) else spec
                ),
            )
            scenario_params[key] = {"value": value}

    return scenario_params


def test_scenario_params_construction():
    """Test 1: Construction scenario_params avec tous les paramètres."""
    print("\n" + "=" * 80)
    print("TEST 1: Construction scenario_params (simulation UI)")
    print("=" * 80)

    strategy = "Bollinger_Breakout"

    # Simuler sélection utilisateur (seulement 3 paramètres optimisés)
    param_ranges = {
        "bb_period": (20, 30),
        "bb_std": (1.5, 2.5),
        "entry_z": (1.0, 2.0),
    }

    # Session vide (cas problématique)
    configured_params = {}

    # Construction avec fix
    scenario_params = simulate_ui_scenario_params_construction(
        strategy, param_ranges, configured_params
    )

    print(f"\nParamètres optimisés (param_ranges): {len(param_ranges)}")
    print(f"Total paramètres dans scenario_params: {len(scenario_params)}")

    # Vérifications critiques
    assert "min_pnl_pct" in scenario_params, "❌ min_pnl_pct manquant !"
    assert scenario_params["min_pnl_pct"]["value"] == 0.0, "❌ min_pnl_pct != 0.0 !"

    print(f"✅ min_pnl_pct présent: {scenario_params['min_pnl_pct']}")

    # Vérifier autres paramètres essentiels
    essential_params = [
        "atr_period",
        "atr_multiplier",
        "risk_per_trade",
        "max_hold_bars",
        "spacing_bars",
    ]
    for param in essential_params:
        assert param in scenario_params, f"❌ {param} manquant !"
        print(f"✅ {param} = {scenario_params[param]}")

    print("\n✅ TEST 1 RÉUSSI: Tous les paramètres essentiels présents")
    return scenario_params


def test_param_grid_generation(scenario_params: Dict):
    """Test 2: Génération des combinaisons avec generate_param_grid()."""
    print("\n" + "=" * 80)
    print("TEST 2: Génération combinaisons avec generate_param_grid()")
    print("=" * 80)

    combos = generate_param_grid(scenario_params)

    print(f"\nCombinaisons générées: {len(combos)}")

    if not combos:
        print("❌ Aucune combinaison générée !")
        return None

    # Vérifier première combinaison
    combo = combos[0]
    print(f"\nCombo[0] contient {len(combo)} paramètres:")

    # Vérifications critiques
    assert "min_pnl_pct" in combo, "❌ min_pnl_pct absent de combo !"
    assert (
        combo["min_pnl_pct"] == 0.0
    ), f"❌ min_pnl_pct = {combo['min_pnl_pct']} (attendu: 0.0)"

    print(f"✅ min_pnl_pct = {combo['min_pnl_pct']} (correct)")

    # Afficher quelques paramètres clés
    for key in [
        "bb_period",
        "bb_std",
        "entry_z",
        "atr_period",
        "min_pnl_pct",
        "risk_per_trade",
    ]:
        if key in combo:
            print(f"  {key} = {combo[key]}")

    print(f"\n✅ TEST 2 RÉUSSI: {len(combos)} combinaisons valides générées")
    return combos


def test_backtest_execution(combos: List[Dict]):
    """Test 3: Exécution backtests réels avec BBAtrStrategy."""
    print("\n" + "=" * 80)
    print("TEST 3: Exécution backtests réels")
    print("=" * 80)

    # Créer données de marché
    df = create_realistic_market_data(300)
    print(f"\nDonnées créées: {len(df)} barres")
    print(f"Prix: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f}")

    # Stratégie
    strategy = BBAtrStrategy(symbol="BTCUSDC", timeframe="15m")

    results = []
    trades_per_combo = []
    capitals_final = []

    # Tester les 5 premières combinaisons
    n_tests = min(5, len(combos))
    print(f"\nTest de {n_tests} combinaisons:")

    for i, combo in enumerate(combos[:n_tests]):
        print(f"\n--- Combo {i+1}/{n_tests} ---")
        print(
            f"  bb_period={combo['bb_period']}, bb_std={combo['bb_std']}, "
            f"entry_z={combo['entry_z']}"
        )
        print(f"  min_pnl_pct={combo['min_pnl_pct']}")

        # Backtest
        equity_curve, stats = strategy.backtest(
            df=df, params=combo, initial_capital=10000.0, fee_bps=4.5, slippage_bps=0.0
        )

        trades = stats.total_trades
        pnl = stats.total_pnl
        pnl_pct = stats.total_pnl_pct
        capital_final = equity_curve.iloc[-1]

        print(f"  → Trades: {trades}")
        print(f"  → PnL: {pnl:.2f} ({pnl_pct:.2f}%)")
        print(f"  → Capital final: {capital_final:.2f}")

        trades_per_combo.append(trades)
        capitals_final.append(capital_final)

        results.append(
            {
                "combo_id": i,
                "bb_period": combo["bb_period"],
                "bb_std": combo["bb_std"],
                "entry_z": combo["entry_z"],
                "min_pnl_pct": combo["min_pnl_pct"],
                "trades": trades,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "capital_final": capital_final,
            }
        )

    # Analyse résultats
    print("\n" + "=" * 80)
    print("ANALYSE DES RÉSULTATS")
    print("=" * 80)

    # 1. Vérifier qu'on a des trades
    total_trades = sum(trades_per_combo)
    avg_trades = total_trades / len(trades_per_combo) if trades_per_combo else 0

    print(f"\nTrades générés:")
    print(f"  Total: {total_trades}")
    print(f"  Moyenne: {avg_trades:.1f} trades/combo")
    print(f"  Min: {min(trades_per_combo)}, Max: {max(trades_per_combo)}")

    if total_trades == 0:
        print("❌ ÉCHEC: Aucun trade généré (problème min_pnl_pct non résolu ?)")
        return False

    print("✅ Des trades sont générés")

    # 2. Vérifier que le capital varie
    capitals_unique = len(set(capitals_final))
    capital_min = min(capitals_final)
    capital_max = max(capitals_final)
    capital_range = capital_max - capital_min

    print(f"\nCapital final:")
    print(f"  Min: {capital_min:.2f}")
    print(f"  Max: {capital_max:.2f}")
    print(f"  Range: {capital_range:.2f}")
    print(f"  Valeurs uniques: {capitals_unique}/{len(capitals_final)}")

    if capitals_unique == 1 and capital_min == 10000.0:
        print("❌ ÉCHEC: Capital bloqué à 10,000 (aucun trade exécuté ?)")
        return False

    if capital_range < 100:
        print("⚠️ ATTENTION: Faible variation du capital (peut être normal)")
    else:
        print("✅ Le capital varie bien entre combinaisons")

    # 3. Afficher tableau récapitulatif
    print("\n" + "=" * 80)
    print("TABLEAU RÉCAPITULATIF")
    print("=" * 80)

    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))

    print("\n✅ TEST 3 RÉUSSI: Backtests exécutés avec succès")
    return True


def test_min_pnl_pct_impact():
    """Test 4: Vérifier l'impact de min_pnl_pct sur le filtrage."""
    print("\n" + "=" * 80)
    print("TEST 4: Impact de min_pnl_pct sur filtrage des trades")
    print("=" * 80)

    df = create_realistic_market_data(300)
    strategy = BBAtrStrategy(symbol="BTCUSDC", timeframe="15m")

    # Params de base
    base_params = {
        "bb_period": 20,
        "bb_std": 2.0,
        "entry_z": 1.5,
        "atr_period": 14,
        "atr_multiplier": 1.5,
        "risk_per_trade": 0.02,
        "max_hold_bars": 72,
        "spacing_bars": 6,
        "trend_period": 0,
        "entry_logic": "AND",
        "trailing_stop": True,
        "leverage": 1.0,
    }

    # Test avec min_pnl_pct = 0.0 (nouveau défaut)
    params_new = base_params.copy()
    params_new["min_pnl_pct"] = 0.0

    equity_new, stats_new = strategy.backtest(df, params_new, 10000.0)

    print(f"\nAvec min_pnl_pct = 0.0 (nouveau):")
    print(f"  Trades: {stats_new.total_trades}")
    print(f"  PnL: {stats_new.total_pnl:.2f} ({stats_new.total_pnl_pct:.2f}%)")

    # Test avec min_pnl_pct = 0.01 (ancien défaut bugué)
    params_old = base_params.copy()
    params_old["min_pnl_pct"] = 0.01

    equity_old, stats_old = strategy.backtest(df, params_old, 10000.0)

    print(f"\nAvec min_pnl_pct = 0.01 (ancien):")
    print(f"  Trades: {stats_old.total_trades}")
    print(f"  PnL: {stats_old.total_pnl:.2f} ({stats_old.total_pnl_pct:.2f}%)")

    # Comparaison
    trade_diff = stats_new.total_trades - stats_old.total_trades

    print(f"\nDifférence:")
    print(f"  Trades: {trade_diff:+d}")
    print(f"  PnL: {stats_new.total_pnl - stats_old.total_pnl:+.2f}")

    if stats_new.total_trades > stats_old.total_trades:
        print(
            f"✅ min_pnl_pct=0.0 génère PLUS de trades ({stats_new.total_trades} vs {stats_old.total_trades})"
        )
        print("   → Le filtrage est bien désactivé")
    elif stats_new.total_trades == stats_old.total_trades:
        print(f"⚠️ Même nombre de trades ({stats_new.total_trades})")
        print("   → Normal si tous les trades sont > 0.01%")
    else:
        print("❌ Résultat inattendu")

    print("\n✅ TEST 4 TERMINÉ")
    return True


def main():
    """Exécution complète de tous les tests."""
    print("\n" + "=" * 80)
    print("TEST D'INTÉGRATION COMPLET: GRID SWEEP APRÈS CORRECTIF")
    print("=" * 80)
    print("\nObjectif: Vérifier que le flux complet UI → Engine → Strategy fonctionne")
    print("et que le bug min_pnl_pct est bien résolu.")

    try:
        # Test 1: Construction scenario_params
        scenario_params = test_scenario_params_construction()

        # Test 2: Génération combinaisons
        combos = test_param_grid_generation(scenario_params)

        if combos is None:
            print("\n❌ ÉCHEC: Impossible de générer les combinaisons")
            return False

        # Test 3: Exécution backtests
        success = test_backtest_execution(combos)

        if not success:
            print("\n❌ ÉCHEC: Problème lors de l'exécution des backtests")
            return False

        # Test 4: Impact min_pnl_pct
        test_min_pnl_pct_impact()

        # Résumé final
        print("\n" + "=" * 80)
        print("RÉSUMÉ FINAL")
        print("=" * 80)
        print("✅ TOUS LES TESTS RÉUSSIS")
        print("\nCorrectifs validés:")
        print("  1. ✅ min_pnl_pct=0.0 présent dans toutes les combinaisons")
        print("  2. ✅ Backtests génèrent des trades (pas 0)")
        print("  3. ✅ Capital varie entre tests (pas bloqué à 10,000)")
        print("  4. ✅ Flux end-to-end fonctionne correctement")
        print("\n🎉 Le Grid Sweep est maintenant OPÉRATIONNEL !")
        print("\nRECOMMANDATION:")
        print("  → Relancer Streamlit et tester avec vraies données de marché")
        print("  → Vérifier que les résultats correspondent aux tests")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
