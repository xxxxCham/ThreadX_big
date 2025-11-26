#!/usr/bin/env python
"""Analyse de fiabilité et crédibilité des résultats Multi-LLM"""

import json
from datetime import datetime
from pathlib import Path


def main():
    data = json.load(open(r"c:\Users\o3-Pro\Downloads\test.json", encoding="utf-8"))

    print("=" * 70)
    print("🔍 ANALYSE DE FIABILITÉ ET CRÉDIBILITÉ DES RÉSULTATS")
    print("=" * 70)

    issues = []  # Collecter les problèmes
    warnings = []  # Collecter les avertissements

    sweep = data["sweep"]
    top_configs = sweep.get("top_configs", [])
    tests = data.get("tests", {})

    # =========================================================================
    # 1. COHÉRENCE DU SWEEP
    # =========================================================================
    print("\n📊 1. COHÉRENCE DU SWEEP GPU")
    print("-" * 50)

    # Vérifier diversité des top configs
    if top_configs:
        sharpes = [c["stats"]["sharpe_ratio"] for c in top_configs]
        unique_sharpes = len(set([round(s, 6) for s in sharpes]))
        print(f"  Top {len(top_configs)} configs - Sharpes uniques: {unique_sharpes}")

        if unique_sharpes < len(top_configs) // 2:
            warnings.append(
                f"Faible diversité: seulement {unique_sharpes} Sharpes uniques sur {len(top_configs)}"
            )
            print(f"  ⚠️  ALERTE: Faible diversité (plateau d'optimisation?)")
        else:
            print(f"  ✅ Diversité acceptable")

    # =========================================================================
    # 2. VÉRIFICATION ARITHMÉTIQUE DES TRADES
    # =========================================================================
    print("\n📈 2. VÉRIFICATION ARITHMÉTIQUE DES TRADES")
    print("-" * 50)

    if top_configs:
        best_stats = top_configs[0]["stats"]
        trades = best_stats.get("meta", {}).get("trades", [])

        if trades:
            # Recalculer le P&L total
            calc_pnl = sum(t.get("pnl_realized", 0) for t in trades)
            reported_pnl = best_stats.get("total_pnl", 0)
            pnl_diff = abs(reported_pnl - calc_pnl)

            print(f"  P&L rapporté: ${reported_pnl:.2f}")
            print(f"  P&L calculé (somme trades): ${calc_pnl:.2f}")
            print(f"  Écart: ${pnl_diff:.4f}")

            if pnl_diff < 0.01:
                print("  ✅ P&L COHÉRENT")
            elif pnl_diff < 1.0:
                print("  ⚠️  Écart mineur (arrondi)")
                warnings.append(f"Écart P&L mineur: ${pnl_diff:.4f}")
            else:
                print("  ❌ ÉCART SIGNIFICATIF!")
                issues.append(f"Écart P&L significatif: ${pnl_diff:.2f}")

            # Vérifier comptage trades
            wins = len([t for t in trades if t.get("pnl_realized", 0) > 0])
            losses = len([t for t in trades if t.get("pnl_realized", 0) < 0])
            zeros = len([t for t in trades if t.get("pnl_realized", 0) == 0])

            print(f"\n  Trades rapportés: {best_stats.get('total_trades', 0)}")
            print(f"  Trades comptés: {len(trades)}")

            if best_stats.get("total_trades") == len(trades):
                print("  ✅ Nombre de trades cohérent")
            else:
                issues.append(
                    f"Incohérence nombre trades: {best_stats.get('total_trades')} vs {len(trades)}"
                )
                print("  ❌ INCOHÉRENCE!")

            print(
                f"\n  Wins: rapporté={best_stats.get('win_trades', 0)} | calculé={wins}"
            )
            print(
                f"  Losses: rapporté={best_stats.get('loss_trades', 0)} | calculé={losses}"
            )
            print(f"  Trades à zéro: {zeros}")

            if (
                best_stats.get("win_trades") == wins
                and best_stats.get("loss_trades") == losses
            ):
                print("  ✅ Comptage wins/losses cohérent")
            else:
                issues.append("Incohérence comptage wins/losses")
                print("  ❌ INCOHÉRENCE COMPTAGE!")

            # Win rate
            calc_winrate = 100 * wins / len(trades) if trades else 0
            reported_winrate = best_stats.get("win_rate_pct", 0)
            print(f"\n  Win Rate rapporté: {reported_winrate:.2f}%")
            print(f"  Win Rate calculé: {calc_winrate:.2f}%")

            if abs(calc_winrate - reported_winrate) < 0.1:
                print("  ✅ Win rate cohérent")
            else:
                issues.append(
                    f"Écart win rate: {abs(calc_winrate - reported_winrate):.2f}%"
                )
                print("  ❌ ÉCART WIN RATE!")

            # Vérifier equity finale
            initial = best_stats.get("initial_capital", 10000)
            final = best_stats.get("final_equity", 0)
            calc_final = initial + calc_pnl

            print(f"\n  Capital initial: ${initial:.2f}")
            print(f"  Equity finale rapportée: ${final:.2f}")
            print(f"  Equity finale calculée: ${calc_final:.2f}")

            if abs(final - calc_final) < 0.01:
                print("  ✅ Equity finale cohérente")
            else:
                issues.append(f"Écart equity finale: ${abs(final - calc_final):.2f}")
                print("  ⚠️  Écart détecté")

    # =========================================================================
    # 3. COHÉRENCE TESTS vs BASELINE
    # =========================================================================
    print("\n🧪 3. COHÉRENCE DES TESTS DE PROPOSITIONS")
    print("-" * 50)

    tested = tests.get("tested_proposals", [])
    baseline_sharpe = tests.get("baseline_sharpe", 0)

    if tested:
        # Trouver la baseline dans les tests
        baseline_test = next(
            (t for t in tested if "baseline" in t.get("name", "").lower()), None
        )

        if baseline_test:
            print(f"  Baseline Sharpe (déclaré): {baseline_sharpe:.4f}")
            print(
                f"  Baseline Sharpe (testé): {baseline_test.get('sharpe_ratio', 0):.4f}"
            )

            if abs(baseline_sharpe - baseline_test.get("sharpe_ratio", 0)) < 0.0001:
                print("  ✅ Baseline cohérente")
            else:
                warnings.append("Écart baseline déclarée vs testée")
                print("  ⚠️  Légère différence")

        # Vérifier les vs_baseline_sharpe
        print("\n  Vérification des deltas vs baseline:")
        for prop in tested:
            calc_delta = prop.get("sharpe_ratio", 0) - baseline_sharpe
            reported_delta = prop.get("vs_baseline_sharpe", 0)

            status = "✅" if abs(calc_delta - reported_delta) < 0.0001 else "⚠️"
            print(
                f"    {status} {prop['name'][:20]:<20}: delta calculé={calc_delta:+.4f}, rapporté={reported_delta:+.4f}"
            )

    # =========================================================================
    # 4. VÉRIFICATION TEMPORELLE DES TRADES
    # =========================================================================
    print("\n⏱️  4. COHÉRENCE TEMPORELLE DES TRADES")
    print("-" * 50)

    if top_configs and trades:
        # Vérifier l'ordre chronologique
        trade_times = []
        for t in trades:
            entry = t.get("entry_time", "")
            if entry:
                trade_times.append(entry)

        sorted_times = sorted(trade_times)
        is_chronological = trade_times == sorted_times

        print(f"  Premier trade: {trades[0].get('entry_time', 'N/A')}")
        print(f"  Dernier trade: {trades[-1].get('entry_time', 'N/A')}")
        print(
            f"  Trades en ordre chronologique: {'✅ Oui' if is_chronological else '❌ Non'}"
        )

        if not is_chronological:
            issues.append("Trades pas en ordre chronologique")

        # Vérifier qu'il n'y a pas de trades simultanés (overlapping)
        overlaps = 0
        for i in range(len(trades) - 1):
            exit_time = trades[i].get("exit_time", "")
            next_entry = trades[i + 1].get("entry_time", "")
            if exit_time and next_entry and exit_time > next_entry:
                overlaps += 1

        print(f"  Trades qui se chevauchent: {overlaps}")
        if overlaps > 0:
            warnings.append(f"{overlaps} trades se chevauchent")
            print("  ⚠️  Possibles positions simultanées (vérifier si autorisé)")
        else:
            print("  ✅ Pas de chevauchement")

    # =========================================================================
    # 5. VRAISEMBLANCE DES MÉTRIQUES
    # =========================================================================
    print("\n📏 5. VRAISEMBLANCE DES MÉTRIQUES")
    print("-" * 50)

    if top_configs:
        stats = top_configs[0]["stats"]

        # Sharpe ratio plausible
        sharpe = stats.get("sharpe_ratio", 0)
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        if -3 < sharpe < 5:
            print("    ✅ Valeur plausible")
        else:
            warnings.append(f"Sharpe ratio suspect: {sharpe}")
            print("    ⚠️  Valeur inhabituellement extrême")

        # Win rate plausible
        winrate = stats.get("win_rate_pct", 0)
        print(f"  Win Rate: {winrate:.2f}%")
        if 15 < winrate < 85:
            print("    ✅ Valeur plausible pour MA Crossover")
        elif winrate < 15:
            warnings.append(f"Win rate très bas: {winrate}%")
            print("    ⚠️  Win rate très bas (mais possible avec bon R:R)")
        else:
            warnings.append(f"Win rate très haut: {winrate}%")
            print("    ⚠️  Win rate inhabituellement haut")

        # Drawdown plausible
        dd = stats.get("max_drawdown_pct", 0)
        print(f"  Max Drawdown: {dd:.2f}%")
        if -50 < dd < 0:
            print("    ✅ Valeur plausible")
        elif dd > 0:
            issues.append(f"Drawdown positif suspect: {dd}%")
            print("    ❌ Drawdown positif (erreur de signe?)")
        else:
            warnings.append(f"Drawdown très élevé: {dd}%")
            print("    ⚠️  Drawdown très élevé")

        # Profit factor
        avg_win = stats.get("avg_win", 0)
        avg_loss = abs(stats.get("avg_loss", 1))
        win_count = stats.get("win_trades", 0)
        loss_count = stats.get("loss_trades", 1)

        if avg_loss > 0 and loss_count > 0:
            calc_pf = (avg_win * win_count) / (avg_loss * loss_count)
            reported_pf = stats.get("profit_factor", 0)
            print(f"\n  Profit Factor rapporté: {reported_pf:.4f}")
            print(f"  Profit Factor calculé: {calc_pf:.4f}")

            if abs(calc_pf - reported_pf) < 0.01:
                print("    ✅ Cohérent")
            else:
                warnings.append(
                    f"Écart profit factor: {abs(calc_pf - reported_pf):.4f}"
                )
                print("    ⚠️  Légère différence (méthode de calcul?)")

    # =========================================================================
    # 6. COHÉRENCE LLM
    # =========================================================================
    print("\n🤖 6. COHÉRENCE ANALYSE LLM")
    print("-" * 50)

    analysis = data.get("analysis", {})
    proposals = data.get("proposals", {})

    # Vérifier que l'analyse a bien été faite
    if analysis.get("duration_seconds", 0) > 0:
        print(f"  ✅ Analyse complétée en {analysis['duration_seconds']:.1f}s")
    else:
        issues.append("Analyse LLM non complétée")
        print("  ❌ Analyse non complétée")

    # Vérifier les propositions
    props = proposals.get("proposals", [])
    if len(props) >= 2:
        print(f"  ✅ {len(props)} propositions générées")
    else:
        warnings.append(f"Seulement {len(props)} proposition(s)")
        print(f"  ⚠️  Seulement {len(props)} proposition(s)")

    # Vérifier que les propositions sont différentes
    if props:
        params_str = [str(p.get("params", {})) for p in props]
        unique_props = len(set(params_str))
        print(f"  Propositions uniques: {unique_props}/{len(props)}")

        if unique_props == len(props):
            print("    ✅ Toutes différentes")
        else:
            warnings.append("Propositions dupliquées")
            print("    ⚠️  Certaines sont identiques")

    # =========================================================================
    # 7. VÉRIFICATION AMÉLIORATION FINALE
    # =========================================================================
    print("\n🏆 7. VÉRIFICATION DE L'AMÉLIORATION")
    print("-" * 50)

    best_sharpe = data.get("best_sharpe", 0)
    sweep_best = top_configs[0]["stats"]["sharpe_ratio"] if top_configs else 0

    print(f"  Meilleur Sharpe du sweep: {sweep_best:.4f}")
    print(f"  Meilleur Sharpe final: {best_sharpe:.4f}")
    print(
        f"  Amélioration: {((best_sharpe - sweep_best) / sweep_best * 100) if sweep_best else 0:.2f}%"
    )

    if best_sharpe > sweep_best:
        print("  ✅ Amélioration confirmée par les LLMs")
    elif best_sharpe == sweep_best:
        print("  ⚠️  Pas d'amélioration vs sweep (LLM n'a pas trouvé mieux)")
        warnings.append("Aucune amélioration trouvée par LLM")
    else:
        issues.append("Best sharpe final inférieur au sweep")
        print("  ❌ RÉGRESSION (problème!)")

    # =========================================================================
    # RÉSUMÉ FINAL
    # =========================================================================
    print("\n" + "=" * 70)
    print("📋 RÉSUMÉ DE L'AUDIT")
    print("=" * 70)

    print(f"\n  ❌ Problèmes critiques: {len(issues)}")
    for i, issue in enumerate(issues, 1):
        print(f"     {i}. {issue}")

    print(f"\n  ⚠️  Avertissements: {len(warnings)}")
    for i, warn in enumerate(warnings, 1):
        print(f"     {i}. {warn}")

    # Score de fiabilité
    score = 100 - (len(issues) * 20) - (len(warnings) * 5)
    score = max(0, min(100, score))

    print(f"\n  📊 SCORE DE FIABILITÉ: {score}/100")

    if score >= 90:
        print("     🟢 EXCELLENT - Données très fiables")
    elif score >= 70:
        print("     🟡 BON - Données globalement fiables, quelques points à vérifier")
    elif score >= 50:
        print("     🟠 MOYEN - Vérifications recommandées")
    else:
        print("     🔴 FAIBLE - Problèmes significatifs détectés")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
