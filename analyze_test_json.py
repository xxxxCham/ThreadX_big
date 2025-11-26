#!/usr/bin/env python
"""Analyse du fichier test.json généré par Multi-LLM Optimizer"""

import json
from pathlib import Path


def main():
    filepath = Path(r"c:\Users\o3-Pro\Downloads\test.json")
    data = json.load(open(filepath, encoding="utf-8"))

    print("=" * 70)
    print("📊 ANALYSE COMPLÈTE DU FICHIER test.json (Multi-LLM Optimizer)")
    print("=" * 70)

    # === METADATA ===
    print(f"\n🔖 Run ID: {data['run_id']}")
    print(f"📅 Timestamp: {data['timestamp']}")
    print(f"🎯 Stratégie: {data['strategy_name']}")
    print(
        f"⏱️  Durée totale: {data['total_duration_seconds']:.1f}s ({data['total_duration_seconds']/60:.1f} min)"
    )

    # === CONFIGURATION ===
    config = data.get("config", {})
    print(f"\n{'='*70}")
    print("⚙️  CONFIGURATION")
    print(f"{'='*70}")
    print(f"  GPU: {config.get('use_gpu', 'N/A')}")
    print(f"  Multi-GPU: {config.get('use_multigpu', 'N/A')}")
    print(f"  Max Workers: {config.get('max_workers', 'N/A')}")
    print(f"  Feeder Aggr: {config.get('feeder_aggr', 'N/A')}")
    print(f"  N Proposals: {config.get('n_proposals', 'N/A')}")
    print(f"  Top N Analysis: {config.get('top_n_analysis', 'N/A')}")
    print(f"  Memory Saver: {config.get('memory_saver', 'N/A')}")

    # === SWEEP GPU ===
    sweep = data["sweep"]
    print(f"\n{'='*70}")
    print("🔄 SWEEP GPU")
    print(f"{'='*70}")
    print(f"  Configurations testées: {sweep['total_configs']:,}")
    print(f"  Durée: {sweep['duration_seconds']:.1f}s")
    print(f"  Top configs sauvegardées: {len(sweep.get('top_configs', []))}")

    if sweep.get("top_configs"):
        print(f"\n  📈 TOP 5 CONFIGURATIONS DU SWEEP:")
        print(
            f"  {'Rank':<5} {'Sharpe':<10} {'Return%':<10} {'Trades':<8} {'WinRate%':<10} {'MaxDD%':<10}"
        )
        print(f"  {'-'*55}")
        for i, cfg in enumerate(sweep["top_configs"][:5], 1):
            stats = cfg["stats"]
            print(
                f"  {i:<5} {stats['sharpe_ratio']:<10.4f} {stats['total_pnl_pct']:<10.2f} "
                f"{stats['total_trades']:<8} {stats['win_rate_pct']:<10.2f} {stats['max_drawdown_pct']:<10.2f}"
            )

        # Paramètres de la meilleure config
        best = sweep["top_configs"][0]
        print(f"\n  🏆 Meilleurs paramètres (sweep):")
        print(f"     fast_period: {best['params']['fast_period']}")
        print(f"     slow_period: {best['params']['slow_period']}")
        print(f"     stop_loss_pct: {best['params']['stop_loss_pct']}%")
        print(f"     take_profit_pct: {best['params']['take_profit_pct']}%")
        print(f"     risk_per_trade: {best['params']['risk_per_trade']}")
        print(f"     max_hold_bars: {best['params']['max_hold_bars']}")

    # === ANALYSE LLM ===
    analysis = data.get("analysis", {})
    print(f"\n{'='*70}")
    print("🤖 ANALYSE LLM (Analyst)")
    print(f"{'='*70}")
    print(f"  Modèle: {analysis.get('model_used', 'N/A')}")
    print(f"  Durée: {analysis.get('duration_seconds', 0):.1f}s")

    if analysis.get("patterns"):
        print(f"\n  📌 PATTERNS IDENTIFIÉS ({len(analysis['patterns'])}):")
        for i, pattern in enumerate(analysis["patterns"], 1):
            print(f"     {i}. {pattern}")

    if analysis.get("key_metrics"):
        print(f"\n  📊 MÉTRIQUES CLÉS:")
        for metric, value in analysis["key_metrics"].items():
            print(f"     • {metric}: {value}")

    if analysis.get("recommendations"):
        print(f"\n  💡 RECOMMANDATIONS ({len(analysis['recommendations'])}):")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"     {i}. {rec[:80]}..." if len(rec) > 80 else f"     {i}. {rec}")

    # === PROPOSALS ===
    proposals = data.get("proposals", {})
    print(f"\n{'='*70}")
    print("📝 PROPOSITIONS (Strategist)")
    print(f"{'='*70}")
    print(f"  Modèle: {proposals.get('model_used', 'N/A')}")
    print(f"  Durée: {proposals.get('duration_seconds', 0):.1f}s")
    print(f"  Propositions générées: {len(proposals.get('proposals', []))}")

    if proposals.get("proposals"):
        print(f"\n  📋 DÉTAIL DES PROPOSITIONS:")
        for i, prop in enumerate(proposals["proposals"], 1):
            print(f"\n     [{i}] {prop.get('name', 'Sans nom')}")
            params = prop.get("params", {})
            print(
                f"         fast: {params.get('fast_period')}, slow: {params.get('slow_period')}"
            )
            print(
                f"         SL: {params.get('stop_loss_pct')}%, TP: {params.get('take_profit_pct')}%"
            )
            if prop.get("rationale"):
                rationale = (
                    prop["rationale"][:100] + "..."
                    if len(prop.get("rationale", "")) > 100
                    else prop.get("rationale", "")
                )
                print(f"         Rationale: {rationale}")

    # === TESTS DES PROPOSITIONS ===
    tests = data.get("tests", {})
    print(f"\n{'='*70}")
    print("🧪 TESTS DES PROPOSITIONS")
    print(f"{'='*70}")
    print(f"  Durée: {tests.get('duration_seconds', 0):.1f}s")
    print(
        f"  Amélioration trouvée: {'✅ Oui' if tests.get('improvement_found') else '❌ Non'}"
    )
    print(f"  Meilleure proposition: {tests.get('best_proposal', 'N/A')}")
    print(f"  Meilleur Sharpe: {tests.get('best_sharpe', 0):.4f}")
    print(f"  Baseline Sharpe: {tests.get('baseline_sharpe', 0):.4f}")

    if tests.get("tested_proposals"):
        print(f"\n  📊 RÉSULTATS DES TESTS:")
        print(
            f"  {'Nom':<25} {'Sharpe':<10} {'Return%':<10} {'Trades':<8} {'vs Base':<12} {'OK?':<5}"
        )
        print(f"  {'-'*75}")
        for prop in tests["tested_proposals"]:
            vs_base = prop.get("vs_baseline_sharpe", 0)
            vs_str = f"+{vs_base:.4f}" if vs_base > 0 else f"{vs_base:.4f}"
            ok = "✅" if prop.get("is_improvement") else "❌"
            print(
                f"  {prop['name']:<25} {prop.get('sharpe_ratio', 0):<10.4f} "
                f"{prop.get('total_return', 0):<10.2f} {prop.get('total_trades', 0):<8} "
                f"{vs_str:<12} {ok:<5}"
            )

    # === MEILLEURE CONFIG FINALE ===
    print(f"\n{'='*70}")
    print("🏆 MEILLEURE CONFIGURATION FINALE")
    print(f"{'='*70}")
    best_config = data.get("best_config", {})
    print(f"  Sharpe Ratio: {data.get('best_sharpe', 0):.4f}")
    print(f"\n  Paramètres:")
    for k, v in best_config.items():
        print(f"    • {k}: {v}")

    # === RÉSUMÉ ===
    print(f"\n{'='*70}")
    print("📋 RÉSUMÉ")
    print(f"{'='*70}")
    print(data.get("summary", "Pas de résumé disponible"))

    # === STATISTIQUES DES TRADES (si disponible) ===
    if sweep.get("top_configs") and sweep["top_configs"][0]["stats"].get(
        "meta", {}
    ).get("trades"):
        trades = sweep["top_configs"][0]["stats"]["meta"]["trades"]
        print(f"\n{'='*70}")
        print(f"📈 ANALYSE DES {len(trades)} TRADES DE LA MEILLEURE CONFIG")
        print(f"{'='*70}")

        total_pnl = sum(t.get("pnl_realized", 0) for t in trades)
        wins = [t for t in trades if t.get("pnl_realized", 0) > 0]
        losses = [t for t in trades if t.get("pnl_realized", 0) < 0]
        longs = [t for t in trades if t.get("side") == "LONG"]
        shorts = [t for t in trades if t.get("side") == "SHORT"]

        print(f"  Total P&L: ${total_pnl:.2f}")
        print(f"  Trades gagnants: {len(wins)} ({100*len(wins)/len(trades):.1f}%)")
        print(f"  Trades perdants: {len(losses)} ({100*len(losses)/len(trades):.1f}%)")
        print(f"  LONG: {len(longs)} | SHORT: {len(shorts)}")

        if wins:
            avg_win = sum(t["pnl_realized"] for t in wins) / len(wins)
            max_win = max(t["pnl_realized"] for t in wins)
            print(f"  Gain moyen: ${avg_win:.2f} | Max: ${max_win:.2f}")

        if losses:
            avg_loss = sum(t["pnl_realized"] for t in losses) / len(losses)
            max_loss = min(t["pnl_realized"] for t in losses)
            print(f"  Perte moyenne: ${avg_loss:.2f} | Max: ${max_loss:.2f}")


if __name__ == "__main__":
    main()
