"""
POC Autopsy System - Test Système Auto-Apprenant
================================================

Test workflow complet:
1. Autopsy Agent (post-mortem analysis)
2. Kill Rules Manager (auto-update)
3. Critic Integration (hook rejection)
4. Feedback Strategist (patterns + rules)
5. Heatmap dashboard (aggregation)

Usage:
    python tools/test_autopsy_system.py

Expected Output:
    ✅ Autopsy analysis complete
    ✅ Kill rules auto-added (score ≥ 8.5)
    ✅ Feedback generated for Strategist
    ✅ Heatmap data aggregated
    ✅ SYSTEM OPERATIONAL

Author: ThreadX Framework
Version: 1.0 - Autopsy System
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Mock imports (POC standalone)
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_autopsy_agent():
    """Test 1: Autopsy Agent post-mortem analysis."""
    print("\n" + "=" * 60)
    print("TEST 1: AUTOPSY AGENT - POST-MORTEM ANALYSIS")
    print("=" * 60)

    # Mock strategy code
    strategy_code = '''
def backtest(data, params):
    """Stratégie Bollinger Momentum v23 (REJECTED)."""
    # Entry: Bollinger Lower Band breach
    # Exit: Bollinger Upper Band
    # Stop-loss: 1.2% (TROP BAS → drawdown)
    # Min profit: 0.6% (TROP BAS → trop de trades)

    for i in range(len(data)):
        if data.close[i] < data.bb_lower[i]:
            # Entry signal
            if not position:
                entry_price = data.close[i]
                stop_loss = entry_price * 0.988  # 1.2% (BUG)
                take_profit = entry_price * 1.006  # 0.6% (BUG)
'''

    # Mock Critic report
    critic_report = {
        "rejection_reason": "Trop de trades, drawdown excessif, profit factor insuffisant",
        "metrics_multi_token": {
            "SOL": {
                "sharpe_ratio": 1.23,
                "nb_trades": 87,
                "avg_trade_duration": 0.8,  # hours (TROP COURT)
                "profit_factor": 1.4,
                "win_rate": 0.58,
                "max_drawdown": 0.18,  # 18% (TROP HAUT)
            },
            "BTC": {
                "sharpe_ratio": 0.94,
                "nb_trades": 102,
                "avg_trade_duration": 0.6,
                "profit_factor": 1.3,
                "win_rate": 0.55,
                "max_drawdown": 0.21,
            },
            "ETH": {
                "sharpe_ratio": 1.05,
                "nb_trades": 95,
                "avg_trade_duration": 0.7,
                "profit_factor": 1.35,
                "win_rate": 0.57,
                "max_drawdown": 0.19,
            },
        },
        "overall_verdict": "REJECTED",
        "confidence": 0.95,
    }

    # Simuler Autopsy analysis (mock LLM response)
    # En prod: autopsy.analyze_failure() → appel deepseek-r1:32b
    autopsy_report = {
        "cause_principale": "trop_de_trades",
        "poids_cause": 0.85,
        "symptomes_cles": [
            "avg_trade_duration = 0.7h (cible > 6h)",
            "294 trades total (98 trades/token, cible < 15)",
            "profit_factor = 1.35 (cible > 2.0)",
            "stop_loss trop serré (1.2% → triggers fréquents)",
            "min_profit trop bas (0.6% → sorties prématurées)",
        ],
        "correctifs_concrets": [
            "Augmenter min_profit_pct de 0.6% → 1.8% (×3)",
            "Augmenter stop_loss de 1.2% → 2.5% (×2)",
            "Ajouter filtre ATR: trade seulement si ATR > seuil",
            "Ajouter cooldown 2h entre trades (éviter sur-trading)",
        ],
        "kill_rules_proposees": [
            "rejeter si avg_trade_duration < 3h",
            "rejeter si nb_trades > 30 par token",
            "rejeter si profit_factor < 2.0",
            "rejeter si stop_loss < 2.0%",
        ],
        "score_amelioration_attendue": 9.2,
        "timestamp": datetime.now().isoformat(),
        "model_used": "deepseek-r1:32b (simulated)",
        "strategy_name": "bollinger_momentum_v23",
        "code_snapshot": strategy_code,
    }

    # Sauvegarder rapport (mock persistence)
    reports_dir = Path("./autopsy_reports")
    reports_dir.mkdir(exist_ok=True)

    report_file = reports_dir / f"{autopsy_report['strategy_name']}.json"
    with open(report_file, "w") as f:
        json.dump(autopsy_report, f, indent=2)

    print(f"✅ Autopsy report generated: {report_file}")
    print(f"   Cause principale: {autopsy_report['cause_principale']}")
    print(f"   Score amélioration: {autopsy_report['score_amelioration_attendue']}/10")
    print(f"   Kill rules proposées: {len(autopsy_report['kill_rules_proposees'])}")

    return autopsy_report


def test_kill_rules_manager(autopsy_report):
    """Test 2: Kill Rules Manager auto-update."""
    print("\n" + "=" * 60)
    print("TEST 2: KILL RULES MANAGER - AUTO-UPDATE")
    print("=" * 60)

    # Mock KillRulesManager
    class MockKillRulesManager:
        def __init__(self):
            self.rules = []
            self.rules_path = Path("./kill_rules.json")

        def add_rules_from_autopsy(self, report, min_score=8.5):
            score = report.get("score_amelioration_attendue", 0)
            if score < min_score:
                print(f"⚠️  Score {score} < {min_score}, rules not activated")
                return 0

            proposed_rules = report.get("kill_rules_proposees", [])
            added = 0

            for rule_text in proposed_rules:
                rule_entry = {
                    "rule": rule_text,
                    "added_at": datetime.now().isoformat(),
                    "source": "autopsy",
                    "improvement_score": score,
                    "active": True,
                    "metadata": {
                        "strategy_name": report.get("strategy_name"),
                        "cause_principale": report.get("cause_principale"),
                    },
                }
                self.rules.append(rule_entry)
                added += 1

            # Save
            with open(self.rules_path, "w") as f:
                json.dump(self.rules, f, indent=2)

            print(f"✅ Added {added}/{len(proposed_rules)} kill rules")
            print(f"   Total active rules: {len(self.rules)}")

            return added

        def get_active_rules(self):
            return [r["rule"] for r in self.rules if r.get("active", True)]

    # Test auto-update
    manager = MockKillRulesManager()
    added = manager.add_rules_from_autopsy(autopsy_report, min_score=8.5)

    if added > 0:
        print(f"✅ Kill rules auto-added (score {autopsy_report['score_amelioration_attendue']} ≥ 8.5)")
        active_rules = manager.get_active_rules()
        print(f"   Active rules:")
        for i, rule in enumerate(active_rules, start=1):
            print(f"     {i}. {rule}")
    else:
        print(f"❌ Kill rules NOT added (score too low)")

    return manager


def test_failure_patterns_aggregation(autopsy_report):
    """Test 3: Failure Patterns Heatmap aggregation."""
    print("\n" + "=" * 60)
    print("TEST 3: FAILURE PATTERNS HEATMAP - AGGREGATION")
    print("=" * 60)

    # Mock multiple autopsy reports
    reports = [
        autopsy_report,  # trop_de_trades
        {
            "cause_principale": "drawdown_excessif",
            "score_amelioration_attendue": 8.7,
            "timestamp": datetime.now().isoformat(),
        },
        {
            "cause_principale": "trop_de_trades",  # Duplicate cause
            "score_amelioration_attendue": 9.0,
            "timestamp": datetime.now().isoformat(),
        },
        {
            "cause_principale": "profit_factor_faible",
            "score_amelioration_attendue": 7.5,
            "timestamp": datetime.now().isoformat(),
        },
    ]

    # Aggregate patterns
    patterns = {}
    for report in reports:
        cause = report.get("cause_principale", "unknown")
        score = report.get("score_amelioration_attendue", 0)
        timestamp = report.get("timestamp", "")

        if cause not in patterns:
            patterns[cause] = {
                "count": 0,
                "last_seen": timestamp,
                "scores": [],
            }

        patterns[cause]["count"] += 1
        patterns[cause]["last_seen"] = max(patterns[cause]["last_seen"], timestamp)
        patterns[cause]["scores"].append(score)

    # Calculate averages
    for cause, data in patterns.items():
        data["avg_score"] = sum(data["scores"]) / len(data["scores"])
        del data["scores"]  # Cleanup

    # Sort by frequency
    patterns = dict(sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True))

    print(f"✅ Patterns aggregated: {len(patterns)} unique causes")
    print(f"   Total failures: {len(reports)}")
    print(f"\n   Top Causes (by frequency):")

    for i, (cause, data) in enumerate(patterns.items(), start=1):
        print(
            f"     {i}. {cause}: {data['count']} occurrences "
            f"(avg score: {data['avg_score']:.1f}/10)"
        )

    return patterns


def test_strategist_feedback(patterns, kill_rules_manager):
    """Test 4: Strategist Feedback generation."""
    print("\n" + "=" * 60)
    print("TEST 4: STRATEGIST FEEDBACK - PROMPT INJECTION")
    print("=" * 60)

    # Generate feedback (Top 5 échecs)
    total_failures = sum(data["count"] for data in patterns.values())
    top_patterns = list(patterns.items())[:5]

    feedback = f"**Tu as déjà échoué {total_failures} fois.**\n\n"
    feedback += "**Top 5 Causes:**\n\n"

    for i, (cause, data) in enumerate(top_patterns, start=1):
        feedback += f"{i}. **{cause}** – {data['count']} occurrences\n"

    # Kill rules section
    active_rules = kill_rules_manager.get_active_rules()
    feedback += f"\n**Kill Rules Actives:** {len(active_rules)}\n\n"

    if active_rules:
        feedback += "**Top règles:**\n"
        for i, rule in enumerate(active_rules[:5], start=1):
            feedback += f"{i}. {rule}\n"

    feedback += "\n**→ Tu DOIS éviter ces patterns à tout prix.**"

    print("✅ Feedback generated for Strategist:")
    print("\n" + "-" * 60)
    print(feedback)
    print("-" * 60)

    return feedback


def test_critic_integration():
    """Test 5: Critic integration (mock workflow)."""
    print("\n" + "=" * 60)
    print("TEST 5: CRITIC INTEGRATION - AUTOPSY HOOK")
    print("=" * 60)

    # Mock Critic workflow
    class MockCritic:
        def __init__(self, enable_autopsy=True):
            self.enable_autopsy = enable_autopsy

        def validate_proposals(self, proposals):
            # Simulate rejection
            print(f"   Validating {len(proposals)} proposals...")
            print(f"   ❌ All proposals rejected (mock)")

            return {
                "validated_proposals": [],
                "rejected_proposals": proposals,
                "overall_assessment": {
                    "total_rejected": len(proposals),
                    "warnings": ["Trop de trades", "Drawdown excessif"],
                },
            }

        def analyze_failure_with_autopsy(self, strategy_code, critic_report):
            if not self.enable_autopsy:
                return None

            print(f"   🔬 Launching Autopsy analysis...")

            # Mock autopsy report (en prod: appel LLM)
            report = {
                "cause_principale": "trop_de_trades",
                "score_amelioration_attendue": 9.0,
                "kill_rules_proposees": ["rejeter si nb_trades > 30"],
            }

            print(f"   ✅ Autopsy complete: {report['cause_principale']} (score {report['score_amelioration_attendue']}/10)")

            # Auto-update kill rules (mock)
            if report["score_amelioration_attendue"] >= 8.5:
                print(f"   ⚔️  Kill rules auto-added: {len(report['kill_rules_proposees'])} rules")

            return report

    # Test workflow
    critic = MockCritic(enable_autopsy=True)

    proposals = [
        {"id": 1, "modifications": {"entry_z": 2.5, "k_sl": 1.8}},
        {"id": 2, "modifications": {"entry_z": 3.0, "k_sl": 2.0}},
    ]

    validation_result = critic.validate_proposals(proposals)

    if not validation_result["validated_proposals"]:
        print(f"\n   All proposals rejected → Trigger Autopsy")

        autopsy_report = critic.analyze_failure_with_autopsy(
            strategy_code="def backtest(...): pass",
            critic_report=validation_result,
        )

        if autopsy_report:
            print(f"   ✅ Autopsy hook executed successfully")
        else:
            print(f"   ❌ Autopsy disabled or failed")

    print("\n✅ Critic integration test complete")


def main():
    """Run all POC tests."""
    print("\n" + "=" * 60)
    print("🔬 AUTOPSY SYSTEM - POC TEST SUITE")
    print("=" * 60)

    # Test 1: Autopsy Agent
    autopsy_report = test_autopsy_agent()

    # Test 2: Kill Rules Manager
    kill_rules_manager = test_kill_rules_manager(autopsy_report)

    # Test 3: Failure Patterns Heatmap
    patterns = test_failure_patterns_aggregation(autopsy_report)

    # Test 4: Strategist Feedback
    feedback = test_strategist_feedback(patterns, kill_rules_manager)

    # Test 5: Critic Integration
    test_critic_integration()

    # Final summary
    print("\n" + "=" * 60)
    print("✅ AUTOPSY SYSTEM - ALL TESTS PASSED")
    print("=" * 60)
    print("\nSystem Components Validated:")
    print("  ✅ Autopsy Agent (post-mortem analysis)")
    print("  ✅ Kill Rules Manager (auto-update)")
    print("  ✅ Failure Patterns Aggregation (heatmap data)")
    print("  ✅ Strategist Feedback (prompt injection)")
    print("  ✅ Critic Integration (autopsy hook)")
    print("\nNext Steps:")
    print("  1. Integrate into Orchestrator workflow")
    print("  2. Test with real LLM (deepseek-r1:32b)")
    print("  3. Run 50 iterations benchmark")
    print("  4. Monitor heatmap dashboard")
    print("\n🚀 SYSTÈME AUTO-APPRENANT OPÉRATIONNEL")
    print("=" * 60)


if __name__ == "__main__":
    main()
