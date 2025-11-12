"""
Profiler détaillé pour analyser les performances du sweep.
Mesure le temps de chaque composant pour identifier les bottlenecks.
"""

import time
import pandas as pd
import numpy as np
from collections import defaultdict
import json
from pathlib import Path

from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec

# ==================== INSTRUMENTATION ====================


class PerformanceProfiler:
    """Collecte les métriques de performance détaillées."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.counters = defaultdict(int)
        self.start_times = {}

    def start(self, label: str):
        """Démarre un timer pour un label."""
        self.start_times[label] = time.perf_counter()

    def stop(self, label: str):
        """Arrête le timer et enregistre la durée."""
        if label in self.start_times:
            elapsed = time.perf_counter() - self.start_times[label]
            self.timings[label].append(elapsed * 1000)  # en ms
            del self.start_times[label]

    def count(self, label: str, value: int = 1):
        """Incrémente un compteur."""
        self.counters[label] += value

    def get_stats(self):
        """Retourne les statistiques agrégées."""
        stats = {}

        for label, times in self.timings.items():
            if times:
                stats[label] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "mean_ms": np.mean(times),
                    "median_ms": np.median(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "std_ms": np.std(times),
                }

        for label, count in self.counters.items():
            if label not in stats:
                stats[label] = {}
            stats[label]["total_count"] = count

        return stats


# Profiler global
profiler = PerformanceProfiler()

# ==================== PATCH DU CODE ====================

# Patch BBAtrStrategy._ensure_indicators
from threadx.strategy.bb_atr import BBAtrStrategy

original_ensure_indicators = BBAtrStrategy._ensure_indicators


def profiled_ensure_indicators(self, df, params, precomputed_indicators=None):
    profiler.start("ensure_indicators")

    # Mesure FAST PATH vs calcul
    if precomputed_indicators:
        profiler.start("ensure_indicators.fast_path_check")

    result = original_ensure_indicators(self, df, params, precomputed_indicators)

    if precomputed_indicators:
        profiler.stop("ensure_indicators.fast_path_check")
        profiler.count("fast_path_used")
    else:
        profiler.count("indicators_calculated")

    profiler.stop("ensure_indicators")
    return result


BBAtrStrategy._ensure_indicators = profiled_ensure_indicators

# Patch BBAtrStrategy.generate_signals
original_generate_signals = BBAtrStrategy.generate_signals


def profiled_generate_signals(self, df, params):
    profiler.start("generate_signals")
    profiler.start("generate_signals.calculations")

    result = original_generate_signals(self, df, params)

    profiler.stop("generate_signals.calculations")
    profiler.stop("generate_signals")
    profiler.count("signals_generated")

    return result


BBAtrStrategy.generate_signals = profiled_generate_signals

# Patch BBAtrStrategy.backtest
original_backtest = BBAtrStrategy.backtest


def profiled_backtest(self, *args, **kwargs):
    profiler.start("backtest_full")

    result = original_backtest(self, *args, **kwargs)

    profiler.stop("backtest_full")
    profiler.count("backtests_executed")

    return result


BBAtrStrategy.backtest = profiled_backtest

# Patch SweepRunner._evaluate_single_combination
from threadx.optimization.engine import SweepRunner as SR

original_evaluate = SR._evaluate_single_combination


def profiled_evaluate(self, *args, **kwargs):
    profiler.start("evaluate_combination")
    profiler.start("evaluate_combination.strategy_creation")

    result = original_evaluate(self, *args, **kwargs)

    profiler.stop("evaluate_combination.strategy_creation")
    profiler.stop("evaluate_combination")

    return result


SR._evaluate_single_combination = profiled_evaluate

# ==================== TEST ====================

print("=" * 80)
print("🔬 PROFILER DÉTAILLÉ - ANALYSE PERFORMANCES")
print("=" * 80)
print()

# Données de test
dates = pd.date_range(start="2024-01-01", periods=5000, freq="1h")
test_data = pd.DataFrame(
    {
        "open": 50000 + np.random.randn(5000) * 100,
        "high": 50100 + np.random.randn(5000) * 100,
        "low": 49900 + np.random.randn(5000) * 100,
        "close": 50000 + np.random.randn(5000) * 100,
        "volume": np.random.randint(1000, 10000, 5000),
    },
    index=dates,
)

# Configuration sweep (plus petit pour analyse détaillée)
grid_spec = ScenarioSpec(
    type="grid",
    params={
        "bb_window": [10, 20, 30],  # 3 valeurs
        "bb_num_std": [1.5, 2.0, 2.5],  # 3 valeurs
        "atr_window": [14, 20],  # 2 valeurs
        "atr_multiplier": [1.0, 1.5],  # 2 valeurs
    },
    sampler="grid",
)

total_combos = 3 * 3 * 2 * 2  # = 36 combinaisons

print(f"📊 Configuration:")
print(f"   - Données: {len(test_data)} barres")
print(f"   - Combinaisons: {total_combos}")
print()

# Runner
runner = SweepRunner(max_workers=8)

print("⏱️  Démarrage du profiling...")
start = time.perf_counter()

# Exécution
results = runner.run_grid(
    grid_spec=grid_spec, real_data=test_data, symbol="TEST", timeframe="1h"
)

elapsed = time.perf_counter() - start

print()
print("=" * 80)
print("✅ RÉSULTATS")
print("=" * 80)
print(f"⏱️  Temps total: {elapsed:.2f}s")
print(f"📊 Combinaisons: {len(results)}")
print(f"🚀 Vitesse: {len(results) / elapsed:.1f} tests/sec")
print()

# ==================== ANALYSE DÉTAILLÉE ====================

stats = profiler.get_stats()

print("=" * 80)
print("📈 ANALYSE DÉTAILLÉE DES PERFORMANCES")
print("=" * 80)
print()

# Trier par temps total
sorted_stats = sorted(
    [(k, v) for k, v in stats.items() if "total_ms" in v],
    key=lambda x: x[1]["total_ms"],
    reverse=True,
)

total_time_ms = elapsed * 1000

print("🔥 TOP COMPOSANTS PAR TEMPS TOTAL:")
print("-" * 80)
print(
    f"{'Composant':<40} {'Total (ms)':<12} {'% Total':<10} {'Moy (ms)':<12} {'Appels':<10}"
)
print("-" * 80)

for label, data in sorted_stats[:15]:
    pct = (data["total_ms"] / total_time_ms) * 100
    print(
        f"{label:<40} {data['total_ms']:>10.1f}  {pct:>8.1f}%  {data['mean_ms']:>10.2f}  {data['count']:>8}"
    )

print()
print("📊 STATISTIQUES PAR OPÉRATION:")
print("-" * 80)

# Regrouper par catégorie
categories = {
    "Indicateurs": ["ensure_indicators", "fast_path"],
    "Signaux": ["generate_signals"],
    "Backtest": ["backtest_full"],
    "Évaluation": ["evaluate_combination"],
}

for category, keywords in categories.items():
    print(f"\n🔹 {category}:")
    cat_total = 0
    cat_count = 0

    for label, data in sorted_stats:
        if any(kw in label for kw in keywords):
            if "total_ms" in data:
                cat_total += data["total_ms"]
                cat_count += data["count"]
                print(
                    f"   {label:<35} {data['mean_ms']:>8.2f}ms/call  ({data['count']} appels)"
                )

    if cat_count > 0:
        print(
            f"   {'TOTAL ' + category:<35} {cat_total:>8.1f}ms  ({cat_total/total_time_ms*100:.1f}% du total)"
        )

# Compteurs
print()
print("🔢 COMPTEURS:")
print("-" * 80)
for label, count in profiler.counters.items():
    print(f"   {label:<40} {count:>10}")

# Ratio FAST PATH
fast_path = profiler.counters.get("fast_path_used", 0)
calculated = profiler.counters.get("indicators_calculated", 0)
total_ind = fast_path + calculated

if total_ind > 0:
    print()
    print(f"⚡ FAST PATH: {fast_path}/{total_ind} ({fast_path/total_ind*100:.1f}%)")
else:
    print()
    print("⚡ FAST PATH: Aucun indicateur mesuré (erreurs pendant l'exécution)")

# ==================== GÉNÉRATION HTML ====================

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ThreadX Performance Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{ margin: 0; color: white; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #999;
            margin-top: 5px;
        }}
        .chart {{
            background: #2d2d2d;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        table {{
            width: 100%;
            background: #2d2d2d;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #404040;
        }}
        th {{
            background: #3d3d3d;
            color: #667eea;
            font-weight: bold;
        }}
        tr:hover {{
            background: #353535;
        }}
        .progress-bar {{
            height: 20px;
            background: #404040;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 ThreadX Performance Analysis</h1>
        <p>Analyse détaillée des performances du sweep optimization</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{len(results)}</div>
            <div class="stat-label">Combinaisons testées</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{elapsed:.2f}s</div>
            <div class="stat-label">Temps total</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(results)/elapsed:.1f}</div>
            <div class="stat-label">Tests/sec</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{fast_path/total_ind*100 if total_ind > 0 else 0:.0f}%</div>
            <div class="stat-label">FAST PATH hit rate</div>
        </div>
    </div>

    <div class="chart">
        <h2>⏱️ Répartition du temps par composant</h2>
        <div id="pieChart"></div>
    </div>

    <div class="chart">
        <h2>📊 Temps moyen par opération</h2>
        <div id="barChart"></div>
    </div>

    <div class="chart">
        <h2>📈 Détails des performances</h2>
        <table>
            <thead>
                <tr>
                    <th>Composant</th>
                    <th>Temps total (ms)</th>
                    <th>% du total</th>
                    <th>Temps moyen (ms)</th>
                    <th>Appels</th>
                    <th>Répartition</th>
                </tr>
            </thead>
            <tbody>
"""

for label, data in sorted_stats[:15]:
    pct = (data["total_ms"] / total_time_ms) * 100
    html_content += f"""
                <tr>
                    <td>{label}</td>
                    <td>{data['total_ms']:.1f}</td>
                    <td>{pct:.1f}%</td>
                    <td>{data['mean_ms']:.2f}</td>
                    <td>{data['count']}</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min(pct, 100):.1f}%"></div>
                        </div>
                    </td>
                </tr>
"""

html_content += f"""
            </tbody>
        </table>
    </div>

    <script>
        // Pie chart - Répartition du temps
        var pieData = [{{
            values: {json.dumps([v["total_ms"] for k, v in sorted_stats[:10]])},
            labels: {json.dumps([k for k, v in sorted_stats[:10]])},
            type: 'pie',
            hole: 0.4,
            marker: {{
                colors: ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe',
                         '#43e97b', '#fa709a', '#fee140', '#30cfd0', '#a8edea']
            }},
            textinfo: 'label+percent',
            textposition: 'outside'
        }}];

        var pieLayout = {{
            paper_bgcolor: '#2d2d2d',
            plot_bgcolor: '#2d2d2d',
            font: {{ color: '#e0e0e0' }},
            showlegend: true,
            height: 500
        }};

        Plotly.newPlot('pieChart', pieData, pieLayout);

        // Bar chart - Temps moyen
        var barData = [{{
            x: {json.dumps([k for k, v in sorted_stats[:15]])},
            y: {json.dumps([v["mean_ms"] for k, v in sorted_stats[:15]])},
            type: 'bar',
            marker: {{
                color: '#667eea',
                line: {{ color: '#764ba2', width: 2 }}
            }}
        }}];

        var barLayout = {{
            paper_bgcolor: '#2d2d2d',
            plot_bgcolor: '#2d2d2d',
            font: {{ color: '#e0e0e0' }},
            xaxis: {{
                tickangle: -45,
                gridcolor: '#404040'
            }},
            yaxis: {{
                title: 'Temps moyen (ms)',
                gridcolor: '#404040'
            }},
            height: 500,
            margin: {{ b: 150 }}
        }};

        Plotly.newPlot('barChart', barData, barLayout);
    </script>
</body>
</html>
"""

# Sauvegarder le rapport HTML
report_path = Path("performance_report.html")
report_path.write_text(html_content, encoding="utf-8")

print()
print("=" * 80)
print(f"📄 Rapport HTML généré: {report_path.absolute()}")
print("=" * 80)
print()
print("🌐 Ouverture dans le navigateur...")

# Ouvrir dans le navigateur
import webbrowser

webbrowser.open(f"file:///{report_path.absolute()}")
