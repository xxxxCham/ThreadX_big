"""
Profiling avec cProfile pour identifier les bottlenecks.
Génère un rapport HTML interactif.
"""

import cProfile
import pstats
import io
import time
import pandas as pd
import numpy as np
from pathlib import Path

from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec

print("=" * 80)
print("🔬 PROFILING DÉTAILLÉ - cProfile")
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

# Configuration sweep
grid_spec = ScenarioSpec(
    type="grid",
    params={
        "bb_window": [10, 20, 30],
        "bb_num_std": [1.5, 2.0, 2.5],
        "atr_window": [14, 20],
        "atr_multiplier": [1.0, 1.5],
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

print("⏱️  Démarrage du profiling avec cProfile...")
print()

# Profiling
profiler = cProfile.Profile()
start = time.perf_counter()

profiler.enable()
results = runner.run_grid(
    grid_spec=grid_spec, real_data=test_data, symbol="TEST", timeframe="1h"
)
profiler.disable()

elapsed = time.perf_counter() - start

print()
print("=" * 80)
print("✅ RÉSULTATS")
print("=" * 80)
print(f"⏱️  Temps total: {elapsed:.2f}s")
print(f"📊 Combinaisons: {len(results)}")
print(f"🚀 Vitesse: {len(results) / elapsed:.1f} tests/sec")
print()

# ==================== ANALYSE STATS ====================

# Créer un stream pour les stats
s = io.StringIO()
ps = pstats.Stats(profiler, stream=s)
ps.strip_dirs()
ps.sort_stats("cumulative")

print("=" * 80)
print("📈 TOP 30 FONCTIONS PAR TEMPS CUMULATIF")
print("=" * 80)
ps.print_stats(30)
print(s.getvalue())

# ==================== GÉNÉRATION HTML ====================

# Extraire les stats pour HTML
s2 = io.StringIO()
ps2 = pstats.Stats(profiler, stream=s2)
ps2.strip_dirs()
ps2.sort_stats("cumulative")
ps2.print_stats(50)

stats_text = s2.getvalue()

# Parser les stats
import re

lines = stats_text.split("\n")
functions = []

for line in lines:
    # Format: ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    match = re.match(
        r"\s*(\d+(?:/\d+)?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)", line
    )
    if match:
        ncalls, tottime, percall_tot, cumtime, percall_cum, func = match.groups()
        functions.append(
            {
                "ncalls": ncalls,
                "tottime": float(tottime),
                "percall_tot": float(percall_tot),
                "cumtime": float(cumtime),
                "percall_cum": float(percall_cum),
                "function": func.strip(),
            }
        )

# Top 20 pour les graphiques
top20 = functions[:20]

import json

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ThreadX cProfile Analysis</title>
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
            font-size: 0.9em;
        }}
        th {{
            background: #3d3d3d;
            color: #667eea;
            font-weight: bold;
        }}
        tr:hover {{
            background: #353535;
        }}
        .function-name {{
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
        }}
        .highlight {{
            background: #4a4a00;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 ThreadX cProfile Analysis</h1>
        <p>Analyse détaillée des performances avec cProfile</p>
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
            <div class="stat-value">{len(functions)}</div>
            <div class="stat-label">Fonctions profilées</div>
        </div>
    </div>

    <div class="chart">
        <h2>⏱️ Top 20 - Temps cumulatif</h2>
        <div id="cumtimeChart"></div>
    </div>

    <div class="chart">
        <h2>🔥 Top 20 - Temps propre (tottime)</h2>
        <div id="tottimeChart"></div>
    </div>

    <div class="chart">
        <h2>📊 Détails des fonctions (Top 40)</h2>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Fonction</th>
                    <th>Appels</th>
                    <th>Temps total (s)</th>
                    <th>Par appel (ms)</th>
                    <th>Cumulatif (s)</th>
                    <th>Par appel cum (ms)</th>
                </tr>
            </thead>
            <tbody>
"""

for i, func in enumerate(functions[:40], 1):
    # Highlight ThreadX functions
    row_class = "highlight" if "threadx" in func["function"] else ""
    html_content += f"""
                <tr class="{row_class}">
                    <td>{i}</td>
                    <td class="function-name">{func["function"]}</td>
                    <td>{func["ncalls"]}</td>
                    <td>{func["tottime"]:.3f}</td>
                    <td>{func["percall_tot"]*1000:.2f}</td>
                    <td>{func["cumtime"]:.3f}</td>
                    <td>{func["percall_cum"]*1000:.2f}</td>
                </tr>
"""

html_content += f"""
            </tbody>
        </table>
    </div>

    <script>
        // Chart cumtime
        var cumData = [{{
            x: {json.dumps([f["function"][:50] for f in top20])},
            y: {json.dumps([f["cumtime"] for f in top20])},
            type: 'bar',
            marker: {{
                color: '#667eea',
                line: {{ color: '#764ba2', width: 2 }}
            }}
        }}];

        var cumLayout = {{
            paper_bgcolor: '#2d2d2d',
            plot_bgcolor: '#2d2d2d',
            font: {{ color: '#e0e0e0' }},
            xaxis: {{
                tickangle: -45,
                gridcolor: '#404040'
            }},
            yaxis: {{
                title: 'Temps cumulatif (s)',
                gridcolor: '#404040'
            }},
            height: 500,
            margin: {{ b: 200 }}
        }};

        Plotly.newPlot('cumtimeChart', cumData, cumLayout);

        // Chart tottime
        var totData = [{{
            x: {json.dumps([f["function"][:50] for f in top20])},
            y: {json.dumps([f["tottime"] for f in top20])},
            type: 'bar',
            marker: {{
                color: '#f093fb',
                line: {{ color: '#f5576c', width: 2 }}
            }}
        }}];

        var totLayout = {{
            paper_bgcolor: '#2d2d2d',
            plot_bgcolor: '#2d2d2d',
            font: {{ color: '#e0e0e0' }},
            xaxis: {{
                tickangle: -45,
                gridcolor: '#404040'
            }},
            yaxis: {{
                title: 'Temps propre (s)',
                gridcolor: '#404040'
            }},
            height: 500,
            margin: {{ b: 200 }}
        }};

        Plotly.newPlot('tottimeChart', totData, totLayout);
    </script>
</body>
</html>
"""

# Sauvegarder le rapport HTML
report_path = Path("cprofile_report.html")
report_path.write_text(html_content, encoding="utf-8")

print()
print("=" * 80)
print(f"📄 Rapport cProfile HTML généré: {report_path.absolute()}")
print("=" * 80)
print()

# Ouvrir dans le navigateur
import webbrowser

webbrowser.open(str(report_path.absolute()))
print("🌐 Rapport ouvert dans le navigateur !")
