"""
Analyseur de performance pour le moteur de backtest ThreadX.
Mesure, diagnostique et génère des rapports visuels.
"""

import time
import json
import psutil
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from collections import defaultdict
import threading
from pathlib import Path


@dataclass
class ComponentMetrics:
    """Métriques d'un composant."""

    name: str
    total_time: float  # Temps total en secondes
    call_count: int
    avg_time: float  # Temps moyen par appel
    min_time: float
    max_time: float
    percent_of_total: float
    memory_delta_mb: float  # Différence mémoire avant/après
    diagnostic: str  # "optimize", "monitor", "ok"
    potential_gain: str  # Description du gain potentiel
    priority: int  # 1=haute, 2=moyenne, 3=basse
    details: Dict[str, Any]  # Infos supplémentaires


class PerformanceAnalyzer:
    """
    Analyseur de performance avec diagnostic automatique.

    Usage:
        analyzer = PerformanceAnalyzer()

        with analyzer.measure("data_loading"):
            load_data()

        with analyzer.measure("backtest"):
            run_backtest()

        report = analyzer.generate_report()
        analyzer.save_html_report("report.html")
    """

    def __init__(self):
        self.components: Dict[str, List[float]] = defaultdict(list)
        self.memory_snapshots: Dict[str, tuple] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_duration: float = 0
        self.process = psutil.Process(os.getpid())
        self._lock = threading.Lock()

    def start(self):
        """Démarre le profiling global."""
        self.start_time = time.perf_counter()

    def stop(self):
        """Arrête le profiling global."""
        self.end_time = time.perf_counter()
        if self.start_time:
            self.total_duration = self.end_time - self.start_time

    @contextmanager
    def measure(self, component_name: str, **metadata):
        """
        Context manager pour mesurer un composant.

        Args:
            component_name: Nom du composant
            **metadata: Métadonnées additionnelles à stocker
        """
        # Snapshot mémoire avant
        mem_before = self.process.memory_info().rss / 1024 / 1024  # MB

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start

            # Snapshot mémoire après
            mem_after = self.process.memory_info().rss / 1024 / 1024  # MB

            with self._lock:
                self.components[component_name].append(elapsed)
                self.memory_snapshots[component_name] = (
                    mem_before,
                    mem_after,
                    metadata,
                )

    def _diagnose_component(self, name: str, metrics: Dict) -> tuple[str, str, int]:
        """
        Diagnostique automatique d'un composant.

        Returns:
            (diagnostic, potential_gain, priority)
            diagnostic: "optimize", "monitor", "ok"
            potential_gain: Description du gain
            priority: 1 (haute), 2 (moyenne), 3 (basse)
        """
        percent = metrics["percent_of_total"]
        avg_time_ms = metrics["avg_time"] * 1000
        call_count = metrics["call_count"]

        # Critères de diagnostic
        if percent > 30:
            # Composant dominant (>30% du temps total)
            if avg_time_ms > 100:
                return (
                    "optimize",
                    f"Gain potentiel: {percent:.1f}% du temps total. Optimisation prioritaire !",
                    1,
                )
            else:
                return (
                    "optimize",
                    f"Composant très sollicité ({call_count} appels). Réduire appels ou paralléliser.",
                    1,
                )

        elif percent > 10:
            # Composant significatif (10-30%)
            if avg_time_ms > 50:
                return (
                    "monitor",
                    f"Gain modéré possible ({percent:.1f}%). Optimiser si facile.",
                    2,
                )
            elif call_count > 1000:
                return (
                    "monitor",
                    f"Nombreux appels ({call_count}). Envisager batching ou cache.",
                    2,
                )
            else:
                return (
                    "monitor",
                    f"Surveiller l'évolution ({percent:.1f}% du temps).",
                    2,
                )

        elif percent > 3:
            # Composant mineur mais visible (3-10%)
            if avg_time_ms > 100:
                return (
                    "monitor",
                    f"Temps unitaire élevé. Gain faible ({percent:.1f}%) mais possible.",
                    3,
                )
            else:
                return (
                    "ok",
                    f"Impact faible ({percent:.1f}%). Aucune action prioritaire.",
                    3,
                )

        else:
            # Composant négligeable (<3%)
            return (
                "ok",
                f"Impact négligeable ({percent:.1f}%). Aucune action nécessaire.",
                3,
            )

    def generate_report(self) -> Dict[str, Any]:
        """
        Génère un rapport complet avec diagnostic.

        Returns:
            Dict avec toutes les métriques et diagnostics
        """
        if not self.total_duration and self.start_time:
            self.stop()

        components_metrics = []

        for name, timings in self.components.items():
            total_time = sum(timings)
            call_count = len(timings)
            avg_time = total_time / call_count if call_count > 0 else 0
            min_time = min(timings) if timings else 0
            max_time = max(timings) if timings else 0
            percent = (
                (total_time / self.total_duration * 100)
                if self.total_duration > 0
                else 0
            )

            # Mémoire
            mem_before, mem_after, metadata = self.memory_snapshots.get(
                name, (0, 0, {})
            )
            memory_delta = mem_after - mem_before

            # Diagnostic automatique
            metrics_dict = {
                "percent_of_total": percent,
                "avg_time": avg_time,
                "call_count": call_count,
            }
            diagnostic, potential_gain, priority = self._diagnose_component(
                name, metrics_dict
            )

            component = ComponentMetrics(
                name=name,
                total_time=total_time,
                call_count=call_count,
                avg_time=avg_time,
                min_time=min_time,
                max_time=max_time,
                percent_of_total=percent,
                memory_delta_mb=memory_delta,
                diagnostic=diagnostic,
                potential_gain=potential_gain,
                priority=priority,
                details=metadata,
            )
            components_metrics.append(component)

        # Tri par temps total décroissant
        components_metrics.sort(key=lambda x: x.total_time, reverse=True)

        # Statistiques globales
        total_measured = sum(c.total_time for c in components_metrics)
        overhead = self.total_duration - total_measured

        report = {
            "summary": {
                "total_duration": self.total_duration,
                "total_measured": total_measured,
                "overhead": overhead,
                "overhead_percent": (
                    (overhead / self.total_duration * 100)
                    if self.total_duration > 0
                    else 0
                ),
                "component_count": len(components_metrics),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "components": [asdict(c) for c in components_metrics],
            "recommendations": self._generate_recommendations(components_metrics),
        }

        return report

    def _generate_recommendations(
        self, components: List[ComponentMetrics]
    ) -> List[Dict[str, str]]:
        """Génère des recommandations d'optimisation."""
        recommendations = []

        # Top 3 composants à optimiser
        to_optimize = [c for c in components if c.diagnostic == "optimize"]
        to_optimize.sort(key=lambda x: x.priority)

        for i, comp in enumerate(to_optimize[:3], 1):
            recommendations.append(
                {
                    "rank": i,
                    "component": comp.name,
                    "action": "Optimiser en priorité",
                    "reason": comp.potential_gain,
                    "impact": f"Potentiel: {comp.percent_of_total:.1f}% du temps total",
                }
            )

        # Composants avec nombreux appels
        high_call_count = [
            c for c in components if c.call_count > 500 and c.percent_of_total > 5
        ]
        for comp in high_call_count[:2]:
            if comp.diagnostic != "optimize":  # Éviter doublons
                recommendations.append(
                    {
                        "rank": len(recommendations) + 1,
                        "component": comp.name,
                        "action": "Réduire nombre d'appels",
                        "reason": f"{comp.call_count} appels détectés",
                        "impact": f"Batching ou cache pourrait réduire de ~{comp.percent_of_total/2:.1f}%",
                    }
                )

        # Composants avec variance élevée (min/max très différents)
        for comp in components:
            if comp.max_time > comp.min_time * 10 and comp.call_count > 10:
                recommendations.append(
                    {
                        "rank": len(recommendations) + 1,
                        "component": comp.name,
                        "action": "Investiguer variance",
                        "reason": f"Max time {comp.max_time*1000:.1f}ms vs min {comp.min_time*1000:.1f}ms",
                        "impact": "Performance incohérente détectée",
                    }
                )
                break  # Un seul suffit

        return recommendations

    def save_json_report(self, filepath: str):
        """Sauvegarde le rapport en JSON."""
        report = self.generate_report()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"📊 Rapport JSON sauvegardé: {filepath}")

    def save_html_report(self, filepath: str = "performance_report.html"):
        """Génère et sauvegarde un rapport HTML interactif."""
        report = self.generate_report()

        html = self._generate_html(report)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"🌐 Rapport HTML sauvegardé: {filepath}")

        # Ouvrir automatiquement dans le navigateur
        import webbrowser

        webbrowser.open(f"file://{os.path.abspath(filepath)}")

    def _generate_html(self, report: Dict) -> str:
        """Génère le HTML du rapport."""
        components = report["components"]
        summary = report["summary"]
        recommendations = report["recommendations"]

        # Génération des cartes de composants
        component_cards = []
        for comp in components:
            # Couleur selon diagnostic
            color_map = {
                "optimize": "#ef4444",  # Rouge
                "monitor": "#f59e0b",  # Orange
                "ok": "#10b981",  # Vert
            }
            bg_color_map = {
                "optimize": "#fee2e2",
                "monitor": "#fef3c7",
                "ok": "#d1fae5",
            }
            color = color_map.get(comp["diagnostic"], "#6b7280")
            bg_color = bg_color_map.get(comp["diagnostic"], "#f3f4f6")

            # Icône selon diagnostic
            icon_map = {
                "optimize": "🔴",
                "monitor": "🟡",
                "ok": "🟢",
            }
            icon = icon_map.get(comp["diagnostic"], "⚪")

            card = f"""
            <div class="component-card" style="border-left: 4px solid {color}; background: {bg_color}">
                <div class="component-header">
                    <h3>{icon} {comp['name']}</h3>
                    <span class="percent">{comp['percent_of_total']:.1f}%</span>
                </div>
                <div class="metrics-grid">
                    <div class="metric">
                        <span class="label">Temps total</span>
                        <span class="value">{comp['total_time']:.3f}s</span>
                    </div>
                    <div class="metric">
                        <span class="label">Appels</span>
                        <span class="value">{comp['call_count']}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Temps moyen</span>
                        <span class="value">{comp['avg_time']*1000:.2f}ms</span>
                    </div>
                    <div class="metric">
                        <span class="label">Min / Max</span>
                        <span class="value">{comp['min_time']*1000:.2f} / {comp['max_time']*1000:.2f}ms</span>
                    </div>
                </div>
                <div class="diagnostic">
                    <strong>Diagnostic:</strong> {comp['potential_gain']}
                </div>
                {f'<div class="memory">💾 Mémoire: {comp["memory_delta_mb"]:+.1f} MB</div>' if abs(comp['memory_delta_mb']) > 0.1 else ''}
            </div>
            """
            component_cards.append(card)

        # Génération des recommandations
        recommendation_items = []
        for rec in recommendations:
            recommendation_items.append(
                f"""
            <div class="recommendation">
                <div class="rec-rank">#{rec['rank']}</div>
                <div class="rec-content">
                    <h4>{rec['component']}</h4>
                    <p><strong>{rec['action']}</strong></p>
                    <p>{rec['reason']}</p>
                    <p class="impact">{rec['impact']}</p>
                </div>
            </div>
            """
            )

        # Graphique de distribution (simple bar chart CSS)
        bars = []
        for comp in components[:10]:  # Top 10
            width = min(comp["percent_of_total"], 100)
            color_map = {
                "optimize": "#ef4444",
                "monitor": "#f59e0b",
                "ok": "#10b981",
            }
            color = color_map.get(comp["diagnostic"], "#6b7280")
            bars.append(
                f"""
            <div class="bar-item">
                <div class="bar-label">{comp['name'][:30]}</div>
                <div class="bar-container">
                    <div class="bar" style="width: {width}%; background: {color}"></div>
                    <span class="bar-value">{comp['percent_of_total']:.1f}%</span>
                </div>
            </div>
            """
            )

        html_template = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThreadX Performance Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f9fafb;
            color: #111827;
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        .timestamp {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            color: #6b7280;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 10px;
        }}
        .summary-card .value {{
            font-size: 2rem;
            font-weight: 700;
            color: #111827;
        }}
        .summary-card .unit {{
            font-size: 1rem;
            color: #6b7280;
            margin-left: 5px;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .section h2 {{
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #111827;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }}
        .component-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .component-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .component-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .component-header h3 {{
            font-size: 1.2rem;
            color: #111827;
        }}
        .percent {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }}
        .metric {{
            display: flex;
            flex-direction: column;
        }}
        .metric .label {{
            font-size: 0.85rem;
            color: #6b7280;
            margin-bottom: 5px;
        }}
        .metric .value {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #111827;
        }}
        .diagnostic {{
            background: #f9fafb;
            padding: 12px;
            border-radius: 6px;
            font-size: 0.95rem;
            margin-top: 10px;
        }}
        .memory {{
            margin-top: 10px;
            font-size: 0.9rem;
            color: #6b7280;
        }}
        .recommendation {{
            display: flex;
            gap: 15px;
            padding: 15px;
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            border-radius: 6px;
            margin-bottom: 15px;
        }}
        .rec-rank {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #f59e0b;
            min-width: 40px;
        }}
        .rec-content h4 {{
            color: #111827;
            margin-bottom: 8px;
        }}
        .rec-content p {{
            margin: 5px 0;
            font-size: 0.95rem;
        }}
        .impact {{
            color: #6b7280;
            font-style: italic;
        }}
        .bar-item {{
            margin-bottom: 15px;
        }}
        .bar-label {{
            font-size: 0.9rem;
            color: #111827;
            margin-bottom: 5px;
            font-weight: 500;
        }}
        .bar-container {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .bar {{
            height: 30px;
            background: #667eea;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .bar-value {{
            font-size: 0.9rem;
            font-weight: 600;
            color: #111827;
            min-width: 50px;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 20px;
            padding: 15px;
            background: #f9fafb;
            border-radius: 6px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚡ ThreadX Performance Report</h1>
            <div class="timestamp">Généré le {summary['timestamp']}</div>
        </header>

        <div class="summary">
            <div class="summary-card">
                <h3>Durée Totale</h3>
                <div class="value">{summary['total_duration']:.3f}<span class="unit">s</span></div>
            </div>
            <div class="summary-card">
                <h3>Temps Mesuré</h3>
                <div class="value">{summary['total_measured']:.3f}<span class="unit">s</span></div>
            </div>
            <div class="summary-card">
                <h3>Overhead</h3>
                <div class="value">{summary['overhead_percent']:.1f}<span class="unit">%</span></div>
            </div>
            <div class="summary-card">
                <h3>Composants</h3>
                <div class="value">{summary['component_count']}</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Distribution du Temps</h2>
            {''.join(bars)}
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #ef4444"></div>
                    <span>🔴 À optimiser en priorité</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f59e0b"></div>
                    <span>🟡 À surveiller</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #10b981"></div>
                    <span>🟢 OK</span>
                </div>
            </div>
        </div>

        {f'''
        <div class="section">
            <h2>🎯 Recommandations Prioritaires</h2>
            {''.join(recommendation_items)}
        </div>
        ''' if recommendations else ''}

        <div class="section">
            <h2>🔍 Détails des Composants</h2>
            {''.join(component_cards)}
        </div>
    </div>
</body>
</html>
        """

        return html_template

    def print_summary(self):
        """Affiche un résumé dans la console."""
        report = self.generate_report()
        summary = report["summary"]
        components = report["components"]

        print("\n" + "=" * 80)
        print("⚡ RAPPORT DE PERFORMANCE THREADX")
        print("=" * 80)
        print(f"Durée totale: {summary['total_duration']:.3f}s")
        print(f"Temps mesuré: {summary['total_measured']:.3f}s")
        print(
            f"Overhead: {summary['overhead']:.3f}s ({summary['overhead_percent']:.1f}%)"
        )
        print(f"Composants: {summary['component_count']}")
        print("\n" + "-" * 80)
        print("TOP COMPOSANTS PAR TEMPS:")
        print("-" * 80)

        for i, comp in enumerate(components[:10], 1):
            icon_map = {"optimize": "🔴", "monitor": "🟡", "ok": "🟢"}
            icon = icon_map.get(comp["diagnostic"], "⚪")
            print(
                f"{i:2}. {icon} {comp['name']:30} {comp['percent_of_total']:6.2f}% ({comp['total_time']:.3f}s, {comp['call_count']} appels)"
            )

        if report["recommendations"]:
            print("\n" + "-" * 80)
            print("🎯 RECOMMANDATIONS:")
            print("-" * 80)
            for rec in report["recommendations"][:5]:
                print(f"{rec['rank']}. [{rec['component']}] {rec['action']}")
                print(f"   → {rec['reason']}")
                print(f"   💡 {rec['impact']}\n")

        print("=" * 80 + "\n")
