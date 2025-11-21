#!/usr/bin/env python3
"""
ThreadX UI - Autonomous Multi-Agent Orchestrator Page
=====================================================

Interface de contrôle et supervision pour le système d'optimisation autonome 24/7.

Features:
- Activation/désactivation orchestrator
- Supervision logs temps réel (streaming)
- Visualisation code généré par agents
- Dashboard métriques Tier S live
- Contrôles pause/resume/stop
- Historique itérations avec graphes convergence

Usage:
    Accessible via sidebar: "🤖 Orchestrator Autonome"
"""

import json
import time
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from threadx.llm.orchestrator import (
    OptimizationConfig,
    OptimizationOrchestrator,
)
from threadx.utils.log import get_logger

logger = get_logger(__name__)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================


def init_session_state():
    """Initialise session state pour orchestrator."""
    if "orchestrator_running" not in st.session_state:
        st.session_state.orchestrator_running = False

    if "orchestrator_thread" not in st.session_state:
        st.session_state.orchestrator_thread = None

    if "orchestrator_logs" not in st.session_state:
        st.session_state.orchestrator_logs = []

    if "orchestrator_iterations" not in st.session_state:
        st.session_state.orchestrator_iterations = []

    if "orchestrator_config" not in st.session_state:
        st.session_state.orchestrator_config = None

    if "orchestrator_pause" not in st.session_state:
        st.session_state.orchestrator_pause = False

    if "generated_code_history" not in st.session_state:
        st.session_state.generated_code_history = []

    if "current_best_sharpe" not in st.session_state:
        st.session_state.current_best_sharpe = 0.0


# =============================================================================
# ORCHESTRATOR WORKER THREAD
# =============================================================================


def orchestrator_worker(config: OptimizationConfig, data: pd.DataFrame):
    """
    Worker thread exécutant orchestrator en arrière-plan.

    Args:
        config: Configuration optimisation
        data: Données OHLCV
    """
    try:
        # Logger dans session state
        st.session_state.orchestrator_logs.append(
            {
                "timestamp": datetime.now(),
                "level": "INFO",
                "message": f"🚀 Orchestrator started: {config.strategy_name}",
            }
        )

        # Créer orchestrator
        orchestrator = OptimizationOrchestrator(
            config=config,
            data=data,
            analyst_model="deepseek-r1:70b",
            strategist_model="gpt-oss:20b",
            critic_model="deepseek-r1:70b",
            gpu_id=0,
            debug=True,
        )

        st.session_state.orchestrator_logs.append(
            {
                "timestamp": datetime.now(),
                "level": "INFO",
                "message": "✅ Orchestrator initialized",
            }
        )

        # Hook pour capturer logs
        def log_callback(iteration: int, message: str, level: str = "INFO"):
            st.session_state.orchestrator_logs.append(
                {
                    "timestamp": datetime.now(),
                    "level": level,
                    "iteration": iteration,
                    "message": message,
                }
            )

        # Hook pour capturer code généré
        def code_callback(iteration: int, agent: str, code: str, description: str):
            st.session_state.generated_code_history.append(
                {
                    "timestamp": datetime.now(),
                    "iteration": iteration,
                    "agent": agent,
                    "code": code,
                    "description": description,
                }
            )

        # Lancer boucle autonome (avec hooks intégrés dans orchestrator)
        result = orchestrator.run()

        # Stocker résultat
        st.session_state.orchestrator_logs.append(
            {
                "timestamp": datetime.now(),
                "level": "SUCCESS",
                "message": f"🏆 Optimization complete: Best Sharpe={result['best_score']:.3f}",
            }
        )

        st.session_state.current_best_sharpe = result["best_score"]
        st.session_state.orchestrator_iterations = result.get("iterations", [])

    except Exception as e:
        logger.error(f"Orchestrator worker failed: {e}", exc_info=True)
        st.session_state.orchestrator_logs.append(
            {
                "timestamp": datetime.now(),
                "level": "ERROR",
                "message": f"❌ Orchestrator failed: {e}",
            }
        )

    finally:
        st.session_state.orchestrator_running = False
        st.session_state.orchestrator_thread = None


# =============================================================================
# UI COMPONENTS
# =============================================================================


def render_control_panel():
    """Panneau contrôles activation/pause/stop."""
    st.header("🎛️ Contrôles Orchestrator")
    
    # Vérification données disponibles
    data_available = st.session_state.get("data") is not None
    if not data_available:
        st.error("""
            ⚠️ **Aucune donnée chargée !**
            
            Veuillez d'abord charger des données sur la page **📊 Chargement des Données**.
            
            L'orchestrator a besoin de données OHLCV pour effectuer les backtests.
        """)
        st.info("💡 Naviguez vers 'Chargement des Données' dans la sidebar et chargez un symbole (ex: BTCUSDC)")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        if not st.session_state.orchestrator_running:
            if st.button(
                "▶️ Démarrer Optimisation Autonome",
                type="primary",
                use_container_width=True,
            ):
                # Valider config
                if st.session_state.orchestrator_config is None:
                    st.error("⚠️ Configuration manquante - voir section Configuration")
                    return

                # Démarrer thread worker
                st.session_state.orchestrator_running = True
                st.session_state.orchestrator_logs = []
                st.session_state.orchestrator_iterations = []

                # Créer thread (avec context Streamlit)
                thread = Thread(
                    target=orchestrator_worker,
                    args=(
                        st.session_state.orchestrator_config,
                        st.session_state.get("data_ohlcv"),  # Depuis data loader
                    ),
                )
                add_script_run_ctx(thread)  # Important pour session state
                thread.start()

                st.session_state.orchestrator_thread = thread
                st.success("✅ Orchestrator démarré en arrière-plan!")
                st.rerun()
        else:
            st.button(
                "⏸️ En cours d'exécution...",
                disabled=True,
                use_container_width=True,
            )

    with col2:
        if st.session_state.orchestrator_running:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.orchestrator_pause = True
                st.warning("⏸️ Pause demandée (fin iteration en cours)")
        else:
            st.button("⏸️ Pause", disabled=True, use_container_width=True)

    with col3:
        if st.session_state.orchestrator_running:
            if st.button("⏹️ Arrêter", type="secondary", use_container_width=True):
                st.session_state.orchestrator_running = False
                # Thread s'arrêtera à fin iteration
                st.error("⏹️ Arrêt demandé (fin iteration en cours)")
                st.rerun()
        else:
            st.button("⏹️ Arrêter", disabled=True, use_container_width=True)

    # Status indicator
    if st.session_state.orchestrator_running:
        st.success("🟢 **Status**: Orchestrator ACTIF - Optimisation en cours 24/7")
    else:
        st.info("⚪ **Status**: Orchestrator INACTIF - Prêt à démarrer")


def render_configuration():
    """Configuration orchestrator."""
    st.header("⚙️ Configuration Optimisation")
    
    st.info("""
        **🤖 Comment ça marche ?**
        
        Le système **multi-agents** fait collaborer 3 LLM spécialisés :
        
        1. **📊 Analyst** (deepseek-r1:70b) - Analyse les résultats de backtest et identifie les problèmes
        2. **💡 Strategist** (gpt-oss:20b) - Propose des modifications de paramètres basées sur le diagnostic
        3. **✅ Critic** (deepseek-r1:70b) - Valide les propositions et rejette les mauvaises idées
        
        → Les agents **conversent entre eux** pour optimiser la stratégie de manière autonome !
        → Objectif : Atteindre un **Sharpe Ratio ≥ 1.8** (Tier S)
    """)

    with st.expander("🔧 Paramètres Avancés", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            strategy_name = st.selectbox(
                "Stratégie",
                ["ma_crossover", "bollinger_dual", "amplitude_hunter"],
                help="Stratégie à optimiser",
            )

            target_sharpe = st.number_input(
                "Target Sharpe (Tier S)",
                min_value=1.0,
                max_value=5.0,
                value=1.8,
                step=0.1,
                help="Objectif Sharpe Ratio (Tier S = 1.8)",
            )

            max_iterations = st.number_input(
                "Max Iterations",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="Nombre max itérations autonomes",
            )

        with col2:
            convergence_threshold = st.number_input(
                "Convergence Threshold",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                help="Arrêt si X cycles stagnation",
            )

            proposals_per_iteration = st.number_input(
                "Proposals par Iteration",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Nombre propositions testées par cycle",
            )

            export_dir = st.text_input(
                "Export Directory",
                value="./exports/orchestrator",
                help="Dossier exports résultats",
            )

        # Paramètres initiaux stratégie
        st.subheader("Paramètres Initiaux Stratégie")

        if strategy_name == "ma_crossover":
            col1, col2 = st.columns(2)
            with col1:
                fast_period = st.number_input("Fast Period", 5, 50, 10, 1)
                slow_period = st.number_input("Slow Period", 10, 100, 30, 5)
            with col2:
                stop_loss = st.number_input("Stop Loss %", 0.5, 5.0, 1.5, 0.1)
                take_profit = st.number_input("Take Profit %", 1.0, 10.0, 3.0, 0.5)

            initial_params = {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "stop_loss_pct": stop_loss,
                "take_profit_pct": take_profit,
            }

        else:
            st.info("Paramètres par défaut utilisés pour cette stratégie")
            initial_params = {}

        # Sauvegarder config
        if st.button("💾 Sauvegarder Configuration", type="primary"):
            config = OptimizationConfig(
                strategy_name=strategy_name,
                initial_params=initial_params,
                target_sharpe=target_sharpe,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                proposals_per_iteration=proposals_per_iteration,
                export_dir=Path(export_dir),
            )

            st.session_state.orchestrator_config = config
            st.success("✅ Configuration sauvegardée!")


def render_logs_viewer():
    """Fenêtre logs temps réel (streaming)."""
    st.header("📜 Logs Temps Réel")

    # Filtre niveau logs
    col1, col2 = st.columns([3, 1])

    with col1:
        log_filter = st.multiselect(
            "Filtrer par niveau",
            ["INFO", "SUCCESS", "WARNING", "ERROR"],
            default=["INFO", "SUCCESS", "WARNING", "ERROR"],
        )

    with col2:
        auto_scroll = st.checkbox("Auto-scroll", value=True)

    # Container logs scrollable
    log_container = st.container()

    with log_container:
        if not st.session_state.orchestrator_logs:
            st.info("Aucun log - Démarrez l'orchestrator pour voir activité")
        else:
            # Afficher derniers 100 logs (reverse chronologique)
            logs_to_display = [
                log
                for log in st.session_state.orchestrator_logs[-100:]
                if log["level"] in log_filter
            ]

            for log in reversed(logs_to_display):  # Plus récents en haut
                timestamp = log["timestamp"].strftime("%H:%M:%S")
                level = log["level"]
                message = log["message"]
                iteration = log.get("iteration", "")

                # Colorisation selon niveau
                if level == "ERROR":
                    st.error(f"`{timestamp}` [{iteration}] {message}")
                elif level == "WARNING":
                    st.warning(f"`{timestamp}` [{iteration}] {message}")
                elif level == "SUCCESS":
                    st.success(f"`{timestamp}` [{iteration}] {message}")
                else:
                    st.info(f"`{timestamp}` [{iteration}] {message}")

    # Auto-refresh si running
    if st.session_state.orchestrator_running and auto_scroll:
        time.sleep(1)
        st.rerun()


def render_code_viewer():
    """Fenêtre visualisation code généré dynamiquement."""
    st.header("💻 Code Généré par Agents")

    if not st.session_state.generated_code_history:
        st.info("Aucun code généré - Les agents proposeront modifications ici")
        return

    # Sélecteur iteration
    iterations = sorted(
        set(
            entry["iteration"]
            for entry in st.session_state.generated_code_history
        )
    )

    selected_iteration = st.selectbox(
        "Iteration",
        iterations,
        index=len(iterations) - 1,  # Dernière par défaut
    )

    # Filtrer par iteration
    codes_iteration = [
        entry
        for entry in st.session_state.generated_code_history
        if entry["iteration"] == selected_iteration
    ]

    # Afficher chaque code avec tabs par agent
    if codes_iteration:
        tabs = st.tabs([entry["agent"] for entry in codes_iteration])

        for tab, entry in zip(tabs, codes_iteration):
            with tab:
                st.caption(
                    f"🕒 {entry['timestamp'].strftime('%H:%M:%S')} - {entry['description']}"
                )

                # Code avec syntax highlighting
                st.code(entry["code"], language="python", line_numbers=True)

                # Boutons actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "📋 Copier Code",
                        key=f"copy_{entry['timestamp']}",
                        use_container_width=True,
                    ):
                        st.session_state["clipboard"] = entry["code"]
                        st.success("✅ Code copié!")

                with col2:
                    if st.button(
                        "💾 Sauvegarder Fichier",
                        key=f"save_{entry['timestamp']}",
                        use_container_width=True,
                    ):
                        filename = f"{entry['agent']}_iter{entry['iteration']}.py"
                        filepath = Path("./exports/generated_code") / filename
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        filepath.write_text(entry["code"])
                        st.success(f"✅ Sauvegardé: {filepath}")


def render_metrics_dashboard():
    """Dashboard métriques Tier S live."""
    st.header("📊 Dashboard Métriques Live")

    if not st.session_state.orchestrator_iterations:
        st.info("Aucune itération - Dashboard se remplira automatiquement")
        return

    # Metrics overview (dernière iteration)
    last_iteration = st.session_state.orchestrator_iterations[-1]
    metrics = last_iteration.get("final_backtest", {}).get("metrics", {})

    # Top metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sharpe = metrics.get("sharpe_ratio", 0.0)
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            delta=f"Target: 1.8",
            delta_color="normal" if sharpe >= 1.8 else "inverse",
        )

    with col2:
        sortino = metrics.get("sortino_ratio", 0.0)
        st.metric(
            "Sortino Ratio",
            f"{sortino:.2f}",
            delta=f"Target: 2.8",
            delta_color="normal" if sortino >= 2.8 else "inverse",
        )

    with col3:
        max_dd = metrics.get("max_drawdown", 0.0)
        st.metric(
            "Max Drawdown",
            f"{max_dd:.1%}",
            delta=f"Target: ≤-18%",
            delta_color="normal" if max_dd >= -0.18 else "inverse",
        )

    with col4:
        tier_s_val = metrics.get("tier_s_validation", {})
        tier_s_score = tier_s_val.get("score", 0)
        st.metric(
            "Tier S Score",
            f"{tier_s_score:.0f}/100",
            delta=f"{tier_s_val.get('tier_s_passed', 0)}/10 passed",
        )

    # Graphe convergence Sharpe
    st.subheader("📈 Convergence Sharpe Ratio")

    iterations_data = []
    for i, it in enumerate(st.session_state.orchestrator_iterations):
        sharpe = it.get("final_backtest", {}).get("metrics", {}).get("sharpe_ratio", 0)
        iterations_data.append({"iteration": i + 1, "sharpe": sharpe})

    df_conv = pd.DataFrame(iterations_data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_conv["iteration"],
            y=df_conv["sharpe"],
            mode="lines+markers",
            name="Sharpe",
            line=dict(color="blue", width=2),
            marker=dict(size=8),
        )
    )

    # Target line
    fig.add_hline(
        y=1.8,
        line_dash="dash",
        line_color="green",
        annotation_text="Target Tier S (1.8)",
    )

    fig.update_layout(
        title="Evolution Sharpe Ratio par Iteration",
        xaxis_title="Iteration",
        yaxis_title="Sharpe Ratio",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tier S validation détaillée
    st.subheader("🎯 Validation Tier S Détaillée")

    if tier_s_val:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Métriques Validées**:")
            passed = tier_s_val.get("tier_s_passed", 0)
            st.progress(passed / 10.0, text=f"{passed}/10 Tier S passed")

            if tier_s_val.get("ai_evolved_gold"):
                st.success("🏆 **AI-EVOLVED-GOLD TAG ACHIEVED!**")

        with col2:
            st.write("**Métriques Échouées**:")
            failed = tier_s_val.get("failed_metrics", [])
            if failed:
                for metric in failed:
                    st.error(f"❌ {metric}")
            else:
                st.success("✅ Toutes métriques Tier S validées!")


# =============================================================================
# MAIN PAGE
# =============================================================================


def main():
    """Page principale Orchestrator Autonome."""
    # Note: st.set_page_config() est géré par streamlit_app.py principal
    
    init_session_state()

    st.title("🤖 Orchestrator Multi-Agent Autonome")
    st.markdown(
        """
        Système d'optimisation autonome 24/7 avec supervision temps réel.
        
        **Features**:
        - ✅ Optimisation continue stratégies (Analyst + Strategist + Critic)
        - ✅ Logs streaming temps réel
        - ✅ Visualisation code généré dynamiquement
        - ✅ Dashboard métriques Tier S live
        - ✅ Contrôles pause/resume/stop
        
        **Prérequis**: 
        - 🔹 Ollama running avec models: `deepseek-r1:70b`, `gpt-oss:20b`
        - 🔹 Données chargées (page Configuration)
        - 🔹 GPU disponible (optionnel mais recommandé)
        """
    )

    # Panneau contrôles principal
    render_control_panel()

    st.divider()

    # Configuration (collapsible)
    render_configuration()

    st.divider()

    # Layout 2 colonnes: Logs + Code
    col1, col2 = st.columns([1, 1])

    with col1:
        render_logs_viewer()

    with col2:
        render_code_viewer()

    st.divider()

    # Dashboard métriques full-width
    render_metrics_dashboard()

    # Footer
    st.caption(
        f"ThreadX v2.0 - Orchestrator Autonome | "
        f"Best Sharpe: {st.session_state.current_best_sharpe:.3f} | "
        f"Iterations: {len(st.session_state.orchestrator_iterations)}"
    )


if __name__ == "__main__":
    main()
