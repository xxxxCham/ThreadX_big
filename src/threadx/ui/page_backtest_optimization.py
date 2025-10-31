"""
ThreadX - Page Backtest & Optimisation
=======================================

Page fusionnée combinant le backtest simple et l'optimisation Sweep.
Interface organisée en onglets pour une navigation intuitive.

Author: ThreadX Framework
Version: 2.0.0 - UI Redesign
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .backtest_bridge import BacktestResult, run_backtest, run_backtest_gpu
from .system_monitor import get_global_monitor
from .fast_sweep import fast_parameter_sweep, get_strategy_function
from ..data_access import load_ohlcv
from .strategy_registry import (
    base_params_for,
    list_strategies,
    resolve_range,
    tunable_parameters_for,
)
from threadx.indicators.bank import IndicatorBank, IndicatorSettings
from threadx.optimization.engine import SweepRunner
from threadx.optimization.scenarios import ScenarioSpec


def _require_configuration() -> Dict[str, Any]:
    """Vérifie que la configuration est complète."""
    required_keys = ("symbol", "timeframe", "start_date", "end_date", "strategy")
    missing = [key for key in required_keys if key not in st.session_state]

    if missing:
        st.warning(
            f"⚠️ Configuration incomplète. "
            f"Veuillez d'abord configurer : {', '.join(missing)}"
        )
        st.info("👈 Allez sur la page **Configuration & Stratégie** pour commencer.")
        st.stop()

    data_frame = st.session_state.get("data")
    if not isinstance(data_frame, pd.DataFrame) or data_frame.empty:
        st.warning("⚠️ Aucune donnée chargée.")
        st.info("👈 Retournez sur **Configuration & Stratégie** et cliquez sur 'Charger & Prévisualiser'.")
        st.stop()

    return {key: st.session_state[key] for key in required_keys}


def _render_config_badge(context: Dict[str, Any]) -> None:
    """Affiche un badge récapitulatif de la configuration."""
    st.info(
        f"📊 **{context['symbol']}** @ {context['timeframe']} | "
        f"📅 {context['start_date']} → {context['end_date']} | "
        f"⚙️ {context['strategy']}"
    )


def _render_price_chart(df: pd.DataFrame, indicators: Dict[str, Dict[str, Any]]) -> None:
    """Graphique OHLC avec indicateurs."""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
        )
    )

    # Bollinger Bands si configuré
    bollinger = indicators.get("bollinger", {})
    if {"window", "std"} <= set(bollinger.keys()) and not df["close"].empty:
        window = int(bollinger["window"])
        std_mult = float(bollinger["std"])
        rolling_close = df["close"].rolling(window, min_periods=window)
        mid = rolling_close.mean()
        std = rolling_close.std()

        fig.add_trace(go.Scatter(
            x=df.index, y=mid, name="BB Mid",
            mode="lines", line=dict(color='#ffa726', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=mid + std_mult * std, name="BB Upper",
            mode="lines", line=dict(color='#42a5f5', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=mid - std_mult * std, name="BB Lower",
            mode="lines", line=dict(color='#42a5f5', width=1, dash='dash')
        ))

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Prix (USD)",
        xaxis=dict(rangeslider=dict(visible=False), gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a8b2d1', size=11),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="backtest_chart")


def _render_equity_curve(equity: pd.Series) -> None:
    """Courbe d'équité moderne."""
    if equity.empty:
        st.warning("⚠️ Courbe d'équité vide.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='Équité',
        line=dict(color='#26a69a', width=2),
        fill='tozeroy',
        fillcolor='rgba(38, 166, 154, 0.1)',
    ))

    # Ligne initiale
    fig.add_hline(y=equity.iloc[0], line_dash="dash",
                  line_color="gray", opacity=0.5,
                  annotation_text="Capital initial",
                  annotation_position="right")

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Équité ($)",
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a8b2d1', size=11),
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True, key="equity_curve")


def _render_metrics(metrics: Dict[str, Any]) -> None:
    """Métriques de performance en cartes."""
    if not metrics:
        st.info("ℹ️ Aucune métrique calculée.")
        return

    # Organiser métriques en colonnes
    metrics_list = list(metrics.items())
    n_metrics = len(metrics_list)
    n_cols = min(4, n_metrics)

    # Afficher en grille
    for i in range(0, n_metrics, n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            if i + j < n_metrics:
                key, value = metrics_list[i + j]
                with col:
                    formatted = f"{value:.4f}" if isinstance(value, float) else value
                    # Couleur delta basée sur valeur
                    delta_color = "normal"
                    if isinstance(value, (int, float)):
                        delta_color = "normal" if value > 0 else "inverse"

                    st.metric(
                        label=key.replace("_", " ").title(),
                        value=formatted,
                    )

    # Bouton export
    st.markdown("")
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Métrique", "Valeur"])
    csv = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Exporter les métriques (CSV)",
        csv,
        "metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )



def _build_sweep_grid(min_value: float, max_value: float, step: float, value_type: str) -> np.ndarray:
    """Crée une grille de valeurs pour le sweep en gérant int/float proprement."""
    if max_value < min_value:
        min_value, max_value = max_value, min_value

    if value_type == "int":
        min_int = int(round(min_value))
        max_int = int(round(max_value))
        step_int = max(1, int(round(step)))
        return np.arange(min_int, max_int + step_int, step_int, dtype=int)

    step_float = float(step) if step else 0.1
    if step_float <= 0:
        step_float = 0.1

    span = max_value - min_value
    if span <= 0:
        return np.array([min_value], dtype=float)

    count = int(round(span / step_float)) + 1
    values = min_value + np.arange(count) * step_float
    values = values[values <= max_value + step_float * 1e-6]
    if len(values) == 0 or values[-1] < max_value:
        values = np.append(values, max_value)
    return np.round(values, 8)


def _render_monte_carlo_tab() -> None:
    """Onglet d'optimisation Monte-Carlo."""
    st.markdown("### 🎲 Optimisation Monte-Carlo")

    context = _require_configuration()
    data = st.session_state.get("data")

    if not isinstance(data, pd.DataFrame) or data.empty:
        st.warning("⚠️ Chargez d'abord des données.")
        return

    strategies = list_strategies()
    if not strategies:
        st.error("❌ Aucune stratégie disponible.")
        return

    _render_config_badge(context)

    st.markdown("#### Configuration Monte-Carlo")
    col_strategy, col_gpu, col_multigpu, col_workers = st.columns(4)

    with col_strategy:
        strategy = st.selectbox(
            "Stratégie",
            strategies,
            index=strategies.index(context["strategy"]) if context["strategy"] in strategies else 0,
            key="mc_strategy"
        )

    with col_gpu:
        use_gpu = st.checkbox("Activer GPU",
                               value=st.session_state.get("mc_use_gpu", True),
                               key="mc_use_gpu")

    with col_multigpu:
        use_multigpu = st.checkbox("Multi-GPU (5090+2060)",
                                    value=st.session_state.get("mc_use_multigpu", True),
                                    key="mc_use_multigpu")

    with col_workers:
        # Récupérer la sélection précédente depuis session_state
        current_mode = st.session_state.get("mc_workers_mode", "Auto (Dynamique)")
        mode_index = 1 if current_mode == "Manuel" else 0

        workers_mode = st.selectbox("Workers", ["Auto (Dynamique)", "Manuel"],
                                     index=mode_index,
                                     key="mc_workers_mode")
        if workers_mode == "Manuel":
            max_workers = st.number_input("Nb Workers", min_value=2, max_value=32,
                                         value=st.session_state.get("mc_manual_workers", 8),
                                         step=1, key="mc_manual_workers")
        else:
            max_workers = None

    tunable_specs = tunable_parameters_for(strategy)
    if not tunable_specs:
        st.info("ℹ️ Aucun paramètre optimisable pour cette stratégie.")
        return

    configured_params = st.session_state.get("strategy_params", {}) or {}
    base_strategy_params = base_params_for(strategy)

    range_preferences = st.session_state.get("strategy_param_ranges", {}).copy()
    st.markdown("##### Plages de paramètres")
    param_ranges: Dict[str, Tuple[float, float]] = {}
    param_types: Dict[str, str] = {}

    for key, spec in tunable_specs.items():
        label = spec.get('label') or key.replace('_', ' ').title()
        param_type = spec.get('type') or ('float' if isinstance(spec.get('default'), float) else 'int')
        param_types[key] = param_type

        default_val = configured_params.get(key, spec.get('default'))
        if default_val is None:
            default_val = base_strategy_params.get(key, 0 if param_type == 'int' else 0.0)

        min_val = spec.get('min')
        max_val = spec.get('max')
        opt_min, opt_max = resolve_range(spec)
        if min_val is None:
            min_val = opt_min if opt_min is not None else default_val
        if max_val is None:
            max_val = opt_max if opt_max is not None else default_val
        if min_val is None:
            min_val = 0 if param_type == 'int' else 0.0
        if max_val is None or max_val <= min_val:
            max_val = min_val + (spec.get('step') or (1 if param_type == 'int' else 0.1))

        stored_range = range_preferences.get(key)

        # Créer 2 colonnes: plage + sensibilité (Monte-Carlo)
        col_range, col_sense = st.columns([3, 1])

        with col_range:
            if param_type == 'int':
                min_val = int(round(min_val))
                max_val = int(round(max_val))
                if stored_range:
                    stored_low, stored_high = map(int, stored_range)
                    default_tuple = (max(min_val, stored_low), min(max_val, stored_high))
                else:
                    default_tuple = (min(int(default_val), max_val), max(int(default_val), min_val)) if isinstance(default_val, (int, float)) else (min_val, max_val)
                selected_range = st.slider(label, min_value=min_val, max_value=max_val, value=(int(default_tuple[0]), int(default_tuple[1])), step=1, key=f"mc_range_{key}")
            else:
                min_val = float(min_val)
                max_val = float(max_val)
                float_step = float(spec.get('step') or 0.1)
                if stored_range:
                    stored_low = float(stored_range[0])
                    stored_high = float(stored_range[1])
                    default_tuple = (max(min_val, stored_low), min(max_val, stored_high))
                else:
                    if default_val is not None:
                        default_min = float(default_val) - 0.1 * abs(float(default_val))
                        default_max = float(default_val) + 0.1 * abs(float(default_val))
                        default_tuple = (max(min_val, default_min), min(max_val, default_max))
                    else:
                        default_tuple = (min_val, max_val)
                selected_range = st.slider(label, min_value=min_val, max_value=max_val, value=(float(default_tuple[0]), float(default_tuple[1])), step=float_step, key=f"mc_range_{key}")

        # Sensibilité : Slider de STEP (granularité d'exploration) - Monte-Carlo
        with col_sense:
            range_span = selected_range[1] - selected_range[0]
            base_step = float(spec.get('step') or 0.1)

            if param_type == 'int':
                # Pour entiers : step de 1 à range_span
                step_val_display = st.slider(
                    "📊 Step",
                    min_value=1,
                    max_value=max(1, int(range_span)),
                    value=1,
                    step=1,
                    key=f"mc_step_{key}",
                    label_visibility="collapsed"
                )
            else:
                # Pour floats : step de (base_step/10) à range_span
                step_val_display = st.slider(
                    "📊 Step",
                    min_value=base_step / 10,
                    max_value=range_span if range_span > 0 else base_step,
                    value=base_step,
                    step=base_step / 100,
                    format="%.4f",
                    key=f"mc_step_{key}",
                    label_visibility="collapsed"
                )

        range_preferences[key] = (selected_range[0], selected_range[1])
        param_ranges[key] = (selected_range[0], selected_range[1])

        # Display combination count for this parameter (Monte-Carlo)
        range_min, range_max = selected_range
        span = range_max - range_min

        if param_type == 'int':
            # For integers: count the values in the range with this step
            n_combinations = len(range(int(range_min), int(range_max) + 1, max(1, int(step_val_display))))
        else:
            # For floats: (span / step)
            n_combinations = span / step_val_display if step_val_display > 0 else 1

        # Show the combination count
        comb_text = f"📊 Plage: {range_min} → {range_max} | Sensibilité: {step_val_display} | Combinaisons: {n_combinations:.1f}"
        st.caption(comb_text)

    st.session_state["strategy_param_ranges"] = range_preferences
    st.markdown("##### Paramètres d'échantillonnage")
    col_count, col_seed = st.columns(2)
    with col_count:
        n_scenarios = st.number_input("Nombre de scénarios", min_value=50, max_value=10000, value=st.session_state.get("mc_n", 500), step=50, key="mc_n")
    with col_seed:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=st.session_state.get("mc_seed", 42), step=1, key="mc_seed")

    if st.button("🎲 Lancer Monte-Carlo", type="primary", use_container_width=True, key="run_mc_btn"):
        indicator_settings = IndicatorSettings(use_gpu=use_gpu)
        indicator_bank = IndicatorBank(indicator_settings)
        runner = SweepRunner(indicator_bank=indicator_bank, max_workers=max_workers, use_multigpu=use_multigpu)

        scenario_params: Dict[str, Any] = {}
        for key, (min_v, max_v) in param_ranges.items():
            if param_types[key] == 'int':
                values = list(range(int(min_v), int(max_v) + 1))
            else:
                values = np.linspace(min_v, max_v, num=50).tolist()
            scenario_params[key] = {"values": values}

        for key, value in configured_params.items():
            if key not in scenario_params:
                scenario_params[key] = {"value": value}

        spec = ScenarioSpec(type="monte_carlo", params=scenario_params, n_scenarios=int(n_scenarios), seed=int(seed))

        # Récupérer les données réelles pour le backtest
        real_data = st.session_state.get("data")
        symbol = st.session_state.get("symbol", "BTC")
        timeframe = st.session_state.get("timeframe", "1h")

        try:
            # Lancer le Monte-Carlo avec barre de progression
            st.markdown("### 🎲 Exécution du Monte-Carlo")
            results = _run_monte_carlo_with_progress(runner, spec, real_data, symbol, timeframe, strategy, int(n_scenarios))
            st.session_state["monte_carlo_results"] = results

            # Afficher les informations de configuration
            st.markdown("---")
            st.markdown("### ⚙️ Configuration d'exécution")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Mode Multi-GPU", "Activé ✅" if use_multigpu else "Désactivé ⊘")
            with col_info2:
                actual_workers = runner.max_workers if runner.max_workers else "Auto"
                st.metric("Workers utilisés", str(actual_workers))
            with col_info3:
                st.metric("Total des résultats", len(results) if isinstance(results, pd.DataFrame) else 0)
        except Exception as exc:
            st.error(f"❌ Erreur Monte-Carlo: {exc}")
            import traceback
            st.code(traceback.format_exc())
            return

    results_df = st.session_state.get("monte_carlo_results")

    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        st.markdown("---")
        st.markdown("### 📈 Résultats Monte-Carlo")

        score_col = None
        for candidate in ["score", "objective", "sharpe", "total_return"]:
            if candidate in results_df.columns:
                score_col = candidate
                break

        if score_col:
            results_sorted = results_df.sort_values(by=score_col, ascending=False)
        else:
            results_sorted = results_df

        st.dataframe(results_sorted.head(100), use_container_width=True, height=400)

        best_row = results_sorted.iloc[0]
        st.markdown("#### 🏆 Meilleur scénario")
        st.json(best_row.to_dict())

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Exporter les résultats Monte-Carlo (CSV)",
            csv,
            "monte_carlo_results.csv",
            "text/csv",
            use_container_width=True,
        )


def _run_sweep_with_progress(runner, spec, real_data, symbol, timeframe, strategy, total_combinations):
    """Lance un sweep avec barre de progression et statistiques de vitesse."""
    import threading

    # Créer les placeholders pour l'UI
    progress_placeholder = st.empty()
    stats_cols = st.columns(4)

    # État partagé (thread-safe via GIL Python)
    shared_state = {
        "running": False,
        "current": 0,
        "total": 0,
        "start_time": time.time(),
        "should_stop": False,  # Signal d'arrêt
    }

    # Démarrer le sweep dans un thread pour ne pas bloquer Streamlit
    def run_sweep_thread():
        """Thread qui exécute le sweep (pas de Streamlit calls ici!)."""
        try:
            shared_state["running"] = True
            shared_state["start_time"] = time.time()
            results = runner.run_grid(spec, real_data, symbol, timeframe, strategy_name=strategy, reuse_cache=True)
            shared_state["results"] = results
            shared_state["error"] = None
        except Exception as e:
            # Ignorer les erreurs si arrêt demandé
            if shared_state["should_stop"]:
                shared_state["error"] = "Arrêt demandé par l'utilisateur"
                shared_state["results"] = None
            else:
                shared_state["error"] = str(e)
                shared_state["results"] = None
        finally:
            shared_state["running"] = False

    # Démarrer le sweep
    sweep_thread = threading.Thread(target=run_sweep_thread, daemon=True)
    sweep_thread.start()

    # Boucle de mise à jour UI (thread principal, synchrone avec Streamlit)
    start_time = time.time()
    status_placeholder = stats_cols[0].empty()
    speed_placeholder = stats_cols[1].empty()
    eta_placeholder = stats_cols[2].empty()
    completed_placeholder = stats_cols[3].empty()

    # Progress initial
    progress_placeholder.progress(0, text="🚀 Initialisation du Sweep...")
    status_placeholder.metric("📊 Status", "Initialisation...", delta=None)

    # Boucle: mettre à jour l'UI jusqu'à fin du sweep
    while shared_state["running"]:
        try:
            # Vérifier si l'utilisateur a demandé l'arrêt
            if st.session_state.get("run_stop_requested", False):
                shared_state["should_stop"] = True
                st.session_state.run_stop_requested = False  # Réinitialiser le flag
                progress_placeholder.progress(0, text="⏹️ Arrêt en cours...")
                status_placeholder.metric("📊 Status", "Arrêt en cours...", delta=None)
                break  # Quitter la boucle d'affichage

            if runner.total_scenarios > 0:
                current = runner.current_scenario
                total = runner.total_scenarios
                progress = min(current / total, 0.99)
                elapsed = time.time() - start_time

                if current > 0 and elapsed > 0:
                    speed = current / elapsed
                    remaining = total - current
                    eta_seconds = remaining / speed if speed > 0 else 0
                    eta_minutes, eta_secs = divmod(eta_seconds, 60)
                    eta_str = f"{int(eta_minutes)}m {int(eta_secs)}s"

                    # Mise à jour UI (thread principal)
                    progress_placeholder.progress(progress, text=f"⏳ {current}/{total} ({progress*100:.0f}%)")
                    status_placeholder.metric("📊 Status", "Exécution...", delta=None)
                    speed_placeholder.metric("🚀 Vitesse", f"{speed:.1f} tests/sec")
                    eta_placeholder.metric("⏱️ ETA", eta_str)
                    completed_placeholder.metric("📈 Complétés", f"{current}")

            time.sleep(0.1)  # Mettre à jour plus fréquemment (100ms) pour réactivité d'arrêt
        except:
            pass  # Ignorer erreurs de mise à jour

    # Attendre fin du thread
    sweep_thread.join(timeout=5)

    # Afficher résultats final
    elapsed_time = time.time() - start_time

    if shared_state["error"]:
        progress_placeholder.progress(0, text=f"❌ Erreur après {elapsed_time:.1f}s")
        status_placeholder.metric("📊 Status", "Erreur ❌", delta=None)
        st.error(f"Sweep échoué: {shared_state['error']}")
        raise Exception(shared_state["error"])

    results = shared_state.get("results")
    if results is None:
        results = pd.DataFrame()

    completed = len(results) if isinstance(results, pd.DataFrame) else 0
    tests_per_second = completed / elapsed_time if elapsed_time > 0 else 0
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{int(minutes)}m {int(seconds)}s"

    # Stats finales
    progress_placeholder.progress(1.0, text=f"✅ Sweep terminé en {time_str}")
    status_placeholder.metric("📊 Status", "Complété ✅", delta=None)
    speed_placeholder.metric("🚀 Vitesse", f"{tests_per_second:.1f} tests/sec")
    eta_placeholder.metric("⏱️ Temps Total", time_str)
    completed_placeholder.metric("📈 Résultats", f"{completed}")

    return results


def _run_monte_carlo_with_progress(runner, spec, real_data, symbol, timeframe, strategy, n_scenarios):
    """Lance un Monte-Carlo avec barre de progression et statistiques de vitesse."""
    import threading

    # Créer les placeholders pour l'UI
    progress_placeholder = st.empty()
    stats_cols = st.columns(4)

    # État partagé (thread-safe via GIL Python)
    shared_state = {
        "running": False,
        "current": 0,
        "total": 0,
        "start_time": time.time(),
        "should_stop": False,  # Signal d'arrêt
    }

    # Démarrer le Monte-Carlo dans un thread
    def run_monte_carlo_thread():
        """Thread qui exécute le Monte-Carlo (pas de Streamlit calls ici!)."""
        try:
            shared_state["running"] = True
            shared_state["start_time"] = time.time()
            results = runner.run_monte_carlo(spec, real_data, symbol, timeframe, strategy_name=strategy, reuse_cache=True)
            shared_state["results"] = results
            shared_state["error"] = None
        except Exception as e:
            # Ignorer les erreurs si arrêt demandé
            if shared_state["should_stop"]:
                shared_state["error"] = "Arrêt demandé par l'utilisateur"
                shared_state["results"] = None
            else:
                shared_state["error"] = str(e)
                shared_state["results"] = None
        finally:
            shared_state["running"] = False

    # Démarrer le Monte-Carlo
    mc_thread = threading.Thread(target=run_monte_carlo_thread, daemon=True)
    mc_thread.start()

    # Boucle de mise à jour UI (thread principal, synchrone avec Streamlit)
    start_time = time.time()
    status_placeholder = stats_cols[0].empty()
    speed_placeholder = stats_cols[1].empty()
    eta_placeholder = stats_cols[2].empty()
    completed_placeholder = stats_cols[3].empty()

    # Progress initial
    progress_placeholder.progress(0, text="🎲 Initialisation du Monte-Carlo...")
    status_placeholder.metric("📊 Status", "Initialisation...", delta=None)

    # Boucle: mettre à jour l'UI jusqu'à fin du Monte-Carlo
    while shared_state["running"]:
        try:
            # Vérifier si l'utilisateur a demandé l'arrêt
            if st.session_state.get("run_stop_requested", False):
                shared_state["should_stop"] = True
                st.session_state.run_stop_requested = False  # Réinitialiser le flag
                progress_placeholder.progress(0, text="⏹️ Arrêt en cours...")
                status_placeholder.metric("📊 Status", "Arrêt en cours...", delta=None)
                break  # Quitter la boucle d'affichage

            if runner.total_scenarios > 0:
                current = runner.current_scenario
                total = runner.total_scenarios
                progress = min(current / total, 0.99)
                elapsed = time.time() - start_time

                if current > 0 and elapsed > 0:
                    speed = current / elapsed
                    remaining = total - current
                    eta_seconds = remaining / speed if speed > 0 else 0
                    eta_minutes, eta_secs = divmod(eta_seconds, 60)
                    eta_str = f"{int(eta_minutes)}m {int(eta_secs)}s"

                    # Mise à jour UI (thread principal)
                    progress_placeholder.progress(progress, text=f"⏳ {current}/{total} ({progress*100:.0f}%)")
                    status_placeholder.metric("📊 Status", "Exécution...", delta=None)
                    speed_placeholder.metric("🚀 Vitesse", f"{speed:.1f} scén/sec")
                    eta_placeholder.metric("⏱️ ETA", eta_str)
                    completed_placeholder.metric("📈 Complétés", f"{current}")

            time.sleep(0.1)  # Mettre à jour plus fréquemment (100ms) pour réactivité d'arrêt
        except:
            pass  # Ignorer erreurs de mise à jour

    # Attendre fin du thread
    mc_thread.join(timeout=5)

    # Afficher résultats final
    elapsed_time = time.time() - start_time

    if shared_state["error"]:
        progress_placeholder.progress(0, text=f"❌ Erreur après {elapsed_time:.1f}s")
        status_placeholder.metric("📊 Status", "Erreur ❌", delta=None)
        st.error(f"Monte-Carlo échoué: {shared_state['error']}")
        raise Exception(shared_state["error"])

    results = shared_state.get("results")
    if results is None:
        results = pd.DataFrame()

    completed = len(results) if isinstance(results, pd.DataFrame) else 0
    scenarios_per_second = completed / elapsed_time if elapsed_time > 0 else 0
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{int(minutes)}m {int(seconds)}s"

    # Stats finales
    progress_placeholder.progress(1.0, text=f"✅ Monte-Carlo terminé en {time_str}")
    status_placeholder.metric("📊 Status", "Complété ✅", delta=None)
    speed_placeholder.metric("🚀 Vitesse", f"{scenarios_per_second:.1f} scén/sec")
    eta_placeholder.metric("⏱️ Temps Total", time_str)
    completed_placeholder.metric("📈 Résultats", f"{completed}")

    return results


def _format_param_value(value: float, value_type: str, decimals: int = 4) -> str:
    if value_type == "int":
        return str(int(round(value)))
    formatted = f"{value:.{decimals}f}"
    return formatted.rstrip("0").rstrip(".")


def _render_trades_table(trades: List[Dict[str, Any]]) -> None:
    """Table des transactions."""
    if not trades:
        st.info("ℹ️ Aucune transaction enregistrée.")
        return

    trades_df = pd.DataFrame(trades)

    # Formater si colonnes spécifiques existent
    if 'profit' in trades_df.columns:
        trades_df['profit'] = trades_df['profit'].apply(lambda x: f"${x:.2f}")

    st.dataframe(
        trades_df,
        use_container_width=True,
        height=300,
    )

    # Bouton export
    csv = trades_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Exporter les trades (CSV)",
        csv,
        "trades.csv",
        "text/csv",
        use_container_width=True,
    )




def _render_monitoring_section(metadata: Optional[Dict[str, Any]], history: Optional[pd.DataFrame]) -> None:
    """Affiche les diagnostics GPU/CPU et les courbes de monitoring."""
    has_metadata = isinstance(metadata, dict) and bool(metadata)
    has_history = isinstance(history, pd.DataFrame) and not history.empty

    if not has_metadata and not has_history:
        return

    st.markdown("#### 🔍 Diagnostics Système & GPU")

    if has_metadata:
        devices = metadata.get("devices_used", [])
        gpu_enabled = metadata.get("gpu_enabled", False)
        multi_gpu = metadata.get("multi_gpu_enabled", False)
        gpu_balance = metadata.get("gpu_balance", {})
        exec_time = metadata.get("execution_time_sec")
        monitor_stats = metadata.get("monitoring_stats", {})

        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.metric("GPU activé", "Oui" if gpu_enabled else "Non")
            st.metric("Multi-GPU", "Oui" if multi_gpu else "Non")
        with col_meta2:
            st.metric("Durée (s)", f"{exec_time:.2f}" if exec_time else "N/A")
            st.metric("GPU 1 moyen (%)", f"{monitor_stats.get('gpu1_mean', 0):.1f}" if monitor_stats else "N/A")
        with col_meta3:
            st.metric("GPU 2 moyen (%)", f"{monitor_stats.get('gpu2_mean', 0):.1f}" if monitor_stats else "N/A")
            st.metric("CPU moyen (%)", f"{monitor_stats.get('cpu_mean', 0):.1f}" if monitor_stats else "N/A")

        with st.expander("Détails GPU", expanded=False):
            st.write("Périphériques :", devices or "Inconnu")
            if gpu_balance:
                st.write("Balance de charge :", gpu_balance)
            if monitor_stats:
                st.json(monitor_stats)

    if has_history:
        df = history.copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time"], y=df["cpu"], name="CPU (%)", line=dict(color="#26a69a")))
        fig.add_trace(go.Scatter(x=df["time"], y=df["gpu1"], name="GPU 1 (%)", line=dict(color="#42a5f5")))
        fig.add_trace(go.Scatter(x=df["time"], y=df["gpu2"], name="GPU 2 (%)", line=dict(color="#ef5350")))
        fig.update_layout(
            height=320,
            template="plotly_dark",
            xaxis_title="Temps (s)",
            yaxis_title="Utilisation (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, key="monitoring_chart")


def _render_backtest_tab() -> None:
    """Onglet Backtest simple avec option GPU."""
    context = _require_configuration()
    indicators = st.session_state.get("indicators", {})
    params = st.session_state.get("strategy_params", {}) or {}

    _render_config_badge(context)

    st.markdown("### 🚀 Lancer le Backtest")
    col_mode, col_monitor = st.columns(2)
    with col_mode:
        use_gpu = st.checkbox("Activer le moteur GPU (BacktestEngine)",
                               value=st.session_state.get("backtest_use_gpu", True),
                               key="backtest_use_gpu")
    with col_monitor:
        monitoring = st.checkbox("Monitoring CPU/GPU en temps réel",
                                 value=st.session_state.get("backtest_monitoring", True),
                                 key="backtest_monitoring")

    if st.button("🚀 Exécuter le Backtest", type="primary", use_container_width=True, key="run_backtest_btn"):
        with st.spinner("🛠️ Exécution du backtest en cours..."):
            monitor_history = None
            try:
                df = load_ohlcv(
                    context["symbol"],
                    context["timeframe"],
                    start=context["start_date"],
                    end=context["end_date"],
                )

                if df.empty:
                    st.error("⚠️ Dataset vide pour cette plage.")
                    return

                run_params = dict(params) if isinstance(params, dict) else {}

                if use_gpu:
                    monitor = get_global_monitor() if monitoring else None
                    if monitor:
                        if monitor.is_running():
                            monitor.stop()
                        monitor.clear_history()

                    result = run_backtest_gpu(
                        df=df,
                        strategy=context["strategy"],
                        params=run_params,
                        symbol=context["symbol"],
                        timeframe=context["timeframe"],
                        use_gpu=True,
                        enable_monitoring=monitoring,
                    )

                    if monitoring:
                        monitor = get_global_monitor()
                        if monitor.is_running():
                            monitor.stop()
                        monitor_history = monitor.get_history_df()
                        monitor.clear_history()
                else:
                    result = run_backtest(df=df, strategy=context["strategy"], params=run_params)

                    monitor = get_global_monitor()
                    if monitor.is_running():
                        monitor.stop()
                    monitor.clear_history()

                st.session_state.backtest_results = result
                st.session_state.data = df
                st.session_state['monitor_history'] = monitor_history

                st.success("✅ Backtest terminé avec succès !")

            except FileNotFoundError as exc:
                st.error(f"⚠️ {exc}")
                return
            except Exception as exc:
                st.error(f"❌ Erreur lors du backtest: {exc}")
                return

    result: BacktestResult = st.session_state.get("backtest_results")
    if result:
        st.markdown("---")
        st.markdown("### 📊 Résultats du Backtest")

        res_tab1, res_tab2, res_tab3 = st.tabs(["🔍 Graphiques", "📈 Métriques", "👥 Transactions"])

        with res_tab1:
            st.markdown("#### Prix & Indicateurs")
            data_df = st.session_state.get("data")
            if isinstance(data_df, pd.DataFrame):
                _render_price_chart(data_df, indicators)

            st.markdown("#### Courbe d'équité")
            _render_equity_curve(result.equity)

            history_df = st.session_state.get("monitor_history")
            _render_monitoring_section(result.metadata, history_df)

        with res_tab2:
            _render_metrics(result.metrics)

        with res_tab3:
            _render_trades_table(result.trades)


def _render_optimization_tab() -> None:
    """Onglet d'optimisation par balayage exhaustif de paramètres (Sweep)."""
    st.markdown("### 🔬 Optimisation par Sweep (Grille Exhaustive)")

    context = _require_configuration()
    data = st.session_state.get("data")

    if not isinstance(data, pd.DataFrame) or data.empty:
        st.warning("⚠️ Chargez d'abord des données sur la page 'Chargement des Données'.")
        return

    strategies = list_strategies()
    if not strategies:
        st.error("❌ Aucune stratégie disponible.")
        return

    _render_config_badge(context)

    st.markdown("#### Configuration du Sweep")
    col_strategy, col_gpu, col_multigpu, col_workers = st.columns(4)

    with col_strategy:
        strategy = st.selectbox(
            "Stratégie à optimiser",
            strategies,
            index=strategies.index(context["strategy"]) if context["strategy"] in strategies else 0,
            key="sweep_strategy"
        )

    with col_gpu:
        use_gpu = st.checkbox("Activer GPU",
                               value=st.session_state.get("sweep_use_gpu", True),
                               key="sweep_use_gpu")

    with col_multigpu:
        use_multigpu = st.checkbox("Multi-GPU (5090+2060)",
                                    value=st.session_state.get("sweep_use_multigpu", True),
                                    key="sweep_use_multigpu")

    with col_workers:
        # Récupérer la sélection précédente depuis session_state
        current_mode = st.session_state.get("sweep_workers_mode", "Auto (Dynamique)")
        mode_index = 1 if current_mode == "Manuel" else 0

        workers_mode = st.selectbox("Workers", ["Auto (Dynamique)", "Manuel"],
                                     index=mode_index,
                                     key="sweep_workers_mode")
        if workers_mode == "Manuel":
            max_workers = st.number_input("Nb Workers", min_value=2, max_value=32,
                                         value=st.session_state.get("sweep_manual_workers", 8),
                                         step=1, key="sweep_manual_workers")
        else:
            max_workers = None

    try:
        tunable_specs = tunable_parameters_for(strategy)
    except KeyError:
        st.error(f"❌ Stratégie inconnue: {strategy}")
        return

    if not tunable_specs:
        st.info("ℹ️ Aucun paramètre optimisable pour cette stratégie.")
        return

    configured_params = st.session_state.get("strategy_params", {}) or {}
    base_strategy_params = base_params_for(strategy)

    # Configuration des plages pour TOUS les paramètres
    range_preferences = st.session_state.get("strategy_param_ranges", {}).copy()
    st.markdown("##### Plages de paramètres à optimiser")

    param_ranges: Dict[str, Tuple[float, float]] = {}
    param_types: Dict[str, str] = {}
    param_steps: Dict[str, float] = {}

    for key, spec in tunable_specs.items():
        label = spec.get('label') or key.replace('_', ' ').title()
        param_type = spec.get('type') or ('float' if isinstance(spec.get('default'), float) else 'int')
        param_types[key] = param_type

        default_val = configured_params.get(key, spec.get('default'))
        if default_val is None:
            default_val = base_strategy_params.get(key, 0 if param_type == 'int' else 0.0)

        min_val = spec.get('min')
        max_val = spec.get('max')
        step_val = spec.get('step') or (1 if param_type == 'int' else 0.1)
        opt_min, opt_max = resolve_range(spec)

        if min_val is None:
            min_val = opt_min if opt_min is not None else default_val
        if max_val is None:
            max_val = opt_max if opt_max is not None else default_val
        if min_val is None:
            min_val = 0 if param_type == 'int' else 0.0
        if max_val is None or max_val <= min_val:
            max_val = min_val + (step_val * 10)

        stored_range = range_preferences.get(key)

        # Créer 2 colonnes: plage + sensibilité
        col_range, col_sense = st.columns([3, 1])

        with col_range:
            if param_type == 'int':
                min_val = int(round(min_val))
                max_val = int(round(max_val))
                step_val = max(1, int(round(step_val)))

                if stored_range:
                    stored_low, stored_high = map(int, stored_range)
                    default_tuple = (max(min_val, stored_low), min(max_val, stored_high))
                else:
                    default_tuple = (min_val, max_val)

                selected_range = st.slider(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=(int(default_tuple[0]), int(default_tuple[1])),
                    step=1,
                    key=f"sweep_range_{key}"
                )
            else:
                min_val = float(min_val)
                max_val = float(max_val)
                step_val = float(step_val)

                if stored_range:
                    stored_low = float(stored_range[0])
                    stored_high = float(stored_range[1])
                    default_tuple = (max(min_val, stored_low), min(max_val, stored_high))
                else:
                    default_tuple = (min_val, max_val)

                selected_range = st.slider(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=(float(default_tuple[0]), float(default_tuple[1])),
                    step=step_val,
                    key=f"sweep_range_{key}"
                )

        # Sensibilité : Slider de STEP (granularité d'exploration)
        with col_sense:
            range_span = selected_range[1] - selected_range[0]

            if param_type == 'int':
                # Pour entiers : step de 1 à range_span
                step_input = st.slider(
                    "📊 Step",
                    min_value=1,
                    max_value=max(1, int(range_span)),
                    value=step_val,
                    step=1,
                    key=f"sweep_step_{key}",
                    label_visibility="collapsed"
                )
            else:
                # Pour floats : step de (step_val/10) à range_span
                step_input = st.slider(
                    "📊 Step",
                    min_value=step_val / 10,
                    max_value=range_span if range_span > 0 else step_val,
                    value=step_val,
                    step=step_val / 100,
                    format="%.4f",
                    key=f"sweep_step_{key}",
                    label_visibility="collapsed"
                )

        range_preferences[key] = (selected_range[0], selected_range[1])
        param_ranges[key] = (selected_range[0], selected_range[1])
        param_steps[key] = step_input

        # Display combination count for this parameter
        range_min, range_max = selected_range
        step = step_input
        span = range_max - range_min

        if param_type == 'int':
            # For integers: count the values in the range with this step
            n_combinations = len(range(int(range_min), int(range_max) + 1, max(1, int(step))))
        else:
            # For floats: (span / step)
            n_combinations = span / step if step > 0 else 1

        # Show the combination count
        comb_text = f"📊 Plage: {range_min} → {range_max} | Sensibilité: {step} | Combinaisons: {n_combinations:.1f}"
        st.caption(comb_text)

    st.session_state["strategy_param_ranges"] = range_preferences

    # Calculer le nombre total de combinaisons
    total_combinations = 1
    for key, (min_v, max_v) in param_ranges.items():
        step = param_steps[key]
        if param_types[key] == 'int':
            n_values = len(range(int(min_v), int(max_v) + 1, max(1, int(step))))
        else:
            # Utiliser le même calcul que np.linspace pour cohérence
            n_values = max(2, int((max_v - min_v) / step) + 1)
        total_combinations *= n_values

    # Affichage du nombre total de combinaisons
    if total_combinations <= 100000:
        st.success(f"✅ **{total_combinations} combinaisons** - Grille optimale (rapide)")
    elif total_combinations <= 1000000:
        st.info(f"📊 **{total_combinations} combinaisons** - Grille normale (quelques minutes)")
    elif total_combinations <= 3000000:
        st.warning(f"⚠️ **ATTENTION: {total_combinations} combinaisons** - Peut prendre 30-60 min avec GPU")
        st.info("💡 **Note:** Grille large mais faisable avec multi-GPU et 30 workers")
    else:
        st.error(f"❌ **BLOKÉ: {total_combinations} combinaisons trop nombreuses!**")
        st.error("🛑 Cette grille causera un MemoryError (>3M même avec GPU).")
        st.info("✨ **Solutions:**\n1. Augmentez le step (sensibilité) pour tous les paramètres\n2. Réduisez les plages (min/max)\n3. Utilisez Monte-Carlo à la place")

    # Bouton de lancement (désactivé si grille > 3 millions)
    button_disabled = total_combinations > 3000000
    if st.button("🔬 Lancer le Sweep", type="primary", use_container_width=True, key="run_sweep_btn", disabled=button_disabled):
        indicator_settings = IndicatorSettings(use_gpu=use_gpu)
        indicator_bank = IndicatorBank(indicator_settings)
        runner = SweepRunner(indicator_bank=indicator_bank, max_workers=max_workers, use_multigpu=use_multigpu)

        # Construire les paramètres pour le sweep
        scenario_params: Dict[str, Any] = {}
        for key, (min_v, max_v) in param_ranges.items():
            step = param_steps[key]
            if param_types[key] == 'int':
                values = list(range(int(min_v), int(max_v) + 1, max(1, int(step))))
            else:
                values = np.linspace(min_v, max_v, num=max(2, int((max_v - min_v) / step) + 1)).tolist()
            scenario_params[key] = {"values": values}

        # Ajouter les paramètres non-optimisés
        for key, value in configured_params.items():
            if key not in scenario_params:
                scenario_params[key] = {"value": value}

        # Utiliser run_grid pour explorer toutes les combinaisons
        spec = ScenarioSpec(type="grid", params=scenario_params)

        # Récupérer les données réelles pour le backtest
        real_data = st.session_state.get("data")
        symbol = st.session_state.get("symbol", "BTC")
        timeframe = st.session_state.get("timeframe", "1h")

        try:
            # Lancer le sweep avec barre de progression
            st.markdown("### 🚀 Exécution du Sweep")
            results = _run_sweep_with_progress(runner, spec, real_data, symbol, timeframe, strategy, total_combinations)
            st.session_state["sweep_results"] = results

            # Afficher les informations de configuration
            st.markdown("---")
            st.markdown("### ⚙️ Configuration d'exécution")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Mode Multi-GPU", "Activé ✅" if use_multigpu else "Désactivé ⊘")
            with col_info2:
                actual_workers = runner.max_workers if runner.max_workers else "Auto"
                st.metric("Workers utilisés", str(actual_workers))
            with col_info3:
                st.metric("Total des résultats", len(results) if isinstance(results, pd.DataFrame) else 0)
        except Exception as exc:
            st.error(f"❌ Erreur Sweep: {exc}")
            import traceback
            st.code(traceback.format_exc())
            return

    # Affichage des résultats
    results_df = st.session_state.get("sweep_results")

    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        st.markdown("---")
        st.markdown("### 📊 Résultats du Sweep")

        score_col = None
        for candidate in ["score", "objective", "sharpe", "total_return", "pnl"]:
            if candidate in results_df.columns:
                score_col = candidate
                break

        if score_col:
            results_sorted = results_df.sort_values(by=score_col, ascending=False)
        else:
            results_sorted = results_df

        st.dataframe(results_sorted.head(100), use_container_width=True, height=400)

        best_row = results_sorted.iloc[0]
        st.markdown("#### 🏆 Meilleure configuration")
        st.json(best_row.to_dict())

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Exporter les résultats Sweep (CSV)",
            csv,
            "sweep_results.csv",
            "text/csv",
            use_container_width=True,
        )


def _build_sweep_grid(min_value: float, max_value: float, step: float, value_type: str) -> np.ndarray:
    """Crée une grille de valeurs pour le sweep en gérant int/float proprement."""
    if max_value < min_value:
        min_value, max_value = max_value, min_value

    if value_type == "int":
        min_int = int(round(min_value))
        max_int = int(round(max_value))
        step_int = max(1, int(round(step)))
        return np.arange(min_int, max_int + step_int, step_int, dtype=int)

    step_float = float(step) if step else 0.1
    if step_float <= 0:
        step_float = 0.1

    span = max_value - min_value
    if span <= 0:
        return np.array([min_value], dtype=float)

    count = int(round(span / step_float)) + 1
    values = min_value + np.arange(count) * step_float
    values = values[values <= max_value + step_float * 1e-6]
    if len(values) == 0 or values[-1] < max_value:
        values = np.append(values, max_value)
    return np.round(values, 8)


def _format_param_value(value: float, value_type: str, decimals: int = 4) -> str:
    if value_type == "int":
        return str(int(round(value)))
    formatted = f"{value:.{decimals}f}"
    return formatted.rstrip("0").rstrip(".")


def _render_trades_table(trades: List[Dict[str, Any]]) -> None:
    """Table des transactions."""
    if not trades:
        st.info("ℹ️ Aucune transaction enregistrée.")
        return

    trades_df = pd.DataFrame(trades)

    # Formater si colonnes spécifiques existent
    if 'profit' in trades_df.columns:
        trades_df['profit'] = trades_df['profit'].apply(lambda x: f"${x:.2f}")

    st.dataframe(
        trades_df,
        use_container_width=True,
        height=300,
    )

    # Bouton export
    csv = trades_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Exporter les trades (CSV)",
        csv,
        "trades.csv",
        "text/csv",
        use_container_width=True,
    )




def _render_monitoring_section(metadata: Optional[Dict[str, Any]], history: Optional[pd.DataFrame]) -> None:
    """Affiche les diagnostics GPU/CPU et les courbes de monitoring."""
    has_metadata = isinstance(metadata, dict) and bool(metadata)
    has_history = isinstance(history, pd.DataFrame) and not history.empty

    if not has_metadata and not has_history:
        return

    st.markdown("#### 🔍 Diagnostics Système & GPU")

    if has_metadata:
        devices = metadata.get("devices_used", [])
        gpu_enabled = metadata.get("gpu_enabled", False)
        multi_gpu = metadata.get("multi_gpu_enabled", False)
        gpu_balance = metadata.get("gpu_balance", {})
        exec_time = metadata.get("execution_time_sec")
        monitor_stats = metadata.get("monitoring_stats", {})

        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.metric("GPU activé", "Oui" if gpu_enabled else "Non")
            st.metric("Multi-GPU", "Oui" if multi_gpu else "Non")
        with col_meta2:
            st.metric("Durée (s)", f"{exec_time:.2f}" if exec_time else "N/A")
            st.metric("GPU 1 moyen (%)", f"{monitor_stats.get('gpu1_mean', 0):.1f}" if monitor_stats else "N/A")
        with col_meta3:
            st.metric("GPU 2 moyen (%)", f"{monitor_stats.get('gpu2_mean', 0):.1f}" if monitor_stats else "N/A")
            st.metric("CPU moyen (%)", f"{monitor_stats.get('cpu_mean', 0):.1f}" if monitor_stats else "N/A")

        with st.expander("Détails GPU", expanded=False):
            st.write("Périphériques :", devices or "Inconnu")
            if gpu_balance:
                st.write("Balance de charge :", gpu_balance)
            if monitor_stats:
                st.json(monitor_stats)

    if has_history:
        df = history.copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time"], y=df["cpu"], name="CPU (%)", line=dict(color="#26a69a")))
        fig.add_trace(go.Scatter(x=df["time"], y=df["gpu1"], name="GPU 1 (%)", line=dict(color="#42a5f5")))
        fig.add_trace(go.Scatter(x=df["time"], y=df["gpu2"], name="GPU 2 (%)", line=dict(color="#ef5350")))
        fig.update_layout(
            height=320,
            template="plotly_dark",
            xaxis_title="Temps (s)",
            yaxis_title="Utilisation (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, key="monitoring_chart")


def main() -> None:
    """Point d'entrée de la page Optimisation."""
    st.title("🔬 Optimisation de Stratégies")
    # Unified run-state across UI
    if "run_active" not in st.session_state:
        st.session_state.run_active = False
    if "run_kind" not in st.session_state:
        st.session_state.run_kind = None
    if "run_stop_requested" not in st.session_state:
        st.session_state.run_stop_requested = False
    if "current_runner" not in st.session_state:
        st.session_state.current_runner = None

    # Global Stop control in sidebar
    with st.sidebar:
        if st.button("⏹ Arrêter l'exécution", use_container_width=True, key="global_stop_btn"):
            st.session_state.run_stop_requested = True
            try:
                from threadx.optimization.engine import request_global_stop
                request_global_stop()
            except Exception:
                pass
            st.warning("Arrêt demandé — tentative d'interruption des tâches en cours.")
    st.markdown("*Optimisez vos paramètres de trading avec Sweep ou Monte-Carlo*")
    st.markdown("---")

    # Onglets principaux (Backtest Simple supprimé)
    tab1, tab2 = st.tabs(["🔬 Sweep", "🎲 Monte-Carlo"])

    with tab1:
        _render_optimization_tab()

    with tab2:
        _render_monte_carlo_tab()


if __name__ == "__main__":
    main()
