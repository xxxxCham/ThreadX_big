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
    col_strategy, col_gpu, col_workers = st.columns(3)

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

    with col_workers:
        max_workers = st.slider("Workers", 1, 8, st.session_state.get("mc_workers", 4), key="mc_workers")

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

        range_preferences[key] = (selected_range[0], selected_range[1])
        param_ranges[key] = (selected_range[0], selected_range[1])

    st.session_state["strategy_param_ranges"] = range_preferences
    st.markdown("##### Paramètres d'échantillonnage")
    col_count, col_seed = st.columns(2)
    with col_count:
        n_scenarios = st.number_input("Nombre de scénarios", min_value=50, max_value=10000, value=st.session_state.get("mc_n", 500), step=50, key="mc_n")
    with col_seed:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=st.session_state.get("mc_seed", 42), step=1, key="mc_seed")

    if st.button("🎲 Lancer Monte-Carlo", type="primary", use_container_width=True, key="run_mc_btn"):
        with st.spinner("🎲 Génération des scénarios Monte-Carlo..."):
            indicator_settings = IndicatorSettings(use_gpu=use_gpu)
            indicator_bank = IndicatorBank(indicator_settings)
            runner = SweepRunner(indicator_bank=indicator_bank, max_workers=max_workers)

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
            try:
                results = runner.run_monte_carlo(spec, reuse_cache=True)
                st.session_state["monte_carlo_results"] = results
                st.success("✅ Monte-Carlo terminé !")
            except Exception as exc:
                st.error(f"❌ Erreur Monte-Carlo: {exc}")
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
    """Onglet Optimisation Sweep."""
    st.markdown("### 🔬 Optimisation des Paramètres (Sweep)")

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

    st.markdown("#### Configuration du Sweep")

    col1, col2, col3 = st.columns(3)

    with col1:
        strategy = st.selectbox(
            "Stratégie à optimiser",
            strategies,
            index=strategies.index(context["strategy"]) if context["strategy"] in strategies else 0,
            key="sweep_strategy"
        )

    try:
        tunable_specs = tunable_parameters_for(strategy)
    except KeyError:
        st.error(f"❌ Stratégie inconnue: {strategy}")
        return

    if not tunable_specs:
        st.info("ℹ️ Aucun paramètre numérique optimisable pour cette stratégie.")
        return

    param_options = list(tunable_specs.keys())

    def _option_label(key: str) -> str:
        spec = tunable_specs[key]
        return spec.get('label') or key.replace('_', ' ').title()

    default_param = st.session_state.get("sweep_selected_param")
    if default_param not in param_options:
        default_param = param_options[0]

    with col2:
        param_name = st.selectbox(
            "Paramètre à optimiser",
            param_options,
            index=param_options.index(default_param),
            format_func=_option_label,
            key="sweep_param"
        )
    st.session_state["sweep_selected_param"] = param_name

    with col3:
        capital_initial = st.number_input(
            "Capital initial (€)",
            min_value=100,
            max_value=1_000_000,
            value=int(st.session_state.get("sweep_capital", 10000)),
            step=1000,
            key="sweep_capital",
            help="Capital de départ pour calculer le PnL en euros"
        )

    param_spec = tunable_specs[param_name]
    param_label = param_spec.get('label') or param_name.replace('_', ' ').title()
    param_type = param_spec.get('type') or ('float' if isinstance(param_spec.get('default'), float) else 'int')

    configured_params = st.session_state.get("strategy_params", {}) or {}
    range_preferences = dict(st.session_state.get("strategy_param_ranges", {}) or {})
    base_strategy_params = base_params_for(strategy)

    default_value = configured_params.get(param_name, param_spec.get("default"))
    stored_range = range_preferences.get(param_name)
    if default_value is None:
        default_value = base_strategy_params.get(param_name)
    if default_value is None:
        default_value = 0 if param_type == 'int' else 0.0

    slider_min = param_spec.get("min")
    slider_max = param_spec.get("max")
    opt_min, opt_max = resolve_range(param_spec)

    if slider_min is None:
        slider_min = opt_min if opt_min is not None else default_value
    if slider_max is None:
        slider_max = opt_max if opt_max is not None else default_value

    if slider_min is None:
        slider_min = 0 if param_type == 'int' else 0.0
    if slider_max is None or slider_max <= slider_min:
        slider_max = slider_min + (param_spec.get('step') or (1 if param_type == 'int' else 0.1))

    default_min = max(slider_min, opt_min) if opt_min is not None else max(slider_min, default_value)
    default_max = min(slider_max, opt_max) if opt_max is not None else min(slider_max, default_value)
    if default_max < default_min:
        default_max = default_min

    if stored_range:
        stored_low, stored_high = stored_range
        if param_type == 'int':
            stored_low = int(round(stored_low))
            stored_high = int(round(stored_high))
        else:
            stored_low = float(stored_low)
            stored_high = float(stored_high)
        default_min = max(default_min, stored_low)
        default_max = min(default_max, stored_high)
        if default_max < default_min:
            default_min, default_max = stored_low, stored_high

    step_default = param_spec.get('step') or (1 if param_type == 'int' else 0.05)
    if step_default <= 0:
        step_default = 1 if param_type == 'int' else 0.05

    col4, col5 = st.columns(2)

    if param_type == 'int':
        slider_min_int = int(round(slider_min))
        slider_max_int = int(round(slider_max))
        default_min_int = int(round(default_min))
        default_max_int = int(round(default_max))
        step_default_int = max(1, int(round(step_default)))

        with col4:
            min_val, max_val = st.slider(
                "Plage de valeurs",
                min_value=slider_min_int,
                max_value=slider_max_int,
                value=(default_min_int, default_max_int),
                step=step_default_int,
                key="sweep_range"
            )

        with col5:
            max_step = max(1, max_val - min_val)
            step = st.number_input(
                "Pas d'incrémentation",
                min_value=1,
                max_value=max_step,
                value=step_default_int,
                step=1,
                key="sweep_step"
            )

        decimals = 0
    else:
        slider_min_float = float(slider_min)
        slider_max_float = float(slider_max)
        default_min_float = float(default_min)
        default_max_float = float(default_max)
        step_default_float = float(step_default)

        with col4:
            min_val, max_val = st.slider(
                "Plage de valeurs",
                min_value=slider_min_float,
                max_value=slider_max_float,
                value=(default_min_float, default_max_float),
                step=step_default_float,
                key="sweep_range"
            )

        with col5:
            step = st.number_input(
                "Pas d'incrémentation",
                min_value=max(step_default_float / 10, 0.0001),
                value=float(step_default_float),
                step=float(step_default_float),
                format="%.4f",
                key="sweep_step"
            )

        step_str = f"{float(step):.8f}".rstrip("0")
        decimals = len(step_str.split(".")[1]) if "." in step_str else 2
        decimals = min(max(decimals, 1), 6)

    if param_type == 'int':
        step = int(step) if step else 1
        if step <= 0:
            step = 1
    else:
        step = float(step) if step else step_default
        if step <= 0:
            step = step_default

    grid = _build_sweep_grid(min_val, max_val, step, param_type)
    if param_type == 'int':
        range_preferences[param_name] = (int(min_val), int(max_val))
    else:
        range_preferences[param_name] = (float(min_val), float(max_val))
    st.session_state["strategy_param_ranges"] = range_preferences
    test_count = len(grid)

    st.caption(f"{test_count} combinaisons à tester pour {param_label}")

    if st.button("🔬 Lancer l'Optimisation", type="primary", use_container_width=True, key="run_sweep_btn"):
        configured_params = st.session_state.get("strategy_params", {}) or {}

        # Préparer les valeurs de la grille
        param_values = [int(v) if param_type == 'int' else float(v) for v in grid]

        with st.spinner(f"⚡ Optimisation RAPIDE en cours... {test_count} tests"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            # Récupérer la fonction de stratégie optimisée
            strategy_func = get_strategy_function(strategy)

            # Callback pour mise à jour UI (seulement tous les 50 runs)
            def update_ui(current_idx, total, _result):
                progress = current_idx / total
                progress_bar.progress(progress)

                elapsed = time.time() - start_time
                rate = current_idx / elapsed if elapsed > 0 else 0
                eta = (total - current_idx) / rate if rate > 0 else 0

                status_text.text(
                    f"⚡ {current_idx}/{total} tests "
                    f"({rate:.0f} runs/sec, ETA: {eta:.1f}s)"
                )

            try:
                # FAST SWEEP - Utilise calculs vectorisés numpy
                results_df = fast_parameter_sweep(
                    data=data,
                    param_name=param_name,
                    param_values=param_values,
                    strategy_func=strategy_func,
                    capital_initial=capital_initial,
                    update_callback=update_ui,
                    update_frequency=50,  # Mise à jour UI tous les 50 runs
                )

                # Ajouter les colonnes de formatage
                if param_type == 'int':
                    results_df["param_display"] = results_df["param"].apply(
                        lambda x: str(int(x))
                    )
                else:
                    results_df["param_display"] = results_df["param"].apply(
                        lambda x: _format_param_value(x, param_type, decimals)
                    )

                # Stocker résultats
                st.session_state.sweep_results = results_df
                st.session_state["sweep_capital_used"] = capital_initial
                st.session_state["sweep_param_label"] = param_label
                st.session_state["sweep_param_type"] = param_type
                st.session_state["sweep_param_decimals"] = decimals

                # Stats finales
                total_time = time.time() - start_time
                throughput = test_count / total_time if total_time > 0 else 0

                status_text.text(
                    f"✅ Optimisation terminée ! "
                    f"{test_count} tests en {total_time:.2f}s "
                    f"({throughput:.0f} runs/sec)"
                )
                progress_bar.empty()

                st.success(f"🚀 Performance: **{throughput:.0f} runs/seconde** !")

            except Exception as exc:
                status_text.text("❌ Optimisation interrompue")
                st.error(f"❌ Sweep interrompu: {exc}")
                import traceback
                st.code(traceback.format_exc())
                return

    sweep_df = st.session_state.get("sweep_results")
    capital = st.session_state.get("sweep_capital_used", capital_initial)
    stored_param_label = st.session_state.get("sweep_param_label", param_label)
    stored_param_type = st.session_state.get("sweep_param_type", param_type)
    stored_param_decimals = st.session_state.get("sweep_param_decimals", decimals)

    if isinstance(sweep_df, pd.DataFrame) and not sweep_df.empty:
        st.markdown("---")
        st.markdown("### 📊 Résultats de l'Optimisation")

        best_idx = sweep_df["sharpe"].idxmax()
        best_param = sweep_df.loc[best_idx, "param"]
        best_param_display = sweep_df.loc[best_idx, "param_display"] if "param_display" in sweep_df.columns else _format_param_value(best_param, stored_param_type, stored_param_decimals)
        best_sharpe = sweep_df.loc[best_idx, "sharpe"]
        best_pnl = sweep_df.loc[best_idx, "pnl_euros"]

        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        with col_sum1:
            st.metric("🏆 Meilleur Paramètre", best_param_display)
        with col_sum2:
            st.metric("📈 Sharpe Ratio", f"{best_sharpe:.3f}")
        with col_sum3:
            st.metric("💶 PnL", f"{best_pnl:.2f} €", delta=f"{best_pnl/capital*100:.1f}%")
        with col_sum4:
            best_equity = sweep_df.loc[best_idx, "equity_final"]
            st.metric("💰 Capital Final", f"{best_equity:.2f} €")

        st.markdown("---")

        st.markdown("#### 📈 Visualisations")

        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(
            x=sweep_df["param"],
            y=sweep_df["sharpe"],
            mode="lines+markers",
            name="Sharpe Ratio",
            line=dict(color='#26a69a', width=3),
            marker=dict(size=10, line=dict(width=2, color='#1e5c4f'))
        ))
        fig_sharpe.add_vline(
            x=best_param,
            line_dash="dash",
            line_color="#ffd700",
            line_width=2,
            annotation_text=f"⭐ Optimal: {best_param_display}",
            annotation_position="top"
        )
        fig_sharpe.update_layout(
            xaxis_title=f"Paramètre: {stored_param_label}",
            yaxis_title="Sharpe Ratio",
            template="plotly_dark",
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#a8b2d1'),
            hovermode='x unified',
        )
        if stored_param_type == "float":
            fig_sharpe.update_xaxes(tickformat=f".{stored_param_decimals}f")
        else:
            fig_sharpe.update_xaxes(tickformat="d")
        st.plotly_chart(fig_sharpe, use_container_width=True, key="sweep_sharpe_main")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("##### 💶 PnL en Euros")
            fig_pnl = go.Figure()
            colors = ['#26a69a' if x >= 0 else '#ef5350' for x in sweep_df["pnl_euros"]]
            fig_pnl.add_trace(go.Bar(
                x=sweep_df["param"],
                y=sweep_df["pnl_euros"],
                name="PnL (€)",
                marker=dict(color=colors, line=dict(width=1, color='#ffffff'))
            ))
            fig_pnl.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
            fig_pnl.update_layout(
                xaxis_title=stored_param_label,
                yaxis_title="PnL (€)",
                template="plotly_dark",
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11, color='#a8b2d1'),
            )
            if stored_param_type == "float":
                fig_pnl.update_xaxes(tickformat=f".{stored_param_decimals}f")
            else:
                fig_pnl.update_xaxes(tickformat="d")
            st.plotly_chart(fig_pnl, use_container_width=True, key="sweep_pnl")

        with col_g2:
            st.markdown("##### 📉 Max Drawdown")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=sweep_df["param"],
                y=sweep_df["max_dd"],
                mode="lines+markers",
                name="Max DD (%)",
                line=dict(color='#ef5350', width=2),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(239, 83, 80, 0.1)',
            ))
            fig_dd.update_layout(
                xaxis_title=stored_param_label,
                yaxis_title="Max Drawdown (%)",
                template="plotly_dark",
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11, color='#a8b2d1'),
            )
            if stored_param_type == "float":
                fig_dd.update_xaxes(tickformat=f".{stored_param_decimals}f")
            else:
                fig_dd.update_xaxes(tickformat="d")
            st.plotly_chart(fig_dd, use_container_width=True, key="sweep_dd")

        st.markdown("##### 🔍 Analyse des Trades: Nombre vs Taille Moyenne")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=sweep_df["nb_trades"],
            y=sweep_df["avg_trade_size"],
            mode="markers",
            name="Tests",
            marker=dict(
                size=12,
                color=sweep_df["sharpe"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe"),
                line=dict(width=1, color='white')
            ),
            text=[f"{stored_param_label}={label}" for label in sweep_df.get("param_display", sweep_df["param"])],
            hovertemplate="<b>%{text}</b><br>Trades: %{x}<br>Taille moy: %{y:.2f}<extra></extra>"
        ))
        best_nb_trades = sweep_df.loc[best_idx, "nb_trades"]
        best_avg_size = sweep_df.loc[best_idx, "avg_trade_size"]
        fig_scatter.add_trace(go.Scatter(
            x=[best_nb_trades],
            y=[best_avg_size],
            mode="markers",
            name="Optimal",
            marker=dict(size=20, color='#ffd700', symbol='star', line=dict(width=2, color='white'))
        ))
        fig_scatter.update_layout(
            xaxis_title="Nombre de Trades",
            yaxis_title="Taille Moyenne du Trade",
            template="plotly_dark",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color='#a8b2d1'),
            hovermode='closest',
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="sweep_scatter")

        st.markdown("#### 📋 Tableau Détail")
        display_df = sweep_df.copy()
        if "param_display" in display_df.columns:
            display_df["param"] = display_df["param_display"]
            display_df.drop(columns=["param_display"], inplace=True, errors="ignore")

        display_df = display_df.rename(columns={
            "param": stored_param_label,
            "sharpe": "Sharpe Ratio",
            "return_pct": "Rendement (%)",
            "pnl_euros": f"PnL (€) - Capital: {capital}€",
            "equity_final": "Capital Final (€)",
            "max_dd": "Max DD (%)",
            "nb_trades": "Nb Trades",
            "avg_trade_size": "Taille Moy Trade",
        })

        display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].apply(lambda x: f"{x:.3f}")
        display_df["Rendement (%)"] = display_df["Rendement (%)"].apply(lambda x: f"{x:.2f}%")
        display_df[f"PnL (€) - Capital: {capital}€"] = display_df[f"PnL (€) - Capital: {capital}€"].apply(lambda x: f"{x:.2f} €")
        display_df["Capital Final (€)"] = display_df["Capital Final (€)"].apply(lambda x: f"{x:.2f} €")
        display_df["Max DD (%)"] = display_df["Max DD (%)"].apply(lambda x: f"{x:.2f}%")
        display_df["Taille Moy Trade"] = display_df["Taille Moy Trade"].apply(lambda x: f"{x:.2f}")

        st.dataframe(display_df, use_container_width=True, height=400)

        st.markdown("---")
        st.markdown(f"#### 💡 Simulation avec {capital}€ de capital")
        best_return = sweep_df.loc[best_idx, "return_pct"] / 100
        pnl_capital = capital * best_return
        equity_capital = capital * (1 + best_return)

        col_cap1, col_cap2, col_cap3 = st.columns(3)
        with col_cap1:
            st.metric("Capital Initial", f"{capital:.0f} €")
        with col_cap2:
            st.metric("PnL", f"{pnl_capital:.2f} €", delta=f"{best_return*100:.2f}%")
        with col_cap3:
            st.metric("Capital Final", f"{equity_capital:.2f} €")

        st.markdown("---")
        csv = sweep_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Exporter les résultats (CSV)",
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
    """Point d'entrée de la page Backtest & Optimisation."""
    st.title("📊 Backtest & Optimisation")
    st.markdown("*Testez et optimisez vos stratégies de trading*")
    st.markdown("---")

    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["🎯 Backtest Simple", "🔬 Optimisation Sweep", "🎲 Monte-Carlo"])

    with tab1:
        _render_backtest_tab()

    with tab2:
        _render_optimization_tab()

    with tab3:
        _render_monte_carlo_tab()


if __name__ == "__main__":
    main()
