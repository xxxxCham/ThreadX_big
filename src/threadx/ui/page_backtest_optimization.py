"""
ThreadX - Page Backtest & Optimisation
=======================================

Page fusionnée combinant le backtest simple et l'optimisation Sweep.
Interface organisée en onglets pour une navigation intuitive.

Author: ThreadX Framework
Version: 2.0.0 - UI Redesign
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from threadx.data_access import load_ohlcv
from threadx.indicators.bank import IndicatorBank, IndicatorSettings
from threadx.optimization.engine import SweepRunner
from threadx.optimization.parallel_sweep_manager import probe_parallel_configs
from threadx.optimization.scenarios import ScenarioSpec
from threadx.ui.backtest_bridge import BacktestResult, run_backtest, run_backtest_gpu
from threadx.ui.strategy_registry import (
    base_params_for,
    list_strategies,
    parameter_specs_for,
    resolve_range,
    tunable_parameters_for,
)
from threadx.ui.system_monitor import get_global_monitor
from threadx.utils.log import get_logger

logger = get_logger(__name__)


def _get_param_description(key: str) -> str:
    """Retourne une description détaillée pour un paramètre donné."""
    descriptions = {
        # Bollinger Bands
        "bb_period": "Nombre de périodes pour calculer la moyenne mobile (SMA) des Bandes de Bollinger. Plus élevé = bandes plus lisses.",
        "bb_std": "Multiplicateur de l'écart-type (σ) pour les bandes supérieure et inférieure. 2.0 = ±2 écarts-types (95% de confiance).",
        "bb_window": "Nombre de périodes pour la moyenne mobile des Bandes de Bollinger.",

        # ATR (Average True Range)
        "atr_period": "Nombre de périodes pour calculer l'Average True Range (volatilité). Classique : 14 périodes.",
        "atr_multiplier": "Multiplicateur de l'ATR pour définir les stops/trailing stops. Plus élevé = stops plus larges.",
        "atr_window": "Fenêtre de calcul pour l'Average True Range.",
        "sl_atr_multiplier": "Multiplicateur ATR pour le Stop Loss initial (ex: 2.0 × ATR = stop à 2 ATR de distance).",

        # Entrées/Sorties
        "entry_z": "Seuil de Z-score pour déclencher une entrée. Plus bas = entrées plus agressives. 1.0 = 1 écart-type.",
        "entry_logic": "Logique pour combiner les conditions d'entrée : AND (toutes) ou OR (au moins une).",
        "pb_entry_threshold_min": "Valeur %B minimale pour entrée (0.0 = bande basse, 1.0 = bande haute).",
        "pb_entry_threshold_max": "Valeur %B maximale pour entrée.",

        # Risk Management
        "risk_per_trade": "Fraction du capital risquée par trade. 0.02 = 2% du capital par position.",
        "min_pnl_pct": "Filtre de profit/perte minimum en %. Trades < ce seuil sont ignorés. 0.0 = désactivé.",
        "max_hold_bars": "Durée maximale d'une position en nombre de barres (chandelier). Force la sortie après expiration.",
        "spacing_bars": "Nombre minimal de barres à attendre entre deux trades consécutifs (anti-overtrading).",
        "min_spacing_bars": "Espacement minimum entre trades pour éviter le surtrading.",
        "stop_loss_pct": "Stop Loss fixe en pourcentage du prix d'entrée (ex: 2.0 = -2%).",
        "take_profit_pct": "Take Profit fixe en pourcentage du prix d'entrée (ex: 4.0 = +4%).",
        "leverage": "Effet de levier appliqué. 1.0 = sans levier, 2.0 = doublement de l'exposition.",
        "short_stop_pct": "Stop Loss spécifique pour les positions SHORT, en %.",
        "sl_min_pct": "Stop Loss minimum en % pour protéger le capital.",

        # Trailing Stops
        "trailing_stop": "Active/désactive le Trailing Stop basé sur ATR.",
        "trailing_activation_pb_threshold": "Seuil %B pour activer le trailing stop (ex: 1.0 = au-dessus de la bande haute).",
        "trailing_activation_gain_r": "Gain en ratio R (Risk/Reward) nécessaire pour activer le trailing.",
        "trailing_type": "Type de trailing stop : chandelier (ATR), pb_floor (%B), ou macd_fade (MACD).",
        "trailing_chandelier_atr_mult": "Multiplicateur ATR pour le Chandelier Exit (trailing stop ATR).",
        "trailing_pb_floor": "Valeur %B plancher pour sortie via trailing stop.",

        # Filtres et Tendance
        "trend_period": "Période de l'EMA pour filtrer la tendance. 0 = désactivé. Évite les trades contre-tendance.",
        "ema_filter_period": "Période EMA pour filtrer les trades par tendance. 0 = pas de filtre.",
        "bbwidth_percentile_threshold": "Percentile de BBWidth pour filtrer les régimes de volatilité (30-70 = médian).",
        "bbwidth_lookback": "Nombre de barres pour calculer le percentile de BBWidth.",
        "volume_zscore_threshold": "Seuil de Z-score pour le volume (filtre d'activité du marché).",
        "volume_lookback": "Fenêtre de calcul pour le Z-score du volume.",
        "use_adx_filter": "Active le filtre ADX (Average Directional Index) pour détecter les tendances.",
        "adx_threshold": "Seuil ADX pour considérer une tendance. < 20 = range, > 25 = tendance.",
        "adx_period": "Période de calcul de l'ADX.",

        # Moyennes Mobiles
        "fast_period": "Période de la moyenne mobile rapide (SMA).",
        "slow_period": "Période de la moyenne mobile lente (SMA).",
        "fast_window": "Fenêtre EMA rapide pour croisements.",
        "slow_window": "Fenêtre EMA lente pour croisements.",
        "atr_mult": "Multiplicateur ATR pour les canaux de prix.",

        # MACD
        "macd_fast": "Période rapide du MACD (généralement 12).",
        "macd_slow": "Période lente du MACD (généralement 26).",
        "macd_signal": "Période de la ligne de signal MACD (généralement 9).",

        # AmplitudeHunter spécifique
        "spring_lookback": "Nombre de barres pour détecter un 'spring' (fausse cassure).",
        "amplitude_score_threshold": "Seuil du score d'amplitude pour valider une opportunité (0-1).",
        "amplitude_w1_bbwidth": "Poids de BBWidth dans le score d'amplitude (0-1).",
        "amplitude_w2_pb": "Poids de %B dans le score d'amplitude (0-1).",
        "amplitude_w3_macd_slope": "Poids de la pente MACD dans le score d'amplitude (0-1).",
        "amplitude_w4_volume": "Poids du volume dans le score d'amplitude (0-1).",
        "pyramiding_enabled": "Active/désactive l'ajout de positions (pyramiding).",
        "pyramiding_max_adds": "Nombre maximum d'ajouts de positions (1-2).",
        "use_bip_target": "Active la sortie partielle à la cible BIP (Break-In-Profit).",
        "bip_partial_exit_pct": "Pourcentage de la position à clôturer à la cible BIP (ex: 0.5 = 50%).",

        # Frais
        "fee_bps": "Frais de transaction en basis points (1 bp = 0.01%). Ex: 4.5 = 0.045%.",
        "slippage_bps": "Slippage estimé en basis points.",
    }
    return descriptions.get(key, f"Paramètre {key.replace('_', ' ').title()}")


def _save_config_to_history(
    strategy: str,
    strategy_params: dict,
    param_ranges: dict,
    global_sensitivity: float = 1.0,
    n_scenarios: int | None = None,
    config_type: str = "Sweep",
) -> None:
    """Sauvegarde la configuration actuelle dans l'historique."""
    if "config_history" not in st.session_state:
        st.session_state.config_history = []

    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy,
        "type": config_type,
        "strategy_params": strategy_params.copy() if strategy_params else {},
        "param_ranges": param_ranges.copy() if param_ranges else {},
        "global_sensitivity": global_sensitivity,
        "n_scenarios": n_scenarios,
    }

    st.session_state.config_history.append(config)

    # Limiter l'historique à 20 configurations
    if len(st.session_state.config_history) > 20:
        st.session_state.config_history = st.session_state.config_history[-20:]


def _render_config_history(key_prefix: str = "") -> dict | None:
    """Affiche l'historique des configurations avec navigation."""
    with st.expander("📜 Historique des Configurations", expanded=False):
        history = st.session_state.get("config_history", [])

        if not history:
            st.caption("Aucune configuration enregistrée pour l'instant.")
            return None

        # Navigation dans l'historique
        st.caption(f"**{len(history)} configuration(s) sauvegardée(s)**")

        # Afficher les configurations récentes
        for idx, cfg in enumerate(reversed(history)):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(
                        f"**{cfg['type']} - {cfg['strategy']}**  \n"
                        f"📅 {cfg['timestamp']}  \n"
                        f"🎚️ Sensibilité: {cfg['global_sensitivity']}x"
                    )
                with col2:
                    if st.button(
                        "📥 Charger",
                        key=f"{key_prefix}load_hist_{len(history) - 1 - idx}",
                        use_container_width=True,
                    ):
                        return cfg
                with col3:
                    if st.button(
                        "🗑️ Suppr.",
                        key=f"{key_prefix}del_hist_{len(history) - 1 - idx}",
                        use_container_width=True,
                    ):
                        st.session_state.config_history.pop(len(history) - 1 - idx)
                        st.rerun()

                st.markdown("---")

        return None


def _sort_results_by_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Trie les résultats par PNL décroissant, avec fallback robuste.

    Priorité des colonnes considérées comme PNL:
    - 'pnl', 'PNL', 'total_pnl', 'net_pnl', 'net_profit', 'profit', 'total_profit'
    - Fallbacks usuels si aucun PNL explicite: 'total_return', 'sharpe', 'objective', 'score'

    Args:
        df: DataFrame de résultats

    Returns:
        DataFrame trié (ou inchangé si aucune colonne trouvée)
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    candidates = [
        "pnl",
        "PNL",
        "total_pnl",
        "net_pnl",
        "net_profit",
        "profit",
        "total_profit",
        # Fallbacks si PNL absent
        "total_return",
        "sharpe",
        "objective",
        "score",
    ]

    for col in candidates:
        if col in df.columns:
            try:
                return df.sort_values(by=col, ascending=False)
            except Exception:
                continue

    return df


def _extract_params_from_row(strategy: str, row: pd.Series) -> dict[str, Any]:
    """Construit un dict de paramètres depuis une ligne de résultats.

    On lit d'abord les paramètres attendus par la stratégie (registry), puis on
    complète par les valeurs courantes/base si la colonne est absente.
    """
    try:
        specs = parameter_specs_for(strategy)
    except Exception:
        specs = {}
    base_params = base_params_for(strategy)
    current = (st.session_state.get("strategy_params") or {}).copy()

    params: dict[str, Any] = {}
    keys = list(specs.keys()) if isinstance(specs, dict) else []
    for k in keys:
        if k in row.index:
            params[k] = row[k]
        elif k in current:
            params[k] = current[k]
        elif k in base_params:
            params[k] = base_params[k]

    # Inclure aussi d'éventuelles colonnes param non listées par specs (numériques)
    for k, v in row.items():
        if (
            k not in params
            and isinstance(v, (int, float))
            and k
            not in (
                "score",
                "objective",
                "sharpe",
                "total_return",
                "pnl",
            )
        ):
            params[k] = v

    return params


def _render_price_with_trades(
    df: pd.DataFrame, trades: list[dict[str, Any]], title: str = "📈 OHLC + Trades"
) -> None:
    """Trace un graphique OHLC avec repères d'entrées/sorties de trades."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.warning("⚠️ Données OHLCV indisponibles pour le tracé")
        return

    if not {"open", "high", "low", "close"} <= set(df.columns):
        st.warning("⚠️ Colonnes OHLC manquantes pour le tracé")
        return

    st.markdown(f"#### {title}")
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index.to_list(),
            open=df["open"].tolist(),
            high=df["high"].tolist(),
            low=df["low"].tolist(),
            close=df["close"].tolist(),
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )

    # Points d'entrée/sortie
    entries_x, entries_y, exits_x, exits_y = [], [], [], []
    entries_color, exits_color = [], []
    for t in trades or []:
        side = str(t.get("side", "LONG")).upper()
        # Entrée
        if "entry_time" in t and "entry_price" in t:
            entries_x.append(t["entry_time"])
            entries_y.append(t["entry_price"])
            entries_color.append("#42a5f5" if side == "LONG" else "#ab47bc")
        # Sortie
        if "exit_time" in t and "exit_price" in t:
            exits_x.append(t["exit_time"])
            exits_y.append(t["exit_price"])
            exits_color.append("#ffa726" if side == "LONG" else "#ff7043")

    if entries_x:
        fig.add_trace(
            go.Scatter(
                x=list(entries_x),
                y=list(entries_y),
                mode="markers",
                name="Entrée",
                marker=dict(symbol="triangle-up", size=10, color=entries_color),
            )
        )
    if exits_x:
        fig.add_trace(
            go.Scatter(
                x=list(exits_x),
                y=list(exits_y),
                mode="markers",
                name="Sortie",
                marker=dict(symbol="triangle-down", size=10, color=exits_color),
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Prix",
        xaxis=dict(rangeslider=dict(visible=False), gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ohlc_trades_{title}")


def _require_configuration() -> dict[str, Any]:
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
        st.info(
            "👈 Retournez sur **Configuration & Stratégie** et cliquez sur 'Charger & Prévisualiser'."
        )
        st.stop()

    return {key: st.session_state[key] for key in required_keys}


def _render_config_badge(context: dict[str, Any]) -> None:
    """Affiche un badge récapitulatif de la configuration."""
    st.info(
        f"📊 **{context['symbol']}** @ {context['timeframe']} | "
        f"📅 {context['start_date']} → {context['end_date']} | "
        f"⚙️ {context['strategy']}"
    )


def _ensure_sweep_ui_defaults() -> None:
    """Applique les presets globaux demandés pour le sweep."""
    defaults = {
        "sweep_force_processpool": True,
        "sweep_workers_mode": "Manuel",
        "sweep_manual_workers": 30,
        "sweep_enable_llm": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_price_chart(
    df: pd.DataFrame, indicators: dict[str, dict[str, Any]]
) -> None:
    """Graphique OHLC avec indicateurs."""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index.to_list(),
            open=df["open"].tolist(),
            high=df["high"].tolist(),
            low=df["low"].tolist(),
            close=df["close"].tolist(),
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
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

        fig.add_trace(
            go.Scatter(
                x=df.index.to_list(),
                y=mid.tolist(),
                name="BB Mid",
                mode="lines",
                line=dict(color="#ffa726", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_list(),
                y=(mid + std_mult * std).tolist(),
                name="BB Upper",
                mode="lines",
                line=dict(color="#42a5f5", width=1, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index.to_list(),
                y=(mid - std_mult * std).tolist(),
                name="BB Lower",
                mode="lines",
                line=dict(color="#42a5f5", width=1, dash="dash"),
            )
        )

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Prix (USD)",
        xaxis=dict(rangeslider=dict(visible=False), gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a8b2d1", size=11),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True, key="backtest_chart")


def _render_equity_curve(equity: pd.Series) -> None:
    """Courbe d'équité moderne."""
    if equity.empty:
        st.warning("⚠️ Courbe d'équité vide.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity.index.to_list(),
            y=equity.values.tolist(),
            mode="lines",
            name="Équité",
            line=dict(color="#26a69a", width=2),
            fill="tozeroy",
            fillcolor="rgba(38, 166, 154, 0.1)",
        )
    )

    # Ligne initiale
    fig.add_hline(
        y=equity.iloc[0],
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Capital initial",
        annotation_position="right",
    )

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Équité ($)",
        xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a8b2d1", size=11),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, key="equity_curve")


def _render_metrics(metrics: dict[str, Any]) -> None:
    """Métriques de performance en cartes."""
    if not metrics:
        st.info("ℹ️ Aucune métrique calculée.")
        return

    # Filtrer llm_interpretation pour affichage séparé
    metrics_filtered = {k: v for k, v in metrics.items() if k != "llm_interpretation"}

    if not metrics_filtered:
        st.info("ℹ️ Aucune métrique numérique calculée.")
        return

    # Organiser métriques en colonnes
    metrics_list = list(metrics_filtered.items())
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
                    if isinstance(value, (int, float)):
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


def _render_llm_insights(llm_interpretation: dict[str, Any] | None) -> None:
    """Affiche les insights IA si disponibles."""
    if not llm_interpretation:
        return

    st.markdown("---")
    st.markdown("### 🤖 AI Insights")

    # Interprétation globale
    interpretation = llm_interpretation.get("interpretation", "")
    if interpretation:
        st.info(interpretation)

    # Forces
    strengths = llm_interpretation.get("strengths", [])
    if strengths and len(strengths) > 0:
        with st.expander("💪 Forces identifiées", expanded=True):
            for strength in strengths:
                st.markdown(f"✓ {strength}")

    # Faiblesses
    weaknesses = llm_interpretation.get("weaknesses", [])
    if weaknesses and len(weaknesses) > 0:
        with st.expander("⚠️ Points d'attention", expanded=True):
            for weakness in weaknesses:
                st.markdown(f"⚡ {weakness}")

    # Recommandations
    recommendations = llm_interpretation.get("recommendations", [])
    if recommendations and len(recommendations) > 0:
        with st.expander("💡 Recommandations", expanded=True):
            for rec in recommendations:
                st.markdown(f"→ {rec}")

    # Métadonnées additionnelles
    risk_level = llm_interpretation.get("risk_level", "UNKNOWN")
    suitability = llm_interpretation.get("suitability", "")

    if risk_level or suitability:
        col1, col2 = st.columns(2)
        with col1:
            # Couleur selon niveau de risque
            risk_colors = {
                "LOW": "🟢",
                "MODERATE": "🟡",
                "HIGH": "🔴",
                "UNKNOWN": "⚪"
            }
            risk_icon = risk_colors.get(risk_level, "⚪")
            st.metric("Niveau de risque", f"{risk_icon} {risk_level}")

        with col2:
            if suitability:
                st.markdown(f"**Profil adapté:**  \n{suitability}")


# NOTE: _build_sweep_grid déjà défini plus haut — suppression de la redéfinition.


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

    _ensure_sweep_ui_defaults()
    _render_config_badge(context)

    # === TEMPLATES ET HISTORIQUE ===
    st.markdown("#### 🎯 Templates & Historique")
    col_template, col_history = st.columns([1, 1])

    with col_template:
        template_mc = st.selectbox(
            "📦 Template de configuration",
            [
                "Aucun (personnalisé)",
                "🎲 Monte-Carlo Rapide (500 scénarios)",
                "🎯 Monte-Carlo Standard (2000 scénarios)",
                "🔬 Monte-Carlo Précis (5000 scénarios)",
            ],
            key="mc_template_selector",
        )

        # Appliquer le template
        if template_mc == "🎲 Monte-Carlo Rapide (500 scénarios)":
            st.session_state.mc_n = 500
            st.session_state.mc_global_sensitivity = 0.8
            st.info("✨ Template appliqué : 500 scénarios, sensibilité 0.8x")
        elif template_mc == "🎯 Monte-Carlo Standard (2000 scénarios)":
            st.session_state.mc_n = 2000
            st.session_state.mc_global_sensitivity = 1.0
            st.info("✨ Template appliqué : 2000 scénarios, sensibilité 1.0x")
        elif template_mc == "🔬 Monte-Carlo Précis (5000 scénarios)":
            st.session_state.mc_n = 5000
            st.session_state.mc_global_sensitivity = 1.5
            st.info("✨ Template appliqué : 5000 scénarios, sensibilité 1.5x")

    with col_history:
        # Afficher l'historique et gérer le chargement
        loaded_config = _render_config_history(key_prefix="mc_")
        if loaded_config:
            if loaded_config["type"] == "Monte-Carlo":
                st.session_state.strategy = loaded_config["strategy"]
                st.session_state.strategy_params = loaded_config["strategy_params"]
                st.session_state["strategy_param_ranges"] = loaded_config["param_ranges"]
                st.session_state.mc_global_sensitivity = loaded_config["global_sensitivity"]
                if loaded_config.get("n_scenarios"):
                    st.session_state.mc_n = loaded_config["n_scenarios"]
                st.success(f"✅ Configuration chargée : {loaded_config['timestamp']}")
                st.rerun()
            else:
                st.warning("⚠️ Cette configuration est pour un Sweep, pas Monte-Carlo")

    st.markdown("---")

    st.markdown("#### Configuration Monte-Carlo")
    col_strategy, col_gpu, col_multigpu, col_workers = st.columns(4)

    with col_strategy:
        strategy = st.selectbox(
            "Stratégie",
            strategies,
            index=(
                strategies.index(context["strategy"])
                if context["strategy"] in strategies
                else 0
            ),
            key="mc_strategy",
        )

    with col_gpu:
        use_gpu = st.checkbox(
            "Activer GPU",
            value=st.session_state.get("mc_use_gpu", True),
            key="mc_use_gpu",
        )

    with col_multigpu:
        use_multigpu = st.checkbox(
            "Multi-GPU (5090+2060)",
            value=st.session_state.get("mc_use_multigpu", True),
            key="mc_use_multigpu",
        )

    with col_workers:
        # Récupérer la sélection précédente depuis session_state
        current_mode = st.session_state.get("mc_workers_mode", "Auto (Dynamique)")
        mode_index = 1 if current_mode == "Manuel" else 0

        workers_mode = st.selectbox(
            "Workers",
            ["Auto (Dynamique)", "Manuel"],
            index=mode_index,
            key="mc_workers_mode",
        )
        if workers_mode == "Manuel":
            max_workers = st.number_input(
                "Nb Workers",
                min_value=2,
                max_value=64,
                value=st.session_state.get("mc_manual_workers", 30),
                step=1,
                key="mc_manual_workers",
            )
        else:
            max_workers = None

    # Option Fast Sweep (expérimental)
    st.checkbox(
        "Activer Fast Sweep (expérimental)",
        value=st.session_state.get("sweep_fast_mode", False),
        key="sweep_fast_mode",
        help="Utilise une implémentation vectorisée ultra-rapide quand un seul paramètre varie.",
    )

    tunable_specs = tunable_parameters_for(strategy)
    if not tunable_specs:
        st.info("ℹ️ Aucun paramètre optimisable pour cette stratégie.")
        return

    configured_params = st.session_state.get("strategy_params", {}) or {}
    base_strategy_params = base_params_for(strategy)

    range_preferences = st.session_state.get("strategy_param_ranges", {}).copy()

    # Curseur global de sensibilité (ajuste tous les steps proportionnellement)
    st.markdown("##### 🎚️ Sensibilité Globale")
    global_sensitivity = st.slider(
        "Ajuste la granularité de tous les paramètres simultanément",
        min_value=0.5,
        max_value=2.0,
        value=st.session_state.get("mc_global_sensitivity", 1.0),
        step=0.1,
        key="mc_global_sensitivity",
        help="0.5x = Moins de combinaisons (rapide), 2.0x = Plus de combinaisons (précis)",
    )

    st.markdown("##### 🎲 Plages de Paramètres À ÉCHANTILLONNER")
    st.caption("Définissez les intervalles pour l'échantillonnage Monte-Carlo (tirages aléatoires)")

    param_ranges: dict[str, tuple[float, float]] = {}
    param_types: dict[str, str] = {}

    for key, spec in tunable_specs.items():
        label = spec.get("label") or key.replace("_", " ").title()
        param_type = spec.get("type") or (
            "float" if isinstance(spec.get("default"), float) else "int"
        )
        param_types[key] = param_type

        default_val = configured_params.get(key, spec.get("default"))
        if default_val is None:
            default_val = base_strategy_params.get(
                key, 0 if param_type == "int" else 0.0
            )

        min_val = spec.get("min")
        max_val = spec.get("max")
        opt_min, opt_max = resolve_range(spec)
        if min_val is None:
            min_val = opt_min if opt_min is not None else default_val
        if max_val is None:
            max_val = opt_max if opt_max is not None else default_val
        if min_val is None:
            min_val = 0 if param_type == "int" else 0.0
        if max_val is None or max_val <= min_val:
            max_val = min_val + (
                spec.get("step") or (1 if param_type == "int" else 0.1)
            )

        stored_range = range_preferences.get(key)

        # Récupérer la description détaillée
        param_description = _get_param_description(key)

        # Séparateur visuel entre les paramètres
        st.markdown(f"**{label}** ({key})")
        st.caption(param_description)

        # Créer 2 colonnes: plage + sensibilité (Monte-Carlo)
        col_range, col_sense = st.columns([3, 1])

        with col_range:
            if param_type == "int":
                min_val = int(round(min_val))
                max_val = int(round(max_val))
                if stored_range:
                    stored_low, stored_high = map(int, stored_range)
                    default_tuple = (
                        max(min_val, stored_low),
                        min(max_val, stored_high),
                    )
                else:
                    default_tuple = (
                        (min(int(default_val), max_val), max(int(default_val), min_val))
                        if isinstance(default_val, (int, float))
                        else (min_val, max_val)
                    )
                selected_range = st.slider(
                    "Plage",
                    min_value=min_val,
                    max_value=max_val,
                    value=(int(default_tuple[0]), int(default_tuple[1])),
                    step=1,
                    key=f"mc_range_{key}",
                    label_visibility="collapsed",
                )
            else:
                min_val = float(min_val)
                max_val = float(max_val)
                float_step = float(spec.get("step") or 0.1)
                if stored_range:
                    stored_low_f = float(stored_range[0])
                    stored_high_f = float(stored_range[1])
                    default_tuple = (
                        max(min_val, stored_low_f),
                        min(max_val, stored_high_f),
                    )
                else:
                    if default_val is not None:
                        default_min = float(default_val) - 0.1 * abs(float(default_val))
                        default_max = float(default_val) + 0.1 * abs(float(default_val))
                        default_tuple = (
                            max(min_val, default_min),
                            min(max_val, default_max),
                        )
                    else:
                        default_tuple = (min_val, max_val)
                selected_range = st.slider(
                    "Plage",
                    min_value=min_val,
                    max_value=max_val,
                    value=(float(default_tuple[0]), float(default_tuple[1])),
                    step=float_step,
                    key=f"mc_range_{key}",
                    label_visibility="collapsed",
                )

        # Sensibilité : Appliquer le multiplicateur global au step de base - Monte-Carlo
        with col_sense:
            base_step = float(spec.get("step") or 0.1)

            # Calculer le step ajusté avec le multiplicateur global
            adjusted_step = base_step * global_sensitivity

            if param_type == "int":
                # Pour entiers : step ajusté (minimum 1)
                adjusted_step = max(1, int(round(adjusted_step)))
                # Afficher l'information sur l'ajustement
                st.metric(
                    "📊 Step",
                    f"{int(adjusted_step)}",
                    delta=f"×{global_sensitivity:.1f}",
                    label_visibility="collapsed",
                )
            else:
                # Pour floats : afficher le step ajusté avec précision
                st.metric(
                    "📊 Step",
                    f"{adjusted_step:.4f}",
                    delta=f"×{global_sensitivity:.1f}",
                    label_visibility="collapsed",
                )

        range_preferences[key] = (selected_range[0], selected_range[1])
        param_ranges[key] = (selected_range[0], selected_range[1])

        # Display combination count for this parameter (using adjusted step) - Monte-Carlo
        range_min, range_max = selected_range
        span = range_max - range_min

        if param_type == "int":
            # For integers: count the values in the range with this step
            n_combinations = len(
                range(int(range_min), int(range_max) + 1, max(1, int(adjusted_step)))
            )
        else:
            # For floats: (span / step)
            n_combinations = span / adjusted_step if adjusted_step > 0 else 1

        # Show the combination count with adjusted step
        comb_text = f"📊 Plage: {range_min} → {range_max} | Step ajusté: {adjusted_step} | Combinaisons: {n_combinations:.1f}"
        st.caption(comb_text)

        # Séparateur visuel entre paramètres
        st.markdown("---")

    st.session_state["strategy_param_ranges"] = range_preferences
    st.markdown("##### Paramètres d'échantillonnage")
    col_count, col_seed = st.columns(2)
    with col_count:
        n_scenarios = st.number_input(
            "Nombre de scénarios",
            min_value=50,
            max_value=10000,
            value=st.session_state.get("mc_n", 500),
            step=50,
            key="mc_n",
        )
    with col_seed:
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999,
            value=st.session_state.get("mc_seed", 42),
            step=1,
            key="mc_seed",
        )

    # Option IA pour analyse de la meilleure configuration
    st.markdown("---")
    col_llm_mc, col_spacer_mc = st.columns([3, 1])
    with col_llm_mc:
        enable_llm_mc = st.checkbox(
            "🤖 Activer l'analyse IA pour le meilleur scénario",
            value=st.session_state.get("mc_enable_llm", False),
            key="mc_enable_llm",
            help="Génère une interprétation intelligente des résultats du meilleur scénario via LLM (ajoute ~10s)",
        )

    if st.button(
        "🎲 Lancer Monte-Carlo",
        type="primary",
        use_container_width=True,
        key="run_mc_btn",
    ):
        indicator_settings = IndicatorSettings(use_gpu=use_gpu)
        indicator_bank = IndicatorBank(indicator_settings)
        runner = SweepRunner(
            indicator_bank=indicator_bank,
            max_workers=max_workers,
            use_multigpu=use_multigpu,
        )
        scenario_params: dict[str, Any] = {}
        for key, (min_v, max_v) in param_ranges.items():
            if param_types[key] == "int":
                values = list(range(int(min_v), int(max_v) + 1))
            else:
                values = np.linspace(min_v, max_v, num=50).tolist()
            scenario_params[key] = {"values": values}

        # 🔥 FIX CRITIQUE: Ajouter TOUS les paramètres par défaut manquants
        # Garantir que min_pnl_pct et autres params sont TOUJOURS présents
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
                logger.debug(f"[MC] Param par défaut ajouté: {key} = {value}")

        mc_spec = ScenarioSpec(
            type="monte_carlo",
            params=scenario_params,
            n_scenarios=int(n_scenarios),
            seed=int(seed),
        )

        # Récupérer les données réelles pour le backtest
        symbol = st.session_state.get("symbol", "BTC")
        timeframe = st.session_state.get("timeframe", "1h")
        start_date = st.session_state.get("start_date")
        end_date = st.session_state.get("end_date")

        # 🔥 FIX CRITIQUE: Recharger les données avec les dates correctes
        # Les données en session peuvent être obsolètes si l'utilisateur a changé les dates
        try:
            real_data = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)
            if real_data.empty:
                st.error(
                    f"⚠️ Aucune donnée disponible pour {symbol}/{timeframe} entre {start_date} et {end_date}"
                )
                return
            # Mettre à jour le cache pour cohérence
            st.session_state.data = real_data
        except Exception as e:
            st.error(f"❌ Erreur chargement données: {e}")
            return

        # Sauvegarder la configuration dans l'historique
        _save_config_to_history(
            strategy=strategy,
            strategy_params=configured_params,
            param_ranges=param_ranges,
            global_sensitivity=global_sensitivity,
            n_scenarios=int(n_scenarios),
            config_type="Monte-Carlo",
        )

        try:
            # Lancer le Monte-Carlo avec barre de progression
            st.markdown("### 🎲 Exécution du Monte-Carlo")
            results = _run_monte_carlo_with_progress(
                runner,
                mc_spec,
                real_data,
                symbol,
                timeframe,
                strategy,
                int(n_scenarios),
            )
            st.session_state["monte_carlo_results"] = results

            # Afficher les informations de configuration
            st.markdown("---")
            st.markdown("### ⚙️ Configuration d'exécution")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric(
                    "Mode Multi-GPU", "Activé ✅" if use_multigpu else "Désactivé ⊘"
                )
            with col_info2:
                actual_workers = runner.max_workers if runner.max_workers else "Auto"
                st.metric("Workers utilisés", str(actual_workers))
            with col_info3:
                st.metric(
                    "Total des résultats",
                    len(results) if isinstance(results, pd.DataFrame) else 0,
                )
        except Exception as exc:
            st.error(f"❌ Erreur Monte-Carlo: {exc}")
            import traceback

            st.code(traceback.format_exc())
            return

    results_df = st.session_state.get("monte_carlo_results")

    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        st.markdown("---")
        st.markdown("### 📈 Résultats Monte-Carlo")

        # Tri par PNL décroissant (fallback si colonne PNL absente)
        results_sorted = _sort_results_by_pnl(results_df)

        # === GRAPHIQUE DE CONVERGENCE MONTE-CARLO ===
        pnl_candidates = [
            "pnl",
            "PNL",
            "total_pnl",
            "net_pnl",
            "net_profit",
            "profit",
            "total_profit",
        ]
        pnl_col = next((c for c in pnl_candidates if c in results_sorted.columns), None)

        if pnl_col and len(results_sorted) > 1:
            st.markdown("#### 📈 Convergence du Meilleur PNL")

            # Calculer l'évolution du meilleur PNL au fil des scénarios
            pnl_values = results_sorted[pnl_col].tolist()
            best_history = []
            current_best = None

            for val in pnl_values:
                if current_best is None or val > current_best:
                    current_best = val
                best_history.append(current_best)

            # Créer le graphique de convergence
            fig_conv = go.Figure()

            fig_conv.add_trace(
                go.Scatter(
                    x=list(range(1, len(best_history) + 1)),
                    y=best_history,
                    mode="lines",
                    name="Meilleur PNL",
                    line=dict(color="#26a69a", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(38, 166, 154, 0.2)",
                )
            )

            # Ligne du meilleur final
            fig_conv.add_hline(
                y=best_history[-1],
                line_dash="dash",
                line_color="#ffa726",
                opacity=0.7,
                annotation_text=f"Meilleur: {best_history[-1]:.2f}",
                annotation_position="right",
            )

            fig_conv.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=20, b=0),
                template="plotly_dark",
                xaxis_title="Numéro du scénario",
                yaxis_title=f"Meilleur {pnl_col}",
                xaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                yaxis=dict(gridcolor="rgba(128,128,128,0.2)"),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a8b2d1", size=11),
                hovermode="x unified",
            )

            st.plotly_chart(fig_conv, use_container_width=True, key="mc_convergence_chart")

            # Analyse de convergence
            if len(best_history) > 10:
                # Comparer les 10 premiers vs 10 derniers scénarios
                first_10_improvement = best_history[9] - best_history[0]
                last_10_improvement = best_history[-1] - best_history[-10]

                if last_10_improvement < first_10_improvement * 0.1:
                    st.success(
                        "✅ **Convergence atteinte** : Les derniers scénarios n'améliorent que peu le résultat."
                    )
                else:
                    st.warning(
                        "⚠️ **Convergence incomplète** : Augmentez le nombre de scénarios pour explorer davantage."
                    )

        st.markdown("---")
        st.dataframe(results_sorted.head(100), use_container_width=True, height=400)

        best_row = results_sorted.iloc[0]
        st.markdown("#### 🏆 Meilleur scénario")
        st.json(best_row.to_dict())

        with st.expander("🔎 OHLC + trades du meilleur scénario", expanded=True):
            strategy_name = st.session_state.get("mc_strategy", context["strategy"])
            best_params = _extract_params_from_row(strategy_name, best_row)

            df_price = st.session_state.get("data")
            if not isinstance(df_price, pd.DataFrame) or df_price.empty:
                try:
                    df_price = load_ohlcv(
                        context["symbol"],
                        context["timeframe"],
                        start=context["start_date"],
                        end=context["end_date"],
                    )
                    st.session_state.data = df_price
                except Exception as e:
                    st.error(f"❌ Erreur chargement données prix: {e}")
                    df_price = None

            if isinstance(df_price, pd.DataFrame) and not df_price.empty:
                try:
                    use_gpu_pref = st.session_state.get("mc_use_gpu", True)
                    enable_llm_mc_analysis = st.session_state.get("mc_enable_llm", False)
                    result_best = run_backtest_gpu(
                        df=df_price,
                        strategy=strategy_name,
                        params=best_params,
                        symbol=context["symbol"],
                        timeframe=context["timeframe"],
                        use_gpu=use_gpu_pref,
                        enable_monitoring=False,
                        enable_llm=enable_llm_mc_analysis,
                        llm_model="gpt-oss:20b",
                    )
                    authentic = (
                        bool(result_best.metadata.get("gpu_enabled"))
                        if isinstance(result_best.metadata, dict)
                        else False
                    )
                    if not authentic:
                        st.warning(
                            "GPU non utilisé: les trades peuvent être approximatifs (CPU)."
                        )
                    _render_price_with_trades(
                        df_price,
                        result_best.trades,
                        title="Meilleur scénario — OHLC + trades",
                    )
                    
                    # Afficher l'analyse IA si disponible
                    llm_interp_mc = result_best.metrics.get("llm_interpretation")
                    if llm_interp_mc:
                        st.markdown("---")
                        _render_llm_insights(llm_interp_mc)
                    
                    with st.expander("Voir la table des trades", expanded=False):
                        _render_trades_table(result_best.trades)
                except Exception as e:
                    st.error(f"❌ Erreur lors du backtest du meilleur scénario: {e}")

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Exporter les résultats Monte-Carlo (CSV)",
            csv,
            "monte_carlo_results.csv",
            "text/csv",
            use_container_width=True,
        )


def _run_sweep_with_progress(
    runner, spec, real_data, symbol, timeframe, strategy, total_combinations
):
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
    # Valeurs par défaut pour éviter KeyError
    shared_state["error"] = None
    shared_state["results"] = None

    # Démarrer le sweep dans un thread pour ne pas bloquer Streamlit
    def run_sweep_thread():
        """Thread qui exécute le sweep (pas de Streamlit calls ici!)."""
        try:
            shared_state["running"] = True
            shared_state["start_time"] = time.time()
            results = runner.run_grid(
                spec,
                real_data,
                symbol,
                timeframe,
                strategy_name=strategy,
                reuse_cache=True,
            )
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
    # Réinitialiser l'historique de lissage pour un nouveau run
    try:
        st.session_state.pop("sweep_speed_samples", None)
    except Exception:
        pass
    sweep_thread = threading.Thread(target=run_sweep_thread, daemon=True)
    sweep_thread.start()

    # Boucle de mise à jour UI (thread principal, synchrone avec Streamlit)
    start_time = time.time()
    status_placeholder = stats_cols[0].empty()
    speed_placeholder = stats_cols[1].empty()
    eta_placeholder = stats_cols[2].empty()
    completed_placeholder = stats_cols[3].empty()

    # Progress initial
    # Throttle des mises a jour UI
    last_current = -1
    last_ui_update = 0.0
    progress_placeholder.progress(0, text="🚀 Initialisation du Sweep...")
    status_placeholder.metric("📊 Statut", "Initialisation...", delta=None)

    # Boucle: mettre à jour l'UI jusqu'à fin du sweep
    if "sweep_speed_samples" not in st.session_state:
        st.session_state["sweep_speed_samples"] = []
    while shared_state["running"]:
        try:
            # Vérifier si l'utilisateur a demandé l'arrêt
            if st.session_state.get("run_stop_requested", False):
                # Tentative silencieuse d'arrêt global (optionnel, peut ne pas être disponible)
                try:  # pragma: no cover - mécanique d'arrêt best-effort
                    from threadx.optimization.engine import request_global_stop  # type: ignore

                    request_global_stop()
                except Exception:
                    pass
                shared_state["should_stop"] = True
                st.session_state.run_stop_requested = False  # Réinitialiser le flag
                progress_placeholder.progress(0, text="⏹️ Arrêt en cours...")
                status_placeholder.metric("📊 Statut", "Arrêt en cours...", delta=None)
                break  # Quitter la boucle d'affichage

            if runner.total_scenarios > 0:
                current = runner.current_scenario
                total = runner.total_scenarios
                progress = min(current / total, 0.99)
                elapsed = time.time() - start_time

                now = time.time()
                if current > 0 and elapsed > 0 and (current != last_current or (now - last_ui_update) >= 0.2):
                    # Débit instantané et lissé (fenêtre ~3s pour plus de réactivité)
                    delta_c = (current - last_current) if last_current >= 0 else 0
                    delta_t = (now - last_ui_update) if last_ui_update > 0 else elapsed
                    inst_speed = (delta_c / delta_t) if delta_t > 0 else 0.0

                    samples = st.session_state.get("sweep_speed_samples", [])
                    samples.append((now, current))
                    cutoff = now - 3.0  # Fenêtre de 3 secondes pour un lissage plus réactif
                    samples = [(t, c) for (t, c) in samples if t >= cutoff]
                    st.session_state["sweep_speed_samples"] = samples
                    if len(samples) >= 2:
                        t0, c0 = samples[0]
                        t1, c1 = samples[-1]
                        smoothed = (c1 - c0) / max(1e-6, (t1 - t0))
                    else:
                        smoothed = inst_speed
                    # Pondération 80% lissé / 20% instantané pour plus de stabilité
                    speed = max(0.0, 0.8 * smoothed + 0.2 * inst_speed)
                    remaining = total - current
                    eta_seconds = remaining / speed if speed > 0 else 0
                    eta_hours, eta_remainder = divmod(eta_seconds, 3600)
                    eta_minutes, eta_secs = divmod(eta_remainder, 60)

                    # Format ETA avec heures si nécessaire
                    if eta_hours >= 1:
                        eta_str = f"{int(eta_hours)}h {int(eta_minutes)}m"
                    else:
                        eta_str = f"{int(eta_minutes)}m {int(eta_secs)}s"

                    last_ui_update = now
                    last_current = current

                    # Mise à jour UI (thread principal) - Style amélioré
                    progress_placeholder.progress(
                        progress, text=f"⏳ {current:,}/{total:,} scénarios ({progress*100:.1f}%)"
                    )
                    status_placeholder.metric(
                        "📊 Statut",
                        "En cours ⚡",
                        delta=f"+{delta_c} en {delta_t:.1f}s",
                        delta_color="normal"
                    )
                    speed_placeholder.metric(
                        "🚀 Vitesse",
                        f"{speed:.1f}",
                        delta="tests/sec",
                        delta_color="off"
                    )
                    eta_placeholder.metric("⏱️ ETA", eta_str)
                    completed_placeholder.metric(
                        "✅ Complétés",
                        f"{current:,}",
                        delta=f"{(current/total*100):.1f}%",
                        delta_color="normal"
                    )

            time.sleep(
                0.2
            )  # Légère réduction de fréquence (200ms) pour alléger l'UI
        except Exception:
            pass  # Ignorer erreurs de mise à jour

    # Attendre fin du thread
    sweep_thread.join(timeout=5)

    # Afficher résultats final
    elapsed_time = time.time() - start_time

    if shared_state["error"]:
        progress_placeholder.progress(0, text=f"❌ Erreur après {elapsed_time:.1f}s")
        status_placeholder.metric("📊 Statut", "Erreur ❌", delta=None)
        st.error(f"Sweep échoué: {shared_state['error']}")
        raise Exception(shared_state["error"])

    results = shared_state.get("results")
    if results is None:
        results = pd.DataFrame()

    completed = len(results) if isinstance(results, pd.DataFrame) else 0
    tests_per_second = completed / elapsed_time if elapsed_time > 0 else 0
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{int(minutes)}m {int(seconds)}s"

    # Stats finales avec style amélioré
    progress_placeholder.progress(1.0, text=f"✅ Sweep terminé en {time_str} | {completed:,} résultats")
    status_placeholder.metric("📊 Statut", "✅ Terminé", delta="100%", delta_color="normal")
    speed_placeholder.metric(
        "🚀 Vitesse Moyenne",
        f"{tests_per_second:.1f}",
        delta="tests/sec",
        delta_color="off"
    )
    eta_placeholder.metric("⏱️ Durée Totale", time_str)
    completed_placeholder.metric("✅ Résultats", f"{completed:,}", delta="100%", delta_color="normal")

    return results


def _run_monte_carlo_with_progress(
    runner, spec, real_data, symbol, timeframe, strategy, n_scenarios
):
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
    # Valeurs par défaut pour éviter KeyError
    shared_state["error"] = None
    shared_state["results"] = None

    # Démarrer le Monte-Carlo dans un thread
    def run_monte_carlo_thread():
        """Thread qui exécute le Monte-Carlo (pas de Streamlit calls ici!)."""
        try:
            shared_state["running"] = True
            shared_state["start_time"] = time.time()
            results = runner.run_monte_carlo(
                spec,
                real_data,
                symbol,
                timeframe,
                strategy_name=strategy,
                reuse_cache=True,
            )
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

    # Réinitialiser l'historique de lissage pour un nouveau run
    try:
        st.session_state.pop("mc_speed_samples", None)
    except Exception:
        pass

    # Démarrer le Monte-Carlo
    mc_thread = threading.Thread(target=run_monte_carlo_thread, daemon=True)
    mc_thread.start()

    # Boucle de mise à jour UI (thread principal, synchrone avec Streamlit)
    start_time = time.time()
    status_placeholder = stats_cols[0].empty()
    speed_placeholder = stats_cols[1].empty()
    eta_placeholder = stats_cols[2].empty()
    completed_placeholder = stats_cols[3].empty()

    # Variables de suivi pour lissage
    last_ui_update = 0.0
    last_current = -1

    # Progress initial
    if "mc_speed_samples" not in st.session_state:
        st.session_state["mc_speed_samples"] = []
    progress_placeholder.progress(0, text="🎲 Initialisation du Monte-Carlo...")
    status_placeholder.metric("📊 Statut", "Initialisation...", delta=None)

    # Boucle: mettre à jour l'UI jusqu'à fin du Monte-Carlo
    while shared_state["running"]:
        try:
            # Vérifier si l'utilisateur a demandé l'arrêt
            if st.session_state.get("run_stop_requested", False):
                try:  # pragma: no cover - arrêt best-effort
                    from threadx.optimization.engine import request_global_stop  # type: ignore

                    request_global_stop()
                except Exception:
                    pass
                shared_state["should_stop"] = True
                st.session_state.run_stop_requested = False  # Réinitialiser le flag
                progress_placeholder.progress(0, text="⏹️ Arrêt en cours...")
                status_placeholder.metric("📊 Statut", "Arrêt en cours...", delta=None)
                break  # Quitter la boucle d'affichage

            if runner.total_scenarios > 0:
                current = runner.current_scenario
                total = runner.total_scenarios
                progress = min(current / total, 0.99)
                elapsed = time.time() - start_time

                now = time.time()
                if current > 0 and elapsed > 0 and (current != last_current or (now - last_ui_update) >= 0.2):
                    # Débit instantané et lissé (fenêtre ~3s pour cohérence avec Sweep)
                    delta_c = (current - last_current) if last_current >= 0 else 0
                    delta_t = (now - last_ui_update) if last_ui_update > 0 else elapsed
                    inst_speed = (delta_c / delta_t) if delta_t > 0 else 0.0

                    samples = st.session_state.get("mc_speed_samples", [])
                    samples.append((now, current))
                    cutoff = now - 3.0  # Fenêtre de 3 secondes
                    samples = [(t, c) for (t, c) in samples if t >= cutoff]
                    st.session_state["mc_speed_samples"] = samples
                    if len(samples) >= 2:
                        t0, c0 = samples[0]
                        t1, c1 = samples[-1]
                        smoothed = (c1 - c0) / max(1e-6, (t1 - t0))
                    else:
                        smoothed = inst_speed
                    # Pondération 80% lissé / 20% instantané
                    speed = max(0.0, 0.8 * smoothed + 0.2 * inst_speed)

                    remaining = total - current
                    eta_seconds = remaining / speed if speed > 0 else 0
                    eta_hours, eta_remainder = divmod(eta_seconds, 3600)
                    eta_minutes, eta_secs = divmod(eta_remainder, 60)

                    # Format ETA avec heures si nécessaire
                    if eta_hours >= 1:
                        eta_str = f"{int(eta_hours)}h {int(eta_minutes)}m"
                    else:
                        eta_str = f"{int(eta_minutes)}m {int(eta_secs)}s"

                    last_ui_update = now
                    last_current = current

                    # Mise à jour UI (thread principal) - Style amélioré
                    progress_placeholder.progress(
                        progress, text=f"⏳ {current:,}/{total:,} scénarios ({progress*100:.1f}%)"
                    )
                    status_placeholder.metric(
                        "📊 Statut",
                        "En cours 🎲",
                        delta=f"+{delta_c} en {delta_t:.1f}s",
                        delta_color="normal"
                    )
                    speed_placeholder.metric(
                        "🚀 Vitesse",
                        f"{speed:.1f}",
                        delta="scén/sec",
                        delta_color="off"
                    )
                    eta_placeholder.metric("⏱️ ETA", eta_str)
                    completed_placeholder.metric(
                        "✅ Complétés",
                        f"{current:,}",
                        delta=f"{(current/total*100):.1f}%",
                        delta_color="normal"
                    )

            time.sleep(
                0.2
            )  # Légère réduction de fréquence (200ms) pour alléger l'UI
        except Exception:
            pass  # Ignorer erreurs de mise à jour

    # Attendre fin du thread
    mc_thread.join(timeout=5)

    # Afficher résultats final
    elapsed_time = time.time() - start_time

    if shared_state["error"]:
        progress_placeholder.progress(0, text=f"❌ Erreur après {elapsed_time:.1f}s")
        status_placeholder.metric("📊 Statut", "Erreur ❌", delta=None)
        st.error(f"Monte-Carlo échoué: {shared_state['error']}")
        raise Exception(shared_state["error"])

    results = shared_state.get("results")
    if results is None:
        results = pd.DataFrame()

    completed = len(results) if isinstance(results, pd.DataFrame) else 0
    scenarios_per_second = completed / elapsed_time if elapsed_time > 0 else 0
    minutes, seconds = divmod(elapsed_time, 60)
    time_str = f"{int(minutes)}m {int(seconds)}s"

    # Stats finales avec style amélioré
    progress_placeholder.progress(1.0, text=f"✅ Monte-Carlo terminé en {time_str} | {completed:,} scénarios")
    status_placeholder.metric("📊 Statut", "✅ Terminé", delta="100%", delta_color="normal")
    speed_placeholder.metric(
        "🚀 Vitesse Moyenne",
        f"{scenarios_per_second:.1f}",
        delta="scén/sec",
        delta_color="off"
    )
    eta_placeholder.metric("⏱️ Durée Totale", time_str)
    completed_placeholder.metric("✅ Scénarios", f"{completed:,}", delta="100%", delta_color="normal")

    return results

    # NOTE: duplication de fonctions supprimée précédemment — ce bloc est volontairement vidé.


def _render_backtest_tab() -> None:
    """Onglet Backtest simple avec option GPU."""
    context = _require_configuration()
    indicators = st.session_state.get("indicators", {})
    params = st.session_state.get("strategy_params", {}) or {}

    _render_config_badge(context)

    st.markdown("### 🚀 Lancer le Backtest")
    col_mode, col_monitor = st.columns(2)
    with col_mode:
        use_gpu = st.checkbox(
            "Activer le moteur GPU (BacktestEngine)",
            value=st.session_state.get("backtest_use_gpu", True),
            key="backtest_use_gpu",
        )
    with col_monitor:
        monitoring = st.checkbox(
            "Monitoring CPU/GPU en temps réel",
            value=st.session_state.get("backtest_monitoring", True),
            key="backtest_monitoring",
        )

    # Option IA
    enable_ai = st.checkbox(
        "🤖 Activer l'analyse IA (LLM)",
        value=st.session_state.get("backtest_enable_llm", False),
        key="backtest_enable_llm",
        help="Génère une interprétation intelligente des résultats via DeepSeek-R1 (ajoute ~10s)",
    )

    if st.button(
        "🚀 Exécuter le Backtest",
        type="primary",
        use_container_width=True,
        key="run_backtest_btn",
    ):
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
                        enable_llm=enable_ai,
                        llm_model="deepseek-r1",
                    )

                    if monitoring:
                        monitor = get_global_monitor()
                        if monitor.is_running():
                            monitor.stop()
                        monitor_history = monitor.get_history_df()
                        monitor.clear_history()
                else:
                    result = run_backtest(
                        df=df, strategy=context["strategy"], params=run_params
                    )

                    monitor = get_global_monitor()
                    if monitor.is_running():
                        monitor.stop()
                    monitor.clear_history()

                st.session_state.backtest_results = result
                st.session_state.data = df
                st.session_state["monitor_history"] = monitor_history

                st.success("✅ Backtest terminé avec succès !")

            except FileNotFoundError as exc:
                st.error(f"⚠️ {exc}")
                return
            except Exception as exc:
                st.error(f"❌ Erreur lors du backtest: {exc}")
                return

    stored_result: BacktestResult = st.session_state.get("backtest_results")
    if stored_result:
        st.markdown("---")
        st.markdown("### 📊 Résultats du Backtest")

        res_tab1, res_tab2, res_tab3 = st.tabs(
            ["🔍 Graphiques", "📈 Métriques", "👥 Transactions"]
        )

        with res_tab1:
            st.markdown("#### Prix & Indicateurs")
            data_df = st.session_state.get("data")
            if isinstance(data_df, pd.DataFrame):
                _render_price_chart(data_df, indicators)

            st.markdown("#### Courbe d'équité")
            _render_equity_curve(stored_result.equity)

            history_df = st.session_state.get("monitor_history")
            _render_monitoring_section(stored_result.metadata, history_df)

        with res_tab2:
            _render_metrics(stored_result.metrics)

            # Afficher AI Insights si disponibles
            llm_interp = stored_result.metrics.get("llm_interpretation")
            if llm_interp:
                _render_llm_insights(llm_interp)

        with res_tab3:
            _render_trades_table(stored_result.trades)


def _render_optimization_tab() -> None:
    """Onglet d'optimisation par balayage exhaustif de paramètres (Sweep)."""
    st.markdown("### 🔬 Optimisation par Sweep (Grille Exhaustive)")

    context = _require_configuration()
    data = st.session_state.get("data")

    if not isinstance(data, pd.DataFrame) or data.empty:
        st.warning(
            "⚠️ Chargez d'abord des données sur la page 'Chargement des Données'."
        )
        return

    strategies = list_strategies()
    if not strategies:
        st.error("❌ Aucune stratégie disponible.")
        return

    _render_config_badge(context)

    # === TEMPLATES ET HISTORIQUE ===
    st.markdown("#### 🎯 Templates & Historique")
    col_template, col_history = st.columns([1, 1])

    with col_template:
        template_sweep = st.selectbox(
            "📦 Template de configuration",
            [
                "Aucun (personnalisé)",
                "🚀 Quick Test (~100 combinaisons)",
                "⚡ Standard (~10k combinaisons)",
                "🔬 Recherche Exhaustive (~100k combinaisons)",
            ],
            key="sweep_template_selector",
        )

        # Appliquer le template
        if template_sweep == "🚀 Quick Test (~100 combinaisons)":
            st.session_state.sweep_global_sensitivity = 0.5
            st.info("✨ Template appliqué : Sensibilité 0.5x (rapide)")
        elif template_sweep == "⚡ Standard (~10k combinaisons)":
            st.session_state.sweep_global_sensitivity = 1.0
            st.info("✨ Template appliqué : Sensibilité 1.0x (équilibré)")
        elif template_sweep == "🔬 Recherche Exhaustive (~100k combinaisons)":
            st.session_state.sweep_global_sensitivity = 1.8
            st.info("✨ Template appliqué : Sensibilité 1.8x (précis)")

    with col_history:
        # Afficher l'historique et gérer le chargement
        loaded_config = _render_config_history(key_prefix="sweep_")
        if loaded_config:
            if loaded_config["type"] == "Sweep":
                st.session_state.strategy = loaded_config["strategy"]
                st.session_state.strategy_params = loaded_config["strategy_params"]
                st.session_state["strategy_param_ranges"] = loaded_config["param_ranges"]
                st.session_state.sweep_global_sensitivity = loaded_config["global_sensitivity"]
                st.success(f"✅ Configuration chargée : {loaded_config['timestamp']}")
                st.rerun()
            else:
                st.warning("⚠️ Cette configuration est pour Monte-Carlo, pas Sweep")

    st.markdown("---")

    st.markdown("#### Configuration du Sweep")
    col_strategy, col_gpu, col_multigpu, col_workers = st.columns(4)

    with col_strategy:
        # Préréglage MA_Crossover
        default_strategy = "MA_Crossover"
        strategy = st.selectbox(
            "Stratégie à optimiser",
            strategies,
            index=(
                strategies.index(default_strategy)
                if default_strategy in strategies
                else strategies.index(context["strategy"]) if context["strategy"] in strategies
                else 0
            ),
            key="sweep_strategy",
        )

    with col_gpu:
        use_gpu = st.checkbox(
            "Activer GPU",
            value=st.session_state.get("sweep_use_gpu", True),
            key="sweep_use_gpu",
        )

    with col_multigpu:
        use_multigpu = st.checkbox(
            "Multi-GPU (5090+2060)",
            value=st.session_state.get("sweep_use_multigpu", True),
            key="sweep_use_multigpu",
        )

    with col_workers:
        # Récupérer la sélection précédente depuis session_state
        current_mode = st.session_state.get("sweep_workers_mode", "Auto (Dynamique)")
        mode_index = 1 if current_mode == "Manuel" else 0

        workers_mode = st.selectbox(
            "Workers",
            ["Auto (Dynamique)", "Manuel"],
            index=mode_index,
            key="sweep_workers_mode",
        )
        if workers_mode == "Manuel":
            max_workers = st.number_input(
                "Nb Workers",
                min_value=2,
                max_value=64,
                value=st.session_state.get("sweep_manual_workers", 30),
                step=1,
                key="sweep_manual_workers",
            )
        else:
            max_workers = None

    # Réglage d'agressivité du feeder (in-flight sizing du moteur)
    st.markdown("#### Réglages avancés")
    st.select_slider(
        "Agressivité feeder",
        options=[1, 2, 4, 6, 8, 10, 12, 16],
        value=st.session_state.get("sweep_feeder_aggr", 10),
        key="sweep_feeder_aggr",
        help="Contrôle la fenêtre de tâches en vol. Plus haut = pipeline plus rempli",
    )
    # Option avancée: forcer l'utilisation d'un ProcessPool (contourner le GIL)
    st.checkbox(
        "Forcer ProcessPool (CPU-bound)",
        value=st.session_state.get("sweep_force_processpool", True),
        key="sweep_force_processpool",
        help="Active un pool de processus (plus coûteux en mémoire) quand la stratégie est GIL-bound",
    )

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

    # Curseur global de sensibilité (ajuste tous les steps proportionnellement)
    st.markdown("##### 🎚️ Sensibilité Globale")
    global_sensitivity = st.slider(
        "Ajuste la granularité de tous les paramètres simultanément",
        min_value=0.5,
        max_value=2.0,
        value=st.session_state.get("sweep_global_sensitivity", 1.0),
        step=0.1,
        key="sweep_global_sensitivity",
        help="0.5x = Moins de combinaisons (rapide), 2.0x = Plus de combinaisons (précis)",
    )

    st.markdown("##### 📊 Plages de Paramètres À OPTIMISER")
    st.caption("Définissez les intervalles à explorer pour trouver la meilleure configuration")

    param_ranges: dict[str, tuple[float, float]] = {}
    param_types: dict[str, str] = {}
    param_steps: dict[str, float] = {}

    for key, spec in tunable_specs.items():
        label = spec.get("label") or key.replace("_", " ").title()
        param_type = spec.get("type") or (
            "float" if isinstance(spec.get("default"), float) else "int"
        )
        param_types[key] = param_type

        default_val = configured_params.get(key, spec.get("default"))
        if default_val is None:
            default_val = base_strategy_params.get(
                key, 0 if param_type == "int" else 0.0
            )

        min_val = spec.get("min")
        max_val = spec.get("max")
        step_val = spec.get("step") or (1 if param_type == "int" else 0.1)
        opt_min, opt_max = resolve_range(spec)

        if min_val is None:
            min_val = opt_min if opt_min is not None else default_val
        if max_val is None:
            max_val = opt_max if opt_max is not None else default_val
        if min_val is None:
            min_val = 0 if param_type == "int" else 0.0
        if max_val is None or max_val <= min_val:
            max_val = min_val + (step_val * 10)

        stored_range = range_preferences.get(key)

        # Récupérer la description détaillée
        param_description = _get_param_description(key)

        # Séparateur visuel entre les paramètres
        st.markdown(f"**{label}** ({key})")
        st.caption(param_description)

        # Préréglages spéciaux pour max_hold_bars et risk_per_trade (valeurs fixes)
        if key == "max_hold_bars":
            # Valeur fixe à 300 (pas de plage)
            st.info("🔒 Préréglé à 300 (valeur fixe)")
            selected_range = (300, 300)
            adjusted_step = 1
        elif key == "risk_per_trade":
            # Valeur fixe à 0.02 (pas de plage)
            st.info("🔒 Préréglé à 0.02 (valeur fixe)")
            selected_range = (0.02, 0.02)
            adjusted_step = 0.01
        else:
            # Comportement normal pour les autres paramètres
            # Créer 2 colonnes: plage + sensibilité
            col_range, col_sense = st.columns([3, 1])

            with col_range:
                if param_type == "int":
                    min_val = int(round(min_val))
                    max_val = int(round(max_val))
                    step_val = max(1, int(round(step_val)))

                    if stored_range:
                        stored_low, stored_high = map(int, stored_range)
                        default_tuple = (
                            max(min_val, stored_low),
                            min(max_val, stored_high),
                        )
                    else:
                        default_tuple = (min_val, max_val)

                    selected_range = st.slider(
                        "Plage",
                        min_value=min_val,
                        max_value=max_val,
                        value=(int(default_tuple[0]), int(default_tuple[1])),
                        step=1,
                        key=f"sweep_range_{key}",
                        label_visibility="collapsed",
                    )
                else:
                    min_val = float(min_val)
                    max_val = float(max_val)
                    step_val = float(step_val)

                    if stored_range:
                        stored_low_f = float(stored_range[0])
                        stored_high_f = float(stored_range[1])
                        default_tuple = (
                            max(min_val, stored_low_f),
                            min(max_val, stored_high_f),
                        )
                    else:
                        default_tuple = (min_val, max_val)

                    selected_range = st.slider(
                        "Plage",
                        min_value=min_val,
                        max_value=max_val,
                        value=(float(default_tuple[0]), float(default_tuple[1])),
                        step=step_val,
                        key=f"sweep_range_{key}",
                        label_visibility="collapsed",
                    )

            # Sensibilité : Appliquer le multiplicateur global au step de base
            with col_sense:
                # Calculer le step ajusté avec le multiplicateur global
                base_step = step_val
                adjusted_step = base_step * global_sensitivity

                if param_type == "int":
                    # Pour entiers : step ajusté (minimum 1)
                    adjusted_step = max(1, int(round(adjusted_step)))
                    # Afficher l'information sur l'ajustement
                    st.metric(
                        "📊 Step",
                        f"{int(adjusted_step)}",
                        delta=f"×{global_sensitivity:.1f}",
                        label_visibility="collapsed",
                    )
                else:
                    # Pour floats : afficher le step ajusté avec précision
                    st.metric(
                        "📊 Step",
                        f"{adjusted_step:.4f}",
                        delta=f"×{global_sensitivity:.1f}",
                        label_visibility="collapsed",
                    )

        range_preferences[key] = (selected_range[0], selected_range[1])
        param_ranges[key] = (selected_range[0], selected_range[1])
        param_steps[key] = adjusted_step

        # Display combination count for this parameter (using adjusted step)
        range_min, range_max = selected_range
        span = range_max - range_min

        if param_type == "int":
            # For integers: count the values in the range with this step
            n_combinations = len(
                range(int(range_min), int(range_max) + 1, max(1, int(adjusted_step)))
            )
        else:
            # For floats: (span / step)
            n_combinations = span / adjusted_step if adjusted_step > 0 else 1

        # Show the combination count with adjusted step
        comb_text = f"📊 Plage: {range_min} → {range_max} | Step ajusté: {adjusted_step} | Combinaisons: {n_combinations:.1f}"
        st.caption(comb_text)

        # Séparateur visuel entre paramètres
        st.markdown("---")

    st.session_state["strategy_param_ranges"] = range_preferences

    # Calculer le nombre total de combinaisons
    total_combinations = 1
    for key, (min_v, max_v) in param_ranges.items():
        step = param_steps[key]
        if param_types[key] == "int":
            n_values = len(range(int(min_v), int(max_v) + 1, max(1, int(step))))
        else:
            # Utiliser le même calcul que np.linspace pour cohérence
            n_values = max(2, int((max_v - min_v) / step) + 1)
        total_combinations *= n_values

    # === GRAPHIQUE DE DISTRIBUTION DES PLAGES ===
    if param_ranges:
        st.markdown("#### 📊 Visualisation de l'Espace de Recherche")
        col_graph, col_est = st.columns([2, 1])

        with col_graph:
            # Calculer la largeur de chaque plage (normalisée)
            spans = []
            labels = []
            for key, (min_v, max_v) in param_ranges.items():
                span = abs(max_v - min_v)
                spans.append(span)
                labels.append(key)

            # Créer le graphique radar
            fig_dist = go.Figure(
                data=go.Scatterpolar(
                    r=spans,
                    theta=labels,
                    fill="toself",
                    line=dict(color="#26a69a", width=2),
                    fillcolor="rgba(38, 166, 154, 0.3)",
                )
            )
            fig_dist.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(spans) * 1.1] if spans else [0, 1],
                        showticklabels=False,
                    ),
                    bgcolor="rgba(0,0,0,0)",
                ),
                showlegend=False,
                height=300,
                margin=dict(l=40, r=40, t=20, b=20),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dist, use_container_width=True, key="param_distribution_sweep")

        with col_est:
            # === ESTIMATEUR DE TEMPS ===
            st.markdown("#### ⏱️ Estimation")
            if total_combinations > 0:
                # Estimation basée sur benchmarks réels
                # ~2000 tests/sec avec GPU, ~500 tests/sec sans GPU
                tests_per_sec = 2000 if use_gpu else 500
                if use_multigpu:
                    tests_per_sec *= 1.8  # Boost multi-GPU

                estimated_seconds = total_combinations / tests_per_sec
                hours, remainder = divmod(estimated_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)

                if hours >= 1:
                    time_str = f"{int(hours)}h {int(minutes)}m"
                elif minutes >= 1:
                    time_str = f"{int(minutes)}m {int(seconds)}s"
                else:
                    time_str = f"{int(seconds)}s"

                st.metric(
                    "Temps estimé",
                    time_str,
                    delta=f"~{int(tests_per_sec)} tests/sec",
                    delta_color="off",
                )

                # Indicateur de performance
                if estimated_seconds < 60:
                    st.success("🚀 Très rapide")
                elif estimated_seconds < 300:
                    st.info("⚡ Rapide")
                elif estimated_seconds < 1800:
                    st.warning("⏳ Moyen")
                else:
                    st.error("🐢 Long")

                st.caption(f"GPU: {'✅ Multi' if use_multigpu else '✅ Activé' if use_gpu else '❌ Désactivé'}")

    st.markdown("---")

    # Affichage du nombre total de combinaisons
    if total_combinations <= 100000:
        st.success(
            f"✅ **{total_combinations} combinaisons** - Grille optimale (rapide)"
        )
    elif total_combinations <= 1000000:
        st.info(
            f"📊 **{total_combinations} combinaisons** - Grille normale (quelques minutes)"
        )
    elif total_combinations <= 3000000:
        st.warning(
            f"⚠️ **ATTENTION: {total_combinations} combinaisons** - Peut prendre 30-60 min avec GPU"
        )
        st.info("💡 **Note:** Grille large mais faisable avec multi-GPU et 30 workers")
    else:
        st.error(f"❌ **BLOKÉ: {total_combinations} combinaisons trop nombreuses!**")
        st.error("🛑 Cette grille causera un MemoryError (>3M même avec GPU).")
        st.info(
            "✨ **Solutions:**\n1. Augmentez le step (sensibilité) pour tous les paramètres\n2. Réduisez les plages (min/max)\n3. Utilisez Monte-Carlo à la place"
        )

    # Bouton de lancement (désactivé si grille > 3 millions)
    # Bouton d'optimisation de la parallelisation
    opt_disabled = total_combinations > 3000000
    if st.button(
        "Optimiser la parallelisation",
        type="secondary",
        use_container_width=True,
        key="optimize_parallel_btn",
        disabled=opt_disabled,
    ):
        # Construire les parametres pour le sweep (meme logique que le bouton principal)
        scenario_params: dict[str, Any] = {}
        for key, (min_v, max_v) in param_ranges.items():
            step = param_steps[key]
            if param_types[key] == "int":
                values = list(range(int(min_v), int(max_v) + 1, max(1, int(step))))
            else:
                values = np.linspace(
                    min_v, max_v, num=max(2, int((max_v - min_v) / step) + 1)
                ).tolist()
            scenario_params[key] = {"values": values}

        # Completer avec les parametres fixes par defaut
        try:
            all_param_specs = parameter_specs_for(strategy)
        except Exception:
            all_param_specs = {}
        for key, spec in all_param_specs.items():
            if key not in scenario_params:
                value = configured_params.get(
                    key,
                    base_strategy_params.get(
                        key, spec.get("default") if isinstance(spec, dict) else spec
                    ),
                )
                scenario_params[key] = {"value": value}

        # Charger les donnees
        symbol = st.session_state.get("symbol", "BTC")
        timeframe = st.session_state.get("timeframe", "1h")
        start_date = st.session_state.get("start_date")
        end_date = st.session_state.get("end_date")
        try:
            real_data = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)
            if real_data.empty:
                st.error(
                    f"Aucune donnee disponible pour {symbol}/{timeframe} entre {start_date} et {end_date}"
                )
                st.stop()
        except Exception as e:
            st.error(f"Erreur chargement donnees: {e}")
            st.stop()

        st.info("Recherche de la configuration optimale de parallelisation...")
        try:
            report = probe_parallel_configs(
                params_grid=scenario_params,
                data=real_data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_name=strategy,
                try_processes=True,
            )
        except Exception as e:
            st.warning(f"Optimisation de la parallelisation indisponible ({e}). Utilisation d'une configuration par defaut.")
            report = {
                "chosen": {"use_processes": False, "max_workers": max_workers or 8, "probe_throughput_cps": 0.0},
                "total_combos": total_combinations,
                "probes": [],
            }

        chosen = report.get("chosen", {})
        total_combos = int(report.get("total_combos", 0))
        probe_cps = float(chosen.get("probe_throughput_cps", 0.0))
        est_sec = (total_combos / probe_cps) if probe_cps > 0 else 0.0
        est_min, est_s = divmod(int(est_sec), 60)
        if probe_cps > 0:
            st.info(
                f"Config: {'ProcessPool' if chosen.get('use_processes') else 'ThreadPool'} "
                f"| workers={chosen.get('max_workers')} | ETA≈ {est_min}m {est_s}s (sondage)"
            )
        else:
            st.info(
                f"Config: {'ProcessPool' if chosen.get('use_processes') else 'ThreadPool'} | workers={chosen.get('max_workers')}"
            )

        # Lancer le sweep complet avec barre de progression standard
        os.environ["THREADX_FEEDER_AGGR"] = str(st.session_state.get("sweep_feeder_aggr", 10))
        indicator_settings = IndicatorSettings(use_gpu=use_gpu)
        indicator_bank = IndicatorBank(indicator_settings)
        runner = SweepRunner(
            indicator_bank=indicator_bank,
            max_workers=int(chosen.get("max_workers", max_workers or 8)),
            use_multigpu=use_multigpu,
            # Respecte le choix du probe, mais permet de forcer ProcessPool côté UI
            use_processes=bool(chosen.get("use_processes", False) or st.session_state.get("sweep_force_processpool", False)),
        )

        scenario_spec = ScenarioSpec(type="grid", params=scenario_params)
        results = _run_sweep_with_progress(
            runner,
            scenario_spec,
            real_data,
            symbol,
            timeframe,
            strategy,
            total_combos,
        )
        st.session_state["sweep_results"] = results

    # Option IA pour analyse de la meilleure configuration
    st.markdown("---")
    col_llm, col_spacer = st.columns([3, 1])
    with col_llm:
        enable_llm = st.checkbox(
            "🤖 Activer l'analyse IA pour la meilleure configuration",
            value=st.session_state.get("sweep_enable_llm", True),
            key="sweep_enable_llm",
            help="Génère une interprétation intelligente des résultats de la meilleure config via LLM (ajoute ~10s)",
        )

    button_disabled = total_combinations > 3000000
    if st.button(
        "🔬 Lancer le Sweep",
        type="primary",
        use_container_width=True,
        key="run_sweep_btn",
        disabled=button_disabled,
    ):
        # Appliquer l'agressivité du feeder au chemin de lancement direct également
        os.environ["THREADX_FEEDER_AGGR"] = str(st.session_state.get("sweep_feeder_aggr", 10))
        indicator_settings = IndicatorSettings(use_gpu=use_gpu)
        indicator_bank = IndicatorBank(indicator_settings)
        runner = SweepRunner(
            indicator_bank=indicator_bank,
            max_workers=max_workers,
            use_multigpu=use_multigpu,
            use_processes=bool(st.session_state.get("sweep_force_processpool", False)),
        )

        # Sauvegarder la configuration dans l'historique
        _save_config_to_history(
            strategy=strategy,
            strategy_params=configured_params,
            param_ranges=param_ranges,
            global_sensitivity=global_sensitivity,
            config_type="Sweep",
        )

        # Construire les paramètres pour le sweep
        scenario_params: dict[str, Any] = {}
        for key, (min_v, max_v) in param_ranges.items():
            step = param_steps[key]
            if param_types[key] == "int":
                values = list(range(int(min_v), int(max_v) + 1, max(1, int(step))))
            else:
                values = np.linspace(
                    min_v, max_v, num=max(2, int((max_v - min_v) / step) + 1)
                ).tolist()
            scenario_params[key] = {"values": values}

        # 🔥 FIX CRITIQUE: Ajouter TOUS les paramètres par défaut manquants
        # Garantir que min_pnl_pct et autres params sont TOUJOURS présents
        all_param_specs = parameter_specs_for(strategy)
        for key, spec in all_param_specs.items():
            if key not in scenario_params:
                value = configured_params.get(
                    key,
                    base_strategy_params.get(
                        key, spec.get("default") if isinstance(spec, dict) else spec
                    ),
                )
                scenario_params[key] = {"value": value}
                logger.debug(f"Param par défaut ajouté: {key} = {value}")

        # Utiliser run_grid pour explorer toutes les combinaisons
        scenario_spec = ScenarioSpec(type="grid", params=scenario_params)

        # Récupérer les données réelles pour le backtest
        symbol = st.session_state.get("symbol", "BTC")
        timeframe = st.session_state.get("timeframe", "1h")
        start_date = st.session_state.get("start_date")
        end_date = st.session_state.get("end_date")

        try:
            real_data = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)
            if real_data.empty:
                st.error(
                    f"⚠️ Aucune donnée disponible pour {symbol}/{timeframe} entre {start_date} et {end_date}"
                )
                return
            st.session_state.data = real_data
            st.info(
                f"📊 Données chargées: {len(real_data)} barres "
                f"({real_data.index[0].date()} → {real_data.index[-1].date()})"
            )
        except Exception as e:
            st.error(f"❌ Erreur chargement données: {e}")
            return

        try:
            st.markdown("### 🚀 Exécution du Sweep")
            results = _run_sweep_with_progress(
                runner,
                scenario_spec,
                real_data,
                symbol,
                timeframe,
                strategy,
                total_combinations,
            )
            st.session_state["sweep_results"] = results

            st.markdown("---")
            st.markdown("### ⚙️ Configuration d'exécution")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric(
                    "Mode Multi-GPU", "Activé ✅" if use_multigpu else "Désactivé ⊘"
                )
            with col_info2:
                actual_workers = runner.max_workers if runner.max_workers else "Auto"
                st.metric("Workers utilisés", str(actual_workers))
            with col_info3:
                st.metric(
                    "Total des résultats",
                    len(results) if isinstance(results, pd.DataFrame) else 0,
                )
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

        # Tri par PNL décroissant (fallback si colonne PNL absente)
        results_sorted = _sort_results_by_pnl(results_df)

        st.dataframe(results_sorted.head(100), use_container_width=True, height=400)

        best_row = results_sorted.iloc[0]
        st.markdown("#### 🏆 Meilleure configuration")
        st.json(best_row.to_dict())

        with st.expander(
            "🔎 OHLC + trades de la meilleure configuration", expanded=True
        ):
            strategy_name = st.session_state.get(
                "sweep_strategy", context["strategy"]
            )  # noqa: E202
            best_params = _extract_params_from_row(strategy_name, best_row)

            df_price = st.session_state.get("data")
            if not isinstance(df_price, pd.DataFrame) or df_price.empty:
                try:
                    df_price = load_ohlcv(
                        context["symbol"],
                        context["timeframe"],
                        start=context["start_date"],
                        end=context["end_date"],
                    )
                    st.session_state.data = df_price
                except Exception as e:
                    st.error(f"❌ Erreur chargement données prix: {e}")
                    df_price = None

            if isinstance(df_price, pd.DataFrame) and not df_price.empty:
                try:
                    use_gpu_pref = st.session_state.get("sweep_use_gpu", True)
                    enable_llm_analysis = st.session_state.get("sweep_enable_llm", False)
                    result_best = run_backtest_gpu(
                        df=df_price,
                        strategy=strategy_name,
                        params=best_params,
                        symbol=context["symbol"],
                        timeframe=context["timeframe"],
                        use_gpu=use_gpu_pref,
                        enable_monitoring=False,
                        enable_llm=enable_llm_analysis,
                        llm_model="gpt-oss:20b",
                    )
                    authentic = (
                        bool(result_best.metadata.get("gpu_enabled"))
                        if isinstance(result_best.metadata, dict)
                        else False
                    )
                    if not authentic:
                        st.warning(
                            "GPU non utilisé: les trades peuvent être approximatifs (CPU)."
                        )
                    _render_price_with_trades(
                        df_price,
                        result_best.trades,
                        title="Meilleure configuration — OHLC + trades",
                    )
                    
                    # Afficher l'analyse IA si disponible
                    llm_interp = result_best.metrics.get("llm_interpretation")
                    if llm_interp:
                        st.markdown("---")
                        _render_llm_insights(llm_interp)
                    
                    with st.expander("Voir la table des trades", expanded=False):
                        _render_trades_table(result_best.trades)
                except Exception as e:
                    st.error(f"❌ Erreur lors du backtest de la meilleure config: {e}")

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Exporter les résultats Sweep (CSV)",
            csv,
            "sweep_results.csv",
            "text/csv",
            use_container_width=True,
        )


def _build_sweep_grid(
    min_value: float, max_value: float, step: float, value_type: str
) -> np.ndarray:
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


def _render_trades_table(trades: list[dict[str, Any]]) -> None:
    """Table des transactions."""
    if not trades:
        st.info("ℹ️ Aucune transaction enregistrée.")
        return

    trades_df = pd.DataFrame(trades)

    # Formater si colonnes spécifiques existent
    if "profit" in trades_df.columns:
        trades_df["profit"] = trades_df["profit"].apply(lambda x: f"${x:.2f}")

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


def _render_monitoring_section(
    metadata: dict[str, Any] | None, history: pd.DataFrame | None
) -> None:
    """Affiche les diagnostics GPU/CPU et les courbes de monitoring."""
    has_metadata = isinstance(metadata, dict) and bool(metadata)
    has_history = isinstance(history, pd.DataFrame) and not history.empty

    if not has_metadata and not has_history:
        return

    st.markdown("#### 🔍 Diagnostics Système & GPU")

    if has_metadata and metadata is not None:
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
            st.metric(
                "GPU 1 moyen (%)",
                f"{monitor_stats.get('gpu1_mean', 0):.1f}" if monitor_stats else "N/A",
            )
        with col_meta3:
            st.metric(
                "GPU 2 moyen (%)",
                f"{monitor_stats.get('gpu2_mean', 0):.1f}" if monitor_stats else "N/A",
            )
            st.metric(
                "CPU moyen (%)",
                f"{monitor_stats.get('cpu_mean', 0):.1f}" if monitor_stats else "N/A",
            )

        with st.expander("Détails GPU", expanded=False):
            st.write("Périphériques :", devices or "Inconnu")
            if gpu_balance:
                st.write("Balance de charge :", gpu_balance)
            if monitor_stats:
                st.json(monitor_stats)

    if has_history and history is not None:
        df = history.copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["time"].tolist(),
                y=df["cpu"].tolist(),
                name="CPU (%)",
                line=dict(color="#26a69a"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["time"].tolist(),
                y=df["gpu1"].tolist(),
                name="GPU 1 (%)",
                line=dict(color="#42a5f5"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["time"].tolist(),
                y=df["gpu2"].tolist(),
                name="GPU 2 (%)",
                line=dict(color="#ef5350"),
            )
        )
        fig.update_layout(
            height=320,
            template="plotly_dark",
            xaxis_title="Temps (s)",
            yaxis_title="Utilisation (%)",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
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
        if st.button(
            "⏹ Arrêter l'exécution", use_container_width=True, key="global_stop_btn"
        ):
            st.session_state.run_stop_requested = True
            try:  # pragma: no cover - arrêt best-effort
                from threadx.optimization.engine import request_global_stop  # type: ignore

                request_global_stop()
            except Exception:
                pass
            st.warning("Arrêt demandé — tentative d'interruption des tâches en cours.")
    st.markdown("*Optimisez vos paramètres de trading avec Sweep ou Monte-Carlo*")
    st.markdown("---")

    # Onglets principaux
    tab1, tab2 = st.tabs(["🔬 Sweep", "🎲 Monte-Carlo"])

    with tab1:
        _render_optimization_tab()

    with tab2:
        _render_monte_carlo_tab()


if __name__ == "__main__":
    main()


