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
    SWEEP_PRESETS,
    base_params_for,
    get_sweep_preset,
    list_strategies,
    parameter_specs_for,
    resolve_range,
    tunable_parameters_for,
)
from threadx.ui.system_monitor import get_global_monitor
from threadx.utils.log import get_logger
from .styles import apply_custom_styles
from .components.charts import (
    render_equity_curve,
    render_ohlcv_chart,
    render_price_chart,
    render_price_with_trades,
)
from .components.config import (
    render_config_history,
    save_config_to_history,
)
from .components.metrics import render_llm_insights, render_metrics

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


<<<<<<< HEAD
=======
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
    """Affiche l'historique des configurations avec navigation.
    
    Args:
        key_prefix: Préfixe pour les clés Streamlit (évite conflits entre onglets)
    
    Returns:
        Configuration chargée ou None
    """
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


>>>>>>> 1b119cb971277c69eb4e50ee864485c021549ced
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
<<<<<<< HEAD
        loaded_config = render_config_history(key_prefix="mc_")
=======
        loaded_config = _render_config_history(key_prefix="mc_")
>>>>>>> 1b119cb971277c69eb4e50ee864485c021549ced
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

    # Configuration depuis sidebar (valeurs globales)
    use_gpu = True  # Toujours activé
    use_multigpu = st.session_state.get("global_use_multigpu", True)
    max_workers = st.session_state.get("global_workers", 30)

    # Afficher statut
    with col_gpu:
        st.metric("🖥️ GPU", "Activé ✅")
    with col_multigpu:
        if use_multigpu:
            st.metric("🔥 Multi-GPU", "Activé ✅")
        else:
            st.metric("🔥 Multi-GPU", "Désactivé")
    with col_workers:
        st.metric("⚡ Workers", f"{max_workers}")

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
        save_config_to_history(
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


def _render_backtest_tab() -> None:
    """Onglet Backtest simple avec option GPU."""
    context = _require_configuration()
    indicators = st.session_state.get("indicators", {})
    params = st.session_state.get("strategy_params", {}) or {}

    _render_config_badge(context)

    st.markdown("### 🚀 Lancer le Backtest")

    # Configuration depuis sidebar (valeurs globales)
    use_gpu = True  # Toujours activé
    monitoring = st.session_state.get("global_monitoring", True)
    enable_ai = st.session_state.get("global_enable_llm", False)

    # Afficher statut de configuration
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        st.metric("🖥️ GPU", "Activé ✅", help="BacktestEngine GPU")
    with col_status2:
        if monitoring:
            st.metric("📊 Monitoring", "Activé ✅", help="Stats temps réel CPU/GPU")
        else:
            st.metric("📊 Monitoring", "Désactivé", help="Gain léger en perfs")

    if enable_ai:
        st.info("🤖 Analyse LLM activée (définie dans sidebar)")
    else:
        st.caption("💡 Analyse LLM désactivée (modifier dans sidebar si besoin)")

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
                        llm_model="gpt-oss:20b",
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
                render_price_chart(data_df, indicators, key="backtest_chart")

            st.markdown("#### Courbe d'équité")
            render_equity_curve(stored_result.equity, key="equity_curve")

            history_df = st.session_state.get("monitor_history")
            _render_monitoring_section(stored_result.metadata, history_df)

        with res_tab2:
            render_metrics(stored_result.metrics)

            # Afficher AI Insights si disponibles
            llm_interp = stored_result.metrics.get("llm_interpretation")
            if llm_interp:
                render_llm_insights(llm_interp)

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
<<<<<<< HEAD
        loaded_config = render_config_history(key_prefix="sweep_")
=======
        loaded_config = _render_config_history(key_prefix="sweep_")
>>>>>>> 1b119cb971277c69eb4e50ee864485c021549ced
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

    # Configuration depuis sidebar (valeurs globales)
    use_gpu = True  # Toujours activé
    use_multigpu = st.session_state.get("global_use_multigpu", True)
    max_workers = st.session_state.get("global_workers", 30)
    feeder_aggr = st.session_state.get("global_feeder_aggr", 16)

    # Afficher statut de configuration
    with col_gpu:
        st.metric("🖥️ GPU", "Activé ✅")
    with col_multigpu:
        if use_multigpu:
            st.metric("🔥 Multi-GPU", "Activé ✅")
        else:
            st.metric("🔥 Multi-GPU", "Désactivé")
    with col_workers:
        st.metric("⚡ Workers", f"{max_workers}", help=f"Feeder: {feeder_aggr}")

    st.caption("💡 Configuration définie dans **sidebar** (section Configuration Globale)")

    # Réglages avancés
    st.markdown("#### Réglages avancés")
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

        # Vérifier si c'est un préréglage centralisé (source unique de vérité)
        preset = get_sweep_preset(key)
        if preset:
            preset_min = preset["min"]
            preset_max = preset["max"]
            preset_step = preset.get("step")
            reason = SWEEP_PRESETS[key]["reason"]
            st.info(f"🔒 Préréglé {preset_min} → {preset_max} - {reason}")
            selected_range = (preset_min, preset_max)
            if preset_step is not None:
                adjusted_step = preset_step if param_type != "int" else max(1, int(round(preset_step)))
            else:
                adjusted_step = 1 if param_type == "int" else step_val * global_sensitivity
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

        # Sélecteur de type de visualisation
        viz_type = st.radio(
            "Type de visualisation",
            ["🎯 Radar normalisé", "📊 Barres horizontales", "📈 Barres avec échelles individuelles"],
            horizontal=True,
            help="Radar = vue d'ensemble, Barres = valeurs précises par paramètre"
        )

        col_graph, col_est = st.columns([2, 1])

        with col_graph:
            # Préparer les données
            spans_raw = []
            labels = []
            min_values = []
            max_values = []

            for key, (min_v, max_v) in param_ranges.items():
                span = abs(max_v - min_v)
                spans_raw.append(span)
                labels.append(key)
                min_values.append(min_v)
                max_values.append(max_v)

            if viz_type == "🎯 Radar normalisé":
                # === RADAR CHART (normalisé) ===
                spans_normalized = []
                hover_texts = []

                # Normaliser chaque span à 100%
                max_span = max(spans_raw) if spans_raw else 1
                for i, (key, span) in enumerate(zip(labels, spans_raw)):
                    spans_normalized.append((span / max_span) * 100)
                    hover_texts.append(f"{key}<br>Min: {min_values[i]:.4g}<br>Max: {max_values[i]:.4g}<br>Span: {span:.4g}")

                fig_dist = go.Figure(
                    data=go.Scatterpolar(
                        r=spans_normalized,
                        theta=labels,
                        fill="toself",
                        line=dict(color="#26a69a", width=2),
                        fillcolor="rgba(38, 166, 154, 0.3)",
                        hovertext=hover_texts,
                        hoverinfo="text",
                    )
                )
                fig_dist.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 110],
                            tickmode='linear',
                            tick0=0,
                            dtick=25,
                            ticksuffix="%",
                            showticklabels=True,
                        ),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    showlegend=False,
                    height=400,
                    margin=dict(l=60, r=60, t=40, b=40),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.caption("📊 Échelle normalisée 0-100% (hover pour valeurs réelles)")

            elif viz_type == "📊 Barres horizontales":
                # === BARRES HORIZONTALES (valeurs normalisées) ===
                max_span = max(spans_raw) if spans_raw else 1
                spans_normalized = [(s / max_span) * 100 for s in spans_raw]

                fig_dist = go.Figure(
                    data=go.Bar(
                        y=labels,
                        x=spans_normalized,
                        orientation='h',
                        marker=dict(
                            color=spans_normalized,
                            colorscale='Teal',
                            showscale=False,
                        ),
                        text=[f"{s:.1f}%<br>({spans_raw[i]:.4g})" for i, s in enumerate(spans_normalized)],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Min: %{customdata[0]:.4g}<br>Max: %{customdata[1]:.4g}<br>Span: %{customdata[2]:.4g}<extra></extra>',
                        customdata=list(zip(min_values, max_values, spans_raw)),
                    )
                )
                fig_dist.update_layout(
                    xaxis=dict(title="Étendue normalisée (%)", range=[0, 110]),
                    yaxis=dict(title=""),
                    height=max(300, len(labels) * 50),
                    margin=dict(l=150, r=80, t=20, b=40),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.caption("📊 Barres horizontales avec valeurs normalisées (hover pour détails)")

            else:  # Barres avec échelles individuelles
                # === SUBPLOTS avec échelle par paramètre ===
                from plotly.subplots import make_subplots

                n_params = len(labels)
                fig_dist = make_subplots(
                    rows=n_params, cols=1,
                    subplot_titles=labels,
                    vertical_spacing=0.08,
                    specs=[[{"type": "bar"}] for _ in range(n_params)]
                )

                for i, (label, min_v, max_v, span) in enumerate(zip(labels, min_values, max_values, spans_raw), 1):
                    fig_dist.add_trace(
                        go.Bar(
                            x=[span],
                            y=[label],
                            orientation='h',
                            marker=dict(color='#26a69a'),
                            text=f"{min_v:.4g} → {max_v:.4g}",
                            textposition='outside',
                            showlegend=False,
                            hovertemplate=f'<b>{label}</b><br>Min: {min_v:.4g}<br>Max: {max_v:.4g}<br>Span: {span:.4g}<extra></extra>',
                        ),
                        row=i, col=1
                    )
                    # Échelle individuelle par axe
                    fig_dist.update_xaxes(range=[0, span * 1.2], row=i, col=1)

                fig_dist.update_layout(
                    height=max(400, n_params * 80),
                    margin=dict(l=150, r=80, t=60, b=40),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.caption("📈 Chaque paramètre a sa propre échelle (valeurs réelles)")

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
        os.environ["THREADX_FEEDER_AGGR"] = str(st.session_state.get("global_feeder_aggr", 16))
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
        os.environ["THREADX_FEEDER_AGGR"] = str(st.session_state.get("global_feeder_aggr", 16))
        indicator_settings = IndicatorSettings(use_gpu=use_gpu)
        indicator_bank = IndicatorBank(indicator_settings)
        runner = SweepRunner(
            indicator_bank=indicator_bank,
            max_workers=max_workers,
            use_multigpu=use_multigpu,
            use_processes=bool(st.session_state.get("sweep_force_processpool", False)),
        )

        # Sauvegarder la configuration dans l'historique
        save_config_to_history(
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
                    render_price_with_trades(
                        df_price,
                        result_best.trades,
                        title="Meilleure configuration — OHLC + trades",
                    )
                    
                    # Afficher l'analyse IA si disponible
                    llm_interp = result_best.metrics.get("llm_interpretation")
                    if llm_interp:
                        st.markdown("---")
                        render_llm_insights(llm_interp)
                    
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


def _render_llm_assisted_sweep() -> None:
    """Onglet Sweep + LLM : Sweep classique avec analyse LLM des résultats."""
    st.info("🚧 **Fonctionnalité en cours de développement**")
    st.markdown("""
    Cet onglet permettra de :
    - Lancer un Sweep classique
    - Analyser les résultats via LLM (DeepSeek, GPT-OSS, etc.)
    - Obtenir des suggestions d'amélioration
    - Ré-itérer avec des plages ajustées
    
    **Pour l'instant, utilisez :**
    - **Sweep Classique** : Optimisation manuelle sans LLM
    - **Multi-Agents Autonome** : Système autonome complet avec 3 agents
    """)
    
    # Placeholder pour futures fonctionnalités
    with st.expander("📋 Roadmap LLM-Assisted Sweep"):
        st.markdown("""
        1. ✅ Interface de base
        2. 🔄 Intégration moteur Sweep existant
        3. 🔄 Analyse LLM post-sweep (un seul agent)
        4. 🔄 Suggestions de nouvelles plages
        5. ⏳ Re-run automatique avec paramètres ajustés
        """)


def _render_autonomous_multi_agents() -> None:
    """Onglet Multi-Agents Autonome : Redirection vers page dédiée."""
    st.success("✅ **Système Multi-Agents actif sur page dédiée**")
    
    st.markdown("""
    Le système Multi-Agents autonome (Analyst, Strategist, Critic) est disponible 
    sur une **page dédiée** avec interface complète.
    
    ### 🧠 Navigation
    """)
    
    col_nav, col_info = st.columns([1, 2])
    
    with col_nav:
        if st.button("🚀 Ouvrir Multi-LLM Optimizer", type="primary", use_container_width=True):
            st.session_state["page"] = "multi_llm"
            st.rerun()
    
    with col_info:
        st.caption("Redirige vers la page 'Multi-LLM Optimizer' avec :")
        st.caption("- 🕵️ **Analyst** : Diagnostic des résultats")
        st.caption("- 💡 **Strategist** : Génération de variantes")
        st.caption("- 🔍 **Critic** : Validation des propositions")
    
    st.markdown("---")
    
    # Aperçu des fonctionnalités
    with st.expander("ℹ️ Aperçu du système Multi-Agents"):
        st.markdown("""
        ### Fonctionnalités principales :
        
        **1. Orchestration autonome**
        - Boucle d'optimisation automatique
        - 3 agents collaboratifs (Analyst, Strategist, Critic)
        - Sélection automatique de la meilleure variante
        
        **2. Interfaces de visualisation**
        - Affichage temps réel des logs
        - Code généré par chaque agent (3 fenêtres)
        - Historique des itérations
        
        **3. Configuration flexible**
        - Mode Simple : Paramètres par défaut auto-chargés
        - Mode Avancé : JSON personnalisé
        - Support de toutes les 9 stratégies
        
        **4. Monitoring système**
        - Surveillance GPU/CPU en temps réel
        - Métriques de performance
        - Historique des ressources
        
        ### 📊 Stratégies supportées (9) :
        - Bollinger Breakout
        - EMA Cross
        - ATR Channel
        - Bollinger Dual
        - Amplitude Hunter
        - MA Crossover
        - Volume Profile Breakout ⭐ (2025)
        - VWAP Momentum Reversion 🏆 (2025)
        - EMA Stoch Scalp ⚡ (2025)
        """)


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
    
    st.markdown("*Optimisez vos paramètres avec 3 approches différentes*")
    st.markdown("---")

<<<<<<< HEAD
    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["� Backtest", "�🔬 Sweep", "🎲 Monte-Carlo"])

    with tab1:
        _render_backtest_tab()

    with tab2:
        _render_optimization_tab()

    with tab3:
        _render_monte_carlo_tab()
=======
    # 3 ONGLETS DISTINCTS : Sweep Classique | Sweep + LLM | Multi-Agents Autonome
    tab_sweep, tab_llm, tab_autonomous = st.tabs([
        "🔬 Sweep Classique", 
        "🤖 Sweep + LLM", 
        "🧠 Multi-Agents Autonome"
    ])

    with tab_sweep:
        st.markdown("### 🔬 Optimisation par Sweep (Grille Exhaustive)")
        st.caption("Balayage exhaustif de paramètres avec réglages manuels")
        _render_optimization_tab()

    with tab_llm:
        st.markdown("### 🤖 Sweep Assisté par LLM")
        st.caption("Analyse LLM des résultats pour suggestions d'amélioration")
        _render_llm_assisted_sweep()

    with tab_autonomous:
        st.markdown("### 🧠 Multi-Agents Autonome")
        st.caption("Système autonome avec Analyst, Strategist, Critic")
        _render_autonomous_multi_agents()
>>>>>>> 1b119cb971277c69eb4e50ee864485c021549ced


if __name__ == "__main__":
    main()

