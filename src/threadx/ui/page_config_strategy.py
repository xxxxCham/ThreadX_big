"""
ThreadX - Page Configuration & Stratégie
=========================================

Page fusionnée combinant la sélection des données et la configuration de stratégie.
Interface moderne et intuitive avec sections collapsibles et organisation optimisée.

Author: ThreadX Framework
Version: 2.0.0 - UI Redesign
"""

from __future__ import annotations

from datetime import date
from typing import List, Dict, Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..data_access import (
    DATA_DIR,
    discover_tokens_and_timeframes,
    get_available_timeframes_for_token,
    load_ohlcv,
)
from ..dataset.validate import validate_dataset
from .strategy_registry import (
    base_params_for,
    indicators_for,
    list_strategies,
)

DEFAULT_SYMBOL = "BTC"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_START_DATE = date(2024, 9, 1)
DEFAULT_END_DATE = date(2024, 9, 10)


def _render_ohlcv_chart(df: pd.DataFrame) -> None:
    """Affiche un graphique en chandelier moderne des données OHLCV."""
    fig = go.Figure()

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

    fig.update_layout(
        height=450,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Prix (USD)",
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor='rgba(128,128,128,0.2)',
        ),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a8b2d1', size=11),
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True, key="chart_preview")


def _render_data_section() -> None:
    """Section de sélection et prévisualisation des données."""
    st.markdown("### 📊 Sélection des Données")

    tokens, _ = discover_tokens_and_timeframes()
    if not tokens:
        st.error("❌ Aucun dataset trouvé. Ajoutez vos fichiers dans le dossier data/.")
        tokens = [DEFAULT_SYMBOL]

    # Récupérer valeurs session
    default_symbol = st.session_state.get("symbol", tokens[0] if tokens else DEFAULT_SYMBOL)
    if default_symbol not in tokens:
        default_symbol = tokens[0]

    # Layout en colonnes pour sélection compacte
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        symbol = st.selectbox(
            "Token",
            options=tokens,
            index=tokens.index(default_symbol) if default_symbol in tokens else 0,
            key="sel_symbol"
        )

    # Timeframes dynamiques
    timeframes = get_available_timeframes_for_token(symbol)
    if not timeframes:
        timeframes = [DEFAULT_TIMEFRAME]

    default_timeframe = st.session_state.get("timeframe", timeframes[0])
    if default_timeframe not in timeframes:
        default_timeframe = timeframes[0]

    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            options=timeframes,
            index=timeframes.index(default_timeframe) if default_timeframe in timeframes else 0,
            key="sel_timeframe"
        )

    default_start = st.session_state.get("start_date", DEFAULT_START_DATE)
    default_end = st.session_state.get("end_date", DEFAULT_END_DATE)

    with col3:
        start_date = st.date_input("Début", value=default_start, key="sel_start")

    with col4:
        end_date = st.date_input("Fin", value=default_end, key="sel_end")

    # Sauvegarder dans session
    st.session_state.symbol = symbol
    st.session_state.timeframe = timeframe
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date

    # Validation
    if start_date > end_date:
        st.error("⚠️ La date de début doit être antérieure à la date de fin.")
        return

    # Bouton de chargement/prévisualisation
    st.markdown("")
    if st.button("🔄 Charger & Prévisualiser", type="primary", use_container_width=True):
        with st.spinner("⏳ Chargement des données..."):
            try:
                df = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)

                if df.empty:
                    st.warning("⚠️ Aucune donnée disponible pour cette plage.")
                    return

                # Sauvegarder dans session
                st.session_state.data = df

                # Afficher résumé
                st.success(
                    f"✅ {len(df)} lignes chargées | "
                    f"{df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')}"
                )

                # Graphique
                st.markdown("#### 📈 Aperçu du Marché")
                _render_ohlcv_chart(df)

                # Stats rapides en colonnes
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Prix moyen", f"${df['close'].mean():.2f}")
                with col_s2:
                    st.metric("Min / Max", f"${df['low'].min():.2f} / ${df['high'].max():.2f}")
                with col_s3:
                    volatility = df['close'].pct_change().std() * 100
                    st.metric("Volatilité", f"{volatility:.2f}%")
                with col_s4:
                    st.metric("Volume moyen", f"{df['volume'].mean():.0f}")

            except FileNotFoundError as exc:
                st.error(f"❌ {exc}")
            except Exception as exc:
                st.error(f"❌ Erreur: {exc}")

    # Afficher info si données déjà chargées
    current_df = st.session_state.get("data")
    if isinstance(current_df, pd.DataFrame) and not current_df.empty:
        st.info(
            f"💾 {len(current_df)} lignes en mémoire pour {symbol}/{timeframe} "
            f"({current_df.index.min().strftime('%Y-%m-%d')} → {current_df.index.max().strftime('%Y-%m-%d')})"
        )


def _render_indicator_inputs(name: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Rendu des inputs pour un indicateur."""
    prev_indicators = st.session_state.get("indicators", {})
    saved = prev_indicators.get(name, {})
    result: Dict[str, Any] = {}

    for key, default in defaults.items():
        prefill = saved.get(key, default)
        label = f"{key}"
        col_key = f"{name}_{key}"

        if isinstance(default, bool):
            result[key] = st.checkbox(label, value=bool(prefill), key=col_key)
        elif isinstance(default, float):
            max_val = max(float(default) * 5, float(default) + 10, 10.0)
            result[key] = st.slider(
                label,
                min_value=0.0,
                max_value=max_val,
                value=float(prefill),
                step=0.1,
                key=col_key,
            )
        elif isinstance(default, int):
            result[key] = st.number_input(
                label, value=int(prefill), min_value=1, step=1, key=col_key
            )
        else:
            result[key] = st.text_input(label, value=str(prefill), key=col_key)

    return result


def _render_strategy_section() -> None:
    """Section de configuration de la stratégie."""
    st.markdown("### ⚙️ Configuration de la Stratégie")

    strategies = list_strategies()
    if not strategies:
        st.error("❌ Aucune stratégie disponible dans le registre.")
        return

    default_strat = st.session_state.get("strategy", strategies[0])
    if default_strat not in strategies:
        default_strat = strategies[0]

    strategy = st.selectbox(
        "Sélectionnez une stratégie",
        strategies,
        index=strategies.index(default_strat),
        key="sel_strategy"
    )

    try:
        indicator_defs = indicators_for(strategy)
        strategy_defaults = base_params_for(strategy)
    except KeyError:
        st.error(f"❌ Stratégie inconnue: {strategy}")
        return

    # Tabs pour organisation
    tab1, tab2 = st.tabs(["📊 Indicateurs", "🎯 Paramètres de Stratégie"])

    with tab1:
        if not indicator_defs:
            st.info("ℹ️ Cette stratégie n'a pas d'indicateurs configurables.")
            indicator_values = {}
        else:
            indicator_values = {}
            for ind_name, ind_defaults in indicator_defs.items():
                with st.expander(f"📈 {ind_name}", expanded=True):
                    indicator_values[ind_name] = _render_indicator_inputs(ind_name, ind_defaults)

    with tab2:
        if not strategy_defaults:
            st.info("ℹ️ Cette stratégie n'a pas de paramètres configurables.")
            strategy_params = {}
        else:
            strategy_params = {}
            prev_params = st.session_state.get("strategy_params", {})

            # Afficher en colonnes si peu de paramètres
            if len(strategy_defaults) <= 4:
                cols = st.columns(2)
                items = list(strategy_defaults.items())
                for idx, (key, default) in enumerate(items):
                    with cols[idx % 2]:
                        prefill = prev_params.get(key, default)
                        param_key = f"strat_param_{key}"

                        if isinstance(default, bool):
                            strategy_params[key] = st.checkbox(key, value=bool(prefill), key=param_key)
                        elif isinstance(default, float):
                            max_val = max(float(default) * 5, float(default) + 10, 10.0)
                            strategy_params[key] = st.slider(
                                key, 0.0, max_val, float(prefill), 0.1, key=param_key
                            )
                        elif isinstance(default, int):
                            strategy_params[key] = st.number_input(
                                key, value=int(prefill), step=1, key=param_key
                            )
                        else:
                            strategy_params[key] = st.text_input(key, value=str(prefill), key=param_key)
            else:
                # Beaucoup de paramètres : affichage vertical
                for key, default in strategy_defaults.items():
                    prefill = prev_params.get(key, default)
                    param_key = f"strat_param_{key}"

                    if isinstance(default, bool):
                        strategy_params[key] = st.checkbox(key, value=bool(prefill), key=param_key)
                    elif isinstance(default, float):
                        max_val = max(float(default) * 5, float(default) + 10, 10.0)
                        strategy_params[key] = st.slider(
                            key, 0.0, max_val, float(prefill), 0.1, key=param_key
                        )
                    elif isinstance(default, int):
                        strategy_params[key] = st.number_input(
                            key, value=int(prefill), step=1, key=param_key
                        )
                    else:
                        strategy_params[key] = st.text_input(key, value=str(prefill), key=param_key)

    # Sauvegarder configuration
    st.session_state.strategy = strategy
    st.session_state.indicators = indicator_values
    st.session_state.strategy_params = strategy_params

    st.success(f"✅ Configuration enregistrée : **{strategy}**")


def main() -> None:
    """Point d'entrée de la page Configuration & Stratégie."""
    st.title("🎯 Configuration & Stratégie")
    st.markdown("*Configurez vos données de marché et votre stratégie de trading*")
    st.markdown("---")

    # Section 1 : Données
    _render_data_section()

    st.markdown("---")

    # Section 2 : Stratégie
    _render_strategy_section()


if __name__ == "__main__":
    main()
