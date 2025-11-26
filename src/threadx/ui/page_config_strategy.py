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
from typing import Any

import pandas as pd
import streamlit as st

from ..data_access import (
    discover_tokens_and_timeframes,
    get_available_timeframes_for_token,
    load_ohlcv,
)
from .strategy_registry import indicator_specs_for, list_strategies, parameter_specs_for
from .styles import apply_custom_styles
from .components.charts import render_ohlcv_chart
from .components.config import normalize_spec

DEFAULT_SYMBOL = "BTCUSDC"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_START_DATE = date(2024, 12, 1)
DEFAULT_END_DATE = date(2025, 1, 31)


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
                render_ohlcv_chart(df, key="chart_preview")

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
        indicator_specs = indicator_specs_for(strategy)
        param_specs = parameter_specs_for(strategy)
    except KeyError:
        st.error(f"❌ Stratégie inconnue: {strategy}")
        return

    # Initialiser valeurs d'indicateurs avec leurs défauts (onglet supprimé)
    if indicator_specs:
        indicator_values = {
            ind_name: {
                key: normalize_spec(spec).get("default")
                for key, spec in ind_spec.items()
            }
            for ind_name, ind_spec in indicator_specs.items()
        }
    else:
        indicator_values = {}

    # Afficher les indicateurs avec valeurs par défaut pour la stratégie sélectionnée
    if indicator_specs:
        with st.expander("📈 Indicateurs techniques (préréglés)", expanded=False):
            indicator_rows = []
            for ind_name, ind_spec in indicator_specs.items():
                for key, spec in ind_spec.items():
                    normalized = normalize_spec(spec)
                    label = normalized.get("label") or key.replace("_", " ").title()
                    default_val = normalized.get("default")
                    min_val = normalized.get("min")
                    max_val = normalized.get("max")
                    step_val = normalized.get("step")

                    if isinstance(default_val, float):
                        default_fmt = f"{default_val:.4f}".rstrip("0").rstrip(".")
                    else:
                        default_fmt = str(default_val)

                    indicator_rows.append(
                        {
                            "Indicateur": ind_name,
                            "Paramètre": label,
                            "Défaut": default_fmt,
                            "Min": min_val,
                            "Max": max_val,
                            "Pas": step_val,
                        }
                    )

            import pandas as pd

            df_ind = pd.DataFrame(indicator_rows)
            st.dataframe(df_ind, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ Aucun indicateur spécifique pour cette stratégie.")

    # Récupérer les paramètres par défaut depuis le registre (pas besoin de sliders ici)
    from .strategy_registry import base_params_for
    strategy_params = base_params_for(strategy)

    st.markdown("#### 📋 Paramètres par Défaut")
    if not param_specs:
        st.info("ℹ️ Cette stratégie n'a pas de paramètres configurables.")
    else:
        st.info(
            "💡 **Les paramètres par défaut sont chargés automatiquement.** "
            "Allez sur la page **⚡ Optimisation** pour définir des plages à explorer."
        )

        # Afficher les paramètres en lecture seule (tableau compact)
        with st.expander("🔍 Voir les paramètres par défaut", expanded=False):
            params_data = []
            for key, spec in param_specs.items():
                normalized = normalize_spec(spec)
                label = normalized.get("label") or key.replace("_", " ").title()
                default_val = normalized.get("default")
                param_type = normalized.get("type", "")

                # Formater la valeur
                if isinstance(default_val, bool):
                    formatted_val = "✅ Oui" if default_val else "❌ Non"
                elif isinstance(default_val, float):
                    formatted_val = f"{default_val:.4f}".rstrip('0').rstrip('.')
                else:
                    formatted_val = str(default_val)

                params_data.append({
                    "Paramètre": label,
                    "Valeur par défaut": formatted_val,
                    "Type": param_type.upper() if param_type else "AUTO"
                })

            # Afficher comme dataframe compact
            import pandas as pd
            df_params = pd.DataFrame(params_data)
            st.dataframe(df_params, use_container_width=True, hide_index=True)

    st.session_state.strategy = strategy
    st.session_state.indicators = indicator_values
    st.session_state.strategy_params = strategy_params

    st.success(f"✅ Configuration enregistrée : **{strategy}** avec paramètres par défaut")


def main() -> None:
    """Point d'entrée de la page Chargement & Visualisation."""
    apply_custom_styles()
    
    st.title("📊 Chargement des Données")
    st.markdown("*Sélectionnez et prévisualisez vos données de marché*")
    st.markdown("---")

    # Section unique : Chargement et visualisation
    _render_data_section()
    st.markdown("---")
    _render_strategy_section()


if __name__ == "__main__":
    main()
