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
import plotly.graph_objects as go
import streamlit as st

from ..data_access import (
    discover_tokens_and_timeframes,
    get_available_timeframes_for_token,
    load_ohlcv,
)
from .strategy_registry import indicator_specs_for, list_strategies, parameter_specs_for

DEFAULT_SYMBOL = "BTCUSDC"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_START_DATE = date(2024, 12, 1)
DEFAULT_END_DATE = date(2025, 1, 31)


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


def _normalize_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, dict):
        normalized = dict(spec)
        if "type" not in normalized:
            default = normalized.get("default")
            if isinstance(default, bool):
                normalized["type"] = "bool"
            elif isinstance(default, int) and not isinstance(default, bool):
                normalized["type"] = "int"
            elif isinstance(default, float):
                normalized["type"] = "float"
            elif "options" in normalized:
                normalized["type"] = "select"
            else:
                normalized["type"] = "text"
        return normalized

    default = spec
    if isinstance(default, bool):
        inferred_type = "bool"
    elif isinstance(default, int) and not isinstance(default, bool):
        inferred_type = "int"
    elif isinstance(default, float):
        inferred_type = "float"
    else:
        inferred_type = "text"

    return {
        "default": default,
        "type": inferred_type,
    }





def _render_param_control(
    label: str,
    widget_key: str,
    spec: dict[str, Any],
    prefill: Any,
    range_store: dict[str, tuple[Any, Any]] | None = None,
    store_key: str | None = None,
) -> Any:
    normalized = _normalize_spec(spec)
    param_type = normalized.get("type", "text")
    default = normalized.get("default")
    min_value = normalized.get("min")
    max_value = normalized.get("max")
    step = normalized.get("step")
    options = normalized.get("options")
    control = normalized.get("control")
    opt_range = normalized.get("opt_range")

    if prefill is None:
        prefill = default

    # Gestion des sliders de plage pour les paramètres numériques
    if (
        range_store is not None
        and store_key
        and param_type in {"int", "float"}
        and normalized.get("range_slider", True)
    ):
        if opt_range and min_value is None:
            min_value = opt_range[0]
        if opt_range and max_value is None:
            max_value = opt_range[1]

        if param_type == "int":
            step_val = int(step or 1)
            if min_value is None:
                min_value = int(prefill) if prefill is not None else 0
            else:
                min_value = int(min_value)
            if max_value is None:
                max_value = int(prefill + step_val * 10) if prefill is not None else min_value + step_val * 10
            else:
                max_value = int(max_value)

            stored_range = range_store.get(store_key) or opt_range
            if stored_range:
                low, high = map(int, stored_range)
            else:
                center = int(prefill) if prefill is not None else (min_value + max_value) // 2
                low = center - step_val * 5
                high = center + step_val * 5

            low = max(min_value, low)
            high = min(max_value, high)
            if low > high:
                low, high = min_value, max_value

            slider_value = st.slider(
                label,
                min_value=min_value,
                max_value=max_value,
                value=(int(low), int(high)),
                step=step_val,
                key=widget_key,
            )
            slider_value = (int(slider_value[0]), int(slider_value[1]))
            range_store[store_key] = slider_value
            return int(round((slider_value[0] + slider_value[1]) / 2))

        if param_type == "float":
            step_val = float(step or 0.05)
            if min_value is None:
                min_value = float(prefill) - step_val * 10 if prefill is not None else 0.0
            else:
                min_value = float(min_value)
            if max_value is None:
                if prefill is not None:
                    max_value = float(prefill) + step_val * 10
                else:
                    max_value = min_value + step_val * 20
            else:
                max_value = float(max_value)

            stored_range = range_store.get(store_key) or opt_range
            if stored_range:
                low, high = map(float, stored_range)
            else:
                center = float(prefill) if prefill is not None else (min_value + max_value) / 2.0
                span = step_val * 5
                low = center - span
                high = center + span

            low = max(min_value, low)
            high = min(max_value, high)
            if low > high:
                low, high = min_value, max_value

            slider_value = st.slider(
                label,
                min_value=float(min_value),
                max_value=float(max_value),
                value=(float(low), float(high)),
                step=step_val,
                key=widget_key,
            )
            slider_value = (float(slider_value[0]), float(slider_value[1]))
            range_store[store_key] = slider_value
            return float((slider_value[0] + slider_value[1]) / 2.0)

    if min_value is not None and prefill is not None:
        prefill = max(prefill, min_value)
    if max_value is not None and prefill is not None:
        prefill = min(prefill, max_value)

    if param_type == "bool":
        return st.checkbox(label, value=bool(prefill), key=widget_key)

    if options:
        try:
            index = options.index(prefill)
        except ValueError:
            index = 0
        return st.selectbox(label, options=options, index=index, key=widget_key)

    if param_type == "int":
        step_val = int(step or 1)
        if control == "number_input" or min_value is None or max_value is None:
            return st.number_input(
                label,
                value=int(prefill) if prefill is not None else int(default or 0),
                step=step_val,
                key=widget_key,
            )

        min_int = int(min_value)
        max_int = int(max_value)
        value = int(prefill) if prefill is not None else int(default or min_int)
        value = min(max(value, min_int), max_int)
        return st.slider(
            label,
            min_value=min_int,
            max_value=max_int,
            value=value,
            step=step_val,
            key=widget_key,
        )

    if param_type == "float":
        step_val = float(step or 0.1)
        if control == "number_input" or min_value is None or max_value is None:
            return st.number_input(
                label,
                value=float(prefill) if prefill is not None else float(default or 0.0),
                step=step_val,
                key=widget_key,
            )

        min_float = float(min_value)
        max_float = float(max_value)
        value = float(prefill) if prefill is not None else float(default or min_float)
        value = min(max(value, min_float), max_float)
        return st.slider(
            label,
            min_value=min_float,
            max_value=max_float,
            value=value,
            step=step_val,
            key=widget_key,
        )

    return st.text_input(label, value=str(prefill) if prefill is not None else "", key=widget_key)


def _render_indicator_inputs(name: str, specs: dict[str, Any], range_store: dict[str, tuple[Any, Any]]) -> dict[str, Any]:
    """Rendu des inputs pour un indicateur."""
    prev_indicators = st.session_state.get("indicators", {})
    saved = prev_indicators.get(name, {})
    result: dict[str, Any] = {}

    for key, spec in specs.items():
        normalized = _normalize_spec(spec)
        prefill = saved.get(key, normalized.get("default"))
        label = normalized.get("label") or f"{key}".replace("_", " ").title()
        col_key = f"{name}_{key}"
        store_key = f"{name}.{key}"
        result[key] = _render_param_control(label, col_key, normalized, prefill, range_store, store_key)

    return result


def _render_strategy_section() -> None:
    """Section de configuration de la stratégie."""
    st.markdown("### ?? Configuration de la Stratégie")

    strategies = list_strategies()
    if not strategies:
        st.error("? Aucune stratégie disponible dans le registre.")
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
        st.error(f"? Stratégie inconnue: {strategy}")
        return

    # Initialiser valeurs d'indicateurs avec leurs défauts (onglet supprimé)
    if indicator_specs:
        indicator_values = {
            ind_name: {
                key: _normalize_spec(spec).get("default")
                for key, spec in ind_spec.items()
            }
            for ind_name, ind_spec in indicator_specs.items()
        }
    else:
        indicator_values = {}

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
                normalized = _normalize_spec(spec)
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
    st.title("📊 Chargement des Données")
    st.markdown("*Sélectionnez et prévisualisez vos données de marché*")
    st.markdown("---")

    # Section unique : Chargement et visualisation
    _render_data_section()
    st.markdown("---")
    _render_strategy_section()


if __name__ == "__main__":
    main()
