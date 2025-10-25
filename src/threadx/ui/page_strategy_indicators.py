from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from .strategy_registry import (
    base_params_for,
    indicators_for,
    list_strategies,
)


def _pick_strategy(strategies: List[str]) -> str:
    """Return the UI-selected strategy with a safe default."""
    if not strategies:
        st.error("Aucune strategie disponible dans le registre.")
        st.stop()

    default = st.session_state.get("strategy", strategies[0])
    if default not in strategies:
        default = strategies[0]

    index = strategies.index(default)
    return st.selectbox("Strategie", strategies, index=index)


def _coerce_numeric_range(value: float) -> float:
    """Derive a reasonable slider max based on the default value."""
    if value <= 0:
        return 10.0
    return max(value * 5, value + 1)


def _indicator_inputs(name: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render inputs for a given indicator and return updated values."""
    prev_indicators = st.session_state.get("indicators", {})
    saved = prev_indicators.get(name, {})
    result: Dict[str, Any] = {}

    with st.expander(name, expanded=True):
        for key, default in defaults.items():
            prefill = saved.get(key, default)
            label = f"{name}.{key}"

            if isinstance(default, bool):
                result[key] = st.checkbox(label, value=bool(prefill))
            elif isinstance(default, float):
                max_val = float(_coerce_numeric_range(float(default)))
                result[key] = st.slider(
                    label,
                    min_value=0.0,
                    max_value=max_val,
                    value=float(prefill),
                    step=0.1,
                )
            elif isinstance(default, int):
                result[key] = st.number_input(
                    label, value=int(prefill), min_value=1, step=1
                )
            else:
                result[key] = st.text_input(label, value=str(prefill))

    return result


def _strategy_param_inputs(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Render inputs for strategy-level parameters."""
    saved = st.session_state.get("strategy_params", {})
    result: Dict[str, Any] = {}

    for key, default in defaults.items():
        prefill = saved.get(key, default)

        if isinstance(default, bool):
            result[key] = st.checkbox(key, value=bool(prefill))
        elif isinstance(default, float):
            max_val = float(_coerce_numeric_range(float(default)))
            result[key] = st.slider(
                key,
                min_value=0.0,
                max_value=max_val,
                value=float(prefill),
                step=0.1,
            )
        elif isinstance(default, int):
            result[key] = st.number_input(key, value=int(prefill), step=1)
        else:
            result[key] = st.text_input(key, value=str(prefill))

    return result


def main() -> None:
    """Configure la strategie et les indicateurs associes."""
    st.title("Strategie & Indicateurs")

    strategies = list_strategies()
    strategy = _pick_strategy(strategies)

    try:
        indicator_defs = indicators_for(strategy)
    except KeyError:
        st.error(f"Strategie inconnue: {strategy}.")
        st.stop()
    if not indicator_defs:
        st.warning(f"Aucun indicateur defini pour '{strategy}'.")

    st.subheader("Indicateurs requis")
    indicator_values = {
        name: _indicator_inputs(name, defaults) for name, defaults in indicator_defs.items()
    }

    strategy_defaults = base_params_for(strategy)
    st.subheader("Parametres de strategie")
    strategy_params = _strategy_param_inputs(strategy_defaults)

    st.session_state.strategy = strategy
    st.session_state.indicators = indicator_values
    st.session_state.strategy_params = strategy_params

    st.success("Strategie et indicateurs enregistres.")


if __name__ == "__main__":
    main()



