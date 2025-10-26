from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from .strategy_registry import indicator_specs_for, list_strategies, parameter_specs_for


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


def _normalize_spec(spec: Any) -> Dict[str, Any]:
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


def _render_param_control(label: str, widget_key: str, spec: Dict[str, Any], prefill: Any) -> Any:
    normalized = _normalize_spec(spec)
    param_type = normalized.get("type", "text")
    default = normalized.get("default")
    min_value = normalized.get("min")
    max_value = normalized.get("max")
    step = normalized.get("step")
    options = normalized.get("options")
    control = normalized.get("control")

    if prefill is None:
        prefill = default

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


def _indicator_inputs(name: str, specs: Dict[str, Any]) -> Dict[str, Any]:
    """Render inputs for a given indicator and return updated values."""
    prev_indicators = st.session_state.get("indicators", {})
    saved = prev_indicators.get(name, {})
    result: Dict[str, Any] = {}

    with st.expander(name, expanded=True):
        for key, raw_spec in specs.items():
            spec = _normalize_spec(raw_spec)
            prefill = saved.get(key, spec.get("default"))
            label = spec.get("label") or f"{name}.{key}"
            widget_key = f"{name}_{key}"
            result[key] = _render_param_control(label, widget_key, spec, prefill)

    return result


def _strategy_param_inputs(specs: Dict[str, Any]) -> Dict[str, Any]:
    """Render inputs for strategy-level parameters."""
    saved = st.session_state.get("strategy_params", {})
    result: Dict[str, Any] = {}

    for key, raw_spec in specs.items():
        spec = _normalize_spec(raw_spec)
        label = spec.get("label") or key.replace("_", " ").title()
        prefill = saved.get(key, spec.get("default"))
        widget_key = f"strat_param_{key}"
        result[key] = _render_param_control(label, widget_key, spec, prefill)

    return result


def main() -> None:
    """Configure la strategie et les indicateurs associes."""
    st.title("Strategie & Indicateurs")

    strategies = list_strategies()
    strategy = _pick_strategy(strategies)

    try:
        indicator_specs = indicator_specs_for(strategy)
        param_specs = parameter_specs_for(strategy)
    except KeyError:
        st.error(f"Strategie inconnue: {strategy}.")
        st.stop()

    if not indicator_specs:
        st.warning(f"Aucun indicateur defini pour '{strategy}'.")

    st.subheader("Indicateurs requis")
    indicator_values = {
        name: _indicator_inputs(name, defaults) for name, defaults in indicator_specs.items()
    }

    st.subheader("Parametres de strategie")
    strategy_params = _strategy_param_inputs(param_specs)

    st.session_state.strategy = strategy
    st.session_state.indicators = indicator_values
    st.session_state.strategy_params = strategy_params

    st.success("Strategie et indicateurs enregistres.")


if __name__ == "__main__":
    main()



