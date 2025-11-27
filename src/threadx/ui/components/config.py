"""
ThreadX UI Configuration
========================

Reusable configuration and parameter control components.
"""

from datetime import datetime
from typing import Any

import streamlit as st


def normalize_spec(spec: Any) -> dict[str, Any]:
    """Normalizes a parameter specification into a standard dict format."""
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


def render_param_control(
    label: str,
    widget_key: str,
    spec: dict[str, Any],
    prefill: Any,
    range_store: dict[str, tuple[Any, Any]] | None = None,
    store_key: str | None = None,
) -> Any:
    """Renders a control widget for a single parameter."""
    normalized = normalize_spec(spec)
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

    # Range sliders for numeric parameters (optimization context)
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

    # Standard controls
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


def render_indicator_inputs(
    name: str, specs: dict[str, Any], range_store: dict[str, tuple[Any, Any]]
) -> dict[str, Any]:
    """Renders inputs for a set of indicators."""
    prev_indicators = st.session_state.get("indicators", {})
    saved = prev_indicators.get(name, {})
    result: dict[str, Any] = {}

    for key, spec in specs.items():
        normalized = normalize_spec(spec)
        prefill = saved.get(key, normalized.get("default"))
        label = normalized.get("label") or f"{key}".replace("_", " ").title()
        col_key = f"{name}_{key}"
        store_key = f"{name}.{key}"
        result[key] = render_param_control(label, col_key, normalized, prefill, range_store, store_key)

    return result


def render_config_history(key_prefix: str = "") -> dict | None:
    """Renders configuration history with load/delete options."""
    with st.expander("📜 Configuration History", expanded=False):
        history = st.session_state.get("config_history", [])

        if not history:
            st.caption("No saved configurations.")
            return None

        st.caption(f"**{len(history)} saved configuration(s)**")

        for idx, cfg in enumerate(reversed(history)):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(
                        f"**{cfg['type']} - {cfg['strategy']}**  \n"
                        f"📅 {cfg['timestamp']}  \n"
                        f"🎚️ Sensitivity: {cfg['global_sensitivity']}x"
                    )
                with col2:
                    if st.button(
                        "📥 Load",
                        key=f"{key_prefix}load_hist_{len(history) - 1 - idx}",
                        use_container_width=True,
                    ):
                        return cfg
                with col3:
                    if st.button(
                        "🗑️ Del",
                        key=f"{key_prefix}del_hist_{len(history) - 1 - idx}",
                        use_container_width=True,
                    ):
                        st.session_state.config_history.pop(len(history) - 1 - idx)
                        st.rerun()

                st.markdown("---")

        return None


def save_config_to_history(
    strategy: str,
    strategy_params: dict,
    param_ranges: dict,
    global_sensitivity: float = 1.0,
    n_scenarios: int | None = None,
    config_type: str = "Sweep",
) -> None:
    """Saves current configuration to history."""
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

    # Limit history to 20 items
    if len(st.session_state.config_history) > 20:
        st.session_state.config_history = st.session_state.config_history[-20:]
