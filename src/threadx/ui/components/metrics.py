"""
ThreadX UI Metrics
==================

Reusable metric display components.
"""

from typing import Any

import pandas as pd
import streamlit as st


def render_metrics(metrics: dict[str, Any]) -> None:
    """Renders performance metrics in a grid."""
    if not metrics:
        st.info("ℹ️ No metrics calculated.")
        return

    # Filter out non-numeric or special keys
    metrics_filtered = {k: v for k, v in metrics.items() if k != "llm_interpretation"}

    if not metrics_filtered:
        st.info("ℹ️ No numeric metrics available.")
        return

    # Display in grid
    metrics_list = list(metrics_filtered.items())
    n_metrics = len(metrics_list)
    n_cols = min(4, n_metrics)

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

    # Export button
    st.markdown("")
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    csv = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Metrics (CSV)",
        csv,
        "metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_llm_insights(llm_interpretation: dict[str, Any] | None) -> None:
    """Renders AI insights if available."""
    if not llm_interpretation:
        return

    st.markdown("---")
    st.markdown("### 🤖 AI Insights")

    # Global interpretation
    interpretation = llm_interpretation.get("interpretation", "")
    if interpretation:
        st.info(interpretation)

    # Strengths
    strengths = llm_interpretation.get("strengths", [])
    if strengths:
        with st.expander("💪 Identified Strengths", expanded=True):
            for strength in strengths:
                st.markdown(f"✓ {strength}")

    # Weaknesses
    weaknesses = llm_interpretation.get("weaknesses", [])
    if weaknesses:
        with st.expander("⚠️ Areas for Improvement", expanded=True):
            for weakness in weaknesses:
                st.markdown(f"⚡ {weakness}")

    # Recommendations
    recommendations = llm_interpretation.get("recommendations", [])
    if recommendations:
        with st.expander("💡 Recommendations", expanded=True):
            for rec in recommendations:
                st.markdown(f"→ {rec}")

    # Metadata
    risk_level = llm_interpretation.get("risk_level", "UNKNOWN")
    suitability = llm_interpretation.get("suitability", "")

    if risk_level or suitability:
        col1, col2 = st.columns(2)
        with col1:
            risk_colors = {
                "LOW": "🟢",
                "MODERATE": "🟡",
                "HIGH": "🔴",
                "UNKNOWN": "⚪"
            }
            risk_icon = risk_colors.get(risk_level, "⚪")
            st.metric("Risk Level", f"{risk_icon} {risk_level}")

        with col2:
            if suitability:
                st.markdown(f"**Suitability:**  \n{suitability}")
