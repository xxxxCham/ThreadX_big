"""
ThreadX UI Charts
=================

Reusable chart components using Plotly.
"""

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_ohlcv_chart(df: pd.DataFrame, key: str = "chart_preview", height: int = 450) -> None:
    """Renders a modern OHLCV candlestick chart."""
    if df.empty:
        st.warning("⚠️ No data to display")
        return

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

    _apply_chart_layout(fig, height=height)
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_price_with_trades(
    df: pd.DataFrame, trades: list[dict[str, Any]], title: str = "📈 OHLC + Trades", key: str = "ohlc_trades"
) -> None:
    """Renders OHLC chart with trade entry/exit markers."""
    if df.empty:
        st.warning("⚠️ OHLCV data unavailable")
        return

    if not {"open", "high", "low", "close"} <= set(df.columns):
        st.warning("⚠️ Missing OHLC columns")
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

    # Trade markers
    entries_x, entries_y, exits_x, exits_y = [], [], [], []
    entries_color, exits_color = [], []
    
    for t in trades or []:
        side = str(t.get("side", "LONG")).upper()
        # Entry
        if "entry_time" in t and "entry_price" in t:
            entries_x.append(t["entry_time"])
            entries_y.append(t["entry_price"])
            entries_color.append("#42a5f5" if side == "LONG" else "#ab47bc")
        # Exit
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
                name="Entry",
                marker=dict(symbol="triangle-up", size=10, color=entries_color),
            )
        )
    if exits_x:
        fig.add_trace(
            go.Scatter(
                x=list(exits_x),
                y=list(exits_y),
                mode="markers",
                name="Exit",
                marker=dict(symbol="triangle-down", size=10, color=exits_color),
            )
        )

    _apply_chart_layout(fig, height=520, y_title="Price")
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_price_chart(
    df: pd.DataFrame, indicators: dict[str, dict[str, Any]], key: str = "backtest_chart"
) -> None:
    """Renders OHLC chart with indicators (e.g. Bollinger Bands)."""
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

    # Bollinger Bands
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

    _apply_chart_layout(fig, height=500, y_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_equity_curve(equity: pd.Series, key: str = "equity_curve") -> None:
    """Renders equity curve."""
    if equity.empty:
        st.warning("⚠️ Empty equity curve.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=equity.index.to_list(),
            y=equity.values.tolist(),
            mode="lines",
            name="Equity",
            line=dict(color="#26a69a", width=2),
            fill="tozeroy",
            fillcolor="rgba(38, 166, 154, 0.1)",
        )
    )

    # Initial capital line
    fig.add_hline(
        y=equity.iloc[0],
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Initial Capital",
        annotation_position="right",
    )

    _apply_chart_layout(fig, height=300, y_title="Equity ($)")
    st.plotly_chart(fig, use_container_width=True, key=key)


def _apply_chart_layout(fig: go.Figure, height: int = 450, y_title: str = "") -> None:
    """Applies common styling to Plotly figures."""
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        xaxis_title="",
        yaxis_title=y_title,
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor='rgba(128,128,128,0.1)',
        ),
        yaxis=dict(gridcolor='rgba(128,128,128,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a8b2d1', size=11),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
