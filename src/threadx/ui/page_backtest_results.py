from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .backtest_bridge import run_backtest, BacktestResult
from ..data_access import load_ohlcv


def _require_selection() -> Dict[str, Any]:
    """Retrieve the selection stored in session_state or stop the page."""
    required_keys = ("symbol", "timeframe", "start_date", "end_date", "strategy")
    missing = [key for key in required_keys if key not in st.session_state]
    if missing:
        st.info(
            "Veuillez completer les etapes precedentes (selection et strategie) "
            "avant de lancer un backtest."
        )
        st.stop()

    data_frame = st.session_state.get("data")
    if not isinstance(data_frame, pd.DataFrame) or data_frame.empty:
        st.warning("Chargez des donnees avant d'executer un backtest.")
        st.stop()

    return {key: st.session_state[key] for key in required_keys}


def _render_selection_badge(context: Dict[str, Any]) -> None:
    st.caption(
        f"{context['symbol']} @ {context['timeframe']} | "
        f"{context['start_date']} -> {context['end_date']} | {context['strategy']}"
    )


def _render_price_chart(
    df: pd.DataFrame,
    indicators: Dict[str, Dict[str, Any]],
    trades: List[Dict[str, Any]] | None = None,
    marker_scale: float = 1.0,
    offset_factor: float = 0.02,
) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        )
    )

    bollinger = indicators.get("bollinger", {})
    if {"window", "std"} <= set(bollinger.keys()) and not df["close"].empty:
        window = int(bollinger["window"])
        std_mult = float(bollinger["std"])
        rolling_close = df["close"].rolling(window, min_periods=window)
        mid = rolling_close.mean()
        std = rolling_close.std()
        fig.add_trace(go.Scatter(x=df.index, y=mid, name="BB mid", mode="lines"))
        fig.add_trace(go.Scatter(x=df.index, y=mid + std_mult * std, name="BB up", mode="lines"))
        fig.add_trace(go.Scatter(x=df.index, y=mid - std_mult * std, name="BB low", mode="lines"))

    trades = trades or []
    marker_scale = max(marker_scale, 0.1)
    offset_factor = max(offset_factor, 0.001)

    if trades:
        entry_x: List[pd.Timestamp] = []
        entry_y: List[float] = []
        entry_text: List[str] = []
        entry_color: List[str] = []
        entry_size: List[float] = []
        entry_symbols: List[str] = []

        exit_x: List[pd.Timestamp] = []
        exit_y: List[float] = []
        exit_text: List[str] = []
        exit_color: List[str] = []
        exit_size: List[float] = []

        marker_shapes: List[go.layout.Shape] = []

        entry_base_size = 11.0 * marker_scale
        exit_base_size = 13.0 * marker_scale

        index_tz = getattr(df.index, "tz", None)

        def _resolve_timestamp(raw: Any) -> pd.Timestamp | None:
            if raw is None:
                return None
            try:
                ts = pd.to_datetime(raw)
            except (TypeError, ValueError):
                return None
            if isinstance(ts, pd.DatetimeIndex):
                ts = ts[0]
            if pd.isna(ts):
                return None
            if index_tz is not None:
                if ts.tzinfo is None:
                    ts = ts.tz_localize(index_tz)
                else:
                    ts = ts.tz_convert(index_tz)
            else:
                if ts.tzinfo is not None:
                    ts = ts.tz_convert(None)
            return ts

        def _resolve_price(raw: Any) -> float | None:
            if raw is None:
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            return value

        def _lookup_row(ts: pd.Timestamp | None) -> pd.Series | None:
            if ts is None:
                return None
            try:
                row = df.loc[ts]
            except KeyError:
                try:
                    idx = df.index.get_indexer([ts], method="nearest")[0]
                except Exception:
                    return None
                if idx == -1:
                    return None
                row = df.iloc[idx]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            return row

        for trade in trades:
            side_raw = trade.get("side") or trade.get("direction") or trade.get("type")
            side = str(side_raw).upper() if isinstance(side_raw, str) else ""
            is_short = side == "SHORT"
            is_long = not is_short

            entry_time = trade.get("entry_time") or trade.get("entry_ts")
            exit_time = trade.get("exit_time") or trade.get("exit_ts")
            entry_ts = _resolve_timestamp(entry_time)
            exit_ts = _resolve_timestamp(exit_time)

            entry_price = _resolve_price(
                trade.get("price_entry", trade.get("entry_price"))
            )
            exit_price = _resolve_price(
                trade.get("price_exit", trade.get("exit_price"))
            )

            pnl_value = _resolve_price(trade.get("pnl"))

            entry_row = _lookup_row(entry_ts)
            exit_row = _lookup_row(exit_ts)

            def _compute_offset(row: pd.Series | None, reference: float | None) -> tuple[float, float, float]:
                if row is None:
                    if reference is None:
                        return (0.0, 0.0, 0.0)
                    return (reference, reference, abs(reference) * offset_factor)
                try:
                    high_val = _resolve_price(row.get("high"))
                    low_val = _resolve_price(row.get("low"))
                except AttributeError:
                    high_val = None
                    low_val = None
                if high_val is None:
                    high_val = reference if reference is not None else _resolve_price(row.get("close"))
                if low_val is None:
                    low_val = reference if reference is not None else _resolve_price(row.get("close"))
                if high_val is None or low_val is None:
                    high_val = low_val = reference if reference is not None else 0.0
                candle_range = max(high_val - low_val, abs(reference or 0.0) * 0.001, 0.1)
                vertical_offset = candle_range * offset_factor
                return float(high_val), float(low_val), float(vertical_offset)

            entry_high, entry_low, entry_offset = _compute_offset(entry_row, entry_price)
            exit_high, exit_low, exit_offset = _compute_offset(exit_row, exit_price)

            if entry_ts is not None and entry_price is not None:
                marker_y = (
                    (entry_low - entry_offset) if is_long else (entry_high + entry_offset)
                )
                entry_x.append(entry_ts)
                entry_y.append(marker_y)
                label_side = "Long" if is_long else "Short"
                entry_text.append(f"{label_side} • Entrée {entry_price:.2f}")
                entry_color.append("#00d26a" if is_long else "#ff4d4f")
                entry_symbols.append("triangle-up" if is_long else "triangle-down")
                entry_size.append(entry_base_size)
                marker_shapes.append(
                    go.layout.Shape(
                        type="line",
                        x0=entry_ts,
                        x1=entry_ts,
                        y0=entry_price,
                        y1=marker_y,
                        xref="x",
                        yref="y",
                        line=dict(
                            color="#00d26a" if is_long else "#ff4d4f",
                            width=1,
                            dash="dot",
                        ),
                    )
                )

            if exit_ts is not None and exit_price is not None:
                marker_y_exit = (
                    (exit_high + exit_offset) if is_long else (exit_low - exit_offset)
                )
                exit_x.append(exit_ts)
                exit_y.append(marker_y_exit)
                direction = "Gain" if pnl_value is not None and pnl_value >= 0 else "Perte"
                exit_label = f"{direction} • Sortie {exit_price:.2f}"
                if pnl_value is not None:
                    exit_label = f"{exit_label} • PnL: {pnl_value:.2f}"
                exit_text.append(exit_label)
                exit_color.append("#00d26a" if pnl_value is not None and pnl_value >= 0 else "#ff4d4f")
                exit_size.append(exit_base_size)
                marker_shapes.append(
                    go.layout.Shape(
                        type="line",
                        x0=exit_ts,
                        x1=exit_ts,
                        y0=exit_price,
                        y1=marker_y_exit,
                        xref="x",
                        yref="y",
                        line=dict(
                            color="#00d26a" if pnl_value is not None and pnl_value >= 0 else "#ff4d4f",
                            width=1,
                            dash="dot",
                        ),
                    )
                )

        if entry_x:
            fig.add_trace(
                go.Scatter(
                    x=entry_x,
                    y=entry_y,
                    mode="markers",
                    marker=dict(
                        symbol=entry_symbols,
                        size=entry_size,
                        color=entry_color,
                        line=dict(width=1, color="#1f1f2e"),
                    ),
                    name="Entrées trade",
                    text=entry_text,
                    hovertemplate="%{text}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
                )
            )

        if exit_x:
            fig.add_trace(
                go.Scatter(
                    x=exit_x,
                    y=exit_y,
                    mode="markers",
                    marker=dict(
                        symbol="diamond",
                        size=exit_size,
                        color=exit_color,
                        line=dict(width=1, color="#1f1f2e"),
                    ),
                    name="Sorties trade",
                    text=exit_text,
                    hovertemplate="%{text}<br>%{x|%Y-%m-%d %H:%M}<extra></extra>",
                )
            )

        if marker_shapes:
            existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
            fig.update_layout(shapes=existing_shapes + marker_shapes)

    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _render_equity_curve(equity: pd.Series) -> None:
    if equity.empty:
        st.warning("Courbe d'equite vide.")
        return
    st.line_chart(equity, use_container_width=True)


def _render_metrics(metrics: Dict[str, Any]) -> None:
    if not metrics:
        st.info("Aucune metrique calculee.")
        return

    columns = st.columns(len(metrics))
    for column, (key, value) in zip(columns, metrics.items()):
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        column.metric(key, formatted)

    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metrique", "Valeur"])
    csv = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Exporter metriques (CSV)",
        csv,
        "metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_trades(trades: List[Dict[str, Any]]) -> None:
    if not trades:
        st.info("Aucune transaction enregistree.")
        return

    st.dataframe(pd.DataFrame(trades), use_container_width=True)


def main() -> None:
    """Page 3 : chargement des donnees, execution et affichage des resultats."""
    st.title("Backtest & Resultats")

    context = _require_selection()
    indicators = st.session_state.get("indicators", {})
    params = st.session_state.get("strategy_params", {})

    _render_selection_badge(context)

    run_clicked = st.button("Lancer le backtest", type="primary", use_container_width=True)

    if run_clicked:
        try:
            df = load_ohlcv(
                context["symbol"],
                context["timeframe"],
                start=context["start_date"],
                end=context["end_date"],
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # pragma: no cover - defensive UI safeguard
            st.error(f"Impossible de charger les donnees: {exc}")
            return

        if df.empty:
            st.warning("Dataset vide pour cette plage.")
            return

        try:
            result = run_backtest(df=df, strategy=context["strategy"], params=params)
        except Exception as exc:
            st.error(f"Backtest interrompu: {exc}")
            return

        st.session_state.backtest_results = result
        st.session_state.data = df

    stored_result = st.session_state.get("backtest_results")
    stored_df = st.session_state.get("data")

    marker_scale_default = st.session_state.get("trade_marker_scale", 1.0)
    marker_offset_default = st.session_state.get("trade_marker_offset", 0.02)

    with st.expander("Configuration des marqueurs de trades"):
        marker_scale = st.select_slider(
            "Taille des symboles",
            options=[1.0, 1.3, 1.5, 2.0],
            value=marker_scale_default if marker_scale_default in [1.0, 1.3, 1.5, 2.0] else 1.0,
            format_func=lambda value: f"{value:.1f}×",
        )
        marker_offset_percent = st.slider(
            "Décalage vertical (%)",
            min_value=1,
            max_value=10,
            value=int(round(marker_offset_default * 100)),
            step=1,
        )

    st.session_state.trade_marker_scale = marker_scale
    st.session_state.trade_marker_offset = marker_offset_percent / 100.0

    if (
        isinstance(stored_result, BacktestResult)
        and isinstance(stored_df, pd.DataFrame)
        and not stored_df.empty
    ):
        st.subheader("Graphique OHLC")
        _render_price_chart(
            stored_df,
            indicators,
            stored_result.trades,
            marker_scale=st.session_state.trade_marker_scale,
            offset_factor=st.session_state.trade_marker_offset,
        )

        st.subheader("Courbe d'equite")
        _render_equity_curve(stored_result.equity)

        st.subheader("Metriques")
        _render_metrics(stored_result.metrics)

        st.subheader("Transactions")
        _render_trades(stored_result.trades)
    else:
        st.info("Lancez un backtest pour afficher les resultats.")


if __name__ == "__main__":
    main()



