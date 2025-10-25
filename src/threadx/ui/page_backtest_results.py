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


def _render_price_chart(df: pd.DataFrame, indicators: Dict[str, Dict[str, Any]]) -> None:
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

    if st.button("Lancer le backtest", type="primary", use_container_width=True):
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

        st.subheader("Graphique OHLC")
        _render_price_chart(df, indicators)

        st.subheader("Courbe d'equite")
        _render_equity_curve(result.equity)

        st.subheader("Metriques")
        _render_metrics(result.metrics)

        st.subheader("Transactions")
        _render_trades(result.trades)


if __name__ == "__main__":
    main()



