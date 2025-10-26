"""
ThreadX - Page Backtest & Optimisation
=======================================

Page fusionnée combinant le backtest simple et l'optimisation Sweep.
Interface organisée en onglets pour une navigation intuitive.

Author: ThreadX Framework
Version: 2.0.0 - UI Redesign
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .backtest_bridge import run_backtest, BacktestResult
from ..data_access import load_ohlcv
from .strategy_registry import list_strategies


def _require_configuration() -> Dict[str, Any]:
    """Vérifie que la configuration est complète."""
    required_keys = ("symbol", "timeframe", "start_date", "end_date", "strategy")
    missing = [key for key in required_keys if key not in st.session_state]

    if missing:
        st.warning(
            f"⚠️ Configuration incomplète. "
            f"Veuillez d'abord configurer : {', '.join(missing)}"
        )
        st.info("👈 Allez sur la page **Configuration & Stratégie** pour commencer.")
        st.stop()

    data_frame = st.session_state.get("data")
    if not isinstance(data_frame, pd.DataFrame) or data_frame.empty:
        st.warning("⚠️ Aucune donnée chargée.")
        st.info("👈 Retournez sur **Configuration & Stratégie** et cliquez sur 'Charger & Prévisualiser'.")
        st.stop()

    return {key: st.session_state[key] for key in required_keys}


def _render_config_badge(context: Dict[str, Any]) -> None:
    """Affiche un badge récapitulatif de la configuration."""
    st.info(
        f"📊 **{context['symbol']}** @ {context['timeframe']} | "
        f"📅 {context['start_date']} → {context['end_date']} | "
        f"⚙️ {context['strategy']}"
    )


def _render_price_chart(df: pd.DataFrame, indicators: Dict[str, Dict[str, Any]]) -> None:
    """Graphique OHLC avec indicateurs."""
    fig = go.Figure()

    # Candlestick
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

    # Bollinger Bands si configuré
    bollinger = indicators.get("bollinger", {})
    if {"window", "std"} <= set(bollinger.keys()) and not df["close"].empty:
        window = int(bollinger["window"])
        std_mult = float(bollinger["std"])
        rolling_close = df["close"].rolling(window, min_periods=window)
        mid = rolling_close.mean()
        std = rolling_close.std()

        fig.add_trace(go.Scatter(
            x=df.index, y=mid, name="BB Mid",
            mode="lines", line=dict(color='#ffa726', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=mid + std_mult * std, name="BB Upper",
            mode="lines", line=dict(color='#42a5f5', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=mid - std_mult * std, name="BB Lower",
            mode="lines", line=dict(color='#42a5f5', width=1, dash='dash')
        ))

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Prix (USD)",
        xaxis=dict(rangeslider=dict(visible=False), gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a8b2d1', size=11),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True, key="backtest_chart")


def _render_equity_curve(equity: pd.Series) -> None:
    """Courbe d'équité moderne."""
    if equity.empty:
        st.warning("⚠️ Courbe d'équité vide.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='Équité',
        line=dict(color='#26a69a', width=2),
        fill='tozeroy',
        fillcolor='rgba(38, 166, 154, 0.1)',
    ))

    # Ligne initiale
    fig.add_hline(y=equity.iloc[0], line_dash="dash",
                  line_color="gray", opacity=0.5,
                  annotation_text="Capital initial",
                  annotation_position="right")

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        template="plotly_dark",
        xaxis_title="",
        yaxis_title="Équité ($)",
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a8b2d1', size=11),
        hovermode='x unified',
    )

    st.plotly_chart(fig, use_container_width=True, key="equity_curve")


def _render_metrics(metrics: Dict[str, Any]) -> None:
    """Métriques de performance en cartes."""
    if not metrics:
        st.info("ℹ️ Aucune métrique calculée.")
        return

    # Organiser métriques en colonnes
    metrics_list = list(metrics.items())
    n_metrics = len(metrics_list)
    n_cols = min(4, n_metrics)

    # Afficher en grille
    for i in range(0, n_metrics, n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            if i + j < n_metrics:
                key, value = metrics_list[i + j]
                with col:
                    formatted = f"{value:.4f}" if isinstance(value, float) else value
                    # Couleur delta basée sur valeur
                    delta_color = "normal"
                    if isinstance(value, (int, float)):
                        delta_color = "normal" if value > 0 else "inverse"

                    st.metric(
                        label=key.replace("_", " ").title(),
                        value=formatted,
                    )

    # Bouton export
    st.markdown("")
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Métrique", "Valeur"])
    csv = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Exporter les métriques (CSV)",
        csv,
        "metrics.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_trades_table(trades: List[Dict[str, Any]]) -> None:
    """Table des transactions."""
    if not trades:
        st.info("ℹ️ Aucune transaction enregistrée.")
        return

    trades_df = pd.DataFrame(trades)

    # Formater si colonnes spécifiques existent
    if 'profit' in trades_df.columns:
        trades_df['profit'] = trades_df['profit'].apply(lambda x: f"${x:.2f}")

    st.dataframe(
        trades_df,
        use_container_width=True,
        height=300,
    )

    # Bouton export
    csv = trades_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Exporter les trades (CSV)",
        csv,
        "trades.csv",
        "text/csv",
        use_container_width=True,
    )


def _render_backtest_tab() -> None:
    """Onglet Backtest Simple."""
    context = _require_configuration()
    indicators = st.session_state.get("indicators", {})
    params = st.session_state.get("strategy_params", {})

    _render_config_badge(context)

    st.markdown("### 🚀 Lancer le Backtest")

    if st.button("▶️ Exécuter le Backtest", type="primary", use_container_width=True, key="run_backtest_btn"):
        with st.spinner("⏳ Exécution du backtest en cours..."):
            try:
                # Charger données
                df = load_ohlcv(
                    context["symbol"],
                    context["timeframe"],
                    start=context["start_date"],
                    end=context["end_date"],
                )

                if df.empty:
                    st.error("❌ Dataset vide pour cette plage.")
                    return

                # Exécuter backtest
                result = run_backtest(df=df, strategy=context["strategy"], params=params)

                # Sauvegarder résultats
                st.session_state.backtest_results = result
                st.session_state.data = df

                st.success("✅ Backtest terminé avec succès!")

            except FileNotFoundError as exc:
                st.error(f"❌ {exc}")
                return
            except Exception as exc:
                st.error(f"❌ Erreur lors du backtest: {exc}")
                return

    # Afficher résultats si disponibles
    result: BacktestResult = st.session_state.get("backtest_results")
    if result:
        st.markdown("---")
        st.markdown("### 📈 Résultats du Backtest")

        # Onglets pour organiser les résultats
        res_tab1, res_tab2, res_tab3 = st.tabs(["📊 Graphiques", "📉 Métriques", "💼 Transactions"])

        with res_tab1:
            st.markdown("#### Prix & Indicateurs")
            data_df = st.session_state.get("data")
            if isinstance(data_df, pd.DataFrame):
                _render_price_chart(data_df, indicators)

            st.markdown("#### Courbe d'Équité")
            _render_equity_curve(result.equity)

        with res_tab2:
            _render_metrics(result.metrics)

        with res_tab3:
            _render_trades_table(result.trades)


def _render_optimization_tab() -> None:
    """Onglet Optimisation Sweep."""
    st.markdown("### 🎯 Optimisation des Paramètres (Sweep)")

    context = _require_configuration()
    data = st.session_state.get("data")

    if not isinstance(data, pd.DataFrame) or data.empty:
        st.warning("⚠️ Chargez d'abord des données.")
        return

    strategies = list_strategies()
    if not strategies:
        st.error("❌ Aucune stratégie disponible.")
        return

    _render_config_badge(context)

    # Configuration du sweep
    st.markdown("#### Configuration du Sweep")

    col1, col2, col3 = st.columns(3)

    with col1:
        strategy = st.selectbox(
            "Stratégie à optimiser",
            strategies,
            index=strategies.index(context["strategy"]) if context["strategy"] in strategies else 0,
            key="sweep_strategy"
        )

    with col2:
        param_name = st.text_input(
            "Paramètre à optimiser",
            value="window",
            key="sweep_param"
        )

    with col3:
        capital_initial = st.number_input(
            "Capital initial (€)",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=1000,
            key="sweep_capital",
            help="Capital de départ pour calculer le PNL en euros"
        )

    # Plage de valeurs
    col4, col5 = st.columns(2)
    with col4:
        min_val, max_val = st.slider(
            "Plage de valeurs",
            5, 50, (10, 30),
            key="sweep_range"
        )

    with col5:
        step = st.number_input(
            "Pas d'incrémentation",
            min_value=1,
            max_value=10,
            value=1,
            key="sweep_step"
        )

    # Lancer le sweep
    if st.button("🔬 Lancer l'Optimisation", type="primary", use_container_width=True, key="run_sweep_btn"):
        with st.spinner("⏳ Optimisation en cours... Cela peut prendre du temps."):
            grid = np.arange(min_val, max_val + step, step)
            results = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, value in enumerate(grid):
                status_text.text(f"Test {idx+1}/{len(grid)}: {param_name}={value}")
                progress_bar.progress((idx + 1) / len(grid))

                params = {param_name: value}
                try:
                    result = run_backtest(data, strategy, params)

                    # Extraire métriques
                    total_return = result.metrics.get("total_return", 0)
                    sharpe = result.metrics.get("sharpe_ratio", 0)

                    # Calculer PNL en euros
                    pnl_euros = capital_initial * total_return

                    # Calculer équité finale
                    equity_final = capital_initial * (1 + total_return)

                    # Calculer max drawdown
                    equity = result.equity * capital_initial
                    running_max = equity.expanding().max()
                    drawdown = (equity - running_max) / running_max
                    max_dd = drawdown.min() if len(drawdown) > 0 else 0

                    # Compter trades et calculer taille moyenne
                    nb_trades = len(result.trades)
                    avg_trade_size = 0
                    if nb_trades > 0 and result.trades:
                        # Taille moyenne = PNL moyen par trade
                        avg_trade_pnl = sum(t.get("pnl", 0) for t in result.trades) / nb_trades
                        avg_trade_size = abs(avg_trade_pnl)

                    results.append({
                        "param": value,
                        "sharpe": sharpe,
                        "return_pct": total_return * 100,  # En pourcentage
                        "pnl_euros": pnl_euros,
                        "equity_final": equity_final,
                        "max_dd": max_dd * 100,  # En pourcentage
                        "nb_trades": nb_trades,
                        "avg_trade_size": avg_trade_size,
                    })
                except Exception as exc:
                    st.error(f"❌ Sweep interrompu: {exc}")
                    return

            st.session_state.sweep_results = pd.DataFrame(results)
            st.session_state.sweep_capital = capital_initial
            status_text.text("✅ Optimisation terminée!")
            progress_bar.empty()

    # Afficher résultats sweep
    sweep_df = st.session_state.get("sweep_results")
    capital = st.session_state.get("sweep_capital", 10000)

    if isinstance(sweep_df, pd.DataFrame) and not sweep_df.empty:
        st.markdown("---")
        st.markdown("### 📊 Résultats de l'Optimisation")

        # Identifier meilleur paramètre
        best_idx = sweep_df["sharpe"].idxmax()
        best_param = sweep_df.loc[best_idx, "param"]
        best_sharpe = sweep_df.loc[best_idx, "sharpe"]
        best_pnl = sweep_df.loc[best_idx, "pnl_euros"]

        # Afficher résumé
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        with col_sum1:
            st.metric("🏆 Meilleur Paramètre", f"{best_param}")
        with col_sum2:
            st.metric("📈 Sharpe Ratio", f"{best_sharpe:.3f}")
        with col_sum3:
            st.metric("💰 PNL", f"{best_pnl:.2f} €", delta=f"{best_pnl/capital*100:.1f}%")
        with col_sum4:
            best_equity = sweep_df.loc[best_idx, "equity_final"]
            st.metric("💼 Capital Final", f"{best_equity:.2f} €")

        st.markdown("---")

        # GRAPHIQUES PRINCIPAUX
        st.markdown("#### 📈 Visualisations")

        # Grande section: Sharpe Ratio (principal)
        st.markdown("##### 🎯 Sharpe Ratio - Indicateur de Performance Ajusté au Risque")
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(
            x=sweep_df["param"],
            y=sweep_df["sharpe"],
            mode="lines+markers",
            name="Sharpe Ratio",
            line=dict(color='#26a69a', width=3),
            marker=dict(size=10, line=dict(width=2, color='#1e5c4f'))
        ))
        # Marquer l'optimal
        fig_sharpe.add_vline(
            x=best_param,
            line_dash="dash",
            line_color="#ffd700",
            line_width=2,
            annotation_text=f"⭐ Optimal: {best_param}",
            annotation_position="top"
        )
        fig_sharpe.update_layout(
            xaxis_title=f"Paramètre: {param_name}",
            yaxis_title="Sharpe Ratio",
            template="plotly_dark",
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#a8b2d1'),
            hovermode='x unified',
        )
        st.plotly_chart(fig_sharpe, use_container_width=True, key="sweep_sharpe_main")

        # Deuxième ligne: PNL et Max Drawdown
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("##### 💰 PNL en Euros")
            fig_pnl = go.Figure()

            # Colorer en vert/rouge selon positif/négatif
            colors = ['#26a69a' if x >= 0 else '#ef5350' for x in sweep_df["pnl_euros"]]

            fig_pnl.add_trace(go.Bar(
                x=sweep_df["param"],
                y=sweep_df["pnl_euros"],
                name="PNL (€)",
                marker=dict(color=colors, line=dict(width=1, color='#ffffff')),
            ))
            fig_pnl.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
            fig_pnl.update_layout(
                xaxis_title=param_name,
                yaxis_title="PNL (€)",
                template="plotly_dark",
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11, color='#a8b2d1'),
            )
            st.plotly_chart(fig_pnl, use_container_width=True, key="sweep_pnl")

        with col_g2:
            st.markdown("##### 📉 Max Drawdown")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=sweep_df["param"],
                y=sweep_df["max_dd"],
                mode="lines+markers",
                name="Max DD (%)",
                line=dict(color='#ef5350', width=2),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(239, 83, 80, 0.1)',
            ))
            fig_dd.update_layout(
                xaxis_title=param_name,
                yaxis_title="Max Drawdown (%)",
                template="plotly_dark",
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=11, color='#a8b2d1'),
            )
            st.plotly_chart(fig_dd, use_container_width=True, key="sweep_dd")

        # Troisième ligne: Scatter plot Nombre de trades vs Taille moyenne
        st.markdown("##### 📊 Analyse des Trades: Nombre vs Taille Moyenne")
        fig_scatter = go.Figure()

        # Colorier les points selon le Sharpe Ratio
        fig_scatter.add_trace(go.Scatter(
            x=sweep_df["nb_trades"],
            y=sweep_df["avg_trade_size"],
            mode="markers",
            name="Tests",
            marker=dict(
                size=12,
                color=sweep_df["sharpe"],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe"),
                line=dict(width=1, color='white')
            ),
            text=[f"{param_name}={p}" for p in sweep_df["param"]],
            hovertemplate="<b>%{text}</b><br>Trades: %{x}<br>Taille moy: %{y:.2f}<extra></extra>"
        ))

        # Marquer le meilleur point
        best_nb_trades = sweep_df.loc[best_idx, "nb_trades"]
        best_avg_size = sweep_df.loc[best_idx, "avg_trade_size"]
        fig_scatter.add_trace(go.Scatter(
            x=[best_nb_trades],
            y=[best_avg_size],
            mode="markers",
            name="Optimal",
            marker=dict(size=20, color='#ffd700', symbol='star', line=dict(width=2, color='white'))
        ))

        fig_scatter.update_layout(
            xaxis_title="Nombre de Trades",
            yaxis_title="Taille Moyenne du Trade",
            template="plotly_dark",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11, color='#a8b2d1'),
            hovermode='closest',
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="sweep_scatter")

        # Table de données avec formatage
        st.markdown("---")
        st.markdown("#### 📋 Tableau Détaillé des Résultats")

        # Créer une copie avec formatage pour l'affichage
        display_df = sweep_df.copy()
        display_df = display_df.rename(columns={
            "param": f"{param_name}",
            "sharpe": "Sharpe Ratio",
            "return_pct": "Rendement (%)",
            "pnl_euros": f"PNL (€) - Capital: {capital}€",
            "equity_final": "Capital Final (€)",
            "max_dd": "Max DD (%)",
            "nb_trades": "Nb Trades",
            "avg_trade_size": "Taille Moy Trade",
        })

        # Formater les colonnes
        display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].apply(lambda x: f"{x:.3f}")
        display_df["Rendement (%)"] = display_df["Rendement (%)"].apply(lambda x: f"{x:.2f}%")
        display_df[f"PNL (€) - Capital: {capital}€"] = display_df[f"PNL (€) - Capital: {capital}€"].apply(lambda x: f"{x:.2f} €")
        display_df["Capital Final (€)"] = display_df["Capital Final (€)"].apply(lambda x: f"{x:.2f} €")
        display_df["Max DD (%)"] = display_df["Max DD (%)"].apply(lambda x: f"{x:.2f}%")
        display_df["Taille Moy Trade"] = display_df["Taille Moy Trade"].apply(lambda x: f"{x:.2f}")

        st.dataframe(display_df, use_container_width=True, height=400)

        # Exemple avec 1000€
        st.markdown("---")
        st.markdown(f"#### 💡 Simulation avec 1000€ de capital")
        capital_1k = 1000
        best_return = sweep_df.loc[best_idx, "return_pct"] / 100
        pnl_1k = capital_1k * best_return
        equity_1k = capital_1k * (1 + best_return)

        col_1k1, col_1k2, col_1k3 = st.columns(3)
        with col_1k1:
            st.metric("Capital Initial", f"{capital_1k} €")
        with col_1k2:
            st.metric("PNL", f"{pnl_1k:.2f} €", delta=f"{best_return*100:.2f}%")
        with col_1k3:
            st.metric("Capital Final", f"{equity_1k:.2f} €")

        # Export
        st.markdown("---")
        csv = sweep_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Exporter les résultats (CSV)",
            csv,
            "sweep_results.csv",
            "text/csv",
            use_container_width=True,
        )


def main() -> None:
    """Point d'entrée de la page Backtest & Optimisation."""
    st.title("📊 Backtest & Optimisation")
    st.markdown("*Testez et optimisez vos stratégies de trading*")
    st.markdown("---")

    # Onglets principaux
    tab1, tab2 = st.tabs(["🎯 Backtest Simple", "🔬 Optimisation Sweep"])

    with tab1:
        _render_backtest_tab()

    with tab2:
        _render_optimization_tab()


if __name__ == "__main__":
    main()
