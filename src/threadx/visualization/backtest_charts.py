"""
ThreadX - Graphiques de Backtest
==================================

Génération graphiques interactifs avec Plotly pour visualiser résultats.

Features:
- Bougies japonaises (OHLCV)
- Marqueurs entrées/sorties (▲ vert / ▼ rouge)
- Bandes de Bollinger overlay
- Courbe d'équité
- Indicateurs techniques (sous-graphique)

Usage:
    >>> from threadx.visualization.backtest_charts import generate_backtest_chart
    >>> generate_backtest_chart(
    ...     results_df=best_results,
    ...     ohlcv_data=ohlcv,
    ...     best_combo={'bb_window': 20, 'bb_num_std': 2.0},
    ...     symbol='BTCUSDC',
    ...     timeframe='1h',
    ...     output_path='backtest_BTCUSDC_1h.html'
    ... )
"""

from pathlib import Path

import pandas as pd

from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Imports optionnels
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly non disponible. Installez avec: pip install plotly kaleido")


def generate_backtest_chart(
    results_df: pd.DataFrame,
    ohlcv_data: pd.DataFrame,
    best_combo: dict[str, float],
    symbol: str,
    timeframe: str,
    output_path: str | Path,
    show_browser: bool = False,
) -> Path | None:
    """
    Génère graphique interactif Plotly avec résultats backtest.

    Args:
        results_df: DataFrame avec colonnes:
            - timestamp: Index temporel
            - position: 1 (long), -1 (short), 0 (flat)
            - equity: Valeur du portefeuille
            - entry_price: Prix d'entrée (NaN si pas d'entrée)
            - exit_price: Prix de sortie (NaN si pas de sortie)
        ohlcv_data: DataFrame OHLCV avec colonnes:
            - timestamp: Index
            - open, high, low, close, volume
        best_combo: Paramètres optimaux (ex: {'bb_window': 20, 'bb_num_std': 2.0})
        symbol: Symbole (ex: 'BTCUSDC')
        timeframe: Intervalle temporel (ex: '1h', '4h')
        output_path: Chemin fichier HTML (ex: 'backtest_BTCUSDC_1h.html')
        show_browser: Si True, ouvre dans navigateur

    Returns:
        Path du fichier HTML généré (ou None si erreur)

    Example:
        >>> chart_path = generate_backtest_chart(
        ...     results_df=best_results,
        ...     ohlcv_data=ohlcv,
        ...     best_combo={'bb_window': 20, 'bb_num_std': 2.0},
        ...     symbol='BTCUSDC',
        ...     timeframe='1h',
        ...     output_path='charts/backtest_BTCUSDC_1h.html'
        ... )
        >>> logger.info(f"Graphique généré: {chart_path}")
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly non disponible. Impossible de générer graphique.")
        return None

    try:
        # Conversion path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Merge OHLCV + results
        merged = pd.merge(
            ohlcv_data,
            results_df[["position", "equity", "entry_price", "exit_price"]],
            left_index=True,
            right_index=True,
            how="left",
        )

        # Calcul Bollinger Bands si paramètres fournis
        bb_window = best_combo.get("bb_window", 20)
        bb_std = best_combo.get("bb_num_std", 2.0)

        if "close" in merged.columns:
            merged["bb_middle"] = merged["close"].rolling(window=int(bb_window)).mean()
            std = merged["close"].rolling(window=int(bb_window)).std()
            merged["bb_upper"] = merged["bb_middle"] + (bb_std * std)
            merged["bb_lower"] = merged["bb_middle"] - (bb_std * std)

        # Création figure avec 3 sous-graphiques
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(
                f"{symbol} {timeframe} - Prix & Signaux",
                "Courbe d'Équité",
                "Indicateurs Techniques",
            ),
        )

        # === ROW 1: CANDLESTICKS + BOLLINGER BANDS + SIGNAUX ===

        # Bougies japonaises
        fig.add_trace(
            go.Candlestick(
                x=merged.index,
                open=merged["open"],
                high=merged["high"],
                low=merged["low"],
                close=merged["close"],
                name="OHLC",
                increasing_line_color="#26A69A",
                decreasing_line_color="#EF5350",
            ),
            row=1,
            col=1,
        )

        # Bollinger Bands
        if "bb_upper" in merged.columns:
            # Bande supérieure
            fig.add_trace(
                go.Scatter(
                    x=merged.index,
                    y=merged["bb_upper"],
                    name="BB Sup",
                    line=dict(color="rgba(250, 128, 114, 0.5)", width=1, dash="dash"),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Bande moyenne
            fig.add_trace(
                go.Scatter(
                    x=merged.index,
                    y=merged["bb_middle"],
                    name="BB Mid",
                    line=dict(color="rgba(135, 206, 250, 0.7)", width=1.5),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

            # Bande inférieure
            fig.add_trace(
                go.Scatter(
                    x=merged.index,
                    y=merged["bb_lower"],
                    name="BB Inf",
                    line=dict(color="rgba(250, 128, 114, 0.5)", width=1, dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(135, 206, 250, 0.1)",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Marqueurs d'entrée (▲ vert)
        entries = merged[merged["entry_price"].notna()]
        if not entries.empty:
            fig.add_trace(
                go.Scatter(
                    x=entries.index,
                    y=entries["entry_price"],
                    mode="markers",
                    name="Entrée",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="#4CAF50",
                        line=dict(color="white", width=1.5),
                    ),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Marqueurs de sortie (▼ rouge)
        exits = merged[merged["exit_price"].notna()]
        if not exits.empty:
            fig.add_trace(
                go.Scatter(
                    x=exits.index,
                    y=exits["exit_price"],
                    mode="markers",
                    name="Sortie",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="#F44336",
                        line=dict(color="white", width=1.5),
                    ),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # === ROW 2: COURBE D'ÉQUITÉ ===

        fig.add_trace(
            go.Scatter(
                x=merged.index,
                y=merged["equity"],
                name="Équité",
                line=dict(color="#2196F3", width=2),
                fill="tozeroy",
                fillcolor="rgba(33, 150, 243, 0.1)",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

        # Ligne de base (capital initial = première valeur équité)
        if "equity" in merged.columns and not merged["equity"].isna().all():
            initial_equity = merged["equity"].iloc[0]
            fig.add_hline(
                y=initial_equity,
                line_dash="dot",
                line_color="gray",
                opacity=0.5,
                row=2,
                col=1,
            )

        # === ROW 3: POSITION (LONG/SHORT/FLAT) ===

        # Convertir position en zone colorée
        position_colors = {
            1: "#4CAF50",  # Long = vert
            -1: "#F44336",  # Short = rouge
            0: "#9E9E9E",  # Flat = gris
        }

        merged["position_color"] = (
            merged["position"].map(position_colors).fillna("#9E9E9E")
        )

        fig.add_trace(
            go.Bar(
                x=merged.index,
                y=merged["position"].abs(),
                name="Position",
                marker=dict(color=merged["position_color"]),
                showlegend=True,
            ),
            row=3,
            col=1,
        )

        # === MISE EN FORME ===

        # Titre global
        params_str = ", ".join([f"{k}={v}" for k, v in best_combo.items()])
        title = (
            f"<b>{symbol} {timeframe}</b> - Backtest Optimisé<br>"
            f"<sub>Paramètres: {params_str}</sub>"
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            xaxis_rangeslider_visible=False,
            height=900,
            template="plotly_dark",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Prix", row=1, col=1)
        fig.update_yaxes(title_text="Équité ($)", row=2, col=1)
        fig.update_yaxes(title_text="Position", row=3, col=1)

        # Sauvegarde HTML
        fig.write_html(
            str(output_path), config={"displayModeBar": True, "displaylogo": False}
        )

        logger.info(f"✅ Graphique généré: {output_path}")

        # Ouvrir dans navigateur si demandé
        if show_browser:
            import webbrowser

            webbrowser.open(str(output_path.absolute()))

        return output_path

    except Exception as e:
        logger.error(f"❌ Erreur génération graphique: {e}")
        return None


def generate_multi_timeframe_chart(
    results_dict: dict[str, pd.DataFrame],
    ohlcv_dict: dict[str, pd.DataFrame],
    best_combos: dict[str, dict[str, float]],
    symbol: str,
    output_path: str | Path,
    show_browser: bool = False,
) -> Path | None:
    """
    Génère graphique multi-timeframes (1h + 4h + 1d par exemple).

    Args:
        results_dict: Dict {timeframe: results_df}
        ohlcv_dict: Dict {timeframe: ohlcv_df}
        best_combos: Dict {timeframe: best_params}
        symbol: Symbole (ex: 'BTCUSDC')
        output_path: Chemin fichier HTML
        show_browser: Ouvrir dans navigateur

    Returns:
        Path du fichier HTML

    Example:
        >>> generate_multi_timeframe_chart(
        ...     results_dict={'1h': results_1h, '4h': results_4h},
        ...     ohlcv_dict={'1h': ohlcv_1h, '4h': ohlcv_4h},
        ...     best_combos={'1h': {...}, '4h': {...}},
        ...     symbol='BTCUSDC',
        ...     output_path='multi_tf_BTCUSDC.html'
        ... )
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly non disponible.")
        return None

    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_timeframes = len(results_dict)

        # Création figure avec N sous-graphiques (un par timeframe)
        fig = make_subplots(
            rows=n_timeframes,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.05,
            subplot_titles=[f"{symbol} {tf}" for tf in results_dict.keys()],
        )

        # Pour chaque timeframe
        for i, (tf, results_df) in enumerate(results_dict.items(), start=1):
            ohlcv = ohlcv_dict[tf]
            best_combos[tf]

            # Merge données
            merged = pd.merge(
                ohlcv,
                results_df[["equity", "entry_price", "exit_price"]],
                left_index=True,
                right_index=True,
                how="left",
            )

            # Candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=merged.index,
                    open=merged["open"],
                    high=merged["high"],
                    low=merged["low"],
                    close=merged["close"],
                    name=f"OHLC {tf}",
                    increasing_line_color="#26A69A",
                    decreasing_line_color="#EF5350",
                ),
                row=i,
                col=1,
            )

            # Entrées
            entries = merged[merged["entry_price"].notna()]
            if not entries.empty:
                fig.add_trace(
                    go.Scatter(
                        x=entries.index,
                        y=entries["entry_price"],
                        mode="markers",
                        name=f"Entrée {tf}",
                        marker=dict(symbol="triangle-up", size=10, color="#4CAF50"),
                        showlegend=False,
                    ),
                    row=i,
                    col=1,
                )

            # Sorties
            exits = merged[merged["exit_price"].notna()]
            if not exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exits.index,
                        y=exits["exit_price"],
                        mode="markers",
                        name=f"Sortie {tf}",
                        marker=dict(symbol="triangle-down", size=10, color="#F44336"),
                        showlegend=False,
                    ),
                    row=i,
                    col=1,
                )

        # Mise en forme
        fig.update_layout(
            title=f"<b>{symbol}</b> - Multi-Timeframes Backtest",
            height=400 * n_timeframes,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            showlegend=False,
        )

        # Sauvegarde
        fig.write_html(str(output_path))
        logger.info(f"✅ Graphique multi-TF généré: {output_path}")

        if show_browser:
            import webbrowser

            webbrowser.open(str(output_path.absolute()))

        return output_path

    except Exception as e:
        logger.error(f"❌ Erreur graphique multi-TF: {e}")
        return None


__all__ = [
    "generate_backtest_chart",
    "generate_multi_timeframe_chart",
]
