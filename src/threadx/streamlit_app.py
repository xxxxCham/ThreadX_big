"""
ThreadX v2.0 - Interface Streamlit Moderne
===========================================

Application de trading quantitatif avec interface fusionnée et moderne.

Architecture:
- Page 1: Configuration & Stratégie (fusion anciennes pages 1+2)
- Page 2: Backtest & Optimisation (fusion anciennes pages 3+4)

Author: ThreadX Framework
Version: 2.0.0 - UI Redesign
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

# Ensure package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threadx.data_access import DATA_DIR
from threadx.ui.page_config_strategy import main as config_page_main
from threadx.ui.page_backtest_optimization import main as backtest_page_main

# Configuration
st.set_page_config(
    page_title="ThreadX v2.0 - Trading Quantitatif",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styles CSS Modernes
st.markdown(
    """
<style>
    .main { background: linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #0f3460 100%); }
    h1 { color: #4fc3f7 !important; font-weight: 700 !important; font-size: 2.5rem !important; text-shadow: 0 0 20px rgba(79, 195, 247, 0.3); }
    h2 { color: #81c784 !important; font-weight: 600 !important; margin-top: 2rem !important; }
    h3 { color: #a8b2d1 !important; font-weight: 500 !important; }
    .stButton>button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 2rem !important; font-weight: 600 !important; transition: all 0.3s ease !important; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important; }
    .stButton>button:hover { transform: translateY(-3px) !important; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; color: #4fc3f7 !important; }
    [data-testid="stMetricLabel"] { color: #a8b2d1 !important; font-size: 0.9rem !important; }
    [data-testid="stExpander"] { background: rgba(255, 255, 255, 0.03) !important; border: 1px solid rgba(255, 255, 255, 0.08) !important; border-radius: 15px !important; backdrop-filter: blur(10px) !important; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%) !important; border-right: 1px solid rgba(79, 195, 247, 0.1) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: rgba(255, 255, 255, 0.02); padding: 8px; border-radius: 12px; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #a8b2d1; padding: 12px 24px; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; }
    hr { margin: 2rem 0 !important; border-color: rgba(79, 195, 247, 0.2) !important; }
</style>
""",
    unsafe_allow_html=True,
)

PAGE_TITLES = {"config": "📊 Chargement des Données", "backtest": "⚡ Optimisation"}
PAGE_RENDERERS = {"config": config_page_main, "backtest": backtest_page_main}


def init_session() -> None:
    """
    Initialise la session avec les réglages par défaut.
    Force l'application des paramètres BTC préréglés UNIQUEMENT à la première ouverture.
    Les modifications de l'utilisateur sont conservées entre les pages.
    """
    # Vérifier si c'est la première initialisation
    if "session_initialized" not in st.session_state:
        st.session_state.session_initialized = False

    defaults = {
        "page": "config",
        "symbol": "BTCUSDC",  # Bitcoin préréglé - OBLIGATOIRE
        "timeframe": "15m",  # 15 minutes préréglé - OBLIGATOIRE
        "start_date": date(2024, 12, 1),  # 1er décembre 2024 - OBLIGATOIRE
        "end_date": date(2025, 1, 31),  # 31 janvier 2025 - OBLIGATOIRE
        "strategy": "Bollinger_Breakout",  # Stratégie Bollinger+ATR préréglée
        "indicators": {},
        # Paramètres de stratégie préréglés selon le tableau classique
        "strategy_params": {
            "bb_period": 20,  # Milieu de la plage 10→50
            "bb_std": 2.0,  # Milieu de la plage 1.5→3.0
            "entry_z": 1.0,  # Seuil Z-score standard
            "entry_logic": "AND",  # Logique d'entrée standard
            "atr_period": 14,  # Milieu de la plage 7→21 (classique)
            "atr_multiplier": 1.5,  # Milieu de la plage 1.0→3.0
            "trailing_stop": True,  # Activer trailing stop
            "risk_per_trade": 0.02,  # 2% de risque par trade (préréglé)
            "min_pnl_pct": 0.01,  # Filtre minimum 0.01%
            "leverage": 1.0,  # Sans levier
            "max_hold_bars": 72,  # 3 jours en 1h (72 barres de 1h)
            "spacing_bars": 6,  # 6 barres minimum entre trades
            "trend_period": 0,  # Sans filtre tendance EMA
        },
        "data": None,
        "backtest_results": None,
        "sweep_results": None,
        "data_dir": str(DATA_DIR),
    }

    # Initialiser les clés manquantes
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # FORCER les paramètres par défaut UNIQUEMENT lors de la première initialisation
    # Après, les modifications utilisateur sont conservées
    if not st.session_state.session_initialized:
        st.session_state.symbol = "BTCUSDC"
        st.session_state.timeframe = "15m"
        st.session_state.start_date = date(2024, 12, 1)
        st.session_state.end_date = date(2025, 1, 31)

        # FORCER le risque par trade à 2% (0.02) - ne jamais le laisser à 0.01
        if "strategy_params" in st.session_state:
            st.session_state.strategy_params["risk_per_trade"] = 0.02

        # Marquer comme initialisé pour ne plus forcer les valeurs
        st.session_state.session_initialized = True


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("# ThreadX v2.0")
        st.markdown("*Trading Quantitatif Haute Performance*")
        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        labels = list(PAGE_TITLES.values())
        current_key = st.session_state.get("page", "config")
        current_label = PAGE_TITLES.get(current_key, labels[0])
        selected_label = st.radio(
            "Navigation",
            labels,
            index=labels.index(current_label),
            key="nav_radio",
            label_visibility="collapsed",
        )
        selected_key = next(k for k, v in PAGE_TITLES.items() if v == selected_label)
        if selected_key != current_key:
            st.session_state.page = selected_key
            st.rerun()
        st.markdown("---")
        st.markdown("### ⚙️ Système")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Backend", "NumPy")
        with col2:
            st.metric("Config", "TOML")
        st.markdown("---")
        if st.button("🔄 Rafraîchir Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("✅ Cache vidé!")
        st.markdown("---")
        st.caption("**ThreadX v2.0** | © 2025")


def main() -> None:
    init_session()
    render_sidebar()
    page_key = st.session_state.get("page", "config")
    renderer = PAGE_RENDERERS.get(page_key, config_page_main)
    renderer()


if __name__ == "__main__":
    main()
