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
from threadx.dataset.validate import validate_dataset

# Configuration
st.set_page_config(
    page_title="ThreadX v2.0 - Trading Quantitatif",
    page_icon="📈",
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

PAGE_TITLES = {"config": "📊 Chargement des Données", "backtest": "🔬 Optimisation"}
PAGE_RENDERERS = {"config": config_page_main, "backtest": backtest_page_main}

@st.cache_resource
def init_session() -> None:
    defaults = {
        "page": "config", "symbol": "BTC", "timeframe": "1h",
        "start_date": date(2024, 9, 1), "end_date": date(2024, 9, 10),
        "strategy": None, "indicators": {}, "strategy_params": {},
        "data": None, "backtest_results": None, "sweep_results": None,
        "data_dir": str(DATA_DIR),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("# ThreadX v2.0")
        st.markdown("*Trading Quantitatif Haute Performance*")
        st.markdown("---")
        st.markdown("### 📍 Navigation")
        labels = list(PAGE_TITLES.values())
        current_key = st.session_state.get("page", "config")
        current_label = PAGE_TITLES.get(current_key, labels[0])
        selected_label = st.radio("Navigation", labels, index=labels.index(current_label), key="nav_radio", label_visibility="collapsed")
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
        if st.button("✅ Valider Datasets", use_container_width=True):
            report = validate_dataset(str(DATA_DIR))
            st.success(f"✅ OK") if report.get("ok") else st.error("❌ Erreur")
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
