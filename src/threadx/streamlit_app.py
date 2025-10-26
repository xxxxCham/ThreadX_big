"""
ThreadX - Interface Streamlit unifiee.
Lancer avec: streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from datetime import date
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure package root (src/) is on sys.path so `import threadx.*` works when running
# the script directly from the repository root.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from threadx.data_access import DATA_DIR, load_ohlcv
    from threadx.ui.page_selection_token import main as selection_page_main
    from threadx.ui.page_strategy_indicators import main as strategy_page_main
    from threadx.ui.page_backtest_results import main as backtest_page_main
    from threadx.ui.strategy_registry import list_strategies
    from threadx.ui.backtest_bridge import run_backtest, BacktestResult
    from threadx.dataset.validate import validate_dataset
except ImportError:  # pragma: no cover - script execution fallback
    from pathlib import Path
    import importlib.util

    base_dir = Path(__file__).resolve().parent

    def _load(name: str, relative_path: str):
        module_path = base_dir / relative_path
        # Derive a sensible package and full module name so relative imports work
        rel_parent = Path(relative_path).parent
        if rel_parent and rel_parent.parts:
            package = "threadx." + ".".join(rel_parent.parts)
        else:
            package = "threadx"
        module_name = Path(relative_path).stem
        fullname = f"{package}.{module_name}"
        # Ensure the parent of `threadx` (src folder) is on sys.path so imports like
        # `import threadx.data` can be resolved when running the script directly.
        sys.path.insert(0, str(base_dir.parent))

        spec = importlib.util.spec_from_file_location(fullname, module_path)
        module = importlib.util.module_from_spec(spec)
        # Ensure the module has the correct package so "from .foo import ..." works
        module.__package__ = package
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
        return module

    data_access = _load("threadx_ui_data_access", "ui/data_access.py")
    # Load data validation module early so UI modules using relative imports
    # like `from ..dataset.validate import ...` can resolve successfully.
    try:
        # Ensure package entries for 'threadx' and 'threadx.data' exist so
        # relative imports inside UI modules (eg. `from ..dataset.validate`) can
        # be resolved by the import machinery.
        import types

        if "threadx" not in sys.modules:
            pkg = types.ModuleType("threadx")
            pkg.__path__ = [str(base_dir)]
            sys.modules["threadx"] = pkg

        data_pkg_name = "threadx.data"
        if data_pkg_name not in sys.modules:
            data_pkg = types.ModuleType(data_pkg_name)
            data_pkg.__path__ = [str(base_dir / "data")]
            sys.modules[data_pkg_name] = data_pkg

        validate_module = _load("threadx_data_validate", "data/validate.py")
    except FileNotFoundError:
        # Provide a lightweight fallback so the UI can still run when the
        # validation module is absent (useful in trimmed branches).
        import types

        fallback = types.SimpleNamespace()

        def _fallback_validate(path: str):
            return {
                "ok": True,
                "type": "no-validate-module",
                "note": "validation module not available",
            }

        fallback.validate_dataset = _fallback_validate
        validate_module = fallback
    selection_page = _load("threadx_ui_selection", "ui/page_selection_token.py")
    strategy_page = _load("threadx_ui_strategy", "ui/page_strategy_indicators.py")
    backtest_page = _load("threadx_ui_backtest", "ui/page_backtest_results.py")
    registry_module = _load("threadx_ui_registry", "ui/strategy_registry.py")
    bridge_module = _load("threadx_ui_bridge", "ui/backtest_bridge.py")

    DATA_DIR, load_ohlcv = data_access.DATA_DIR, data_access.load_ohlcv
    selection_page_main = selection_page.main
    strategy_page_main = strategy_page.main
    backtest_page_main = backtest_page.main
    list_strategies = registry_module.list_strategies
    run_backtest, BacktestResult = (
        bridge_module.run_backtest,
        bridge_module.BacktestResult,
    )
    validate_dataset = validate_module.validate_dataset

st.set_page_config(
    page_title="ThreadX - Trading Quantitatif",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main { background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%); }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 1.5rem; font-weight: 600;
        transition: all 0.3s ease; width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px; padding: 1.5rem;
        backdrop-filter: blur(10px);
    }
    h1 { color: #667eea; font-weight: 700; }
    h2, h3 { color: #a8b2d1; }
</style>
""",
    unsafe_allow_html=True,
)

PAGE_TITLES: Dict[str, str] = {
    "selection": "Selection Donnees",
    "strategy": "Strategie & Indicateurs",
    "backtest": "Backtest & Resultats",
    "sweep": "Optimisation Sweep",
}

DATA_ROOT = DATA_DIR.resolve()
PROCESSED_DIR = DATA_ROOT / "processed"


def _available_strategies() -> List[str]:
    strategies = list_strategies()
    if not strategies:
        st.info("Aucune strategie disponible dans le registre.")
    return strategies


@st.cache_resource
def init_session() -> None:
    # IMPORTANT: Invalider le cache LRU des fonctions de découverte de données
    # pour s'assurer que les nouveaux fichiers sont détectés au démarrage
    import threadx.data_access
    threadx.data_access._iter_data_files.cache_clear()
    threadx.data_access.discover_tokens_and_timeframes.cache_clear()

    defaults = {
        "page": "selection",
        "symbol": "BTC",
        "timeframe": "1h",
        "start_date": date(2025, 1, 1),
        "end_date": date(2025, 1, 15),
        "strategy": None,
        "indicators": {},
        "strategy_params": {},
        "data": None,
        "backtest_results": None,
        "sweep_results": None,
        "data_dir": str(DATA_ROOT),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("# ThreadX v2.0")
        st.markdown("*Trading quantitatif haute performance*")
        st.divider()

        st.markdown("### Navigation")
        labels = list(PAGE_TITLES.values())
        current_key = st.session_state.get("page", "selection")
        current_label = PAGE_TITLES.get(current_key, labels[0])
        selected_label = st.radio(
            "Page:",
            labels,
            index=labels.index(current_label),
            key="nav_radio",
        )
        selected_key = next(k for k, v in PAGE_TITLES.items() if v == selected_label)
        if selected_key != current_key:
            st.session_state.page = selected_key
            st.rerun()

        st.divider()
        st.markdown("### Systeme")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Backend", "Numpy optimise")
        with col2:
            st.metric("Configs", "TOML/Pydantic")

        st.divider()
        st.markdown("### Actions")
        if st.button("Rafraichir cache", use_container_width=True):
            # Invalider le cache Streamlit
            st.cache_data.clear()
            # Invalider le cache LRU des données
            import threadx.data_access
            threadx.data_access._iter_data_files.cache_clear()
            threadx.data_access.discover_tokens_and_timeframes.cache_clear()
            st.success("Cache vide (Streamlit + LRU data).")
            st.info("Rafraîchissez la page (F5) pour recharger les tokens.")
        if st.button("Valider datasets", use_container_width=True):
            report = validate_dataset(str(DATA_ROOT))
            if report.get("ok"):
                st.success(f"Donnees conformes ({report.get('type', 'inconnu')}).")
            else:
                st.error("Donnees non conformes. Voir rapport ci-dessous.")
                st.json(report)

        st.divider()
        st.caption("(c) 2025 ThreadX Core Team")


def render_selection() -> None:
    init_session()
    selection_page_main()

    symbol = st.session_state.get("symbol")
    timeframe = st.session_state.get("timeframe")
    start_date = st.session_state.get("start_date")
    end_date = st.session_state.get("end_date")

    if not all([symbol, timeframe, start_date, end_date]):
        return

    if start_date > end_date:
        st.error("La date de debut doit etre anterieure ou egale a la date de fin.")
        return

    st.markdown("---")
    st.subheader("Gestion des donnees")
    st.caption(f"Dossier de travail : `{DATA_ROOT}`")

    if st.button(
        "Charger les donnees selectionnees", type="primary", use_container_width=True
    ):
        with st.spinner("Chargement en cours..."):
            try:
                df = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Impossible de charger les donnees: {exc}")
                return

            if df.empty:
                st.warning("Dataset vide. Verifiez la plage selectionnee.")
                return

            st.session_state.data = df
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            output_path = PROCESSED_DIR / f"{symbol}_{timeframe}.parquet"
            try:
                df.to_parquet(output_path)
                st.caption(f"Copie sauvegardee : `{output_path}`")
            except Exception as exc:
                st.warning(f"Impossible de sauvegarder la copie parquet: {exc}")

            st.success(f"Donnees chargees ({len(df)} lignes).")
            st.dataframe(df.head(), use_container_width=True)

    current_df = st.session_state.get("data")
    if isinstance(current_df, pd.DataFrame) and not current_df.empty:
        st.info(
            f"{len(current_df)} lignes actuellement en memoire pour {symbol}/{timeframe}."
        )


def _normalize_indicator_state() -> None:
    raw = st.session_state.get("indicators")
    if not raw:
        st.session_state.indicators = {}
        return
    if all(isinstance(value, dict) for value in raw.values()):
        return

    normalized: Dict[str, Dict[str, Any]] = {}
    for full_key, value in raw.items():
        if not isinstance(full_key, str) or "." not in full_key:
            continue
        name, key = full_key.split(".", 1)
        normalized.setdefault(name, {})[key] = value
    st.session_state.indicators = normalized


def render_strategy() -> None:
    init_session()
    strategy_page_main()
    _normalize_indicator_state()

    strategy = st.session_state.get("strategy")
    if strategy:
        st.caption(f"Strategie active : {strategy}")


def render_backtest() -> None:
    init_session()
    backtest_page_main()

    result: BacktestResult | None = st.session_state.get("backtest_results")
    if not result:
        return

    st.markdown("---")
    st.subheader("Exports & diagnostics")

    trades = result.trades
    if trades:
        trades_df = pd.DataFrame(trades)
        csv = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Exporter trades (CSV)",
            csv,
            "trades.csv",
            "text/csv",
            use_container_width=True,
        )
        st.caption(f"{len(trades_df)} trades simules.")
    else:
        st.caption("Aucun trade disponible pour cette execution.")

    metadata = getattr(result, "metadata", {})
    if metadata:
        st.json(metadata)


def render_sweep() -> None:
    st.title("Optimisation Sweep")
    init_session()

    data = st.session_state.get("data")
    if not isinstance(data, pd.DataFrame) or data.empty:
        st.warning("Chargez d'abord des donnees.")
        return

    strategies = _available_strategies()
    if not strategies:
        return

    strategy = st.selectbox("Strategie", strategies, index=0)
    param_name = st.text_input("Parametre a optimiser", value="window")
    min_val, max_val, step = st.slider("Plage", 5, 50, (10, 30), step=1)

    if st.button("Lancer Sweep", type="primary"):
        with st.spinner("Optimisation en cours..."):
            grid = np.arange(min_val, max_val + step, step)
            results = []
            for value in grid:
                params = {param_name: value}
                try:
                    result = run_backtest(data, strategy, params)
                except Exception as exc:
                    st.error(f"Sweep interrompu: {exc}")
                    return
                results.append(
                    {
                        "param": value,
                        "sharpe": result.metrics.get("sharpe_ratio", 0),
                        "return": result.metrics.get("total_return", 0),
                    }
                )
            st.session_state.sweep_results = pd.DataFrame(results)

    sweep_df = st.session_state.get("sweep_results")
    if isinstance(sweep_df, pd.DataFrame) and not sweep_df.empty:
        st.subheader("Resultats Sweep")
        st.dataframe(sweep_df, use_container_width=True)
        fig = go.Figure(
            data=go.Scatter(
                x=sweep_df["param"], y=sweep_df["sharpe"], mode="lines+markers"
            )
        )
        fig.update_layout(title="Sharpe par parametre", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)


PAGE_RENDERERS = {
    "selection": render_selection,
    "strategy": render_strategy,
    "backtest": render_backtest,
    "sweep": render_sweep,
}


def main() -> None:
    init_session()
    render_sidebar()

    page_key = st.session_state.get("page", "selection")
    renderer = PAGE_RENDERERS.get(page_key, render_selection)
    renderer()


if __name__ == "__main__":
    main()




