from __future__ import annotations

from datetime import date
from typing import List

import streamlit as st

from ..data_access import DATA_DIR, discover_tokens_and_timeframes, load_ohlcv
from ..dataset.validate import validate_dataset

DEFAULT_SYMBOL = "BTC"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_START_DATE = date(2025, 1, 1)
DEFAULT_END_DATE = date(2025, 1, 15)


def _ensure_option(options: List[str], value: str) -> List[str]:
    """Make sure the currently selected value remains visible in the UI."""
    if value and value not in options:
        return [value, *options]
    return options


def _sync_session_state(
    symbol: str,
    timeframe: str,
    start_date: date,
    end_date: date,
) -> None:
    """Persist the current selection in Streamlit's session_state."""
    st.session_state.symbol = symbol
    st.session_state.timeframe = timeframe
    st.session_state.start_date = start_date
    st.session_state.end_date = end_date


def _show_validation_widget(data_dir: str) -> None:
    """Small helper that runs `validate_dataset` and displays the outcome."""
    st.markdown("---")
    col_path, col_button = st.columns([3, 1])
    with col_path:
        st.write(f"Dossier de donnees: `{data_dir}`")
    with col_button:
        if st.button("Valider donnees", use_container_width=True):
            with st.spinner("Validation des datasets en cours..."):
                try:
                    report = validate_dataset(data_dir)
                except Exception as exc:  # pragma: no cover - defensive UI safeguard
                    st.error(f"Validation echouee: {exc}")
                    return
            if report.get("ok"):
                label = "Donnees conformes"
                if report.get("type"):
                    label += f" (type {report['type']})"
                st.success(label)
            elif report.get("errors"):
                st.error("Donnees non conformes - details ci-dessous.")
                st.json(report)
            else:
                st.warning("Validation inconclusive - aucun fichier supporte detecte.")


def main() -> None:
    """Page 1 : selection du token et de la plage temporelle."""
    st.title("Selection - Token & Plage temporelle")
    st.caption(f"Donnees exploitees depuis: `{DATA_DIR}`")

    tokens, timeframes = discover_tokens_and_timeframes()
    if not tokens:
        st.error(
            "Aucun dataset exploitable n'a ete trouve dans ce dossier. "
            "Ajoutez vos fichiers Parquet/CSV avant de poursuivre."
        )
        tokens = [DEFAULT_SYMBOL]
    if not timeframes:
        timeframes = [DEFAULT_TIMEFRAME]

    default_symbol = st.session_state.get("symbol", tokens[0])
    default_timeframe = st.session_state.get("timeframe", timeframes[0])
    default_start = st.session_state.get("start_date", DEFAULT_START_DATE)
    default_end = st.session_state.get("end_date", DEFAULT_END_DATE)

    tokens = _ensure_option(tokens, default_symbol)
    timeframes = _ensure_option(timeframes, default_timeframe)

    symbol_index = tokens.index(default_symbol) if default_symbol in tokens else 0
    timeframe_index = (
        timeframes.index(default_timeframe) if default_timeframe in timeframes else 0
    )

    symbol = st.selectbox("Token", options=tokens, index=symbol_index)
    timeframe = st.selectbox("Timeframe", options=timeframes, index=timeframe_index)

    col_start, col_end = st.columns(2)
    start_date = col_start.date_input("Debut", value=default_start)
    end_date = col_end.date_input("Fin", value=default_end)

    _sync_session_state(symbol, timeframe, start_date, end_date)

    if start_date > end_date:
        st.error("La date de debut doit etre anterieure ou egale a la date de fin.")
    else:
        st.success(f"Selection : {symbol} @ {timeframe} : {start_date} -> {end_date}")

    _show_validation_widget(str(DATA_DIR))

    if st.button("Previsualiser donnees", use_container_width=True):
        with st.spinner("Chargement de l'aperu..."):
            try:
                preview = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return
            except Exception as exc:  # pragma: no cover - defensive UI safeguard
                st.error(f"Previsualisation impossible: {exc}")
                return

            if preview.empty:
                st.warning("Aucune donnee disponible pour cette plage.")
                return

            st.dataframe(preview.head(10), use_container_width=True)
            st.caption(
                f"Lignes affichees: {len(preview)} | "
                f"Periode: {preview.index.min()} -> {preview.index.max()}"
            )


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    main()



