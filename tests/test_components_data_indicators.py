"""
ThreadX UI Tests - Data Manager & Indicators Components
========================================================

Tests de structure et IDs pour les composants Data Manager
et Indicators (P5).

Vérifie:
    - Data: IDs data-upload, data-registry-table, validate-data-btn
    - Indicators: IDs indicators-symbol, build-indicators-btn, etc.
    - Présence dcc.Loading autour outputs
    - Grilles responsive (dbc.Row/Col)

Author: ThreadX Framework
Version: Prompt 8 - Tests & Qualité
"""

import pytest

pytestmark = pytest.mark.ui


# ═══════════════════════════════════════════════════════════════
# DATA MANAGER TESTS
# ═══════════════════════════════════════════════════════════════


def test_data_upload_exists(dash_app):
    """Test that data upload component exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "data-upload", dcc.Upload)


def test_data_validate_button_exists(dash_app):
    """Test that validate data button exists."""
    import dash_bootstrap_components as dbc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "validate-data-btn", dbc.Button)


def test_data_registry_table_exists(dash_app):
    """Test that data registry table/container exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout

    # Table might be html.Div or dash_table.DataTable
    table = find_component_by_id(layout, "data-registry-table")
    assert table is not None, "data-registry-table should exist"


def test_data_alert_exists(dash_app):
    """Test that data alert container exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    alert = find_component_by_id(layout, "data-alert")
    assert alert is not None, "data-alert should exist"


def test_data_loading_exists(dash_app):
    """Test that data loading spinner exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    loading = find_component_by_id(layout, "data-loading")
    assert loading is not None, "data-loading should exist"


def test_data_dropdowns_exist(dash_app):
    """Test that data dropdowns (source, symbol, timeframe) exist."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout

    # Source dropdown
    source = find_component_by_id(layout, "data-source")
    assert source is not None, "data-source dropdown should exist"

    # Symbol dropdown
    symbol = find_component_by_id(layout, "data-symbol")
    assert symbol is not None, "data-symbol dropdown should exist"

    # Timeframe dropdown
    timeframe = find_component_by_id(layout, "data-timeframe")
    assert timeframe is not None, "data-timeframe dropdown should exist"


# ═══════════════════════════════════════════════════════════════
# INDICATORS TESTS
# ═══════════════════════════════════════════════════════════════


def test_indicators_symbol_dropdown_exists(dash_app):
    """Test that indicators symbol dropdown exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "indicators-symbol", dcc.Dropdown)


def test_indicators_timeframe_dropdown_exists(dash_app):
    """Test that indicators timeframe dropdown exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "indicators-timeframe", dcc.Dropdown)


def test_indicators_build_button_exists(dash_app):
    """Test that build indicators button exists."""
    import dash_bootstrap_components as dbc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "build-indicators-btn", dbc.Button)


def test_indicators_params_inputs_exist(dash_app):
    """Test that indicator parameter inputs exist."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout

    # EMA period
    ema_period = find_component_by_id(layout, "ema-period")
    assert ema_period is not None, "ema-period input should exist"

    # RSI period
    rsi_period = find_component_by_id(layout, "rsi-period")
    assert rsi_period is not None, "rsi-period input should exist"

    # Bollinger std
    bb_std = find_component_by_id(layout, "bollinger-std")
    assert bb_std is not None, "bollinger-std input should exist"


def test_indicators_cache_table_exists(dash_app):
    """Test that indicators cache table exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    table = find_component_by_id(layout, "indicators-cache-table")
    assert table is not None, "indicators-cache-table should exist"


def test_indicators_alert_exists(dash_app):
    """Test that indicators alert container exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    alert = find_component_by_id(layout, "indicators-alert")
    assert alert is not None, "indicators-alert should exist"


def test_indicators_loading_exists(dash_app):
    """Test that indicators loading spinner exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    loading = find_component_by_id(layout, "indicators-loading")
    assert loading is not None, "indicators-loading should exist"


# ═══════════════════════════════════════════════════════════════
# GRID LAYOUT TESTS
# ═══════════════════════════════════════════════════════════════


def test_data_panel_has_responsive_grid(dash_app):
    """Test that data panel uses Bootstrap responsive grid."""
    import dash_bootstrap_components as dbc

    layout = dash_app.layout

    # Find any dbc.Row in layout
    def find_rows(component):
        rows = []
        if isinstance(component, dbc.Row):
            rows.append(component)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                rows.extend(find_rows(child))
        return rows

    rows = find_rows(layout)
    assert len(rows) > 0, "Layout should contain dbc.Row components"


def test_indicators_panel_has_responsive_grid(dash_app):
    """Test that indicators panel uses Bootstrap responsive grid."""
    import dash_bootstrap_components as dbc

    layout = dash_app.layout

    # Find dbc.Col components
    def find_cols(component):
        cols = []
        if isinstance(component, dbc.Col):
            cols.append(component)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                cols.extend(find_cols(child))
        return cols

    cols = find_cols(layout)
    assert len(cols) > 0, "Layout should contain dbc.Col components"
