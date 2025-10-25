"""
ThreadX UI Tests - Backtest Component
======================================

Tests de structure et IDs pour le composant Backtest (P6).

Vérifie:
    - IDs bt-strategy, bt-symbol, bt-timeframe (dropdowns)
    - IDs bt-period, bt-std (inputs params)
    - ID bt-run-btn (bouton trigger)
    - IDs bt-equity-graph, bt-drawdown-graph (graphiques Plotly)
    - IDs bt-trades-table, bt-metrics-table (tables résultats)
    - Présence dcc.Loading autour outputs
    - Grille responsive (dbc.Row/Col)

Author: ThreadX Framework
Version: Prompt 8 - Tests & Qualité
"""

import pytest

pytestmark = pytest.mark.ui


def test_bt_strategy_dropdown_exists(dash_app):
    """Test that backtest strategy dropdown exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "bt-strategy", dcc.Dropdown)


def test_bt_symbol_dropdown_exists(dash_app):
    """Test that backtest symbol dropdown exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "bt-symbol", dcc.Dropdown)


def test_bt_timeframe_dropdown_exists(dash_app):
    """Test that backtest timeframe dropdown exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "bt-timeframe", dcc.Dropdown)


def test_bt_period_input_exists(dash_app):
    """Test that backtest period input exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "bt-period", dcc.Input)


def test_bt_std_input_exists(dash_app):
    """Test that backtest std deviation input exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "bt-std", dcc.Input)


def test_bt_run_button_exists(dash_app):
    """Test that backtest run button exists."""
    import dash_bootstrap_components as dbc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "bt-run-btn", dbc.Button)


def test_bt_equity_graph_exists(dash_app):
    """Test that backtest equity graph exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "bt-equity-graph", dcc.Graph)


def test_bt_drawdown_graph_exists(dash_app):
    """Test that backtest drawdown graph exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "bt-drawdown-graph", dcc.Graph)


def test_bt_trades_table_exists(dash_app):
    """Test that backtest trades table container exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    table = find_component_by_id(layout, "bt-trades-table")
    assert table is not None, "bt-trades-table should exist"


def test_bt_metrics_table_exists(dash_app):
    """Test that backtest metrics table container exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    table = find_component_by_id(layout, "bt-metrics-table")
    assert table is not None, "bt-metrics-table should exist"


def test_bt_status_exists(dash_app):
    """Test that backtest status container exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    status = find_component_by_id(layout, "bt-status")
    assert status is not None, "bt-status should exist"


def test_bt_loading_exists(dash_app):
    """Test that backtest loading spinner exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    loading = find_component_by_id(layout, "bt-loading")
    assert loading is not None, "bt-loading should exist"


def test_bt_panel_controls_exist(dash_app):
    """Test that backtest configuration controls exist in layout."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout

    strategy_dropdown = find_component_by_id(layout, "bt-strategy")
    assert strategy_dropdown is not None, "bt-strategy dropdown should exist"

    run_button = find_component_by_id(layout, "bt-run-btn")
    assert run_button is not None, "bt-run-btn should exist"


def test_bt_panel_has_responsive_grid(dash_app):
    """Test that backtest panel uses Bootstrap responsive grid."""
    import dash_bootstrap_components as dbc

    layout = dash_app.layout

    # Find dbc.Row and dbc.Col
    def find_grid_components(component):
        grid = {"rows": [], "cols": []}
        if isinstance(component, dbc.Row):
            grid["rows"].append(component)
        if isinstance(component, dbc.Col):
            grid["cols"].append(component)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                child_grid = find_grid_components(child)
                grid["rows"].extend(child_grid["rows"])
                grid["cols"].extend(child_grid["cols"])
        return grid

    grid = find_grid_components(layout)
    assert len(grid["rows"]) > 0, "Layout should contain dbc.Row"
    assert len(grid["cols"]) > 0, "Layout should contain dbc.Col"


def test_bt_graphs_have_dark_theme(dash_app):
    """Test that backtest graphs use dark theme template."""
    from dash import dcc

    from tests.conftest import find_component_by_id

    layout = dash_app.layout

    # Check equity graph
    equity_graph = find_component_by_id(layout, "bt-equity-graph")
    assert equity_graph is not None
    assert isinstance(equity_graph, dcc.Graph)

    # Check if figure has dark template (if initialized)
    if hasattr(equity_graph, "figure") and equity_graph.figure:
        fig = equity_graph.figure
        if hasattr(fig, "layout") and hasattr(fig.layout, "template"):
            template = fig.layout.template
            # Template is an object, check its name attribute if exists
            if hasattr(template, "name"):
                template_name = str(template.name).lower()
                assert "dark" in template_name, "Graph should use dark template"
            elif isinstance(template, str):
                assert "dark" in template.lower(), "Graph should use dark template"
