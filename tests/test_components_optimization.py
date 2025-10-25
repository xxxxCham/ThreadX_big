"""
ThreadX UI Tests - Optimization Component
==========================================

Tests de structure et IDs pour le composant Optimization (P6).

Vérifie:
    - IDs opt-strategy, opt-symbol, opt-timeframe (dropdowns)
    - IDs opt-period-min/max/step, opt-std-min/max/step (grille params)
    - ID opt-run-btn (bouton trigger)
    - ID opt-results-table (table top combinations)
    - ID opt-heatmap (graphique 2D Plotly)
    - Présence dcc.Loading autour outputs
    - Grille responsive (dbc.Row/Col)

Author: ThreadX Framework
Version: Prompt 8 - Tests & Qualité
"""

import pytest

pytestmark = pytest.mark.ui


def test_opt_strategy_dropdown_exists(dash_app):
    """Test that optimization strategy dropdown exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "opt-strategy", dcc.Dropdown)


def test_opt_symbol_dropdown_exists(dash_app):
    """Test that optimization symbol dropdown exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "opt-symbol", dcc.Dropdown)


def test_opt_timeframe_dropdown_exists(dash_app):
    """Test that optimization timeframe dropdown exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "opt-timeframe", dcc.Dropdown)


def test_opt_period_grid_inputs_exist(dash_app):
    """Test that optimization period grid inputs exist."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout

    # Period min/max/step
    assert_component_exists(layout, "opt-period-min", dcc.Input)
    assert_component_exists(layout, "opt-period-max", dcc.Input)
    assert_component_exists(layout, "opt-period-step", dcc.Input)


def test_opt_std_grid_inputs_exist(dash_app):
    """Test that optimization std grid inputs exist."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout

    # Std min/max/step
    assert_component_exists(layout, "opt-std-min", dcc.Input)
    assert_component_exists(layout, "opt-std-max", dcc.Input)
    assert_component_exists(layout, "opt-std-step", dcc.Input)


def test_opt_run_button_exists(dash_app):
    """Test that optimization run button exists."""
    import dash_bootstrap_components as dbc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "opt-run-btn", dbc.Button)


def test_opt_results_table_exists(dash_app):
    """Test that optimization results table exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    table = find_component_by_id(layout, "opt-results-table")
    assert table is not None, "opt-results-table should exist"


def test_opt_heatmap_exists(dash_app):
    """Test that optimization heatmap graph exists."""
    from dash import dcc

    from tests.conftest import assert_component_exists

    layout = dash_app.layout
    assert_component_exists(layout, "opt-heatmap", dcc.Graph)


def test_opt_status_exists(dash_app):
    """Test that optimization status container exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    status = find_component_by_id(layout, "opt-status")
    assert status is not None, "opt-status should exist"


def test_opt_loading_exists(dash_app):
    """Test that optimization loading spinner exists."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    loading = find_component_by_id(layout, "opt-loading")
    assert loading is not None, "opt-loading should exist"


def test_opt_panel_has_tabs(dash_app):
    """Test that optimization panel exposes result tabs."""
    import dash_bootstrap_components as dbc

    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    tabs = find_component_by_id(layout, "opt-tabs")
    assert tabs is not None, "opt-tabs should exist in layout"
    assert isinstance(tabs, dbc.Tabs), "opt-tabs should be a dbc.Tabs component"


def test_opt_panel_has_responsive_grid(dash_app):
    """Test that optimization panel uses Bootstrap responsive grid."""
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


def test_opt_heatmap_has_dark_theme(dash_app):
    """Test that optimization heatmap uses dark theme."""
    from dash import dcc

    from tests.conftest import find_component_by_id

    layout = dash_app.layout

    # Check heatmap graph
    heatmap = find_component_by_id(layout, "opt-heatmap")
    assert heatmap is not None
    assert isinstance(heatmap, dcc.Graph)

    # Check if figure has dark template (if initialized)
    if hasattr(heatmap, "figure") and heatmap.figure:
        fig = heatmap.figure
        if hasattr(fig, "layout") and hasattr(fig.layout, "template"):
            template = fig.layout.template
            # Template is an object, check its name attribute if exists
            if hasattr(template, "name"):
                template_name = str(template.name).lower()
                assert "dark" in template_name, "Heatmap should use dark template"
            elif isinstance(template, str):
                assert "dark" in template.lower(), "Heatmap should use dark template"
