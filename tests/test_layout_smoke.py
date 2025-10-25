"""
ThreadX UI Tests - Layout Smoke Tests
======================================

Tests de fumée pour le layout pipeline:
    - App démarre sans erreur
    - Container principal Bootstrap
    - Présence des colonnes Entrées / Préparation / Analyse
    - Stores & interval nécessaires au polling
    - Navbar & footer présents
"""

import pytest

pytestmark = pytest.mark.ui


def test_app_layout_exists(dash_app):
    """Le layout doit être instancié."""
    assert dash_app is not None, "Dash app should be created"
    assert dash_app.layout is not None, "App layout should not be None"


def test_main_container_exists(dash_app):
    """Le container racine doit être un dbc.Container fluide en thème dark."""
    import dash_bootstrap_components as dbc

    layout = dash_app.layout
    assert isinstance(layout, dbc.Container), "Root layout should be dbc.Container"
    assert layout.fluid is True, "Container should be fluid"
    assert "bg-dark" in (layout.className or ""), "Container should have bg-dark class"


def test_pipeline_columns_present(dash_app):
    """Les trois colonnes du pipeline doivent être présentes."""
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    for column_id in (
        "configuration-column",
        "preparation-column",
        "analysis-column",
    ):
        component = find_component_by_id(layout, column_id)
        assert component is not None, f"Column '{column_id}' should exist"


def test_stores_present(dash_app):
    """Les stores de tâches doivent être présents pour le polling async."""
    from dash import dcc
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    for store_id in (
        "data-task-store",
        "indicators-task-store",
        "bt-task-store",
        "opt-task-store",
    ):
        store = find_component_by_id(layout, store_id)
        assert store is not None, f"Store '{store_id}' should exist in layout"
        assert isinstance(store, dcc.Store), f"'{store_id}' should be dcc.Store"


def test_interval_present(dash_app):
    """L'intervalle global doit exister pour le polling."""
    from dash import dcc
    from tests.conftest import find_component_by_id

    layout = dash_app.layout
    interval = find_component_by_id(layout, "global-interval")
    assert interval is not None, "global-interval should exist"
    assert isinstance(interval, dcc.Interval), "Should be dcc.Interval"
    assert interval.interval == 500, "Interval should be 500ms"


def test_header_present(dash_app):
    """La navbar doit exister et être en thème sombre."""
    import dash_bootstrap_components as dbc

    layout = dash_app.layout

    def find_navbar(component):
        if isinstance(component, dbc.Navbar):
            return component
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                result = find_navbar(child)
                if result:
                    return result
        return None

    navbar = find_navbar(layout)
    assert navbar is not None, "Navbar should exist in header"
    assert navbar.color == "dark", "Navbar should have dark theme"
    assert navbar.dark is True, "Navbar should be dark variant"


def test_footer_present(dash_app):
    """Le footer doit être présent dans le layout."""
    from dash import html

    layout = dash_app.layout

    def find_footer(component):
        if isinstance(component, html.Footer):
            return component
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                result = find_footer(child)
                if result:
                    return result
        return None

    footer = find_footer(layout)
    assert footer is not None, "Footer should exist"
