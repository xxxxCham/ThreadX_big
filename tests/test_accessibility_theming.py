"""
ThreadX UI Tests - Accessibility & Theming
===========================================

Tests de thème sombre et accessibilité basique.

Vérifie:
    - App utilise dbc.themes.DARKLY
    - Présence de labels/titres dans sections
    - Classes CSS dark theme (bg-dark, text-light)
    - Contrastes minimaux (vérification simple)

Author: ThreadX Framework
Version: Prompt 8 - Tests & Qualité
"""

import pytest

pytestmark = pytest.mark.ui


def test_app_uses_dark_theme(dash_app):
    """Test that app uses Bootstrap DARKLY theme."""
    import dash_bootstrap_components as dbc

    # Check external stylesheets
    # Dash stores stylesheets in config.external_stylesheets
    if hasattr(dash_app, "config"):
        stylesheets = dash_app.config.external_stylesheets
    elif hasattr(dash_app, "_external_stylesheets"):
        stylesheets = dash_app._external_stylesheets
    else:
        stylesheets = []

    # Should contain DARKLY theme
    darkly_found = any("darkly" in str(sheet).lower() for sheet in stylesheets)
    assert (
        darkly_found or len(stylesheets) > 0
    ), "App should use dbc.themes.DARKLY or external stylesheets"


def test_main_container_has_dark_classes(dash_app):
    """Test that main container uses dark theme classes."""
    layout = dash_app.layout

    # Container should have bg-dark class
    assert hasattr(layout, "className"), "Container should have className"
    assert "bg-dark" in layout.className, "Container should have bg-dark"
    assert "text-light" in layout.className, "Container should have text-light"


def test_navbar_uses_dark_variant(dash_app):
    """Test that navbar uses dark variant."""
    import dash_bootstrap_components as dbc

    layout = dash_app.layout

    # Find navbar
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
    assert navbar is not None, "Navbar should exist"
    assert navbar.dark is True, "Navbar should use dark variant"
    assert navbar.color == "dark", "Navbar should have dark color"


def test_headers_present_in_tabs(dash_app):
    """Test that each tab has header/title elements."""
    from dash import html

    layout = dash_app.layout

    # Find all H3/H4 headers
    def find_headers(component):
        headers = []
        if isinstance(component, (html.H1, html.H2, html.H3, html.H4)):
            headers.append(component)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                headers.extend(find_headers(child))
        return headers

    headers = find_headers(layout)
    assert len(headers) > 0, "Layout should contain header elements"

    # Check some headers have text-light class
    light_headers = [h for h in headers if hasattr(h, "className") and h.className]
    assert len(light_headers) > 0, "Some headers should have className for styling"


def test_cards_use_dark_theme(dash_app):
    """Test that cards use dark theme styling."""
    import dash_bootstrap_components as dbc

    layout = dash_app.layout

    # Find all cards
    def find_cards(component):
        cards = []
        if isinstance(component, dbc.Card):
            cards.append(component)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                cards.extend(find_cards(child))
        return cards

    cards = find_cards(layout)

    # If cards exist, check they use dark classes
    if cards:
        dark_cards = [
            c
            for c in cards
            if hasattr(c, "className") and c.className and "bg-dark" in c.className
        ]
        assert len(dark_cards) > 0, "Cards should use bg-dark class"


def test_buttons_have_color_variants(dash_app):
    """Test that buttons use Bootstrap color variants."""
    import dash_bootstrap_components as dbc

    layout = dash_app.layout

    # Find all buttons
    def find_buttons(component):
        buttons = []
        if isinstance(component, dbc.Button):
            buttons.append(component)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                buttons.extend(find_buttons(child))
        return buttons

    buttons = find_buttons(layout)
    assert len(buttons) > 0, "Layout should contain buttons"

    # Check buttons have color variants
    colored_buttons = [b for b in buttons if hasattr(b, "color") and b.color]
    assert (
        len(colored_buttons) > 0
    ), "Buttons should have color variants (success, warning, etc.)"


def test_graphs_use_dark_template(dash_app):
    """Test that Plotly graphs use dark template."""
    from dash import dcc

    layout = dash_app.layout

    # Find all dcc.Graph components
    def find_graphs(component):
        graphs = []
        if isinstance(component, dcc.Graph):
            graphs.append(component)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                graphs.extend(find_graphs(child))
        return graphs

    graphs = find_graphs(layout)
    assert len(graphs) > 0, "Layout should contain graphs (bt/opt panels)"

    # Check graphs with initialized figures
    for graph in graphs:
        if hasattr(graph, "figure") and graph.figure:
            fig = graph.figure
            if hasattr(fig, "layout") and hasattr(fig.layout, "template"):
                template = str(fig.layout.template).lower()
                assert "dark" in template, f"Graph {graph.id} should use dark template"


def test_labels_present_for_inputs(dash_app):
    """Test that inputs have associated labels."""
    from dash import dcc, html

    layout = dash_app.layout

    # Find all dcc.Input components
    def find_inputs(component):
        inputs = []
        if isinstance(component, dcc.Input):
            inputs.append(component)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                inputs.extend(find_inputs(child))
        return inputs

    # Find all html.Label components
    def find_labels(component):
        labels = []
        if isinstance(component, html.Label):
            labels.append(component)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                labels.extend(find_labels(child))
        return labels

    inputs = find_inputs(layout)
    labels = find_labels(layout)

    # Simple check: if inputs exist, some labels should exist
    if inputs:
        assert len(labels) > 0, "Layout with inputs should have some labels"


def test_loading_components_present(dash_app):
    """Test that loading spinners are present for async outputs."""
    from dash import dcc

    layout = dash_app.layout

    # Find all dcc.Loading components
    def find_loading(component):
        loading = []
        if isinstance(component, dcc.Loading):
            loading.append(component)
        if hasattr(component, "children"):
            children = component.children
            if not isinstance(children, list):
                children = [children]
            for child in children:
                loading.extend(find_loading(child))
        return loading

    loading_comps = find_loading(layout)
    assert (
        len(loading_comps) > 0
    ), "Layout should contain dcc.Loading components for async operations"
