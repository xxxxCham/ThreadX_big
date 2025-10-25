"""Configuration pytest pour les tests ThreadX."""

import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Ajouter src au path si nécessaire
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def pytest_configure():
    """Configure l'environnement de test."""
    # Évite d'afficher le warning Pandera dans les tests
    os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")


# ═══════════════════════════════════════════════════════════════
# FIXTURES - Bridge Mock
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def bridge_mock():
    """Create mock ThreadXBridge instance for UI tests.

    Stubs all async methods to return task IDs and simulate
    completed events. No real computation or I/O.

    Returns:
        Mock: ThreadXBridge mock with stubbed methods.
    """
    mock = Mock()

    # Stub async submit methods (return task IDs)
    mock.run_backtest_async.return_value = "bt-task-123"
    mock.run_sweep_async.return_value = "opt-task-456"
    mock.validate_data_async.return_value = "data-task-789"
    mock.build_indicators_async.return_value = "ind-task-012"

    # Stub get_event (return completed events)
    def mock_get_event(task_id, timeout=None):
        """Simulate completed event with dummy data."""
        if task_id == "bt-task-123":
            return {
                "status": "completed",
                "result": Mock(
                    equity_curve=[10000, 10500, 11000],
                    drawdown_curve=[0, -2.5, -1.0],
                    trades=[{"entry": "2024-01-01", "pnl": 500}],
                    metrics={"total_return": 10.0, "sharpe": 1.5},
                    total_return=10.0,
                ),
            }
        elif task_id == "opt-task-456":
            return {
                "status": "completed",
                "results": [
                    {"period": 20, "std": 2.0, "return": 12.0},
                    {"period": 25, "std": 2.5, "return": 10.5},
                ],
            }
        elif task_id == "data-task-789":
            return {
                "status": "completed",
                "data": [{"symbol": "BTCUSDT", "timeframe": "1h", "rows": 1000}],
            }
        elif task_id == "ind-task-012":
            return {
                "status": "completed",
                "indicators": {
                    "ema": [100, 101, 102],
                    "rsi": [50, 55, 48],
                },
            }
        return None

    mock.get_event.side_effect = mock_get_event

    # Config attribute
    mock.config.max_workers = 4

    return mock


# ═══════════════════════════════════════════════════════════════
# FIXTURES - Dash App
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def dash_app(bridge_mock):
    """Create Dash app instance with mocked Bridge.

    Args:
        bridge_mock: Mocked ThreadXBridge from bridge_mock fixture.

    Returns:
        dash.Dash: Configured Dash app with layout.
    """
    try:
        import dash
        import dash_bootstrap_components as dbc

        from threadx.ui.layout import create_layout

        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True,
        )
        app.layout = create_layout(bridge_mock)
        return app
    except ImportError as e:
        pytest.skip(f"Dash dependencies not available: {e}")


# ═══════════════════════════════════════════════════════════════
# HELPERS - Component Search
# ═══════════════════════════════════════════════════════════════


def find_component_by_id(layout, component_id):
    """Recursively search for component by ID in Dash layout.

    Args:
        layout: Dash layout tree (Component or dict).
        component_id: String ID to search for.

    Returns:
        Component if found, None otherwise.

    Example:
        >>> comp = find_component_by_id(app.layout, 'bt-run-btn')
        >>> assert comp is not None
        >>> assert isinstance(comp, dbc.Button)
    """
    if layout is None:
        return None

    # Check if current component has matching ID
    if hasattr(layout, "id") and layout.id == component_id:
        return layout

    # Recursively search children
    if hasattr(layout, "children"):
        children = layout.children
        if not isinstance(children, list):
            children = [children]

        for child in children:
            result = find_component_by_id(child, component_id)
            if result is not None:
                return result

    return None


def assert_component_exists(layout, component_id, component_type=None):
    """Assert component exists in layout with optional type check.

    Args:
        layout: Dash layout tree.
        component_id: String ID to search for.
        component_type: Optional type to check (e.g., dbc.Button).

    Raises:
        AssertionError: If component not found or wrong type.

    Example:
        >>> assert_component_exists(app.layout, 'bt-run-btn', dbc.Button)
    """
    comp = find_component_by_id(layout, component_id)
    assert comp is not None, f"Component '{component_id}' not found in layout"

    if component_type is not None:
        assert isinstance(
            comp, component_type
        ), f"Component '{component_id}' is not {component_type.__name__}"

    return comp
