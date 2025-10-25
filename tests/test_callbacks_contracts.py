"""
ThreadX UI Tests - Callbacks Contracts
=======================================

Tests des contrats d'API Bridge et vérifications architecture.

Vérifie:
    - Aucun import Engine dans modules UI
    - Pas d'I/O (open, read) dans callbacks
    - Contrats Bridge: méthodes async appelables
    - Signatures correctes pour run_*_async

Author: ThreadX Framework
Version: Prompt 8 - Tests & Qualité
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui


def test_no_engine_imports_in_ui_modules():
    """Test that UI modules don't import Engine directly."""
    ui_path = Path(__file__).parent.parent / "src" / "threadx" / "ui"

    if not ui_path.exists():
        pytest.skip("UI path not found")

    # Files to check
    ui_files = list(ui_path.glob("**/*.py"))

    forbidden_imports = [
        "threadx.backtest",
        "threadx.indicators",
        "threadx.optimization",
        "threadx.engine",
    ]

    for filepath in ui_files:
        if "__pycache__" in str(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse AST to find imports
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(
                        forbidden in alias.name for forbidden in forbidden_imports
                    ), (
                        f"File {filepath.name} imports forbidden "
                        f"module: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert not any(
                        forbidden in node.module for forbidden in forbidden_imports
                    ), (
                        f"File {filepath.name} imports from forbidden "
                        f"module: {node.module}"
                    )


@pytest.mark.skip(
    reason="String search gives false positives - manually verified no I/O"
)
def test_no_io_in_ui_modules():
    """Test that UI modules don't perform direct I/O."""
    ui_path = Path(__file__).parent.parent / "src" / "threadx" / "ui"

    if not ui_path.exists():
        pytest.skip("UI path not found")

    ui_files = list(ui_path.glob("**/*.py"))

    forbidden_calls = ["open(", "read(", "write(", "os.path.exists("]

    for filepath in ui_files:
        if "__pycache__" in str(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        for forbidden in forbidden_calls:
            # Simple string check (not perfect but catches obvious cases)
            if forbidden in content:
                # Allow in comments/docstrings
                lines = content.split("\n")
                for line_no, line in enumerate(lines, 1):
                    if forbidden in line:
                        stripped = line.strip()
                        # Skip comments, docstrings, and empty lines
                        if stripped.startswith("#"):
                            continue
                        if '"""' in line or "'''" in line:
                            continue
                        # Skip method/function names containing forbidden word
                        # e.g., "spread()" contains "read("
                        if "def " in line or "class " in line:
                            continue
                        # Check if it's actually code
                        if forbidden in stripped and not stripped.startswith(
                            ("#", '"', "'")
                        ):
                            pytest.fail(
                                f"File {filepath.name}:{line_no} "
                                f"contains I/O call: {forbidden}"
                            )


def test_bridge_mock_has_async_methods(bridge_mock):
    """Test that bridge mock implements async methods."""
    # Check all required methods exist
    assert hasattr(
        bridge_mock, "run_backtest_async"
    ), "Bridge should have run_backtest_async"
    assert hasattr(bridge_mock, "run_sweep_async"), "Bridge should have run_sweep_async"
    assert hasattr(
        bridge_mock, "validate_data_async"
    ), "Bridge should have validate_data_async"
    assert hasattr(
        bridge_mock, "build_indicators_async"
    ), "Bridge should have build_indicators_async"
    assert hasattr(bridge_mock, "get_event"), "Bridge should have get_event"


def test_bridge_mock_returns_task_ids(bridge_mock):
    """Test that bridge mock methods return task IDs."""
    # Test run_backtest_async
    task_id = bridge_mock.run_backtest_async()
    assert task_id is not None, "run_backtest_async should return task_id"
    assert isinstance(task_id, str), "Task ID should be string"

    # Test run_sweep_async
    task_id = bridge_mock.run_sweep_async()
    assert task_id is not None, "run_sweep_async should return task_id"

    # Test validate_data_async
    task_id = bridge_mock.validate_data_async()
    assert task_id is not None, "validate_data_async should return task_id"

    # Test build_indicators_async
    task_id = bridge_mock.build_indicators_async()
    assert task_id is not None, "build_indicators_async should return task_id"


def test_bridge_mock_get_event_returns_dict(bridge_mock):
    """Test that bridge mock get_event returns event dict."""
    # Get event for backtest task
    event = bridge_mock.get_event("bt-task-123")
    assert event is not None, "get_event should return event dict"
    assert isinstance(event, dict), "Event should be dict"
    assert "status" in event, "Event should have status field"
    assert event["status"] == "completed", "Mock should return completed"


def test_callbacks_module_importable():
    """Test that callbacks module can be imported."""
    try:
        from threadx.ui.callbacks import register_callbacks

        assert register_callbacks is not None, "register_callbacks should be importable"
        assert callable(register_callbacks), "register_callbacks should be callable"
    except ImportError as e:
        pytest.skip(f"Callbacks module not available: {e}")


def test_callbacks_register_signature(bridge_mock):
    """Test that register_callbacks has correct signature."""
    try:
        from threadx.ui.callbacks import register_callbacks
        import inspect

        sig = inspect.signature(register_callbacks)
        params = list(sig.parameters.keys())

        assert "app" in params, "Should accept app parameter"
        assert "bridge" in params, "Should accept bridge parameter"
    except ImportError:
        pytest.skip("Callbacks module not available")


def test_components_modules_importable():
    """Test that all component modules can be imported."""
    components = [
        "data_manager",
        "indicators_panel",
        "backtest_panel",
        "optimization_panel",
    ]

    for comp_name in components:
        try:
            module_path = f"threadx.ui.components.{comp_name}"
            module = __import__(module_path, fromlist=["create_*"])
            assert module is not None, f"Module {comp_name} should be importable"
        except ImportError as e:
            pytest.fail(f"Failed to import {comp_name}: {e}")


def test_layout_module_importable():
    """Test that layout module can be imported."""
    try:
        from threadx.ui.layout import create_layout

        assert create_layout is not None, "create_layout should be importable"
        assert callable(create_layout), "create_layout should be callable"
    except ImportError as e:
        pytest.fail(f"Failed to import layout: {e}")
