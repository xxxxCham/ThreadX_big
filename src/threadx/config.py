"""
Minimal configuration stubs for local UI execution.

These stubs satisfy imports from threadx.__init__ without requiring the full
configuration subsystem. Replace with real implementations when available.
"""

from __future__ import annotations

import os

# Global switch to silence logs by default during current session.
# Set to "0" to re-enable logs.
os.environ.setdefault("THREADX_SILENCE_LOGS", "1")


class Settings:  # noqa: D401 - simple stub
    """Container for application settings (stub)."""

    pass


def get_settings() -> Settings:
    """Return default Settings instance (stub)."""
    return Settings()


def load_settings(*args, **kwargs) -> Settings:
    """Load Settings from a path or dict (stub)."""
    return Settings()


class ConfigurationError(Exception):
    """Raised on invalid configuration (stub)."""

    pass


class PathValidationError(Exception):
    """Raised on invalid path in configuration (stub)."""

    pass




