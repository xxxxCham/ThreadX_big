"""Configuration-related exceptions for ThreadX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConfigurationError(Exception):
    path: Optional[str]
    reason: str
    details: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass validation trivial
        super().__init__(self.reason)

    @property
    def user_message(self) -> str:
        location = f" (file: {self.path})" if self.path else ""
        return f"Configuration error{location}: {self.reason}"

    def __str__(self) -> str:
        message = self.user_message
        if self.details:
            message = f"{message}\n{self.details}"
        return message


class PathValidationError(Exception):
    """Raised when configuration paths do not pass validation."""

    def __init__(self, message: str):
        super().__init__(message)
