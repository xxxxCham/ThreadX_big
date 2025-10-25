"""
ThreadX CLI Utilities
=====================

Shared helper functions for CLI commands:
- Logging setup
- JSON formatting
- Async polling wrapper
- Error handling

Author: ThreadX Framework
Version: Prompt 9 - CLI Utilities
"""

import json
import logging
import sys
import time
from typing import Any, Callable, Dict, Optional

# Configure logger for CLI module
logger = logging.getLogger("threadx.cli")


def setup_logger(level: int = logging.INFO) -> None:
    """
    Configure logging for CLI with consistent format.

    Args:
        level: Logging level (logging.DEBUG, INFO, WARNING, etc.)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(level)
    logger.debug(f"Logger initialized at level {logging.getLevelName(level)}")


def print_json(data: Dict[str, Any], indent: int = 2) -> None:
    """
    Print data as formatted JSON to stdout.

    Args:
        data: Dictionary to print as JSON.
        indent: Number of spaces for indentation (default: 2).
    """
    try:
        json_str = json.dumps(data, indent=indent, default=str)
        print(json_str)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize JSON: {e}")
        print(json.dumps({"error": "JSON serialization failed", "detail": str(e)}))


def async_runner(
    func: Callable,
    task_id: str,
    timeout: float = 60.0,
    poll_interval: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """
    Poll async function results with timeout.

    Wrapper for Bridge.get_event() that polls until result or timeout.
    Non-blocking polling loop with configurable interval.

    Args:
        func: Callable that takes task_id (e.g., bridge.get_event).
        task_id: Task ID to poll for.
        timeout: Maximum time to wait in seconds (default: 60s).
        poll_interval: Time between polls in seconds (default: 0.5s).

    Returns:
        Event dict if found, None if timeout.

    Example:
        >>> bridge = ThreadXBridge()
        >>> task_id = bridge.run_backtest_async(request)
        >>> result = async_runner(bridge.get_event, task_id)
    """
    logger.debug(
        f"Polling task {task_id} (timeout={timeout}s, interval={poll_interval}s)"
    )

    start_time = time.time()
    attempts = 0

    try:
        while (time.time() - start_time) < timeout:
            attempts += 1

            try:
                event = func(task_id, timeout=poll_interval)

                if event is not None:
                    elapsed = time.time() - start_time
                    logger.debug(
                        f"Task {task_id} completed after {attempts} attempts "
                        f"({elapsed:.2f}s)"
                    )
                    return event

                # No event yet, continue polling
                time.sleep(poll_interval)

            except KeyboardInterrupt:
                # FIX A3: Propagate pour cleanup externe
                logger.warning("Polling interrupted by user (Ctrl+C)")
                raise
            except Exception as e:
                logger.error(f"Error polling task {task_id}: {e}")
                return {"status": "error", "error": str(e)}
    finally:
        # FIX A3: Garantir cleanup même si exception
        logger.debug(f"Polling loop exited for task {task_id}")

    # Timeout reached
    logger.warning(f"Task {task_id} timed out after {timeout}s ({attempts} attempts)")
    return {"status": "timeout", "task_id": task_id, "timeout": timeout}


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string (e.g., "1m 23.4s", "45.2s").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.1f}s"


def print_summary(title: str, data: Dict[str, Any], json_mode: bool = False) -> None:
    """
    Print command result summary (text or JSON).

    Args:
        title: Summary title (e.g., "Backtest Results").
        data: Result data dictionary.
        json_mode: If True, output JSON; else human-readable text.
    """
    if json_mode:
        print_json({"title": title, "data": data})
    else:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print("=" * 60)

        for key, value in data.items():
            # Format key (snake_case → Title Case)
            formatted_key = key.replace("_", " ").title()

            # Format value
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            elif isinstance(value, dict):
                formatted_value = json.dumps(value, indent=2)
            else:
                formatted_value = str(value)

            print(f"  {formatted_key:<25} {formatted_value}")

        print("=" * 60 + "\n")


def handle_bridge_error(error: Exception, json_mode: bool = False) -> None:
    """
    Handle Bridge errors with consistent formatting.

    Args:
        error: Exception raised by Bridge.
        json_mode: If True, output JSON error; else text.
    """
    error_data = {
        "status": "error",
        "type": type(error).__name__,
        "message": str(error),
    }

    if json_mode:
        print_json(error_data)
    else:
        logger.error(f"Bridge error: {error}")
        print(f"\n❌ Error: {error}\n")

    sys.exit(1)
