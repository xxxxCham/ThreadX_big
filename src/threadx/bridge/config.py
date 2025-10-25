"""
ThreadX Bridge Config - Shared Configuration Constants
======================================================

Réexporte les configurations du moteur pour utilisation au-delà du Bridge.
Permet aux couches UI/CLI d'accéder aux configurations sans importer directement Engine.

Author: ThreadX Framework
Version: Prompt 2 - Bridge Foundation
"""

from threadx.optimization.engine import DEFAULT_SWEEP_CONFIG

__all__ = [
    "DEFAULT_SWEEP_CONFIG",
]
