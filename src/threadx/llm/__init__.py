"""
ThreadX LLM Integration Module
================================

Module d'intégration LLM local pour l'analyse intelligente de backtests,
la recommandation de paramètres et l'assistance interactive.

Composants:
- LLMClient: Interface unifiée pour modèles locaux (Ollama)
- LLMIterationEngine: Moteur d'itération pour optimisation automatique
- Prompts: Templates de prompts réutilisables
- Interpreters: Parsers pour structurer les réponses LLM

Author: ThreadX Framework
Version: 2.0.0 - Ajout LLMIterationEngine
"""

from threadx.llm.client import LLMClient
from threadx.llm.interpreters import parse_backtest_interpretation
from threadx.llm.iteration_engine import (
    EngineResult,
    IterationConfig,
    IterationResult,
    LLMIterationEngine,
)

__all__ = [
    "LLMClient",
    "parse_backtest_interpretation",
    "LLMIterationEngine",
    "IterationConfig",
    "IterationResult",
    "EngineResult",
]
