"""
ThreadX LLM Integration Module
================================

Module d'intégration LLM local pour l'analyse intelligente de backtests,
la recommandation de paramètres et l'assistance interactive.

Composants:
- LLMClient: Interface unifiée pour modèles locaux (Ollama)
- Prompts: Templates de prompts réutilisables
- Interpreters: Parsers pour structurer les réponses LLM

Author: ThreadX Framework
Version: 1.0.0 - Initial LLM Integration
"""

from threadx.llm.client import LLMClient
from threadx.llm.interpreters import parse_backtest_interpretation

__all__ = ["LLMClient", "parse_backtest_interpretation"]
