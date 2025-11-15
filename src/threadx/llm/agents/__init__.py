"""
ThreadX Multi-Agent LLM System
================================

Système multi-agents pour optimisation automatique de stratégies de trading.

Agents disponibles:
- Analyst: Analyse quantitative des résultats de backtests
- Strategist: Génération de propositions créatives de modifications
- Critic: Validation et critique des propositions (future)

Usage:
    >>> from threadx.llm.agents import Analyst, Strategist
    >>> analyst = Analyst(model="deepseek-r1:70b")
    >>> strategist = Strategist(model="gpt-oss:20b")
    >>>
    >>> # Analyse de résultats Sweep
    >>> analysis = analyst.analyze_sweep_results(sweep_df, top_n=5)
    >>>
    >>> # Propositions de modifications
    >>> proposals = strategist.propose_modifications(
    ...     analysis=analysis,
    ...     current_params=baseline_params,
    ...     n_proposals=3
    ... )
"""

from threadx.llm.agents.analyst import Analyst
from threadx.llm.agents.strategist import Strategist

__all__ = ["Analyst", "Strategist"]
