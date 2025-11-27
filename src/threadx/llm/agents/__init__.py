"""
ThreadX Multi-Agent LLM System
================================

Système multi-agents pour optimisation automatique de stratégies de trading.

Agents disponibles:
- Analyst: Analyse quantitative des résultats de backtests
- Strategist: Génération de propositions créatives de modifications
- Critic: Validation et critique des propositions
- Autopsy: Post-mortem analysis échecs (auto-learning system)

Usage:
    >>> from threadx.llm.agents import Analyst, Strategist, Critic, Autopsy
    >>> analyst = Analyst(model="deepseek-r1:70b")
    >>> strategist = Strategist(model="gpt-oss:20b")
    >>> critic = Critic(enable_autopsy=True)  # Active auto-learning
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
    >>>
    >>> # Post-mortem échec
    >>> autopsy = Autopsy()
    >>> report = autopsy.analyze_failure(strategy_path, critic_report)
"""

from threadx.llm.agents.analyst import Analyst
from threadx.llm.agents.autopsy import Autopsy
from threadx.llm.agents.codewriter import CodeWriter
from threadx.llm.agents.critic import Critic
from threadx.llm.agents.strategist import Strategist

__all__ = ["Analyst", "Strategist", "Critic", "Autopsy", "CodeWriter"]
