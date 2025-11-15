"""
Base Agent Class for ThreadX LLM System
========================================

Classe abstraite fournissant les fonctionnalités communes à tous les agents LLM.

Features:
- Gestion du timeout et retries automatiques
- Logging structuré avec contexte agent
- Validation des réponses LLM
- Métriques de performance (latence, token usage)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from threadx.llm.client import LLMClient


class BaseAgent(ABC):
    """
    Classe abstraite pour agents LLM spécialisés.

    Attributes:
        name: Nom de l'agent (utilisé pour logging)
        model: Modèle Ollama à utiliser
        client: Instance LLMClient configurée
        timeout: Timeout par défaut pour requêtes LLM
        logger: Logger avec préfixe [AgentName]
    """

    def __init__(
        self,
        name: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 2,
        debug: bool = False,
    ):
        """
        Initialise un agent LLM.

        Args:
            name: Nom de l'agent (ex: "Analyst", "Strategist")
            model: Modèle Ollama (ex: "deepseek-r1:70b", "gpt-oss:20b")
            timeout: Timeout en secondes pour requêtes LLM
            max_retries: Nombre de tentatives en cas d'échec
            debug: Active logging détaillé
        """
        self.name = name
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.debug = debug

        # Logger avec préfixe agent
        self.logger = logging.getLogger(f"threadx.llm.agents.{name.lower()}")
        if debug:
            self.logger.setLevel(logging.DEBUG)

        # Client LLM partagé
        self.client = LLMClient(
            model=model, timeout=timeout, max_retries=max_retries, debug=debug
        )

        # Métriques de performance
        self._metrics = {"total_calls": 0, "total_time": 0.0, "errors": 0}

        self.logger.info(
            f"🤖 Agent {name} initialisé (model={model}, timeout={timeout}s)"
        )

    def _call_llm(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Appel LLM avec tracking des métriques.

        Args:
            prompt: Prompt utilisateur
            system: Message système optionnel
            temperature: Température de génération (0.0 = déterministe)
            max_tokens: Nombre max de tokens générés

        Returns:
            Réponse texte du LLM

        Raises:
            RuntimeError: Si LLM échoue après max_retries
        """
        start_time = time.time()
        self._metrics["total_calls"] += 1

        try:
            if self.debug:
                self.logger.debug(f"📤 LLM Call - Prompt preview: {prompt[:200]}...")

            response = self.client.complete(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            elapsed = time.time() - start_time
            self._metrics["total_time"] += elapsed

            if self.debug:
                self.logger.debug(
                    f"📥 LLM Response ({elapsed:.2f}s): {response[:150]}..."
                )

            return response

        except Exception as e:
            self._metrics["errors"] += 1
            self.logger.error(f"❌ LLM call failed: {e}")
            raise

    def _call_llm_structured(
        self,
        prompt: str,
        expected_schema: dict[str, Any] | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """
        Appel LLM avec parsing JSON et validation de schéma.

        Args:
            prompt: Prompt utilisateur (doit demander JSON en output)
            expected_schema: Schéma attendu {key: type} pour validation (optionnel)
            system: Message système optionnel
            temperature: Température de génération
            max_tokens: Nombre max de tokens

        Returns:
            Dict parsé depuis la réponse JSON du LLM

        Raises:
            ValueError: Si réponse non-JSON ou schéma invalide
            RuntimeError: Si LLM échoue après retries
        """
        start_time = time.time()
        self._metrics["total_calls"] += 1

        try:
            if self.debug:
                self.logger.debug(
                    f"📤 LLM Structured Call - Expected schema: {expected_schema}"
                )

            # Note: LLMClient.complete_structured() n'a pas de param expected_schema
            # On appelle directement et valide manuellement si besoin
            response = self.client.complete_structured(
                prompt=prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Validation optionnelle du schéma (basique)
            if expected_schema:
                missing_keys = set(expected_schema.keys()) - set(response.keys())
                if missing_keys:
                    self.logger.warning(
                        "Schema validation: missing keys %s", missing_keys
                    )

            elapsed = time.time() - start_time
            self._metrics["total_time"] += elapsed

            if self.debug:
                self.logger.debug(
                    "Structured Response (%.2fs): %s...",
                    elapsed,
                    json.dumps(response, indent=2)[:300],
                )

            return response

        except Exception as e:
            self._metrics["errors"] += 1
            self.logger.error("Structured LLM call failed: %s", e)
            raise

    def get_metrics(self) -> dict[str, Any]:
        """
        Récupère les métriques de performance de l'agent.

        Returns:
            {
                "total_calls": int,
                "total_time": float,
                "avg_time_per_call": float,
                "errors": int,
                "success_rate": float
            }
        """
        total_calls = self._metrics["total_calls"]
        avg_time = (
            self._metrics["total_time"] / total_calls if total_calls > 0 else 0.0
        )
        success_rate = (
            (total_calls - self._metrics["errors"]) / total_calls
            if total_calls > 0
            else 0.0
        )

        return {
            "agent_name": self.name,
            "model": self.model,
            "total_calls": total_calls,
            "total_time": self._metrics["total_time"],
            "avg_time_per_call": avg_time,
            "errors": self._metrics["errors"],
            "success_rate": success_rate,
        }

    def reset_metrics(self):
        """Reset les métriques de performance."""
        self._metrics = {"total_calls": 0, "total_time": 0.0, "errors": 0}
        self.logger.debug("📊 Métriques reset")

    @abstractmethod
    def analyze(self, *args, **kwargs) -> dict[str, Any]:
        """
        Méthode abstraite pour analyse principale de l'agent.

        Chaque agent spécialisé doit implémenter cette méthode
        avec sa logique spécifique (ex: analyze_sweep_results pour Analyst).
        """
        raise NotImplementedError("Subclasses must implement analyze()")


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model={self.model})"
