"""
ThreadX LLM Client
==================

Interface unifiée pour interagir avec des modèles LLM locaux via Ollama.

Features:
- Support multi-modèles (DeepSeek-R1, Gemma, Qwen, etc.)
- Timeout configurable et fallback gracieux
- Validation des réponses avec retry automatique
- Mode debug pour logging détaillé

Usage:
    >>> from threadx.llm.client import LLMClient
    >>> client = LLMClient(model="deepseek-r1:8b")
    >>> response = client.complete("Analyse ces résultats...")
    >>> print(response)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

try:
    import ollama

    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False
    logging.warning("ollama package not installed. LLM features will be disabled.")


class LLMClient:
    """
    Client LLM pour Ollama avec gestion d'erreurs robuste.

    Attributes:
        model: Nom du modèle Ollama (e.g., "deepseek-r1:8b", "deepseek-r1:32b")
        endpoint: URL de l'API Ollama (default: "http://localhost:11434")
        timeout: Timeout en secondes pour les requêtes (default: 60)
        max_retries: Nombre de tentatives en cas d'échec (default: 2)
        debug: Active le logging détaillé (default: False)
    """

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        endpoint: str = "http://localhost:11434",
        timeout: float = 60.0,
        max_retries: int = 2,
        debug: bool = False,
    ):
        if not HAS_OLLAMA:
            raise RuntimeError(
                "ollama package not installed. Install with: pip install ollama"
            )

        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.debug = debug
        self.logger = logging.getLogger(__name__)

        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        # Vérifier que le modèle est disponible
        self._verify_model()

    def _verify_model(self) -> None:
        """Vérifie que le modèle Ollama est disponible."""
        try:
            models = ollama.list()
            available_models = [m.model for m in models.models]

            if self.model not in available_models:
                self.logger.warning(
                    f"Model {self.model} not found locally. "
                    f"Available: {available_models}. "
                    f"Ollama will attempt to download it on first use."
                )
        except Exception as e:
            self.logger.error(f"Failed to verify Ollama models: {e}")

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """
        Génère une complétion simple.

        Args:
            prompt: Prompt utilisateur
            system: Message système optionnel (instructions)
            temperature: Température de sampling (0.0 = déterministe, 1.0 = créatif)
            max_tokens: Nombre maximum de tokens générés

        Returns:
            Réponse textuelle du LLM

        Raises:
            RuntimeError: Si la requête échoue après max_retries tentatives
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                start = time.time()

                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                )

                elapsed = time.time() - start
                content = response["message"]["content"]

                if self.debug:
                    self.logger.debug(
                        f"LLM completion successful in {elapsed:.2f}s "
                        f"(model={self.model}, tokens≈{len(content)//4})"
                    )

                return content

            except Exception as e:
                self.logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {e}")
                time.sleep(1)  # Backoff

        return ""  # Unreachable but for type checker

    def complete_structured(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        """
        Génère une complétion avec réponse structurée JSON.

        Args:
            prompt: Prompt utilisateur (doit demander un JSON)
            system: Message système optionnel
            temperature: Température de sampling
            max_tokens: Nombre maximum de tokens

        Returns:
            Dict parsé depuis la réponse JSON

        Raises:
            RuntimeError: Si le parsing JSON échoue après tentatives
        """
        # Modifier le prompt pour forcer JSON
        json_prompt = f"{prompt}\n\nIMPORTANT: Réponds UNIQUEMENT avec du JSON valide, sans texte avant ou après."

        response_text = self.complete(json_prompt, system, temperature, max_tokens)

        # Extraire JSON (souvent entouré de ```json ... ```)
        try:
            # Nettoyer la réponse
            cleaned = response_text.strip()

            # Chercher bloc ```json
            if "```json" in cleaned:
                start = cleaned.find("```json") + 7
                end = cleaned.find("```", start)
                cleaned = cleaned[start:end].strip()
            elif "```" in cleaned:
                start = cleaned.find("```") + 3
                end = cleaned.find("```", start)
                cleaned = cleaned[start:end].strip()

            # Parser JSON
            parsed = json.loads(cleaned)

            if self.debug:
                self.logger.debug(f"Structured JSON parsed successfully: {list(parsed.keys())}")

            return parsed

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM JSON response: {e}\nRaw: {response_text[:500]}")
            raise RuntimeError(f"LLM returned invalid JSON: {e}")

    def interpret_backtest_results(
        self,
        summary: dict[str, Any],
        params: dict[str, Any],
        trades_df: Any = None,
    ) -> dict[str, Any]:
        """
        Interprète des résultats de backtest avec analyse intelligente.

        Args:
            summary: Dict de métriques (sharpe, drawdown, win_rate, etc.)
            params: Dict de paramètres de stratégie testés
            trades_df: DataFrame de trades optionnel (pour analyse approfondie)

        Returns:
            Dict avec clés:
            - interpretation: str (résumé global)
            - strengths: list[str] (forces)
            - weaknesses: list[str] (faiblesses)
            - recommendations: list[str] (actions concrètes)
            - risk_level: str (LOW/MODERATE/HIGH)
            - suitability: str (profil investisseur)
        """
        from threadx.llm.prompts import BACKTEST_INTERPRETATION_PROMPT

        # Formater les métriques pour le prompt
        metrics_str = "\n".join([f"  - {k}: {v}" for k, v in summary.items() if v is not None])
        params_str = "\n".join([f"  - {k}: {v}" for k, v in params.items()])

        # Contexte additionnel sur les trades
        trades_context = ""
        if trades_df is not None and hasattr(trades_df, "__len__") and len(trades_df) > 0:
            trades_context = f"\n  - Nombre de trades: {len(trades_df)}"

        prompt = BACKTEST_INTERPRETATION_PROMPT.format(
            metrics=metrics_str, params=params_str, trades_context=trades_context
        )

        system = (
            "Tu es un analyste quantitatif expert avec 10+ ans d'expérience "
            "en trading algorithmique. Analyse les résultats avec rigueur et pragmatisme."
        )

        try:
            result = self.complete_structured(prompt, system=system, temperature=0.6, max_tokens=1500)

            # Valider les clés attendues
            required_keys = [
                "interpretation",
                "strengths",
                "weaknesses",
                "recommendations",
                "risk_level",
                "suitability",
            ]
            for key in required_keys:
                if key not in result:
                    self.logger.warning(f"Missing key '{key}' in LLM response, using default")
                    result[key] = [] if key in ["strengths", "weaknesses", "recommendations"] else "UNKNOWN"

            return result

        except Exception as e:
            self.logger.error(f"Backtest interpretation failed: {e}", exc_info=True)
            # Fallback gracieux
            return {
                "interpretation": f"Erreur d'interprétation LLM: {e}",
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "risk_level": "UNKNOWN",
                "suitability": "Non analysé (erreur LLM)",
            }
