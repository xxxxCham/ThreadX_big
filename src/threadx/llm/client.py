"""
ThreadX LLM Client
==================

Interface unifiée pour interagir avec des modèles LLM locaux via Ollama.

Features:
- Support multi-modèles (DeepSeek-R1, Gemma, Qwen, etc.)
- Timeout configurable et fallback gracieux
- Validation des réponses avec retry automatique
- Mode debug pour logging détaillé

Notes:
- UnicodeDecodeError dans thread Ollama: Erreur connue non-bloquante
  (thread interne de subprocess, n'affecte pas les résultats)

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


class LLMNotConfiguredError(Exception):
    """Exception levée quand un agent sans LLM tente d'appeler le LLM."""

    pass


class LLMClient:
    """
    Client LLM pour Ollama avec gestion d'erreurs robuste.

    Attributes:
        model: Nom du modèle Ollama (e.g., "deepseek-r1:8b", "deepseek-r1:32b")
        endpoint: URL de l'API Ollama (default: "http://localhost:11434")
        timeout: Timeout en secondes pour les requêtes (default: 180 = 3 min)
        max_retries: Nombre de tentatives en cas d'échec (default: 2)
        debug: Active le logging détaillé (default: False)
    """

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        endpoint: str = "http://localhost:11434",
        timeout: float = 180.0,
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
                error_msg = str(e).lower()

                # Détection erreur CUDA - arrêt immédiat sans retry
                if (
                    "cuda error" in error_msg
                    or "llama runner process has terminated" in error_msg
                ):
                    self.logger.error(
                        f"Erreur CUDA détectée avec Ollama. Le GPU n'est pas disponible ou surchargé. "
                        f"Redémarrez Ollama ou utilisez un modèle CPU: {e}"
                    )
                    raise RuntimeError(
                        f"Erreur GPU Ollama (CUDA): {e}\n"
                        f"Solutions:\n"
                        f"1. Redémarrez Ollama: 'ollama serve'\n"
                        f"2. Vérifiez disponibilité GPU\n"
                        f"3. Utilisez un modèle plus petit\n"
                        f"4. Fermez autres applications GPU"
                    )

                self.logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"LLM request failed after {self.max_retries} attempts: {e}"
                    )
                time.sleep(2)  # Backoff augmenté pour stabilité

        return ""  # Unreachable but for type checker

    def complete_streaming(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        """
        Génère une complétion en streaming (génère des chunks au fur et à mesure).

        Args:
            prompt: Prompt utilisateur
            system: Message système optionnel
            temperature: Température de sampling
            max_tokens: Nombre maximum de tokens

        Yields:
            str: Chunks de texte générés progressivement

        Raises:
            RuntimeError: Si la requête échoue
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
                stream=True,
            )

            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except Exception as e:
            error_msg = str(e).lower()

            if (
                "cuda error" in error_msg
                or "llama runner process has terminated" in error_msg
            ):
                self.logger.error(f"Erreur CUDA détectée: {e}")
                raise RuntimeError(f"Erreur GPU Ollama (CUDA): {e}")

            raise RuntimeError(f"LLM streaming failed: {e}")

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

        # Stocker pour retry si besoin
        self._last_raw_response = response_text

        # Parsing JSON ultra-tolérant avec multiples stratégies
        parsed = self._parse_json_tolerant(response_text)

        if self.debug:
            self.logger.debug(
                f"Structured JSON parsed successfully: {list(parsed.keys())}"
            )

        return parsed

    def _parse_json_tolerant(self, response_text: str) -> dict[str, Any]:
        """
        Parsing JSON ultra-tolérant avec fallbacks multiples.

        Stratégies:
        1. Extraction blocs markdown ```json...```
        2. Extraction objets JSON nus {...}
        3. Recherche première ligne commençant par {
        4. Correction échappements invalides

        Raises:
            RuntimeError: Si toutes stratégies échouent
        """
        import re

        json_candidates = []

        # STRATÉGIE 1: Chercher blocs markdown ```json ... ```
        for match in re.finditer(
            r"```(?:json)?\s*\n(.*?)\n```", response_text, re.DOTALL
        ):
            json_candidates.append(("markdown_block", match.group(1)))

        # STRATÉGIE 2: Chercher objets JSON nus {...} (avec nested support)
        for match in re.finditer(
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response_text, re.DOTALL
        ):
            candidate = match.group(0)
            # Filtrer faux positifs (trop courts)
            if len(candidate) > 20:
                json_candidates.append(("naked_object", candidate))

        # STRATÉGIE 3: Chercher première ligne commençant par {
        lines = response_text.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_text = "\n".join(lines[i:])
                json_candidates.append(("first_brace_line", json_text))
                break

        # Tenter parsing sur chaque candidat
        for strategy, candidate in json_candidates:
            try:
                # Nettoyer et corriger échappements
                cleaned = self._fix_json_escapes(candidate.strip())

                # Parser avec tolérance
                parsed = json.loads(cleaned, strict=False)

                if isinstance(parsed, dict) and len(parsed) > 0:
                    if self.debug:
                        self.logger.debug(
                            f"✅ JSON parsed via strategy '{strategy}' "
                            f"(keys: {list(parsed.keys())})"
                        )
                    return parsed

            except json.JSONDecodeError as e:
                if self.debug:
                    self.logger.debug(
                        f"Strategy '{strategy}' failed: {e} "
                        f"(pos {e.lineno}:{e.colno})"
                    )
                continue

        # ÉCHEC FINAL: Log détaillé et erreur explicite
        self.logger.error(
            f"❌ JSON parsing failed after all strategies.\n"
            f"Response (first 1000 chars):\n{response_text[:1000]}\n"
            f"Candidates tried: {len(json_candidates)} "
            f"({[s for s, _ in json_candidates]})"
        )
        raise RuntimeError(
            f"LLM returned invalid JSON after exhaustive parsing. "
            f"Response length: {len(response_text)} chars. "
            f"Tried {len(json_candidates)} extraction strategies."
        )

    def _fix_json_escapes(self, text: str) -> str:
        """
        Corrige les séquences d'échappement invalides dans le JSON.

        Échappements valides en JSON: quote, backslash, slash, b, f, n, r, t, uXXXX
        Tout autre backslash-X est invalide et sera converti en X.
        """
        import re

        def replace_escape(match):
            escaped_char = match.group(1)
            # Garder les échappements valides
            if escaped_char in ['"', "\\", "/", "b", "f", "n", "r", "t", "u"]:
                return match.group(0)  # Garder tel quel
            else:
                # Remplacer \X par X (enlever le backslash)
                return escaped_char

        # Remplacer les échappements invalides
        fixed = re.sub(r"\\(.)", replace_escape, text)
        return fixed

    def complete_structured_with_retry(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_json_retries: int = 2,
    ) -> dict[str, Any]:
        """
        Appel LLM structuré avec retry intelligent si JSON invalide.

        Si parsing échoue:
        1. Extraire fragment JSON cassé
        2. Redemander au LLM de corriger avec feedback
        3. Parser à nouveau

        Args:
            prompt: Prompt utilisateur
            system: Message système optionnel
            temperature: Température de sampling
            max_tokens: Nombre max de tokens
            max_json_retries: Nombre de tentatives de réparation JSON

        Returns:
            Dict parsé depuis la réponse JSON

        Raises:
            RuntimeError: Si échec après toutes tentatives
        """
        current_prompt = prompt

        for attempt in range(max_json_retries):
            try:
                return self.complete_structured(
                    current_prompt, system, temperature, max_tokens
                )

            except RuntimeError as e:
                if "invalid JSON" in str(e) and attempt < max_json_retries - 1:
                    # Extraire réponse brute
                    last_response = getattr(self, "_last_raw_response", "N/A")

                    # Prompt de réparation
                    repair_prompt = f"""Ta dernière réponse contenait du JSON invalide.

Réponse brute (premiers 500 chars):
{last_response[:500]}

Erreur de parsing: {str(e)}

Réponds UNIQUEMENT avec du JSON valide, sans texte avant ou après.
Format attendu: {{"key": "value", ...}}

JSON corrigé:
"""
                    self.logger.warning(
                        f"⚠️ JSON parse failed (attempt {attempt+1}/{max_json_retries}), "
                        f"asking LLM to repair..."
                    )

                    # Redemander avec prompt réparation
                    current_prompt = repair_prompt
                    continue

                # Échec final ou erreur non-JSON
                raise

        raise RuntimeError(
            f"JSON parsing failed after {max_json_retries} repair attempts"
        )

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
        metrics_str = "\n".join(
            [f"  - {k}: {v}" for k, v in summary.items() if v is not None]
        )
        params_str = "\n".join([f"  - {k}: {v}" for k, v in params.items()])

        # Contexte additionnel sur les trades
        trades_context = ""
        if (
            trades_df is not None
            and hasattr(trades_df, "__len__")
            and len(trades_df) > 0
        ):
            trades_context = f"\n  - Nombre de trades: {len(trades_df)}"

        prompt = BACKTEST_INTERPRETATION_PROMPT.format(
            metrics=metrics_str, params=params_str, trades_context=trades_context
        )

        system = (
            "Tu es un analyste quantitatif expert avec 10+ ans d'expérience "
            "en trading algorithmique. Analyse les résultats avec rigueur et pragmatisme."
        )

        try:
            result = self.complete_structured(
                prompt, system=system, temperature=0.6, max_tokens=1500
            )

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
                    self.logger.warning(
                        f"Missing key '{key}' in LLM response, using default"
                    )
                    result[key] = (
                        []
                        if key in ["strengths", "weaknesses", "recommendations"]
                        else "UNKNOWN"
                    )

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
