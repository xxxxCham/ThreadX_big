"""Agent chargé de générer et de sauvegarder des stratégies expérimentales."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from threadx.llm.agents.base_agent import BaseAgent
from threadx.strategy.experimental import StrategyMeta, register_generated_strategy

__all__ = ["CodeWriter"]


class CodeWriter(BaseAgent):
    """Génère du code de stratégie et l'enregistre sous forme testable."""

    def __init__(self, model: str = "gpt-oss:8b", debug: bool = False, output_dir: str | Path | None = None) -> None:
        super().__init__(name="CodeWriter", model=model, debug=debug)
        default_dir = Path(__file__).resolve().parents[2] / "strategy" / "experimental"
        self.output_dir = Path(output_dir) if output_dir else default_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write(self, file_path: Path, content: str) -> None:
        tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(file_path)

    def write_strategy(
        self,
        strategy_name: str,
        strategy_code: str,
        class_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyMeta:
        """
        Sauvegarde le code d'une stratégie et l'enregistre dans le registre expérimental.

        Args:
            strategy_name: Nom lisible de la stratégie
            strategy_code: Code Python complet de la stratégie
            class_name: Nom de la classe principale (défaut: strategy_name CamelCase)
            metadata: Informations additionnelles à stocker dans metrics

        Returns:
            StrategyMeta persisté dans le registre
        """

        sanitized_name = strategy_name.replace(" ", "_").lower()
        file_path = self.output_dir / f"{sanitized_name}.py"
        self.logger.info("✍️  Écriture stratégie %s -> %s", strategy_name, file_path)
        self._atomic_write(file_path, strategy_code)

        meta = StrategyMeta(
            id=str(uuid.uuid4()),
            file_path=str(file_path),
            class_name=class_name or strategy_name,
            status="generated",
            metrics=metadata or {},
            created_at=time.time(),
            updated_at=time.time(),
        )
        register_generated_strategy(meta)
        return meta

    def generate_code_from_prompt(self, prompt: str, temperature: float = 0.4, max_tokens: int = 2048) -> str:
        """Utilise le LLM pour produire du code source brut."""

        return self._call_llm(prompt=prompt, temperature=temperature, max_tokens=max_tokens)

    def generate_and_write(
        self,
        strategy_name: str,
        prompt: str,
        class_name: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> StrategyMeta:
        """Chaîne complète: génération LLM puis sauvegarde + enregistrement."""

        code = self.generate_code_from_prompt(prompt, temperature=temperature, max_tokens=max_tokens)
        return self.write_strategy(strategy_name, code, class_name=class_name)
