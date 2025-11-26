import random
from enum import Enum
from typing import Dict, List, Optional


class TaskType(Enum):
    INITIALIZATION = "initialization"
    OPTIMIZATION = "optimization"
    AUDIT = "audit"
    GENERAL = "general"


class ModelRouter:
    """
    Handles the selection of LLM models based on the task type and iteration context.
    Implements the "Architect & Builder" strategy with Guest Auditors.
    """

    DEFAULT_ARCHITECT_MODEL = "deepseek-r1:32b"
    DEFAULT_BUILDER_MODEL = "deepseek-r1:8b"  # Plus rapide pour itérations
    DEFAULT_GUEST_MODELS = [
        # Modèles installés localement (vérifiés 26/11/2025)
        "gemma3:27b",
        "gemma3:12b",
        "deepseek-r1-distill:14b",
        "mistral:7b-instruct",
    ]

    def __init__(
        self,
        audit_interval: int = 5,
        architect_model: str | None = None,
        builder_model: str | None = None,
        guest_models: list[str] | None = None,
    ):
        """
        Args:
            audit_interval: Number of optimization steps between guest audits.
            architect_model: Override for architect.
            builder_model: Override for builder.
            guest_models: Override list for auditors (round-robin).
        """
        self.audit_interval = audit_interval
        self._step_counter = 0
        self._guest_index = 0
        self.architect_model = architect_model or self.DEFAULT_ARCHITECT_MODEL
        self.builder_model = builder_model or self.DEFAULT_BUILDER_MODEL
        self.guest_models = guest_models or list(self.DEFAULT_GUEST_MODELS)

    def get_model_for_task(
        self, task_type: TaskType, step_number: Optional[int] = None
    ) -> str:
        """
        Determines the best model to use for a given task.

        Args:
            task_type: The type of task (INITIALIZATION, OPTIMIZATION, etc.)
            step_number: The current generation/step number (for optimization loops).

        Returns:
            The name of the ollama model to use.
        """
        if task_type == TaskType.INITIALIZATION:
            return self.architect_model

        if task_type == TaskType.OPTIMIZATION:
            # If step_number is provided, check for audit rotation
            if step_number is not None and step_number > 0:
                if step_number % self.audit_interval == 0:
                    return self._get_next_guest_model()

            # Default builder for optimization
            return self.builder_model

        if task_type == TaskType.AUDIT:
            return self._get_random_guest_model()

        # Default fallback
        return self.builder_model

    def _get_next_guest_model(self) -> str:
        """Selects the next guest model in round-robin order."""
        if not self.guest_models:
            return self.builder_model
        selected = self.guest_models[self._guest_index % len(self.guest_models)]
        self._guest_index += 1
        return selected

    def _get_random_guest_model(self) -> str:
        """Selects a random guest model for audit tasks."""
        if not self.guest_models:
            return self.builder_model
        return random.choice(self.guest_models)

    def get_model_info(self, model_name: str) -> Dict[str, str]:
        """Returns metadata about the model role."""
        if model_name == self.architect_model:
            return {"role": "Architect", "desc": "High-reasoning foundation builder"}
        elif model_name == self.builder_model:
            return {"role": "Builder", "desc": "Fast iterative optimizer"}
        elif model_name in self.guest_models:
            return {
                "role": "Guest Auditor",
                "desc": "External perspective for validation",
            }
        else:
            return {"role": "Unknown", "desc": "Generic model"}
