from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from importlib import util
from pathlib import Path
from typing import Any, Iterable, List

__all__ = [
    "StrategyMeta",
    "register_generated_strategy",
    "update_strategy_status",
    "list_experimental_strategies",
    "load_strategy_class",
]


@dataclass
class StrategyMeta:
    """Métadonnées d'une stratégie expérimentale."""

    id: str
    file_path: str
    class_name: str
    status: str
    metrics: dict[str, Any] | None = None
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyMeta":
        return cls(
            id=data.get("id", ""),
            file_path=data.get("file_path", ""),
            class_name=data.get("class_name", ""),
            status=data.get("status", "unknown"),
            metrics=data.get("metrics"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


def _registry_path() -> Path:
    base_dir = Path(__file__).resolve().parent
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / "registry.json"


def _load_registry() -> dict[str, StrategyMeta]:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.getLogger(__name__).error("Registry file is corrupted: %s", path)
        return {}
    return {
        entry.get("id", str(uuid.uuid4())): StrategyMeta.from_dict(entry)
        for entry in raw
        if isinstance(entry, dict)
    }


def _write_registry(strategies: Iterable[StrategyMeta]) -> None:
    path = _registry_path()
    entries = [asdict(meta) for meta in strategies]
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def register_generated_strategy(meta: StrategyMeta) -> StrategyMeta:
    """Enregistre une stratégie générée par CodeWriter."""

    registry = _load_registry()
    meta_id = meta.id or str(uuid.uuid4())
    now = time.time()
    meta.updated_at = now
    if not meta.created_at:
        meta.created_at = now
    meta.status = meta.status or "generated"
    meta.id = meta_id
    registry[meta_id] = meta
    _write_registry(registry.values())
    return meta


def update_strategy_status(
    strategy_id: str, status: str, metrics: dict[str, Any] | None = None
) -> StrategyMeta:
    """Met à jour le statut/les métriques d'une stratégie."""

    registry = _load_registry()
    if strategy_id not in registry:
        raise KeyError(f"Strategy {strategy_id} not found in registry")
    meta = registry[strategy_id]
    meta.status = status
    meta.updated_at = time.time()
    if metrics is not None:
        meta.metrics = metrics
    registry[strategy_id] = meta
    _write_registry(registry.values())
    return meta


def list_experimental_strategies(status_filter: str | None = None) -> List[StrategyMeta]:
    """Liste les stratégies expérimentales (optionnellement filtrées)."""

    registry = _load_registry()
    metas = list(registry.values())
    if status_filter:
        metas = [m for m in metas if m.status == status_filter]
    metas.sort(key=lambda m: m.updated_at, reverse=True)
    return metas


def load_strategy_class(file_path: str | Path, class_name: str) -> type:
    """Charge dynamiquement une classe de stratégie depuis un fichier."""

    module_path = Path(file_path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {module_path}")

    spec = util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]

    try:
        strategy_class = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Class {class_name} not found in module {module_path}"
        ) from exc

    return strategy_class
