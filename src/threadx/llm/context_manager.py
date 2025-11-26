"""
Context Manager - Gestion Intelligente Contexte pour Agents LLM
==============================================================

Fournit aux agents LLM :
1. Inventaire complet historiques disponibles (tokens, dates, qualité)
2. Catalogue stratégies existantes (params, performances, versions)
3. Détection automatique données invalides/manquantes
4. Gestion évolution tokens au cours du temps
5. Système CRUD stratégies avec versioning

Architecture:
    ContextManager → DataInventory + StrategyRegistry + ValidationEngine
    → Agents reçoivent contexte structuré JSON
    → Peuvent requêter données spécifiques
    → Peuvent créer/modifier stratégies persistantes

Author: ThreadX Framework
Version: 1.0 - Intelligent Context Management
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from threadx.utils.log import get_logger

logger = get_logger(__name__)


# =============================================================================
# DATA INVENTORY - Inventaire Historiques Disponibles
# =============================================================================


@dataclass
class TokenAvailability:
    """Disponibilité token sur période."""

    symbol: str
    """Symbole (ex: BTCUSDC)."""

    start_date: datetime
    """Date début disponibilité."""

    end_date: datetime
    """Date fin disponibilité (None = actif)."""

    timeframes: list[str]
    """Timeframes disponibles (ex: ['1m', '15m', '1h'])."""

    data_quality: float
    """Score qualité 0-1 (1=parfait, 0=gaps massifs)."""

    total_bars: int
    """Nombre total barres disponibles."""

    gaps_detected: int
    """Nombre gaps détectés."""

    last_update: datetime
    """Dernière mise à jour données."""

    source: str = "binance"
    """Source données (binance, bybit, etc)."""


@dataclass
class DataInventory:
    """Inventaire complet historiques disponibles."""

    tokens: dict[str, TokenAvailability] = field(default_factory=dict)
    """Disponibilité par token."""

    global_start_date: datetime = datetime(2024, 1, 1)
    """Date début globale historiques (1 jan 2024)."""

    global_end_date: datetime = field(default_factory=datetime.now)
    """Date fin globale (aujourd'hui)."""

    total_tokens: int = 0
    """Nombre total tokens disponibles."""

    def add_token(self, token: TokenAvailability):
        """Ajoute token à inventaire."""
        self.tokens[token.symbol] = token
        self.total_tokens = len(self.tokens)

    def get_available_tokens(
        self, start_date: datetime, end_date: datetime, timeframe: str
    ) -> list[str]:
        """
        Retourne tokens disponibles sur période + timeframe.

        Args:
            start_date: Date début recherchée
            end_date: Date fin recherchée
            timeframe: Timeframe requis

        Returns:
            Liste symboles disponibles
        """
        available = []

        for symbol, token in self.tokens.items():
            # Vérifier timeframe disponible
            if timeframe not in token.timeframes:
                continue

            # Vérifier couverture période
            if token.start_date <= start_date and (
                token.end_date is None or token.end_date >= end_date
            ):
                available.append(symbol)

        return available

    def validate_token_period(
        self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str
    ) -> tuple[bool, str]:
        """
        Valide disponibilité token sur période.

        Args:
            symbol: Symbole à valider
            start_date: Date début
            end_date: Date fin
            timeframe: Timeframe

        Returns:
            (valide: bool, raison: str)
        """
        if symbol not in self.tokens:
            return False, f"Token {symbol} non disponible dans inventaire"

        token = self.tokens[symbol]

        # Vérifier timeframe
        if timeframe not in token.timeframes:
            return (
                False,
                f"Timeframe {timeframe} non disponible pour {symbol}. "
                f"Disponibles: {', '.join(token.timeframes)}",
            )

        # Vérifier début période
        if token.start_date > start_date:
            return (
                False,
                f"{symbol} seulement disponible depuis {token.start_date.date()}. "
                f"Début demandé: {start_date.date()}",
            )

        # Vérifier fin période
        if token.end_date and token.end_date < end_date:
            return (
                False,
                f"{symbol} disponible jusqu'à {token.end_date.date()}. "
                f"Fin demandée: {end_date.date()}",
            )

        # Vérifier qualité données
        if token.data_quality < 0.8:
            return (
                False,
                f"{symbol} qualité données insuffisante: {token.data_quality:.1%}. "
                f"Gaps détectés: {token.gaps_detected}",
            )

        return True, "OK"

    def to_llm_context(self) -> dict[str, Any]:
        """
        Convertit inventaire en contexte LLM.

        Returns:
            Dict structuré pour agents
        """
        return {
            "global_period": {
                "start": self.global_start_date.isoformat(),
                "end": self.global_end_date.isoformat(),
                "description": "Historiques disponibles du 1 jan 2024 à aujourd'hui",
            },
            "total_tokens": self.total_tokens,
            "tokens": {
                symbol: {
                    "available_since": token.start_date.date().isoformat(),
                    "available_until": (
                        token.end_date.date().isoformat()
                        if token.end_date
                        else "today"
                    ),
                    "timeframes": token.timeframes,
                    "quality_score": f"{token.data_quality:.1%}",
                    "total_bars": token.total_bars,
                    "gaps": token.gaps_detected,
                    "last_update": token.last_update.date().isoformat(),
                }
                for symbol, token in self.tokens.items()
            },
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Génère recommandations basées sur inventaire."""
        recs = []

        # Tokens haute qualité
        high_quality = [
            s for s, t in self.tokens.items() if t.data_quality >= 0.95
        ]
        if high_quality:
            recs.append(
                f"Tokens haute qualité (≥95%): {', '.join(high_quality[:5])}"
            )

        # Tokens récents
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent = [
            s for s, t in self.tokens.items() if t.last_update >= recent_cutoff
        ]
        if recent:
            recs.append(f"{len(recent)} tokens mis à jour derniers 30 jours")

        # Warnings gaps
        gaps_tokens = [s for s, t in self.tokens.items() if t.gaps_detected > 10]
        if gaps_tokens:
            recs.append(
                f"⚠️ {len(gaps_tokens)} tokens avec gaps >10: {', '.join(gaps_tokens[:3])}"
            )

        return recs


# =============================================================================
# STRATEGY REGISTRY - Catalogue Stratégies Existantes
# =============================================================================


@dataclass
class StrategyVersion:
    """Version stratégie avec métadonnées."""

    name: str
    """Nom stratégie (ex: ma_crossover)."""

    version: str
    """Version (ex: v1.0, v2.1, optimized_2024-11-21)."""

    params: dict[str, Any]
    """Paramètres stratégie."""

    performance: dict[str, float] | None = None
    """Métriques performances (sharpe, sortino, etc)."""

    tier_s_score: float | None = None
    """Score Tier S (0-100)."""

    created_at: datetime = field(default_factory=datetime.now)
    """Date création."""

    created_by: str = "human"
    """Créateur (human, analyst, strategist, optimizer)."""

    parent_version: str | None = None
    """Version parente (si optimisation/fork)."""

    description: str = ""
    """Description modifications."""

    status: str = "active"
    """Status: active, archived, deprecated."""

    backtests: list[dict[str, Any]] = field(default_factory=list)
    """Historique backtests (symbol, period, results)."""


@dataclass
class StrategyRegistry:
    """Catalogue stratégies avec versioning."""

    strategies: dict[str, list[StrategyVersion]] = field(default_factory=dict)
    """Stratégies par nom → versions."""

    registry_path: Path = Path("./exports/strategy_registry.json")
    """Chemin fichier registry persistant."""

    def add_strategy(self, strategy: StrategyVersion):
        """Ajoute version stratégie."""
        if strategy.name not in self.strategies:
            self.strategies[strategy.name] = []

        self.strategies[strategy.name].append(strategy)
        self.save()

        logger.info(
            f"Strategy registered: {strategy.name} {strategy.version} "
            f"(created by {strategy.created_by})"
        )

    def get_latest_version(self, name: str) -> StrategyVersion | None:
        """Retourne dernière version stratégie."""
        if name not in self.strategies:
            return None

        # Trier par date création
        versions = sorted(
            self.strategies[name], key=lambda v: v.created_at, reverse=True
        )

        return versions[0]

    def get_best_version(self, name: str, metric: str = "sharpe_ratio") -> StrategyVersion | None:
        """
        Retourne meilleure version selon métrique.

        Args:
            name: Nom stratégie
            metric: Métrique tri (sharpe_ratio, sortino_ratio, tier_s_score)

        Returns:
            Meilleure version ou None
        """
        if name not in self.strategies:
            return None

        versions_with_perf = [
            v for v in self.strategies[name] if v.performance is not None
        ]

        if not versions_with_perf:
            return None

        # Trier par métrique
        if metric == "tier_s_score":
            return max(
                versions_with_perf,
                key=lambda v: v.tier_s_score or 0,
            )
        else:
            return max(
                versions_with_perf,
                key=lambda v: v.performance.get(metric, 0),
            )

    def get_evolution_tree(self, name: str) -> dict[str, Any]:
        """
        Retourne arbre évolution stratégie.

        Args:
            name: Nom stratégie

        Returns:
            Tree structure parent→children
        """
        if name not in self.strategies:
            return {}

        versions = self.strategies[name]

        # Construire graph
        tree = {}
        for v in versions:
            tree[v.version] = {
                "params": v.params,
                "performance": v.performance,
                "tier_s": v.tier_s_score,
                "parent": v.parent_version,
                "created_at": v.created_at.isoformat(),
                "created_by": v.created_by,
            }

        return tree

    def save(self):
        """Sauvegarde registry sur disque."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Convertir en JSON sérialisable
        data = {
            name: [
                {
                    **asdict(v),
                    "created_at": v.created_at.isoformat(),
                }
                for v in versions
            ]
            for name, versions in self.strategies.items()
        }

        self.registry_path.write_text(json.dumps(data, indent=2))

    def load(self):
        """Charge registry depuis disque."""
        if not self.registry_path.exists():
            logger.warning(f"Registry not found: {self.registry_path}")
            return

        data = json.loads(self.registry_path.read_text())

        self.strategies = {}
        for name, versions in data.items():
            self.strategies[name] = [
                StrategyVersion(
                    **{
                        **v,
                        "created_at": datetime.fromisoformat(v["created_at"]),
                    }
                )
                for v in versions
            ]

        logger.info(
            f"Registry loaded: {len(self.strategies)} strategies, "
            f"{sum(len(v) for v in self.strategies.values())} versions"
        )

    def to_llm_context(self, name: str | None = None) -> dict[str, Any]:
        """
        Convertit registry en contexte LLM.

        Args:
            name: Nom stratégie spécifique (None = toutes)

        Returns:
            Dict structuré pour agents
        """
        if name:
            # Contexte stratégie spécifique
            if name not in self.strategies:
                return {"error": f"Strategy {name} not found"}

            latest = self.get_latest_version(name)
            best = self.get_best_version(name)
            tree = self.get_evolution_tree(name)

            return {
                "strategy_name": name,
                "total_versions": len(self.strategies[name]),
                "latest_version": {
                    "version": latest.version,
                    "params": latest.params,
                    "performance": latest.performance,
                    "tier_s_score": latest.tier_s_score,
                    "created_at": latest.created_at.date().isoformat(),
                    "created_by": latest.created_by,
                },
                "best_version": {
                    "version": best.version,
                    "params": best.params,
                    "performance": best.performance,
                    "tier_s_score": best.tier_s_score,
                } if best else None,
                "evolution_tree": tree,
            }
        else:
            # Contexte global
            return {
                "total_strategies": len(self.strategies),
                "strategies": {
                    name: {
                        "total_versions": len(versions),
                        "latest": versions[-1].version,
                        "best_sharpe": max(
                            (
                                v.performance.get("sharpe_ratio", 0)
                                for v in versions
                                if v.performance
                            ),
                            default=0,
                        ),
                        "best_tier_s": max(
                            (v.tier_s_score or 0 for v in versions), default=0
                        ),
                    }
                    for name, versions in self.strategies.items()
                },
            }


# =============================================================================
# CONTEXT MANAGER - Orchestrateur Contexte Global
# =============================================================================


class ContextManager:
    """Manager contexte global pour agents LLM."""

    def __init__(
        self,
        data_dir: Path = Path("./data"),
        registry_path: Path = Path("./exports/strategy_registry.json"),
    ):
        """
        Initialise context manager.

        Args:
            data_dir: Dossier données OHLCV
            registry_path: Chemin registry stratégies
        """
        self.data_dir = data_dir
        self.inventory = DataInventory()
        self.registry = StrategyRegistry(registry_path=registry_path)

        # Charger registry existant
        self.registry.load()

        # Scanner données disponibles
        self._scan_data_directory()

        logger.info(
            f"ContextManager initialized: {self.inventory.total_tokens} tokens, "
            f"{len(self.registry.strategies)} strategies"
        )

    def _scan_data_directory(self):
        """Scanne dossier data pour inventorier tokens."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return

        # Scanner fichiers parquet/csv
        for file in self.data_dir.glob("**/*.parquet"):
            self._process_data_file(file)

        for file in self.data_dir.glob("**/*.csv"):
            self._process_data_file(file)

    def _process_data_file(self, file_path: Path):
        """
        Traite fichier données pour extraire métadonnées.

        Args:
            file_path: Chemin fichier
        """
        try:
            # Parser nom fichier (ex: BTCUSDC_15m_2024-01-01_2024-12-31.parquet)
            parts = file_path.stem.split("_")
            if len(parts) < 2:
                return

            symbol = parts[0]
            timeframe = parts[1] if len(parts) > 1 else "unknown"

            # Charger metadata (éviter chargement complet)
            if file_path.suffix == ".parquet":
                df = pd.read_parquet(file_path, columns=["timestamp"])
            else:
                df = pd.read_csv(file_path, usecols=["timestamp"], nrows=10000)

            if df.empty:
                return

            # Extraire dates
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            start_date = df["timestamp"].min()
            end_date = df["timestamp"].max()

            # Détecter gaps (simpliste)
            expected_freq = self._get_expected_frequency(timeframe)
            gaps = self._detect_gaps(df["timestamp"], expected_freq)

            # Calculer qualité
            quality = 1.0 - (gaps / max(len(df), 1))

            # Créer/update token
            if symbol in self.inventory.tokens:
                token = self.inventory.tokens[symbol]
                if timeframe not in token.timeframes:
                    token.timeframes.append(timeframe)
                token.start_date = min(token.start_date, start_date)
                if token.end_date:
                    token.end_date = max(token.end_date, end_date)
                token.total_bars += len(df)
                token.gaps_detected += gaps
            else:
                token = TokenAvailability(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframes=[timeframe],
                    data_quality=quality,
                    total_bars=len(df),
                    gaps_detected=gaps,
                    last_update=datetime.fromtimestamp(file_path.stat().st_mtime),
                )
                self.inventory.add_token(token)

        except Exception as e:
            logger.debug(f"Failed to process {file_path}: {e}")

    def _get_expected_frequency(self, timeframe: str) -> str:
        """Convertit timeframe en pandas frequency."""
        mapping = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }
        return mapping.get(timeframe, "1H")

    def _detect_gaps(self, timestamps: pd.Series, freq: str) -> int:
        """Détecte nombre gaps dans série temporelle."""
        timestamps = timestamps.sort_values()
        expected = pd.date_range(
            start=timestamps.min(), end=timestamps.max(), freq=freq
        )
        return len(expected) - len(timestamps)

    def get_full_context(
        self, strategy_name: str | None = None
    ) -> dict[str, Any]:
        """
        Retourne contexte complet pour agents.

        Args:
            strategy_name: Nom stratégie focus (None = global)

        Returns:
            Dict contexte LLM-ready
        """
        return {
            "data_inventory": self.inventory.to_llm_context(),
            "strategy_registry": self.registry.to_llm_context(strategy_name),
            "system_info": {
                "current_date": datetime.now().date().isoformat(),
                "data_range": "2024-01-01 to today",
                "total_tokens_tracked": self.inventory.total_tokens,
                "total_strategies": len(self.registry.strategies),
            },
        }

    def validate_optimization_request(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        strategy_name: str,
    ) -> tuple[bool, str, dict[str, Any]]:
        """
        Valide requête optimisation avant démarrage.

        Args:
            symbol: Token à trader
            start_date: Date début backtest
            end_date: Date fin backtest
            timeframe: Timeframe
            strategy_name: Nom stratégie

        Returns:
            (valide, message, contexte_enrichi)
        """
        issues = []
        warnings = []

        # 1. Valider token disponible
        valid_token, token_msg = self.inventory.validate_token_period(
            symbol, start_date, end_date, timeframe
        )
        if not valid_token:
            issues.append(f"❌ Token: {token_msg}")
        else:
            warnings.append(f"✅ Token: {token_msg}")

        # 2. Valider stratégie existe
        strategy_version = self.registry.get_latest_version(strategy_name)
        if not strategy_version:
            warnings.append(
                f"⚠️ Strategy {strategy_name} not found in registry. Will use default params."
            )
        else:
            warnings.append(
                f"✅ Strategy: {strategy_name} {strategy_version.version} loaded"
            )

        # 3. Recommandations alternatives
        alternatives = self.inventory.get_available_tokens(
            start_date, end_date, timeframe
        )
        if alternatives and symbol not in alternatives:
            warnings.append(
                f"💡 Alternative tokens disponibles: {', '.join(alternatives[:5])}"
            )

        # Construire message
        if issues:
            return False, "\n".join(issues), {}
        else:
            return True, "\n".join(warnings), self.get_full_context(strategy_name)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_default_inventory() -> DataInventory:
    """Crée inventaire par défaut avec tokens principaux."""
    inventory = DataInventory()

    # Tokens majeurs disponibles depuis 2024-01-01
    major_tokens = [
        ("BTCUSDC", 0.98, 500000),
        ("ETHUSDC", 0.97, 480000),
        ("BNBUSDC", 0.95, 450000),
        ("SOLUSDC", 0.96, 460000),
        ("XRPUSDC", 0.94, 440000),
    ]

    for symbol, quality, bars in major_tokens:
        token = TokenAvailability(
            symbol=symbol,
            start_date=datetime(2024, 1, 1),
            end_date=None,  # Actif
            timeframes=["1m", "5m", "15m", "1h", "4h", "1d"],
            data_quality=quality,
            total_bars=bars,
            gaps_detected=int(bars * (1 - quality)),
            last_update=datetime.now(),
        )
        inventory.add_token(token)

    return inventory
