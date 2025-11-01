"""
ThreadX - Indicator Range Presets
==================================

Gestion des plages d'optimisation pré-définies pour les indicateurs techniques.

Ce module charge les configurations depuis indicator_ranges.toml et fournit
des méthodes pour mapper automatiquement les paramètres de stratégies aux plages
d'optimisation recommandées.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import toml
import logging

from threadx.utils.log import get_logger

logger = get_logger(__name__)

# Chemin vers le fichier de configuration
RANGES_FILE = Path(__file__).parent / "indicator_ranges.toml"
EXECUTION_PRESETS_FILE = Path(__file__).parent / "execution_presets.toml"


@dataclass
class IndicatorRangePreset:
    """
    Représente un pré-réglage de plage d'optimisation pour un indicateur.

    Attributes:
        name: Nom de l'indicateur (ex: "bollinger.period")
        min: Valeur minimale
        max: Valeur maximale
        step: Pas d'incrémentation
        default: Valeur par défaut
        type: Type de paramètre ("numeric", "boolean", "categorical", "fixed")
        values: Valeurs possibles (pour categorical)
        description: Description du paramètre
    """

    name: str
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    default: Optional[Any] = None
    type: str = "numeric"  # "numeric", "boolean", "categorical", "fixed"
    values: Optional[List[Any]] = None
    value: Optional[Any] = None  # Pour type="fixed"
    description: str = ""

    def get_range(self) -> Tuple[float, float]:
        """Retourne la plage (min, max)"""
        if self.type != "numeric":
            raise ValueError(
                f"get_range() n'est applicable que pour type='numeric', got: {self.type}"
            )
        if self.min is None or self.max is None:
            raise ValueError(f"min ou max non défini pour {self.name}")
        return (self.min, self.max)

    def get_grid_values(self) -> List[Any]:
        """Génère une liste de valeurs pour grid search"""
        if self.type == "numeric":
            if self.min is None or self.max is None or self.step is None:
                raise ValueError(f"min, max, ou step non défini pour {self.name}")

            # Générer la grille
            values = []
            current = self.min
            while current <= self.max:
                values.append(current)
                current += self.step

            # S'assurer que max est inclus
            if values and abs(values[-1] - self.max) > 1e-10:
                values.append(self.max)

            return values

        elif self.type == "categorical":
            if self.values is None:
                raise ValueError(f"values non défini pour categorical {self.name}")
            return self.values.copy()

        elif self.type == "boolean":
            return [False, True]

        elif self.type == "fixed":
            if self.value is None:
                raise ValueError(f"value non défini pour fixed {self.name}")
            return [self.value]

        else:
            raise ValueError(f"Type inconnu: {self.type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "default": self.default,
            "type": self.type,
            "values": self.values,
            "value": self.value,
            "description": self.description,
        }


def load_all_presets() -> Dict[str, IndicatorRangePreset]:
    """
    Charge tous les presets depuis indicator_ranges.toml

    Returns:
        Dictionnaire {nom_indicateur: IndicatorRangePreset}
    """
    if not RANGES_FILE.exists():
        logger.warning(f"Fichier de presets non trouvé: {RANGES_FILE}")
        return {}

    logger.info(f"Chargement des presets depuis: {RANGES_FILE}")

    try:
        config = toml.load(RANGES_FILE)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier TOML: {e}")
        return {}

    presets = {}

    # Parcourir les catégories et indicateurs
    for category, indicators in config.items():
        for indicator_name, params in indicators.items():
            # Nom complet: category.indicator_name
            full_name = f"{category}.{indicator_name}"

            # Déterminer le type
            param_type = params.get("type", "numeric")

            # Créer le preset
            preset = IndicatorRangePreset(
                name=full_name,
                min=params.get("min"),
                max=params.get("max"),
                step=params.get("step"),
                default=params.get("default"),
                type=param_type,
                values=params.get("values"),
                value=params.get("value"),
                description=params.get("description", ""),
            )

            presets[full_name] = preset

            logger.debug(f"Preset chargé: {full_name} ({param_type})")

    logger.info(f"✓ {len(presets)} presets chargés")
    return presets


def get_indicator_range(indicator_name: str) -> Optional[IndicatorRangePreset]:
    """
    Récupère le preset pour un indicateur spécifique.

    Args:
        indicator_name: Nom de l'indicateur (ex: "bollinger.period", "macd.fast_period")

    Returns:
        IndicatorRangePreset ou None si non trouvé
    """
    presets = load_all_presets()
    return presets.get(indicator_name)


def list_available_indicators() -> List[str]:
    """
    Liste tous les indicateurs disponibles dans les presets.

    Returns:
        Liste des noms d'indicateurs
    """
    presets = load_all_presets()
    return sorted(presets.keys())


class StrategyPresetMapper:
    """
    Classe pour mapper automatiquement les paramètres d'une stratégie
    aux presets d'indicateurs disponibles.

    Usage:
        >>> mapper = StrategyPresetMapper("AmplitudeHunter")
        >>> mapper.add_mapping("bb_period", "bollinger.period")
        >>> mapper.add_mapping("macd_fast", "macd.fast_period")
        >>> ranges = mapper.get_optimization_ranges()
    """

    def __init__(self, strategy_name: str):
        """
        Initialise le mapper pour une stratégie.

        Args:
            strategy_name: Nom de la stratégie
        """
        self.strategy_name = strategy_name
        self.mappings: Dict[str, str] = {}  # {param_strategy: indicator_name}
        self.presets = load_all_presets()
        logger.info(f"StrategyPresetMapper initialisé pour: {strategy_name}")

    def add_mapping(self, strategy_param: str, indicator_name: str) -> None:
        """
        Ajoute un mapping entre un paramètre de stratégie et un indicateur.

        Args:
            strategy_param: Nom du paramètre dans la stratégie (ex: "bb_period")
            indicator_name: Nom de l'indicateur dans les presets (ex: "bollinger.period")
        """
        if indicator_name not in self.presets:
            logger.warning(
                f"Indicateur '{indicator_name}' non trouvé dans les presets. "
                f"Mapping ignoré pour '{strategy_param}'."
            )
            return

        self.mappings[strategy_param] = indicator_name
        logger.debug(f"Mapping ajouté: {strategy_param} -> {indicator_name}")

    def add_mappings(self, mappings: Dict[str, str]) -> None:
        """
        Ajoute plusieurs mappings en une seule fois.

        Args:
            mappings: Dictionnaire {strategy_param: indicator_name}
        """
        for strategy_param, indicator_name in mappings.items():
            self.add_mapping(strategy_param, indicator_name)

    def get_optimization_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Retourne les plages d'optimisation pour les paramètres mappés.

        Returns:
            Dictionnaire {strategy_param: (min, max)}
        """
        ranges = {}

        for strategy_param, indicator_name in self.mappings.items():
            preset = self.presets.get(indicator_name)
            if preset and preset.type == "numeric":
                try:
                    ranges[strategy_param] = preset.get_range()
                except ValueError as e:
                    logger.warning(
                        f"Impossible de récupérer la plage pour {strategy_param}: {e}"
                    )

        logger.info(
            f"✓ {len(ranges)} plages d'optimisation générées pour {self.strategy_name}"
        )
        return ranges

    def get_grid_parameters(self) -> Dict[str, List[Any]]:
        """
        Retourne les grilles de paramètres pour grid search.

        Returns:
            Dictionnaire {strategy_param: [valeurs]}
        """
        grid = {}

        for strategy_param, indicator_name in self.mappings.items():
            preset = self.presets.get(indicator_name)
            if preset:
                try:
                    grid[strategy_param] = preset.get_grid_values()
                except ValueError as e:
                    logger.warning(
                        f"Impossible de générer la grille pour {strategy_param}: {e}"
                    )

        logger.info(
            f"✓ {len(grid)} grilles de paramètres générées pour {self.strategy_name}"
        )
        return grid

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Retourne les valeurs par défaut pour les paramètres mappés.

        Returns:
            Dictionnaire {strategy_param: default_value}
        """
        defaults = {}

        for strategy_param, indicator_name in self.mappings.items():
            preset = self.presets.get(indicator_name)
            if preset and preset.default is not None:
                defaults[strategy_param] = preset.default

        logger.info(
            f"✓ {len(defaults)} valeurs par défaut générées pour {self.strategy_name}"
        )
        return defaults

    def get_preset_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Retourne les informations complètes sur les presets mappés.

        Returns:
            Dictionnaire {strategy_param: preset_dict}
        """
        info = {}

        for strategy_param, indicator_name in self.mappings.items():
            preset = self.presets.get(indicator_name)
            if preset:
                info[strategy_param] = preset.to_dict()

        return info


def get_strategy_preset(strategy_name: str) -> StrategyPresetMapper:
    """
    Factory pour créer un mapper pré-configuré pour une stratégie connue.

    Args:
        strategy_name: Nom de la stratégie ("AmplitudeHunter", "BBAtr", etc.)

    Returns:
        StrategyPresetMapper pré-configuré

    Example:
        >>> preset = get_strategy_preset("AmplitudeHunter")
        >>> ranges = preset.get_optimization_ranges()
    """
    mapper = StrategyPresetMapper(strategy_name)

    # Configuration des mappings selon la stratégie
    if strategy_name.lower() in ["amplitudehunter", "amplitude_hunter"]:
        mapper.add_mappings(
            {
                # Bollinger Bands
                "bb_period": "bollinger.period",
                "bb_std": "bollinger.std_dev",
                # MACD
                "macd_fast": "macd.fast_period",
                "macd_slow": "macd.slow_period",
                "macd_signal": "macd.signal_period",
                # ADX
                "adx_period": "adx.period",
                "adx_threshold": "adx.trend_threshold",
                # ATR
                "atr_period": "atr.period",
                "sl_atr_multiplier": "atr.stop_multiplier",
                # Paramètres spécifiques AmplitudeHunter
                "bbwidth_percentile_threshold": "amplitude_hunter.bbwidth_percentile_threshold",
                "volume_zscore_threshold": "amplitude_hunter.volume_zscore_threshold",
                "spring_lookback": "amplitude_hunter.spring_lookback",
                "pb_entry_threshold_min": "amplitude_hunter.pb_entry_threshold_min",
                "pb_entry_threshold_max": "amplitude_hunter.pb_entry_threshold_max",
                "amplitude_score_threshold": "amplitude_hunter.amplitude_score_threshold",
                "trailing_activation_gain_r": "amplitude_hunter.trailing_activation_gain_r",
                "trailing_chandelier_atr_mult": "amplitude_hunter.trailing_chandelier_atr_mult",
                "bip_partial_exit_pct": "amplitude_hunter.bip_partial_exit_pct",
                "risk_per_trade": "amplitude_hunter.risk_per_trade",
                "max_hold_bars": "amplitude_hunter.max_hold_bars",
                "pyramiding_max_adds": "amplitude_hunter.pyramiding_max_adds",
            }
        )

    elif strategy_name.lower() in ["bbatr", "bb_atr"]:
        mapper.add_mappings(
            {
                # Bollinger Bands
                "bb_period": "bollinger.period",
                "bb_std": "bollinger.std_dev",
                # ATR
                "atr_period": "atr.period",
                "atr_multiplier": "atr.stop_multiplier",
                # Risk management
                "risk_per_trade": "amplitude_hunter.risk_per_trade",  # Réutilise le même preset
                "max_hold_bars": "amplitude_hunter.max_hold_bars",
            }
        )

    elif strategy_name.lower() in ["bollingerdual", "bollinger_dual"]:
        mapper.add_mappings(
            {
                # Bollinger Bands
                "bb_period": "bollinger.period",
                "bb_std": "bollinger.std_dev",
                # Moving Average
                "ma_window": "sma.short_period",
                # Risk management
                "risk_per_trade": "amplitude_hunter.risk_per_trade",
                "max_hold_bars": "amplitude_hunter.max_hold_bars",
            }
        )

    else:
        logger.warning(
            f"Stratégie '{strategy_name}' non reconnue. "
            f"Mapper créé sans mappings pré-définis."
        )

    return mapper


def load_execution_presets() -> Dict[str, Dict[str, Any]]:
    """
    🆕 Charge les presets d'exécution (workers, batch size, GPU targets).

    Returns:
        Dict avec structure: {
            "workers": {"auto": {...}, "manuel_30": {...}, ...},
            "batch": {...},
            "gpu": {...},
            "combined": {...}
        }
    """
    if not EXECUTION_PRESETS_FILE.exists():
        logger.warning(
            f"Fichier execution presets non trouvé: {EXECUTION_PRESETS_FILE}"
        )
        return {}

    try:
        data = toml.load(EXECUTION_PRESETS_FILE)
        logger.info(f"✅ Presets d'exécution chargés: {len(data)} catégories")
        return data
    except Exception as e:
        logger.error(f"Erreur chargement execution presets: {e}")
        return {}


def get_execution_preset(preset_name: str = "manuel_30") -> Dict[str, Any]:
    """
    🆕 Récupère un preset d'exécution par nom.

    Args:
        preset_name: Nom du preset (ex: "manuel_30", "auto", "aggressive")

    Returns:
        Configuration du preset

    Example:
        >>> preset = get_execution_preset("manuel_30")
        >>> print(preset["max_workers"])  # 30
        >>> print(preset["batch_size"])   # 2000
    """
    presets = load_execution_presets()

    # Chercher dans toutes les catégories
    for category, category_presets in presets.items():
        for name, config in category_presets.items():
            if name == preset_name or name.endswith(preset_name):
                logger.info(f"✅ Preset trouvé: {category}.{name}")
                return config

    logger.warning(f"Preset '{preset_name}' non trouvé, utilisant défaut")
    return {"max_workers": 30, "batch_size": 2000, "description": "Défaut manuel"}


# ==========================================================================
# MODULE EXPORTS
# ==========================================================================

__all__ = [
    "IndicatorRangePreset",
    "StrategyPresetMapper",
    "get_indicator_range",
    "get_strategy_preset",
    "list_available_indicators",
    "load_all_presets",
    "load_execution_presets",
    "get_execution_preset",
]
