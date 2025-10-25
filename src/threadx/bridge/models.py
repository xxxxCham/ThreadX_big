"""
ThreadX Bridge Models - DataClasses Request/Result
==================================================

DataClasses typées pour requêtes et réponses Bridge.
Aucune logique métier, uniquement structures de données.

Usage:
    >>> from threadx.bridge.models import BacktestRequest, BacktestResult
    >>> req = BacktestRequest(
    ...     symbol='BTCUSDT',
    ...     timeframe='1h',
    ...     strategy='bollinger_reversion',
    ...     params={'period': 20, 'std': 2.0}
    ... )
    >>> # Controller consommera req et retournera BacktestResult

Author: ThreadX Framework
Version: Prompt 2 - Bridge Foundation
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestRequest:
    """Requête pour lancer un backtest.

    Utilisée par CLI et Dash callbacks pour définir tous les paramètres
    nécessaires à l'exécution d'un backtest via BacktestController.

    Attributes:
        symbol: Paire de trading (ex. 'BTCUSDT', 'BNBUSDT').
        timeframe: Timeframe OHLCV (ex. '1h', '15m', '1d').
        strategy: Nom stratégie enregistrée (ex. 'bollinger_reversion').
        params: Paramètres stratégie {key: value} (ex. {'period': 20}).
        start_date: Date début ISO 8601 ou None (utilise dataset complet).
        end_date: Date fin ISO 8601 ou None.
        initial_cash: Capital initial en USD (default: 10000.0).
        use_gpu: Activer accélération GPU si disponible (default: False).
    """

    symbol: str
    timeframe: str
    strategy: str
    params: dict[str, Any] = field(default_factory=dict)
    start_date: str | None = None
    end_date: str | None = None
    initial_cash: float = 10000.0
    use_gpu: bool = False

    def validate(self) -> bool:
        """Validation basique des champs requis (non métier).

        Returns:
            True si symbol, timeframe et strategy sont non-vides.

        Note:
            Validation métier (params, dates) effectuée par Engine.
        """
        return bool(self.symbol and self.timeframe and self.strategy)


@dataclass
class BacktestResult:
    """Résultat d'un backtest.

    Retourné par BacktestController.run_backtest() après exécution.
    Contient tous les KPIs, trades et courbes pour analyse.

    Attributes:
        total_profit: PnL total en USD.
        total_return: Rendement total en pourcentage.
        sharpe_ratio: Ratio de Sharpe annualisé.
        max_drawdown: Drawdown maximum en pourcentage (négatif).
        win_rate: Taux de trades gagnants (0.0 à 1.0).
        trades: Liste des trades [{entry_time, exit_time, pnl, ...}].
        equity_curve: Courbe d'equity (valeur portefeuille par step).
        drawdown_curve: Courbe de drawdown en pourcentage.
        metrics: Métriques supplémentaires {key: value}.
        execution_time: Temps d'exécution en secondes.
        metadata: Informations additionnelles (cache hits, GPU usage, etc.).
    """

    total_profit: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: list[dict[str, Any]]
    equity_curve: list[float]
    drawdown_curve: list[float]
    metrics: dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndicatorRequest:
    """Requête pour construire des indicateurs techniques.

    Utilisée pour demander le calcul d'indicateurs via IndicatorController,
    avec support cache automatique.

    Attributes:
        symbol: Paire de trading (ex. 'BTCUSDT').
        timeframe: Timeframe des données (ex. '1h').
        indicators: Dict d'indicateurs {nom: params}
                   (ex. {'ema': {'period': 50}}).
        data_path: Chemin vers données Parquet ou None (auto-detect).
        force_recompute: Ignorer cache et recalculer (default: False).
        use_gpu: Utiliser GPU pour calculs si disponible (default: False).
    """

    symbol: str
    timeframe: str
    indicators: dict[str, dict[str, Any]]
    data_path: str | None = None
    force_recompute: bool = False
    use_gpu: bool = False

    def validate(self) -> bool:
        """Validation basique des champs requis.

        Returns:
            True si symbol, timeframe et indicators sont valides.
        """
        return bool(
            self.symbol
            and self.timeframe
            and self.indicators
            and len(self.indicators) > 0
        )


@dataclass
class IndicatorResult:
    """Résultat du calcul d'indicateurs.

    Retourné par IndicatorController.build_indicators() avec valeurs
    calculées et informations de cache.

    Attributes:
        indicator_values: Dict {nom_indicateur: valeurs_array}.
        cache_hits: Nombre d'indicateurs chargés depuis cache.
        cache_misses: Nombre d'indicateurs recalculés.
        build_time: Temps total de construction en secondes.
        metadata: Informations cache (paths, checksums, etc.).
    """

    indicator_values: dict[str, Any]
    cache_hits: int = 0
    cache_misses: int = 0
    build_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepRequest:
    """Requête pour exécuter un parameter sweep.

    Utilisée pour l'optimisation paramétrique via SweepController.
    Explore une grille de paramètres et retourne les meilleures combinaisons.

    Attributes:
        symbol: Paire de trading (ex. 'BTCUSDT').
        timeframe: Timeframe des données (ex. '1h').
        strategy: Stratégie à optimiser (ex. 'bollinger_reversion').
        param_grid: Grille de paramètres {param: [val1, val2, ...]} ou
                   {param: {'min': x, 'max': y, 'step': z}}.
        optimization_criteria: Critères de tri
                              ['sharpe_ratio', 'total_return', ...].
        top_n: Nombre de meilleurs résultats à retourner (default: 10).
        max_workers: Nombre de workers parallèles (default: 4).
        use_gpu: Activer GPU pour backtests (default: False).
    """

    symbol: str
    timeframe: str
    strategy: str
    param_grid: dict[str, Any]
    optimization_criteria: list[str] = field(
        default_factory=lambda: ["sharpe_ratio", "total_return"]
    )
    top_n: int = 10
    max_workers: int = 4
    use_gpu: bool = False

    def validate(self) -> bool:
        """Validation basique des champs requis.

        Returns:
            True si tous champs requis sont valides.
        """
        return bool(
            self.symbol
            and self.timeframe
            and self.strategy
            and self.param_grid
            and len(self.param_grid) > 0
        )


@dataclass
class SweepResult:
    """Résultat d'un parameter sweep.

    Retourné par SweepController.run_sweep() avec les meilleures
    combinaisons de paramètres trouvées.

    Attributes:
        best_params: Meilleurs paramètres trouvés {param: value}.
        best_sharpe: Meilleur Sharpe ratio obtenu.
        best_return: Meilleur rendement total obtenu.
        top_results: Liste des top N résultats (sorted).
        total_combinations: Nombre total de combinaisons testées.
        pruned_combinations: Nombre de combinaisons élaguées.
        execution_time: Temps total d'exécution en secondes.
        metadata: Informations détaillées (cache, GPU, etc.).
    """

    best_params: dict[str, Any]
    best_sharpe: float
    best_return: float
    top_results: list[dict[str, Any]]
    total_combinations: int
    pruned_combinations: int = 0
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataRequest:
    """Requête pour chargement et validation de données.

    Utilisée par DataController pour charger et valider des données OHLCV.

    Attributes:
        symbol: Paire de trading (ex. 'BTCUSDT').
        timeframe: Timeframe des données (ex. '1h').
        data_path: Chemin vers fichier Parquet ou None (auto-detect).
        validate: Activer validation qualité données (default: True).
        required_columns: Colonnes obligatoires à vérifier.
        start_date: Date début ou None (tout le dataset).
        end_date: Date fin ou None.
    """

    symbol: str
    timeframe: str
    data_path: str | None = None
    validate: bool = True
    required_columns: list[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )
    start_date: str | None = None
    end_date: str | None = None

    def validate_request(self) -> bool:
        """Validation basique de la requête.

        Returns:
            True si symbol et timeframe sont valides.
        """
        return bool(self.symbol and self.timeframe)


@dataclass
class DataValidationResult:
    """Résultat de validation de données.

    Retourné par DataController.validate_data() après vérification qualité.

    Attributes:
        valid: True si données passent toutes validations.
        row_count: Nombre de lignes dans le dataset.
        missing_values: Nombre de valeurs manquantes détectées.
        duplicate_rows: Nombre de lignes dupliquées.
        date_gaps: Nombre d'écarts temporels anormaux.
        outliers_count: Nombre de valeurs aberrantes détectées.
        quality_score: Score qualité global (0.0 à 10.0).
        errors: Liste des erreurs de validation détectées.
        warnings: Liste des avertissements (non-bloquants).
        metadata: Informations détaillées (colonnes, types, stats, etc.).
    """

    valid: bool
    row_count: int
    missing_values: int = 0
    duplicate_rows: int = 0
    date_gaps: int = 0
    outliers_count: int = 0
    quality_score: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Configuration:
    """Configuration globale Bridge.

    Utilisée pour configurer tous les controllers avec paramètres communs.

    Attributes:
        max_workers: Nombre max de workers parallèles (default: 4).
        gpu_enabled: Activer support GPU global (default: False).
        xp_layer: Backend calcul 'numpy' ou 'cupy' (default: 'numpy').
        cache_path: Chemin vers dossier cache (default: 'cache/').
        log_level: Niveau de logging 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
        validate_requests: Valider requêtes avant exécution (default: True).
        enable_profiling: Activer profiling performances (default: False).
    """

    max_workers: int = 4
    gpu_enabled: bool = False
    xp_layer: str = "numpy"
    cache_path: str = "cache/"
    log_level: str = "INFO"
    validate_requests: bool = True
    enable_profiling: bool = False

    def validate(self) -> bool:
        """Validation de la configuration.

        Returns:
            True si configuration est cohérente.

        Raises:
            ValueError: Si xp_layer invalide ou max_workers < 1.
        """
        if self.xp_layer not in ("numpy", "cupy"):
            raise ValueError(f"Invalid xp_layer: {self.xp_layer}")
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        return True
