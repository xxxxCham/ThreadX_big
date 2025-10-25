"""
ThreadX Bridge Exceptions - Error Hierarchy
===========================================

Hiérarchie d'exceptions pour la couche Bridge.
Permet distinction fine des erreurs (backtest, indicateur, data, sweep).

Usage:
    >>> from threadx.bridge.exceptions import BacktestError
    >>> try:
    ...     controller.run_backtest(request)
    ... except BacktestError as e:
    ...     print(f"Backtest failed: {e}")

Author: ThreadX Framework
Version: Prompt 2 - Bridge Foundation
"""


class BridgeError(Exception):
    """Exception de base pour toutes erreurs Bridge.

    Héritée par toutes exceptions spécifiques (Backtest, Indicator, etc.).
    Utilisée pour catch générique ou erreurs non-classifiées.

    Example:
        >>> try:
        ...     # Bridge operations
        ... except BridgeError as e:
        ...     print(f"Bridge error: {e}")
    """

    pass


class BacktestError(BridgeError):
    """Exception pour erreurs lors de backtests.

    Levée par BacktestController quand:
    - Validation BacktestRequest échoue
    - BacktestEngine lève une exception
    - Mapping résultat impossible

    Example:
        >>> raise BacktestError("Invalid strategy parameters")
    """

    pass


class IndicatorError(BridgeError):
    """Exception pour erreurs lors de calculs d'indicateurs.

    Levée par IndicatorController quand:
    - Validation IndicatorRequest échoue
    - IndicatorBank ne peut pas calculer indicateur
    - Cache corrompu ou inaccessible

    Example:
        >>> raise IndicatorError("EMA calculation failed: missing data")
    """

    pass


class SweepError(BridgeError):
    """Exception pour erreurs lors de parameter sweeps.

    Levée par SweepController quand:
    - Validation SweepRequest échoue
    - UnifiedOptimizationEngine ne peut pas générer grille
    - Aucun résultat valide après sweep

    Example:
        >>> raise SweepError("No valid combinations in param_grid")
    """

    pass


class DataError(BridgeError):
    """Exception pour erreurs lors de chargement/validation données.

    Levée par DataController quand:
    - Validation DataRequest échoue
    - Fichier Parquet introuvable ou corrompu
    - Données invalides (colonnes manquantes, types incorrects)

    Example:
        >>> raise DataError("Missing required column: 'close'")
    """

    pass


class ConfigurationError(BridgeError):
    """Exception pour erreurs de configuration Bridge.

    Levée quand:
    - Configuration.validate() détecte incohérences
    - Paramètres invalides (max_workers < 1, xp_layer inconnu)
    - Cache path inaccessible

    Example:
        >>> raise ConfigurationError("Invalid xp_layer: must be numpy|cupy")
    """

    pass


class ValidationError(BridgeError):
    """Exception pour erreurs de validation de requêtes.

    Levée quand validation automatique détecte champs manquants
    ou valeurs invalides dans Request DataClasses.

    Example:
        >>> raise ValidationError("BacktestRequest.symbol cannot be empty")
    """

    pass
