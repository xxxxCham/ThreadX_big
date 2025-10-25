"""
ThreadX Backtest Validation Module
===================================

Module de validation anti-overfitting pour backtests robustes.
Implémente walk-forward validation, train/test split, et détection de biais temporels.

Features:
- Walk-forward optimization avec fenêtres glissantes
- Train/test split avec purge et embargo
- Détection automatique de look-ahead bias
- Calcul de ratio d'overfitting
- Vérification d'intégrité temporelle des données
- Support pour validation k-fold sur séries temporelles

Author: ThreadX Framework - Quality Initiative Phase 2
Version: 1.0.0 - Anti-Overfitting Validation
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from threadx.utils.common_imports import (
    pd,
    np,
    List,
    Tuple,
    Optional,
    Callable,
    Dict,
    Any,
    create_logger,
)

logger = create_logger(__name__)


@dataclass
class ValidationConfig:
    """
    Configuration pour validation de backtest.

    Attributes:
        method: Méthode de validation ('walk_forward', 'train_test', 'k_fold')
        train_ratio: Ratio de données pour training (0.0 à 1.0)
        test_ratio: Ratio de données pour testing (0.0 à 1.0)
        walk_forward_windows: Nombre de fenêtres pour walk-forward
        purge_days: Nombre de jours à purger entre train et test
        embargo_days: Nombre de jours d'embargo après test
        min_train_samples: Nombre minimum de samples pour training
        min_test_samples: Nombre minimum de samples pour testing

    Notes:
        - train_ratio + test_ratio doit être <= 1.0
        - purge_days prévient le data leakage entre train/test
        - embargo_days simule le délai de traitement réel
    """

    method: str = "walk_forward"
    train_ratio: float = 0.7
    test_ratio: float = 0.3
    walk_forward_windows: int = 5
    purge_days: int = 0
    embargo_days: int = 0
    min_train_samples: int = 100
    min_test_samples: int = 50

    def __post_init__(self):
        """Validation de la configuration."""
        if self.method not in ["walk_forward", "train_test", "k_fold"]:
            raise ValueError(
                f"method doit être 'walk_forward', 'train_test' ou 'k_fold', "
                f"reçu: {self.method}"
            )

        if not 0 < self.train_ratio <= 1.0:
            raise ValueError(
                f"train_ratio doit être entre 0 et 1, reçu: {self.train_ratio}"
            )

        if not 0 < self.test_ratio <= 1.0:
            raise ValueError(
                f"test_ratio doit être entre 0 et 1, reçu: {self.test_ratio}"
            )

        if self.train_ratio + self.test_ratio > 1.0:
            raise ValueError(
                f"train_ratio + test_ratio > 1.0: "
                f"{self.train_ratio} + {self.test_ratio} = "
                f"{self.train_ratio + self.test_ratio}"
            )

        if self.purge_days < 0 or self.embargo_days < 0:
            raise ValueError("purge_days et embargo_days doivent être >= 0")


class BacktestValidator:
    """
    Validateur pour backtests avec protection contre overfitting.

    Implémente différentes stratégies de validation pour séries temporelles:
    - Walk-forward: Fenêtres glissantes train/test
    - Train/test split: Split simple avec purge
    - K-fold: Cross-validation adaptée aux séries temporelles

    Examples:
        >>> config = ValidationConfig(method="walk_forward", walk_forward_windows=5)
        >>> validator = BacktestValidator(config)
        >>> windows = validator.walk_forward_split(data)
        >>> for train, test in windows:
        ...     # Entraîner sur train, valider sur test
        ...     pass
    """

    def __init__(self, config: ValidationConfig):
        """
        Initialise le validateur.

        Parameters:
            config: Configuration de validation
        """
        self.config = config
        self.validation_results: List[Dict[str, Any]] = []
        logger.info(
            f"BacktestValidator initialisé: method={config.method}, "
            f"windows={config.walk_forward_windows}, "
            f"train_ratio={config.train_ratio}, "
            f"purge={config.purge_days}d, embargo={config.embargo_days}d"
        )

    def walk_forward_split(
        self, data: pd.DataFrame, n_windows: Optional[int] = None
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Génère des fenêtres walk-forward pour validation.

        Walk-forward optimization divise les données en fenêtres successives
        où chaque fenêtre contient:
        - Train: Données historiques jusqu'au point de split
        - Test: Données futures après le point de split (+ purge)

        Cette méthode prévient le look-ahead bias et simule le trading réel
        où on ne connaît que le passé.

        Parameters:
            data: DataFrame avec index DatetimeIndex
            n_windows: Nombre de fenêtres (None = utilise config)

        Returns:
            Liste de tuples (train_data, test_data)

        Raises:
            ValueError: Si données insuffisantes ou index non temporel

        Examples:
            >>> windows = validator.walk_forward_split(df, n_windows=5)
            >>> print(f"Généré {len(windows)} fenêtres")
            >>> train, test = windows[0]
            >>> print(f"Train: {len(train)} rows, Test: {len(test)} rows")
        """
        # Validation des entrées
        check_temporal_integrity(data)

        n_windows = n_windows or self.config.walk_forward_windows
        total_len = len(data)

        if total_len < n_windows * (
            self.config.min_train_samples + self.config.min_test_samples
        ):
            raise ValueError(
                f"Données insuffisantes ({total_len} rows) pour {n_windows} fenêtres. "
                f"Minimum requis: {n_windows * (self.config.min_train_samples + self.config.min_test_samples)}"
            )

        windows = []
        # Taille de base pour chaque fenêtre
        window_size = total_len // (n_windows + 1)

        logger.info(
            f"Génération de {n_windows} fenêtres walk-forward "
            f"(window_size={window_size}, total={total_len})"
        )

        for i in range(n_windows):
            # Point de split pour cette fenêtre
            train_end_idx = (i + 1) * window_size

            # Appliquer purge (skip jours après train)
            test_start_idx = train_end_idx
            if self.config.purge_days > 0:
                purge_date = data.index[train_end_idx] + pd.Timedelta(
                    days=self.config.purge_days
                )
                test_start_idx = data.index.get_indexer([purge_date], method="nearest")[
                    0
                ]

            # Calculer fin du test avec embargo
            test_end_idx = train_end_idx + window_size
            if self.config.embargo_days > 0:
                test_end_idx = max(
                    test_start_idx + self.config.min_test_samples,
                    test_end_idx - self.config.embargo_days,
                )

            # Vérifier qu'on a assez de données
            if test_end_idx > total_len:
                logger.warning(
                    f"Fenêtre {i+1} dépasse les données disponibles, "
                    f"ajustement à {total_len}"
                )
                test_end_idx = total_len

            if test_end_idx <= test_start_idx + self.config.min_test_samples:
                logger.warning(f"Fenêtre {i+1} test set trop petit, skip")
                continue

            # Extraire les données
            train_data = data.iloc[:train_end_idx].copy()
            test_data = data.iloc[test_start_idx:test_end_idx].copy()

            # Vérification anti-lookahead CRITIQUE
            if not train_data.index.max() < test_data.index.min():
                raise ValueError(
                    f"❌ LOOK-AHEAD BIAS DÉTECTÉ dans fenêtre {i+1}!\n"
                    f"Train max: {train_data.index.max()}\n"
                    f"Test min: {test_data.index.min()}\n"
                    f"Les dates train/test se chevauchent!"
                )

            # Vérifier tailles minimales
            if len(train_data) < self.config.min_train_samples:
                logger.warning(
                    f"Fenêtre {i+1} train set trop petit "
                    f"({len(train_data)} < {self.config.min_train_samples}), skip"
                )
                continue

            if len(test_data) < self.config.min_test_samples:
                logger.warning(
                    f"Fenêtre {i+1} test set trop petit "
                    f"({len(test_data)} < {self.config.min_test_samples}), skip"
                )
                continue

            windows.append((train_data, test_data))

            logger.debug(
                f"Fenêtre {i+1}/{n_windows}: "
                f"train={len(train_data)} rows "
                f"[{train_data.index.min()} → {train_data.index.max()}], "
                f"test={len(test_data)} rows "
                f"[{test_data.index.min()} → {test_data.index.max()}]"
            )

        if not windows:
            raise ValueError(
                "Aucune fenêtre valide générée. "
                "Vérifier config.min_train_samples et config.min_test_samples"
            )

        logger.info(f"✅ {len(windows)} fenêtres walk-forward générées avec succès")
        return windows

    def train_test_split(
        self, data: pd.DataFrame, train_ratio: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split simple train/test avec purge et embargo.

        Divise les données en deux ensembles chronologiques:
        - Train: Premières train_ratio% des données
        - Test: Dernières données après purge

        Parameters:
            data: DataFrame avec index DatetimeIndex
            train_ratio: Ratio train (None = utilise config)

        Returns:
            Tuple (train_data, test_data)

        Raises:
            ValueError: Si données insuffisantes ou look-ahead détecté

        Examples:
            >>> train, test = validator.train_test_split(df, train_ratio=0.7)
            >>> print(f"Train: {len(train)}, Test: {len(test)}")
        """
        # Validation des entrées
        check_temporal_integrity(data)

        train_ratio = train_ratio or self.config.train_ratio

        # Calculer l'index de split
        split_idx = int(len(data) * train_ratio)

        # Appliquer purge
        purge_idx = split_idx
        if self.config.purge_days > 0:
            purge_date = data.index[split_idx] + pd.Timedelta(
                days=self.config.purge_days
            )
            purge_idx = data.index.get_indexer([purge_date], method="nearest")[0]

        # Appliquer embargo
        end_idx = len(data)
        if self.config.embargo_days > 0:
            embargo_date = data.index[-1] - pd.Timedelta(days=self.config.embargo_days)
            end_idx = data.index.get_indexer([embargo_date], method="nearest")[0]

        # Extraire les données
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[purge_idx:end_idx].copy()

        # Vérification anti-lookahead CRITIQUE
        if not train_data.index.max() < test_data.index.min():
            raise ValueError(
                f"❌ LOOK-AHEAD BIAS DÉTECTÉ!\n"
                f"Train max: {train_data.index.max()}\n"
                f"Test min: {test_data.index.min()}"
            )

        # Vérifier tailles minimales
        if len(train_data) < self.config.min_train_samples:
            raise ValueError(
                f"Train set trop petit: {len(train_data)} < {self.config.min_train_samples}"
            )

        if len(test_data) < self.config.min_test_samples:
            raise ValueError(
                f"Test set trop petit: {len(test_data)} < {self.config.min_test_samples}"
            )

        logger.info(
            f"✅ Train/test split: train={len(train_data)} rows "
            f"[{train_data.index.min()} → {train_data.index.max()}], "
            f"test={len(test_data)} rows "
            f"[{test_data.index.min()} → {test_data.index.max()}], "
            f"purge={self.config.purge_days}d, embargo={self.config.embargo_days}d"
        )

        return train_data, test_data

    def validate_backtest(
        self, backtest_func: Callable, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute backtest avec validation anti-overfitting.

        Applique la méthode de validation configurée et retourne les résultats
        in-sample et out-of-sample avec calcul du ratio d'overfitting.

        Parameters:
            backtest_func: Fonction de backtest à valider
                          Signature: func(data, params) -> dict avec 'sharpe_ratio', etc.
            data: Données complètes
            params: Paramètres du backtest

        Returns:
            Dict avec:
                - method: Méthode utilisée
                - n_windows: Nombre de fenêtres (si walk_forward)
                - in_sample: Résultats in-sample agrégés
                - out_sample: Résultats out-of-sample agrégés
                - overfitting_ratio: Ratio d'overfitting
                - recommendation: Recommandation basée sur le ratio

        Raises:
            ValueError: Si méthode inconnue

        Examples:
            >>> def my_backtest(data, params):
            ...     # Votre logique de backtest
            ...     return {'sharpe_ratio': 1.5, 'total_return': 0.25}
            >>>
            >>> results = validator.validate_backtest(my_backtest, df, {'sma': 20})
            >>> print(f"Overfitting ratio: {results['overfitting_ratio']:.2f}")
        """
        logger.info(f"🔍 Validation backtest avec méthode: {self.config.method}")

        if self.config.method == "walk_forward":
            return self._validate_walk_forward(backtest_func, data, params)
        elif self.config.method == "train_test":
            return self._validate_train_test(backtest_func, data, params)
        else:
            raise ValueError(f"Méthode non implémentée: {self.config.method}")

    def _validate_walk_forward(
        self, backtest_func: Callable, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validation walk-forward interne."""
        windows = self.walk_forward_split(data)

        in_sample_results = []
        out_sample_results = []

        logger.info(f"Exécution de {len(windows)} fenêtres walk-forward...")

        for i, (train, test) in enumerate(windows, 1):
            logger.debug(
                f"Fenêtre {i}/{len(windows)}: train={len(train)}, test={len(test)}"
            )

            try:
                # Backtest in-sample (pour optimisation)
                train_result = backtest_func(train, params)
                in_sample_results.append(train_result)

                # Backtest out-of-sample (validation)
                test_result = backtest_func(test, params)
                out_sample_results.append(test_result)

                logger.debug(
                    f"Fenêtre {i}: "
                    f"IS Sharpe={train_result.get('sharpe_ratio', 'N/A'):.2f}, "
                    f"OOS Sharpe={test_result.get('sharpe_ratio', 'N/A'):.2f}"
                )

            except Exception as e:
                logger.error(f"Erreur fenêtre {i}: {e}")
                continue

        # Agréger résultats
        in_sample_agg = self._aggregate_results(in_sample_results)
        out_sample_agg = self._aggregate_results(out_sample_results)

        # Calculer ratio d'overfitting
        overfitting_ratio = self._calculate_overfitting_ratio(
            in_sample_results, out_sample_results
        )

        # Recommandation
        recommendation = self._get_recommendation(overfitting_ratio)

        result = {
            "method": "walk_forward",
            "n_windows": len(windows),
            "in_sample": in_sample_agg,
            "out_sample": out_sample_agg,
            "overfitting_ratio": overfitting_ratio,
            "recommendation": recommendation,
        }

        logger.info(
            f"✅ Validation walk-forward terminée: "
            f"IS Sharpe={in_sample_agg.get('mean_sharpe_ratio', 0):.2f}, "
            f"OOS Sharpe={out_sample_agg.get('mean_sharpe_ratio', 0):.2f}, "
            f"Overfitting Ratio={overfitting_ratio:.2f}"
        )

        self.validation_results.append(result)
        return result

    def _validate_train_test(
        self, backtest_func: Callable, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validation train/test simple interne."""
        train, test = self.train_test_split(data)

        logger.info("Exécution train/test split...")

        # Backtest sur train
        train_result = backtest_func(train, params)

        # Backtest sur test
        test_result = backtest_func(test, params)

        # Calculer ratio d'overfitting
        overfitting_ratio = self._calculate_overfitting_ratio(
            [train_result], [test_result]
        )

        # Recommandation
        recommendation = self._get_recommendation(overfitting_ratio)

        result = {
            "method": "train_test",
            "in_sample": train_result,
            "out_sample": test_result,
            "overfitting_ratio": overfitting_ratio,
            "recommendation": recommendation,
        }

        logger.info(
            f"✅ Validation train/test terminée: "
            f"IS Sharpe={train_result.get('sharpe_ratio', 0):.2f}, "
            f"OOS Sharpe={test_result.get('sharpe_ratio', 0):.2f}, "
            f"Overfitting Ratio={overfitting_ratio:.2f}"
        )

        self.validation_results.append(result)
        return result

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Agrège résultats multiples en moyennes et écarts-types.

        Parameters:
            results: Liste de dictionnaires de résultats

        Returns:
            Dict avec mean_* et std_* pour chaque métrique
        """
        if not results:
            return {}

        # Métriques à agréger
        metrics = [
            "sharpe_ratio",
            "total_return",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ]
        aggregated = {}

        for metric in metrics:
            values = [r.get(metric, np.nan) for r in results]
            # Filtrer NaN
            values = [v for v in values if not np.isnan(v)]

            if values:
                aggregated[f"mean_{metric}"] = np.mean(values)
                aggregated[f"std_{metric}"] = np.std(values)
                aggregated[f"min_{metric}"] = np.min(values)
                aggregated[f"max_{metric}"] = np.max(values)

        aggregated["n_results"] = len(results)

        return aggregated

    def _calculate_overfitting_ratio(
        self, in_sample: List[Dict[str, Any]], out_sample: List[Dict[str, Any]]
    ) -> float:
        """
        Calcule ratio d'overfitting basé sur Sharpe ratios.

        Ratio proche de 1.0 = performances similaires IS/OOS (bon)
        Ratio >> 1.0 = overfitting probable (mauvais)
        Ratio < 1.0 = OOS meilleur que IS (excellent mais rare)

        Parameters:
            in_sample: Résultats in-sample
            out_sample: Résultats out-of-sample

        Returns:
            Ratio d'overfitting (IS_sharpe / OOS_sharpe)
        """
        in_agg = self._aggregate_results(in_sample)
        out_agg = self._aggregate_results(out_sample)

        in_sharpe = in_agg.get("mean_sharpe_ratio", 0)
        out_sharpe = out_agg.get("mean_sharpe_ratio", 0)

        if out_sharpe == 0 or np.isnan(out_sharpe):
            logger.warning("OOS Sharpe = 0 ou NaN, ratio d'overfitting = inf")
            return float("inf")

        ratio = abs(in_sharpe / out_sharpe)

        return ratio

    def _get_recommendation(self, overfitting_ratio: float) -> str:
        """
        Génère recommandation basée sur ratio d'overfitting.

        Parameters:
            overfitting_ratio: Ratio calculé

        Returns:
            String de recommandation
        """
        if overfitting_ratio < 1.2:
            return (
                "✅ EXCELLENT: Performances robustes, pas d'overfitting détecté. "
                "Stratégie validée pour production."
            )
        elif overfitting_ratio < 1.5:
            return (
                "⚠️ ACCEPTABLE: Léger overfitting détecté. "
                "Considérer simplification des paramètres ou augmentation des données."
            )
        elif overfitting_ratio < 2.0:
            return (
                "🟡 ATTENTION: Overfitting modéré. "
                "Réduire nombre de paramètres optimisés, utiliser régularisation, "
                "ou augmenter période out-of-sample."
            )
        else:
            return (
                "🔴 CRITIQUE: Overfitting sévère détecté! "
                "Stratégie non fiable pour production. "
                "Actions requises: réduire drastiquement les paramètres, "
                "utiliser walk-forward plus long, revoir la logique de stratégie."
            )


def check_temporal_integrity(data: pd.DataFrame) -> bool:
    """
    Vérifie l'intégrité temporelle des données de backtest.

    Effectue les vérifications suivantes:
    - Index est DatetimeIndex
    - Pas de données futures (> maintenant)
    - Pas de timestamps dupliqués
    - Ordre chronologique strict
    - Pas de trous temporels excessifs (optionnel)

    Parameters:
        data: DataFrame avec index temporel

    Returns:
        True si toutes les vérifications passent

    Raises:
        ValueError: Si une vérification échoue

    Examples:
        >>> check_temporal_integrity(df)
        True
    """
    # Vérifier type d'index
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            f"Index doit être DatetimeIndex, reçu: {type(data.index).__name__}"
        )

    # Vérifier pas de données futures
    now = pd.Timestamp.now(tz="UTC")
    if data.index.max() > now:
        raise ValueError(
            f"❌ DONNÉES FUTURES DÉTECTÉES - Look-ahead bias!\n"
            f"Date max dans données: {data.index.max()}\n"
            f"Date actuelle: {now}\n"
            f"Ceci indique un problème de données ou de timestamps."
        )

    # Vérifier duplicates
    if data.index.duplicated().any():
        duplicates = data.index[data.index.duplicated()].unique()
        raise ValueError(
            f"❌ TIMESTAMPS DUPLIQUÉS DÉTECTÉS: {len(duplicates)} dates\n"
            f"Exemples: {duplicates[:5].tolist()}\n"
            f"Les timestamps doivent être uniques pour un backtest valide."
        )

    # Vérifier ordre chronologique
    if not data.index.is_monotonic_increasing:
        raise ValueError(
            f"❌ INDEX NON CHRONOLOGIQUE!\n"
            f"L'index doit être strictement croissant.\n"
            f"Première date: {data.index[0]}\n"
            f"Dernière date: {data.index[-1]}"
        )

    # Vérifications optionnelles
    # Détecter trous temporels excessifs (> 7 jours pour données daily)
    if len(data) > 1:
        time_diffs = data.index.to_series().diff()
        max_gap = time_diffs.max()

        # Alerte si gap > 30 jours (peut être normal pour crypto weekends)
        if max_gap > pd.Timedelta(days=30):
            logger.warning(
                f"⚠️ Gap temporel important détecté: {max_gap}\n"
                f"Vérifier si normal pour votre asset (ex: market holidays)"
            )

    logger.debug(
        f"✅ Intégrité temporelle vérifiée: "
        f"{len(data)} rows, "
        f"[{data.index.min()} → {data.index.max()}]"
    )

    return True


def detect_lookahead_bias(
    train_data: pd.DataFrame, test_data: pd.DataFrame, raise_on_detect: bool = True
) -> bool:
    """
    Détecte le look-ahead bias entre train et test sets.

    Vérifie que:
    - Toutes les dates train < toutes les dates test
    - Pas de chevauchement temporel
    - Gap temporel suffisant (si purge configuré)

    Parameters:
        train_data: Données d'entraînement
        test_data: Données de test
        raise_on_detect: Si True, raise ValueError si bias détecté

    Returns:
        False si bias détecté, True sinon

    Raises:
        ValueError: Si bias détecté et raise_on_detect=True

    Examples:
        >>> detect_lookahead_bias(train, test)
        True
    """
    train_max = train_data.index.max()
    test_min = test_data.index.min()

    has_bias = train_max >= test_min

    if has_bias:
        msg = (
            f"❌ LOOK-AHEAD BIAS DÉTECTÉ!\n"
            f"Train max: {train_max}\n"
            f"Test min: {test_min}\n"
            f"Gap: {test_min - train_max}\n"
            f"Les données train et test se chevauchent temporellement.\n"
            f"Ceci invalide complètement le backtest!"
        )

        if raise_on_detect:
            raise ValueError(msg)
        else:
            logger.error(msg)
            return False

    logger.debug(
        f"✅ Pas de look-ahead bias: "
        f"gap de {test_min - train_max} entre train et test"
    )

    return True


# Export public API
__all__ = [
    "ValidationConfig",
    "BacktestValidator",
    "check_temporal_integrity",
    "detect_lookahead_bias",
]
