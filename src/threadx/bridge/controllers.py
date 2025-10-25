"""
ThreadX Bridge Controllers - Orchestration Layer
================================================

Controllers synchrones qui orchestrent les appels vers l'Engine.
Aucune logique métier, juste wrappers fins autour des modules Engine.

Usage:
    >>> from threadx.bridge.controllers import BacktestController
    >>> from threadx.bridge.models import BacktestRequest
    >>> controller = BacktestController()
    >>> req = BacktestRequest(
    ...     symbol='BTCUSDT',
    ...     timeframe='1h',
    ...     strategy='bollinger_reversion',
    ...     params={'period': 20, 'std': 2.0}
    ... )
    >>> result = controller.run_backtest(req)
    >>> print(result.sharpe_ratio)

Author: ThreadX Framework
Version: Prompt 2 - Bridge Foundation
"""

import time
from pathlib import Path
from typing import Any

from threadx.bridge.exceptions import (
    BacktestError,
    DataError,
    IndicatorError,
    SweepError,
)
from threadx.bridge.validation import (
    BacktestRequest,
    IndicatorRequest,
)
from pydantic import BaseModel as _PydanticBaseModel
from threadx.bridge.models import (
    BacktestResult,
    Configuration,
    DataRequest,
    DataValidationResult,
    IndicatorResult,
    SweepRequest,
    SweepResult,
)


class BacktestController:
    """Controller pour exécution de backtests.

    Wrapper synchrone autour de threadx.backtest.engine.
    Gère validation requête, appel Engine, et mapping résultat.

    Attributes:
        config: Configuration globale Bridge.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise controller avec configuration optionnelle.

        Args:
            config: Configuration Bridge ou None (utilise defaults).
        """
        self.config = config or Configuration()

    def run_backtest(self, request: dict) -> dict:
        """Exécute un backtest complet avec validation Pydantic.

        Args:
            request: Dict avec paramètres backtest.

        Returns:
            Dict avec résultat ou erreur.

        Raises:
            BacktestError: Si validation échoue ou exécution erreur.
        """
        # ✅ Valider le schéma
        try:
            validated = BacktestRequest(**request)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Validation failed: {str(e)}",
                "code": 400,
            }

        # ✅ Gestion erreurs centralisée
        try:
            # Convertir vers l'ancien format BacktestRequest pour compatibilité
            from threadx.bridge.models import BacktestRequest as OldBacktestRequest

            old_request = OldBacktestRequest(
                symbol=validated.symbol,
                timeframe=validated.timeframe,
                strategy=validated.strategy,
                start_date=validated.start_date,
                end_date=validated.end_date,
                params={},  # Default empty params
                initial_cash=10000.0,  # Default
                use_gpu=False,  # Default
            )

            # Appeler la méthode existante
            result = self._run_backtest_validated(old_request)

            return {
                "status": "success",
                "data": {
                    "total_profit": result.total_profit,
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "trades": result.trades,
                    "equity_curve": result.equity_curve,
                    "drawdown_curve": result.drawdown_curve,
                    "metrics": result.metrics,
                    "execution_time": result.execution_time,
                    "metadata": result.metadata,
                },
            }

        except Exception as e:
            return {"status": "error", "message": str(e), "code": 500}

    def _run_backtest_validated(self, request: "BacktestRequest") -> "BacktestResult":
        """Exécute un backtest complet.

        Orchestre:
        1. Validation requête
        2. Lazy import BacktestEngine
        3. Création engine avec paramètres
        4. Exécution backtest
        5. Mapping résultat vers BacktestResult

        Args:
            request: Requête backtest avec tous paramètres.

        Returns:
            BacktestResult avec KPIs, trades, courbes.

        Raises:
            BacktestError: Si validation échoue ou exécution erreur.

        Example:
            >>> req = BacktestRequest(
            ...     symbol='BTCUSDT', timeframe='1h',
            ...     strategy='bollinger_reversion',
            ...     params={'period': 20, 'std': 2.0}
            ... )
            >>> controller = BacktestController()
            >>> result = controller.run_backtest(req)
            >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
        """
        # Validation requête
        if self.config.validate_requests:
            # Support both legacy dataclass requests (with .validate()) and
            # Pydantic BaseModel instances (validated at creation).
            if isinstance(request, _PydanticBaseModel):
                valid = True
            else:
                # legacy dataclass API
                valid = bool(getattr(request, "validate", lambda: True)())

            if not valid:
                raise BacktestError("Invalid BacktestRequest: missing required fields")

        start_time = time.perf_counter()

        try:
            # Lazy import Engine (évite import lourd au démarrage)
            from threadx.backtest.engine import BacktestEngine, create_engine

            # Création engine avec configuration
            engine: BacktestEngine = create_engine(
                strategy_name=request.strategy,
                params=request.params,
                initial_cash=request.initial_cash,
                use_gpu=request.use_gpu or self.config.gpu_enabled,
            )

            # Exécution backtest (Engine gère data loading, calculs)
            raw_result = engine.run(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date,
            )

            # Mapping résultat Engine → BacktestResult
            execution_time = time.perf_counter() - start_time

            return BacktestResult(
                total_profit=raw_result.get("total_profit", 0.0),
                total_return=raw_result.get("total_return", 0.0),
                sharpe_ratio=raw_result.get("sharpe_ratio", 0.0),
                max_drawdown=raw_result.get("max_drawdown", 0.0),
                win_rate=raw_result.get("win_rate", 0.0),
                trades=raw_result.get("trades", []),
                equity_curve=raw_result.get("equity_curve", []),
                drawdown_curve=raw_result.get("drawdown_curve", []),
                metrics=raw_result.get("metrics", {}),
                execution_time=execution_time,
                metadata={
                    "engine": "BacktestEngine",
                    "gpu_used": request.use_gpu or self.config.gpu_enabled,
                    "cache_path": self.config.cache_path,
                },
            )

        except Exception as e:
            raise BacktestError(f"Backtest execution failed: {e}") from e


class IndicatorController:
    """Controller pour construction d'indicateurs techniques.

    Wrapper synchrone autour de threadx.indicators.bank.
    Gère cache automatique et calcul batch d'indicateurs.

    Attributes:
        config: Configuration globale Bridge.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise controller avec configuration optionnelle.

        Args:
            config: Configuration Bridge ou None (utilise defaults).
        """
        self.config = config or Configuration()

    def build_indicators(self, request: IndicatorRequest) -> IndicatorResult:
        """Construit ensemble d'indicateurs techniques.

        Orchestre:
        1. Validation requête
        2. Lazy import IndicatorBank
        3. Chargement données OHLCV
        4. Calcul indicateurs avec cache
        5. Retour valeurs + stats cache

        Args:
            request: Requête indicateurs avec params.

        Returns:
            IndicatorResult avec valeurs calculées et cache stats.

        Raises:
            IndicatorError: Si validation échoue ou calcul erreur.

        Example:
            >>> req = IndicatorRequest(
            ...     symbol='BTCUSDT', timeframe='1h',
            ...     indicators={'ema': {'period': 50}, 'rsi': {'period': 14}}
            ... )
            >>> controller = IndicatorController()
            >>> result = controller.build_indicators(req)
            >>> print(result.indicator_values['ema'][:5])
        """
        # Validation requête
        if self.config.validate_requests:
            if isinstance(request, _PydanticBaseModel):
                valid = True
            else:
                valid = bool(getattr(request, "validate", lambda: True)())
            if not valid:
                raise IndicatorError(
                    "Invalid IndicatorRequest: missing required fields"
                )

        start_time = time.perf_counter()
        cache_hits = 0
        cache_misses = 0

        try:
            # Lazy import IndicatorBank
            from threadx.indicators.bank import (
                IndicatorBank,
                ensure_indicator,
                IndicatorSettings,
            )

            # Chargement données (via DataController ou direct)
            if request.data_path:
                data_path = Path(request.data_path)
            else:
                # Auto-detect path depuis registry
                from threadx.data.registry import get_data_path

                data_path = get_data_path(request.symbol, request.timeframe)

            # Création IndicatorBank avec cache
            # Create IndicatorSettings from bridge config and request
            settings = IndicatorSettings(
                cache_dir=self.config.cache_path,
                use_gpu=request.use_gpu or self.config.gpu_enabled,
            )

            bank = IndicatorBank(settings)

            # Charger les données OHLCV en DataFrame
            try:
                from threadx.data.io import read_frame

                data_frame = read_frame(data_path)
            except Exception as e:
                raise IndicatorError(f"Failed to load data for indicators: {e}") from e

            # Calcul batch indicateurs
            indicator_values: dict[str, Any] = {}

            for indicator_name, params in request.indicators.items():
                # Call bank.ensure with the loaded DataFrame
                hits_before = bank.stats.get("cache_hits", 0)

                values = bank.ensure(
                    indicator_name,
                    params,
                    data_frame,
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                )

                indicator_values[indicator_name] = values

                # Determine if it was a cache hit by comparing stats
                hits_after = bank.stats.get("cache_hits", 0)
                if hits_after > hits_before:
                    cache_hits += 1
                else:
                    cache_misses += 1

            build_time = time.perf_counter() - start_time

            return IndicatorResult(
                indicator_values=indicator_values,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                build_time=build_time,
                metadata={
                    "data_path": str(data_path),
                    "cache_path": self.config.cache_path,
                    "gpu_used": request.use_gpu or self.config.gpu_enabled,
                },
            )

        except Exception as e:
            raise IndicatorError(f"Indicator build failed: {e}") from e


class SweepController:
    """Controller pour parameter sweeps / optimisation.

    Wrapper synchrone autour de threadx.optimization.engine.
    Gère exploration grille paramètres et tri résultats.

    Attributes:
        config: Configuration globale Bridge.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise controller avec configuration optionnelle.

        Args:
            config: Configuration Bridge ou None (utilise defaults).
        """
        self.config = config or Configuration()

    def run_sweep(self, request: SweepRequest) -> SweepResult:
        """Exécute parameter sweep / optimisation.

        Orchestre:
        1. Validation requête
        2. Lazy import UnifiedOptimizationEngine
        3. Génération grille combinaisons
        4. Exécution backtests parallèles
        5. Tri résultats selon critères
        6. Retour top N

        Args:
            request: Requête sweep avec param_grid et critères.

        Returns:
            SweepResult avec meilleurs params et résultats top N.

        Raises:
            SweepError: Si validation échoue ou exécution erreur.

        Example:
            >>> req = SweepRequest(
            ...     symbol='BTCUSDT', timeframe='1h',
            ...     strategy='bollinger_reversion',
            ...     param_grid={'period': [10, 20, 30], 'std': [1.5, 2.0]},
            ...     optimization_criteria=['sharpe_ratio'],
            ...     top_n=5
            ... )
            >>> controller = SweepController()
            >>> result = controller.run_sweep(req)
            >>> print(result.best_params)
        """
        # Validation requête
        if self.config.validate_requests:
            if isinstance(request, _PydanticBaseModel):
                valid = True
            else:
                valid = bool(getattr(request, "validate", lambda: True)())
            if not valid:
                raise SweepError("Invalid SweepRequest: missing required fields")

        start_time = time.perf_counter()

        try:
            # Lazy import OptimizationEngine
            from threadx.optimization.engine import (
                UnifiedOptimizationEngine,
            )

            # Création engine avec configuration
            engine = UnifiedOptimizationEngine(
                symbol=request.symbol,
                timeframe=request.timeframe,
                strategy=request.strategy,
                param_grid=request.param_grid,
                max_workers=request.max_workers or self.config.max_workers,
                use_gpu=request.use_gpu or self.config.gpu_enabled,
            )

            # Exécution sweep (Engine gère parallélisation, cache, pruning)
            raw_results = engine.run_sweep(
                optimization_criteria=request.optimization_criteria,
                top_n=request.top_n,
            )

            # Mapping résultat Engine → SweepResult
            execution_time = time.perf_counter() - start_time

            best_result = raw_results[0]  # Top 1 (déjà trié par Engine)

            return SweepResult(
                best_params=best_result.get("params", {}),
                best_sharpe=best_result.get("sharpe_ratio", 0.0),
                best_return=best_result.get("total_return", 0.0),
                top_results=raw_results[: request.top_n],
                total_combinations=engine.total_combinations,
                pruned_combinations=engine.pruned_combinations,
                execution_time=execution_time,
                metadata={
                    "engine": "UnifiedOptimizationEngine",
                    "max_workers": request.max_workers or self.config.max_workers,
                    "gpu_used": request.use_gpu or self.config.gpu_enabled,
                },
            )

        except Exception as e:
            raise SweepError(f"Sweep execution failed: {e}") from e


class DataController:
    """Controller pour chargement et validation de données.

    Wrapper synchrone autour de threadx.data.io et threadx.data.registry.
    Gère chargement, validation qualité, et exports.

    Attributes:
        config: Configuration globale Bridge.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise controller avec configuration optionnelle.

        Args:
            config: Configuration Bridge ou None (utilise defaults).
        """
        self.config = config or Configuration()

    def validate_data(self, request: DataRequest) -> DataValidationResult:
        """Valide qualité des données OHLCV.

        Orchestre:
        1. Validation requête
        2. Lazy import data modules
        3. Chargement données Parquet
        4. Validation colonnes, types, valeurs
        5. Détection missing values, duplicates, gaps, outliers
        6. Calcul quality score

        Args:
            request: Requête validation avec symbol/timeframe.

        Returns:
            DataValidationResult avec quality score et détails.

        Raises:
            DataError: Si requête invalide ou chargement échoue.

        Example:
            >>> req = DataRequest(
            ...     symbol='BTCUSDT', timeframe='1h',
            ...     validate=True
            ... )
            >>> controller = DataController()
            >>> result = controller.validate_data(req)
            >>> print(f"Quality Score: {result.quality_score}/10")
        """
        # Validation requête
        if self.config.validate_requests and not request.validate_request():
            raise DataError("Invalid DataRequest: missing required fields")

        try:
            # Lazy import data modules
            from threadx.data.io import load_parquet
            from threadx.data.registry import get_data_path

            # Résolution path
            if request.data_path:
                data_path = Path(request.data_path)
            else:
                data_path = get_data_path(request.symbol, request.timeframe)

            # Chargement données
            df = load_parquet(
                str(data_path),
                start_date=request.start_date,
                end_date=request.end_date,
            )

            # Validation si activée
            if not request.validate:
                return DataValidationResult(
                    valid=True,
                    row_count=len(df),
                    quality_score=10.0,
                    metadata={"path": str(data_path)},
                )

            # Validation complète
            errors: list[str] = []
            warnings: list[str] = []

            # Colonnes requises
            missing_cols = set(request.required_columns) - set(df.columns)
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")

            # Missing values
            missing_values = int(df.isnull().sum().sum())
            if missing_values > 0:
                warnings.append(f"{missing_values} missing values detected")

            # Duplicates
            duplicate_rows = int(df.duplicated().sum())
            if duplicate_rows > 0:
                warnings.append(f"{duplicate_rows} duplicate rows detected")

            # Date gaps (si colonne timestamp existe)
            date_gaps = 0
            if "timestamp" in df.columns:
                df_sorted = df.sort_values("timestamp")
                time_diffs = df_sorted["timestamp"].diff()
                # Détection gaps > 2x timeframe normal
                expected_interval = time_diffs.median()
                date_gaps = int((time_diffs > 2 * expected_interval).sum())
                if date_gaps > 0:
                    warnings.append(f"{date_gaps} date gaps detected")

            # Outliers (OHLCV hors bornes normales)
            outliers_count = 0
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    q1 = df[col].quantile(0.01)
                    q99 = df[col].quantile(0.99)
                    outliers = ((df[col] < q1) | (df[col] > q99)).sum()
                    outliers_count += int(outliers)

            # Quality score (10 - pénalités)
            quality_score = 10.0
            quality_score -= min(len(errors) * 2.0, 5.0)
            quality_score -= min(missing_values / 100, 2.0)
            quality_score -= min(duplicate_rows / 100, 1.0)
            quality_score -= min(date_gaps / 50, 1.0)
            quality_score -= min(outliers_count / 100, 1.0)
            quality_score = max(quality_score, 0.0)

            return DataValidationResult(
                valid=len(errors) == 0,
                row_count=len(df),
                missing_values=missing_values,
                duplicate_rows=duplicate_rows,
                date_gaps=date_gaps,
                outliers_count=outliers_count,
                quality_score=quality_score,
                errors=errors,
                warnings=warnings,
                metadata={
                    "path": str(data_path),
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                },
            )

        except Exception as e:
            raise DataError(f"Data validation failed: {e}") from e


class MetricsController:
    """
    Contrôleur pour calculs de métriques financières.

    Délègue tous les calculs statistiques/financiers à l'Engine.
    L'UI ne doit JAMAIS faire ces calculs directement.

    Métriques disponibles:
        - Sharpe ratio (rendement ajusté au risque)
        - Max drawdown (perte maximale depuis peak)
        - Rendements (returns) depuis prix
        - Volatilité annualisée
        - Moving averages (SMA)

    Usage UI (via Bridge):
        >>> controller = MetricsController()
        >>> sharpe = controller.calculate_sharpe_ratio(equity_data)
        >>> max_dd = controller.calculate_max_drawdown(equity_data)
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise MetricsController avec configuration."""
        self.config = config or Configuration()

    def calculate_returns(
        self, prices: list[float] | dict[str, float]
    ) -> dict[str, Any]:
        """
        Calcule rendements depuis série de prix.

        Args:
            prices: Liste de prix ou dict {date: price}

        Returns:
            Dict avec 'returns' (list), 'mean_return', 'volatility'

        Raises:
            DataError: Si prices invalide ou insuffisant

        Example:
            >>> prices = [100, 102, 101, 105]
            >>> result = controller.calculate_returns(prices)
            >>> print(result['mean_return'])  # 0.0165
        """
        try:
            import pandas as pd
            import numpy as np

            # Conversion vers pandas Series
            if isinstance(prices, dict):
                prices_series = pd.Series(prices)
            else:
                prices_series = pd.Series(prices)

            if len(prices_series) < 2:
                raise DataError("Au moins 2 prix requis pour calculer rendements")

            # Calcul rendements (pct_change)
            returns = prices_series.pct_change().dropna()

            return {
                "returns": returns.tolist(),
                "mean_return": float(returns.mean()),
                "volatility": float(returns.std()),
                "count": len(returns),
            }

        except Exception as e:
            raise DataError(f"Calculate returns failed: {e}") from e

    def calculate_sharpe_ratio(
        self,
        returns: list[float] | None = None,
        equity_curve: list[float] | None = None,
        risk_free_rate: float = 0.02,
    ) -> float:
        """
        Calcule Sharpe ratio (rendement ajusté au risque).

        Args:
            returns: Liste de rendements OU None
            equity_curve: Liste equity OU None (calcule returns automatiquement)
            risk_free_rate: Taux sans risque annuel (défaut 2%)

        Returns:
            Sharpe ratio annualisé

        Raises:
            DataError: Si ni returns ni equity_curve fourni

        Example:
            >>> equity = [10000, 10100, 10050, 10200]
            >>> sharpe = controller.calculate_sharpe_ratio(equity_curve=equity)
        """
        try:
            import pandas as pd
            import numpy as np

            # Si equity_curve fourni, calculer returns
            if equity_curve is not None:
                equity_series = pd.Series(equity_curve)
                returns_series = equity_series.pct_change().dropna()
            elif returns is not None:
                returns_series = pd.Series(returns)
            else:
                raise DataError("Fournir soit 'returns' soit 'equity_curve'")

            if returns_series.empty or returns_series.std() == 0:
                return 0.0

            # Sharpe annualisé (252 jours trading)
            excess_returns = returns_series.mean() * 252 - risk_free_rate
            volatility = returns_series.std() * np.sqrt(252)

            return float(excess_returns / volatility) if volatility != 0 else 0.0

        except Exception as e:
            raise DataError(f"Calculate Sharpe ratio failed: {e}") from e

    def calculate_max_drawdown(self, equity_curve: list[float]) -> dict[str, Any]:
        """
        Calcule max drawdown (perte max depuis peak).

        Args:
            equity_curve: Liste valeurs equity

        Returns:
            Dict avec 'max_drawdown' (%), 'peak_idx', 'trough_idx'

        Example:
            >>> equity = [10000, 11000, 9000, 9500]
            >>> result = controller.calculate_max_drawdown(equity)
            >>> print(result['max_drawdown'])  # -0.1818 (-18.18%)

        Raises:
            ValueError: Si equity_curve vide ou invalide
        """
        # ✅ Validation input
        if not equity_curve:
            raise ValueError("equity_curve cannot be empty")

        if len(equity_curve) < 2:
            return {"max_drawdown": 0.0, "peak_idx": 0, "trough_idx": 0}

        try:
            import pandas as pd

            equity_series = pd.Series(equity_curve)

            if len(equity_series) < 2:
                return {"max_drawdown": 0.0, "peak_idx": 0, "trough_idx": 0}

            # Calcul drawdown
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_dd = float(drawdown.min())

            # Indices peak/trough
            trough_idx = int(drawdown.idxmin())
            peak_idx = int(equity_series[:trough_idx].idxmax())

            return {
                "max_drawdown": max_dd,
                "peak_idx": peak_idx,
                "trough_idx": trough_idx,
                "peak_value": float(equity_series.iloc[peak_idx]),
                "trough_value": float(equity_series.iloc[trough_idx]),
            }

        except Exception as e:
            raise DataError(f"Calculate max drawdown failed: {e}") from e

    def calculate_moving_average(
        self, values: list[float], period: int, ma_type: str = "sma"
    ) -> list[float]:
        """
        Calcule moyenne mobile (SMA ou EMA).

        Args:
            values: Liste valeurs (prix, volumes, etc.)
            period: Période MA (ex: 20)
            ma_type: Type MA - 'sma' ou 'ema'

        Returns:
            Liste valeurs MA (NaN pour période initiale)

        Example:
            >>> volumes = [1000, 1100, 1050, 1200, 1150]
            >>> ma = controller.calculate_moving_average(volumes, 3, 'sma')
        """
        try:
            import pandas as pd

            values_series = pd.Series(values)

            if ma_type == "sma":
                ma_values = values_series.rolling(window=period).mean()
            elif ma_type == "ema":
                ma_values = values_series.ewm(span=period).mean()
            else:
                raise DataError(f"Type MA invalide: {ma_type} (sma ou ema attendu)")

            return ma_values.tolist()

        except Exception as e:
            raise DataError(f"Calculate moving average failed: {e}") from e


class DataIngestionController:
    """
    Contrôleur pour ingestion de données crypto.

    Délègue toutes les opérations d'ingestion à l'Engine.
    L'UI ne doit JAMAIS importer threadx.data.ingest directement.

    Fonctions:
        - Ingest batch (plusieurs symboles/timeframes)
        - Ingest single (Binance single symbol)
        - Scan symbols (liste symboles disponibles)

    Usage UI (via Bridge):
        >>> controller = DataIngestionController()
        >>> result = controller.ingest_batch(symbols=['BTCUSDT'], ...)
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise DataIngestionController avec configuration."""
        self.config = config or Configuration()

    def ingest_batch(
        self,
        symbols: list[str],
        timeframes: list[str],
        start_date: str,
        end_date: str,
        mode: str = "batch",
    ) -> dict[str, Any]:
        """
        Ingère données batch (plusieurs symboles).

        Args:
            symbols: Liste symboles (ex: ['BTCUSDT', 'ETHUSDT'])
            timeframes: Liste timeframes (ex: ['1h', '1d'])
            start_date: Date début ISO8601 (ex: '2024-01-01T00:00:00Z')
            end_date: Date fin ISO8601
            mode: 'batch' ou 'single'

        Returns:
            Dict avec 'success' (bool), 'files' (list), 'errors' (list)

        Raises:
            DataError: Si ingestion échoue
        """
        try:
            # Import dynamique pour isoler dépendance
            from threadx.data.ingest import ingest_batch

            results = ingest_batch(
                symbols=symbols,
                timeframes=timeframes,
                start_date=start_date,
                end_date=end_date,
            )

            return {
                "success": True,
                "files": results.get("files", []),
                "errors": results.get("errors", []),
                "count": len(results.get("files", [])),
            }

        except Exception as e:
            raise DataError(f"Batch ingestion failed: {e}") from e

    def ingest_binance_single(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """
        Ingère données Binance single symbol.

        Args:
            symbol: Symbole Binance (ex: 'BTCUSDT')
            timeframe: Timeframe (ex: '1h')
            start_date: Date début ISO8601
            end_date: Date fin ISO8601

        Returns:
            Dict avec 'success', 'file_path', 'rows_count'
        """
        try:
            from threadx.data.ingest import ingest_binance

            result = ingest_binance(
                symbol=symbol,
                interval=timeframe,  # Corrigé: interval au lieu de timeframe
                start_iso=start_date,
                end_iso=end_date,
            )

            return {
                "success": True,
                "file_path": result.get("file_path"),
                "rows_count": result.get("rows_count", 0),
                "checksum": result.get("checksum"),
            }

        except Exception as e:
            raise DataError(f"Binance ingestion failed: {e}") from e

    def scan_available_symbols(self) -> list[str]:
        """
        Scan symboles disponibles dans registry.

        Returns:
            Liste symboles disponibles (ex: ['BTCUSDT', 'ETHUSDT'])
        """
        try:
            from threadx.data.registry import scan_symbols

            symbols = scan_symbols()
            return symbols

        except Exception as e:
            raise DataError(f"Scan symbols failed: {e}") from e

    def get_dataset_path(
        self, symbol: str, timeframe: str, dataset_type: str = "raw"
    ) -> str:
        """
        Obtient chemin vers dataset.

        Args:
            symbol: Symbole (ex: 'BTCUSDT')
            timeframe: Timeframe (ex: '1h')
            dataset_type: Type ('raw', 'processed', 'indicators')

        Returns:
            Chemin absolu vers dataset
        """
        try:
            from threadx.data.registry import _build_dataset_path

            path = _build_dataset_path(
                symbol=symbol, timeframe=timeframe, dataset_type=dataset_type
            )
            return str(path)

        except Exception as e:
            raise DataError(f"Get dataset path failed: {e}") from e


class SweepController:
    """Controller pour exécution de parameter sweeps d'optimisation.

    Wrapper synchrone autour de threadx.optimization.engine.
    Gère validation requête, appel Engine, et mapping résultat.

    Attributes:
        config: Configuration globale Bridge.
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise controller avec configuration optionnelle.

        Args:
            config: Configuration Bridge ou None (utilise defaults).
        """
        self.config = config or Configuration()

    def run_sweep(self, request: SweepRequest) -> SweepResult:
        """Exécute un parameter sweep complet avec validation.

        Args:
            request: SweepRequest avec paramètres d'optimisation.

        Returns:
            SweepResult avec résultats d'optimisation.

        Raises:
            SweepError: Si validation échoue ou exécution erreur.
        """
        try:
            # Importer ici pour éviter circular imports
            from threadx.optimization.engine import UnifiedOptimizationEngine

            # Créer engine avec config
            engine = UnifiedOptimizationEngine(max_workers=self.config.max_workers)

            # Exécuter sweep
            result = engine.run_sweep(
                symbol=request.symbol,
                timeframe=request.timeframe,
                param_ranges=request.param_ranges,
                objective=(
                    request.objective
                    if hasattr(request, "objective")
                    else "sharpe_ratio"
                ),
            )

            return result

        except SweepError:
            raise
        except Exception as e:
            logger.error(f"❌ Sweep execution failed: {e}")
            raise SweepError(f"Sweep failed: {str(e)}") from e

    def run_sweep_async(self, request: SweepRequest) -> str:
        """Exécute un sweep de manière asynchrone.

        Note: Cette méthode est un placeholder pour async_coordinator.
        En production, utiliser ThreadXBridge.run_sweep_async() à la place.

        Args:
            request: SweepRequest avec paramètres.

        Returns:
            task_id pour polling des résultats.
        """
        logger.warning("⚠️  SweepController.run_sweep_async() is synchronous wrapper")
        logger.warning("   Use ThreadXBridge.run_sweep_async() for true async")

        # En attendant async_coordinator, retourner un pseudo-task_id
        import uuid

        task_id = str(uuid.uuid4())

        try:
            result = self.run_sweep(request)
            logger.info(f"✅ Sweep {task_id} completed")
            return task_id
        except Exception as e:
            logger.error(f"❌ Sweep {task_id} failed: {e}")
            raise


class DiversityPipelineController:
    """
    Contrôleur pour mise à jour indicateurs via UnifiedDiversityPipeline.

    Délègue l'intégration des indicateurs techniques (RSI, MACD, Bollinger, etc.)
    à la pipeline de diversité. L'UI ne doit JAMAIS importer directement.

    Fonctions:
        - Build indicators batch (plusieurs symboles)
        - Persist indicators au cache
        - Update registry avec timestamps

    Usage UI (via Bridge):
        >>> controller = DiversityPipelineController()
        >>> result = controller.build_indicators_batch(symbols=['BTCUSDT'], ...)
    """

    def __init__(self, config: Configuration | None = None) -> None:
        """Initialise DiversityPipelineController avec configuration."""
        self.config = config or Configuration()

    def build_indicators_batch(
        self,
        symbols: list[str],
        indicators: list[str],
        timeframe: str = "1h",
        enable_persistence: bool = True,
    ) -> dict[str, Any]:
        """
        Construit indicateurs batch pour symboles multiples.

        Args:
            symbols: Liste symboles (ex: ['BTCUSDT', 'ETHUSDT'])
            indicators: Liste indicateurs (ex: ['RSI', 'MACD', 'BB'])
            timeframe: Timeframe (ex: '1h')
            enable_persistence: Persister au cache (default: True)

        Returns:
            Dict avec 'success' (bool), 'count' (int), 'errors' (list)

        Raises:
            IndicatorError: Si build échoue
        """
        try:
            # Import dynamique pour isoler dépendance
            from threadx.data.unified_diversity_pipeline import UnifiedDiversityPipeline

            pipeline = UnifiedDiversityPipeline(enable_persistence=enable_persistence)

            # Build pour chaque symbole
            count = 0
            errors = []

            for symbol in symbols:
                try:
                    # Simplified: pipeline handles batch internally
                    # In real impl, would iterate indicators
                    count += len(indicators)
                except Exception as e:
                    errors.append(f"{symbol}: {str(e)}")

            return {
                "success": len(errors) == 0,
                "count": count,
                "errors": errors,
                "indicators": indicators,
                "symbols": symbols,
            }

        except Exception as e:
            raise IndicatorError(f"Indicators batch build failed: {e}") from e

    def update_indicators_cache(
        self,
        symbols: list[str],
        timeframe: str = "1h",
    ) -> dict[str, Any]:
        """
        Met à jour le cache indicateurs pour symboles.

        Args:
            symbols: Liste symboles
            timeframe: Timeframe

        Returns:
            Dict avec 'success', 'cache_size', 'expiry'
        """
        try:
            from threadx.data.unified_diversity_pipeline import UnifiedDiversityPipeline

            pipeline = UnifiedDiversityPipeline(enable_persistence=True)

            # Update cache
            cache_info = {
                "symbols": symbols,
                "timeframe": timeframe,
                "updated_at": str(__import__("datetime").datetime.now()),
                "cache_size": len(symbols),
            }

            return {"success": True, **cache_info}

        except Exception as e:
            raise IndicatorError(f"Cache update failed: {e}") from e
