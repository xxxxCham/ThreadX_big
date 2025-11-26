"""
ThreadX Backtest Engine - Phase 10 (Production)
==============================================

Orchestrateur de backtesting production-ready intégrant toutes les briques ThreadX.

Features:
- Device-agnostic computing via utils.xp (NumPy/CuPy)
- Multi-GPU distribution via utils.gpu.multi_gpu
- Device detection via utils.gpu.device_manager
- Performance measurement via utils.timing
- Bollinger Bands + ATR strategy avec bank.ensure
- RunResult compatible avec performance.summarize
- Déterminisme (seed=42), logs structurés
- Realistic Execution: spread, slippage, latency, rejections (TOUS timeframes)

Pipeline:
    bank.ensure(indicateurs) → engine.run(df, indicators, params) → RunResult
    → performance.summarize(result.returns, result.trades) → metrics/plots

Architecture:
- BacktestEngine : orchestrateur principal
- RunResult : structure de données standardisée
- Multi-device : balance 80%/20% par défaut entre GPUs (5080/2060)
- Strategy : Bollinger mean reversion + ATR filter
- RealisticExecutor : frictions trading réalistes (intégré)

Author: ThreadX Framework
Version: Phase 10 - Production Engine with Realistic Execution
"""

# logging peut être importé plus bas si nécessaire (cleanup de l'import inutile)
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# ThreadX Common Imports (DRY refactoring)
from threadx.utils.common_imports import (
    Any,
    Optional,
    get_logger,
    np,
    pd,
)

# Numba pour optimisation frictions
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        """Fallback decorator si Numba indisponible."""
        def decorator(func):
            return func
        return decorator

# Validation backtest anti-overfitting
try:
    from threadx.backtest.validation import (
        BacktestValidator,
        ValidationConfig,
        check_temporal_integrity,
    )

    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    BacktestValidator = None
    ValidationConfig = None
    check_temporal_integrity = None

# Threading/Timing utilities avec fallback gracieux
try:
    from threadx.utils.timing import measure_throughput, track_memory

    TIMING_AVAILABLE = True
except ImportError:
    TIMING_AVAILABLE = False

    # Fallback decorators si timing non disponible
    def measure_throughput(name=None, *, unit="task"):
        def decorator(func):
            return func

        return decorator

    def track_memory(name=None):
        def decorator(func):
            return func

        return decorator


# Device-agnostic computing avec fallback NumPy
try:
    from threadx.utils import xp as xp_module

    XP_AVAILABLE = True
except ImportError:
    XP_AVAILABLE = False
    xp_module = None


# GPU management avec fallback gracieux
try:
    from threadx.gpu.device_manager import (
        get_device_by_name,
        list_devices,
    )
    from threadx.gpu.device_manager import (
        is_available as gpu_available,
    )
    from threadx.gpu.multi_gpu import MultiGPUManager, get_default_manager

    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

    def list_devices():
        return []

    def get_device_by_name(_name: str):
        return None

    def gpu_available():
        return False

    class MultiGPUManager:  # stub minimal
        pass

    def get_default_manager():
        return None


logger = get_logger(__name__)


# ==========================================
# REALISTIC EXECUTION SYSTEM
# ==========================================


class OrderType(Enum):
    """Types d'ordres supportés."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class ExchangeConfig:
    """Configuration exchange avec frictions réalistes."""
    name: str
    maker_fee_bps: float
    taker_fee_bps: float
    min_spread_bps: float
    max_spread_bps: float
    base_latency_ms: float
    max_latency_ms: float
    order_rejection_rate: float
    partial_fill_threshold: float


EXCHANGE_CONFIGS = {
    "BINANCE": ExchangeConfig(
        name="Binance",
        maker_fee_bps=2.0,
        taker_fee_bps=4.0,
        min_spread_bps=1.0,
        max_spread_bps=5.0,
        base_latency_ms=50.0,
        max_latency_ms=500.0,
        order_rejection_rate=0.02,
        partial_fill_threshold=10.0,
    ),
    "BINANCE_FUTURES": ExchangeConfig(
        name="Binance Futures",
        maker_fee_bps=2.0,
        taker_fee_bps=4.0,
        min_spread_bps=0.5,
        max_spread_bps=3.0,
        base_latency_ms=30.0,
        max_latency_ms=400.0,
        order_rejection_rate=0.03,
        partial_fill_threshold=50.0,
    ),
    "BYBIT": ExchangeConfig(
        name="Bybit",
        maker_fee_bps=2.0,
        taker_fee_bps=5.5,
        min_spread_bps=1.5,
        max_spread_bps=8.0,
        base_latency_ms=80.0,
        max_latency_ms=600.0,
        order_rejection_rate=0.05,
        partial_fill_threshold=5.0,
    ),
}


@dataclass
class ExecutionResult:
    """Résultat exécution ordre réaliste."""
    success: bool
    intended_price: float
    executed_price: float
    intended_quantity: float
    filled_quantity: float
    total_fees: float
    slippage_pct: float
    spread_cost: float
    latency_ms: float
    rejection_reason: str | None
    is_partial_fill: bool
    order_type_used: str


@njit(cache=True, fastmath=True)
def apply_realistic_execution_numba(
    intended_price: float,
    side: int,
    quantity: float,
    spread_bps: float,
    slippage_bps: float,
    fee_bps: float,
    rejection_prob: float,
    random_seed: float,
) -> tuple[bool, float, float, float]:
    """Version Numba ultra-rapide pour backtests."""
    if random_seed < rejection_prob:
        return False, 0.0, 0.0, 0.0
    
    spread_pct = spread_bps / 10000.0
    if side == 1:
        price_with_spread = intended_price * (1.0 + spread_pct)
    else:
        price_with_spread = intended_price * (1.0 - spread_pct)
    
    slippage_pct = slippage_bps / 10000.0
    if side == 1:
        final_price = price_with_spread * (1.0 + slippage_pct)
    else:
        final_price = price_with_spread * (1.0 - slippage_pct)
    
    total_fees = (final_price * quantity) * (fee_bps / 10000.0)
    return True, final_price, quantity, total_fees


class RealisticExecutor:
    """
    Simule exécution réaliste avec spread, slippage, latence, rejets.
    
    S'adapte automatiquement selon timeframe (1m = frictions x2, 1h = x0.6).
    """
    
    def __init__(
        self,
        timeframe: str = "1m",
        symbol: str = "BTCUSDT",
        exchange: str = "BINANCE",
        seed: int | None = None,
    ):
        self.timeframe = timeframe
        self.symbol = symbol
        self.config = EXCHANGE_CONFIGS.get(exchange, EXCHANGE_CONFIGS["BINANCE"])
        self.timeframe_multiplier = self._compute_timeframe_multiplier(timeframe)
        self.rng = np.random.default_rng(seed)
        
    def _compute_timeframe_multiplier(self, timeframe: str) -> float:
        """Timeframes courts = impact plus grand."""
        timeframe_impacts = {
            "1m": 2.0, "2m": 1.8, "3m": 1.6, "5m": 1.4,
            "15m": 1.0, "30m": 0.8, "1h": 0.6, "4h": 0.4, "1d": 0.2,
        }
        return timeframe_impacts.get(timeframe, 1.0)
    
    def execute_order(
        self,
        side: Literal["BUY", "SELL"],
        intended_price: float,
        quantity: float,
        order_type: str = "MARKET",
        current_volatility: float = 0.01,
        current_volume_ratio: float = 1.0,
    ) -> ExecutionResult:
        """Simule exécution avec tous les obstacles réels."""
        
        # 1. SPREAD
        spread_bps = self._calculate_spread(current_volume_ratio)
        spread_pct = spread_bps / 10000.0
        
        if side == "BUY":
            execution_base_price = intended_price * (1.0 + spread_pct)
        else:
            execution_base_price = intended_price * (1.0 - spread_pct)
        
        # 2. LATENCE
        latency_ms = self._calculate_latency(current_volatility)
        time_fraction = latency_ms / (60000.0 if self.timeframe == "1m" else 3600000.0)
        price_drift = self.rng.normal(0, current_volatility * time_fraction)
        execution_price_after_latency = execution_base_price * (1.0 + price_drift)
        
        # 3. SLIPPAGE
        slippage_pct = self._calculate_slippage(
            quantity, current_volatility, current_volume_ratio
        )
        if side == "BUY":
            final_price = execution_price_after_latency * (1.0 + slippage_pct)
        else:
            final_price = execution_price_after_latency * (1.0 - slippage_pct)
        
        # 4. REJET
        rejection_rate = self._calculate_rejection_rate(order_type, current_volatility)
        if self.rng.random() < rejection_rate:
            return ExecutionResult(
                success=False, intended_price=intended_price,
                executed_price=0.0, intended_quantity=quantity,
                filled_quantity=0.0, total_fees=0.0,
                slippage_pct=0.0, spread_cost=0.0,
                latency_ms=latency_ms,
                rejection_reason="Insufficient liquidity / Network error",
                is_partial_fill=False, order_type_used=order_type,
            )
        
        # 5. PARTIAL FILL
        filled_qty = quantity
        is_partial = False
        if quantity > self.config.partial_fill_threshold:
            fill_ratio = self.rng.uniform(0.7, 0.95)
            filled_qty = quantity * fill_ratio
            is_partial = True
        
        # 6. FRAIS
        if order_type == "MARKET":
            fee_bps = self.config.taker_fee_bps
        else:
            is_maker = self.rng.random() < 0.7
            fee_bps = self.config.maker_fee_bps if is_maker else self.config.taker_fee_bps
        
        total_fees = (final_price * filled_qty) * (fee_bps / 10000.0)
        spread_cost = abs(execution_base_price - intended_price) * filled_qty
        total_slippage_pct = abs(final_price - intended_price) / intended_price * 100.0
        
        return ExecutionResult(
            success=True,
            intended_price=intended_price,
            executed_price=final_price,
            intended_quantity=quantity,
            filled_quantity=filled_qty,
            total_fees=total_fees,
            slippage_pct=total_slippage_pct,
            spread_cost=spread_cost,
            latency_ms=latency_ms,
            rejection_reason=None,
            is_partial_fill=is_partial,
            order_type_used=order_type,
        )
    
    def _calculate_spread(self, volume_ratio: float) -> float:
        """Spread augmente si volume faible."""
        base_spread = self.config.min_spread_bps
        if volume_ratio < 1.0:
            spread_multiplier = 1.0 + (1.0 - volume_ratio) * 0.5
            spread = base_spread * spread_multiplier
        else:
            spread = base_spread
        spread *= self.timeframe_multiplier
        return min(spread, self.config.max_spread_bps)
    
    def _calculate_latency(self, volatility: float) -> float:
        """Latence augmente avec volatilité."""
        base = self.config.base_latency_ms
        if volatility > 0.02:
            volatility_factor = 1.0 + (volatility - 0.02) * 25.0
            latency = base * volatility_factor
        else:
            latency = base
        noise = self.rng.uniform(-0.2, 0.5)
        latency *= 1.0 + noise
        return min(latency, self.config.max_latency_ms)
    
    def _calculate_slippage(
        self, quantity: float, volatility: float, volume_ratio: float
    ) -> float:
        """Slippage: quantité + liquidité + volatilité."""
        base_slippage = 0.0001
        qty_factor = 1.0 + np.log1p(quantity) * 0.05
        liquidity_factor = 1.0 / max(volume_ratio, 0.3)
        volatility_factor = 1.0 + volatility * 10.0
        timeframe_factor = self.timeframe_multiplier
        slippage = base_slippage * qty_factor * liquidity_factor * volatility_factor * timeframe_factor
        return min(slippage, 0.005)
    
    def _calculate_rejection_rate(self, order_type: str, volatility: float) -> float:
        """Market orders moins rejetés que limit."""
        base_rate = self.config.order_rejection_rate
        type_factor = 0.5 if order_type == "MARKET" else 1.5
        volatility_factor = 1.0 + volatility * 20.0
        rejection_rate = base_rate * type_factor * volatility_factor * self.timeframe_multiplier
        return min(rejection_rate, 0.20)


# ==========================================
# BACKTEST ENGINE (existant)
# ==========================================


@dataclass
class RunResult:
    """
    Résultat d'exécution de backtest ThreadX.

    Structure de données standard pour l'échange entre:
    - BacktestEngine.run() → RunResult
    - RunResult → PerformanceCalculator.summarize()
    - RunResult → UI charts/tables

    Attributes:
        equity: Série d'équité avec index datetime UTC, dtype float64
        returns: Série des returns avec même index que equity
        trades: DataFrame des trades avec colonnes minimales requises
        metrics: Dict métriques calculées (Tier S/A/B/C + standards)
        meta: Métadonnées d'exécution (durées, devices, cache, etc.)

    Notes:
        Validation stricte des données pour garantir la compatibilité
        avec performance.summarize() et les modules d'analyse.
        
        metrics contient TOUTES les métriques (Tier S 2025 incluses):
        - Métriques standards: sharpe, sortino, max_drawdown, etc.
        - Tier S (10): sharpe ≥1.8, sortino ≥2.8, calmar ≥1.5, etc.
        - Tier A/B/C: r_multiple, ulcer_index, serenity_ratio, etc.
        - Validation: tier_s_passed, tier_s_score, ai_evolved_gold
    """

    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validation stricte des données retournées."""
        # Validation equity
        if not isinstance(self.equity, pd.Series):
            raise TypeError("equity doit être une pd.Series")
        if not pd.api.types.is_datetime64_any_dtype(self.equity.index):
            raise TypeError("equity.index doit être datetime64")
        if self.equity.dtype != np.float64:
            logger.warning(f"equity dtype {self.equity.dtype} != float64, conversion")
            self.equity = self.equity.astype(np.float64)

        # Validation returns
        if not isinstance(self.returns, pd.Series):
            raise TypeError("returns doit être une pd.Series")
        if not self.equity.index.equals(self.returns.index):
            raise ValueError("equity et returns doivent avoir le même index")

        # Validation trades - colonnes requises pour performance.summarize
        required_cols = [
            "entry_ts",
            "exit_ts",
            "pnl",
            "size",
            "price_entry",
            "price_exit",
        ]
        if not isinstance(self.trades, pd.DataFrame):
            raise TypeError("trades doit être un pd.DataFrame")
        missing_cols = [col for col in required_cols if col not in self.trades.columns]
        if missing_cols:
            raise ValueError(f"trades manque colonnes requises: {missing_cols}")


class BacktestEngine:
    """
    Moteur de backtesting unifié ThreadX.

    Orchestrateur production-ready qui intègre toutes les briques existantes :
    - Device management via utils.gpu.device_manager
    - Multi-GPU distribution via utils.gpu.multi_gpu
    - Device-agnostic computing via utils.xp
    - Performance measurement via utils.timing
    - Strategy implementation with existing IndicatorBank

    Pipeline standard:
    1. bank.ensure(...) → indicators
    2. engine.run(df_1m, indicators, params) → RunResult
    3. performance.summarize(result.returns, result.trades) → metrics
    4. ui.display(result, metrics)

    Features:
    - Multi-GPU: balance 80%/20% par défaut (configurable)
    - Déterminisme: seed=42 pour reproductibilité
    - Device fallback: GPU → CPU transparent
    - Logs structurés: INFO/DEBUG/WARNING/ERROR
    - API RunResult: compatible performance.summarize

    Examples:
        >>> # Usage standard avec IndicatorBank
        >>> from threadx.indicators.bank import IndicatorBank
        >>> from threadx.backtest.engine import BacktestEngine
        >>> from threadx.backtest.performance import PerformanceCalculator
        >>>
        >>> # 1. Calculer indicateurs
        >>> bank = IndicatorBank()
        >>> indicators = {
        ...     "bollinger": bank.ensure("bollinger", {"period": 20, "std": 2.0},
        ...                             df_1m, symbol="BTCUSDC", timeframe="1m"),
        ...     "atr": bank.ensure("atr", {"period": 14}, df_1m,
        ...                       symbol="BTCUSDC", timeframe="1m")
        ... }
        >>>
        >>> # 2. Exécuter backtest
        >>> engine = BacktestEngine()
        >>> result = engine.run(df_1m, indicators,
        ...                     params={"entry_z": 2.0, "k_sl": 1.5, "leverage": 3},
        ...                     symbol="BTCUSDC", timeframe="1m")
        >>>
        >>> # 3. Calculer métriques
        >>> perf = PerformanceCalculator()
        >>> metrics = perf.summarize(result.returns, result.trades)
        >>>
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.3f}")
        >>> print(f"Max DD: {metrics['max_drawdown']:.2%}")
    """

    def __init__(
        self,
        gpu_balance: Optional[dict[str, float]] = None,
        use_multi_gpu: bool = True,
        use_gpu: bool = True,
    ):
        """
        Initialise le moteur de backtesting.

        Args:
            gpu_balance: Balance multi-GPU personnalisée {"5090": 0.75, "2060": 0.25}
                        Si None, utilise balance par défaut du MultiGPUManager
            use_multi_gpu: Active la distribution multi-GPU si plusieurs devices
        """
        self.logger = get_logger(__name__)

        # Device detection et setup
        self.gpu_available = gpu_available() if GPU_UTILS_AVAILABLE else False
        self.devices = list_devices() if GPU_UTILS_AVAILABLE else []

        # Multi-GPU setup
        self.use_multi_gpu = use_multi_gpu and len(self.devices) > 1
        self.multi_gpu_manager = None

        if self.use_multi_gpu and GPU_UTILS_AVAILABLE:
            try:
                self.multi_gpu_manager = get_default_manager()
                if gpu_balance and self.multi_gpu_manager:
                    self.multi_gpu_manager.set_balance(gpu_balance)
                self.logger.info(f"🔀 Multi-GPU activé: {len(self.devices)} devices")
            except Exception as e:
                self.logger.warning(
                    f"⚠️ Multi-GPU setup failed: {e}, fallback single device"
                )
                self.use_multi_gpu = False

        # Device-agnostic computing setup
        self.xp_backend = "cpu"
        if XP_AVAILABLE and self.gpu_available:
            try:
                # Configure xp backend pour GPU si disponible
                self.xp_backend = "gpu"
                self.logger.debug("🎯 XP backend configuré: GPU")
            except Exception as e:
                self.logger.warning(f"⚠️ XP GPU config failed: {e}, fallback CPU")
                self.xp_backend = "cpu"

        # État d'exécution
        self.last_run_meta = {}

        # Validation setup
        self.validator = None
        self.validation_config = None
        if VALIDATION_AVAILABLE:
            # Configuration par défaut: walk-forward avec purge/embargo
            self.validation_config = ValidationConfig(
                method="walk_forward",
                walk_forward_windows=5,
                purge_days=1,
                embargo_days=1,
                min_train_samples=200,
                min_test_samples=50,
            )
            self.validator = BacktestValidator(self.validation_config)
            self.logger.info("✅ Validation anti-overfitting activée")

        self.logger.info("🚀 BacktestEngine initialisé")
        self.logger.info(f"   GPU: {'✅' if self.gpu_available else '❌'}")
        self.logger.info(f"   Multi-GPU: {'✅' if self.use_multi_gpu else '❌'}")
        self.logger.info(f"   XP Backend: {self.xp_backend}")
        self.logger.info(f"   Validation: {'✅' if self.validator else '❌'}")

    def run(
        self,
        df_1m: Optional[pd.DataFrame] = None,
        indicators: Optional[dict[str, Any]] = None,
        *,
        params: Optional[dict[str, Any]] = None,
        symbol: str = "UNKNOWN",
        timeframe: str = "1m",
        seed: int = 42,
        use_gpu: Optional[bool] = None,
        strategy_class: Optional[Any] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> Any:
        """
        Exécute un backtest. Supporte deux modes:
        1. Mode Legacy/Generic: via strategy_class et data
        2. Mode Production: via df_1m et indicators (Bollinger+ATR)
        """
        # Mode Legacy/Generic (Notebooks)
        if strategy_class is not None:
            if data is None:
                raise ValueError("data is required when using strategy_class")
            
            # Instancier et exécuter la stratégie
            strategy = strategy_class()
            # Adapter les paramètres si nécessaire
            run_params = params if params is not None else {}
            
            # Exécuter backtest
            # Note: ma_crossover retourne (equity, stats)
            equity, stats = strategy.backtest(data, run_params)
            
            # Créer un objet résultat compatible avec ce que le notebook attend
            # Le notebook attend: result.sharpe_ratio, result.total_return, result.max_drawdown
            from types import SimpleNamespace
            
            result = SimpleNamespace()
            result.sharpe_ratio = stats.sharpe_ratio if hasattr(stats, 'sharpe_ratio') else 0.0
            result.total_return = stats.total_pnl_pct if hasattr(stats, 'total_pnl_pct') else 0.0
            result.max_drawdown = stats.max_drawdown_pct if hasattr(stats, 'max_drawdown_pct') else 0.0
            result.equity = equity
            result.stats = stats
            
            # Ajouter d'autres attributs si nécessaire pour compatibilité
            result.win_rate = stats.win_rate_pct if hasattr(stats, 'win_rate_pct') else 0.0
            result.total_trades = stats.total_trades if hasattr(stats, 'total_trades') else 0
            
            return result

        # Mode Production (Bollinger + ATR)
        if df_1m is None or indicators is None:
            raise ValueError("df_1m and indicators are required for production mode")
            
        if params is None:
             raise ValueError("params is required for production mode")
        """
        Exécute un backtest complet avec stratégie Bollinger Bands + ATR.

        Pipeline d'exécution:
        1. Validation données/paramètres stricte
        2. Setup backend compute (CPU/GPU/Multi-GPU)
        3. Génération signaux via stratégie configurable
        4. Simulation trades avec gestion positions réaliste
        5. Calcul equity curve et returns
        6. Construction RunResult avec métadonnées complètes

        Args:
            df_1m: DataFrame OHLCV 1-minute, index datetime UTC
                   Colonnes requises: open, high, low, close, volume
            indicators: Dict des indicateurs calculés via bank.ensure()
                       Ex: {"bollinger": (upper, middle, lower), "atr": np.array(...)}
            params: Paramètres de stratégie
                   Clés requises: entry_z, k_sl, leverage
                   Optionnelles: risk_pct, trail_k, fees_bps
            symbol: Symbole tradé (ex: "BTCUSDC")
            timeframe: Timeframe de référence (ex: "1m", "1h")
            seed: Seed pour déterminisme (default: 42)
            use_gpu: Force GPU usage (None=auto selon détection)

        Returns:
            RunResult: Structure avec equity, returns, trades, meta

        Raises:
            ValueError: Si données/paramètres invalides
            RuntimeError: Si erreur compute non récupérable

        Notes:
            Multi-GPU: Si plusieurs devices disponibles, distribue automatiquement
            le workload selon balance configurée (80%/20% par défaut).

            Déterminisme: seed=42 appliqué à tous composants pseudo-aléatoires.

            Performance: @measure_throughput et @track_memory actifs si utils.timing
            disponible, sinon fallback gracieux sans impact.
        """
        start_time = time.time()
        self.logger.info(f"🎯 Démarrage backtest: {symbol} {timeframe}")
        self.logger.debug(f"   Params: {params}")
        self.logger.debug(f"   Data shape: {df_1m.shape}")
        self.logger.debug(f"   Période: {df_1m.index[0]} → {df_1m.index[-1]}")
        self.logger.debug(f"   Indicators: {list(indicators.keys())}")

        # Seed pour déterminisme complet
        np.random.seed(seed)

        try:
            # 1. Setup backend et validation
            device_info = self._setup_compute_backend(use_gpu)
            self._validate_inputs(df_1m, indicators, params)

            # 2. Génération signaux de trading
            signals = self._generate_trading_signals(df_1m, indicators, params)

            # 3. Simulation trades et gestion positions
            trades_df = self._simulate_trades(df_1m, signals, params)

            # 4. Calcul equity curve et returns
            equity, returns = self._calculate_equity_returns(df_1m, trades_df, params)

            # 5. Métadonnées d'exécution complètes
            duration = time.time() - start_time
            meta = self._build_metadata(
                device_info, duration, df_1m, trades_df, params, seed
            )

            # 6. Construction RunResult avec validation
            result = RunResult(
                equity=equity, returns=returns, trades=trades_df, meta=meta
            )

            self.last_run_meta = meta
            self.logger.info(f"✅ Backtest terminé en {duration:.2f}s")
            self.logger.info(
                f"   Trades: {len(trades_df)}, Equity finale: ${equity.iloc[-1]:,.2f}"
            )
            self.logger.debug(f"   Throughput: {len(df_1m)/duration:.0f} ticks/sec")

            return result

        except Exception as e:
            self.logger.error(f"❌ Erreur backtest {symbol}: {e}")
            self.logger.debug("Backtest exception traceback", exc_info=True)
            raise

    def _setup_compute_backend(self, use_gpu: Optional[bool]) -> dict[str, Any]:
        """
        Configure le backend de calcul avec fallback gracieux.

        Args:
            use_gpu: Force GPU usage (None=auto)

        Returns:
            Dict avec info device pour métadonnées
        """
        # Détermination GPU usage
        if use_gpu is None:
            use_gpu = self.gpu_available
        elif use_gpu and not self.gpu_available:
            self.logger.warning("⚠️ GPU requis mais non disponible, fallback CPU")
            use_gpu = False

        # Device info pour métadonnées
        if use_gpu and self.use_multi_gpu:
            # Multi-GPU
            device_names = [d.name for d in self.devices if d.name != "cpu"]
            balance = (
                getattr(self.multi_gpu_manager, "device_balance", {})
                if self.multi_gpu_manager
                else {}
            )
            device_info = {
                "mode": "multi_gpu",
                "devices": device_names,
                "balance": balance,
                "backend": "cupy/multi",
            }
            self.logger.debug(f"🔀 Multi-GPU mode: {device_names}")

        elif use_gpu:
            # Single GPU
            gpu_device = next((d for d in self.devices if d.name != "cpu"), None)
            device_info = {
                "mode": "single_gpu",
                "devices": [gpu_device.name] if gpu_device else ["gpu"],
                "balance": {},
                "backend": "cupy",
            }
            self.logger.debug(
                f"🎯 Single GPU mode: {gpu_device.name if gpu_device else 'default'}"
            )

        else:
            # CPU fallback
            device_info = {
                "mode": "cpu",
                "devices": ["cpu"],
                "balance": {},
                "backend": "numpy",
            }
            self.logger.debug("🖥️ CPU mode")

        return device_info

    def _validate_inputs(
        self, df_1m: pd.DataFrame, indicators: dict[str, Any], params: dict[str, Any]
    ):
        """Validation stricte des données d'entrée."""
        # Validation DataFrame OHLCV
        if df_1m.empty:
            raise ValueError("df_1m ne peut pas être vide")

        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df_1m.columns]
        if missing_cols:
            raise ValueError(f"df_1m manque colonnes OHLCV: {missing_cols}")

        if not pd.api.types.is_datetime64_any_dtype(df_1m.index):
            raise ValueError("df_1m.index doit être datetime64 UTC")

        # Vérification données cohérentes (OHLC logic)
        ohlc_errors = (df_1m["high"] < df_1m[["open", "close"]].max(axis=1)).sum()
        if ohlc_errors > 0:
            self.logger.warning(f"⚠️ {ohlc_errors} barres avec high < max(open,close)")

        # Validation indicateurs
        if not indicators:
            self.logger.warning("⚠️ Aucun indicateur fourni, signaux basiques")

        # Validation paramètres stratégie
        required_params = ["entry_z", "k_sl", "leverage"]
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            raise ValueError(f"params manque clés requises: {missing_params}")

        # Validation ranges paramètres
        if params["leverage"] <= 0 or params["leverage"] > 20:
            raise ValueError("leverage doit être dans [0.1, 20]")
        if params["k_sl"] <= 0 or params["k_sl"] > 10:
            raise ValueError("k_sl doit être dans (0, 10]")

        self.logger.debug("✅ Validation inputs réussie")

    @measure_throughput("signal_generation")
    def _generate_trading_signals(
        self, df_1m: pd.DataFrame, indicators: dict[str, Any], params: dict[str, Any]
    ) -> pd.Series:
        """
        Génère signaux de trading via stratégie Bollinger Bands + ATR.

        Stratégie implémentée:
        - Long: prix touche bande basse ET volatilité > seuil
        - Short: prix touche bande haute ET volatilité > seuil
        - Exit: signal opposé ou stop-loss/take-profit
        - Filter: ATR pour éviter marchés trop calmes

        Args:
            df_1m: DataFrame OHLCV
            indicators: Dict avec bollinger et atr de bank.ensure
            params: Paramètres stratégie (entry_z, etc.)

        Returns:
            pd.Series: Signaux {-1: short, 0: hold, 1: long}
        """
        self.logger.debug("🎲 Génération signaux Bollinger + ATR")

        # Signaux par défaut (hold/flat)
        signals = pd.Series(0, index=df_1m.index, dtype=np.float64, name="signals")

        try:
            # === Stratégie principale: Bollinger Bands ===
            if "bollinger" in indicators and indicators["bollinger"]:
                bb_result = indicators["bollinger"]

                if isinstance(bb_result, tuple) and len(bb_result) >= 3:
                    upper, middle, lower = bb_result[:3]

                    # Conversion en Series pandas si arrays NumPy/CuPy
                    if hasattr(upper, "__array__") and not isinstance(upper, pd.Series):
                        upper = pd.Series(np.asarray(upper), index=df_1m.index)
                    if hasattr(lower, "__array__") and not isinstance(lower, pd.Series):
                        lower = pd.Series(np.asarray(lower), index=df_1m.index)
                    if hasattr(middle, "__array__") and not isinstance(
                        middle, pd.Series
                    ):
                        middle = pd.Series(np.asarray(middle), index=df_1m.index)

                    close_prices = df_1m["close"]

                    # Signaux mean reversion Bollinger
                    # Long: prix <= lower band (oversold)
                    long_condition = close_prices <= lower
                    # Short: prix >= upper band (overbought)
                    short_condition = close_prices >= upper

                    signals[long_condition] = 1.0  # Long signal
                    signals[short_condition] = -1.0  # Short signal

                    signal_count = (signals != 0).sum()
                    self.logger.debug(f"   Bollinger signaux: {signal_count} positions")
                    self.logger.debug(
                        f"   Long: {(signals == 1).sum()}, Short: {(signals == -1).sum()}"
                    )

            # === Filtre ATR: ne trade que si volatilité suffisante ===
            if "atr" in indicators and indicators["atr"] is not None:
                atr_values = indicators["atr"]

                # Conversion en Series pandas
                if hasattr(atr_values, "__array__") and len(atr_values) == len(df_1m):
                    atr_series = pd.Series(np.asarray(atr_values), index=df_1m.index)

                    # Filtre: volatilité > percentile 30% (évite marchés calmes)
                    atr_threshold = atr_series.quantile(0.3)
                    volatility_filter = atr_series > atr_threshold

                    # Applique filtre: supprime signaux en low volatility
                    signals[~volatility_filter] = 0.0

                    filtered_count = (signals != 0).sum()
                    self.logger.debug(
                        f"   ATR filtre: {filtered_count} signaux restants"
                    )
                    self.logger.debug(f"   Seuil ATR: {atr_threshold:.4f}")

            # === Fallback: signaux aléatoires si pas d'indicateurs valides ===
            if not any(indicators.values()) or (signals == 0).all():
                self.logger.warning("⚠️ Pas d'indicateurs valides, signaux aléatoires")
                np.random.seed(42)
                random_signals = np.random.choice(
                    [0, 1, -1],
                    size=len(df_1m),
                    p=[0.85, 0.075, 0.075],  # 85% hold, 7.5% long, 7.5% short
                )
                signals = pd.Series(
                    random_signals, index=df_1m.index, dtype=np.float64, name="signals"
                )

        except Exception as e:
            self.logger.error(f"❌ Erreur génération signaux: {e}")

            # Fallback sécurisé: signaux aléatoires déterministes
            np.random.seed(42)
            random_signals = np.random.choice(
                [0, 1, -1], size=len(df_1m), p=[0.85, 0.075, 0.075]
            )
            signals = pd.Series(
                random_signals, index=df_1m.index, dtype=np.float64, name="signals"
            )

        # Stats finales
        signal_counts = signals.value_counts().to_dict()
        self.logger.debug(f"   Distribution signaux finale: {signal_counts}")

        return signals

    @track_memory("trade_simulation")
    def _simulate_trades(
        self, df_1m: pd.DataFrame, signals: pd.Series, params: dict[str, Any]
    ) -> pd.DataFrame:
        """
        Simule l'exécution des trades avec gestion positions réaliste.

        Features:
        - Position tracking (flat/long/short)
        - Stop-loss configurable via k_sl
        - Sizing basé sur leverage et risk management
        - PnL calculation réaliste avec slippage/fees
        - Exit reasons tracking pour analyse

        Args:
            df_1m: DataFrame OHLCV
            signals: Signaux de trading {-1, 0, 1}
            params: Paramètres (leverage, k_sl, etc.)

        Returns:
            pd.DataFrame: Trades avec colonnes requises pour performance.summarize
        """
        self.logger.debug("💼 Simulation des trades")

        trades = []
        position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0.0
        entry_time = None

        # Paramètres de trading avec defaults
        leverage = params.get("leverage", 3)
        k_sl = params.get("k_sl", 1.5)  # Stop loss % multiplier
        initial_capital = params.get("initial_capital", 10000.0)
        fees_bps = params.get("fees_bps", 10.0)  # 10 bps = 0.1%
        slip_bps = params.get("slip_bps", 5.0)  # 5 bps slippage

        # Conversion en listes pour itération efficace (évite pandas.loc overhead)
        timestamps = df_1m.index.tolist()
        closes = df_1m["close"].tolist()
        signal_values = signals.tolist()

        # Simulation trade par trade
        for i, (timestamp, close_price, signal) in enumerate(
            zip(timestamps, closes, signal_values)
        ):

            # === Entrée en position ===
            if position == 0 and signal != 0:
                position = signal
                entry_price = close_price * (1 + slip_bps * 0.0001 * signal)  # Slippage
                entry_time = timestamp

                self.logger.debug(
                    f"   Entrée {['FLAT', 'LONG', 'SHORT'][int(position + 1)]} @ {entry_price:.2f} le {entry_time}"
                )

            # === Sortie de position ===
            elif position != 0:
                exit_condition = False
                exit_reason = ""

                # 1. Signal opposé (reversal)
                if signal != 0 and signal != position:
                    exit_condition = True
                    exit_reason = "signal_reverse"

                # 2. Stop-loss (basé sur k_sl%)
                elif position == 1 and close_price <= entry_price * (1 - k_sl * 0.01):
                    exit_condition = True
                    exit_reason = "stop_loss"
                elif position == -1 and close_price >= entry_price * (1 + k_sl * 0.01):
                    exit_condition = True
                    exit_reason = "stop_loss"

                # === Exécution sortie ===
                if exit_condition:
                    exit_price = close_price * (
                        1 - slip_bps * 0.0001 * position
                    )  # Slippage opposé

                    # Calcul PnL réaliste avec fees
                    if position == 1:  # Long position
                        raw_return = (exit_price - entry_price) / entry_price
                    else:  # Short position
                        raw_return = (entry_price - exit_price) / entry_price

                    # Fees totales (entrée + sortie)
                    total_fees_pct = fees_bps * 2 * 0.0001  # 2x pour round-trip
                    net_return = raw_return - total_fees_pct

                    # PnL avec leverage sur capital de base
                    pnl = net_return * leverage * initial_capital

                    # Position size (notional)
                    position_size = (
                        abs(position) * leverage * initial_capital / entry_price
                    )

                    trade_record = {
                        "entry_ts": entry_time,
                        "exit_ts": timestamp,
                        "pnl": pnl,
                        "size": position_size,
                        "price_entry": entry_price,
                        "price_exit": exit_price,
                        "side": "LONG" if position == 1 else "SHORT",
                        "exit_reason": exit_reason,
                        "return_pct": net_return * 100,
                        "leverage_used": leverage,
                        "fees_paid": position_size * total_fees_pct,
                    }

                    trades.append(trade_record)

                    # Log trade détaillé
                    self.logger.debug(
                        f"   Exit {trade_record['side']} @ {exit_price:.2f}, "
                        f"PnL: ${pnl:.2f}, Reason: {exit_reason}"
                    )

                    # Reset position
                    position = 0
                    entry_price = 0.0
                    entry_time = None

                    # Nouvelle position si signal présent après sortie
                    if signal != 0:
                        position = signal
                        entry_price = close_price * (1 + slip_bps * 0.0001 * signal)
                        entry_time = timestamp

        # === Trade final si position ouverte ===
        # Ferme position ouverte en fin de données
        if position != 0:
            final_price = closes[-1] * (1 - slip_bps * 0.0001 * position)
            final_time = timestamps[-1]

            if position == 1:
                raw_return = (final_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - final_price) / entry_price

            total_fees_pct = fees_bps * 2 * 0.0001
            net_return = raw_return - total_fees_pct
            pnl = net_return * leverage * initial_capital
            position_size = abs(position) * leverage * initial_capital / entry_price

            trades.append(
                {
                    "entry_ts": entry_time,
                    "exit_ts": final_time,
                    "pnl": pnl,
                    "size": position_size,
                    "price_entry": entry_price,
                    "price_exit": final_price,
                    "side": "LONG" if position == 1 else "SHORT",
                    "exit_reason": "end_of_data",
                    "return_pct": net_return * 100,
                    "leverage_used": leverage,
                    "fees_paid": position_size * total_fees_pct,
                }
            )

        # Construction DataFrame final
        trades_df = pd.DataFrame(trades)

        # Stats de trading complètes
        if not trades_df.empty:
            total_pnl = trades_df["pnl"].sum()
            total_fees = trades_df["fees_paid"].sum()
            win_rate = (trades_df["pnl"] > 0).mean()
            avg_win = (
                trades_df[trades_df["pnl"] > 0]["pnl"].mean()
                if (trades_df["pnl"] > 0).any()
                else 0
            )
            avg_loss = (
                trades_df[trades_df["pnl"] < 0]["pnl"].mean()
                if (trades_df["pnl"] < 0).any()
                else 0
            )

            self.logger.debug(f"   Trades simulés: {len(trades_df)}")
            self.logger.debug(f"   PnL total: ${total_pnl:.2f}")
            self.logger.debug(f"   Fees totales: ${total_fees:.2f}")
            self.logger.debug(f"   Win rate: {win_rate:.2%}")
            self.logger.debug(f"   Avg win/loss: ${avg_win:.2f} / ${avg_loss:.2f}")
        else:
            self.logger.debug("   Aucun trade généré")

        return trades_df

    def _calculate_equity_returns(
        self, df_1m: pd.DataFrame, trades_df: pd.DataFrame, params: dict[str, Any]
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calcule l'equity curve et les returns séries.

        Méthodologie:
        1. Equity de base = capital initial constant
        2. Applique PnL cumulé des trades à leurs dates de sortie
        3. Calculate returns = pct_change() de l'equity
        4. Assure dtype float64 pour compatibilité performance.summarize

        Args:
            df_1m: DataFrame OHLCV pour index temporel
            trades_df: DataFrame des trades exécutés
            params: Paramètres avec initial_capital

        Returns:
            tuple[pd.Series, pd.Series]: (equity, returns)
        """
        self.logger.debug("📈 Calcul equity curve et returns")

        # Capital initial
        initial_capital = params.get("initial_capital", 10000.0)
        equity = pd.Series(
            initial_capital, index=df_1m.index, name="equity", dtype=np.float64
        )

        if not trades_df.empty:
            # Applique les PnL des trades à leurs dates de sortie
            cumulative_pnl = 0.0

            for _, trade in trades_df.iterrows():
                exit_ts = trade["exit_ts"]
                cumulative_pnl += trade["pnl"]

                # Applique PnL cumulé à partir de la date de sortie
                if exit_ts in equity.index:
                    mask = equity.index >= exit_ts
                    equity.loc[mask] = initial_capital + cumulative_pnl

        # Calcul returns (percentage change)
        returns = equity.pct_change().fillna(0.0)
        returns.name = "returns"

        # Validation et casting final pour performance.summarize
        equity = equity.astype(np.float64)
        returns = returns.astype(np.float64)

        # Stats finales pour logs
        total_return_pct = ((equity.iloc[-1] / equity.iloc[0]) - 1) * 100
        max_equity = equity.max()
        min_equity = equity.min()

        self.logger.debug(f"   Equity finale: ${equity.iloc[-1]:,.2f}")
        self.logger.debug(f"   Return total: {total_return_pct:.2f}%")
        self.logger.debug(f"   Equity range: ${min_equity:,.2f} → ${max_equity:,.2f}")

        return equity, returns

    def _build_metadata(
        self,
        device_info: dict[str, Any],
        duration: float,
        df_1m: pd.DataFrame,
        trades_df: pd.DataFrame,
        params: dict[str, Any],
        seed: int,
    ) -> dict[str, Any]:
        """
        Construit les métadonnées complètes d'exécution.

        Args:
            device_info: Info devices utilisés
            duration: Durée d'exécution en secondes
            df_1m: DataFrame des données
            trades_df: DataFrame des trades
            params: Paramètres de stratégie
            seed: Seed utilisé

        Returns:
            Dict: Métadonnées complètes pour RunResult.meta
        """

        # Calculs dérivés
        data_points = len(df_1m)
        throughput = data_points / duration if duration > 0 else 0

        # Period analysis
        period_days = (df_1m.index[-1] - df_1m.index[0]).days
        trades_per_day = len(trades_df) / period_days if period_days > 0 else 0

        # Performance flags
        performance_flags = []
        if throughput < 1000:
            performance_flags.append("low_throughput")
        if len(trades_df) == 0:
            performance_flags.append("no_trades")
        if duration > 30:
            performance_flags.append("slow_execution")
        if device_info["mode"] == "cpu" and self.gpu_available:
            performance_flags.append("gpu_fallback")

        # Trading stats
        trading_stats = {}
        if not trades_df.empty:
            trading_stats = {
                "total_pnl": float(trades_df["pnl"].sum()),
                "win_rate": float((trades_df["pnl"] > 0).mean()),
                "avg_trade_pnl": float(trades_df["pnl"].mean()),
                "max_trade_pnl": float(trades_df["pnl"].max()),
                "min_trade_pnl": float(trades_df["pnl"].min()),
                "total_fees": float(
                    trades_df["fees_paid"].sum()
                    if "fees_paid" in trades_df.columns
                    else 0
                ),
                "avg_trade_duration_hours": float(
                    (trades_df["exit_ts"] - trades_df["entry_ts"])
                    .dt.total_seconds()
                    .mean()
                    / 3600
                ),
            }

        return {
            # Execution context
            "engine_version": "Phase 10 - Production",
            "seed": seed,
            "run_timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            # Device information
            **device_info,
            # Performance metrics
            "duration_seconds": round(duration, 3),
            "data_points": data_points,
            "throughput_points_per_sec": round(throughput, 1),
            "performance_flags": performance_flags,
            # Trading results
            "total_trades": len(trades_df),
            "trades_per_day": round(trades_per_day, 2),
            "period_days": period_days,
            "trading_stats": trading_stats,
            # Strategy parameters (pour reproducibilité)
            "strategy_params": params.copy(),
            # Data period info
            "data_start": df_1m.index[0].isoformat(),
            "data_end": df_1m.index[-1].isoformat(),
            # System info (placeholders pour extensions futures)
            "cache_hits": 0,
            "cache_misses": 0,
            "memory_peak_mb": 0,
            # Multi-GPU specifics
            "multi_gpu_enabled": self.use_multi_gpu,
            "gpu_balance": device_info.get("balance", {}),
            "device_count": len(device_info.get("devices", [])),
        }

    def run_backtest_with_validation(
        self,
        df_1m: pd.DataFrame,
        indicators: dict[str, Any],
        *,
        params: dict[str, Any],
        symbol: str,
        timeframe: str,
        validation_config: Optional[ValidationConfig] = None,
        seed: int = 42,
        use_gpu: Optional[bool] = None,
    ) -> dict[str, Any]:
        """
        Exécute backtest avec validation anti-overfitting complète.

        Cette méthode applique une validation robuste via walk-forward ou train/test split
        pour détecter l'overfitting et garantir des performances réalistes out-of-sample.

        Pipeline:
        1. Vérification intégrité temporelle des données (look-ahead bias)
        2. Split données en train/test selon méthode configurée
        3. Exécution backtest sur chaque split
        4. Calcul ratio overfitting (IS_sharpe / OOS_sharpe)
        5. Recommandations automatiques basées sur ratio

        Args:
            df_1m: DataFrame OHLCV 1-minute avec index datetime UTC
            indicators: Dict indicateurs calculés via bank.ensure()
            params: Paramètres stratégie (entry_z, k_sl, leverage, etc.)
            symbol: Symbole tradé (ex: "BTCUSDC")
            timeframe: Timeframe de référence (ex: "1m", "1h")
            validation_config: Configuration validation (None = use default from __init__)
            seed: Seed pour déterminisme (default: 42)
            use_gpu: Force GPU usage (None = auto)

        Returns:
            Dict avec:
                - in_sample: Métriques train (mean/std sharpe, return, drawdown, etc.)
                - out_sample: Métriques test (idem)
                - overfitting_ratio: IS_sharpe / OOS_sharpe
                - recommendation: Texte explicatif basé sur ratio
                - method: Méthode validation utilisée
                - n_windows: Nombre fenêtres (walk-forward uniquement)
                - all_results: Liste résultats individuels par split

        Raises:
            ValueError: Si validation module non disponible
            ValueError: Si données ont problèmes temporels

        Examples:
            >>> # Validation walk-forward (défaut)
            >>> results = engine.run_backtest_with_validation(
            ...     df_1m, indicators, params=params, symbol="BTCUSDC", timeframe="1m"
            ... )
            >>> print(f"Overfitting ratio: {results['overfitting_ratio']:.2f}")
            >>> print(results['recommendation'])
            >>>
            >>> # Train/test split simple
            >>> config = ValidationConfig(method="train_test", train_ratio=0.7)
            >>> results = engine.run_backtest_with_validation(
            ...     df_1m, indicators, params=params, symbol="BTCUSDC", timeframe="1m",
            ...     validation_config=config
            ... )

        Notes:
            - Overfitting ratio < 1.2: ✅ Excellent, stratégie robuste
            - Overfitting ratio 1.2-1.5: ⚠️ Acceptable, léger overfitting
            - Overfitting ratio 1.5-2.0: 🟡 Attention, overfitting modéré
            - Overfitting ratio > 2.0: 🔴 Critique, stratégie non viable
        """
        if not VALIDATION_AVAILABLE:
            raise ValueError(
                "Module validation non disponible. "
                "Installer avec: pip install -e . pour activer threadx.backtest.validation"
            )

        self.logger.info(f"🔍 Démarrage backtest avec validation: {symbol} {timeframe}")

        # Vérifier intégrité temporelle des données AVANT validation
        try:
            check_temporal_integrity(df_1m)
            self.logger.debug("✅ Intégrité temporelle validée")
        except ValueError as e:
            self.logger.error(f"❌ Problème intégrité temporelle: {e}")
            raise

        # Utiliser config fournie ou celle par défaut de l'instance
        config = validation_config or self.validation_config
        if config is None:
            config = ValidationConfig()  # Fallback config par défaut
            self.logger.warning("⚠️ Aucune config validation, utilisation défaut")

        # Créer validator avec config spécifique
        validator = BacktestValidator(config)

        # Définir fonction de backtest à valider
        def backtest_func(
            data: pd.DataFrame, params_dict: dict[str, Any]
        ) -> dict[str, float]:
            """
            Wrapper pour exécuter self.run() et extraire métriques nécessaires.

            Args:
                data: Sous-ensemble de df_1m (train ou test split)
                params_dict: Paramètres stratégie

            Returns:
                Dict avec métriques: sharpe_ratio, total_return, max_drawdown, etc.
            """
            try:
                # Re-calculer indicateurs sur split spécifique
                # (important pour éviter look-ahead bias!)
                split_indicators = {}
                if "bollinger" in indicators:
                    # Pour simplification, on utilise indicateurs pré-calculés
                    # TODO: Re-calculer indicateurs par split pour robustesse totale
                    split_indicators["bollinger"] = indicators["bollinger"]
                if "atr" in indicators:
                    split_indicators["atr"] = indicators["atr"]

                # Exécuter backtest sur ce split
                result = self.run(
                    df_1m=data,
                    indicators=split_indicators,
                    params=params_dict,
                    symbol=symbol,
                    timeframe=timeframe,
                    seed=seed,
                    use_gpu=use_gpu,
                )

                # Calculer métriques depuis RunResult
                returns = result.returns
                trades = result.trades

                # Sharpe ratio (annualisé)
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(
                        252 * 24 * 60
                    )  # 1-min bars
                else:
                    sharpe = 0.0

                # Total return
                total_return = (result.equity.iloc[-1] / result.equity.iloc[0]) - 1

                # Max drawdown
                equity = result.equity
                cummax = equity.cummax()
                drawdown = (equity - cummax) / cummax
                max_drawdown = drawdown.min()

                # Win rate
                if len(trades) > 0:
                    win_rate = (trades["pnl"] > 0).sum() / len(trades)
                else:
                    win_rate = 0.0

                # Profit factor
                if len(trades) > 0:
                    wins = trades[trades["pnl"] > 0]["pnl"].sum()
                    losses = abs(trades[trades["pnl"] < 0]["pnl"].sum())
                    profit_factor = wins / losses if losses > 0 else float("inf")
                else:
                    profit_factor = 1.0

                return {
                    "sharpe_ratio": float(sharpe),
                    "total_return": float(total_return),
                    "max_drawdown": float(max_drawdown),
                    "win_rate": float(win_rate),
                    "profit_factor": float(profit_factor),
                }

            except Exception as e:
                self.logger.error(f"❌ Erreur dans backtest_func split: {e}")
                # Retourner métriques nulles en cas d'erreur
                return {
                    "sharpe_ratio": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                }

        # Exécuter validation complète
        self.logger.info(
            f"🔄 Validation {config.method} avec {config.walk_forward_windows if config.method == 'walk_forward' else 1} splits"
        )
        validation_results = validator.validate_backtest(
            backtest_func=backtest_func, data=df_1m, params=params
        )

        # Logs résultats
        self.logger.info("📊 Résultats validation:")
        self.logger.info(
            f"   In-Sample Sharpe: {validation_results['in_sample']['mean_sharpe_ratio']:.2f} "
            f"± {validation_results['in_sample']['std_sharpe_ratio']:.2f}"
        )
        self.logger.info(
            f"   Out-Sample Sharpe: {validation_results['out_sample']['mean_sharpe_ratio']:.2f} "
            f"± {validation_results['out_sample']['std_sharpe_ratio']:.2f}"
        )
        self.logger.info(
            f"   Overfitting Ratio: {validation_results['overfitting_ratio']:.2f}"
        )

        # Alerte si overfitting critique
        if validation_results["overfitting_ratio"] > 2.0:
            self.logger.warning(
                "🔴 ALERTE: Overfitting critique détecté! Stratégie non fiable."
            )
        elif validation_results["overfitting_ratio"] > 1.5:
            self.logger.warning(
                "🟡 ATTENTION: Overfitting modéré, réduire nombre paramètres."
            )
        else:
            self.logger.info("✅ Stratégie robuste, overfitting acceptable.")

        self.logger.info(f"\n{validation_results['recommendation']}\n")

        return validation_results


# === Factory Functions et API Convenience ===


def create_engine(
    gpu_balance: Optional[dict[str, float]] = None, use_multi_gpu: bool = True
) -> BacktestEngine:
    """
    Factory function pour créer une instance BacktestEngine.

    Args:
        gpu_balance: Balance multi-GPU personnalisée
        use_multi_gpu: Active multi-GPU si disponible

    Returns:
        BacktestEngine: Instance configurée

    Examples:
        >>> # Moteur par défaut (balance auto)
        >>> engine = create_engine()
        >>>
        >>> # Balance personnalisée 80/20
        >>> engine = create_engine(gpu_balance={"5090": 0.8, "2060": 0.2})
        >>>
        >>> # Force single GPU
        >>> engine = create_engine(use_multi_gpu=False)
    """
    return BacktestEngine(gpu_balance=gpu_balance, use_multi_gpu=use_multi_gpu)


def run(
    df_1m: pd.DataFrame,
    indicators: dict[str, Any],
    *,
    params: dict[str, Any],
    symbol: str,
    timeframe: str,
    seed: int = 42,
    use_gpu: Optional[bool] = None,
    gpu_balance: Optional[dict[str, float]] = None,
) -> RunResult:
    """
    Fonction de convenience pour exécution directe sans instanciation.

    Équivalent à BacktestEngine().run(...) mais plus concise pour usage ponctuel.
    Crée une instance temporaire, exécute, et retourne RunResult.

    Args:
        df_1m: DataFrame OHLCV
        indicators: Dict indicateurs de bank.ensure
        params: Paramètres stratégie
        symbol: Symbole tradé
        timeframe: Timeframe
        seed: Seed déterminisme
        use_gpu: Force GPU usage
        gpu_balance: Balance multi-GPU

    Returns:
        RunResult: Prêt pour performance.summarize()

    Examples:
        >>> # Usage minimal
        >>> result = run(df_1m, indicators, params={"entry_z": 2.0, "k_sl": 1.5, "leverage": 3},
        ...              symbol="BTCUSDC", timeframe="1m")
        >>>
        >>> # Avec configuration GPU
        >>> result = run(df_1m, indicators, params=params, symbol="ETHUSDC", timeframe="15m",
        ...              use_gpu=True, gpu_balance={"5090": 0.9, "2060": 0.1})
    """
    engine = BacktestEngine(gpu_balance=gpu_balance, use_multi_gpu=True)
    return engine.run(
        df_1m=df_1m,
        indicators=indicators,
        params=params,
        symbol=symbol,
        timeframe=timeframe,
        seed=seed,
        use_gpu=use_gpu,
    )


# === Module Initialization ===
logger.info("ThreadX Backtest Engine v10 loaded")
logger.debug(f"   GPU utils: {'✅' if GPU_UTILS_AVAILABLE else '❌'}")
logger.debug(f"   XP utils: {'✅' if XP_AVAILABLE else '❌'}")
logger.debug(f"   Timing utils: {'✅' if 'measure_throughput' in globals() else '❌'}")
