# 📚 SURVOL COMPLET DU CODEBASE THREADX - 113 FICHIERS PYTHON

**Date**: 31 Octobre 2025 | **Version**: ThreadX v2.0
**Total Fichiers**: 113 Python | **Arborescence**: 10+ modules principaux

---

## 📋 TABLE DES MATIÈRES

1. [Section 1: BENCHMARKS](#section-1-benchmarks--3-fichiers)
2. [Section 2: SCRIPTS](#section-2-scripts-root--2-fichiers)
3. [Section 3: EXAMPLES](#section-3-examples--archive)
4. [Section 4: SRC/THREADX - RACINE](#section-4-srcthreadx-racine--4-fichiers)
5. [Section 5: BACKTEST MODULE](#section-5-backtest-module--5-fichiers)
6. [Section 6: BRIDGE MODULE](#section-6-bridge-module--7-fichiers)
7. [Section 7: CLI MODULE](#section-7-cli-module--8-fichiers)
8. [Section 8: CONFIGURATION](#section-8-configuration--5-fichiers)
9. [Section 9: DATA MODULE](#section-9-data-module--4-fichiers)
10. [Section 10: GPU MODULE](#section-10-gpu-module--5-fichiers)
11. [Section 11: INDICATORS MODULE](#section-11-indicators-module--8-fichiers)
12. [Section 12: OPTIMIZATION MODULE](#section-12-optimization-module--11-fichiers)
13. [Section 13: STRATEGY MODULE](#section-13-strategy-module--5-fichiers)
14. [Section 14: UI MODULE](#section-14-ui-module--8-fichiers)
15. [Section 15: UTILS MODULE](#section-15-utils-module--7-fichiers)
16. [Section 16: TESTING & TOOLS](#section-16-testing--tools--3-fichiers)

---

## SECTION 1: BENCHMARKS (3 fichiers)

### Structure
```
benchmarks/
├── _archive/          (ancien code de benchmark)
└── README.md          (documentation benchmarks)
```

**Fichiers identifiés mais non trouvés**:
- `bench_indicators.py` (archive)
- `run_backtests.py` (archive)
- `run_indicators.py` (archive)
- `utils.py` (archive)

**Status**: Ces fichiers semblent avoir été archivés/supprimés. Le dossier `benchmarks/` contient surtout une archive.

---

## SECTION 2: SCRIPTS ROOT (2 fichiers)

Situés à la racine `d:\ThreadX_big\scripts/`

### 1. `dedupe_parquets.py`
- **Responsabilité**: Déduplication de fichiers Parquet
- **Type**: Utilitaire de maintenance
- **Purpose**: Nettoyer les doublonn données

### 2. `inspect_parquet_compare.py`
- **Responsabilité**: Inspection et comparaison de fichiers Parquet
- **Type**: Outil de debug
- **Purpose**: Comparer contenu/structure Parquet

### Dossier `scripts/_legacy/`
Contient des scripts dépréciés archivés (~25 fichiers):
- `apply_batch_regen.py` - Régénération batch obsolète
- `apply_pandera_corrections.py` - Corrections Pandera
- `batch_regenerate_failed.py` - Batch regeneration
- `build_data_inventory.py` - Inventaire données
- `build_validated_mapping.py` - Mapping validé
- Et 20+ autres scripts de migration/validation

---

## SECTION 3: EXAMPLES (Archive)

Situé à `d:\ThreadX_big\examples/`

### Dossier `examples/_archive/`
Contient exemples dépréciés:
- Probablement code legacy v1 ou exemples d'utilisation anciens

---

## SECTION 4: SRC/THREADX - RACINE (4 fichiers)

### 1. `__init__.py`
**Status**: ✅ Minimal package init
- Imports: `Settings`, `get_settings`, `load_settings`, `ConfigurationError`, `PathValidationError`
- **Version**: 1.0.0
- **Role**: Point d'entrée package ThreadX

### 2. `config.py`
**Status**: ⚠️ Stubs minimalistes
- Classes de stub: `Settings`, `ConfigurationError`, `PathValidationError`
- **Functions**: `get_settings()`, `load_settings()`
- **Purpose**: Configuration stubs pour UI locale sans dépendances complètes
- **Note**: La vraie config est dans `configuration/loaders.py`

### 3. `data_access.py` (208 lignes)
**Status**: ✅ Opérationnel
- **Responsabilité**: Accès aux données OHLCV
- **Key Functions**:
  - `_default_data_dir()` - Localise dossier data robustement
  - `discover_tokens_and_timeframes()` - Énumère tokens/TF disponibles
  - `get_available_timeframes_for_token()` - TF pour un token
- **Data Folders**: `crypto_data_parquet/`, `crypto_data_json/`
- **Extensions**: `.parquet`, `.feather`, `.csv`, `.json`
- **Caching**: `@lru_cache` pour performance

### 4. `streamlit_app.py`
**Status**: ✅ Streamlit v2.0
- **Responsabilité**: Application Streamlit principale
- **Architecture**:
  - Page 1: Configuration & Stratégie (`page_config_strategy.py`)
  - Page 2: Backtest & Optimisation (`page_backtest_optimization.py`)
- **Features**:
  - Styles CSS modernes (gradient dark blue)
  - Session state gérée par `init_session()`
  - Sidebar avec navigation
  - 2 pages UI fusionnées (v1 avait 5 pages)
- **Key Functions**:
  - `init_session()` - Initialisation avec defaults BTC
  - `render_sidebar()` - Affichage sidebar
  - `main()` - Orchestration principale

---

## SECTION 5: BACKTEST MODULE (5 fichiers)

Situé à `src/threadx/backtest/`

### 1. `engine.py` (1276 lignes) ⭐ CORE
**Status**: ✅ Production-ready Phase 10
- **Responsabilité**: Orchestrateur principal de backtesting
- **Architecture**:
  - Device-agnostic via `utils.xp` (NumPy/CuPy)
  - Multi-GPU via `utils.gpu.multi_gpu`
  - Performance tracking via `utils.timing`
- **Key Classes**:
  - `BacktestEngine` - Orchestrateur principal
  - `RunResult` - Structure standardisée résultats
- **Strategy**: Bollinger mean reversion + ATR filter
- **Output**: Compatible avec `performance.summarize()`
- **Validation**: Anti-overfitting via Phase 2 validation

### 2. `performance.py` (1207 lignes) ⭐ CORE
**Status**: ✅ Production financière
- **Responsabilité**: Calcul métriques financières
- **GPU Support**: Transparent fallback CPU via `xp()`
- **Key Functions**:
  - Sharpe ratio, Sortino, Max Drawdown, CAGR
  - Visualization avec Matplotlib
  - Robust edge case handling (NaN, inf, trades vides)
- **Robustness**:
  - Validation intégrée
  - Gestion erreurs complète
  - Seed=42 pour déterminisme
- **Windows 11**: Compatible headless mode

### 3. `sweep.py` (865 lignes) ⭐ CORE
**Status**: ✅ Phase 7 Sweep & Logging
- **Responsabilité**: Parameter sweep parallélisé
- **Features**:
  - Multi-threaded grid execution
  - Checkpoint/resume capability
  - Append-only Parquet storage avec file locks
  - GPU/CPU transparent delegation
- **Determinism**: Seed=42
- **File Locking**: Windows + Unix support (msvcrt/fcntl)
- **Data Schema**: Parquet standardisé avec métadonnées

### 4. `validation.py` (742 lignes) ⭐ CORE
**Status**: ✅ Anti-overfitting validation
- **Responsabilité**: Validation backtests robustes
- **Methods**:
  - Walk-forward optimization
  - Train/test split avec purge/embargo
  - Look-ahead bias detection
  - Overfitting ratio calculation
  - K-fold temporal validation
- **@dataclass**: `ValidationConfig` pour paramètres

### 5. `__init__.py`
**Status**: ✅ Module init
- Imports/exports du module backtest

---

## SECTION 6: BRIDGE MODULE (7 fichiers)

Situé à `src/threadx/bridge/`

### 1. `models.py` (359 lignes)
**Status**: ✅ Dataclasses typées
- **Responsabilité**: Structures requête/réponse
- **Key Classes**:
  - `BacktestRequest` - Requête backtest (symbol, timeframe, strategy, params)
  - `BacktestResult` - Résultat backtest (PnL, Sharpe, trades, equity_curve)
  - `DataRequest`, `SweepRequest`, etc.
- **No Business Logic**: Pur structures données
- **Type Safety**: Annotated complètement

### 2. `controllers.py` (Summarized)
**Status**: ✅ Orchestration layer
- **Responsabilité**: Wrappers synchrones autour Engine
- **Key Classes**:
  - `BacktestController` - Lance backtests
  - `IndicatorController` - Construit indicateurs
  - `SweepController` - Parameter sweeps
  - `DataController` - Chargement/validation données
  - `MetricsController` - Calculs financiers
- **Pattern**: Thin wrappers (pas logique métier)

### 3. `exceptions.py`
**Status**: ✅ Exception hierarchy
- `BacktestError`, `DataError`, `IndicatorError`, `SweepError`

### 4. `validation.py`
**Status**: ✅ Request validation
- Pydantic BaseModel pour validation

### 5. `config.py`
**Status**: ✅ Bridge config
- `Configuration` dataclass

### 6. `async_coordinator.py`
**Status**: ⚠️ Async coordination
- Coordination asynchrone (optional)

### 7. `unified_diversity_pipeline.py`
**Status**: ⚠️ Diversity pipeline
- Pipeline de diversité de stratégies

### 8. `__init__.py`
**Status**: ✅ Module init

---

## SECTION 7: CLI MODULE (8 fichiers)

Situé à `src/threadx/cli/`

### 1. `main.py`
**Responsabilité**: Entry point CLI principal
- Argparse orchestration
- Routing vers subcommands

### 2. `backtest_cmd.py`
**Responsabilité**: Command `threadx backtest`
- Parsing arguments backtest
- Appel BacktestController

### 3. `data_cmd.py`
**Responsabilité**: Command `threadx data`
- Gestion données (load, validate, list)

### 4. `indicators_cmd.py`
**Responsabilité**: Command `threadx indicators`
- Calcul/gestion indicateurs

### 5. `optimize_cmd.py`
**Responsabilité**: Command `threadx optimize`
- Parameter sweeps depuis CLI
- TOML config loading

### 6. `utils.py`
**Responsabilité**: Utilitaires CLI
- Format output, table rendering, etc.

### 7. `__init__.py` & `__main__.py`
**Responsabilité**: Module init et entry point
- `python -m threadx` support

---

## SECTION 8: CONFIGURATION (5 fichiers)

Situé à `src/threadx/configuration/`

### 1. `settings.py` (117 lignes) ⭐ CORE
**Status**: ✅ Dataclass configuration
```python
@dataclass(frozen=True)
class Settings:
    # Paths: DATA_ROOT, RAW_JSON, PROCESSED, INDICATORS, RUNS, LOGS, etc.
    # GPU: DEVICES, LOAD_BALANCE, MEMORY_THRESHOLD, AUTO_FALLBACK
    # Performance: TARGET_TASKS_PER_MIN, VECTORIZATION_BATCH_SIZE, CACHE_TTL_SEC
    # Trading: SUPPORTED_TF, BASE_CURRENCY, FEE_RATE, SLIPPAGE_RATE
    # Backtesting: INITIAL_CAPITAL, MAX_POSITIONS, POSITION_SIZE, STOP_LOSS
    # Logging: LOG_LEVEL, LOG_FORMAT, LOG_ROTATE
    # Security: READ_ONLY_DATA, VALIDATE_PATHS
    # Monte Carlo: DEFAULT_SIMULATIONS, MAX_SIMULATIONS
    # Cache: CACHE_ENABLE, CACHE_MAX_SIZE_MB, CACHE_TTL_SECONDS
```
- **Frozen**: True (immutable)
- **Defaults**: Configurés pour trading quantitatif

### 2. `loaders.py` (Summarized)
**Status**: ✅ TOML configuration loader
- **Class**: `TOMLConfigLoader`
- **Functions**:
  - `load_config_dict()` - Charge TOML → dict
  - `load_settings()` - Charge TOML → Settings
  - `get_settings()` - Singleton accessor
- **Validation**: Paths, GPU config, performance config
- **CLI Support**: Argparse integration

### 3. `errors.py`
**Status**: ✅ Exception classes
- `ConfigurationError`
- `PathValidationError`

### 4. `auth.py`
**Status**: ⚠️ Authentication
- Probablement pour futures integrations API

### 5. `__init__.py`
**Status**: ✅ Module init

---

## SECTION 9: DATA MODULE (4 fichiers)

Situé à `src/threadx/data/`

### 1. `schemas.py`
**Status**: ✅ Pandera schemas
- **Responsabilité**: Validation schéma données
- **Contenus**: Schemas OHLCV, indicateurs, etc.

### 2. `validate.py`
**Status**: ✅ Validation données
- **Responsabilité**: Vérifier intégrité/qualité données

### 3. `normalize.py`
**Status**: ✅ Normalisation OHLCV
- **Responsabilité**: Standardiser format OHLCV
- **Config**: `DEFAULT_NORMALIZATION_CONFIG`
- **Function**: `normalize_ohlcv()`

### 4. `__init__.py`
**Status**: ✅ Module init

---

## SECTION 10: GPU MODULE (5 fichiers) ⚡ OPTIMISÉ

Situé à `src/threadx/gpu/`

### 1. `device_manager.py` (413 lignes)
**Status**: ✅ GPU device management
- **Responsabilité**: Gestion GPUs disponibles + détection hétérogène
- **Key Classes**: `DeviceInfo` (dataclass mémoire + compute capability)
- **Features**:
  - Device detection multi-GPU (RTX 5090, RTX 2060, etc.)
  - Friendly name parsing (`_parse_gpu_name()`)
  - NCCL support detection pour synchronisation multi-GPU
  - Memory tracking (total/free/used GB)
  - Compute capability extraction
- **Functions**:
  - `list_devices()` - Liste tous devices (CPU + GPUs)
  - `get_device_by_name()` - Récupère device par nom
  - `check_nccl_support()` - Vérifie NCCL disponible
  - `xp()` - Retourne module backend (CuPy ou NumPy)

### 2. `multi_gpu.py` (918 lignes) ⭐ OPTIMISÉ v2
**Status**: ✅ Multi-GPU orchestration hétérogène
- **Responsabilité**: Distribution travail multi-GPU avec auto-balancing
- **Key Classes**:
  - `MultiGPUManager` - Orchestrateur principal
  - `WorkloadChunk` - Chunk de données avec device assigné
  - `ComputeResult` - Résultat computation avec stats
- **Architecture Pipeline**: Split → Compute → Sync → Merge
- **Load Balancing**: Default 75% RTX 5090 + 25% RTX 2060
- **Optimisations Phase 2**:
  - `profile_auto_balance()` amélioré avec:
    - Warmup runs (2 par défaut) pour stabiliser GPU
    - Mesure temps moyen + écart-type (std)
    - **Efficacité mémoire**: throughput / memory_used
    - Logging détaillé: throughput, std, mem_efficiency
    - Support hétérogène multi-modèles GPU
  - Configuration streams CUDA par device
  - Seed unique par chunk pour reproductibilité
  - NCCL synchronization support
- **Functions**:
  - `distribute_workload()` - Distribution + execution
  - `profile_auto_balance(sample_size, warmup, runs)` - ⚡ AUTO-PROFILING
  - `set_balance()` - Définit ratios manuellement
  - `synchronize()` - Sync NCCL ou CUDA
  - `get_device_stats()` - Stats devices (mémoire, balance)

### 3. `profile_persistence.py`
**Status**: ✅ GPU profile persistence
- **Responsabilité**: Cache profils GPU (benchmarks)
- **Functions**:
  - `stable_hash()` - Hash stable pour signatures
  - `update_gpu_threshold_entry()` - Update profil perf
  - `get_gpu_thresholds()` - Récupère seuils GPU

### 4. `vector_checks.py`
**Status**: ✅ Array validation
- **Class**: `ArrayValidator`
- **Responsabilité**: Validation arrays performante
- **Functions**:
  - `validate_price_data()` - Validation séries temps
  - `validate_indicator_params()` - Validation paramètres
  - `check_array_compatibility()` - Compatibilité arrays
- **Performance**: Non-blocking warnings, optimisée hot-path
- **Features**: NaN/inf detection, shape validation

### 5. `__init__.py`
**Status**: ✅ Module init

**📝 Optimisations GPU Appliquées**:
- ✅ Auto-balance profiling hétérogène (RTX 5090 + RTX 2060)
- ✅ Warmup runs pour mesures précises
- ✅ Efficacité mémoire dans décisions load balancing
- ✅ Logging détaillé throughput + std + mem_efficiency

---

## SECTION 11: INDICATORS MODULE (8 fichiers) ⚡ NUMBA OPTIMISÉ

Situé à `src/threadx/indicators/`

### 1. `bank.py` ⭐ CORE - 1115+ lignes
**Status**: ✅ Indicator Bank centralisé
- **Responsabilité**: Cache centralisé indicateurs + registry
- **Key Features**:
  - Cache disque intelligent (TTL: 3600s)
  - Batch processing automatique (threshold: 100 params)
  - Registry automatique mise à jour (Parquet)
  - GPU multi-carte transparent
  - Validation + recompute forcé
- **Key Classes**:
  - `IndicatorBank` - Main orchestrator
  - `CacheManager` - Gestion cache TTL/checksums
- **Global Functions**:
  - `ensure_indicator()` - Vérifie/recalcule si nécessaire
  - `force_recompute_indicator()` - Recalcul obligatoire
  - `batch_ensure_indicators()` - Batch avec parallélisation
  - `get_bank_stats()` - Statistiques cache
  - `cleanup_indicators_cache()` - Nettoyage
- **Cache Keys**: MD5 sorted params + data checksum
- **Parquet Registry**: Mise à jour automatique

### 2. `bollinger.py`
**Status**: ✅ Bollinger Bands calculator
- **Class**: `BollingerBands`
- **Features**: Mean, std, z-score, %B
- **Params**: Period, std multiplier

### 3. `xatr.py`
**Status**: ✅ ATR calculator
- **Class**: `ATR`
- **Functions**:
  - `compute_atr()` - Simple ATR
  - `compute_atr_batch()` - Batch ATR
  - `validate_atr_results()` - Validation
  - `benchmark_atr_performance()` - Benchmark
- **Performance**: Vectorisé NumPy/CuPy

### 4. `indicators_np.py`
**Status**: ✅ NumPy indicators (core calculations)
- **Functions**: `ema_np()`, `rsi_np()`, `boll_np()`, `macd_np()`, `atr_np()`, `vwap_np()`, `obv_np()`, `vortex_df()`
- **Performance**: 50x faster than pandas rolling
- **Custom EMA**: Optimized implementation

### 5. `numpy_ext.py`
**Status**: ✅ Pandas DataFrame helper functions
- **Helper Functions**:
  - `add_rsi()` - Add RSI column
  - `add_macd()` - Add MACD columns
  - `add_bollinger()` - Add Bollinger columns
  - `add_atr()` - Add ATR column
  - `add_vwap()` - Add VWAP column
  - `add_obv()` - Add OBV column
  - `add_vortex()` - Add Vortex columns
  - `add_all_indicators()` - Add all indicators
- **Purpose**: Easy integration with Streamlit/analysis

### 6. `engine.py`
**Status**: ✅ Indicator engine
- **Responsabilité**: Orchestration calculs indicateurs

### 7. `gpu_integration.py` (969 lignes) ⚡ NUMBA CUDA OPTIMISÉ v2
**Status**: ✅ GPU acceleration + Numba CUDA kernels fusionnés
- **Responsabilité**: Intégration GPU pour indicateurs avec kernels optimisés
- **Key Classes**:
  - `GPUAcceleratedIndicatorBank` - Banque GPU + Numba
- **Optimisations Numba CUDA Phase 2**:
  - ✅ **Kernels CUDA fusionnés**:
    - `_numba_bollinger_kernel()` - SMA + std en un seul kernel
    - `_numba_rsi_kernel()` - Gains/losses + RSI fusionnés
  - ✅ **Config thread/block optimale**: 256 threads/block (RTX 5090/2060)
  - ✅ **Shared memory**: Rolling window en shared memory
  - ✅ **Grid-stride loop**: Support grandes données
  - ✅ **Fallback cascade**: Numba → CuPy → CPU
- **Methods**:
  - `bollinger_bands()` - Bollinger avec auto-dispatch
  - `_bollinger_bands_numba()` - ⚡ Kernel Numba fusionné
  - `_bollinger_bands_gpu()` - CuPy distribution classique
  - `_bollinger_bands_cpu()` - Pandas rolling fallback
  - `atr()` - ATR avec dispatch GPU/CPU
  - `rsi()` - RSI avec dispatch GPU/CPU
  - `_should_use_gpu_dynamic()` - Décision profiling-based
  - `_micro_probe()` - Benchmark CPU vs GPU vs Numba
  - `_dispatch_indicator()` - Dispatch centralisé
- **Performance Features**:
  - Profiling dynamique CPU vs GPU vs Numba
  - Micro-probing pour décision automatique
  - Signature-based caching de décisions
  - Memory efficiency tracking
- **Constants**:
  - `OPTIMAL_THREADS_PER_BLOCK = 256`
  - `OPTIMAL_BLOCKS_PER_SM = 2`
- **Numba Availability**: Fallback gracieux si Numba non installé

### 8. `__init__.py`
**Status**: ✅ Module init

**📝 Optimisations Indicators Appliquées**:
- ✅ Numba CUDA kernels fusionnés (SMA+std, gains+losses)
- ✅ Thread/block configuration optimale (256 threads/block)
- ✅ Shared memory pour rolling windows
- ✅ Cascade fallback: Numba → CuPy → CPU
- ✅ Profiling dynamique pour auto-décision GPU vs CPU

---

## SECTION 12: OPTIMIZATION MODULE (11 fichiers)

Situé à `src/threadx/optimization/`

### 1. `engine.py` (Attached ✅) ⭐ CORE - 1200+ lignes
**Status**: ✅ Phase 10 Unified Optimization Engine
- **Responsabilité**: Moteur d'optimisation paramétrique unifié
- **Key Classes**:
  - `SweepRunner` - Runner sweeps paramétriques
  - `UnifiedOptimizationEngine` - Orchestrateur principal
- **Features**:
  - Batch processing + early stopping
  - IndicatorBank reuse (centralisé)
  - Device-agnostic via xp
  - Dynamic worker adjustment
- **Functions**:
  - `run_grid()` - Grid search
  - `run_monte_carlo()` - Monte Carlo
  - Global stop flag: `set_global_stop()`, `is_global_stop_requested()`, `request_global_stop()`, `clear_global_stop()`
- **Integration**: IndicatorBank + BacktestEngine + PerformanceCalculator

### 2. `scenarios.py`
**Status**: ✅ Scenario specifications
- **Class**: `ScenarioSpec`
- **Functions**:
  - `generate_param_grid()` - Grid generation
  - `generate_monte_carlo()` - MC generation
  - `_normalize_param()` - Param normalization

### 3. `pruning.py`
**Status**: ✅ Pareto pruning
- **Function**: `pareto_soft_prune()` - Pareto front analysis
- **Helper Functions**: Dominance checking, cleaning

### 4. `reporting.py`
**Status**: ✅ Reporting & visualization
- **Functions**:
  - `summarize_distribution()` - Distribution stats
  - `build_heatmaps()` - Parameter heatmaps
  - `write_reports()` - Export reports
  - `validate_results_dataframe()` - Validation

### 5. `ui.py`
**Status**: ✅ UI integration
- **Class**: `ParametricOptimizationUI`
- **Functions**: `create_optimization_ui()`, `init_ui()`

### 6. `run.py` (Attached ✅)
**Status**: ✅ CLI entry point
- **Functions**:
  - `load_config()` - Deprecated wrapper
  - `validate_cli_config()` - Config validation
  - `build_scenario_spec()` - Scenario construction
  - `run_sweep()` - Sweep execution
  - `main()` - CLI orchestration
- **CLI Args**: `--config`, `--dry-run`, `--verbose`

### 7. `presets/ranges.py`
**Status**: ✅ Indicator range presets
- **Class**: `IndicatorRangePreset`, `StrategyPresetMapper`
- **Functions**:
  - `load_all_presets()` - Load all presets
  - `get_indicator_range()` - Get range for indicator
  - `list_available_indicators()` - List indicators
  - `get_strategy_preset()` - Get strategy preset

### 8. `presets/__init__.py`
**Status**: ✅ Presets module init

### 9. `templates/base_optimizer.py`
**Status**: ✅ Base optimizer template
- **Class**: `BaseOptimizer`

### 10. `templates/grid_optimizer.py`
**Status**: ✅ Grid optimizer
- **Function**: `grid_search()`

### 11. `templates/monte_carlo_optimizer.py`
**Status**: ✅ Monte Carlo optimizer
- **Function**: `monte_carlo_search()`

---

## SECTION 13: STRATEGY MODULE (5 fichiers)

Situé à `src/threadx/strategy/`

### 1. `model.py` (850 lignes) ⭐ CORE
**Status**: ✅ Strategy model layer
- **Responsabilité**: Types/structures pour stratégies
- **Key Classes**:
  - `Trade` - Transaction complète
  - `RunStats` - Statistiques performance
  - `TradeDict` - TypedDict optimisé
- **Protocol**: Strategy pattern pour extensibilité
- **Functions**:
  - `validate_ohlcv_dataframe()` - OHLCV validation
  - `validate_strategy_params()` - Param validation
  - `save_run_results()` / `load_run_results()` - Persistence JSON
- **JSON Serialization**: Complète pour persistence

### 2. `amplitude_hunter.py` (Attached ✅) ⭐ ADVANCED
**Status**: ✅ AmplitudeHunter strategy
- **Responsabilité**: Capture amplitude complète Bollinger Bands
- **Strategy Logic**:
  1. Filtre régime multi-critères (BBWidth %ile, Volume z-score, ADX)
  2. Setup "Spring → Drive" (MACD séquentiel)
  3. Score d'Amplitude pour modulation agressivité
  4. Pyramiding intelligent (jusqu'à 2 adds)
  5. Trailing stop conditionnel (%B + MACD)
  6. Cible BIP (Bollinger Implied Price)
  7. Stop loss spécifique SHORT (37% au-dessus entry)
- **Class**: `AmplitudeHunterStrategy`
- **Dataclass**: `AmplitudeHunterParams`
- **Functions**:
  - `generate_signals()` - Signal generation
  - `backtest()` - Backtesting
  - `create_default_params()` - Default params
- **Indicators Used**: Bollinger, MACD, ADX, OBV, Volume
- **Trade Count**: Up to 3 pyramided positions

### 3. `bb_atr.py`
**Status**: ✅ Bollinger Band + ATR strategy
- **Class**: `BBAtrStrategy`
- **Dataclass**: `BBAtrParams`
- **Indicators**: Bollinger Bands + ATR filter

### 4. `bollinger_dual.py`
**Status**: ✅ Dual Bollinger strategy
- **Class**: `BollingerDualStrategy`
- **Dataclass**: `BollingerDualParams`

### 5. `__init__.py`
**Status**: ✅ Module init

---

## SECTION 14: UI MODULE (8 fichiers)

Situé à `src/threadx/ui/`

### 1. `page_config_strategy.py` ⭐ ACTIVE PAGE 1
**Status**: ✅ Streamlit Page 1 (fusion v1)
- **Responsabilité**: Configuration & Stratégie
- **Fusion**: Anciennes pages v1 + v2
  - Data selection (symbol, timeframe, date range)
  - Strategy selection (Bollinger_Breakout, etc.)
  - Parameter configuration
- **Default Presets**:
  - Symbol: BTCUSDC
  - Timeframe: 15m
  - Date range: Dec 1 2024 - Jan 31 2025
- **Key Functions**:
  - `_render_ohlcv_chart()` - Price chart
  - `_render_data_section()` - Data controls
  - `_render_strategy_section()` - Strategy controls
  - `main()` - Page orchestration
- **Session State**: Persiste configuration

### 2. `page_backtest_optimization.py` ⭐ ACTIVE PAGE 2
**Status**: ✅ Streamlit Page 2 (fusion v1)
- **Responsabilité**: Backtest & Optimisation
- **Tabs**:
  1. **Backtest Tab**:
     - Simple backtest avec progress bar
     - Equity curve + metrics
     - Trades table
  2. **Sweep Tab**:
     - Parameter grid configuration
     - Sensitivity sliders (granularité)
     - Combination counter (≤100K optimal, ≤3M max)
     - Progress bar avec vitesse
     - Results export CSV
  3. **Monte Carlo Tab**:
     - Random parameters
     - Scenario count configuration
     - Seed pour reproductibilité
     - Results table
- **Key Functions**:
  - `_render_price_chart()` - Price avec indicateurs
  - `_render_equity_curve()` - Equity visualization
  - `_render_metrics()` - Metrics display
  - `_render_trades_table()` - Trades table
  - `_run_sweep_with_progress()` - Sweep execution
  - `_run_monte_carlo_with_progress()` - MC execution
- **Progress Tracking**: Real-time updates

### 3. `strategy_registry.py`
**Status**: ✅ Strategy registry
- **Responsabilité**: Registre centralisé stratégies
- **Registry**: Bollinger_Breakout, EMA_Cross, ATR_Channel, etc.
- **Key Functions**:
  - `list_strategies()` - List all strategies
  - `parameter_specs_for()` - Get params
  - `indicator_specs_for()` - Get indicators
  - `base_params_for()` - Default params
  - `tunable_parameters_for()` - Tunable params
  - `resolve_range()` - Range resolution
- **Param Types**: Non-tunable (entry_logic, trailing_stop) vs tunable (10 params)

### 4. `fast_sweep.py` (Attached ✅)
**Status**: ✅ Ultra-fast sweep optimisation
- **Responsabilité**: Sweep ultra-rapide pour UI
- **Features**:
  - Batch processing indicateurs (1 calcul seulement)
  - Mise à jour UI espacée (tous les 50 runs)
  - Vectorized NumPy calculations
  - No redundant recalculation
- **Throughput**: 100+ runs/second
- **Key Functions**:
  - `fast_parameter_sweep()` - Main sweep function
  - `simple_bollinger_strategy()` - Ultra-fast Bollinger
  - `bollinger_zscore_strategy()` - Bollinger z-score
  - `adaptive_ma_strategy()` - Adaptive MA
  - `get_strategy_function()` - Strategy lookup
- **Strategies**: Mapping dict par nom

### 5. `backtest_bridge.py`
**Status**: ✅ Bridge UI ↔ Engine
- **Responsabilité**: Interface Streamlit ↔ Backtest Engine
- **Key Functions**:
  - `run_backtest()` - Execute backtest
  - `run_backtest_gpu()` - GPU acceleration
  - `_generate_position()` - Position generation
  - `_compute_equity()` - Equity calculation
- **Class**: `BacktestResult` - Résultats

### 6. `system_monitor.py`
**Status**: ✅ System monitoring
- **Class**: `SystemMonitor`
- **Responsabilité**: CPU/GPU usage monitoring
- **Functions**: `get_global_monitor()`
- **Metrics**: CPU, memory, GPU utilization

### 7. `_legacy_v1/page_selection_token.py`
**Status**: ⚠️ Legacy archive (169 lignes)
- **Note**: Fusionné dans `page_config_strategy.py`

### 8. `_legacy_v1/page_strategy_indicators.py`
**Status**: ⚠️ Legacy archive (202 lignes)
- **Note**: Fusionné dans `page_config_strategy.py`

### 9. `_legacy_v1/page_backtest_results.py`
**Status**: ⚠️ Legacy archive (451 lignes)
- **Note**: Fusionné dans `page_backtest_optimization.py`

### 10. `__init__.py`
**Status**: ✅ Module init

---

## SECTION 15: UTILS MODULE (7 fichiers)

Situé à `src/threadx/utils/`

### 1. `xp.py` ⭐ DEVICE-AGNOSTIC BACKEND
**Status**: ✅ NumPy/CuPy abstraction layer
- **Responsabilité**: Abstraction device-agnostic computing
- **Key Functions**:
  - `get_xp()` - Get backend (NumPy ou CuPy)
  - `gpu_available()` - Check GPU disponibilité
  - `is_gpu_backend()` - Current backend check
  - `get_backend_name()` - Backend name string
  - `to_device()` - Array to GPU
  - `to_host()` - Array to CPU
  - `asnumpy()` - Force NumPy conversion
  - `ascupy()` - Force CuPy conversion
  - `ensure_array_type()` - Type consistency
  - `memory_pool_info()` - GPU memory stats
  - `clear_memory_pool()` - GPU memory cleanup
  - `benchmark_operation()` - Performance testing
  - `device_synchronize()` - GPU sync
  - `get_array_info()` - Array metadata
- **Fallback**: Graceful fallback NumPy si GPU indisponible

### 2. `log.py` ⭐ LOGGING INFRASTRUCTURE
**Status**: ✅ Structured logging
- **Key Functions**:
  - `configure_logging()` - Logger configuration
  - `setup_logging_once()` - One-time setup
  - `get_logger()` - Get logger instance
  - `setup_logging()` - Legacy setup
- **Features**: Structured logging, file rotation, level control

### 3. `timing.py`
**Status**: ✅ Performance timing
- **Decorators**:
  - `@measure_throughput()` - Throughput measurement
  - `@track_memory()` - Memory tracking
  - `@combined_measurement()` - Combined metrics
  - `@performance_context()` - Context manager
- **Features**: Threshold-based logging, adaptive behavior

### 4. `determinism.py`
**Status**: ✅ Deterministic execution
- **Key Functions**:
  - `set_global_seed()` - Set seed (NumPy, CuPy, TensorFlow, etc.)
  - `enforce_deterministic_merges()` - Deterministic DataFrame merge
  - `stable_hash()` - Stable hashing
  - `create_deterministic_splits()` - Reproducible data splits
  - `hash_df()` - DataFrame hashing
  - `validate_determinism()` - Verify reproducibility
  - `get_random_states()` - Capture random state
  - `compare_random_states()` - Compare states
- **Seed=42**: Default throughout codebase

### 5. `cache.py`
**Status**: ✅ Caching utilities
- **Key Classes**:
  - Caching decorators
  - `LRU` cache
  - `TTL` cache
  - Indicators cache with auto-invalidation
- **Functions**:
  - `generate_stable_key()` - Deterministic cache keys
  - `@cached()` - Generic caching decorator
  - `@lru_cache()` - LRU decorator
  - `@ttl_cache()` - TTL decorator
  - `@indicators_cache()` - Specialized indicators cache

### 6. `batching.py`
**Status**: ✅ Batch processing utilities
- **Functions**:
  - `batch_generator()` - Batch generator
  - `adaptive_batch_size()` - Dynamic batch sizing
  - `batch_process()` - Batch processing executor
  - `batch_apply()` - Apply function batched
  - `batch_reduce()` - Reduce function batched
  - `chunked()` - Chunk iterator
- **Adaptive**: Automatic batch size optimization

### 7. `common_imports.py`
**Status**: ✅ DRY common imports
- **Exports**: `pd`, `np`, `logging`, `Dict`, `Any`, `Optional`, etc.
- **Function**: `create_logger()` - Convenient logger creation
- **Purpose**: Reduce import boilerplate across modules

### 8. `__init__.py`
**Status**: ✅ Module init

---

## SECTION 16: TESTING & TOOLS (3 fichiers)

Situé à `tests/` et `tools/`

### 1. `tests/conftest.py`
**Status**: ✅ Pytest configuration
- **Responsabilité**: Pytest fixtures et configuration
- **Fixtures**: Mock data, temporary directories, etc.

### 2. `tests/mocks.py` (via `testing/mocks.py`)
**Status**: ✅ Mock utilities for testing
- **Key Functions**:
  - `get_mock_logger()` - Mock logger
  - `setup_mock_logging_once()` - Mock logging setup
  - `mock_plot_equity()` - Mock plot
  - `mock_plot_drawdown()` - Mock drawdown plot
  - `mock_render_trades_table()` - Mock trades table
  - `mock_render_metrics_table()` - Mock metrics table
  - `mock_export_table()` - Mock export
- **Purpose**: Testing sans Matplotlib/rendering

### 3. `tools/_archive/benchmarks_cpu_gpu.py`
**Status**: ⚠️ Legacy benchmark tool
- **Purpose**: CPU vs GPU benchmarks (archive)

---

## 🏗️ ARCHITECTURE OVERVIEW

### Layer Stack (Bottom to Top)

```
┌─────────────────────────────────────┐
│  STREAMLIT UI (streamlit_app.py)    │ ← User Interface
├─────────────────────────────────────┤
│  UI Pages (page_*.py)               │ ← Page components
├─────────────────────────────────────┤
│  Bridge Controllers (controllers.py)│ ← Orchestration
├─────────────────────────────────────┤
│  Optimization Engine (engine.py)    │ ← Parameter sweeps
│  Backtest Engine (engine.py)        │ ← Backtesting
├─────────────────────────────────────┤
│  Indicator Bank (bank.py)           │ ← Centralized cache
│  Performance Metrics (performance)  │ ← Analytics
├─────────────────────────────────────┤
│  Strategy Models (model.py, *.py)   │ ← Strategy logic
├─────────────────────────────────────┤
│  GPU/Device Layer (xp, gpu/)        │ ← Hardware abstraction
│  Utils (log, cache, timing, etc.)   │ ← Infrastructure
├─────────────────────────────────────┤
│  Configuration (settings, loaders)  │ ← Config
│  Data Access (data_access.py)       │ ← File I/O
└─────────────────────────────────────┘
```

### Data Flow

```
User Input
    ↓
Streamlit Page (UI)
    ↓
Bridge Controller
    ↓
Engine (Backtest/Optimization)
    ↓
Indicator Bank (cache check/compute)
    ↓
Strategy (generate signals)
    ↓
Performance Metrics
    ↓
Results (DataFrame + Visualizations)
    ↓
User Display
```

### Key Dependencies

1. **pandas**: DataFrames (OHLCV, results)
2. **numpy**: Vectorized calculations
3. **cupy** (optional): GPU acceleration
4. **streamlit**: UI framework
5. **pyarrow**: Parquet I/O
6. **matplotlib**: Visualization
7. **pytest**: Testing

---

## 📊 MODULE COUPLING ANALYSIS

### High Coupling (Core)
- `bank.py` ↔ `engine.py` (optimization) - Indicator reuse
- `engine.py` (backtest) ↔ `performance.py` - Results integration
- `streamlit_app.py` ↔ `page_*.py` - UI orchestration
- `strategy/*.py` ↔ `model.py` - Trade structure

### Low Coupling (Modular)
- `utils/*` - Independent utilities
- `configuration/*` - Settings only
- `gpu/*` - Hardware abstraction
- `indicators/xatr.py`, `bollinger.py` - Isolated calculators

### Optional Coupling
- `bridge/` - Asynchronous coordination (optional)
- `cli/` - Command-line only if used
- `testing/` - Test utilities only

---

## 🎯 KEY DESIGN PATTERNS

### 1. **Registry Pattern**
- `strategy_registry.py` - Central strategy lookup

### 2. **Cache Pattern**
- `bank.py` - IndicatorBank with TTL + checksums
- `cache.py` - Decorators `@cached`, `@lru_cache`, `@ttl_cache`

### 3. **Factory Pattern**
- `controllers.py` - Create controllers
- `optimize_cmd.py` - Create scenario specs

### 4. **Bridge Pattern**
- `backtest_bridge.py` - UI ↔ Engine abstraction
- `xp.py` - NumPy ↔ CuPy abstraction

### 5. **Singleton Pattern**
- `get_settings()` - Global settings instance
- `_global_bank` - Global indicator bank

### 6. **Strategy Pattern**
- `strategy/*.py` - Multiple strategies (Amplitude, BB-ATR, etc.)
- `AmplitudeHunterStrategy`, `BBAtrStrategy`, etc.

### 7. **Dataclass Pattern**
- `models.py`, `settings.py` - Configuration as code
- `AmplitudeHunterParams`, `BBAtrParams`, etc.

---

## ⚡ PERFORMANCE CHARACTERISTICS

### Optimization Techniques
1. **Vectorization**: NumPy/CuPy instead of loops
2. **Caching**: IndicatorBank with disk persistence
3. **Batch Processing**: 100+ params → 1 batch
4. **GPU Acceleration**: Transparent CuPy fallback
5. **Early Stopping**: Pareto pruning during sweeps
6. **Worker Adjustment**: Dynamic thread count based on system

### Throughput Targets
- **Fast Sweep**: 100+ runs/second
- **Optimization**: 2500 tasks/minute
- **Batch Indicators**: 1000+ per batch
- **Indicator Cache**: 3600s TTL

### Memory Management
- **Indicator Cache**: 2048 MB max
- **GPU Memory**: 80% threshold before fallback
- **Auto Cleanup**: Stale cache removal

---

## 🐛 ERROR HANDLING & VALIDATION

### Exception Hierarchy
```
Exception
├── ConfigurationError
├── PathValidationError
├── BacktestError
├── DataError
├── IndicatorError
└── SweepError
```

### Validation Points
1. **Data**: `data/validate.py`, `backtest/validation.py`
2. **Configuration**: `configuration/loaders.py`, `settings.py`
3. **Arrays**: `gpu/vector_checks.py`
4. **Strategy Params**: `strategy/model.py`
5. **OHLCV**: `data/schemas.py` (Pandera)

---

## 🔄 UPGRADE PATH (v2.0)

### From v1.0
- **Consolidation**: 5 UI pages → 2 pages
- **Archive**: Legacy v1 in `_legacy_v1/`
- **Fusion**: Config + Backtest pages combined
- **Modern UI**: Gradient styling, responsive layout

### What Changed
- ✅ New pages: `page_config_strategy.py`, `page_backtest_optimization.py`
- ✅ Registry: `strategy_registry.py` centralized
- ✅ Fast sweep: `fast_sweep.py` ultra-optimized
- ✅ Architecture: Bridge pattern for decoupling
- ❌ Old pages: Archived but available

---

## 📈 CODEBASE STATISTICS

| Métrique | Valeur |
|----------|--------|
| Fichiers Python | 113 |
| Modules principaux | 10+ |
| Lignes core code | ~15,000 |
| Lines in Engine (backtest) | 1,276 |
| Lines in Performance | 1,207 |
| Lines in Optimization | 1,200+ |
| Lines in Indicator Bank | 1,115+ |
| Strategies implémentées | 3+ |
| Device support | CPU/GPU |
| UI Framework | Streamlit |
| Data formats | Parquet/JSON/CSV |

---

## 🎓 LEARNING PATH FOR NEWCOMERS

1. **Start**: `streamlit_app.py` - Application entry
2. **UI**: `page_config_strategy.py`, `page_backtest_optimization.py`
3. **Strategy**: `strategy/model.py`, `amplitude_hunter.py`
4. **Backtest**: `backtest/engine.py`, `performance.py`
5. **Optimization**: `optimization/engine.py`, `scenarios.py`
6. **Indicators**: `indicators/bank.py`, `bollinger.py`, `xatr.py`
7. **Utils**: `utils/xp.py`, `log.py`, `cache.py`
8. **Configuration**: `configuration/settings.py`, `loaders.py`

---

## 🚀 DEPLOYMENT NOTES

### Requirements
- Python 3.10+
- pandas, numpy, streamlit
- Optional: cupy (GPU), pyarrow
- Windows 11 compatible

### Configuration
- `paths.toml` - Main config file
- Environment: `THREADX_DATA_DIR`
- GPU: Auto-detection via device_manager

### Running
```bash
# UI
streamlit run src/threadx/streamlit_app.py

# CLI
python -m threadx backtest --config config.toml

# Tests
pytest tests/ -v
```

---

**End of Survey** | Generated: 2025-10-31 | Version: v2.0.0

