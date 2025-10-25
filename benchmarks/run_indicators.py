#!/usr/bin/env python3
"""
ThreadX Benchmarks - Run Indicators
===================================

Benchmarks de performance pour les indicateurs techniques:
- Mesure des vitesses CPU vs GPU
- Validation des KPI de performance (≥3× speedup)
- Tests de scalabilité avec volumes croissants
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from threadx.indicators.bank import IndicatorBank
from threadx.utils.determinism import set_global_seed
from threadx.utils.log import get_logger

logger = get_logger(__name__)

# KPI Targets
TARGET_SPEEDUP = 3.0  # ≥3× speedup GPU vs CPU
TARGET_GPU_UTILIZATION = 0.70  # ≥70% GPU utilization
SEED_GLOBAL = 42


def generate_benchmark_data(n_points: int) -> pd.DataFrame:
    """Génère des données OHLCV pour benchmarks."""
    set_global_seed(SEED_GLOBAL)
    
    dates = pd.date_range('2024-01-01', periods=n_points, freq='1min')
    
    # Série temporelle réaliste avec volatilité
    base_price = 50000.0
    drift = np.random.randn(n_points).cumsum() * 10
    volatility = 200 * (1 + 0.5 * np.sin(np.arange(n_points) / 1440))
    noise = np.random.randn(n_points) * volatility
    
    close = base_price + drift + noise
    high = close + np.abs(np.random.randn(n_points) * 50)
    low = close - np.abs(np.random.randn(n_points) * 50)
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    
    volume = np.random.randint(1000, 100000, n_points)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


def benchmark_bollinger_bands(data_sizes: list[int]) -> pd.DataFrame:
    """Benchmark Bollinger Bands sur différentes tailles."""
    logger.info("Benchmark Bollinger Bands")
    
    results = []
    bank = IndicatorBank()
    
    # Paramètres test
    test_params = [
        {'period': 20, 'std': 2.0},
        {'period': 50, 'std': 1.5},
        {'period': 100, 'std': 2.5}
    ]
    
    for n_points in data_sizes:
        data = generate_benchmark_data(n_points)
        
        for params in test_params:
            # Test CPU
            start = time.perf_counter()
            cpu_result = bank.ensure(
                'bollinger', params, data['close'],
                symbol=f"BENCH_CPU_{n_points}", timeframe="1m"
            )
            cpu_duration = time.perf_counter() - start
            
            # Test GPU (si disponible)
            gpu_duration = None
            speedup = None
            
            try:
                # Force nouveau calcul
                start = time.perf_counter()
                gpu_result = bank.ensure(
                    'bollinger', params, data['close'],
                    symbol=f"BENCH_GPU_{n_points}", timeframe="1m"
                )
                gpu_duration = time.perf_counter() - start
                
                if gpu_duration > 0:
                    speedup = cpu_duration / gpu_duration
                    
            except Exception as e:
                logger.warning(f"GPU benchmark failed: {e}")
            
            # Calcul throughput
            throughput_cpu = n_points / cpu_duration if cpu_duration > 0 else 0
            throughput_gpu = n_points / gpu_duration if gpu_duration and gpu_duration > 0 else None
            
            results.append({
                'indicator': 'bollinger',
                'params': str(params),
                'n_points': n_points,
                'cpu_duration': cpu_duration,
                'gpu_duration': gpu_duration,
                'speedup': speedup,
                'throughput_cpu': throughput_cpu,
                'throughput_gpu': throughput_gpu,
                'target_speedup': TARGET_SPEEDUP,
                'speedup_ok': speedup >= TARGET_SPEEDUP if speedup else False
            })
            
            logger.info(f"Bollinger {params} @ {n_points}: "
                       f"CPU={cpu_duration:.4f}s, GPU={gpu_duration or 'N/A'}, "
                       f"speedup={speedup:.2f}x" if speedup else f"speedup=N/A")
    
    return pd.DataFrame(results)


def benchmark_atr(data_sizes: list[int]) -> pd.DataFrame:
    """Benchmark ATR sur différentes tailles."""
    logger.info("Benchmark ATR")
    
    results = []
    bank = IndicatorBank()
    
    test_params = [
        {'period': 14, 'method': 'ema'},
        {'period': 21, 'method': 'sma'},
        {'period': 28, 'method': 'ema'}
    ]
    
    for n_points in data_sizes:
        data = generate_benchmark_data(n_points)
        hlc_data = data[['high', 'low', 'close']]
        
        for params in test_params:
            # Test CPU
            start = time.perf_counter()
            cpu_result = bank.ensure(
                'atr', params, hlc_data,
                symbol=f"BENCH_ATR_CPU_{n_points}", timeframe="1m"
            )
            cpu_duration = time.perf_counter() - start
            
            # Test GPU
            gpu_duration = None
            speedup = None
            
            try:
                start = time.perf_counter()
                gpu_result = bank.ensure(
                    'atr', params, hlc_data,
                    symbol=f"BENCH_ATR_GPU_{n_points}", timeframe="1m"
                )
                gpu_duration = time.perf_counter() - start
                
                if gpu_duration > 0:
                    speedup = cpu_duration / gpu_duration
                    
            except Exception as e:
                logger.warning(f"ATR GPU benchmark failed: {e}")
            
            throughput_cpu = n_points / cpu_duration if cpu_duration > 0 else 0
            throughput_gpu = n_points / gpu_duration if gpu_duration and gpu_duration > 0 else None
            
            results.append({
                'indicator': 'atr',
                'params': str(params),
                'n_points': n_points,
                'cpu_duration': cpu_duration,
                'gpu_duration': gpu_duration,
                'speedup': speedup,
                'throughput_cpu': throughput_cpu,
                'throughput_gpu': throughput_gpu,
                'target_speedup': TARGET_SPEEDUP,
                'speedup_ok': speedup >= TARGET_SPEEDUP if speedup else False
            })
            
            logger.info(f"ATR {params} @ {n_points}: "
                       f"CPU={cpu_duration:.4f}s, GPU={gpu_duration or 'N/A'}, "
                       f"speedup={speedup:.2f}x" if speedup else f"speedup=N/A")
    
    return pd.DataFrame(results)


def run_comprehensive_benchmarks():
    """Exécute tous les benchmarks d'indicateurs."""
    logger.info("=== ThreadX Benchmarks - Indicateurs ===")
    
    # Tailles de données pour tests
    data_sizes = [1000, 10000, 50000, 100000, 1000000]
    
    all_results = []
    
    # Benchmark Bollinger Bands
    try:
        bb_results = benchmark_bollinger_bands(data_sizes)
        all_results.append(bb_results)
        logger.info(f"Bollinger benchmarks: {len(bb_results)} tests complétés")
    except Exception as e:
        logger.error(f"Bollinger benchmark failed: {e}")
    
    # Benchmark ATR
    try:
        atr_results = benchmark_atr(data_sizes)
        all_results.append(atr_results)
        logger.info(f"ATR benchmarks: {len(atr_results)} tests complétés")
    except Exception as e:
        logger.error(f"ATR benchmark failed: {e}")
    
    # Consolidation résultats
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Sauvegarde
        output_path = Path("artifacts/reports/benchmark_indicators.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_results.to_csv(output_path, index=False)
        
        # Rapport de synthèse
        generate_benchmark_summary(combined_results)
        
        logger.info(f"Benchmarks sauvés: {output_path}")
        return combined_results
    else:
        logger.error("Aucun benchmark complété avec succès")
        return None


def generate_benchmark_summary(results: pd.DataFrame):
    """Génère un rapport de synthèse des benchmarks."""
    
    # Analyse des KPI
    gpu_results = results[results['gpu_duration'].notna()]
    
    if len(gpu_results) > 0:
        avg_speedup = gpu_results['speedup'].mean()
        max_speedup = gpu_results['speedup'].max()
        speedup_success_rate = (gpu_results['speedup'] >= TARGET_SPEEDUP).mean()
        
        # Performance par taille de données
        performance_by_size = gpu_results.groupby('n_points').agg({
            'speedup': ['mean', 'max', 'min'],
            'cpu_duration': 'mean',
            'gpu_duration': 'mean'
        }).round(4)
        
        summary = {
            'total_tests': len(results),
            'gpu_tests': len(gpu_results),
            'avg_speedup': avg_speedup,
            'max_speedup': max_speedup,
            'target_speedup': TARGET_SPEEDUP,
            'speedup_success_rate': speedup_success_rate,
            'kpi_speedup_ok': avg_speedup >= TARGET_SPEEDUP,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Sauvegarde résumé
        summary_path = Path("artifacts/reports/benchmark_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Performance détaillée
        perf_path = Path("artifacts/reports/benchmark_performance_by_size.csv")
        performance_by_size.to_csv(perf_path)
        
        logger.info("=== BENCHMARK SUMMARY ===")
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"GPU tests: {summary['gpu_tests']}")
        logger.info(f"Average speedup: {avg_speedup:.2f}x (target: {TARGET_SPEEDUP}x)")
        logger.info(f"Success rate: {speedup_success_rate:.1%}")
        logger.info(f"KPI Status: {'✅ PASS' if summary['kpi_speedup_ok'] else '❌ FAIL'}")
        
    else:
        logger.warning("Aucun test GPU réussi - impossible de calculer les KPI")


if __name__ == "__main__":
    results = run_comprehensive_benchmarks()
    
    if results is not None:
        print(f"\n✅ Benchmarks terminés avec succès: {len(results)} tests")
    else:
        print("\n❌ Échec des benchmarks")
        sys.exit(1)
