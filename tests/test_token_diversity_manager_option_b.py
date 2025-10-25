#!/usr/bin/env python3
"""
Tests complets pour TokenDiversityManager - Contrat Option B
===========================================================

Suite de tests couvrant :
✅ 1. Tests smoke : prepare_dataframe avec indicateurs
✅ 2. Validation OHLCV : resample, règles agrégation
✅ 3. Qualité colonnes : préfixes, dtype, collisions
✅ 4. Cache TTL : hits/miss, invalidation
✅ 5. Déterminisme : reproductibilité avec seed
✅ 6. Performance : budgets latence, micro-benchmarks
✅ 7. Gestion erreurs : messages clairs, codes stables
"""

import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from threadx.data.tokens import (
    TokenDiversityManager,
    IndicatorSpec,
    PriceSourceSpec,
    RunMetadata,
    create_default_config,
    TokenDiversityDataSource,
)

# Configuration des tests
logging.basicConfig(level=logging.INFO)


class TestTokenDiversityManagerOptionB:
    """Tests complets du contrat Option B."""

    @pytest.fixture
    def manager(self):
        """Manager configuré pour tests."""
        return TokenDiversityManager()

    @pytest.fixture
    def basic_price_source(self) -> PriceSourceSpec:
        """Source de prix par défaut (stub)."""
        return PriceSourceSpec(name="stub", params={})

    @pytest.fixture
    def sample_indicators(self) -> list[IndicatorSpec]:
        """Indicateurs de test standard."""
        return [
            IndicatorSpec(name="rsi", params={"window": 14}),
            IndicatorSpec(name="bbands", params={"window": 20, "n_std": 2.0}),
        ]

    def test_smoke_prepare_dataframe_basic(self, manager, basic_price_source):
        """Test smoke : prepare_dataframe basique sans exception."""
        df, meta = manager.prepare_dataframe(
            market="BTCUSDT",
            timeframe="1h",
            start="2023-01-01",
            end="2023-01-08",  # 1 semaine
            indicators=[],
            price_source=basic_price_source,
        )

        # Vérifications basiques
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert isinstance(meta, dict)

        # Colonnes OHLCV obligatoires
        required_cols = {"open", "high", "low", "close", "volume"}
        assert required_cols.issubset(df.columns)

        # Index UTC monotone
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None  # tz-aware
        assert df.index.is_monotonic_increasing

    def test_smoke_with_indicators(
        self, manager, basic_price_source, sample_indicators
    ):
        """Test smoke : prepare_dataframe avec indicateurs RSI + BBands."""
        try:
            df, meta = manager.prepare_dataframe(
                market="BTCUSDT",
                timeframe="1h",
                start="2023-01-01",
                end="2023-01-08",
                indicators=sample_indicators,
                price_source=basic_price_source,
            )

            # Vérifications structure
            assert not df.empty
            assert len(df) > 50  # Suffisant pour indicateurs

            # Colonnes indicateurs présentes (préfixe ind_)
            indicator_cols = [col for col in df.columns if col.startswith("ind_")]
            assert len(indicator_cols) > 0  # Au moins un indicateur calculé

            # Métadonnées complètes
            assert meta["indicators_count"] == len(sample_indicators)
            assert meta["market"] == "BTCUSDT"
            assert meta["execution_time_ms"] > 0

        except ImportError:
            # IndicatorBank non disponible
            pytest.skip("IndicatorBank not available for indicator tests")

    def test_ohlcv_resample_rules(self, manager, basic_price_source):
        """Test validation OHLCV : règles resample 1m→1h."""
        # Récupérer données 1m puis forcer resample
        df_1m, _ = manager.prepare_dataframe(
            market="ETHUSDT",
            timeframe="1m",
            start="2023-01-01T10:00:00",
            end="2023-01-01T12:00:00",  # 2h de données
            indicators=[],
            price_source=basic_price_source,
        )

        # Resample manuel pour vérifier règles
        df_1h = df_1m.resample("1H").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Vérifications règles OHLC
        assert df_1h["high"].iloc[0] >= df_1h["open"].iloc[0]
        assert df_1h["high"].iloc[0] >= df_1h["close"].iloc[0]
        assert df_1h["low"].iloc[0] <= df_1h["open"].iloc[0]
        assert df_1h["low"].iloc[0] <= df_1h["close"].iloc[0]
        assert df_1h["volume"].iloc[0] > 0

    def test_column_quality_and_prefixes(self, manager, basic_price_source):
        """Test qualité colonnes : préfixes, dtype, absence collisions."""
        indicators = [
            IndicatorSpec(name="rsi", params={"window": 14}),
            IndicatorSpec(name="atr", params={"window": 14}),
        ]

        try:
            df, meta = manager.prepare_dataframe(
                market="BTCUSDT",
                timeframe="1h",
                start="2023-01-01",
                end="2023-01-15",  # 2 semaines
                indicators=indicators,
                price_source=basic_price_source,
            )

            # Préfixes corrects
            ohlcv_cols = [col for col in df.columns if not col.startswith("ind_")]
            indicator_cols = [col for col in df.columns if col.startswith("ind_")]

            assert "open" in ohlcv_cols
            assert "close" in ohlcv_cols
            assert len(indicator_cols) > 0

            # Dtype conformes (float64 ou compatible)
            for col in ["open", "high", "low", "close", "volume"]:
                assert pd.api.types.is_numeric_dtype(df[col])

            # Pas de collisions de noms
            assert len(df.columns) == len(set(df.columns))

            # NaN head après warmup acceptable
            head_nans = df.head(50).isna().sum().sum()
            total_cells = len(df.head(50)) * len(df.columns)
            nan_ratio = head_nans / total_cells
            assert nan_ratio < 0.1  # Moins de 10% NaN après warmup

        except ImportError:
            pytest.skip("IndicatorBank not available")

    def test_cache_ttl_functionality(self, manager, basic_price_source):
        """Test cache TTL : 1er appel (miss), 2e (hit), invalidation."""
        params = {
            "market": "BTCUSDT",
            "timeframe": "1h",
            "start": "2023-01-01",
            "end": "2023-01-03",
            "indicators": [],
            "price_source": basic_price_source,
            "cache_ttl_sec": 10,  # 10 secondes TTL
        }

        # 1er appel : cache miss
        start_time = time.time()
        df1, meta1 = manager.prepare_dataframe(**params)
        first_latency = time.time() - start_time

        assert meta1["cache_hit"] is False

        # 2e appel immédiat : cache hit
        start_time = time.time()
        df2, meta2 = manager.prepare_dataframe(**params)
        second_latency = time.time() - start_time

        assert meta2["cache_hit"] is True
        assert second_latency < first_latency  # Cache plus rapide

        # DataFrames identiques
        pd.testing.assert_frame_equal(df1, df2)

        # Attendre expiration cache
        time.sleep(11)

        # 3e appel : cache miss après TTL
        df3, meta3 = manager.prepare_dataframe(**params)
        assert meta3["cache_hit"] is False

    def test_determinism_with_seed(self, manager, basic_price_source):
        """Test déterminisme : 2 runs même seed → DataFrame identique."""
        params = {
            "market": "ETHUSDT",
            "timeframe": "1h",
            "start": "2023-01-01",
            "end": "2023-01-05",
            "indicators": [],
            "price_source": basic_price_source,
            "seed": 42,
        }

        # Vider cache pour test propre
        manager.clear_cache()

        # Premier run
        df1, meta1 = manager.prepare_dataframe(**params)

        # Vider cache entre les runs
        manager.clear_cache()

        # Deuxième run avec même seed
        df2, meta2 = manager.prepare_dataframe(**params)

        # Comparaison bitwise (tolérance pour float)
        pd.testing.assert_frame_equal(df1, df2, check_exact=False, rtol=1e-10)

        # Métadonnées cohérentes
        assert meta1["rows_processed"] == meta2["rows_processed"]
        assert meta1["market"] == meta2["market"]

    def test_performance_budget(self, manager, basic_price_source):
        """Test performance : budget latence < 500ms pour 1000 bars × 2 indicateurs."""
        indicators = [
            IndicatorSpec(name="rsi", params={"window": 14}),
            IndicatorSpec(name="bbands", params={"window": 20, "n_std": 2.0}),
        ]

        try:
            start_time = time.time()
            df, meta = manager.prepare_dataframe(
                market="BTCUSDT",
                timeframe="1h",
                start="2023-01-01",
                end="2023-03-01",  # ~2 mois ≈ 1400 bars
                indicators=indicators,
                price_source=basic_price_source,
            )
            execution_time = time.time() - start_time

            # Budget performance : < 500ms pour cette charge
            assert execution_time < 0.5, f"Trop lent : {execution_time:.2f}s"

            # Métriques dans metadata
            assert meta["execution_time_ms"] > 0
            assert len(df) > 1000  # Suffisamment de données

        except ImportError:
            pytest.skip("Performance test requires IndicatorBank")

    def test_error_handling_invalid_params(self, manager, basic_price_source):
        """Test gestion erreurs : paramètres invalides avec messages clairs."""

        # Plage inversée
        with pytest.raises(ValueError, match="Invalid date range"):
            manager.prepare_dataframe(
                market="BTCUSDT",
                timeframe="1h",
                start="2023-01-15",  # après end
                end="2023-01-01",
                indicators=[],
                price_source=basic_price_source,
            )

        # Market inconnu avec message explicite
        with pytest.raises(Exception):  # DataNotFoundError ou ValueError
            manager.prepare_dataframe(
                market="INVALIDCOIN",
                timeframe="1h",
                start="2023-01-01",
                end="2023-01-02",
                indicators=[],
                price_source=basic_price_source,
            )

    def test_stats_and_metrics(self, manager, basic_price_source):
        """Test statistiques : cache hits, appels, devices."""
        # Vider stats
        manager.clear_cache()
        initial_stats = manager.get_stats()

        # Quelques appels
        for i in range(3):
            manager.prepare_dataframe(
                market="BTCUSDT",
                timeframe="1h",
                start=f"2023-01-0{i+1}",
                end=f"2023-01-0{i+2}",
                indicators=[],
                price_source=basic_price_source,
                cache_ttl_sec=30,
            )

        final_stats = manager.get_stats()

        # Vérifications stats
        assert final_stats["prepare_calls"] >= 3
        assert final_stats["cache_size"] >= 0
        assert final_stats["total_latency_ms"] > 0
        assert "cache_hit_rate" in final_stats

    def test_integration_with_data_source(self):
        """Test intégration : TokenDiversityManager + TokenDiversityDataSource."""
        config = create_default_config()
        data_source = TokenDiversityDataSource(config)

        # Test symboles disponibles
        symbols = data_source.list_symbols(group="L1")
        assert len(symbols) > 0
        assert "BTCUSDT" in symbols or "BTC" in symbols

        # Test get_frame direct
        df = data_source.get_frame(
            symbol=symbols[0], timeframe="1h", end=datetime.now()
        )

        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)


@pytest.mark.benchmark
class TestTokenDiversityManagerBenchmarks:
    """Benchmarks de performance pour validation budgets."""

    def test_micro_benchmark_ohlcv_only(self, benchmark):
        """Micro-benchmark : OHLCV seul (sans indicateurs)."""
        manager = TokenDiversityManager()
        price_source = PriceSourceSpec(name="stub", params={})

        def run_ohlcv():
            return manager.prepare_dataframe(
                market="BTCUSDT",
                timeframe="1h",
                start="2023-01-01",
                end="2023-01-15",
                indicators=[],
                price_source=price_source,
            )

        result = benchmark(run_ohlcv)
        df, meta = result

        assert not df.empty
        assert meta["execution_time_ms"] > 0

    def test_micro_benchmark_with_indicators(self, benchmark):
        """Micro-benchmark : OHLCV + 3 indicateurs."""
        manager = TokenDiversityManager()
        price_source = PriceSourceSpec(name="stub", params={})
        indicators = [
            IndicatorSpec(name="rsi", params={"window": 14}),
            IndicatorSpec(name="bbands", params={"window": 20, "n_std": 2.0}),
            IndicatorSpec(name="atr", params={"window": 14}),
        ]

        def run_with_indicators():
            return manager.prepare_dataframe(
                market="ETHUSDT",
                timeframe="1h",
                start="2023-01-01",
                end="2023-02-01",
                indicators=indicators,
                price_source=price_source,
            )

        try:
            result = benchmark(run_with_indicators)
            df, meta = result

            assert not df.empty
            assert meta["indicators_count"] == len(indicators)

        except ImportError:
            pytest.skip("IndicatorBank required for benchmark")


if __name__ == "__main__":
    # Exécution directe pour développement
    print("🧪 Tests TokenDiversityManager Option B")
    print("=" * 50)

    # Tests smoke rapides
    manager = TokenDiversityManager()
    price_source = PriceSourceSpec(name="stub", params={})

    try:
        df, meta = manager.prepare_dataframe(
            market="BTCUSDT",
            timeframe="1h",
            start="2023-01-01",
            end="2023-01-08",
            indicators=[],
            price_source=price_source,
        )

        print(f"✅ Smoke test réussi : {len(df)} rows, {len(df.columns)} cols")
        print(f"   Colonnes : {list(df.columns)}")
        print(f"   Index : {df.index[0]} → {df.index[-1]}")
        print(f"   Métadonnées : {meta['execution_time_ms']:.1f}ms")

        # Test cache
        df2, meta2 = manager.prepare_dataframe(
            market="BTCUSDT",
            timeframe="1h",
            start="2023-01-01",
            end="2023-01-08",
            indicators=[],
            price_source=price_source,
        )

        print(f"✅ Cache test : hit={meta2['cache_hit']}")

        # Stats finales
        stats = manager.get_stats()
        print(
            f"✅ Stats : {stats['prepare_calls']} calls, hit_rate={stats['cache_hit_rate']:.2f}"
        )

        print("\n🎯 Tous les tests smoke réussis !")

    except Exception as e:
        print(f"❌ Erreur durant les tests : {e}")
        import traceback

        traceback.print_exc()

