"""
Tests unitaires pour TokenDiversityDataSource
"""

import pytest
from datetime import datetime
from pathlib import Path
import pandas as pd

from threadx.data.tokens import (
    TokenDiversityConfig,
    TokenDiversityDataSource,
    create_default_config,
)


class TestTokenDiversityConfig:
    """Tests pour TokenDiversityConfig"""

    def test_create_default_config(self):
        """Test création config par défaut"""
        config = create_default_config()

        assert config is not None
        assert len(config.groups) == 4  # L1, DeFi, L2, Stable
        assert len(config.symbols) > 0
        assert "1h" in config.supported_tf
        assert config.cache_dir == "./data/diversity_cache"

    def test_config_validation(self):
        """Test validation post_init"""
        # Config valide
        config = TokenDiversityConfig(
            groups={"L1": ["BTC", "ETH"]},
            symbols=["BTC", "ETH"],
            supported_tf=("1h", "4h"),
            cache_dir="./cache",
        )
        assert config is not None

    def test_config_immutable(self):
        """Test que config est frozen"""
        config = create_default_config()

        with pytest.raises(Exception):  # FrozenInstanceError
            config.symbols = []  # type: ignore


class TestTokenDiversityDataSource:
    """Tests pour TokenDiversityDataSource"""

    def test_init_provider(self):
        """Test initialisation provider"""
        config = create_default_config()
        provider = TokenDiversityDataSource(config)

        assert provider.config == config

    def test_list_groups(self):
        """Test list_groups()"""
        config = create_default_config()
        provider = TokenDiversityDataSource(config)

        groups = provider.list_groups()

        assert len(groups) == 4
        assert "L1" in groups
        assert "DeFi" in groups
        assert "L2" in groups
        assert "Stable" in groups

    def test_list_symbols_all(self):
        """Test list_symbols() sans groupe"""
        config = create_default_config()
        provider = TokenDiversityDataSource(config)

        symbols = provider.list_symbols()

        assert len(symbols) > 0
        assert "BTCUSDC" in symbols
        assert "ETHUSDC" in symbols

    def test_list_symbols_by_group(self):
        """Test list_symbols() avec groupe spécifique"""
        config = create_default_config()
        provider = TokenDiversityDataSource(config)

        l1_symbols = provider.list_symbols("L1")

        assert len(l1_symbols) > 0
        assert "BTCUSDC" in l1_symbols
        assert "ETHUSDC" in l1_symbols

    def test_list_symbols_unknown_group(self):
        """Test list_symbols() avec groupe inexistant"""
        config = create_default_config()
        provider = TokenDiversityDataSource(config)

        unknown_symbols = provider.list_symbols("UnknownGroup")

        assert unknown_symbols == []

    def test_validate_symbol(self):
        """Test validate_symbol()"""
        config = create_default_config()
        provider = TokenDiversityDataSource(config)

        assert provider.validate_symbol("BTCUSDC") is True
        assert provider.validate_symbol("ETHUSDC") is True
        assert provider.validate_symbol("INVALID") is False

    def test_validate_timeframe(self):
        """Test validate_timeframe()"""
        config = create_default_config()
        provider = TokenDiversityDataSource(config)

        assert provider.validate_timeframe("1h") is True
        assert provider.validate_timeframe("4h") is True
        assert provider.validate_timeframe("1d") is True
        assert provider.validate_timeframe("7d") is False


class TestFetchOHLCV:
    """Tests pour fetch_ohlcv() - nécessite données locales"""

    @pytest.fixture
    def provider(self):
        """Provider avec config par défaut"""
        config = create_default_config()
        return TokenDiversityDataSource(config)

    def test_fetch_ohlcv_invalid_symbol(self, provider):
        """Test fetch_ohlcv avec symbole invalide"""
        with pytest.raises(ValueError, match="non supporté"):
            provider.fetch_ohlcv("INVALID", "1h")

    def test_fetch_ohlcv_invalid_timeframe(self, provider):
        """Test fetch_ohlcv avec timeframe invalide"""
        with pytest.raises(ValueError, match="non supporté"):
            provider.fetch_ohlcv("BTCUSDC", "7d")

    @pytest.mark.skipif(
        not Path("./data/crypto_data_parquet/BTCUSDC_1h.parquet").exists(),
        reason="Données locales manquantes (BTCUSDC_1h.parquet)",
    )
    def test_fetch_ohlcv_parquet_success(self, provider):
        """Test fetch_ohlcv avec fichier Parquet existant"""
        df = provider.fetch_ohlcv("BTCUSDC", "1h", limit=100)

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 100
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    @pytest.mark.skipif(
        not Path("./data/crypto_data_parquet/BTCUSDC_1h.parquet").exists(),
        reason="Données locales manquantes",
    )
    def test_fetch_ohlcv_with_date_filter(self, provider):
        """Test fetch_ohlcv avec filtrage par date"""
        # Utiliser des dates UTC-aware
        start_date = pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime()
        end_date = pd.Timestamp("2025-10-01", tz="UTC").to_pydatetime()

        df = provider.fetch_ohlcv(
            "BTCUSDC",
            "1h",
            start_date=start_date,
            end_date=end_date,
            limit=500,
        )

        assert df is not None
        assert len(df) <= 500
        # Vérifier que les dates sont dans la plage
        assert df.index[0] >= start_date
        assert df.index[-1] <= end_date

    def test_fetch_ohlcv_file_not_found(self, provider):
        """Test fetch_ohlcv quand fichier absent"""
        # Créer config avec symbole sans données locales
        from threadx.data.tokens import (
            TokenDiversityConfig,
            TokenDiversityDataSource,
        )

        config = TokenDiversityConfig(
            groups={"Test": ["FAKEUSDC"]},
            symbols=["FAKEUSDC"],  # Symbole qui n'existe pas
            supported_tf=("1h",),
        )
        test_provider = TokenDiversityDataSource(config)

        with pytest.raises(FileNotFoundError, match="Aucune donnée trouvée"):
            test_provider.fetch_ohlcv("FAKEUSDC", "1h")


class TestIntegration:
    """Tests d'intégration complets"""

    @pytest.mark.skipif(
        not Path("./data/crypto_data_parquet").exists(),
        reason="Répertoire data/crypto_data_parquet manquant",
    )
    def test_full_workflow(self):
        """Test workflow complet : config → provider → fetch"""
        # 1. Créer config
        config = create_default_config()
        assert config is not None

        # 2. Créer provider
        provider = TokenDiversityDataSource(config)
        assert provider is not None

        # 3. Lister groupes et symboles
        groups = provider.list_groups()
        assert len(groups) > 0

        symbols = provider.list_symbols("L1")
        assert len(symbols) > 0

        # 4. Valider symboles
        first_symbol = symbols[0]
        assert provider.validate_symbol(first_symbol)

        # 5. (Optionnel) Fetch OHLCV si données disponibles
        # Testé séparément car dépend des données locales


# =============================================================================
# TESTS MANUELS (nécessitent données locales)
# =============================================================================


def manual_test_fetch_with_real_data():
    """
    Test manuel avec vraies données Parquet.

    Usage:
        python -c "from tests.test_token_diversity import manual_test_fetch_with_real_data; manual_test_fetch_with_real_data()"
    """
    print("\n" + "=" * 60)
    print("TEST MANUEL - fetch_ohlcv avec données réelles")
    print("=" * 60)

    # Setup
    config = create_default_config()
    provider = TokenDiversityDataSource(config)

    # Test 1: Chargement basique
    print("\n1️⃣ Test chargement BTCUSDC 1h (limit=100)")
    try:
        df = provider.fetch_ohlcv("BTCUSDC", "1h", limit=100)
        print(f"   ✅ Succès: {len(df)} lignes chargées")
        print(f"   Période: {df.index[0]} → {df.index[-1]}")
        print(f"   Colonnes: {list(df.columns)}")
        print(f"\n   Aperçu:\n{df.head(3)}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # Test 2: Filtrage par dates
    print("\n2️⃣ Test filtrage par dates")
    try:
        df = provider.fetch_ohlcv(
            "BTCUSDC",
            "1h",
            start_date=datetime(2025, 9, 1),
            end_date=datetime(2025, 10, 1),
            limit=200,
        )
        print(f"   ✅ Succès: {len(df)} lignes")
        print(f"   Période: {df.index[0]} → {df.index[-1]}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # Test 3: Fallback JSON
    print("\n3️⃣ Test fallback JSON (si Parquet absent)")
    try:
        # Chercher un symbole avec JSON mais pas Parquet
        df = provider.fetch_ohlcv("ETHUSDC", "4h", limit=50)
        print(f"   ✅ Succès: {len(df)} lignes")
    except FileNotFoundError:
        print("   ⚠️ Aucune donnée locale (Parquet/JSON)")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    print("\n" + "=" * 60)
    print("✅ Tests manuels terminés")
    print("=" * 60)


if __name__ == "__main__":
    # Lancer tests manuels si exécuté directement
    manual_test_fetch_with_real_data()

