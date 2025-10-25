"""
Tests complets pour le contrat UDFI 
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from threadx.data.udfi_contract import (
    assert_udfi, apply_column_map, normalize_dtypes, enforce_index_rules,
    UDFIError, UDFIIndexError, UDFITypeError, UDFIColumnError, UDFIIntegrityError,
    REQUIRED_COLS, CRITICAL_COLS, EXPECTED_DTYPES
)

class TestUDFIContract:
    """Suite de tests pour le contrat UDFI."""

    def test_valid_udfi_dataframe(self):
        """Test avec DataFrame parfaitement conforme UDFI."""

        # Données valides
        timestamps = pd.date_range('2024-01-01', periods=5, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 5,
            'open': [47000.0, 47100.0, 47050.0, 47200.0, 47150.0],
            'high': [47200.0, 47300.0, 47250.0, 47400.0, 47350.0],
            'low': [46950.0, 47000.0, 47000.0, 47100.0, 47050.0],
            'close': [47100.0, 47050.0, 47200.0, 47150.0, 47300.0],
            'volume': [12.5, 8.3, 15.7, 22.1, 18.9]
        }, index=timestamps)

        # Validation stricte ne doit pas lever d'exception
        assert_udfi(df, strict=True)

    def test_missing_required_columns(self):
        """Test avec colonnes obligatoires manquantes."""

        timestamps = pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 3,
            'open': [47000.0, 47100.0, 47050.0],
            # 'high' manquant - doit lever UDFIColumnError
            'low': [46950.0, 47000.0, 47000.0],
            'close': [47100.0, 47050.0, 47200.0],
            'volume': [12.5, 8.3, 15.7]
        }, index=timestamps)

        with pytest.raises(UDFIColumnError, match="Colonnes manquantes"):
            assert_udfi(df)

    def test_non_utc_index(self):
        """Test avec index non-UTC."""

        # Index avec timezone Europe/Paris
        timestamps = pd.date_range('2024-01-01', periods=3, freq='1h', tz='Europe/Paris')
        df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 3,
            'open': [47000.0, 47100.0, 47050.0],
            'high': [47200.0, 47300.0, 47250.0],
            'low': [46950.0, 47000.0, 47000.0],
            'close': [47100.0, 47050.0, 47200.0],
            'volume': [12.5, 8.3, 15.7]
        }, index=timestamps)

        with pytest.raises(UDFIIndexError, match="Index non-UTC"):
            assert_udfi(df)

    def test_unsorted_index(self):
        """Test avec index non trié."""

        # Timestamps dans le désordre
        timestamps = pd.to_datetime([
            '2024-01-01 10:00:00',
            '2024-01-01 08:00:00',  # Plus ancien
            '2024-01-01 12:00:00'
        ], utc=True)

        df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 3,
            'open': [47000.0, 47100.0, 47050.0],
            'high': [47200.0, 47300.0, 47250.0],
            'low': [46950.0, 47000.0, 47000.0],
            'close': [47100.0, 47050.0, 47200.0],
            'volume': [12.5, 8.3, 15.7]
        }, index=timestamps)

        with pytest.raises(UDFIIndexError, match="Index non trié"):
            assert_udfi(df)

    def test_duplicate_index(self):
        """Test avec doublons dans l'index."""

        # Même timestamp répété
        timestamp = pd.Timestamp('2024-01-01 10:00:00', tz='UTC')
        timestamps = [timestamp, timestamp, timestamp]

        df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 3,
            'open': [47000.0, 47100.0, 47050.0],
            'high': [47200.0, 47300.0, 47250.0],
            'low': [46950.0, 47000.0, 47000.0],
            'close': [47100.0, 47050.0, 47200.0],
            'volume': [12.5, 8.3, 15.7]
        }, index=timestamps)

        with pytest.raises(UDFIIndexError, match="Index avec doublons"):
            assert_udfi(df)

    def test_critical_nan_values(self):
        """Test avec valeurs NaN sur colonnes critiques."""

        timestamps = pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 3,
            'open': [47000.0, np.nan, 47050.0],  # NaN sur colonne critique
            'high': [47200.0, 47300.0, 47250.0],
            'low': [46950.0, 47000.0, 47000.0],
            'close': [47100.0, 47050.0, 47200.0],
            'volume': [12.5, 8.3, 15.7]
        }, index=timestamps)

        with pytest.raises(UDFIIntegrityError, match="NaN interdits sur open"):
            assert_udfi(df)

    def test_ohlc_integrity_violation(self):
        """Test avec violation de l'intégrité OHLC."""

        timestamps = pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'symbol': ['BTCUSDT'] * 3,
            'open': [47000.0, 47100.0, 47050.0],
            'high': [46900.0, 47300.0, 47250.0],  # high < open (violation)
            'low': [46950.0, 47000.0, 47000.0],
            'close': [47100.0, 47050.0, 47200.0],
            'volume': [12.5, 8.3, 15.7]
        }, index=timestamps)

        with pytest.raises(UDFIIntegrityError, match="Violation high"):
            assert_udfi(df, strict=True)

    def test_invalid_column_names(self):
        """Test avec noms de colonnes non conformes (snake_case)."""

        timestamps = pd.date_range('2024-01-01', periods=3, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'Symbol': ['BTCUSDT'] * 3,  # PascalCase invalide
            'open_Price': [47000.0, 47100.0, 47050.0],  # camelCase invalide
            'high': [47200.0, 47300.0, 47250.0],
            'low': [46950.0, 47000.0, 47000.0],
            'close': [47100.0, 47050.0, 47200.0],
            'volume': [12.5, 8.3, 15.7]
        }, index=timestamps)

        with pytest.raises(UDFIColumnError, match="Nommage invalide"):
            assert_udfi(df)

class TestUDFIHelpers:
    """Tests pour les fonctions helper du contrat UDFI."""

    def test_apply_column_map(self):
        """Test du mapping de colonnes."""

        df = pd.DataFrame({
            'ts': [1640995200, 1640995260, 1640995320],
            'sym': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
            'o': [47000.0, 3800.0, 1.2],
            'h': [47200.0, 3850.0, 1.25],
            'l': [46950.0, 3750.0, 1.18],
            'c': [47100.0, 3820.0, 1.22],
            'v': [12.5, 45.2, 150000.0]
        })

        mapping = {
            'ts': 'timestamp',
            'sym': 'symbol',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }

        result = apply_column_map(df, mapping)

        # Vérification des colonnes renommées
        expected_cols = set(mapping.values())
        assert set(result.columns) == expected_cols

        # Vérification des données préservées
        assert len(result) == len(df)
        assert result['symbol'].iloc[0] == 'BTCUSDT'

    def test_normalize_dtypes(self):
        """Test de normalisation des types."""

        df = pd.DataFrame({
            'timestamp': ['1640995200', '1640995260', '1640995320'],  # string
            'symbol': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
            'open': ['47000.0', '3800.0', '1.2'],  # string → float64
            'volume': [12, 45, 150000]  # int → float64
        })

        dtypes_in = {
            'timestamp': 'string',
            'open': 'string',
            'volume': 'int64'
        }

        dtypes_out = {
            'timestamp': 'int64',
            'symbol': 'string',
            'open': 'float64',
            'volume': 'float64'
        }

        result = normalize_dtypes(df, dtypes_in, dtypes_out)

        # Vérification des types convertis
        assert result['timestamp'].dtype == 'int64'
        assert result['open'].dtype == 'float64'
        assert result['volume'].dtype == 'float64'
        assert result['symbol'].dtype == 'string'

    def test_enforce_index_rules(self):
        """Test d'établissement des règles d'index."""

        df = pd.DataFrame({
            'timestamp': [1640995320, 1640995200, 1640995260, 1640995260],  # Désordre + doublon
            'symbol': ['BTCUSDT'] * 4,
            'open': [47050.0, 47000.0, 47100.0, 47100.0],
            'high': [47250.0, 47200.0, 47300.0, 47300.0],
            'low': [47000.0, 46950.0, 47000.0, 47000.0],
            'close': [47200.0, 47100.0, 47050.0, 47050.0],
            'volume': [15.7, 12.5, 8.3, 8.3]
        })

        result = enforce_index_rules(df, 'timestamp', tz='UTC')

        # Index doit être DatetimeIndex UTC trié
        assert isinstance(result.index, pd.DatetimeIndex)
        assert str(result.index.tz) == 'UTC'
        assert result.index.is_monotonic_increasing
        assert result.index.is_unique

        # Doublon supprimé (keep='last')
        assert len(result) == 3

        # Colonne timestamp supprimée
        assert 'timestamp' not in result.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
