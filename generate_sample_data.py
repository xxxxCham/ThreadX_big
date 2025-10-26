#!/usr/bin/env python3
"""
Script pour générer des données OHLCV d'exemple pour ThreadX.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Configuration
DATA_DIR = Path("/home/user/ThreadX_big/data")
CRYPTO_JSON_DIR = DATA_DIR / "crypto_data_json"
CRYPTO_PARQUET_DIR = DATA_DIR / "crypto_data_parquet"

# Symboles et timeframes
SYMBOLS = ["BTCUSDC", "ETHUSDC", "ADAUSDC"]
TIMEFRAMES = ["1h", "5m", "15m", "30m"]  # Ajoutez d'autres si nécessaire

# Prix de départ approximatifs (pour réalisme)
BASE_PRICES = {
    "BTCUSDC": 45000.0,
    "ETHUSDC": 2500.0,
    "ADAUSDC": 0.50,
}

def generate_ohlcv(symbol: str, timeframe: str, days: int = 30) -> pd.DataFrame:
    """Génère des données OHLCV aléatoires mais réalistes."""

    # Calculer le nombre de bougies selon le timeframe
    if timeframe == "1h":
        periods = days * 24
        freq = "1H"
    elif timeframe == "5m":
        periods = days * 24 * 12
        freq = "5min"
    elif timeframe == "15m":
        periods = days * 24 * 4
        freq = "15min"
    elif timeframe == "30m":
        periods = days * 24 * 2
        freq = "30min"
    else:
        periods = days * 24
        freq = "1H"

    # Créer les timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, periods=periods, tz='UTC')

    # Prix de base
    base_price = BASE_PRICES.get(symbol, 100.0)

    # Générer des variations de prix réalistes
    np.random.seed(hash(symbol) % 2**32)  # Seed basé sur le symbol pour cohérence

    # Random walk pour le prix de clôture
    returns = np.random.normal(0, 0.02, periods)  # Rendements avec volatilité de 2%
    price_multipliers = np.exp(np.cumsum(returns))
    close_prices = base_price * price_multipliers

    # Générer OHLC basé sur close
    data = []
    for i, close in enumerate(close_prices):
        # Variation intraday
        high_offset = abs(np.random.normal(0, 0.005)) * close
        low_offset = abs(np.random.normal(0, 0.005)) * close

        high = close + high_offset
        low = close - low_offset

        # Open est entre high et low
        open_price = np.random.uniform(low, high)

        # Volume aléatoire mais réaliste
        volume = np.random.uniform(100, 10000) * (base_price / 100)

        data.append({
            'time': timestamps[i],
            'open': round(open_price, 4),
            'high': round(high, 4),
            'low': round(low, 4),
            'close': round(close, 4),
            'volume': round(volume, 2)
        })

    df = pd.DataFrame(data)
    df = df.set_index('time')

    return df

def save_data():
    """Génère et sauvegarde toutes les données."""
    print("🚀 Génération des données OHLCV d'exemple...")

    for symbol in SYMBOLS:
        for timeframe in TIMEFRAMES:
            print(f"   Génération {symbol}_{timeframe}...")

            # Générer les données
            df = generate_ohlcv(symbol, timeframe, days=30)

            # Sauvegarder en Parquet (dans crypto_data_json car c'est ce que le code cherche)
            parquet_path = CRYPTO_JSON_DIR / f"{symbol}_{timeframe}.parquet"
            df.to_parquet(parquet_path)

            # Optionnel: sauvegarder aussi en JSON
            json_path = CRYPTO_JSON_DIR / f"{symbol}_{timeframe}.json"
            df.reset_index().to_json(json_path, orient='records', date_format='iso')

            print(f"   ✅ Sauvegardé: {parquet_path.name} ({len(df)} lignes)")

    print(f"\n✅ Données générées avec succès dans {CRYPTO_JSON_DIR}")
    print(f"   - Symboles: {', '.join(SYMBOLS)}")
    print(f"   - Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"   - Format: Parquet + JSON")

if __name__ == "__main__":
    save_data()
