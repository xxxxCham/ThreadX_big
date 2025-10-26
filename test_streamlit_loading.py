#!/usr/bin/env python3
"""
Test de simulation exacte du chargement Streamlit.
Reproduit exactement ce que fait page_selection_token.py
"""

import sys
from pathlib import Path
from datetime import date

# Simuler l'environnement Streamlit
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 80)
print("SIMULATION EXACTE DU CHARGEMENT STREAMLIT")
print("=" * 80)

# Import exact comme dans page_selection_token.py
from threadx.data_access import DATA_DIR, discover_tokens_and_timeframes, load_ohlcv

print(f"\n📁 DATA_DIR: {DATA_DIR}")

# Découverte des tokens (comme dans Streamlit)
print("\n🔍 Découverte des tokens...")
tokens, timeframes = discover_tokens_and_timeframes()

print(f"   Tokens trouvés: {tokens}")
print(f"   Timeframes trouvés: {timeframes}")

if not tokens:
    print("\n❌ ERREUR: Aucun token trouvé !")
    print("   C'est exactement le problème que vous voyez dans Streamlit")
    sys.exit(1)

# Simuler la sélection utilisateur
symbol = "BTC"  # Ce que vous sélectionnez dans le dropdown
timeframe = "1h"
start_date = date(2025, 1, 1)
end_date = date(2025, 1, 15)

print(f"\n📊 Test de chargement avec:")
print(f"   Symbol: {symbol}")
print(f"   Timeframe: {timeframe}")
print(f"   Start: {start_date}")
print(f"   End: {end_date}")

# Reproduire exactement le code de page_selection_token.py ligne 110
try:
    preview = load_ohlcv(symbol, timeframe, start=start_date, end=end_date)

    print(f"\n✅ SUCCÈS ! Données chargées:")
    print(f"   Lignes: {len(preview)}")
    print(f"   Colonnes: {list(preview.columns)}")
    print(f"   Période: {preview.index.min()} -> {preview.index.max()}")
    print("\n   Aperçu:")
    print(preview.head(3).to_string())

except FileNotFoundError as exc:
    print(f"\n❌ ERREUR FileNotFoundError:")
    print(f"   {exc}")
    print("\n   C'est exactement l'erreur que vous voyez dans Streamlit !")

except Exception as exc:
    print(f"\n❌ ERREUR:")
    print(f"   {exc}")

print("\n" + "=" * 80)
