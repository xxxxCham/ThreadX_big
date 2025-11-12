#!/usr/bin/env python3
"""
Data Inventory Tool - ThreadX
==============================

Analyse complète de la structure des données:
- OHLCV dans crypto_data_parquet/
- Indicateurs legacy dans indicateurs_data_parquet/
- Cache d'indicateurs dans indicators_cache/
- Cache général dans cache/

Génère un rapport JSON complet avec:
- Liste des fichiers
- Tailles
- Symboles/timeframes détectés
- Redondances potentielles
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from collections import defaultdict

# Chemins à analyser
BASE_DIR = Path(__file__).parent.parent
LOCATIONS = {
    "ohlcv": BASE_DIR / "src" / "threadx" / "data" / "crypto_data_parquet",
    "legacy_indicators": BASE_DIR
    / "src"
    / "threadx"
    / "data"
    / "indicateurs_data_parquet",
    "indicators_cache": BASE_DIR / "src" / "threadx" / "indicators_cache",
    "general_cache": BASE_DIR / "cache",
}


def get_size_mb(path: Path) -> float:
    """Calcule la taille en MB d'un fichier ou dossier."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)
    return 0.0


def analyze_ohlcv(location: Path) -> Dict[str, Any]:
    """Analyse les fichiers OHLCV bruts."""
    if not location.exists():
        return {"exists": False, "error": f"Location not found: {location}"}

    files = list(location.glob("*.parquet"))

    symbols = set()
    timeframes = set()
    file_details = []

    for f in files:
        # Parse filename: SYMBOL_TIMEFRAME.parquet
        parts = f.stem.split("_")
        if len(parts) >= 2:
            symbol = parts[0]
            timeframe = parts[1]
            symbols.add(symbol)
            timeframes.add(timeframe)

            # Lire les métadonnées
            try:
                df = pd.read_parquet(f)
                file_details.append(
                    {
                        "file": f.name,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "rows": len(df),
                        "start": str(df.index.min()) if len(df) > 0 else None,
                        "end": str(df.index.max()) if len(df) > 0 else None,
                        "size_mb": round(get_size_mb(f), 2),
                    }
                )
            except Exception as e:
                file_details.append({"file": f.name, "error": str(e)})

    return {
        "exists": True,
        "location": str(location),
        "total_files": len(files),
        "total_size_mb": round(get_size_mb(location), 2),
        "symbols": sorted(symbols),
        "symbol_count": len(symbols),
        "timeframes": sorted(timeframes),
        "timeframe_count": len(timeframes),
        "files": file_details,
    }


def analyze_legacy_indicators(location: Path) -> Dict[str, Any]:
    """Analyse les indicateurs legacy."""
    if not location.exists():
        return {"exists": False, "error": f"Location not found: {location}"}

    # Structure attendue: indicateurs_data_parquet/TOKEN/TF/*.parquet
    files = list(location.rglob("*.parquet"))

    tokens = set()
    timeframes_by_token = defaultdict(set)

    file_details = []
    for f in files:
        # Parse structure: indicateurs_data_parquet/TOKEN/TF/indicator.parquet
        parts = f.parts
        if len(parts) >= 3:
            token = parts[-3]
            tf = parts[-2]
            tokens.add(token)
            timeframes_by_token[token].add(tf)

            file_details.append(
                {
                    "file": f.name,
                    "path": str(f.relative_to(location)),
                    "token": token,
                    "timeframe": tf,
                    "size_mb": round(get_size_mb(f), 2),
                }
            )

    return {
        "exists": True,
        "location": str(location),
        "total_files": len(files),
        "total_size_mb": round(get_size_mb(location), 2),
        "tokens": sorted(tokens),
        "token_count": len(tokens),
        "timeframes_per_token": {
            token: sorted(tfs) for token, tfs in timeframes_by_token.items()
        },
        "files": file_details[:20],  # Limite à 20 pour lisibilité
    }


def analyze_indicators_cache(location: Path) -> Dict[str, Any]:
    """Analyse le cache d'indicateurs."""
    if not location.exists():
        return {"exists": False, "error": f"Location not found: {location}"}

    # Structure: indicators_cache/bollinger/*.parquet, indicators_cache/atr/*.parquet
    indicator_types = {}

    for indicator_dir in location.iterdir():
        if indicator_dir.is_dir() and indicator_dir.name != "registry":
            files = list(indicator_dir.glob("*.parquet"))

            # Parse filenames: bollinger_BTCUSDC_15m_{params_hash}_{data_hash}.parquet
            symbols = set()
            timeframes = set()

            for f in files:
                parts = f.stem.split("_")
                if len(parts) >= 3:
                    # bollinger, BTCUSDC, 15m, ...
                    if len(parts) > 1:
                        symbols.add(parts[1])
                    if len(parts) > 2:
                        timeframes.add(parts[2])

            indicator_types[indicator_dir.name] = {
                "files": len(files),
                "size_mb": round(get_size_mb(indicator_dir), 2),
                "symbols": sorted(symbols),
                "timeframes": sorted(timeframes),
            }

    # Registry
    registry_files = (
        list((location / "registry").glob("*.parquet"))
        if (location / "registry").exists()
        else []
    )

    return {
        "exists": True,
        "location": str(location),
        "total_files": sum(ind["files"] for ind in indicator_types.values())
        + len(registry_files),
        "total_size_mb": round(get_size_mb(location), 2),
        "indicator_types": indicator_types,
        "registry_files": [f.name for f in registry_files],
    }


def analyze_general_cache(location: Path) -> Dict[str, Any]:
    """Analyse le cache général."""
    if not location.exists():
        return {"exists": False, "error": f"Location not found: {location}"}

    subdirs = {}
    for subdir in location.iterdir():
        if subdir.is_dir():
            files = list(subdir.rglob("*"))
            subdirs[subdir.name] = {
                "files": len([f for f in files if f.is_file()]),
                "size_mb": round(get_size_mb(subdir), 2),
            }

    return {
        "exists": True,
        "location": str(location),
        "total_size_mb": round(get_size_mb(location), 2),
        "subdirectories": subdirs,
    }


def generate_report() -> Dict[str, Any]:
    """Génère le rapport complet d'inventaire."""
    report = {"timestamp": pd.Timestamp.now().isoformat(), "locations": {}}

    for name, path in LOCATIONS.items():
        print(f"Analysing {name} at {path}...")

        if name == "ohlcv":
            report["locations"][name] = analyze_ohlcv(path)
        elif name == "legacy_indicators":
            report["locations"][name] = analyze_legacy_indicators(path)
        elif name == "indicators_cache":
            report["locations"][name] = analyze_indicators_cache(path)
        elif name == "general_cache":
            report["locations"][name] = analyze_general_cache(path)

    # Résumé global
    total_size = sum(
        loc.get("total_size_mb", 0)
        for loc in report["locations"].values()
        if loc.get("exists", False)
    )

    report["summary"] = {
        "total_size_mb": round(total_size, 2),
        "total_size_gb": round(total_size / 1024, 2),
        "locations_analyzed": len(
            [loc for loc in report["locations"].values() if loc.get("exists")]
        ),
    }

    return report


def main():
    """Point d'entrée principal."""
    print("=== ThreadX Data Inventory ===\n")

    report = generate_report()

    # Sauvegarde JSON
    output_file = BASE_DIR / "DATA_INVENTORY.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Rapport sauvegardé: {output_file}")

    # Affichage résumé
    print(f"\n📊 RÉSUMÉ:")
    print(
        f"   Total: {report['summary']['total_size_mb']:.2f} MB ({report['summary']['total_size_gb']:.2f} GB)"
    )
    print(f"   Locations: {report['summary']['locations_analyzed']}/4\n")

    for name, data in report["locations"].items():
        if data.get("exists"):
            print(f"   {name}:")
            print(f"      Files: {data.get('total_files', 'N/A')}")
            print(f"      Size: {data.get('total_size_mb', 0):.2f} MB")
            if "symbol_count" in data:
                print(f"      Symbols: {data['symbol_count']}")
            if "timeframe_count" in data:
                print(f"      Timeframes: {data['timeframe_count']}")
        else:
            print(f"   {name}: ❌ NOT FOUND")


if __name__ == "__main__":
    main()
