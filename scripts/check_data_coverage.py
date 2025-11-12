#!/usr/bin/env python3
"""
Data Coverage Check - ThreadX
==============================

Vérifie la couverture des données OHLCV:
- Symboles complets (tous les timeframes présents)
- Symboles partiels (timeframes manquants)
- Gaps temporels dans les données
- Recommandations de téléchargement

Usage:
    python scripts/check_data_coverage.py
    python scripts/check_data_coverage.py --symbol BTCUSDC
    python scripts/check_data_coverage.py --min-days 365
"""

import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "src" / "threadx" / "data" / "crypto_data_parquet"

# Timeframes attendus (du plus granulaire au moins granulaire)
EXPECTED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]

# Jours minimum attendus par timeframe
MIN_DAYS = {
    "1m": 30,  # 1 mois pour 1m (lourd)
    "5m": 90,  # 3 mois pour 5m
    "15m": 180,  # 6 mois pour 15m
    "1h": 365,  # 1 an pour 1h
    "4h": 730,  # 2 ans pour 4h
}


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Vérifie la couverture des données OHLCV"
    )
    parser.add_argument("--symbol", type=str, help="Vérifier un symbole spécifique")
    parser.add_argument("--min-days", type=int, help="Jours minimum requis (override)")
    parser.add_argument(
        "--show-gaps", action="store_true", help="Afficher les gaps temporels"
    )
    parser.add_argument("--output", type=str, help="Fichier de sortie JSON")
    return parser.parse_args()


def get_all_symbols() -> List[str]:
    """Récupère tous les symboles disponibles."""
    if not DATA_DIR.exists():
        return []

    symbols = set()
    for file in DATA_DIR.glob("*.parquet"):
        parts = file.stem.split("_")
        if len(parts) >= 1:
            symbols.add(parts[0])

    return sorted(symbols)


def check_file_coverage(
    symbol: str, timeframe: str, min_days: Optional[int] = None
) -> Dict:
    """Vérifie la couverture d'un fichier OHLCV."""
    file_path = DATA_DIR / f"{symbol}_{timeframe}.parquet"

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "exists": file_path.exists(),
        "file_path": str(file_path),
    }

    if not file_path.exists():
        result["status"] = "missing"
        return result

    try:
        # Charger les données
        df = pd.read_parquet(file_path)

        if len(df) == 0:
            result["status"] = "empty"
            return result

        # Métadonnées temporelles
        start_date = pd.Timestamp(df.index.min())
        end_date = pd.Timestamp(df.index.max())
        days_coverage = (end_date - start_date).days

        result.update(
            {
                "rows": len(df),
                "start_date": str(start_date),
                "end_date": str(end_date),
                "days_coverage": days_coverage,
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
            }
        )

        # Vérifier la couverture minimum
        expected_days = min_days if min_days else MIN_DAYS.get(timeframe, 365)

        if days_coverage >= expected_days:
            result["status"] = "complete"
        elif days_coverage >= expected_days * 0.5:
            result["status"] = "partial"
            result["missing_days"] = expected_days - days_coverage
        else:
            result["status"] = "insufficient"
            result["missing_days"] = expected_days - days_coverage

        # Détecter les gaps (optionnel, coûteux en calcul)
        # On vérifie juste le ratio rows vs timespan attendu
        expected_rows = calculate_expected_rows(timeframe, days_coverage)
        coverage_ratio = len(df) / expected_rows if expected_rows > 0 else 0

        result["coverage_ratio"] = round(coverage_ratio, 2)

        if coverage_ratio < 0.95:
            result["has_gaps"] = True
            result["gap_warning"] = (
                f"Couverture: {coverage_ratio*100:.1f}% (attendu: 95%+)"
            )
        else:
            result["has_gaps"] = False

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def calculate_expected_rows(timeframe: str, days: int) -> int:
    """Calcule le nombre de lignes attendues pour un timeframe et une durée."""
    minutes_per_candle = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}

    candles_per_day = (24 * 60) / minutes_per_candle.get(timeframe, 60)
    return int(candles_per_day * days)


def check_symbol_coverage(symbol: str, min_days: Optional[int] = None) -> Dict:
    """Vérifie la couverture complète d'un symbole."""
    coverage = {"symbol": symbol, "timeframes": {}}

    complete_count = 0
    partial_count = 0
    missing_count = 0

    for tf in EXPECTED_TIMEFRAMES:
        result = check_file_coverage(symbol, tf, min_days)
        coverage["timeframes"][tf] = result

        if result["status"] == "complete":
            complete_count += 1
        elif result["status"] in ["partial", "insufficient"]:
            partial_count += 1
        else:
            missing_count += 1

    # Classification globale
    if complete_count == len(EXPECTED_TIMEFRAMES):
        coverage["overall_status"] = "✅ COMPLET"
    elif complete_count + partial_count == len(EXPECTED_TIMEFRAMES):
        coverage["overall_status"] = "⚠️  PARTIEL"
    elif missing_count == len(EXPECTED_TIMEFRAMES):
        coverage["overall_status"] = "❌ MANQUANT"
    else:
        coverage["overall_status"] = "⚠️  INCOMPLET"

    coverage["stats"] = {
        "complete": complete_count,
        "partial": partial_count,
        "missing": missing_count,
    }

    return coverage


def generate_full_report(min_days: Optional[int] = None) -> Dict:
    """Génère le rapport complet de couverture."""
    symbols = get_all_symbols()

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_symbols": len(symbols),
        "expected_timeframes": EXPECTED_TIMEFRAMES,
        "symbols": {},
        "summary": {"complete": [], "partial": [], "incomplete": [], "missing": []},
    }

    for symbol in symbols:
        coverage = check_symbol_coverage(symbol, min_days)
        report["symbols"][symbol] = coverage

        # Classification
        status = coverage["overall_status"]
        if "COMPLET" in status:
            report["summary"]["complete"].append(symbol)
        elif "PARTIEL" in status:
            report["summary"]["partial"].append(symbol)
        elif "MANQUANT" in status:
            report["summary"]["missing"].append(symbol)
        else:
            report["summary"]["incomplete"].append(symbol)

    return report


def print_summary(report: Dict):
    """Affiche le résumé du rapport."""
    print("=" * 60)
    print("RÉSUMÉ DE LA COUVERTURE DES DONNÉES")
    print("=" * 60)
    print(f"\nTotal symboles: {report['total_symbols']}")
    print(f"Timeframes attendus: {', '.join(report['expected_timeframes'])}\n")

    print(f"✅ Complets:   {len(report['summary']['complete']):>4} symboles")
    print(f"⚠️  Partiels:   {len(report['summary']['partial']):>4} symboles")
    print(f"⚠️  Incomplets: {len(report['summary']['incomplete']):>4} symboles")
    print(f"❌ Manquants:  {len(report['summary']['missing']):>4} symboles\n")

    # Détails des symboles partiels (top 10)
    if report["summary"]["partial"]:
        print("\nTOP 10 SYMBOLES PARTIELS (timeframes manquants):")
        for symbol in report["summary"]["partial"][:10]:
            cov = report["symbols"][symbol]
            missing_tfs = [
                tf
                for tf, data in cov["timeframes"].items()
                if data["status"] in ["missing", "insufficient"]
            ]
            print(f"  {symbol:15} → Manque: {', '.join(missing_tfs)}")

    # Détails des symboles incomplets (top 10)
    if report["summary"]["incomplete"]:
        print("\nTOP 10 SYMBOLES INCOMPLETS:")
        for symbol in report["summary"]["incomplete"][:10]:
            cov = report["symbols"][symbol]
            stats = cov["stats"]
            print(
                f"  {symbol:15} → Complet: {stats['complete']}/{len(EXPECTED_TIMEFRAMES)}"
            )


def print_symbol_detail(symbol: str, coverage: Dict):
    """Affiche le détail d'un symbole."""
    print(f"\n{'=' * 60}")
    print(f"COUVERTURE: {symbol}")
    print(f"{'=' * 60}\n")
    print(f"Statut global: {coverage['overall_status']}\n")

    for tf in EXPECTED_TIMEFRAMES:
        data = coverage["timeframes"][tf]

        status_icon = {
            "complete": "✅",
            "partial": "⚠️ ",
            "insufficient": "⚠️ ",
            "missing": "❌",
            "empty": "❌",
            "error": "❌",
        }.get(data["status"], "❓")

        print(f"{status_icon} {tf:>4} : ", end="")

        if data["status"] == "missing":
            print("FICHIER MANQUANT")
        elif data["status"] == "error":
            print(f"ERREUR - {data.get('error', 'Unknown')}")
        elif data["status"] == "empty":
            print("FICHIER VIDE")
        else:
            print(
                f"{data['rows']:,} lignes | {data['days_coverage']} jours | {data['size_mb']} MB"
            )
            print(
                f"         Période: {data['start_date'][:10]} → {data['end_date'][:10]}"
            )

            if data.get("has_gaps"):
                print(f"         ⚠️  {data['gap_warning']}")

            if data.get("missing_days", 0) > 0:
                print(f"         ⚠️  Manque {data['missing_days']} jours")


def main():
    """Point d'entrée principal."""
    args = parse_args()

    if not DATA_DIR.exists():
        print(f"❌ Répertoire de données introuvable: {DATA_DIR}")
        return

    # Vérification d'un symbole spécifique
    if args.symbol:
        coverage = check_symbol_coverage(args.symbol, args.min_days)
        print_symbol_detail(args.symbol, coverage)
        return

    # Rapport complet
    print("🔍 Analyse de la couverture des données...\n")
    report = generate_full_report(args.min_days)

    print_summary(report)

    # Sauvegarde optionnelle
    if args.output:
        import json

        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Rapport détaillé sauvegardé: {output_path}")


if __name__ == "__main__":
    main()
