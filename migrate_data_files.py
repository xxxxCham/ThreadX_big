#!/usr/bin/env python3
"""
Outil de migration et validation des fichiers de données crypto.

Usage:
    python migrate_data_files.py --validate    # Vérifier les fichiers
    python migrate_data_files.py --migrate     # Déplacer les fichiers au bon endroit
    python migrate_data_files.py --fix-names   # Créer des alias BTC->BTCUSDC
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import shutil

# Chemins
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "src" / "threadx" / "data"
PARQUET_DIR = DATA_DIR / "crypto_data_parquet"
JSON_DIR = DATA_DIR / "crypto_data_json"

# Mapping symboles raccourcis -> symboles complets
SYMBOL_MAPPING = {
    "BTC": "BTCUSDC",
    "ETH": "ETHUSDC",
    "ADA": "ADAUSDC",
    "SOL": "SOLUSDC",
    "BNB": "BNBUSDC",
}


def validate_ohlcv_file(file_path: Path) -> dict:
    """Valide qu'un fichier a la structure OHLCV correcte."""
    result = {"valid": False, "errors": [], "warnings": [], "info": {}}

    try:
        # Lire le fichier
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".json":
            df = pd.read_json(file_path)
        elif file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            result["errors"].append(f"Format non supporté: {file_path.suffix}")
            return result

        # Colonnes attendues (insensible à la casse)
        required_cols = ["open", "high", "low", "close", "volume"]
        df_cols_lower = [c.lower() for c in df.columns]

        # Vérifier colonnes OHLCV
        missing_cols = [c for c in required_cols if c not in df_cols_lower]
        if missing_cols:
            result["errors"].append(f"Colonnes manquantes: {missing_cols}")
            return result

        # Vérifier l'index (doit être datetime ou contenir une colonne 'time'/'timestamp')
        has_time_index = df.index.dtype == 'datetime64[ns]' or df.index.dtype == 'datetime64[ns, UTC]'
        has_time_column = 'time' in df_cols_lower or 'timestamp' in df_cols_lower

        if not has_time_index and not has_time_column:
            result["warnings"].append("Pas d'index datetime ni de colonne 'time'/'timestamp'")

        # Vérifier types numériques
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 5:
            result["warnings"].append(f"Seulement {len(numeric_cols)} colonnes numériques trouvées")

        # Infos
        result["info"] = {
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "first_date": str(df.index[0]) if has_time_index else "N/A",
            "last_date": str(df.index[-1]) if has_time_index else "N/A",
        }

        result["valid"] = len(result["errors"]) == 0
        return result

    except Exception as e:
        result["errors"].append(f"Erreur lecture fichier: {e}")
        return result


def find_data_files(search_dir: Path) -> list:
    """Trouve tous les fichiers de données (parquet, json, csv)."""
    files = []
    for ext in [".parquet", ".json", ".csv"]:
        files.extend(search_dir.glob(f"*{ext}"))
    return sorted(files)


def validate_all_files():
    """Valide tous les fichiers de données."""
    print("=" * 80)
    print("VALIDATION DES FICHIERS DE DONNÉES")
    print("=" * 80)

    # Chercher dans le dossier racine data/
    files_in_root = find_data_files(DATA_DIR)

    # Chercher dans les sous-dossiers
    files_in_parquet = find_data_files(PARQUET_DIR) if PARQUET_DIR.exists() else []
    files_in_json = find_data_files(JSON_DIR) if JSON_DIR.exists() else []

    print(f"\n📁 Fichiers trouvés:")
    print(f"   - Racine data/           : {len(files_in_root)} fichiers")
    print(f"   - crypto_data_parquet/   : {len(files_in_parquet)} fichiers")
    print(f"   - crypto_data_json/      : {len(files_in_json)} fichiers")

    if files_in_root:
        print(f"\n⚠️  ATTENTION: {len(files_in_root)} fichiers dans la racine data/")
        print("   Ces fichiers devraient être dans crypto_data_parquet/ ou crypto_data_json/")
        print("   Utilisez --migrate pour les déplacer automatiquement")
        print()
        for f in files_in_root:
            print(f"   - {f.name}")

    # Valider chaque fichier
    all_files = files_in_root + files_in_parquet + files_in_json

    if not all_files:
        print("\n❌ Aucun fichier de données trouvé !")
        return

    print(f"\n🔍 Validation de {len(all_files)} fichiers...\n")

    valid_count = 0
    invalid_count = 0

    for file_path in all_files:
        result = validate_ohlcv_file(file_path)

        status = "✅" if result["valid"] else "❌"
        location = "racine" if file_path.parent == DATA_DIR else file_path.parent.name

        print(f"{status} {file_path.name} ({location})")

        if result["errors"]:
            for err in result["errors"]:
                print(f"      ❌ {err}")

        if result["warnings"]:
            for warn in result["warnings"]:
                print(f"      ⚠️  {warn}")

        if result["valid"]:
            info = result["info"]
            print(f"      ℹ️  {info['rows']} lignes, {len(info['columns'])} colonnes")
            valid_count += 1
        else:
            invalid_count += 1

        print()

    print("=" * 80)
    print(f"✅ Valides  : {valid_count}")
    print(f"❌ Invalides: {invalid_count}")
    print("=" * 80)


def migrate_files():
    """Déplace les fichiers de la racine vers les bons sous-dossiers."""
    print("=" * 80)
    print("MIGRATION DES FICHIERS")
    print("=" * 80)

    # Créer les dossiers si nécessaire
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    files_in_root = find_data_files(DATA_DIR)

    if not files_in_root:
        print("\n✅ Aucun fichier à migrer (tous déjà dans les bons dossiers)")
        return

    print(f"\n📦 Migration de {len(files_in_root)} fichiers...\n")

    for file_path in files_in_root:
        if file_path.suffix == ".parquet":
            dest_dir = PARQUET_DIR
        elif file_path.suffix == ".json":
            dest_dir = JSON_DIR
        else:
            dest_dir = JSON_DIR  # CSV → JSON dir par défaut

        dest_path = dest_dir / file_path.name

        if dest_path.exists():
            print(f"⏭️  {file_path.name} → Déjà existant dans {dest_dir.name}, ignoré")
        else:
            shutil.move(str(file_path), str(dest_path))
            print(f"✅ {file_path.name} → {dest_dir.name}/")

    print(f"\n✅ Migration terminée !")
    print(f"   - Fichiers Parquet : {PARQUET_DIR}")
    print(f"   - Fichiers JSON    : {JSON_DIR}")


def create_aliases():
    """Crée des alias BTC->BTCUSDC, ETH->ETHUSDC, etc."""
    print("=" * 80)
    print("CRÉATION D'ALIAS POUR SYMBOLES RACCOURCIS")
    print("=" * 80)

    print(f"\n📋 Mapping configuré:")
    for short, full in SYMBOL_MAPPING.items():
        print(f"   {short} → {full}")

    print(f"\n🔗 Création des liens symboliques...\n")

    created = 0
    skipped = 0

    for short_symbol, full_symbol in SYMBOL_MAPPING.items():
        # Chercher tous les fichiers du symbole complet
        for ext in [".parquet", ".json"]:
            for timeframe in ["1h", "5m", "15m", "30m", "1d", "4h"]:
                # Fichier source
                src_parquet = PARQUET_DIR / f"{full_symbol}_{timeframe}.parquet"
                src_json = JSON_DIR / f"{full_symbol}_{timeframe}.json"

                # Fichier alias
                alias_parquet = PARQUET_DIR / f"{short_symbol}_{timeframe}.parquet"
                alias_json = JSON_DIR / f"{short_symbol}_{timeframe}.json"

                # Créer alias Parquet
                if src_parquet.exists() and not alias_parquet.exists():
                    shutil.copy2(src_parquet, alias_parquet)
                    print(f"✅ Créé: {short_symbol}_{timeframe}.parquet")
                    created += 1
                elif src_parquet.exists():
                    skipped += 1

                # Créer alias JSON
                if src_json.exists() and not alias_json.exists():
                    shutil.copy2(src_json, alias_json)
                    print(f"✅ Créé: {short_symbol}_{timeframe}.json")
                    created += 1
                elif src_json.exists():
                    skipped += 1

    print(f"\n✅ Création terminée !")
    print(f"   - Fichiers créés : {created}")
    print(f"   - Déjà existants : {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Outil de migration et validation des données crypto ThreadX"
    )
    parser.add_argument("--validate", action="store_true", help="Valider les fichiers")
    parser.add_argument("--migrate", action="store_true", help="Migrer les fichiers vers les bons dossiers")
    parser.add_argument("--fix-names", action="store_true", help="Créer des alias BTC->BTCUSDC")
    parser.add_argument("--all", action="store_true", help="Tout faire (validate + migrate + fix-names)")

    args = parser.parse_args()

    if not any([args.validate, args.migrate, args.fix_names, args.all]):
        parser.print_help()
        print("\n⚠️  Utilisez au moins une option (--validate, --migrate, --fix-names, ou --all)")
        return

    if args.all or args.validate:
        validate_all_files()
        print()

    if args.all or args.migrate:
        migrate_files()
        print()

    if args.all or args.fix_names:
        create_aliases()
        print()


if __name__ == "__main__":
    main()
