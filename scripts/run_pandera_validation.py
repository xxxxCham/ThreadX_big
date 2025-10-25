"""Run strict Pandera validation on files listed in artifacts/validated_mapping_plan.csv

Produces:
 - artifacts/pandera_validation_details.csv (per-file results)
 - artifacts/pandera_validation_report.md (summary + examples)

This script does NOT modify data; it only analyses and suggests fixes.
"""
from pathlib import Path
import csv
import pandas as pd
import argparse
from pandera import Column, DataFrameSchema
import pandera as pa
# validate_dataset not used here; file-level validation via pandera directly


OHLCV = ["open", "high", "low", "close", "volume"]


def validate_candle(df: pd.DataFrame):
    """Return (ok, errors:list, suggested_fix:str)"""
    errors = []
    suggested = []

    cols = [c.lower() for c in df.columns]
    missing = [c for c in OHLCV if c not in cols]
    if missing:
        errors.append(f"missing_columns:{','.join(missing)}")
        # Suggest renaming common variants
        if any(c in cols for c in ["close_price", "closep", "closeprice"]):
            suggested.append("rename_close_variants")
        if "volume" in missing:
            suggested.append("fill_volume_with_0")

    # Check numeric coercion
    non_numeric_cols = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            try:
                pd.to_numeric(df[c].dropna())
            except Exception:
                non_numeric_cols.append(c)
    if non_numeric_cols:
        errors.append(f"non_numeric:{','.join(non_numeric_cols)}")
        suggested.append("coerce_numeric")

    # Timestamp/index checks
    idx = df.index
    if not pd.api.types.is_datetime64_any_dtype(idx):
        # see if there is a timestamp column
        ts_col = None
        for cand in ["timestamp", "time", "date", "datetime"]:
            if cand in cols:
                ts_col = cand
                break
        if ts_col is None:
            # maybe index is integer unix ts
            if pd.api.types.is_integer_dtype(idx):
                suggested.append("convert_index_unix_ts")
            else:
                errors.append("no_datetime_index_or_timestamp_column")
                suggested.append("set_datetime_index_from_column")
        else:
            suggested.append(f"use_{ts_col}_as_datetime_index")

    # Pandera schema validate if possible
    schema = DataFrameSchema(
        {
            "open": Column(float, nullable=False),
            "high": Column(float, nullable=False),
            "low": Column(float, nullable=False),
            "close": Column(float, nullable=False),
            "volume": Column(float, nullable=True),
        }
    )
    try:
        # lower-case columns mapping
        df_lc = df.copy()
        df_lc.columns = [c.lower() for c in df_lc.columns]
        schema.validate(df_lc, lazy=True)
    except pa.errors.SchemaErrors as e:
        errors.append(f"pandera:{str(e)}")
        suggested.append("fix_schema_issues")

    ok = len(errors) == 0
    return ok, errors, ";".join(suggested)


def validate_indicator(df: pd.DataFrame):
    errors = []
    suggested = []
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric) == 0:
        errors.append("no_numeric_columns")
        suggested.append("extract_numeric_columns_or_drop")

    # index check
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        suggested.append("ensure_datetime_index_or_timestamp_column")

    ok = len(errors) == 0
    return ok, errors, ";".join(suggested)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapping", default="artifacts/validated_mapping_plan.csv")
    p.add_argument("--out-csv", default="artifacts/pandera_validation_details.csv")
    p.add_argument("--out-md", default="artifacts/pandera_validation_report.md")
    args = p.parse_args()

    mapping = Path(args.mapping)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    ok_count = 0
    sample_errors = []

    if not mapping.exists():
        print(f"Mapping file not found: {mapping}")
        return

    with open(mapping, newline='', encoding='utf-8') as mf:
        reader = csv.DictReader(mf)
        for r in reader:
            total += 1
            src = Path(r['source_path'])
            dest = r.get('dest_path', '')
            ftype = r.get('type', '')
            detail = {"source": str(src), "dest": dest, "type": ftype, "ok": False, "errors": "", "suggested_fix": ""}

            if not src.exists():
                detail['errors'] = 'missing_file'
                rows.append(detail)
                sample_errors.append(detail)
                continue

            try:
                # read file
                if src.suffix.lower() == '.parquet':
                    df = pd.read_parquet(src)
                elif src.suffix.lower() == '.json':
                    df = pd.read_json(src, orient='records', convert_dates=False)
                elif src.suffix.lower() == '.csv':
                    df = pd.read_csv(src)
                else:
                    detail['errors'] = f'unsupported_ext:{src.suffix}'
                    rows.append(detail)
                    sample_errors.append(detail)
                    continue

                # ensure we look at datetime index if present
                if df.index.name is None and 'timestamp' in (c.lower() for c in df.columns):
                    # do not modify df here; just note
                    pass

                if ftype == 'candle':
                    ok, errors, suggested = validate_candle(df)
                else:
                    ok, errors, suggested = validate_indicator(df)

                detail['ok'] = ok
                detail['errors'] = ' | '.join(errors) if errors else ''
                detail['suggested_fix'] = suggested
                rows.append(detail)
                if not ok and len(sample_errors) < 50:
                    sample_errors.append(detail)
                if ok:
                    ok_count += 1

            except Exception as exc:
                detail['errors'] = 'read_error:' + str(exc)
                detail['suggested_fix'] = 'inspect_read_issue'
                rows.append(detail)
                if len(sample_errors) < 50:
                    sample_errors.append(detail)

    # write CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as outf:
        writer = csv.DictWriter(
            outf,
            fieldnames=['source', 'dest', 'type', 'ok', 'errors', 'suggested_fix'],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # write markdown summary
    with open(out_md, 'w', encoding='utf-8') as md:
        md.write("# Pandera validation report\n\n")
        md.write(f"Total files scanned: {total}\n\n")
        md.write(f"Valid files: {ok_count}\n\n")
        md.write("## Sample errors\n\n")
        for e in sample_errors[:50]:
            md.write(f"- {e['source']}: {e['errors']} -> suggested: {e['suggested_fix']}\n")

    print(f"Wrote details to {out_csv} and summary to {out_md}")


if __name__ == '__main__':
    main()
