"""Provisional corrections script.

Reads:
 - artifacts/pandera_validation_details.csv
 - artifacts/validated_mapping_plan.csv

Produces:
 - artifacts/pandera_correction_plan.csv (actions proposed or applied)
 - artifacts/pandera_correction_report.md

By default runs in dry-run mode. Use --apply to perform copies/writes.
Use --limit N to process only N files (0 => all).
"""

from pathlib import Path
import csv
import argparse
import pandas as pd
from pandera import Column, DataFrameSchema
import pandera as pa
from threadx.data.io import write_frame


COMMON_RENAMES = {
    # common variants -> canonical
    "close_price": "close",
    "closeprice": "close",
    "closep": "close",
    "open_price": "open",
    "high_price": "high",
    "low_price": "low",
}


OHLCV = ["open", "high", "low", "close", "volume"]


def apply_renames(df: pd.DataFrame):
    rename_map = {}
    for c in list(df.columns):
        lc = c.lower()
        if lc in COMMON_RENAMES:
            rename_map[c] = COMMON_RENAMES[lc]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, bool(rename_map)


def coerce_numeric(df: pd.DataFrame):
    coerced = False
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            before_nonnull = df[c].notna().sum()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            after_nonnull = df[c].notna().sum()
            if after_nonnull < before_nonnull:
                # some coercion produced NaNs
                coerced = True
    return df, coerced


def ensure_volume(df: pd.DataFrame):
    if "volume" not in [c.lower() for c in df.columns]:
        df["volume"] = 0
        return df, True
    return df, False


def ensure_datetime_index(df: pd.DataFrame):
    # If index is integer and looks like unix ts, convert
    changed = False
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        # try common timestamp columns
        for col in ["timestamp", "time", "date", "datetime"]:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(
                        df[col], unit=None, errors="coerce", utc=True
                    )
                    df = df.set_index(col)
                    changed = True
                    break
                except Exception:
                    continue

        # else try to interpret index as unix ts
        if not changed and pd.api.types.is_integer_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index, unit="s", utc=True)
                changed = True
            except Exception:
                try:
                    df.index = pd.to_datetime(df.index, unit="ms", utc=True)
                    changed = True
                except Exception:
                    pass

    return df, changed


def pandera_validate(df: pd.DataFrame):
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
        df_lc = df.copy()
        df_lc.columns = [c.lower() for c in df_lc.columns]
        schema.validate(df_lc, lazy=True)
        return True, None
    except pa.errors.SchemaErrors as e:
        return False, str(e)


def process_file(src: Path, dest: Path, ftype: str = "indicator", apply: bool = False):
    """Process a single file and return a detail dict.

    ftype: 'candle' or 'indicator' (string)
    """
    detail = {
        "source": str(src),
        "dest": str(dest),
        "type": ftype,
        "applied": False,
        "errors": "",
        "actions": "",
    }
    try:
        # read file
        if not src.exists():
            detail["errors"] = "missing_source"
            return detail

        if src.suffix.lower() == ".parquet":
            df = pd.read_parquet(src)
        elif src.suffix.lower() == ".json":
            df = pd.read_json(src, orient="records", convert_dates=False)
        elif src.suffix.lower() == ".csv":
            df = pd.read_csv(src)
        else:
            detail["errors"] = f"unsupported_ext:{src.suffix}"
            return detail

        actions = []

        # common fixes
        df, renamed = apply_renames(df)
        if renamed:
            actions.append("renamed_common_columns")

        df, coerced = coerce_numeric(df)
        if coerced:
            actions.append("coerced_numeric")

        # branch by type
        if ftype and str(ftype).lower().startswith("cand"):
            # candles must have volume, datetime index and pass pandera
            # ensure OHLCV columns are float64 before validation (fix common dtype issues)
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    try:
                        df[col] = df[col].astype("float64")
                    except Exception:
                        df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                            "float64"
                        )

            df, vol_added = ensure_volume(df)
            if vol_added:
                actions.append("added_volume_0")

            df, idx_changed = ensure_datetime_index(df)
            if idx_changed:
                actions.append("set_datetime_index")

            ok, pand_err = pandera_validate(df)
            if ok:
                actions.append("pandera_ok")
                if apply:
                    dest_path = Path(dest)
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    # use centralized writer to enforce normalization/dtypes
                    write_frame(df, dest_path, overwrite=True)
                    detail["applied"] = True
            else:
                detail["errors"] = pand_err

        else:
            # indicators: no OHLCV pandera; try to coerce numerics and set datetime index
            df, idx_changed = ensure_datetime_index(df)
            if idx_changed:
                actions.append("set_datetime_index")
            # write as parquet
            actions.append("indicator_normalized")
            if apply:
                dest_path = Path(dest)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                write_frame(df, dest_path, overwrite=True)
                detail["applied"] = True

        detail["actions"] = ";".join(actions)
        return detail

    except Exception as exc:
        detail["errors"] = str(exc)
        return detail


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--validation-details", default="artifacts/pandera_validation_details.csv"
    )
    p.add_argument("--mapping", default="artifacts/validated_mapping_plan.csv")
    p.add_argument("--out", default="artifacts/pandera_correction_plan.csv")
    p.add_argument("--apply", action="store_true", default=False)
    p.add_argument("--limit", type=int, default=100, help="0 => all")
    args = p.parse_args()

    val_details = Path(args.validation_details)
    mapping = Path(args.mapping)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not val_details.exists() or not mapping.exists():
        print(
            "Required artifacts missing. Run run_pandera_validation.py and build_validated_mapping.py first."
        )
        return

    # load mapping into dict source->dest and type
    map_df = pd.read_csv(mapping)
    dest_map = {row["source_path"]: row["dest_path"] for _, row in map_df.iterrows()}
    type_map = {
        row["source_path"]: row.get("type", "indicator") for _, row in map_df.iterrows()
    }

    results = []
    processed = 0

    with open(val_details, newline="", encoding="utf-8") as vf:
        for row in csv.DictReader(vf):
            if args.limit and args.limit > 0 and processed >= args.limit:
                break
            processed += 1
            src = Path(row["source"])
            dest = dest_map.get(row["source"], None)
            ftype = type_map.get(row["source"], "indicator")
            if dest is None:
                # construct fallback dest under validated indicators
                dest = str(
                    Path("data") / "validated" / "indicators" / (src.stem + ".parquet")
                )

            detail = process_file(src, Path(dest), ftype=ftype, apply=args.apply)
            results.append(detail)

    # write out plan
    with open(out, "w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(
            outf, fieldnames=["source", "dest", "type", "applied", "errors", "actions"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # summary md
    md = Path(str(out).replace(".csv", ".md"))
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# Pandera Correction Plan\n\nProcessed: {len(results)}\n\n")
        failed = [r for r in results if r.get("errors")]
        f.write(f"Failures: {len(failed)}\n\n")
        if failed:
            f.write("## Examples:\n")
            for r in failed[:50]:
                f.write(
                    f"- {r['source']}: {r['errors']} -> actions: {r.get('actions', '')}\n"
                )

    print(f"Wrote correction plan to {out} and summary to {md}")


if __name__ == "__main__":
    main()
