from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from threadx.indicators import indicators_np
from threadx.data.normalize import convert_indicator_df_to_parquet, validate_indicator
from threadx.data.loader import BinanceDataLoader


def find_local_price(symbol: str, timeframe: str) -> Path | None:
    base = Path("data/crypto_data_parquet")
    candidate = base / f"{symbol.upper()}_{timeframe.lower()}.parquet"
    if candidate.exists():
        return candidate
    # fallback: try other timeframes or folder variants
    for p in base.glob(f"{symbol.upper()}_*{timeframe.lower()}*.parquet"):
        return p
    return None


def compute_ema_from_df(df: pd.DataFrame, period: int) -> pd.DataFrame:
    # Ensure index and close
    if "close" not in df.columns:
        raise ValueError("Source price DataFrame must contain 'close' column")
    close = df["close"].astype("float64").to_numpy()
    ema = indicators_np.ema_np(close, span=period)
    df_out = pd.DataFrame({f"ema_{period}": ema}, index=df.index)
    return df_out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True, help="Symbol, e.g. AIXBTUSDC")
    p.add_argument("--timeframe", required=True, help="Timeframe e.g. 3m")
    p.add_argument("--period", type=int, default=20, help="EMA period")
    p.add_argument(
        "--days", type=int, default=365, help="Days history to fetch if needed"
    )
    p.add_argument(
        "--out-dir",
        default="artifacts/regenerated",
        help="Where to write regenerated (dry-run writes here)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Move regenerated file into data/validated on success",
    )
    args = p.parse_args(argv)

    symbol = args.symbol.upper()
    timeframe = args.timeframe.lower()
    period = args.period

    corrupted_path = Path(
        f"data/indicators/{symbol}/{timeframe}/ema_period{period}.parquet"
    )
    print(f"Target indicator path: {corrupted_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}_{timeframe}_ema_period{period}.parquet"

    # 1) Try to find local price parquet
    local = find_local_price(symbol, timeframe)
    df_price = None

    if local:
        print(f"Found local price data: {local}")
        try:
            df_price = pd.read_parquet(local)
        except Exception as e:
            print(f"Failed to read local price parquet {local}: {e}")
            df_price = None

    # 2) If not found, try to download via BinanceDataLoader
    if df_price is None:
        print(
            "Local price not available or unreadable; attempting Binance download (dry-run)."
        )
        loader = BinanceDataLoader(
            json_cache_dir=Path("data/cache/json"),
            parquet_cache_dir=Path("data/crypto_data_parquet"),
        )
        try:
            df_price = loader.download_ohlcv(
                symbol, timeframe, days_history=args.days, save_parquet=True
            )
        except Exception as e:
            print(f"Downloader failed: {e}")
            df_price = None

    if df_price is None or df_price.empty:
        print("No price data available to regenerate indicator. Aborting.")
        return 2

    # 3) Compute EMA
    try:
        df_indicator = compute_ema_from_df(df_price, period)
    except Exception as e:
        print(f"Failed to compute EMA: {e}")
        return 3

    # 4) Convert to canonical parquet in out_dir (dry-run)
    try:
        regenerated = convert_indicator_df_to_parquet(df_indicator, out_path)
        print(f"Regenerated indicator written (dry-run) to: {regenerated}")
    except Exception as e:
        print(f"Failed to convert/write regenerated indicator: {e}")
        return 4

    # 5) Validate
    try:
        report = validate_indicator(pd.read_parquet(regenerated))
        print(f"Validation report: {report}")
        if report.get("status") != "OK":
            print("Validation failed; not applying.")
            return 5
    except Exception as e:
        print(f"Validation error: {e}")
        return 6

    # 6) Commit if requested
    if args.apply:
        target = Path("data/validated/indicators") / symbol / timeframe
        target.mkdir(parents=True, exist_ok=True)
        final_path = target / f"ema_period{period}.parquet"
        # Move file
        try:
            Path(regenerated).replace(final_path)
            print(f"Applied: moved regenerated to {final_path}")
        except Exception as e:
            print(f"Failed to move regenerated file into validated path: {e}")
            return 7
    else:
        print("Dry-run complete: use --apply to commit into data/validated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
