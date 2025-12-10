# src/build_price_only_daily.py
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2010-01-01"
END_DATE = "2025-01-01"

def download_spy():
    df = yf.download("SPY", start=START_DATE, end=END_DATE, interval="1d")
    df = df.rename_axis("date").reset_index()
    df.to_csv(RAW_DIR / "spy_price.csv", index=False)
    return df

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Parse dates and drop rows without a real date (like that NaT/SPY row)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    # 2) Sort by date, reset index
    df = df.sort_values("date").reset_index(drop=True)

    # 3) Decide which price column to use
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise ValueError(f"No 'Adj Close' or 'Close' column found. Columns: {df.columns}")

    # 4) Force numeric types for price-related columns
    numeric_cols = [c for c in ["Close", "Open", "High", "Low", "Adj Close", "Volume"] if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop any rows where price_col couldn't be parsed to a number
    df = df[df[price_col].notna()].copy()

    # 5) Basic returns
    df["ret_1d"] = df[price_col].pct_change()
    df["log_ret_1d"] = np.log(df[price_col] / df[price_col].shift(1))

    # 6) Moving averages
    df["ma_close_5"] = df[price_col].rolling(5).mean()
    df["ma_close_20"] = df[price_col].rolling(20).mean()

    # 7) Rolling volatility
    df["vol_5"] = df["ret_1d"].rolling(5).std()
    df["vol_20"] = df["ret_1d"].rolling(20).std()

    # 8) Target: next-day up/down
    df["future_price"] = df[price_col].shift(-1)
    df["future_ret_1d"] = (df["future_price"] / df[price_col]) - 1.0
    df["label_up"] = (df["future_ret_1d"] > 0).astype("int64")

    # 9) Drop NaNs and last row (no future)
    df = df.dropna().reset_index(drop=True)

    return df



def main():
    try:
        df = pd.read_csv(RAW_DIR / "spy_price.csv", parse_dates=["date"])
        print("\n=== RAW DATA PREVIEW ===")
        print(df.head())
        print("\nColumns:", list(df.columns))
        print("Shape:", df.shape)
        print("Date range:", df['date'].min(), "→", df['date'].max())
        print("========================\n")
        print("dtypes BEFORE features:\n", df.dtypes)


    except FileNotFoundError:
        df = download_spy()

    df = add_technical_features(df)
    print("dtypes AFTER features:\n", df.dtypes)
    df.to_parquet(PROC_DIR / "daily_merged.parquet", index=False)
    print("Saved:", PROC_DIR / "daily_merged.parquet")
    print(df.head())
    print(df[["label_up"]].value_counts(normalize=True))

if __name__ == "__main__":
    main()
