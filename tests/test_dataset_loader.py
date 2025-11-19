"""Tests for DatasetLoader in src.volforecast.data.dataset_loader.

Creates temporary CSV files for two tickers and verifies that
`load_all_datasets` concatenates them and `compute_ticker_stats` returns
expected mean/std computed on the first fraction of rows per ticker.
"""
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.volforecast.data.dataset_loader import DatasetLoader


def write_csv(path: Path, dates, values):
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "Value": values})
    df.to_csv(path, index=False)


def test_load_and_compute_stats(tmp_path):
    base = tmp_path / "data" / "stock_info"
    base.mkdir(parents=True)

    # Create two tickers, each with 10 days
    n = 10
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Ticker A: values 1..10
    a_path = base / "TICKA_dataset.csv"
    write_csv(a_path, dates, np.arange(1, n + 1))

    # Ticker B: values 11..20
    b_path = base / "TICKB_dataset.csv"
    write_csv(b_path, dates, np.arange(11, 11 + n))

    loader = DatasetLoader()

    # Load datasets
    df = loader.load_all_datasets(str(tmp_path / "data"))

    # Basic checks
    assert "ticker" in df.columns
    assert "_source_file" in df.columns
    assert len(df) == 2 * n

    # Compute stats using first 80% of rows -> 8 rows per ticker
    fraction = 0.8
    stats = loader.compute_ticker_stats(df, fraction=fraction, value_col="Value")

    assert "mean" in stats.columns and "std" in stats.columns and "n_used" in stats.columns

    # Expected values
    n_use = int(np.floor(n * fraction))
    assert n_use == 8

    expected_mean_a = np.arange(1, n_use + 1).mean()
    expected_std_a = np.std(np.arange(1, n_use + 1), ddof=1)

    expected_mean_b = np.arange(11, 11 + n_use).mean()
    expected_std_b = np.std(np.arange(11, 11 + n_use), ddof=1)

    # Compare with small tolerance
    assert np.isclose(stats.loc["TICKA", "mean"], expected_mean_a)
    assert np.isclose(stats.loc["TICKA", "std"], expected_std_a)
    assert int(stats.loc["TICKA", "n_used"]) == n_use

    assert np.isclose(stats.loc["TICKB", "mean"], expected_mean_b)
    assert np.isclose(stats.loc["TICKB", "std"], expected_std_b)
    assert int(stats.loc["TICKB", "n_used"]) == n_use


def test_normalize_by_ticker_first_fraction_stats(tmp_path):
    """Verify that normalization splits data and produces mean ~0 and std ~1
    on the training portion per ticker.
    """
    base = tmp_path / "data" / "stock_info"
    base.mkdir(parents=True)

    # Small dataset: 6 rows per ticker for fast test
    n = 6
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    a_path = base / "TICKA_dataset.csv"
    write_csv(a_path, dates, np.arange(1, n + 1))

    b_path = base / "TICKB_dataset.csv"
    write_csv(b_path, dates, np.arange(11, 11 + n))

    loader = DatasetLoader()
    df = loader.load_all_datasets(str(tmp_path / "data"))

    fraction = 0.5  # 50% train, 50% test
    df_train, df_test = loader.normalize_by_ticker(df, fraction=fraction, cols=["Value"])

    n_use = int(np.floor(n * fraction))
    assert n_use == 3
    assert len(df_train) == 2 * n_use
    assert len(df_test) == 2 * (n - n_use)

    # Verify training set has mean ~0, std ~1 per ticker
    print("\nNormalized training set statistics:")
    for ticker in ["TICKA", "TICKB"]:
        g_train = df_train[df_train["ticker"] == ticker]["Value"].astype(float)
        mean = g_train.mean()
        std = g_train.std(ddof=1)
        print(f"  {ticker}: mean={mean:.8f}, std={std:.8f}")
        assert np.isclose(mean, 0.0, atol=1e-8)
        assert np.isclose(std, 1.0, atol=1e-8)


if __name__ == "__main__":
    # Quick demo of DatasetLoader functionality
    import argparse
    from pathlib import Path
    import tempfile

    parser = argparse.ArgumentParser(description="Demo DatasetLoader from tests")
    default_data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "stock_info"))
    parser.add_argument("--data-dir", default=default_data_dir, help="Directory to search for dataset CSVs")
    parser.add_argument("--fraction", type=float, default=0.8, help="Fraction for train/test split")
    args = parser.parse_args()

    # Run the normalization test
    print("Running test_normalize_by_ticker_first_fraction_stats...")
    with tempfile.TemporaryDirectory() as tmp:
        test_normalize_by_ticker_first_fraction_stats(Path(tmp))
    
    loader = DatasetLoader()
    try:
        df_all = loader.load_all_datasets(args.data_dir)
        print(f"✓ Loaded {len(df_all)} rows")
        
        df_train, df_test = loader.normalize_by_ticker(df_all, args.fraction, cols=None)
        print(f"✓ Train: {len(df_train)}, Test: {len(df_test)}")
        print("✅ DatasetLoader working correctly")
    except Exception as e:
        print(f"❌ Error: {e}")
