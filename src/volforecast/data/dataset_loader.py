"""Dataset loader utilities for the volforecast project.

This module provides a small, testable loader class that discovers CSV files
matching a glob pattern (default: "*_dataset.csv") under a directory and
returns a concatenated pandas DataFrame. Each loaded DataFrame is augmented
with `_source_file` and `ticker` columns.
"""
from __future__ import annotations

import os
import glob
from typing import List

import numpy as np
import pandas as pd


class DatasetLoader:
    """Helper to load datasets from disk.

    Example usage:
        loader = DatasetLoader()
        df = loader.load_all_datasets("data/stock_info")
    """

    @staticmethod
    def load_all_datasets(data_dir: str, pattern: str = "*_dataset.csv") -> pd.DataFrame:
        """
        Recursively read CSV files under `data_dir` that match `pattern` and return
        a concatenated `pd.DataFrame`.

        Each loaded DataFrame will be augmented with two columns:
        - `_source_file`: full path to the CSV file
        - `ticker`: filename with the trailing `_dataset.csv` removed (when present)

        Args:
            data_dir: path to search for CSV files (recursive)
            pattern: glob pattern to match files (default: "*_dataset.csv")

        Raises:
            FileNotFoundError: if no matching files are found or none are readable

        Returns:
            Concatenated DataFrame of all loaded files.
        """
        search_pattern = os.path.join(data_dir, "**", pattern)
        files: List[str] = glob.glob(search_pattern, recursive=True)
        files = [f for f in files if os.path.isfile(f)]

        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' found under: {data_dir}")

        dfs: List[pd.DataFrame] = []
        for f in sorted(files):
            try:
                df = pd.read_csv(f)
            except Exception as exc:
                # Warn and skip files that fail to parse
                print(f"Warning: failed to read '{f}': {exc}")
                continue

            base = os.path.basename(f)
            ticker = base
            suffix = "_dataset.csv"
            if base.endswith(suffix):
                ticker = base[: -len(suffix)]

            df["_source_file"] = f
            df["ticker"] = ticker
            dfs.append(df)

        if not dfs:
            raise FileNotFoundError(
                f"No readable CSV files found matching '{pattern}' under: {data_dir}"
            )

        return pd.concat(dfs, ignore_index=True)

    def compute_ticker_stats(
        self,
        df: pd.DataFrame,
        fraction: float,
        value_col: str = "Log_Return",
    ) -> pd.DataFrame:
        """
        Compute mean and standard deviation for `value_col` for each `ticker`,
        using only the first `fraction` portion of rows for each ticker.

        Args:
            df: concatenated DataFrame containing a `ticker` column (as produced
                by `load_all_datasets`). Rows for each ticker are considered in
                ascending date order when a `Date` column is present; otherwise
                the current row order is used.
            fraction: float in (0, 1] indicating the percentage of each ticker's
                rows to use for statistics (e.g. 0.8 uses the first 80% rows).
            value_col: column to compute statistics on. If the column is missing
                and a `Close` column exists, log-returns will be computed from
                `Close` and used instead.

        Returns:
            DataFrame indexed by `ticker` with columns `mean`, `std`, `n_used`.
        """
        if not (0 < fraction <= 1):
            raise ValueError("fraction must be > 0 and <= 1")

        if "ticker" not in df.columns:
            raise KeyError("Input DataFrame must contain a 'ticker' column")

        records = []

        for ticker, group in df.groupby("ticker"):
            g = group.copy()
            # Prefer sorting by Date when available
            if "Date" in g.columns:
                try:
                    g = g.sort_values("Date")
                except Exception:
                    # If Date cannot be sorted, keep original order
                    pass

            n_total = len(g)
            if n_total == 0:
                records.append({"ticker": ticker, "mean": np.nan, "std": np.nan, "n_used": 0})
                continue

            n_use = int(np.floor(n_total * fraction))
            n_use = max(1, n_use)

            if value_col in g.columns:
                series = g[value_col].astype(float).reset_index(drop=True)
            elif "Close" in g.columns:
                # compute log returns from Close as fallback
                closes = g["Close"].astype(float).reset_index(drop=True)
                series = np.log(closes).diff().fillna(0)
            else:
                raise KeyError(f"Neither '{value_col}' nor 'Close' columns found for ticker {ticker}")

            sample = series.iloc[:n_use]
            mean = float(sample.mean()) if len(sample) > 0 else float("nan")
            std = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0

            records.append({"ticker": ticker, "mean": mean, "std": std, "n_used": n_use})

        stats = pd.DataFrame.from_records(records).set_index("ticker")
        return stats

    def normalize_by_ticker(
        self,
        df: pd.DataFrame,
        fraction: float,
        cols: list | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return two DataFrames: training and testing sets, both normalized per-ticker
        using mean and std computed on the first `fraction` portion of rows for each ticker.

        Behavior:
        - For each ticker, rows are sorted by `Date` if present before splitting.
        - The first `fraction` portion of each ticker is used to compute statistics
          and becomes the training set.
        - The remaining portion becomes the testing set.
        - Both datasets are normalized using the same per-ticker statistics.
        - If `cols` is None, all numeric columns are standardized except the
          internal columns (`_source_file`, `ticker`).
        - To avoid division-by-zero, when the computed std is zero the std is
          treated as 1.0 (the standardized values will be 0).

        Args:
            df: DataFrame containing `_source_file` and `ticker` columns.
            fraction: float in (0, 1) fraction of each ticker's rows to use
                for training (computing mean/std and providing train data).
                The remainder becomes the test set.
            cols: Optional list of columns to standardize. If None, numeric
                columns are selected automatically.

        Returns:
            Tuple of (train_df, test_df), both normalized with the same per-ticker
            statistics computed from the training portion.
        """
        if not (0 < fraction < 1):
            raise ValueError("fraction must be > 0 and < 1")

        if "ticker" not in df.columns:
            raise KeyError("Input DataFrame must contain a 'ticker' column")

        # Choose columns to normalize
        work = df.copy()
        if cols is None:
            # select numeric columns excluding internal metadata
            numeric = work.select_dtypes(include=[np.number]).columns.tolist()
            exclude = {"_source_file"}
            cols = [c for c in numeric if c not in exclude and c != "ticker"]

        # Compute per-ticker statistics and split
        train_parts = []
        test_parts = []

        for ticker, group in work.groupby("ticker"):
            g = group.copy()
            # sort by Date when available
            if "Date" in g.columns:
                try:
                    g = g.sort_values("Date").reset_index(drop=True)
                except Exception:
                    g = g.reset_index(drop=True)
            else:
                g = g.reset_index(drop=True)

            n_total = len(g)
            if n_total == 0:
                continue

            n_use = int(np.floor(n_total * fraction))
            n_use = max(1, n_use)

            # Split train/test
            g_train = g.iloc[:n_use].copy()
            g_test = g.iloc[n_use:].copy()

            # Compute statistics from train portion only
            stats_dict = {}
            for col in cols:
                if col not in g.columns:
                    continue

                sample = g_train[col].astype(float)
                mu = float(sample.mean()) if len(sample) > 0 else 0.0
                std = float(sample.std(ddof=1)) if len(sample) > 1 else 0.0
                if std == 0.0:
                    std = 1.0

                stats_dict[col] = (mu, std)

            # Apply normalization to both train and test using train statistics
            for col, (mu, std) in stats_dict.items():
                g_train[col] = (g_train[col].astype(float) - mu) / std
                if len(g_test) > 0:
                    g_test[col] = (g_test[col].astype(float) - mu) / std

            train_parts.append(g_train)
            if len(g_test) > 0:
                test_parts.append(g_test)

        train_result = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
        test_result = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

        return train_result, test_result



