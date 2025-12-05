"""Data preprocessing utilities for the volforecast project.

This module provides preprocessing functionality for data cleaning and smoothing,
including winsorization to handle outliers and EWMA filtering to reduce noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing.

    Attributes:
        apply_winsorization: Whether to apply winsorization to remove outliers.
        winsorize_lower: Lower percentile for winsorization (e.g., 0.01 for 1st percentile).
        winsorize_upper: Upper percentile for winsorization (e.g., 0.99 for 99th percentile).
        apply_ewma: Whether to apply exponential weighted moving average filtering.
        ewma_span: Span parameter for EWMA (larger = more smoothing). None to skip EWMA.
        apply_on_columns: List of columns to apply preprocessing to. If None, apply to numeric columns.
    """

    apply_winsorization: bool = True
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    apply_ewma: bool = True
    ewma_span: Optional[int] = 20
    apply_on_columns: Optional[list[str]] = None


class FeaturePreprocessor:
    """Handles data preprocessing including winsorization and EWMA filtering.

    This preprocessor is applied per-ticker to ensure consistency between train/test sets.
    Preprocessing parameters computed on the training set are stored and applied to the test set.
    """

    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()):
        """Initialize the preprocessor with configuration.

        Args:
            config: PreprocessingConfig object with preprocessing parameters.
        """
        self.config = config
        self.preprocessing_params: dict = {}  # Store params per ticker for train/test consistency

    def _apply_winsorization(
        self,
        series: pd.Series,
        lower_percentile: float,
        upper_percentile: float,
    ) -> pd.Series:
        """Apply winsorization to a series.

        Replaces values below the lower percentile with the lower bound value,
        and values above the upper percentile with the upper bound value.

        Args:
            series: Input series to winsorize.
            lower_percentile: Lower percentile (e.g., 0.01 for 1st percentile).
            upper_percentile: Upper percentile (e.g., 0.99 for 99th percentile).

        Returns:
            Winsorized series.
        """
        lower_bound = series.quantile(lower_percentile)
        upper_bound = series.quantile(upper_percentile)
        return series.clip(lower=lower_bound, upper=upper_bound)

    def _apply_ewma(
        self,
        series: pd.Series,
        span: int,
    ) -> pd.Series:
        """Apply exponential weighted moving average filtering to a series.

        EWMA gives more weight to recent observations while smoothing out noise.

        Args:
            series: Input series to filter.
            span: Span parameter for EWMA. Larger values = more smoothing.

        Returns:
            EWMA-filtered series.
        """
        return series.ewm(span=span, adjust=False).mean()

    def preprocess(
        self,
        df: pd.DataFrame,
        ticker: Optional[str] = None,
        fit_params: bool = True,
    ) -> pd.DataFrame:
        """Apply preprocessing to a DataFrame.

        When fit_params=True (typically on training data), this method computes and stores
        preprocessing parameters. When fit_params=False (typically on test data), it uses
        the stored parameters for consistency.

        Args:
            df: Input DataFrame to preprocess. Should contain numeric columns.
            ticker: Optional ticker name for per-ticker parameter tracking.
            fit_params: If True, compute and store preprocessing parameters for this ticker.
                       If False, use already stored parameters.

        Returns:
            Preprocessed DataFrame with winsorization and/or EWMA applied.

        Raises:
            ValueError: If fit_params=False but no stored parameters exist for the ticker.
        """
        work = df.copy()

        # Determine which columns to preprocess
        if self.config.apply_on_columns is not None:
            cols = self.config.apply_on_columns
        else:
            # Select numeric columns only
            cols = work.select_dtypes(include=[np.number]).columns.tolist()

        # Filter out non-existent columns
        cols = [c for c in cols if c in work.columns]

        if not cols:
            return work

        # Generate storage key for this ticker
        param_key = ticker if ticker is not None else "default"

        # Apply preprocessing per column
        for col in cols:
            if col not in work.columns or not pd.api.types.is_numeric_dtype(work[col]):
                continue

            series = work[col].copy()

            # Winsorization
            if self.config.apply_winsorization:
                if fit_params:
                    lower = self.config.winsorize_lower
                    upper = self.config.winsorize_upper
                else:
                    if param_key not in self.preprocessing_params:
                        raise ValueError(
                            f"No preprocessing parameters found for {param_key}. "
                            "fit_params must be True on the training set first."
                        )
                    lower = self.preprocessing_params[param_key]["winsorize_lower"]
                    upper = self.preprocessing_params[param_key]["winsorize_upper"]

                series = self._apply_winsorization(series, lower, upper)

            # EWMA filtering
            if self.config.apply_ewma and self.config.ewma_span is not None:
                series = self._apply_ewma(series, self.config.ewma_span)

            work[col] = series

        # Store parameters if fitting
        if fit_params:
            self.preprocessing_params[param_key] = {
                "winsorize_lower": self.config.winsorize_lower,
                "winsorize_upper": self.config.winsorize_upper,
                "ewma_span": self.config.ewma_span,
                "columns": cols,
            }

        return work

    def preprocess_by_ticker(
        self,
        df: pd.DataFrame,
        ticker_col: str = "ticker",
        fit_params: bool = True,
    ) -> pd.DataFrame:
        """Apply preprocessing separately for each ticker in the DataFrame.

        This is useful when you have a concatenated DataFrame with multiple tickers
        and want to ensure preprocessing parameters are computed per-ticker.

        Args:
            df: Input DataFrame with a ticker column.
            ticker_col: Name of the ticker column.
            fit_params: If True, compute and store parameters. If False, use stored parameters.

        Returns:
            DataFrame with preprocessing applied per ticker.
        """
        if ticker_col not in df.columns:
            raise KeyError(f"Column '{ticker_col}' not found in DataFrame")

        result_parts = []

        for ticker, group in df.groupby(ticker_col):
            # Exclude the ticker column from preprocessing
            preprocessing_cols = [c for c in group.columns if c != ticker_col]
            temp_config = PreprocessingConfig(
                apply_winsorization=self.config.apply_winsorization,
                winsorize_lower=self.config.winsorize_lower,
                winsorize_upper=self.config.winsorize_upper,
                apply_ewma=self.config.apply_ewma,
                ewma_span=self.config.ewma_span,
                apply_on_columns=preprocessing_cols,
            )

            # Create temporary preprocessor or use self with updated config
            preprocessed_group = self.preprocess(
                group,
                ticker=ticker,
                fit_params=fit_params,
            )

            result_parts.append(preprocessed_group)

        return pd.concat(result_parts, ignore_index=True)

    def get_preprocessing_info(self) -> dict:
        """Return information about stored preprocessing parameters.

        Returns:
            Dictionary with preprocessing parameters for each ticker.
        """
        return self.preprocessing_params.copy()

    def reset_parameters(self) -> None:
        """Clear all stored preprocessing parameters."""
        self.preprocessing_params = {}
