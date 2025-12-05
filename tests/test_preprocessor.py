"""Unit tests for data preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from src.volforecast.data.preprocessor import FeaturePreprocessor, PreprocessingConfig


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = PreprocessingConfig()
        assert cfg.apply_winsorization is True
        assert cfg.winsorize_lower == 0.01
        assert cfg.winsorize_upper == 0.99
        assert cfg.apply_ewma is True
        assert cfg.ewma_span == 20

    def test_custom_config(self):
        """Test custom configuration values."""
        cfg = PreprocessingConfig(
            apply_winsorization=False,
            winsorize_lower=0.05,
            ewma_span=30,
        )
        assert cfg.apply_winsorization is False
        assert cfg.winsorize_lower == 0.05
        assert cfg.ewma_span == 30


class TestFeaturePreprocessor:
    """Tests for FeaturePreprocessor class."""

    @pytest.fixture
    def sample_series(self):
        """Create a sample series with some outliers."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        data[10] = -10  # negative outlier
        data[50] = 10  # positive outlier
        return pd.Series(data)

    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame with multiple columns."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "returns": np.random.normal(0, 1, 100),
                "vix": np.random.normal(15, 5, 100),
                "ticker": ["AAPL"] * 100,
            }
        )

    def test_winsorization(self, sample_series):
        """Test winsorization removes outliers."""
        cfg = PreprocessingConfig(
            apply_winsorization=True,
            apply_ewma=False,
            winsorize_lower=0.01,
            winsorize_upper=0.99,
        )
        preprocessor = FeaturePreprocessor(cfg)

        # Apply winsorization directly
        lower = sample_series.quantile(0.01)
        upper = sample_series.quantile(0.99)
        result = preprocessor._apply_winsorization(sample_series, 0.01, 0.99)

        # Check that extreme values are bounded
        assert result.min() >= lower
        assert result.max() <= upper
        assert len(result) == len(sample_series)

    def test_ewma_filtering(self, sample_series):
        """Test EWMA filtering reduces variance."""
        cfg = PreprocessingConfig(
            apply_winsorization=False,
            apply_ewma=True,
            ewma_span=20,
        )
        preprocessor = FeaturePreprocessor(cfg)

        result = preprocessor._apply_ewma(sample_series, span=20)

        # EWMA should generally have lower variance than original
        assert result.std() < sample_series.std() or np.isclose(
            result.std(), sample_series.std(), rtol=0.1
        )
        assert len(result) == len(sample_series)

    def test_preprocess_with_winsorization_only(self, sample_dataframe):
        """Test preprocessing with only winsorization."""
        cfg = PreprocessingConfig(
            apply_winsorization=True,
            apply_ewma=False,
            apply_on_columns=["returns", "vix"],
        )
        preprocessor = FeaturePreprocessor(cfg)

        result = preprocessor.preprocess(sample_dataframe, ticker="AAPL", fit_params=True)

        assert "returns" in result.columns
        assert "vix" in result.columns
        assert len(result) == len(sample_dataframe)
        assert "AAPL" in preprocessor.preprocessing_params  ##FIXME

    def test_preprocess_with_ewma_only(self, sample_dataframe):
        """Test preprocessing with only EWMA."""
        cfg = PreprocessingConfig(
            apply_winsorization=False,
            apply_ewma=True,
            ewma_span=20,
            apply_on_columns=["returns"],
        )
        preprocessor = FeaturePreprocessor(cfg)

        result = preprocessor.preprocess(sample_dataframe, ticker="AAPL", fit_params=True)

        assert "returns" in result.columns
        assert len(result) == len(sample_dataframe)

    def test_preprocess_with_both_winsorization_and_ewma(self, sample_dataframe):
        """Test preprocessing with both winsorization and EWMA."""
        cfg = PreprocessingConfig(
            apply_winsorization=True,
            apply_ewma=True,
            ewma_span=20,
            apply_on_columns=["returns", "vix"],
        )
        preprocessor = FeaturePreprocessor(cfg)

        result = preprocessor.preprocess(sample_dataframe, ticker="AAPL", fit_params=True)

        assert "returns" in result.columns
        assert "vix" in result.columns
        assert len(result) == len(sample_dataframe)

    def test_preprocess_parameter_consistency(self, sample_dataframe):
        """Test that train and test data use consistent parameters."""
        cfg = PreprocessingConfig(
            apply_winsorization=True,
            apply_ewma=True,
            ewma_span=20,
        )
        preprocessor = FeaturePreprocessor(cfg)

        # Split into train and test
        train = sample_dataframe.iloc[:70].copy()
        test = sample_dataframe.iloc[70:].copy()

        # Preprocess train with fit_params=True
        train_proc = preprocessor.preprocess(train, ticker="AAPL", fit_params=True)

        # Preprocess test with fit_params=False (use train parameters)
        test_proc = preprocessor.preprocess(test, ticker="AAPL", fit_params=False)

        # Check that parameters were stored
        params = preprocessor.get_preprocessing_info()
        assert "AAPL" in params
        assert params["AAPL"]["winsorize_lower"] == 0.01
        assert params["AAPL"]["ewma_span"] == 20

    def test_preprocess_without_parameters_raises_error(self, sample_dataframe):
        """Test that using fit_params=False without prior fit raises error."""
        cfg = PreprocessingConfig(
            apply_winsorization=True,
            apply_ewma=False,
        )
        preprocessor = FeaturePreprocessor(cfg)

        with pytest.raises(ValueError):
            preprocessor.preprocess(sample_dataframe, ticker="NEW_TICKER", fit_params=False)

    def test_preprocess_by_ticker(self):
        """Test preprocessing with multiple tickers."""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "returns": np.random.normal(0, 1, 200),
                "ticker": ["AAPL"] * 100 + ["MSFT"] * 100,
            }
        )

        cfg = PreprocessingConfig(
            apply_winsorization=True,
            apply_ewma=False,
            apply_on_columns=["returns"],
        )
        preprocessor = FeaturePreprocessor(cfg)

        result = preprocessor.preprocess_by_ticker(df, ticker_col="ticker", fit_params=True)

        assert len(result) == len(df)
        params = preprocessor.get_preprocessing_info()
        assert "AAPL" in params
        assert "MSFT" in params

    def test_reset_parameters(self, sample_dataframe):
        """Test resetting stored preprocessing parameters."""
        cfg = PreprocessingConfig(apply_winsorization=True, apply_ewma=False)
        preprocessor = FeaturePreprocessor(cfg)

        preprocessor.preprocess(sample_dataframe, ticker="AAPL", fit_params=True)
        assert len(preprocessor.get_preprocessing_info()) > 0

        preprocessor.reset_parameters()
        assert len(preprocessor.get_preprocessing_info()) == 0

    def test_preprocess_with_no_columns(self):
        """Test preprocessing handles DataFrames with no numeric columns."""
        df = pd.DataFrame(
            {
                "name": ["A", "B", "C"],
                "date": pd.date_range("2020-01-01", periods=3),
            }
        )

        cfg = PreprocessingConfig(apply_winsorization=True)
        preprocessor = FeaturePreprocessor(cfg)

        result = preprocessor.preprocess(df, fit_params=True)
        assert len(result) == len(df)
        assert result["name"].tolist() == df["name"].tolist()

    def test_preprocess_preserves_index(self, sample_dataframe):
        """Test that preprocessing preserves DataFrame index."""
        sample_dataframe.index = pd.date_range("2020-01-01", periods=len(sample_dataframe))

        cfg = PreprocessingConfig(apply_winsorization=True)
        preprocessor = FeaturePreprocessor(cfg)

        result = preprocessor.preprocess(sample_dataframe, fit_params=True)

        assert result.index.equals(sample_dataframe.index)

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        cfg = PreprocessingConfig(apply_winsorization=True)
        preprocessor = FeaturePreprocessor(cfg)

        result = preprocessor.preprocess(df, fit_params=True)
        assert len(result) == 0
