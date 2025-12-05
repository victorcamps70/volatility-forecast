# Data Preprocessing Module

## Overview

The `preprocessor.py` module provides robust data cleaning and smoothing capabilities for the volatility forecasting pipeline. It implements two complementary techniques:

1. **Winsorization** - Handling outliers by clipping extreme values
2. **EWMA Filtering** - Exponential weighted moving average for noise reduction

## Motivation

### Why Winsorization?

- **Outlier Management**: Financial data can have extreme moves (gaps, shocks, market disruptions) that don't represent normal market conditions
- **Model Robustness**: Outliers can disproportionately affect model training, especially with linear models (ElasticNet) and tree-based models sensitive to extreme values
- **Interpretability**: Reduces noise that models might incorrectly interpret as signal

### Why EWMA Filtering?

- **Noise Reduction**: Market data contains microstructure noise (bid-ask bounces, tick effects) that obscures true price signals
- **Temporal Smoothing**: Gives more weight to recent observations while gradually discounting older ones
- **Data Quality**: Particularly useful for handling gaps and irregular trading patterns

## Architecture

### `PreprocessingConfig`

Dataclass that configures preprocessing behavior:

```python
@dataclass
class PreprocessingConfig:
    apply_winsorization: bool = True
    winsorize_lower: float = 0.01      # 1st percentile
    winsorize_upper: float = 0.99      # 99th percentile
    apply_ewma: bool = True
    ewma_span: Optional[int] = 20      # span parameter
    apply_on_columns: Optional[list[str]] = None
```

### `FeaturePreprocessor`

Main class handling all preprocessing logic:

#### Key Methods

**`preprocess(df, ticker, fit_params)`**

- Apply preprocessing to a DataFrame
- `fit_params=True`: Compute and store preprocessing parameters (typically on training data)
- `fit_params=False`: Use stored parameters (typically on test data)
- Ensures train/test consistency

**`preprocess_by_ticker(df, ticker_col, fit_params)`**

- Apply preprocessing separately for each ticker
- Useful for multi-ticker datasets where each ticker should have its own parameters
- Maintains ticker independence

**`get_preprocessing_info()`**

- Returns stored preprocessing parameters per ticker
- Useful for auditing and debugging

**`reset_parameters()`**

- Clear all stored parameters
- Call when starting a new preprocessing session

## Integration with Training Pipeline

### Current Pipeline Order

```
1. Load CSV data
    ↓
2. Compute log returns (if needed)
    ↓
3. ✨ Winsorize returns (NEW)
    ↓
4. ✨ Apply EWMA smoothing (NEW)
    ↓
5. Normalize by ticker (z-score)
    ↓
6. Build lagged features
    ↓
7. Train models
```

### Why This Order?

1. **Winsorize first** → Remove extreme outliers before computing statistics
2. **EWMA next** → Smooth the already-cleaned data
3. **Normalize last** → Compute z-score statistics on cleaned, smoothed data
4. **Build features** → Create lags from preprocessed data

### Train/Test Consistency

The preprocessor ensures that train and test datasets use consistent preprocessing:

```python
# On training data split
train_data = df.loc[:train_end]
preprocessor.preprocess(train_data, ticker='AAPL', fit_params=True)  # ✓ Learns parameters

# On test data split
test_data = df.loc[train_end:]
preprocessor.preprocess(test_data, ticker='AAPL', fit_params=False)  # ✓ Uses stored parameters
```

**Important**: Using stored parameters on test data ensures that:

- Percentile thresholds are based only on training data
- No data leakage from test set to training
- Fair out-of-sample evaluation

## Configuration

### Default Settings (configs/default.yaml)

```yaml
preprocessing:
  apply_winsorization: true
  winsorize_lower: 0.01 # 1st percentile
  winsorize_upper: 0.99 # 99th percentile
  apply_ewma: true
  ewma_span: 20 # days of averaging
```

### Tuning Guidelines

**Winsorization percentiles:**

- `0.01, 0.99`: Conservative - removes only extreme outliers (~2% of data)
- `0.05, 0.95`: Moderate - removes more outliers (~10% of data)
- `0.10, 0.90`: Aggressive - affects 20% of data (use with caution)

**EWMA span:**

- `5-10`: Heavy smoothing (fast-decaying weights)
- `20`: Moderate smoothing (default)
- `50+`: Light smoothing (long-memory weights)
- `None`: Skip EWMA (no smoothing)

## Usage Examples

### Example 1: Basic Preprocessing

```python
from src.volforecast.data.preprocessor import FeaturePreprocessor, PreprocessingConfig

# Load data
df = pd.read_csv('data/AAPL_dataset.csv')

# Create preprocessor with default settings
config = PreprocessingConfig()
preprocessor = FeaturePreprocessor(config)

# Preprocess (fit parameters on training data)
df_clean = preprocessor.preprocess(df, ticker='AAPL', fit_params=True)
```

### Example 2: Custom Configuration

```python
config = PreprocessingConfig(
    apply_winsorization=True,
    winsorize_lower=0.05,
    winsorize_upper=0.95,
    apply_ewma=True,
    ewma_span=30,
)
preprocessor = FeaturePreprocessor(config)
```

### Example 3: Train/Test Split with Preprocessing

```python
# Split data
train_df = df.iloc[:300]
test_df = df.iloc[300:]

# Preprocess with train parameters
train_clean = preprocessor.preprocess(train_df, ticker='AAPL', fit_params=True)
test_clean = preprocessor.preprocess(test_df, ticker='AAPL', fit_params=False)

# Concatenate
df_clean = pd.concat([train_clean, test_clean])
```

### Example 4: Multi-ticker Data

```python
# Load all tickers
df_all = load_all_datasets('data/')

# Preprocess per-ticker
preprocessor = FeaturePreprocessor(PreprocessingConfig())
df_clean = preprocessor.preprocess_by_ticker(
    df_all,
    ticker_col='ticker',
    fit_params=True
)

# Each ticker has independent preprocessing parameters
params = preprocessor.get_preprocessing_info()
print(params)  # {'AAPL': {...}, 'MSFT': {...}, ...}
```

## Performance Considerations

- **Memory**: Creates copies of data; consider for large datasets
- **Speed**: Winsorization is O(n), EWMA is O(n) - both very fast
- **Numerical Stability**: Handles edge cases (empty series, single values, etc.)

## Testing

Comprehensive test suite in `tests/test_preprocessor.py`:

```bash
pytest tests/test_preprocessor.py -v
```

Tests cover:

- Configuration validation
- Winsorization correctness
- EWMA filtering behavior
- Parameter consistency between train/test
- Multi-ticker handling
- Edge cases (empty DataFrames, no numeric columns, etc.)

## Impact on Models

### GARCH

- Benefits from outlier removal (less variance in estimation)
- EWMA provides smoother conditional variance estimates
- Recommended settings: Conservative winsorization (0.01, 0.99)

### ElasticNet

- Most sensitive to outliers due to L1/L2 regularization
- EWMA reduces overfitting to noise
- Recommended settings: Moderate winsorization (0.05, 0.95), span=20

### XGBoost

- More robust to outliers than linear models
- EWMA still helps reduce overfitting
- Recommended settings: Conservative winsorization, span=20-30

## Troubleshooting

**Issue: "No preprocessing parameters found"**

- Solution: Call `preprocess(..., fit_params=True)` on training data first

**Issue: Preprocessing removes too much data**

- Solution: Reduce winsorization aggressiveness (e.g., 0.05, 0.95 → 0.01, 0.99)

**Issue: EWMA smoothing is too aggressive**

- Solution: Increase span parameter (e.g., 20 → 50)

## Future Enhancements

- Robust scaling alternatives (median absolute deviation)
- Multiple EWMA stacking
- Per-column adaptive thresholds
- Preprocessing performance monitoring and visualization
