# Volatility Forecasting: GARCH vs ML

V0.3

This repository compares **GARCH** models versus **Machine Learning** methods for **one-step-ahead volatility forecasting** on equity time series.

Disclaimer: Usage of generative AI for code generation

## Structure

```
.
├── configs/
│   └── default.yaml
├── data/                 # Place your CSVs here (kept out of git)
├── docs/
│   ├── 01_theory.md
│   ├── 02_models.md
│   ├── 03_code_reference.md
├── notebooks/
│   ├── 01_presentation_of_results.ipynb
├── reports/
│   └── figures/
├── scripts/
│   └── run_all.py
├── src/
│   └── volforecast/
│       ├── data/
│       │   └── loader.py #To be done
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── backtest.py
│       ├── features/
│       │   ├── builders.py
│       ├── models/
│       │   ├── base.py
│       │   ├── garch_model.py
│       │   ├── elasticnet_regression_model.py
│       │   ├── lstm_model.py #to be done
│       │   └── xgboost_model.py
│       └── visualization/
│           └── plot.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Quickstart

1. **Install** (preferably in a virtualenv):

   ```bash
   pip install -e .   #[dev]
   ```

2. **Put your CSVs** in `data/`. Example expected columns:

   - `Date` (YYYY-MM-DD), `Close`, and optionally market/VIX series.
   - If you already computed log-returns, name them `Log_Return` etc.

3. **Open notebooks**:

   ```bash
   jupyter lab
   ```

4. **Run experiments**:
   ```bash
   python -m scripts.run_all #(Use absolute imports for consistent behavior across environments.)
   ```

## Notes

- Each model lives in its own **Python script**, with a common base in `base.py`
- Data stays in `data/` as CSVs (not committed).
- Results and charts go to `reports/figures/`.

## Overview

Financial volatility — the variability of asset returns — is central in risk management, derivative pricing, and portfolio optimization.
This project compares statistical and machine learning approaches.

| Model Type           | Example               | Description                                                                  |
| -------------------- | --------------------- | ---------------------------------------------------------------------------- |
| **Econometric**      | GARCH(1,1)            | Captures volatility clustering using conditional variance dynamics.          |
| **Machine Learning** | ElasticNet Regression | Linear model with L1/L2 regularization; interpretable and robust.            |
| **Machine Learning** | XGBoost               | Nonlinear gradient-boosted trees learning from lagged features and VIX data. |

---

## Key Components

| File                                                    | Purpose                                                           |
| ------------------------------------------------------- | ----------------------------------------------------------------- |
| `src/volforecast/models/garch_model.py`                 | Implements GARCH(1,1) using `arch`.                               |
| `src/volforecast/models/elasticnet_regression_model.py` | ElasticNet regressor with lagged features.                        |
| `src/volforecast/models/xgboost_model.py`               | Gradient-boosted tree model for nonlinear volatility forecasting. |
| `src/volforecast/features/builders.py`                  | Creates lag features (returns, VIX, day-of-week dummies).         |
| `src/volforecast/evaluation/metrics.py`                 | Defines RMSE, MAE, and QLIKE.                                     |
| `src/volforecast/evaluation/backtest.py`                | Rolling-window backtesting framework.                             |
| `scripts/run_all.py`                                    | Runs all models sequentially and stores results.                  |
| `notebooks/01_presentation_of_results.ipynb`            | Visualizes and compares forecasts.                                |

---

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: measures average forecast error magnitude.
- **MAE (Mean Absolute Error)**: measures average absolute deviation.
- **QLIKE (Quasi-Likelihood)**: penalizes miscalibrated volatility forecasts more heavily — standard in volatility forecasting research.

---

## Theoretical Background

- **GARCH(1,1)** assumes the conditional variance evolves as:

  $$
  \sigma*t^2 = \omega + \alpha \epsilon*{t-1}^2 + \beta \sigma\_{t-1}^2
  $$

  capturing _volatility clustering_.

- **ElasticNet Regression** combines Lasso (L1) and Ridge (L2) regularization:

  $$
  \min_{\beta} \|y - X\beta\|\_2^2 + \alpha[(1 - l_1)\|\beta\|_2^2 + l_1\|\beta\|_1
  $$

  controlling both overfitting and feature selection.

- **XGBoost Volatility Model**

  Uses lagged returns, lagged VIX, and calendar features (day of week).
  Supports log-transform of variance for stability.
  Can be cross-validated via GridSearchCV or TimeSeriesSplit to tune:

  - max_depth
  - learning_rate
  - subsample
  - colsample_bytree
  - n_estimators

The ElasticNet and XGBoost are fed with:

- lagged returns,
- lagged realized volatility,
- rolling mean and standard deviation of volatility,
- optional log transformations for variance stabilization.

---

## Detailed Documentation

See the [`docs/`](docs/) folder for:

- Theoretical background (`01_theory.md`)
- Model details (`02_models.md`)
- Code reference (`03_code_reference.md`)

# TODO:

- Import the data loader and test it
- Add a filter to the models to see how they perform with it
- Containerize the project
- For Saïd: adapt the nn to compare it to other models
