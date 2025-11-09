# Volatility Forecasting: GARCH vs ML

V0.1

This repository compares **GARCH** models versus **Machine Learning** methods for **one-step-ahead volatility forecasting** on equity time series.

## Structure

```
.
├── configs/
│   └── default.yaml
├── data/                 # Place your CSVs here (kept out of git)
├── notebooks/
│   ├── 01_presentation_of_results.ipynb
├── reports/
│   └── figures/
├── scripts/
│   └── run_all.py #To be done
├── src/
│   └── volforecast/
│       ├── data/
│       │   └── loader.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── backtest.py
│       └── models/
│           ├── garch_model.py
│           └── elasticnet_regression_model.py
├── .gitignore
├── pyproject.toml
└── README.md
```

## Quickstart

1. **Install** (preferably in a virtualenv):

   ```bash
   pip install -e .[dev]
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
   python scripts/run_all.py --csv data/AAPL.csv --ticker AAPL
   ```

## Notes

- Each model lives in its own **Python script**, mirroring your style.
- Data stays in `data/` as CSVs (not committed).
- Results and charts go to `reports/figures/`.

## Overview

Financial volatility — the variability of asset returns — plays a central role in risk management, option pricing, and portfolio allocation.  
This project explores two complementary modeling paradigms:

| Model Type           | Example               | Description                                                                            |
| -------------------- | --------------------- | -------------------------------------------------------------------------------------- |
| **Econometric**      | GARCH(1,1)            | Captures conditional heteroskedasticity — the idea that volatility clusters over time. |
| **Machine Learning** | ElasticNet regression | Learns volatility from lagged returns, realized volatility, and rolling statistics.    |

Both models are compared on accuracy metrics such as **RMSE**, **MAE**, and **QLIKE**.

---

## ⚙️ Key Components

| File                                                    | Purpose                                                                         |
| ------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `src/volforecast/models/garch_model.py`                 | Implements the GARCH(1,1) model using the `arch` package.                       |
| `src/volforecast/models/elasticnet_regression_model.py` | Implements an ElasticNet volatility regressor with lagged and rolling features. |
| `src/volforecast/evaluation/metrics.py`                 | Contains RMSE, MAE, and QLIKE loss functions.                                   |
| `src/volforecast/evaluation/backtest.py`                | Implements a rolling backtest framework.                                        |
| `src/volforecast/data/loader.py`                        | Loads CSV files and prepares log returns.                                       |
| `scripts/run_all.py`                                    | Command-line experiment runner comparing models.                                |
| `notebooks/01_preentation_of_results.ipynb`             | Interactive exploration and visualization of results.                           |

---

## 📊 Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: measures average forecast error magnitude.
- **MAE (Mean Absolute Error)**: measures average absolute deviation.
- **QLIKE (Quasi-Likelihood)**: penalizes miscalibrated volatility forecasts more heavily — standard in volatility forecasting research.

---

## 🧠 Theoretical Background

- **GARCH(1,1)** assumes the conditional variance evolves as:

  $$
  \sigma*t^2 = \omega + \alpha \epsilon*{t-1}^2 + \beta \sigma\_{t-1}^2
  $$

  capturing _volatility clustering_.

- **ElasticNet Regression** combines Lasso (L1) and Ridge (L2) regularization:

  $$
  \min_{\beta} \|y - X\beta\|_2^2 + \alpha ((1 - l_1)\|\beta\|_2^2 + l_1\|\beta\|_1)
  $$

  $$
  y - X \beta _2^2
  $$


  controlling both overfitting and feature selection.

The ElasticNet is fed with:

- lagged returns,
- lagged realized volatility,
- rolling mean and standard deviation of volatility,
- optional log transformations for variance stabilization.

---

## 📚 Detailed Documentation

See the [`docs/`](docs/) folder for:

- Theoretical background (`01_theory.md`)
- Model details (`02_models.md`)
- Code reference (`03_code_reference.md`)

# TODO:

- Test the GARCH Model in the notebook
- Comment the results of the GARCH Model
- Comment the results of the ENet CV Model
- Code the LSTM Model
- Code the NN Model
- Test the LSTM Model
- Test the NN Model
- Comment results of the LSTM Model
- Comment results of the NN Model
- Make sure there's no redundancies between Enet and evaluation, and evaluation files
- Comment the choices made on the Enet Model
- Make the .md that are still not there
- Import the results into reports --> not done yet
- Import the data loader and test it
- Update the format of the .py (mettre à niveau mes .py pour qu'ils soient tous pareil)
- Add a run_all.py to make sure that we can run all the results at once
- Finetune the models to have better results
- Add stress and time tests to better the code
