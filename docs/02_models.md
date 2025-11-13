# 02 Model Specifications and Implementation Details

This document provides a formal description of the volatility forecasting models implemented in the repository. All models share a common interface defined by `BaseVolModel`, enabling unified evaluation, backtesting, and comparison.

## 1. Base Framework

All forecasting models inherit from:

```python
class BaseVolModel(Generic[C])
```

where `C` is a configuration dataclass derived from `BaseConfig`.
Every model implements three required components:

- `fit(df)`: estimation on a training window,
- `predict(df)`: one-step-ahead prediction based on available information,
- `build_target(df)`: construction of the realized variance proxy.

The framework uses the following realized variance:

$$
y_{t} = r_{t+\Delta}^2,
\qquad \Delta = 1 \text{ (one-step-ahead shift)}
$$

with configurable shift through `BaseConfig.target_shift`.
The base evaluation method computes:

- RMSE
- MAE
- QLIKE

on the aligned predictions and realized values.

This structure ensures full comparability between GARCH and machine learning models.

---

## 2. GARCH Model

### 2.1. Specification

The implemented model corresponds to:

$$r_t = \sigma_t \epsilon_t, \qquad \epsilon_t \sim \mathcal{D}$$,

$$
\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2 + \gamma r_{t-1}^2 \mathbf{1}{r_{t-1}<0} \quad (\text{if } \sigma>0)
$$

depending on the parameters:

- $p$ (ARCH terms)
- $q$ (GARCH terms)
- $o$ (asymmetry terms for GJR model)
- error distribution $\mathcal{D} \in {\text{Normal}, t, \text{GED}, \text{Skew-}t, \dots}$
- mean specification (zero, constant, AR, or HAR structure)

These options correspond to the parameters in `GARCHConfig`.

---

### 2.2. Data Flow and Implementation

- During `fit(df)`, the model extracts the return series and passes it to `arch_model` from the `arch` library.
- The fit is performed with likelihood maximization (BFGS or similar) via `engle.grt`.
- The fitted object is stored in `self.res_`.

#### Forecasting

The one-step-ahead conditional variance is obtained through:

```python
h = res.forecast(horizon=1).variance.iloc[:, 0]
```

yielding:

$$\widehat{\sigma}_{t+1}^2$$

The prediction is shifted by one period and aligned with the user-provided index.

Optional rescaling (e.g., for normalized return inputs) is handled via:

```python
h * scale_factor**2
```

---

### 2.3. Summary

The `summary()` method returns the raw `arch` model results.
This model serves as the **econometric benchmark**, due to its interpretability and strong theoretical foundation.

---

## 3. ElasticNet Regression Model

### 3.1. Mathematical Formulation

The ElasticNet predictor is based on a linear model:

$$y_t = X_t^\top \beta + \varepsilon_t$$,

where:

- $X_t$ contains lagged returns, lagged VIX values, and optional calendar dummies,
- $y_t = \log(r_{t+1}^2 + \varepsilon)$ if log-target mode is activated.

ElasticNet minimizes:

$$\min_{\beta}\big\lVert y - X\beta \big\rVert_2^2 + \alpha \left( (1 - \lambda) \lVert \beta \rVert_2^2 + \lambda \lVert \beta \rVert_1 \right)$$,

with:

- $\alpha \in {\alpha_1, \dots}$
- $\lambda$ = L1 ratio, typically in ${0.1, 0.5, 0.9}$

Cross-validation is performed using `ElasticNetCV`.

---

### 3.2. Feature Construction

ElasticNet uses the feature builder:

- lagged log-returns: $r_{t-L}$,
- lagged VIX values,
- day-of-week dummy variables.

Let:

$$
X_t = \left(
r_{t-L_1}, \dots,
\mathrm{VIX}_{t-L'_1}, \dots,
\text{DOW}_t
\right)
$$

The resulting feature matrix is passed through:

- optional `StandardScaler`,
- ElasticNetCV estimator.

---

### 3.3. Implementation Details

Key steps in `fit(df)`:

1. Construct features via `FeatureBuilder`.
2. Construct targets via `build_target()`.
3. Log-transform targets if `use_log_target=True`.
4. Remove rows containing NaNs.
5. Fit pipeline: `[StandardScaler?, ElasticNetCV]`.

Prediction inverts the log transformation:

$$\widehat{y}_t = \exp(\widehat{\eta}_t) - \varepsilon$$.

Negative predictions are clipped to zero.

---

### 3.4. Summary Output

The summary includes:

- selected optimal $\alpha$,
- selected optimal L1 ratio,
- coefficient vector as a dictionary.

ElasticNet is the **interpretable ML baseline**.

---

## 4. XGBoost Regression Model

### 4.1. Model Structure

The XGBoost model estimates:

$$y_t = f(X_t; \Theta)$$,

where (f) is a sum of regression trees fitted with gradient boosting.
Just as in the ElasticNet model:

- $X_t$ contains lagged returns, lagged VIX, and day-of-week features.
- $y_t$ is the log-transformed realized variance when `use_log_target=True`.

The fitted function is:

$$f(x) = \sum_{m=1}^M \eta_m h_m(x)$$,

with shrinkage parameter $\eta$ and maximum tree depth specified via configuration.

---

### 4.2. Implementation

During `fit(df)`:

1. Build feature matrix $X$ via `FeatureBuilder`.
2. Compute realized variance target $y_t$.
3. Apply log-transform if activated.
4. Remove rows with missing values.
5. Fit:

```python
XGBRegressor(
    n_estimators,
    learning_rate,
    max_depth,
    subsample,
    colsample_bytree,
)
```

Currently, the implementation uses **direct training** without cross-validation.
Future versions may integrate:

- time-series aware cross-validation (`TimeSeriesSplit`),
- randomized hyperparameter search,
- early stopping.

---

### 4.3. Prediction

The model predicts:

$$\widehat{\eta}_t = f(X_t)$$.

Log inversion is applied:

$$
\widehat{y}_t =
\max\left( \exp(\widehat{\eta}_t) - \varepsilon,; 0 \right)$$.


Predictions are reindexed to align with the full DataFrame index.

---

### 4.4. Summary Output

The summary includes:

* number of features used,
* full set of model hyperparameters,
* feature importances as returned by XGBoost.

XGBoost represents the **nonlinear ML benchmark** in this project.

---

## 5. Shared Feature Builder

The `FeatureBuilder` class provides a standardized set of predictors:

* lagged returns: $r_{t-L}$,
* lagged VIX values,
* optional day-of-week dummies.

Formally:


$$X_t =
\begin{bmatrix}
r_{t-L_1}, \dots, r_{t-L_k}, \
\mathrm{VIX}_{t-L'_1}, \dots, \mathrm{VIX}_{t-L'_\ell}, \
\mathbf{1}_{ \text{DOW}=d }
\end{bmatrix}
$$

This ensures that all ML models are trained on the same input space, enabling controlled comparison against GARCH.

---

## 6. Backtesting and Evaluation

All models are integrated into the unified rolling-window backtesting framework:

- expanding or fixed training window,
- time-based stepping (`1D`, `1B`, etc.),
- per-step re-estimation and prediction.

The backtest outputs:

- aligned series of predictions and true variances,
- evaluation metrics (RMSE, MAE, QLIKE),
- the number of effective predictions.

This procedure enforces strict **no lookahead bias** and ensures methodological consistency.

---

## 7. Summary Table

| Model      | Type         | Functional Form               | Features Used            | Strengths                            | Limitations               |
| ---------- | ------------ | ----------------------------- | ------------------------ | ------------------------------------ | ------------------------- |
| GARCH(1,1) | Econometric  | Parametric variance recursion | Returns only             | Interpretable, theoretical grounding | Rigid functional form     |
| ElasticNet | Linear ML    | Penalized linear model        | Lagged returns, VIX, DOW | Interpretable, stable                | Linear interactions only  |
| XGBoost    | Nonlinear ML | Gradient-boosted trees        | Same as ElasticNet       | Captures nonlinearities              | Higher computational cost |
