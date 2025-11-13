# 01 Theoretical Foundations of Volatility Modelling

## 1. Introduction

Volatility is a fundamental concept in financial econometrics. It quantifies the magnitude of fluctuations in asset returns and is used extensively in risk management, derivative pricing, portfolio allocation, and market microstructure analysis. Its accurate forecasting is thus an essential task.

This document presents the mathematical definition of volatility, its empirical properties, and the theory underlying the Generalized Autoregressive Conditional Heteroskedasticity (GARCH) family of models, which serves as the econometric baseline in this project.

---

## 2. Returns and Volatility

### 2.1. Log-Returns

Let $P_t$ denote the asset price at time $t$. The log-return is defined as:

$$r_t = \log P_t - \log P_{t-1}$$

Log-returns are preferred due to their additive structure and compatibility with continuous-time models.

---

### 2.2. Volatility as Conditional Variance

The unconditional variance of returns is:

$$\sigma^2 = \mathrm{Var}(r_t)$$

However, empirical evidence indicates that return variance varies over time. Volatility is therefore modelled as a **conditional variance**:

$$\sigma_t^2 = \mathrm{Var}(r_t \mid \mathcal{F}_{t-1}) $$,

where $\mathcal{F}_{t-1}$ represents the information set available up to time $t-1$. Forecasting $\sigma_{t+1}^2$ constitutes the primary objective of volatility modelling.

---

## 3. Stylized Facts of Financial Returns

Financial returns exhibit several well-documented empirical regularities:

1. **Volatility clustering**: large returns tend to be followed by large returns, and small returns by small returns; $r_t^2$ shows persistent autocorrelation.
2. **Heavy tails**: return distributions exhibit excess kurtosis relative to the Gaussian distribution.
3. **Time-varying volatility** with slow mean reversion.
4. **Leverage effects**: negative returns often lead to larger increases in volatility than positive returns of the same magnitude.

These characteristics motivate econometric models that explicitly allow for heteroskedasticity in the conditional variance.

---

## 4. ARCH and GARCH Models

### 4.1. ARCH(q) Model

Engle (1982) introduced the **Autoregressive Conditional Heteroskedasticity (ARCH)** model, where returns follow:

$$r_t = \sigma_t \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0,1)$$,

$$\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i r_{t-i}^2$$,

with constraints $\omega > 0$ and $\alpha_i \ge 0$.
ARCH models capture volatility clustering but usually require large $q$ to be effective, making them less parsimonious.

---

### 4.2. GARCH(1,1) Model

To address this issue, Bollerslev (1986) proposed the **Generalized ARCH (GARCH)** model. The most widely used specification is GARCH(1,1):

$$r_t = \sigma_t \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0,1)$$,

$$\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$$,

with parameters satisfying:

$$\omega > 0,\quad \alpha \ge 0,\quad \beta \ge 0,\quad \alpha + \beta < 1$$

This model provides a parsimonious yet effective representation of conditional variance dynamics in practice.

---

### 4.3. Interpretation of Parameters

- $\omega$: long-term average variance level.
- $\alpha$: sensitivity to new information (impact of lagged squared returns).
- $\beta$: persistence of past volatility.

The persistence coefficient $\alpha + \beta$ is typically close to 1 for financial assets, reflecting the slow mean reversion of volatility.

---

## 5. Volatility Forecasting with GARCH

### 5.1. One-Step-Ahead Forecast

Given information up to time $t$, the one-step-ahead conditional variance forecast is:

$$\widehat{\sigma}_{t+1}^2 = \omega + \alpha r_t^2 + \beta \sigma_t^2$$

This project evaluates the accuracy of such forecasts.

---

### 5.2. Realized Variance Proxy

Because the true latent variance $\sigma_t^2$ is not directly observable, a common proxy is:

$$\mathrm{RV}_{t+1} = r_{t+1}^2$$

For daily data, this proxy is widely used, although more refined estimators (e.g., realized volatility from high-frequency data) exist.

---

### 5.3. Evaluation Metrics

Forecast accuracy is assessed using:

- **Root Mean Squared Error (RMSE)**

  $$\mathrm{RMSE} = \sqrt{\mathbb{E}\left[(\widehat{\sigma}_{t+1}^2 - \mathrm{RV}_{t+1})^2\right]}$$,

- **Mean Absolute Error (MAE)**

  $$\mathrm{MAE} = \mathbb{E}\left[\lvert \widehat{\sigma}_{t+1}^2 - \mathrm{RV}_{t+1} \rvert\right]$$,

- **Quasi-Likelihood Loss (QLIKE)**
  $$\mathrm{QLIKE} = \mathbb{E}\left[\frac{\mathrm{RV}_{t+1}}{\widehat{\sigma}_{t+1}^2} + \log \widehat{\sigma}_{t+1}^2 \right]$$

QLIKE is widely used in volatility forecast evaluation because it remains consistent even when realized variance proxies contain measurement error.

---

## 6. Motivation for Machine Learning Comparisons

Although GARCH(1,1) provides a robust and interpretable framework, it imposes restrictive linearity assumptions and uses only a limited set of lagged terms.
Machine learning models allow for:

- nonlinear functional relationships,
- incorporation of exogenous covariates (e.g., VIX),
- richer feature sets (rolling statistics, calendar effects, etc.),
- flexible interactions between predictors.

Comparing GARCH with ML-based models therefore allows us to assess potential improvements in forecasting accuracy in the presence of nonlinearities and high-dimensional information.

---

## 7. Conclusion

This document has presented the theoretical background necessary for understanding volatility as a time-varying conditional variance and described the GARCH(1,1) model in its standard form.
GARCH remains the primary econometric benchmark due to its simplicity and empirical effectiveness.
The next step of the project evaluates whether modern machine learning methods can surpass this benchmark in one-day-ahead volatility forecasting using a unified backtesting and evaluation framework.
