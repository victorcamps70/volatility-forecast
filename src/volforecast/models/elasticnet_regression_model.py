"""
ElasticNet Regression Model
@author: Victor Camps, Saïd Tellez and Steven Yu with contributions from ChatGPT
@author_email: victorcamps70@gmail.com
@date: 17/10/2025
@description: This script implements an ElasticNet regression model using scikit-learn.
"""

# TODO: Define ReadME to explain the model, its parameters, and how to use it.
# TODO: look if we can optimize imports, and remove functions into other .py if needed

## Imports

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from ..evaluation.metrics import qlike_loss, rmse, mae


## Création du setup pour le modèle
@dataclass
class ElasticNetRegression_config:
    # Dataset parameters
    date: str = "Date"
    stock_close: str = "AAPL_Close"
    market_close: str = "SP500_Close"
    vix_close: str = "VIX_Close"
    open: str = "Open"
    high: str = "High"
    low: str = "Low"
    close: str = "Close"
    volume: str = "Volume"
    log_return_stock: str = "Log_Return"
    log_return_SP500: str = "Log_Return_SP500"
    log_return_VIX: str = "Log_Return_VIX"

    # Model parameters
    n_splits: int = 5
    random_state: int = 42
    alphas: np.ndarray = (
        None  # we can define a range of penalization strengths to search over (lower alpha = less regularization)
    )
    l1_ratio: np.ndarray = None  # ElasticNet mixing parameter between Ridge and Lasso
    max_iter: int = 20000

    # Feature engineering parameters
    lags: List[int] = field(
        default_factory=lambda: [5, 10]
    )  # how many days to look back for lagged features
    rolling_windows: List[int] = field(
        default_factory=lambda: [5, 10]
    )  # window sizes for rolling statistics
    include_calendar_features: bool = (
        True  # whether to include calendar features like day of week (Monday, Tuesday, etc.) bcs we may see weekly patterns
    )
    log_transform: bool = (
        True  # whether to log transform the target variable to stabilize variance and make relationships more linear (also gives always positive predictions)
    )
    scale_features: bool = True  # whether to standardize features to have mean 0 and variance 1
    test_size: float = 0.2  # proportion of data to use for testing

    # Target
    target: str = "next_squared_return_stock"
    target_horizon: str = None  # provided if target == "5_day_ahead_return_stock"
    target_offset: float = 1e-6  # small offset to avoid log(0) if log_transform is True


## Création du modèle
class ElasticNetRegression:
    def __init__(self, config: ElasticNetRegression_config):
        """
        Initialize the ElasticNet Regression model with the given configuration.
        """
        self.config = config
        self.features: List[str] = []  # to be filled later
        if self.config.alphas is None:
            self.config.alphas = np.logspace(-5, 1, 60)
        if self.config.l1_ratio is None:
            self.config.l1_ratio = np.linspace(0.1, 0.9, 9)
        self.features: List[str] = []
        self.fitted: bool = False  # Flag to indicate if the model has been fitted
        self.pipeline: Optional[Pipeline] = (
            None  # Pipeline for preprocessing and modeling (c'est pour éviter d'oublier de transformer les données)
        )

    # Data

    def _build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the target column correspondingly to what we're looking for (next squared return or 5-day ahead return).

        Raises:
            ValueError: if the target specified in the configuration is invalid (i.e., not 'next_squared_return_stock' or '5_day_ahead_return_stock')

        Returns:
            pd.DataFrame: data with the target column added

        """
        if self.config.target == "next_squared_return_stock":
            df[self.config.target] = (
                df[self.config.log_return_stock].shift(-1) ** 2
            )  # next day squared return: variance proxy
        elif self.config.target == "5_day_ahead_return_stock":
            df[self.config.target] = (
                df[self.config.log_return_stock]
                .rolling(window=5)
                .apply(lambda x: np.sqrt(np.mean(x**2)))
                .shift(-4)
            )  # 5-day ahead return: volatility proxy
        else:
            raise ValueError("Invalid target specified in configuration.")
        return df[self.config.target]  # return only the target column to be merged later

    def _add_day_of_the_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds day of the week as categorical features to the dataframe.

        Returns:
            pd.DataFrame: data with day of the week features added
        """
        df[self.config.date] = pd.to_datetime(df[self.config.date])
        df["day_of_week"] = df[self.config.date].dt.day_name()
        day_dummies = pd.get_dummies(
            df["day_of_week"], prefix="day", drop_first=True, dtype=int
        )  # one hot encoding of the day of the week, dropping the first to avoid multicollinearity
        df = pd.concat([df.drop(columns=["day_of_week"]), day_dummies], axis=1)
        return df

    def _build_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Builds the feature matrix X and target vector y from the dataframe.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: feature matrix X and target vector y
        """
        # Ensure time ordering
        df = df.sort_values(by=self.config.date).reset_index(drop=True)  # reset index after sorting

        # Create target variable
        df[self.config.target] = self._build_target(df)
        out = df.copy()  # on essaie de faire les choses proprement en faisant une copie

        # We add base realized volatility in the features
        out["realized_volatility"] = out[self.config.log_return_stock] ** 2

        # Adding lagged features
        for lag in self.config.lags:
            out[f"lag_returns_{lag}"] = out[self.config.log_return_stock].shift(lag)
            out[f"lag_volatility_{lag}"] = out["realized_volatility"].shift(lag)

            out[f"lag_vix_{lag}"] = out[self.config.vix_close].shift(lag)
            out[f"lag_log_return_vix_{lag}"] = out[self.config.log_return_VIX].shift(lag)

            out[f"lag_market_close_{lag}"] = out[self.config.market_close].shift(lag)
            out[f"lag_market_return_{lag}"] = out[self.config.log_return_SP500].shift(lag)

        # Rolling stats on volatility
        for window in self.config.rolling_windows:
            out[f"rolling_volatility_mean_{window}"] = (
                out["realized_volatility"].rolling(window).mean()
            )
            out[f"rolling_volatility_std_{window}"] = (
                out["realized_volatility"].rolling(window).std()
            )

        # Include calendar features
        if self.config.include_calendar_features:
            out = self._add_day_of_the_week(out)

        # Dropping non numeric columns (cannot be features) and cols used to calculate other features

        dropped_columns = {
            self.config.date,  # NaN
            self.config.log_return_stock,  # We are using the rolling one
            "realized_volatility",  # wtf ?!
            self.config.market_close,
            self.config.vix_close,
            self.config.target,
            self.config.stock_close,
            self.config.open,
            self.config.high,
            self.config.low,
            self.config.close,
            self.config.volume,
            self.config.log_return_SP500,
        }

        dropped_columns = {
            c for c in dropped_columns if c in out.columns
        }  # We don't keep the column only if it is in the features array

        X = out.drop(columns=list(dropped_columns)).select_dtypes(include=[np.number])
        y = out[self.config.target].astype(float)
        if self.config.log_transform:
            eps = float(self.config.target_offset)
            y = np.log(
                np.clip(y, 0.0, None) + eps
            )  # log transform with small offset to avoid log(0)

        # Drop rows with NaN values resulting from feature engineering
        data = pd.concat([X, y.rename("target")], axis=1).dropna()
        X, y = data.drop(columns=["target"]), data["target"]

        self.features = list(X.columns)
        return X, y

    # Tuning, fitting, predicting, evaluating

    def fit(self, df: pd.DataFrame) -> "ElasticNetRegression":
        """
        Fit the ElasticNet regression model to the provided dataframe.

        Args:
            df (pd.DataFrame): The input dataframe containing features and target.

        Returns:
            ElasticNetRegression: The fitted model instance.
        """
        X, y = self._build_X_y(df)

        # Train-test split but maintaining time order
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)

        # We create the pipeline steps to apply always the same steps to train and test
        steps = []
        if self.config.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "elasticnet",
                ElasticNetCV(
                    alphas=self.config.alphas,
                    l1_ratio=self.config.l1_ratio,
                    max_iter=self.config.max_iter,
                    cv=tscv,
                    random_state=self.config.random_state,
                ),
            )
        )

        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X, y)
        self.fitted = True

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict using the fitted ElasticNet regression model.

        Args:
            df (pd.DataFrame): The input dataframe containing features.

        Returns:
            np.ndarray: The predicted target values.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction.")

        X, _ = self._build_X_y(
            df
        )  # We only need X for prediction, we go through the same feature engineering steps as in fit
        predictions = self.pipeline.predict(X)
        if self.config.log_transform:
            eps = float(self.config.target_offset)
            predictions = np.exp(predictions) - eps  # inverse of log transform with offset
            predictions = np.clip(predictions, 0.0, None)  # ensure non-negative predictions

        return pd.Series(predictions, index=X.index, name="predictions")

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate the fitted model on the provided dataframe.

        Args:
            df (pd.DataFrame): The input dataframe containing features and target.

        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before evaluation.")

        X, _ = self._build_X_y(df)
        target_raw = self._build_target(df)
        target_raw = target_raw.loc[X.index]

        # predictions
        y_pred = self.pipeline.predict(X)
        if self.config.log_transform:
            eps = float(self.config.target_offset)
            y_pred_raw = np.exp(y_pred) - eps  # inverse of log transform with offset
            y_pred_raw = np.clip(y_pred_raw, 0.0, None)  # ensure non-negative predictions
        else:
            y_pred_raw = y_pred

        # clip to avoid negative predictions if target is squared return
        y_true = target_raw.clip(lower=0).astype(float)
        y_hat = np.maximum(y_pred_raw, 0.0).astype(float)

        rmse_value = rmse(y_true, y_hat)
        mae_value = mae(y_true, y_hat)
        qlike_value = qlike_loss(y_true.values, y_hat)

        coefficients = None
        enet = self.pipeline.named_steps.get("elasticnet")
        if enet is not None and hasattr(enet, "coef_"):
            coefficients = dict(zip(self.features, enet.coef_.tolist()))

        return {
            "RMSE": rmse_value,
            "MAE": mae_value,
            "QLIKE": qlike_value,
            "best_alpha": float(self.pipeline.named_steps["elasticnet"].alpha_),
            "best_l1_ratio": float(self.pipeline.named_steps["elasticnet"].l1_ratio_),
            "n_features": len(self.features),
            "coefficients": coefficients,
        }

    # Application of the model

    def train_test_split(
        self, n_test: int, df: pd.DataFrame = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits the dataframe into training, validation, and testing sets while maintaining time order.

        Args:
            n_test (int): Number of samples for the test set.
            df (pd.DataFrame, optional): The input dataframe to split. If None, uses the internal dataframe.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Indices for training, validation, and testing sets.
        Raises:
            ValueError: If n_test is not a positive integer less than the number of samples.
        """
        if df is None:
            raise ValueError("Dataframe must be provided for train-test split.")

        X, _ = self._build_X_y(df)
        n_samples = len(X)
        if n_test <= 0 or n_test >= n_samples:
            raise ValueError("n_test must be a positive integer less than the number of samples.")
        test_idx = np.arange(n_samples - n_test, n_samples)
        train_idx = np.arange(0, n_samples - n_test)
        return train_idx, test_idx

    def fit_on_train_predict_on_test(self, df: pd.DataFrame, n_test: int = 252) -> Dict[str, Any]:
        """
        Fits the model on the training set and predicts on the test set.

        Args:
            n_test (int): Number of samples for the test set.
            df (pd.DataFrame): The input dataframe containing features and target.
        Returns:
            Dict[str, Any]: A dictionary containing evaluation metrics on the test set.
        Raises:
            ValueError: If n_test is not a positive integer less than the number of samples.
        """
        X, y = self._build_X_y(df)
        N = len(X)
        if n_test <= 0 or n_test >= N:
            raise ValueError("n_test must be a positive integer less than the number of samples.")
        X_train, X_test = X.iloc[: N - n_test], X.iloc[N - n_test :]
        y_train, y_test = y.iloc[: N - n_test], y.iloc[N - n_test :]

        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        steps = [
            ("scaler", StandardScaler()) if self.config.scale_features else None,
            (
                "elasticnet",
                ElasticNetCV(
                    alphas=self.config.alphas,
                    l1_ratio=self.config.l1_ratio,
                    max_iter=self.config.max_iter,
                    cv=tscv,
                    random_state=self.config.random_state,
                ),
            ),
        ]
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X_train, y_train)

        # Predictions (raw no log)
        y_pred = self.pipeline.predict(X_test)
        if self.config.log_transform:
            y_pred = np.expm1(y_pred)

        # Raw target
        target_raw = self._build_target(df)
        target_raw = target_raw.loc[X_test.index].clip(lower=0).astype(float)

        metrics = {
            "RMSE": float(np.sqrt(rmse(target_raw, np.maximum(y_pred, 0.0).astype(float)))),
            "MAE": float(mae(target_raw, np.maximum(y_pred, 0.0).astype(float))),
            "QLIKE": qlike_loss(target_raw.values, np.maximum(y_pred, 0.0).astype(float)),
            "best_alpha": float(self.pipeline.named_steps["elasticnet"].alpha_),
            "best_l1_ratio": float(self.pipeline.named_steps["elasticnet"].l1_ratio_),
            "n_features": len(self.features),
        }
        return metrics
