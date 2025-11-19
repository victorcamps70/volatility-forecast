from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.volforecast.models.base import BaseVolModel, BaseConfig
from src.volforecast.features.builders import FeatureBuilder
from src.volforecast.evaluation.metrics import qlike_loss, rmse as rmse_loss, mae as mae_loss


@dataclass
class XGBoostConfig(BaseConfig):
    n_estimators_grid: tuple[int, ...] = (100, 200, 500)
    max_depth_grid: tuple[int, ...] = (2, 3, 4)
    learning_rate_grid: tuple[float, ...] = tuple(np.linspace(0.01, 0.1, 5))
    subsample_grid: tuple[float, ...] = (0.6, 0.8)
    colsample_bytree_grid: tuple[float, ...] = (0.6, 0.8)
    min_child_weight_grid: tuple[float, ...] = (5.0, 10.0, 20.0)
    gamma_grid: tuple[float, ...] = (0.0, 0.1, 0.5)
    reg_lambda_grid: tuple[float, ...] = (1.0, 5.0, 10.0)  # L2
    reg_alpha_grid: tuple[float, ...] = (0.0, 0.1, 1.0)
    use_log_target: bool = True
    cv_splits: int = 5  # Number of time series splits
    n_iter: int = 20  # Number of randomized search iterations
    scoring: str = (
        "qlike"  # Scoring metric for Cross validation: choose between qlike, rmse and mae
    )
    random_state: int = 42


class XGBoostVolModel(BaseVolModel):
    def __init__(self, config: XGBoostConfig, feature_builder: FeatureBuilder):
        super().__init__(config)
        self.feature_builder = feature_builder
        self.model: Optional[XGBRegressor] = None
        self.search_: Optional[RandomizedSearchCV] = None
        self.best_params_: Dict[str, Any] | None = None
        self.fitted_: bool = False

    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.feature_builder.build_features(df)

    def _build_y(self, df: pd.DataFrame) -> pd.Series:
        y = self.build_target(df)
        if self.config.use_log_target:
            y = np.log(y + self.config.eps)
        return y.rename("y")

    def fit(self, df: pd.DataFrame) -> "XGBoostVolModel":
        """
        Fitting function wrapper for the XGBoost model
        Args:
            df: dataframe
        Return:
            pd.Series: prediction
        """
        X = self._build_X(df)
        y = self._build_y(df)
        valid = X.notna().all(axis=1) & y.notna()
        X, y = X[valid], y[valid]

        base_model = XGBRegressor(
            random_state=self.config.random_state,
            objective="reg:squarederror",
            n_jobs=-1,
            tree_method="hist",
        )

        eps = self.config.eps
        use_log = self.config.use_log_target
        metric_name = self.config.scoring

        def _metric_for_cv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            """
            Internal function to the fit function,
            to use a QLIKE-based (or any other metric) random cross validation
            for the XGBoost model
            Args:
                y_true: true target
                y_pred: predicted target
            Returns:
                scorer
            """
            # y_true, y_pred are arrays on the SAME scale as y in _build_y
            if use_log:
                # our y is log(variance + eps) → go back to variance
                y_true_var = np.exp(y_true) - eps
                y_pred_var = np.exp(y_pred) - eps
            else:
                y_true_var = y_true
                y_pred_var = y_pred

            if metric_name == "qlike":
                val = qlike_loss(y_true_var, y_pred_var)
            elif metric_name == "rmse":
                val = rmse_loss(y_true_var, y_pred_var)
            elif metric_name == "mae":
                val = mae_loss(y_true_var, y_pred_var)
            else:
                # fallback: RMSE if someone passes something weird
                val = rmse_loss(y_true_var, y_pred_var)

            # sklearn maximises the score → return negative loss/metric
            return -float(val)

        scorer = make_scorer(_metric_for_cv, greater_is_better=True)

        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)  # Time-series–aware cross-validation

        param_distributions: Dict[str, Any] = {
            "n_estimators": self.config.n_estimators_grid,
            "max_depth": self.config.max_depth_grid,
            "learning_rate": self.config.learning_rate_grid,
            "subsample": self.config.subsample_grid,
            "colsample_bytree": self.config.colsample_bytree_grid,
            "min_child_weight": self.config.min_child_weight_grid,
            "gamma": self.config.gamma_grid,
            "reg_lambda": self.config.reg_lambda_grid,
            "reg_alpha": self.config.reg_alpha_grid,
        }

        if self.best_params_ is None:  # We do only one cross validation to reduce time of training
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=self.config.n_iter,
                cv=tscv,
                scoring=scorer,
                random_state=self.config.random_state,
                n_jobs=1,
                verbose=0,
            )

            search.fit(X, y)
            self.search_ = search
            self.best_params_ = search.best_params_
            self.model = cast(XGBRegressor, search.best_estimator_)

        else:
            best_params = dict(self.best_params_)
            best_params.update(
                {
                    "objective": "reg:squarederror",
                    "random_state": self.config.random_state,
                    "n_jobs": 1,
                    "tree_method": "hist",
                }
            )

            if best_params.get("max_depth", 6) > 5:
                best_params["max_depth"] = 5  # to avoid overfitting
            self.model = XGBRegressor(**best_params)
            self.model.fit(X, y)

        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predicting function wrapper for the XGBoost model
        Args:
            df: dataframe
        Return:
            pd.Series: prediction
        """
        assert self.fitted_, "Call fit() first."
        assert self.model is not None, "Call fit() first"
        X = self._build_X(df)
        valid = X.notna().all(axis=1)
        Xv = X[valid]
        yhat = pd.Series(self.model.predict(Xv), index=Xv.index, name="y_pred")

        if self.config.use_log_target:
            yhat.loc[valid] = np.exp(yhat.loc[valid]) - self.config.eps

        return yhat.reindex(df.index).clip(lower=0.0).rename("y_pred")

    def summary(self) -> Dict[str, Any]:
        if self.model is None:
            return {}

        info: Dict[str, Any] = {
            "n_features": getattr(self.model, "n_features_in_", None),
            "best_params": self.model.get_params(),
            "feature_importances": None,
        }

        # Feature names + importances if available
        if hasattr(self.model, "feature_names_in_") and hasattr(self.model, "feature_importances_"):
            info["feature_importances"] = dict(
                zip(
                    self.model.feature_names_in_,
                    self.model.feature_importances_.tolist(),
                )
            )

        # Extra info from randomized search
        if self.search_ is not None:
            info["cv_best_score"] = getattr(self.search_, "best_score_", None)
            info["cv_best_params"] = getattr(self.search_, "best_params_", None)

        return info


if __name__ == "__main__":
    import sys

    print(sys.executable)
    print("Running basic self-test for XGBoostVolModel with randomized CV...")

    # 1. Create small dummy dataset
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=80, freq="D"),
            "log_return_AAPL": np.random.normal(0, 0.02, size=80),
            "vix": np.random.uniform(15, 25, size=80),
        }
    ).set_index("date")

    # 2. Build config and feature builder
    base_cfg = BaseConfig(date_col="date", return_col="log_return_AAPL")
    xgboost_cfg = XGBoostConfig(**base_cfg.__dict__)
    builder = FeatureBuilder(lags_returns=(1, 2), lags_vix=(1, 2))

    model = XGBoostVolModel(xgboost_cfg, builder)

    # 3. Fit & predict
    model.fit(df)
    y_pred = model.predict(df)

    print("Predictions:")
    print(y_pred.head())

    print("\nSummary:")
    print(model.summary())
