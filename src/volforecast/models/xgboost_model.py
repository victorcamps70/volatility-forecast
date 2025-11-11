from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from src.volforecast.models.base import BaseVolModel, BaseConfig
from ..features.builders import FeatureBuilder


@dataclass
class XGBoostConfig(BaseConfig):
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    use_log_target: bool = True
    random_state: int = 42


class XGBoostVolModel(BaseVolModel):
    def __init__(self, config: XGBoostConfig, feature_builder: FeatureBuilder):
        super().__init__(config)
        self.feature_builder = feature_builder
        self.model: XGBRegressor | None = None
        self.fitted_: bool = False

    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.feature_builder.build_features(df)

    def _build_y(self, df: pd.DataFrame) -> pd.Series:
        y = self.build_target(df)
        if self.config.use_log_target:
            y = np.log(y + self.config.eps)
        return y.rename("y")

    def fit(self, df: pd.DataFrame) -> "XGBoostVolModel":
        X = self._build_X(df)
        y = self._build_y(df)
        valid = X.notna().all(axis=1) & y.notna()
        X, y = X[valid], y[valid]

        self.model = XGBRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
        )

        self.model.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        assert self.fitted_, "Call fit() first."
        X = self._build_X(df)
        valid = X.notna().all(axis=1)
        Xv = X[valid]
        yhat = pd.Series(self.model.predict(Xv), index=Xv.index, name="y_pred")

        if self.config.use_log_target:
            yhat.loc[valid] = np.exp(yhat.loc[valid]) - self.config.eps

        return yhat.reindex(df.index).clip(lower=0.0).rename("y_pred")

    def summary(self) -> Dict[str, Any]:
        if not self.model:
            return {}
        return {
            "n_features": self.model.n_features_in_,
            "best_params": self.model.get_params(),
            "feature_importances": dict(
                zip(self.model.feature_names_in_, self.model.feature_importances_.tolist())
            ),
        }
