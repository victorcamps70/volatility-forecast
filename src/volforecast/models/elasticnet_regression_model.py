from typing import Tuple, List, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

from src.volforecast.models.base import BaseVolModel, BaseConfig
from src.volforecast.features.builders import FeatureBuilder


@dataclass
class ElasticNetConfig(BaseConfig):
    alphas: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    l1_ratio: Tuple[float, ...] = (0.1, 0.5, 0.9)
    cv_splits: int = 5
    use_log_target: bool = True
    scale_features: bool = True


class ElasticNetVolModel(BaseVolModel):
    def __init__(self, config: ElasticNetConfig, feature_builder: FeatureBuilder):
        super().__init__(config)
        self.feature_builder = feature_builder
        self.pipeline: Pipeline | None = None
        self.fitted_: bool = False
        self.coef_: pd.Series | None = None

    # --- Private helpers
    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.feature_builder.build_features(df)

    def _build_y(self, df: pd.DataFrame) -> pd.Series:
        y = self.build_target(df)  # from BaseVolModel
        if self.config.use_log_target:
            y = np.log(y + self.config.eps)
        return y.rename("y")

    # --- Required interface
    def fit(self, df: pd.DataFrame) -> "ElasticNetVolModel":
        X = self._build_X(df)
        y = self._build_y(df)
        valid = X.notna().all(axis=1) & y.notna()
        X, y = X[valid], y[valid]
        tscv = TimeSeriesSplit(
            n_splits=self.config.cv_splits
        )  # to avoid random shuffling in the window we're watching

        steps: List[Tuple[str, Any]] = []
        if self.config.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "enet",
                ElasticNetCV(
                    alphas=self.config.alphas,
                    l1_ratio=self.config.l1_ratio,
                    cv=tscv,
                    max_iter=10000,
                    n_jobs=None,
                    random_state=42,
                ),
            )
        )
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X, y)
        self.fitted_ = True

        # store coefficients for summary (map to feature names if scaler used)
        enet = self.pipeline.named_steps["enet"]
        self.coef_ = pd.Series(enet.coef_, index=X.columns, name="coef")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        assert self.fitted_, "Call fit() first."
        assert self.pipeline is not None, "Call fit() first"
        X = self._build_X(df)

        # keep only rows with all features present
        valid = X.notna().all(axis=1)
        Xv = X[valid]

        # predict only on valid rows
        yhat = pd.Series(self.pipeline.predict(Xv), index=Xv.index, name="y_pred")

        # put predictions back onto the full index (NaN where features are missing)
        yhat = yhat.reindex(X.index)

        # invert log if used
        if self.config.use_log_target:
            # only invert where we have predictions
            yhat.loc[valid] = np.exp(yhat.loc[valid]) - self.config.eps

        return yhat.clip(lower=0.0).rename("y_pred")

    # optional
    def summary(self) -> Dict[str, Any]:
        assert self.pipeline is not None, "Call fit() first"
        return {
            "coefficients": None if self.coef_ is None else self.coef_.to_dict(),
            "best_alpha": getattr(self.pipeline.named_steps["enet"], "alpha_", None),
            "best_l1_ratio": getattr(self.pipeline.named_steps["enet"], "l1_ratio_", None),
        }


if __name__ == "__main__":

    print("Running basic self-test for ElasticNetVolModel...")

    # 1. Create small dummy dataset
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "log_return_AAPL": [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.0, 0.01, -0.005, 0.002],
            "vix": [20, 21, 19, 18, 22, 21, 20, 19, 20, 22],
        }
    ).set_index("date")

    # 2. Build config and feature builder
    base_cfg = BaseConfig(date_col="date", return_col="log_return_AAPL")
    enet_cfg = ElasticNetConfig(**base_cfg.__dict__)
    builder = FeatureBuilder(lags_returns=(1, 2))
    model = ElasticNetVolModel(enet_cfg, builder)

    # 3. Fit & predict
    model.fit(df)
    y_pred = model.predict(df)

    print("Predictions:")
    print(y_pred.head())
