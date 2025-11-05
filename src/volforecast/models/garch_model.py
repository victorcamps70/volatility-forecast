from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional
from arch import arch_model
from ..evaluation.metrics import qlike_loss, rmse, mae

# TODO: add docstrings, reconfigure to match other models


@dataclass
class GARCHConfig:
    date: str = "Date"
    log_return_stock: str = "Log_Return"
    p: int = 1
    q: int = 1
    o: int = 0  # for GJR
    dist: str = "normal"  # "t", "skewt", etc.
    mean: str = "zero"  # "constant" if you want a mean
    vol: str = "GARCH"  # or "EGARCH", "GARCH", "GJR-GARCH"
    scale_features: bool = True


class GARCHModel:
    def __init__(self, config: GARCHConfig):
        """Initialize the GARCH model with the given configuration.

        Args:
            config (GARCHConfig): Configuration parameters for the GARCH model.
        """
        self.config = config
        self._fitted: bool = False
        self.pipeline: Optional[Pipeline] = (
            None  # Pipeline for preprocessing and modeling (c'est pour éviter d'oublier de transformer les données)
        )

    def fit(self, df: pd.DataFrame) -> "GARCHModel":
        """
        Fit the GARCH model to the data.
        Args:
            df (pd.DataFrame): DataFrame containing the log returns.
            Returns:GARCHModel: Fitted GARCHModel instance.
        """

        r = df[self.config.log_return_stock].astype(float).dropna()
        steps = []
        if self.config.scale_features:
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "garch",
                arch_model(
                    r,
                    p=self.config.p,
                    o=self.config.o,
                    q=self.config.q,
                    vol=self.config.vol,
                    dist=self.config.dist,
                    mean=self.config.mean,
                ),
            )
        )
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(disp="off")
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict variance using the fitted GARCH model.

        Args:
            df (pd.DataFrame): DataFrame containing the log returns for prediction.

        Raises:
            ValueError: If the model has not been fitted yet.

        Returns:
            pd.Series: Predicted variance values aligned with the input DataFrame index.
        """
        if not self._fitted:
            raise ValueError("Fit the model first.")
        horizon = len(df)
        # one-step-ahead rolling variance forecast aligned to df index
        fcast = self._fitted_res.forecast(horizon=horizon, reindex=False)
        var = fcast.variance.iloc[-1].values  # last row are next-step forecasts for horizon steps
        out = pd.Series(var, index=df.index, name="predictions")
        # ensure non-negativity
        return out.clip(lower=0)

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self._fitted_res is None:
            raise ValueError("Fit the model first.")
        # Forecast aligned to df
        y_hat = self.predict(df)
        # Proxy "true" variance as squared return
        y_true = (
            (df[self.config.log_return_stock].astype(float) ** 2).reindex(y_hat.index).clip(lower=0)
        )
        return {
            "RMSE": rmse(y_true, y_hat),
            "MAE": mae(y_true, y_hat),
            "QLIKE": qlike_loss(y_true.values, y_hat.values),
        }
