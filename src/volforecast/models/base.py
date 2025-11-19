from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from typing import Dict, Any, Generic, TypeVar
from src.volforecast.evaluation.metrics import rmse as rmse_loss, mae as mae_loss, qlike_loss


@dataclass
class BaseConfig:
    date_col: str = "date"
    return_col: str = "log_return"  # base input for targets
    target_shift: int = -1  # 1-step ahead (align realized at t with info at t-1)
    eps: float = 1e-8  # for log transforms


C = TypeVar("C", bound=BaseConfig)


class BaseVolModel(Generic[C]):
    def __init__(self, config: C):
        self.config = config

    # REQUIRED — uniform across all models
    def fit(self, df: pd.DataFrame) -> "BaseVolModel":
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series indexed like df.index, named 'y_pred' (variance/vol proxy)."""
        raise NotImplementedError

    def build_target(self, df: pd.DataFrame) -> pd.Series:
        """Return the realized variance/vol proxy aligned with df.index (e.g. r_{t}^2 shifted)."""
        r = df[self.config.return_col]
        y = (r**2).shift(-self.config.target_shift)  # realized at t compared to info at t-1
        return y.rename("y_true")

    def evaluate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generic evaluation: compute metrics on predict(df) vs build_target(df)."""

        y_pred = self.predict(df).rename("y_pred")
        y_true = self.build_target(df).reindex(y_pred.index)
        valid = y_true.notna() & y_pred.notna()
        return {
            "RMSE": rmse_loss(y_true[valid], y_pred[valid]),
            "MAE": mae_loss(y_true[valid], y_pred[valid]),
            "QLIKE": qlike_loss(y_true[valid], y_pred[valid]),
            "n": int(valid.sum()),
        }

    # OPTIONAL — models can override to add details (e.g., coefficients)
    def summary(self) -> Dict[str, Any]:
        return {}
