from __future__ import annotations
import pandas as pd
from typing import Tuple
from dataclasses import dataclass


class FeatureBuilder:
    def __init__(
        self,
        lags_returns=(1, 2, 5),
        lags_vix=(1, 2),
        add_dow=True,
        date_col="date",
        return_col="log_return",
        vix_col="vix",
    ):
        self.lags_returns = lags_returns
        self.lags_vix = lags_vix
        self.add_dow = add_dow
        self.date_col = date_col
        self.return_col = return_col
        self.vix_col = vix_col

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        # example features
        for L in self.lags_returns:
            X[f"lag_returns_{L}"] = df[self.return_col].shift(L)
        for L in self.lags_vix:
            X[f"lag_vix_{L}"] = df[self.vix_col].shift(L)
        if self.add_dow and "date" in df.columns:
            dow = pd.to_datetime(df[self.date_col]).dt.dayofweek
            X = X.join(pd.get_dummies(dow, prefix="dow", dtype=int))
        return X
