from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

@dataclass
class FeatureBuilder:
    lags_returns=(1, 2, 5)
    lags_vix=(1, 2)
    add_dow=True
    date_col="date"
    return_col="log_return_AAPL"
    vix_col="vix"

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
