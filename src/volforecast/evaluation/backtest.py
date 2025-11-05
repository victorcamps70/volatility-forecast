from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import qlike_loss, rmse, mae


def rolling_backtest(
    model, df: pd.DataFrame, n_test: int = 252, target_col: str = "target"
) -> dict:
    """Generic rolling-origin evaluation using a model with fit/predict/evaluate signatures."""
    X, y = model._build_X_y(df)  # reuse model's feature builder for consistency
    N = len(X)
    if n_test <= 0 or n_test >= N:
        raise ValueError("n_test must be in (0, N)")
    X_train, X_test = X.iloc[: N - n_test], X.iloc[N - n_test :]
    target = model._build_target(df).loc[X_test.index].clip(lower=0)
    model.fit(df.iloc[: N - n_test])
    y_hat = model.predict(df.iloc[N - n_test :]).reindex(target.index)
    return {
        "RMSE": rmse(target, np.maximum(y_hat, 0.0)),
        "MAE": mae(target, np.maximum(y_hat, 0.0)),
        "QLIKE": qlike_loss(target.values, np.maximum(y_hat, 0.0).values),
    }
