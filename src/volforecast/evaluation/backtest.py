from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any
from src.volforecast.evaluation.metrics import qlike_loss, rmse as rmse_loss, mae as mae_loss


def rolling_backtest(
    model,
    df: pd.DataFrame,
    train_start,
    train_end,
    step: str = "1D",
    *,
    min_history: int | None = None,  # set e.g. 20 if you use lags up to 5 etc.
    fixed_window: int | None = None,  # e.g. last N rows; None = expanding window
) -> Dict[str, Any]:
    """
    Walk-forward validation:
      - fit on df.loc[:current_end] (or last N rows if fixed_window)
      - predict exactly at 'next_start' using *context* up to next_start
      - advance by 'step'
    """
    y_pred_list: list[pd.Series] = []
    y_true_list: list[pd.Series] = []

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("DataFrame must be indexed by DatetimeIndex for time-based backtest.")

    current_end = pd.to_datetime(train_end)
    step_offset = pd.tseries.frequencies.to_offset(step)

    # If user didn't pass min_history, try a safe default
    # (models with lags (1..5) usually need at least ~5-10 points; we pick 10)
    if min_history is None:
        min_history = 10

    while True:
        # Training window: expanding or fixed
        train_df = df.loc[:current_end]
        if fixed_window is not None and len(train_df) > fixed_window:
            train_df = train_df.iloc[-fixed_window:]

        if train_df.empty:
            break

        # Ensure we have enough history before fitting/predicting
        if len(train_df) < min_history:
            next_start = current_end + step_offset
            if next_start not in df.index:
                # If step lands on a gap, advance until we hit data or stop.
                while next_start not in df.index and next_start <= df.index.max():
                    next_start = next_start + step_offset
                if next_start not in df.index:
                    break
            current_end = next_start
            continue

        # Fit on training window
        model.fit(train_df)

        # Next timestamp to predict
        next_start = current_end + step_offset
        if next_start not in df.index:
            # walk forward to the next available timestamp
            while next_start not in df.index and next_start <= df.index.max():
                next_start = next_start + step_offset
            if next_start not in df.index:
                break

        # Context window for feature building up to prediction time
        context_df = df.loc[:next_start]

        # Model predicts for all rows it can; we keep the value at next_start
        pred_full = model.predict(context_df)
        # Be defensive if model returns a DataFrame or misaligned index
        if isinstance(pred_full, pd.DataFrame):
            pred_full = pred_full.squeeze()
        pred_full = pred_full.rename("y_pred")

        # Extract the single step we’re evaluating
        yhat = pred_full.loc[pred_full.index.intersection(pd.Index([next_start]))]

        # Ground truth at that time (model’s own target builder)
        ytrue_full = model.build_target(df)  # uses the whole series for the shift
        ytrue = ytrue_full.loc[next_start:next_start].rename("y_true")

        # Skip if either side missing/NaN
        if yhat.empty or ytrue.empty or yhat.isna().any() or ytrue.isna().any():
            current_end = next_start
            continue

        y_pred_list.append(yhat)
        y_true_list.append(ytrue)

        # Advance the window
        current_end = next_start

    # Concatenate results (handle case where nothing was appended)
    if not y_pred_list or not y_true_list:
        return {
            "metrics": {"RMSE": np.nan, "MAE": np.nan, "QLIKE": np.nan, "n": 0},
            "y_true": pd.Series(dtype=float, name="y_true"),
            "y_pred": pd.Series(dtype=float, name="y_pred"),
        }

    y_pred = pd.concat(y_pred_list).rename("y_pred")
    y_true = pd.concat(y_true_list).rename("y_true")

    # Compute metrics on overlapping non-NaN indices
    valid = y_true.notna() & y_pred.notna()
    if valid.sum() == 0:
        metrics = {"RMSE": np.nan, "MAE": np.nan, "QLIKE": np.nan, "n": 0}
    else:
        metrics = {
            "RMSE": rmse_loss(y_true[valid], y_pred[valid]),
            "MAE": mae_loss(y_true[valid], y_pred[valid]),
            "QLIKE": qlike_loss(y_true[valid], y_pred[valid]),
            "n": int(valid.sum()),
        }

    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}


if __name__ == "__main__":

    print("Running quick self-test for rolling_backtest...")

    # ===  Create small dummy dataset ===
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {"log_return_AAPL": np.random.normal(0, 0.02, size=len(idx))},
        index=idx,
    )

    # ===  Define a simple dummy model ===
    class DummyModel:
        def __init__(self):
            self.is_fitted = False

        def fit(self, train_df: pd.DataFrame):
            # Pretend to train (e.g., store mean)
            self.mean_ = train_df["log_return_AAPL"].mean()
            self.is_fitted = True

        def predict(self, test_df: pd.DataFrame):
            assert self.is_fitted, "Model not fitted!"
            # Predict constant variance estimate
            preds = pd.Series(self.mean_**2, index=test_df.index, name="y_pred")
            return preds

        def build_target(self, test_df: pd.DataFrame):
            return (test_df["log_return_AAPL"] ** 2).rename("y_true")

    # ===  Run backtest ===
    result = rolling_backtest(
        DummyModel(), df, train_start=df.index[0], train_end=df.index[9], step="1D"
    )

    # === 5) Print outputs ===
    print("\n=== Metrics ===")
    for k, v in result["metrics"].items():
        print(f"{k:10s}: {v:.6f}" if isinstance(v, float) else f"{k:10s}: {v}")

    print("\n=== Predictions ===")
    print(result["y_pred"].head())

    print("\n=== True values ===")
    print(result["y_true"].head())

    print("\n Self-test completed successfully!")
