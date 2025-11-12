from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from src.volforecast.evaluation.metrics import qlike_loss, rmse as rmse_loss, mae as mae_loss


def rolling_backtest(model, df: pd.DataFrame, train_start, train_end, step="1D") -> Dict[str, Any]:
    """
    Expanding or sliding window:
    - fit on df.loc[:train_end]
    - predict on next 'step'
    - advance window; repeat
    """
    y_pred_list = []
    y_true_list = []

    current_end = pd.to_datetime(train_end)
    while True:
        train_df = df.loc[:current_end]
        if train_df.empty:
            break

        model.fit(train_df)

        # predict on the next period
        next_start = current_end + pd.tseries.frequencies.to_offset(step)
        test_df = df.loc[next_start:next_start]
        if test_df.empty:
            break

        pred = model.predict(test_df)
        true = model.build_target(test_df)
        y_pred_list.append(pred)
        y_true_list.append(true)

        current_end = next_start  # move forward

    y_pred = pd.concat(y_pred_list).rename("y_pred")
    y_true = pd.concat(y_true_list).rename("y_true")

    # metrics
    from .metrics import rmse, mae, qlike

    valid = y_true.notna() & y_pred.notna()
    metrics = {
        "RMSE": rmse_loss(y_true[valid], y_pred[valid]),
        "MAE": mae_loss(y_true[valid], y_pred[valid]),
        "QLIKE": qlike_loss(y_true[valid], y_pred[valid]),
        "n": int(valid.sum()),
    }
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}
