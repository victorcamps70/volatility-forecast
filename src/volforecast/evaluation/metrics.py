from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> float:
    """
    Calculate the QLIKE loss between true and predicted values.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: The QLIKE loss value.
    """

    y_true = np.asarray(y_true, dtype=np.float64)
    y_true = np.clip(y_true, epsilon, None)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, epsilon, None)
    loss = float(np.mean(np.log(y_pred) + (y_true / y_pred)))

    return loss


def rmse(y_true, y_pred) -> float:
    """Mean squared error

    Args:
        y_true (float): real values
        y_pred (float): predicted values

    Returns:
        float: returns root mean squared error (sklearn metrics)
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error

    Args:
        y_true (float): real values
        y_pred (float): predicted values

    Returns:
        float: returns mean absolute error (sklearn metrics)
    """
    return float(mean_absolute_error(y_true, y_pred))
