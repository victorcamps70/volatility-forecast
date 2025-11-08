import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def plot_and_save_volatility_forecast(
    df: pd.DataFrame,
    model,
    log_return_col: str = "log_return_AAPL",
    title: str = "Volatility Forecast",
    show: bool = True,
    save: bool = True,
    save_name: str | None = None,
):
    """
    Plots and saves actual vs predicted next-day volatility (squared return).

    Args:
        df (pd.DataFrame): DataFrame containing the log return column.
        model: Fitted volatility model with a .predict(df) method.
        log_return_col (str): Column name for log returns.
        title (str): Plot title.
        show (bool): Whether to display the figure inline.
        save (bool): Whether to save the figure to reports/figures/.
        save_name (str | None): Custom filename (without extension).

    Returns:
        target_raw (pd.Series): Actual next-day variance.
        y_pred (pd.Series): Predicted variance.
    """
    # --- Prepare data ---
    y_pred = model.predict(df)
    target_raw = (df[log_return_col] ** 2).shift(-1)
    target_raw = target_raw.loc[y_pred.index].dropna()
    y_pred = y_pred.loc[target_raw.index]

    # --- Plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(target_raw.index, target_raw.values, label="Actual (next squared return)", alpha=0.7)
    plt.plot(y_pred.index, y_pred.values, label="Predicted variance", alpha=0.9)
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Variance (squared return)")
    plt.tight_layout()

    # --- Save figure ---
    if save:
        reports_dir = Path("reports") / "figures"
        reports_dir.mkdir(parents=True, exist_ok=True)

        if save_name is None:
            model_name = model.__class__.__name__
            save_name = f"{model_name}_forecast_plot.png"

        save_path = reports_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path.resolve()}")

    # --- Show or close ---
    if show:
        plt.show()
    else:
        plt.close()

    return target_raw, y_pred
