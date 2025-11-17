import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd


def plot_and_save_volatility_forecast(
    res: dict,
    title: str = "Volatility Forecast",
    show: bool = True,
    save: bool = True,
    save_name: str | None = None,
    plot_as_vol: bool = True,
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
        None
    """
    # --- Prepare data ---
    y_true = res["y_true"]
    y_pred = res["y_pred"]
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid].rename("Realized")
    y_pred = y_pred[valid].rename("Predicted")

    if plot_as_vol:
        y_true = np.sqrt(y_true.clip(lower=0))
        y_pred = np.sqrt(y_pred.clip(lower=0))
        y_label = "Volatility"
    else:
        y_label = "Variance (squared return)"

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(y_true.index, y_true.values, label="Realized")
    plt.plot(y_pred.index, y_pred.values, label="Predicted")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()

    # --- Save figure ---
    if save:
        reports_dir = Path("reports") / "figures"
        reports_dir.mkdir(parents=True, exist_ok=True)

        if save_name is None:
            save_name = f"{title.lower().replace(' ', '_')}_forecast_plot.png"

        save_path = reports_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path.resolve()}")

    # --- Show or close ---
    if show:
        plt.show()

    # plt.close()

    # print metrics for convenience
    print("\nMetrics:")
    for k, v in res.get("metrics", {}).items():
        try:
            print(f"{k}: {float(v):.6f}")
        except Exception:
            print(f"{k}: {v}")
