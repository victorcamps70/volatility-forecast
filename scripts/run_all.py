import pandas as pd
import time
from pathlib import Path
from src.volforecast.models.base import BaseConfig
from src.volforecast.models.elasticnet_regression_model import ElasticNetVolModel, ElasticNetConfig
from src.volforecast.models.garch_model import GARCHVolModel, GARCHConfig
from src.volforecast.models.xgboost_model import XGBoostConfig, XGBoostVolModel
from src.volforecast.features.builders import FeatureBuilder
from src.volforecast.data.preprocessor import FeaturePreprocessor, PreprocessingConfig
from src.volforecast.evaluation.backtest import rolling_backtest
from src.volforecast.visualization.plot import plot_and_save_volatility_forecast


TICKERS = [
    "AAPL",
    "AXP",
    "AMZN",
    "AVGO",
    "CAT",
    "CSCO",
    "KO",
    "GOOGL",
    "GS",
    "JPM",
    "MA",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "QCOM",
    "TSLA",
    "UBER",
]  # 15 biggest american capitalisations

METRICS_CSV = Path("reports") / "metrics_summary.csv"
LAST_YEAR_DAYS = 250


def append_metrics_to_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """
    Append metrics for one ticker to a global CSV.

    df must have columns: ticker, model, RMSE, MAE, QLIKE, n
    Args:
        df: dataframe (dataset)
        csv_path: Path to the data
    Returns:
        None
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def keep_last_year_in_results(model_results: dict, n_days: int = LAST_YEAR_DAYS) -> dict:
    """
    Return a copy of model_results where y_true and y_pred
    are restricted to the last n_days (based on y_true index).
    Args:
        model_results: dict with the metrics, the target predicted and the real from target
        n_days: window of data we want to show
    Returns:
        copy of the model_results restricted to the last n_days
    """
    # Copy to avoid modifying the original results dict in place
    subset = model_results.copy()

    y_true = model_results["y_true"]
    y_pred = model_results["y_pred"]

    # Take the last n_days of y_true
    y_true_last = y_true.iloc[-n_days:]

    # Align y_pred on the same dates
    y_pred_last = y_pred.loc[y_true_last.index]

    subset["y_true"] = y_true_last
    subset["y_pred"] = y_pred_last

    return subset


def run_for_ticker(ticker: str) -> pd.DataFrame:
    """
    Function to run the training, the test and the plot of a ticker.
    Args:
        ticker: name of the ticker
    Returns:
        Metrics of the 4 models on the ticker
    """
    # ----------Parameters--------------------------
    # Data choose
    csv_path = f"data/{ticker}_dataset.csv"
    burning_period = 150
    type_step = "1B"
    min_history_GARCH = 100
    fixed_window_GARCH = 500  # GARCH needs a lot of data
    min_history_ENET_1 = 50
    fixed_window_ENET_1 = (
        50  # LR needs not much points to work well -> if too many points, just a mean
    )
    min_history_ENET_2 = 100
    fixed_window_ENET_2 = 200  # LR to have the best measures (but plot not there)
    min_history_XGBOOST = 50
    fixed_window_XGBOOST = 100

    # Shared Config
    date_col = "date"
    return_col = "log_return"
    target_shift = -5
    eps = 1e-8

    # Feature builder
    lags_returns = (1, 2, 3, 4, 5)
    lags_vix = (1, 2, 3, 4, 5)
    vix_col = "log_return_VIX"

    # GARCH Setup
    p = 1
    q = 1
    dist = "normal"
    mean = "zero"

    # Elasticnet Setup
    alphas = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    l1_ratio = (0.1, 0.5, 0.9)
    cv_splits = 5

    # XGBoost Setup
    n_iter = 30

    # ---------Loading data------------------------------
    print(f"\n===== Ticker: {ticker} | Loading data from {csv_path} =====")
    df = pd.read_csv(csv_path, sep=",", parse_dates=["Date"])
    df = df.rename(columns={"Date": "date"}).sort_values("date").set_index("date")
    train_start = df.index.min()
    train_end = df.index[burning_period]

    # ---------Preprocessing (winsorization + EWMA)------------------------------
    print(f"Applying data preprocessing (winsorization + EWMA) ...")
    preprocessing_cfg = PreprocessingConfig(
        apply_winsorization=True,
        winsorize_lower=0.01,
        winsorize_upper=0.99,
        apply_ewma=True,
        ewma_span=20,
    )
    preprocessor = FeaturePreprocessor(preprocessing_cfg)
    
    # Split data by train_end to apply preprocessing parameters
    train_data = df.loc[:train_end].copy()
    test_data = df.loc[train_end:].copy()
    
    # Fit preprocessing on train data and apply to both
    train_data = preprocessor.preprocess(train_data, ticker=ticker, fit_params=True)
    test_data = preprocessor.preprocess(test_data, ticker=ticker, fit_params=False)
    
    # Recombine datasets
    df = pd.concat([train_data, test_data])
    print(f"Preprocessing complete. Preprocessed columns: {preprocessor.get_preprocessing_info()}")

    # --------Creating models----------------------------
    # Shared config
    base_cfg = BaseConfig(
        date_col=date_col,
        return_col=return_col,
        target_shift=target_shift,
        eps=eps,
    )

    # Feature Builder
    feature_builder = FeatureBuilder(
        lags_returns=lags_returns,
        lags_vix=lags_vix,
        add_dow=True,
        date_col=date_col,
        return_col=return_col,
        vix_col=vix_col,
    )

    # GARCH setup
    garch_cfg = GARCHConfig(**base_cfg.__dict__, p=p, q=q, dist=dist, mean=mean)
    garch_model = GARCHVolModel(garch_cfg)

    # Elasticnet setup
    enet_cfg = ElasticNetConfig(
        **base_cfg.__dict__,
        alphas=alphas,
        l1_ratio=l1_ratio,
        cv_splits=cv_splits,
        use_log_target=True,  # trains on log(target); auto-inverted at predict time
        scale_features=True,
    )
    enet_model = ElasticNetVolModel(enet_cfg, feature_builder)

    # XGBoost setup

    xgb_cfg = XGBoostConfig(
        **base_cfg.__dict__,
        cv_splits=cv_splits,
        n_iter=n_iter,
        use_log_target=True,
        random_state=42,
    )
    xgboost_model = XGBoostVolModel(xgb_cfg, feature_builder)

    # ---------Def of the models--------------------------------

    models = {
        "GARCH": garch_model,
        "Elasticnet1": enet_model,
        "Elasticnet2": enet_model,
        "XGBoost": xgboost_model,
    }

    # -----------Run backtests and display---------------------------------
    Results = {}
    for name, model in models.items():
        print(f"Running {name} backtest for {ticker} ...")

        start_time = time.perf_counter()

        if name == "GARCH":
            model_results = rolling_backtest(
                model=model,
                df=df,
                train_start=train_start,
                train_end=train_end,
                step=type_step,
                min_history=min_history_GARCH,
                fixed_window=fixed_window_GARCH,
            )
        elif name == "Elasticnet1":
            model_results = rolling_backtest(
                model=model,
                df=df,
                train_start=train_start,
                train_end=train_end,
                step=type_step,
                min_history=min_history_ENET_1,
                fixed_window=fixed_window_ENET_1,
            )
        elif name == "Elasticnet2":
            model_results = rolling_backtest(
                model=model,
                df=df,
                train_start=train_start,
                train_end=train_end,
                step=type_step,
                min_history=min_history_ENET_2,
                fixed_window=fixed_window_ENET_2,
            )
        else:
            model_results = rolling_backtest(
                model=model,
                df=df,
                train_start=train_start,
                train_end=train_end,
                step=type_step,
                min_history=min_history_XGBOOST,
                fixed_window=fixed_window_XGBOOST,
            )

        elapsed_sec = time.perf_counter() - start_time
        metrics = model_results["metrics"].copy()
        metrics["runtime_sec"] = elapsed_sec

        Results[name] = metrics
        print(
            f"{name} done ! QLIKE = {model_results['metrics']['QLIKE']:.3e} | "
            f"time = {elapsed_sec:.1f} s ({elapsed_sec/60:.1f} min)"
        )

        last_year_results = keep_last_year_in_results(model_results)
        plot_and_save_volatility_forecast(
            last_year_results,
            title=f"{ticker}_{name}_Next-Day_Volatility_Forecast",
            save=True,
            show=False,
            plot_as_vol=True,
        )

    # ---------- Metrics table (RMSE/MAE in scientific notation) ----------
    metrics_df = pd.DataFrame(Results).T  # index = model name
    metrics_df = metrics_df[["RMSE", "MAE", "QLIKE", "n", "runtime_sec"]]

    metrics_df["ticker"] = ticker
    metrics_df["model"] = metrics_df.index

    display_df = metrics_df.copy()
    display_df["RMSE"] = metrics_df["RMSE"].map(lambda x: f"{x:.2e}")
    display_df["MAE"] = metrics_df["MAE"].map(lambda x: f"{x:.2e}")
    display_df["runtime_min"] = display_df["runtime_sec"] / 60

    print(f"\nModel Performance Comparison for {ticker}")
    print(display_df.set_index("model")[["RMSE", "MAE", "QLIKE", "n", "runtime_min"]].to_string())

    return metrics_df.reset_index(drop=True)


def main():
    all_metrics_list = []

    for ticker in TICKERS:
        try:
            metrics_df = run_for_ticker(ticker)
        except Exception as e:
            print(f"\n!!! Error while processing {ticker}: {e}")
            # you can also import traceback and print traceback.format_exc()
            continue

        # Save to global CSV immediately
        append_metrics_to_csv(metrics_df, METRICS_CSV)
        all_metrics_list.append(metrics_df)

    if not all_metrics_list:
        print("No metrics produced (all tickers failed?).")
        return

    all_metrics_df = pd.concat(all_metrics_list, ignore_index=True)
    all_metrics_df = all_metrics_df.set_index(["ticker", "model"])

    print("\n==================== Global Summary ====================")
    print(all_metrics_df.to_string())


if __name__ == "__main__":
    main()
