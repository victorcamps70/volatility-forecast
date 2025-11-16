import pandas as pd
from src.volforecast.models.base import BaseConfig
from src.volforecast.models.elasticnet_regression_model import ElasticNetVolModel, ElasticNetConfig
from src.volforecast.models.garch_model import GARCHVolModel, GARCHConfig
from src.volforecast.models.xgboost_model import XGBoostConfig, XGBoostVolModel
from src.volforecast.features.builders import FeatureBuilder
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


def run_for_ticker(ticker: str) -> pd.DataFrame:
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
    fixed_window_ENET_2 = 200  # LR to have the best measures (but curve not there)
    min_history_XGBOOST = 50
    fixed_window_XGBOOST = 100

    # Shared Config
    date_col = "date"
    return_col = "log_return"
    target_shift = -1
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
        print(f"Running {name} backtest ...")
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

        Results[model] = model_results["metrics"]
        print(f"{name} done ! QLIKE = {model_results['metrics']['QLIKE']:.3e}")
        plot_and_save_volatility_forecast(
            model_results,
            title=f"{ticker}_{name}_Next-Day_Volatility_Forecast",
            save=True,
            show=False,
            plot_as_vol=True,
        )

    # ---------- Metrics table (RMSE/MAE in scientific notation) ----------
    # ---------- Metrics table (RMSE/MAE in scientific notation) ----------
    metrics_df = pd.DataFrame(Results).T  # index = model name
    metrics_df["RMSE"] = metrics_df["RMSE"].map(lambda x: f"{x:.2e}")
    metrics_df["MAE"] = metrics_df["MAE"].map(lambda x: f"{x:.2e}")
    metrics_df = metrics_df[["RMSE", "MAE", "QLIKE", "n"]]

    print(f"\nModel Performance Comparison for {ticker}")
    print(metrics_df.to_string(index=True))

    return metrics_df


def main():
    all_metrics = []

    for ticker in TICKERS:
        metrics_df = run_for_ticker(ticker)
        # keep track of ticker + model
        metrics_df = metrics_df.copy()
        metrics_df["ticker"] = ticker
        metrics_df["model"] = metrics_df.index
        all_metrics.append(metrics_df.reset_index(drop=True))

    if not all_metrics:
        print("No tickers specified in TICKERS list.")
        return

    # Concatenate all results
    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    all_metrics_df = all_metrics_df.set_index(["ticker", "model"])

    print("\n==================== Global Summary ====================")
    print(all_metrics_df.to_string())

    # ===== Save results to CSV =====
    output_path = "reports/metrics_summary.csv"
    all_metrics_df.to_csv(output_path)
    print(f"\nSaved metrics summary to {output_path}")


if __name__ == "__main__":
    main()
