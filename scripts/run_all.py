import pandas as pd
from src.volforecast.models.base import BaseConfig
from src.volforecast.models.elasticnet_regression_model import ElasticNetVolModel, ElasticNetConfig
from src.volforecast.models.garch_model import GARCHVolModel, GARCHConfig
from src.volforecast.models.xgboost_model import XGBoostConfig, XGBoostVolModel
from src.volforecast.features.builders import FeatureBuilder
from src.volforecast.evaluation.backtest import rolling_backtest
from src.volforecast.visualization.plot import plot_and_save_volatility_forecast


def main():
    # ----------Parameters--------------------------
    # Data choose
    csv_path = "data/dataset_AAPL.csv"
    burning_period = 150
    type_step = "1B"
    min_history = 20
    fixed_window = 750

    # Shared Config
    date_col = "date"
    return_col = "log_return_AAPL"
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
    n_estimators = 500
    learning_rate = 0.05
    max_depth = 4
    subsample = 0.8
    colsample_bytree = 0.8
    random_state = 42

    # ---------Loading data------------------------------
    df = pd.read_csv(csv_path, sep=";", parse_dates=["Date"])
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
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        use_log_target=True,
        random_state=random_state,
    )
    xgboost_model = XGBoostVolModel(xgb_cfg, feature_builder)

    # ---------Def of the models--------------------------------

    models = {"GARCH": garch_model, "Elasticnet": enet_model, "XGBoost": xgboost_model}

    # -----------Run backtests and display---------------------------------
    Results = {}
    for name, model in models.items():
        print(f"Running {name} backtest ...")
        model_results = rolling_backtest(
            model=model,
            df=df,
            train_start=train_start,
            train_end=train_end,
            step=type_step,
            min_history=min_history,
            fixed_window=fixed_window,
        )
        Results[model] = model_results["metrics"]
        print(f"{name} done ! QLIKE = {model_results['metrics']['QLIKE']:.3e}")
        plot_and_save_volatility_forecast(
            model_results,
            title=f"{model} — Next-Day Volatility Forecast",
            save=False,
            show=True,
            plot_as_vol=True,
        )

    # ---------- Metrics table (RMSE/MAE in scientific notation) ----------
    metrics_df = pd.DataFrame({k: v for k, v in Results.items()}).T
    metrics_df["RMSE"] = metrics_df["RMSE"].map(lambda x: f"{x:.2e}")
    metrics_df["MAE"] = metrics_df["MAE"].map(lambda x: f"{x:.2e}")
    metrics_df = metrics_df[["RMSE", "MAE", "QLIKE", "n"]]

    print(" Model Performance Comparison")
    print(metrics_df.to_string(index=True))


if __name__ == "__main__":
    main()
