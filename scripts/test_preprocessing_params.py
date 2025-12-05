"""
Script to test different preprocessing parameter combinations for a single ticker.

This script runs models with various preprocessing configurations and compares their performance.
Useful for hyperparameter tuning of winsorization and EWMA parameters.

Usage:
    python scripts/test_preprocessing_params.py --ticker AAPL --output-dir reports/preprocessing_tests
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.volforecast.models.base import BaseConfig
from src.volforecast.models.elasticnet_regression_model import ElasticNetVolModel, ElasticNetConfig
from src.volforecast.models.garch_model import GARCHVolModel, GARCHConfig
from src.volforecast.models.xgboost_model import XGBoostConfig, XGBoostVolModel
from src.volforecast.features.builders import FeatureBuilder
from src.volforecast.data.preprocessor import FeaturePreprocessor, PreprocessingConfig
from src.volforecast.evaluation.backtest import rolling_backtest


# Define parameter combinations to test
PREPROCESSING_CONFIGS = [
    {
        "name": "no_preprocessing",
        "apply_winsorization": False,
        "apply_ewma": False,
    },
    {
        "name": "winsorization_only_conservative",
        "apply_winsorization": True,
        "winsorize_lower": 0.01,
        "winsorize_upper": 0.99,
        "apply_ewma": False,
    },
    {
        "name": "winsorization_only_moderate",
        "apply_winsorization": True,
        "winsorize_lower": 0.05,
        "winsorize_upper": 0.95,
        "apply_ewma": False,
    },
    {
        "name": "winsorization_only_aggressive",
        "apply_winsorization": True,
        "winsorize_lower": 0.10,
        "winsorize_upper": 0.90,
        "apply_ewma": False,
    },
    {
        "name": "ewma_only_light",
        "apply_winsorization": False,
        "apply_ewma": True,
        "ewma_span": 50,
    },
    {
        "name": "ewma_only_moderate",
        "apply_winsorization": False,
        "apply_ewma": True,
        "ewma_span": 20,
    },
    {
        "name": "ewma_only_heavy",
        "apply_winsorization": False,
        "apply_ewma": True,
        "ewma_span": 10,
    },
    {
        "name": "combined_conservative",
        "apply_winsorization": True,
        "winsorize_lower": 0.01,
        "winsorize_upper": 0.99,
        "apply_ewma": True,
        "ewma_span": 20,
    },
    {
        "name": "combined_moderate",
        "apply_winsorization": True,
        "winsorize_lower": 0.05,
        "winsorize_upper": 0.95,
        "apply_ewma": True,
        "ewma_span": 20,
    },
    {
        "name": "combined_aggressive",
        "apply_winsorization": True,
        "winsorize_lower": 0.10,
        "winsorize_upper": 0.90,
        "apply_ewma": True,
        "ewma_span": 30,
    },
]


def create_preprocessing_config(config_dict):
    """Create PreprocessingConfig from dictionary."""
    return PreprocessingConfig(
        apply_winsorization=config_dict.get("apply_winsorization", False),
        winsorize_lower=config_dict.get("winsorize_lower", 0.01),
        winsorize_upper=config_dict.get("winsorize_upper", 0.99),
        apply_ewma=config_dict.get("apply_ewma", False),
        ewma_span=config_dict.get("ewma_span", 20),
    )


def run_model_with_preprocessing(
    ticker,
    csv_path,
    burning_period,
    preprocessing_cfg,
    model_name,
):
    """Run a single model with given preprocessing configuration."""

    # Load data
    df = pd.read_csv(csv_path, sep=",", parse_dates=["Date"])
    df = df.rename(columns={"Date": "date"}).sort_values("date").set_index("date")
    train_start = df.index.min()
    train_end = df.index[burning_period]

    # Apply preprocessing
    preprocessor = FeaturePreprocessor(preprocessing_cfg)
    train_end_idx = burning_period
    train_data = df.iloc[:train_end_idx].copy()
    test_data = df.iloc[train_end_idx:].copy()

    train_data = preprocessor.preprocess(train_data, ticker=ticker, fit_params=True)
    test_data = preprocessor.preprocess(test_data, ticker=ticker, fit_params=False)

    df = pd.concat([train_data, test_data])
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]

    # Model config
    date_col = "date"
    return_col = "log_return"
    target_shift = -5
    eps = 1e-8

    base_cfg = BaseConfig(
        date_col=date_col,
        return_col=return_col,
        target_shift=target_shift,
        eps=eps,
    )

    # Model-specific parameters
    if model_name == "GARCH":
        cfg = GARCHConfig(**base_cfg.__dict__, p=1, q=1, dist="normal", mean="zero")
        model = GARCHVolModel(cfg)
        min_history = 100
        fixed_window = 500

    elif model_name == "ElasticNet":
        lags_returns = (1, 2, 3, 4, 5)
        lags_vix = (1, 2, 3, 4, 5)
        feature_builder = FeatureBuilder(
            lags_returns=lags_returns,
            lags_vix=lags_vix,
            add_dow=True,
            date_col=date_col,
            return_col=return_col,
            vix_col="log_return_VIX",
        )
        cfg = ElasticNetConfig(
            **base_cfg.__dict__,
            alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
            l1_ratio=(0.1, 0.5, 0.9),
            cv_splits=5,
            use_log_target=True,
            scale_features=True,
        )
        model = ElasticNetVolModel(cfg, feature_builder)
        min_history = 100
        fixed_window = 200

    elif model_name == "XGBoost":
        lags_returns = (1, 2, 3, 4, 5)
        lags_vix = (1, 2, 3, 4, 5)
        feature_builder = FeatureBuilder(
            lags_returns=lags_returns,
            lags_vix=lags_vix,
            add_dow=True,
            date_col=date_col,
            return_col=return_col,
            vix_col="log_return_VIX",
        )
        cfg = XGBoostConfig(
            **base_cfg.__dict__,
            cv_splits=5,
            n_iter=30,
            use_log_target=True,
            random_state=42,
        )
        model = XGBoostVolModel(cfg, feature_builder)
        min_history = 50
        fixed_window = 100

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Run backtest
    try:
        model_results = rolling_backtest(
            model=model,
            df=df,
            train_start=train_start,
            train_end=train_end,
            step="1B",
            min_history=min_history,
            fixed_window=fixed_window,
        )
        return model_results["metrics"]
    except Exception as e:
        print(f"Error running {model_name} with {preprocessing_cfg}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Test different preprocessing parameters for a single ticker."
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol (default: AAPL)")
    parser.add_argument(
        "--output-dir",
        default="reports/preprocessing_tests",
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        default="GARCH,ElasticNet,XGBoost",
        help="Comma-separated list of models to test",
    )
    parser.add_argument(
        "--burning-period",
        type=int,
        default=150,
        help="Burning period for train/test split",
    )

    args = parser.parse_args()

    ticker = args.ticker
    output_dir = Path(args.output_dir)
    models = args.models.split(",")
    burning_period = args.burning_period

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = f"data/{ticker}_dataset.csv"

    print(f"\n{'='*80}")
    print(f"Testing preprocessing parameters for {ticker}")
    print(f"{'='*80}\n")

    # Run tests
    all_results = []

    for config_dict in PREPROCESSING_CONFIGS:
        config_name = config_dict["name"]
        print(f"\nTesting: {config_name}")
        print("-" * 80)

        preprocessing_cfg = create_preprocessing_config(config_dict)

        for model_name in models:
            print(f"  {model_name}...", end=" ", flush=True)

            metrics = run_model_with_preprocessing(
                ticker=ticker,
                csv_path=csv_path,
                burning_period=burning_period,
                preprocessing_cfg=preprocessing_cfg,
                model_name=model_name,
            )

            if metrics is not None:
                result = {
                    "preprocessing_config": config_name,
                    "model": model_name,
                    "ticker": ticker,
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "QLIKE": metrics["QLIKE"],
                    "n": metrics["n"],
                    "timestamp": datetime.now().isoformat(),
                }

                # Add preprocessing details
                result.update(
                    {
                        "apply_winsorization": config_dict.get("apply_winsorization", False),
                        "winsorize_lower": config_dict.get("winsorize_lower", None),
                        "winsorize_upper": config_dict.get("winsorize_upper", None),
                        "apply_ewma": config_dict.get("apply_ewma", False),
                        "ewma_span": config_dict.get("ewma_span", None),
                    }
                )

                all_results.append(result)
                print(f"QLIKE={metrics['QLIKE']:.3e}")
            else:
                print("FAILED")

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save raw results
    results_csv = output_dir / f"{ticker}_preprocessing_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n✓ Results saved to {results_csv}")

    # Create summary tables
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")

    # Pivot for RMSE
    rmse_pivot = results_df.pivot_table(
        values="RMSE", index="preprocessing_config", columns="model", aggfunc="first"
    )
    print("RMSE by Preprocessing Config and Model:")
    print(rmse_pivot.to_string())

    # Pivot for MAE
    mae_pivot = results_df.pivot_table(
        values="MAE", index="preprocessing_config", columns="model", aggfunc="first"
    )
    print("\n\nMAE by Preprocessing Config and Model:")
    print(mae_pivot.to_string())

    # Pivot for QLIKE
    qlike_pivot = results_df.pivot_table(
        values="QLIKE", index="preprocessing_config", columns="model", aggfunc="first"
    )
    print("\n\nQLIKE by Preprocessing Config and Model:")
    print(qlike_pivot.to_string())

    # Find best config per model
    print(f"\n{'='*80}")
    print("BEST CONFIGURATIONS (by QLIKE)")
    print(f"{'='*80}\n")

    for model_name in models:
        model_df = results_df[results_df["model"] == model_name]
        best_idx = model_df["QLIKE"].idxmin()
        best_row = results_df.loc[best_idx]

        print(f"{model_name}:")
        print(f"  Config: {best_row['preprocessing_config']}")
        print(f"  QLIKE: {best_row['QLIKE']:.3e}")
        print(f"  RMSE: {best_row['RMSE']:.3e}")
        print(f"  MAE: {best_row['MAE']:.3e}")

        # Print config details
        if best_row["apply_winsorization"]:
            print(
                f"  Winsorization: [{best_row['winsorize_lower']}, {best_row['winsorize_upper']}]"
            )
        if best_row["apply_ewma"]:
            print(f"  EWMA span: {best_row['ewma_span']}")
        print()

    # Save summary tables
    rmse_pivot.to_csv(output_dir / f"{ticker}_rmse_summary.csv")
    mae_pivot.to_csv(output_dir / f"{ticker}_mae_summary.csv")
    qlike_pivot.to_csv(output_dir / f"{ticker}_qlike_summary.csv")

    print(f"\n✓ Summary tables saved to {output_dir}/")

    # Save recommended config
    recommended = {"ticker": ticker, "test_date": datetime.now().isoformat(), "best_configs": {}}

    for model_name in models:
        model_df = results_df[results_df["model"] == model_name]
        best_idx = model_df["QLIKE"].idxmin()
        best_row = results_df.loc[best_idx]

        recommended["best_configs"][model_name] = {
            "preprocessing_config": best_row["preprocessing_config"],
            "apply_winsorization": bool(best_row["apply_winsorization"]),
            "winsorize_lower": (
                float(best_row["winsorize_lower"])
                if best_row["winsorize_lower"] is not None
                else None
            ),
            "winsorize_upper": (
                float(best_row["winsorize_upper"])
                if best_row["winsorize_upper"] is not None
                else None
            ),
            "apply_ewma": bool(best_row["apply_ewma"]),
            "ewma_span": int(best_row["ewma_span"]) if best_row["ewma_span"] is not None else None,
            "metrics": {
                "QLIKE": float(best_row["QLIKE"]),
                "RMSE": float(best_row["RMSE"]),
                "MAE": float(best_row["MAE"]),
            },
        }

    with open(output_dir / f"{ticker}_recommended_config.json", "w") as f:
        json.dump(recommended, f, indent=2)

    print(f"✓ Recommended config saved to {output_dir}/{ticker}_recommended_config.json")


if __name__ == "__main__":
    main()
