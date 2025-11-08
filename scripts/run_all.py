import pandas as pd
from volforecast.models.base import BaseConfig
from volforecast.models.elasticnet_regression_model import ElasticNetVolModel, ElasticNetConfig
from volforecast.models.garch_model import GARCHVolModel, GARCHConfig
from volforecast.features.builders import ElasticNetFeatureBuilder
from volforecast.evaluation.backtest import rolling_backtest

# 1. Load data
csv_path = "data/dataset_AAPL.csv"
df = pd.read_csv(csv_path, sep=";", parse_dates=["Date"])

# 2. Shared config
base_cfg = BaseConfig(date_col="date", return_col="log_return", target_shift=-1, eps=1e-8)

# 3. ElasticNet setup
enet_cfg = ElasticNetConfig(
    **base_cfg.__dict__, alphas=(1e-4, 1e-3, 1e-2, 1e-1), l1_ratio=(0.1, 0.5, 0.9)
)
features = ElasticNetFeatureBuilder(lags_returns=(1, 2, 5), lags_vix=(1, 2))
enet = ElasticNetVolModel(enet_cfg, features)

# 4. GARCH setup
garch_cfg = GARCHConfig(**base_cfg.__dict__, p=1, q=1, dist="normal", mean="zero")
garch = GARCHVolModel(garch_cfg)

# 5. Run backtests
bt_enet = rolling_backtest(enet, df, train_start=df.index.min(), train_end=df.index[250])
bt_garch = rolling_backtest(garch, df, train_start=df.index.min(), train_end=df.index[250])

# 6. Display metrics
print("ElasticNet metrics:", bt_enet["metrics"])
print("GARCH metrics:", bt_garch["metrics"])

# 7. (optional) Save or plot results
# bt_enet["y_pred"].to_csv("outputs/enet_predictions.csv")
# bt_garch["y_pred"].to_csv("outputs/garch_predictions.csv")
