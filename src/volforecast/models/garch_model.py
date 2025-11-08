from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from arch import arch_model
from src.volforecast.models.base import BaseVolModel, BaseConfig


@dataclass
class GARCHConfig(BaseConfig):
    p: int = 1
    q: int = 1
    o: int = 0
    dist: str = "normal"
    mean: str = "zero"
    scale_factor: float = 1.0  # to rescale returns if needed
    rescale_returns: bool = False  # whether to rescale returns by scale_factor


class GARCHVolModel(BaseVolModel):
    def __init__(self, config: GARCHConfig):
        super().__init__(config)
        self.res_ = None
        self.fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "GARCHVolModel":
        r = df[self.config.return_col].dropna()
        am = arch_model(
            r,
            vol="GARCH",
            p=self.config.p,
            q=self.config.q,
            o=self.config.o,
            dist=self.config.dist,
            mean=self.config.mean,
        )
        self.res_ = am.fit(disp="off")
        self.fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        assert self.fitted_, "Call fit() first."
        h = self.res_.forecast(horizon=1, start=0, reindex=True).variance.iloc[:, 0]

        if self.config.rescale_returns:
            h = h * (self.config.scale_factor**2)

        # align to incoming df index, forward-fill if needed
        yhat = h.reindex(df.index).rename("y_pred")
        return yhat


if __name__ == "__main__":
    import pandas as pd

    print("Running basic self-test for GARCHVolModel...")

    # 1. Create a small dummy dataset
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "log_return": [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, 0.0, 0.01, -0.005, 0.002],
        }
    ).set_index("date")

    # 2. Build config and model

    base_cfg = BaseConfig(return_col="log_return", target_shift=-1)
    garch_cfg = GARCHConfig(**base_cfg.__dict__, p=1, q=1, dist="normal", mean="zero")
    model = GARCHVolModel(garch_cfg)

    # 3. Fit & predict
    model.fit(df)
    y_pred = model.predict(df)

    print("Predictions:")
    print(y_pred.head())

    # 4. (Optional) quick comparison with realized variance
    y_true = model.build_target(df)
    valid = y_true.notna() & y_pred.notna()
    if valid.any():
        mae = (y_true[valid] - y_pred[valid]).abs().mean()
        print(f"\nMean Absolute Error (sanity check): {mae:.6e}")

    print("\n✅ GARCHVolModel self-test finished.")
