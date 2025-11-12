from dataclasses import dataclass
import pandas as pd
import torch.nn as nn

from src.volforecast.models.base import BaseVolModel
from src.volforecast.models.garch_model import GARCHConfig, GARCHVolModel

@dataclass
class DeepEconoNetConfig(GARCHConfig):
    input_size: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 1
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    use_log_target: bool = True
    scale_features: bool = True

class DeepEconoNetModel(BaseVolModel[DeepEconoNetConfig], nn.Module):
    def __init__(self, config: DeepEconoNetConfig):
        super().__init__(config)
        self.config = config
        self.kalman_garch = KalmanGARCH(config)
        self.conv1d = nn.Conv1d(in_channels=config.input_size, out_channels=config.hidden_size, kernel_size=5, padding=1)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True)
        self.fc1 = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.fc2 = nn.Linear(in_features=config.hidden_size, out_features=config.output_size)

class KalmanGARCH:
    def __init__(self, config: DeepEconoNetConfig):
        self.garch_model = GARCHVolModel(config)

    def fit(self, returns: pd.DataFrame):
        self.garch_model.fit(returns)
        return self