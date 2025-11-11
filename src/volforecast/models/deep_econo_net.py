from dataclasses import dataclass
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.volforecast.models.base import BaseVolModel, BaseConfig
from src.volforecast.features.builders import FeatureBuilder
from src.volforecast.models.garch_model import GARCHConfig, GARCHVolModel

@dataclass
class DeepEconoNetConfig(BaseConfig):
    input_size: int = 10
    hidden_size: int = 50
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
        super(DeepEconoNetModel, self).__init__(config)
        super(nn.Module, self).__init__()
        self.config = config
        self.kalman_garch = KalmanGARCH()
        self.conv1d = nn.Conv1d()
        self.lstm = nn.LSTM()
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        
class KalmanGARCH:
    pass