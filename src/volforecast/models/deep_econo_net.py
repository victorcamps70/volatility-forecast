from dataclasses import dataclass
import pandas as pd
import torch.nn as nn
import numpy as np
from typing import List

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
    
    def forward(self, x):
        x = self.kalman_garch.garch_model.predict(x)
        x = self.conv1d(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last time step
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

class KalmanGARCH:
    def __init__(self, config: DeepEconoNetConfig):
        self.garch_model = GARCHVolModel(config)

    def fit(self, returns: pd.DataFrame):
        self.garch_model.fit(returns)
        return self

    # --- 2. Predict / denoise function ---
    def predict(self, log_returns:List[float], Q:float=1e-4, R:float=1e-2):
        """
        Denoise log-returns using GARCH variance as hidden state (scalar EKF).
        
        Inputs:
            log_returns : list or array of log-returns
            params : tuple (omega, alpha, beta)
            Q : process noise
            R : measurement noise
        
        Output:
            denoised_returns : array of same length
        """
        omega, alpha, beta = self.garch_model.res_.params['omega'], self.garch_model.res_.params['alpha[1]'], self.garch_model.res_.params['beta[1]']
        n = len(log_returns)
        
        x_est = np.var(log_returns)  # initial variance estimate
        P = 1.0                      # initial covariance (scalar)
        denoised_returns:List[float] = []

        for t in range(n):
            r_prev = log_returns[t-1] if t > 0 else 0.0

            # --- Predict ---
            x_pred = omega + alpha * r_prev**2 + beta * x_est
            P_pred = beta**2 * P + Q

            # --- Update ---
            H = 0.5 / np.sqrt(max(x_pred, 1e-8))  # linearization
            K = P_pred * H / (H**2 * P_pred + R)
            z = log_returns[t]
            x_est = x_pred + K * (z - np.sqrt(x_pred))
            P = (1 - K * H) * P_pred

            # --- Denoised return ---
            denoised_r = z / np.sqrt(max(x_pred, 1e-8)) * np.sqrt(max(x_est, 1e-8))
            denoised_returns.append(denoised_r)

        return np.array(denoised_returns)
