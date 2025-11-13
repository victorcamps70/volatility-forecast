from dataclasses import dataclass
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
import sys
import os

# Handle imports for both package and direct script execution
try:
    from src.volforecast.models.base import BaseVolModel, BaseConfig
    from src.volforecast.models.garch_model import GARCHConfig, GARCHVolModel
    from src.volforecast.features.builders import FeatureBuilder
except ImportError:
    # If running as a script, add parent directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    from src.volforecast.models.base import BaseVolModel, BaseConfig
    from src.volforecast.models.garch_model import GARCHConfig, GARCHVolModel
    from src.volforecast.features.builders import FeatureBuilder

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
    sequence_length: int = 20
    device: str = "cpu"
    random_state: int = 42

class DeepEconoNetModel(BaseVolModel[DeepEconoNetConfig], nn.Module):
    def __init__(self, config: DeepEconoNetConfig, feature_builder: FeatureBuilder):
        super().__init__(config)
        nn.Module.__init__(self)
        self.config = config
        self.feature_builder = feature_builder
        self.fitted_: bool = False

        self.scaler: StandardScaler | None = None
        self.kalman_garch = KalmanGARCH(config)
        self.conv1d = nn.Conv1d(in_channels=config.input_size, out_channels=config.hidden_size, kernel_size=5, padding=1)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=config.num_layers, batch_first=True, dropout=config.dropout if config.num_layers > 1 else 0.0)
        self.fc1 = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.dropout_layer = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(in_features=config.hidden_size, out_features=config.output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last time step
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x

    # --- Private helpers
    def _build_X(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.feature_builder.build_features(df)

    def _build_y(self, df: pd.DataFrame) -> pd.Series:
        y = self.build_target(df)  # from BaseVolModel
        if self.config.use_log_target:
            y = np.log(y + self.config.eps)
        return y.rename("y")
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
        """Train for one epoch."""
        self.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.config.device)
            y_batch = y_batch.to(self.config.device)
            
            # Forward pass
            y_pred = self(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    # --- Required interface
    def fit(self, df: pd.DataFrame) -> "DeepEconoNetModel":
        X = self._build_X(df)
        y = self._build_y(df)
        valid = X.notna().all(axis=1) & y.notna()
        X, y = X[valid], y[valid]

        # Fit Kalman-GARCH model
        self.kalman_garch.fit(df[valid])

        # Convert X and y to numpy
        X_np = X.values
        y_np = y.values
        
        # Scale features if requested
        if self.config.scale_features:
            self.scaler = StandardScaler()
            X_np = self.scaler.fit_transform(X_np)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_np, y_np, self.config.sequence_length)
        
        # Create data loader
        # X_seq shape: (num_sequences, sequence_length, num_features)
        # Need shape: (num_sequences, num_features, sequence_length) for Conv1d
        X_tensor = torch.FloatTensor(X_seq).transpose(1, 2)  # Convert to [batch, features, seq_len]
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        # Note: shuffle=False because data is time series and temporal order matters
        train_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Setup training
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.MSELoss()
        
        # Train
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        
        for epoch in range(self.config.epochs):
            loss = self._train_epoch(train_loader, optimizer, loss_fn)
            if (epoch + 1) % max(1, self.config.epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{self.config.epochs}, Loss: {loss:.4f}")

        self.fitted_ = True
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        assert self.fitted_, "Call fit() first."
        X = self._build_X(df)
        
        # Keep only rows with all features present
        valid = X.notna().all(axis=1)
        X_valid = X[valid].values
        
        # Scale if we used scaling during fit
        if self.config.scale_features and self.scaler is not None:
            X_valid = self.scaler.transform(X_valid)
        
        # Create sequences for prediction
        predictions = []
        self.eval()
        
        with torch.no_grad():
            for i in range(len(X_valid) - self.config.sequence_length):
                X_seq = X_valid[i:i + self.config.sequence_length]
                # Shape (seq_len, features) -> need to convert to (1, features, seq_len) for Conv1d
                X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).transpose(1, 2).to(self.config.device)
                
                y_pred = self(X_tensor).cpu().numpy()[0, 0]
                predictions.append(y_pred)
        
        # Create result series with proper indexing
        valid_indices = X[valid].index
        pred_indices = valid_indices[self.config.sequence_length:]
        
        yhat = pd.Series(predictions, index=pred_indices, name="y_pred")
        
        # Reindex to original dataframe index (NaN where predictions not available)
        yhat = yhat.reindex(df.index)
        
        # Invert log if used
        if self.config.use_log_target:
            yhat = yhat.where(yhat.isna(), np.exp(yhat) - self.config.eps)
        
        return yhat.clip(lower=0.0).rename("y_pred")

    def summary(self) -> Dict[str, Any]:
        return {
            "model_architecture": str(self),
            "fitted": self.fitted_,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }

class KalmanGARCH:
    def __init__(self, config: DeepEconoNetConfig):
        self.garch_model = GARCHVolModel(config)

    def fit(self, df: pd.DataFrame) -> "KalmanGARCH":
        self.garch_model.fit(df)
        return self

    # --- 2. Predict / denoise function ---
    def predict(self, log_returns:List[float], Q:float=1e-4, R:float=1e-2):
        """
        Kalman filter to denoise log-returns.
        
        State model: x_{t+1} = x_t (constant return)
        Measurement model: z_t = x_t + v_t, where v_t ~ N(0, R)
        
        Inputs:
            log_returns : list or array of log-returns
            Q : process noise (model uncertainty)
            R : measurement noise (observation uncertainty)
        
        Output:
            denoised_returns : array of same length (filtered signal)
        """
        log_returns_array = np.asarray(log_returns, dtype=np.float64)
        n = len(log_returns_array)
        
        # Kalman filter state
        x = log_returns_array[0]  # Initial state estimate
        P = 1.0                    # Initial estimation error covariance
        
        denoised_returns = []
        
        for t in range(n):
            z = log_returns_array[t]  # Measurement (noisy observation)
            
            # Predict step
            x_pred = x                    # State prediction (constant model)
            P_pred = P + Q                # Error covariance prediction
            
            # Update step (measurement update)
            # Kalman gain
            K = P_pred / (P_pred + R)
            
            # State update
            x = x_pred + K * (z - x_pred)
            
            # Error covariance update
            P = (1 - K) * P_pred
            
            denoised_returns.append(x)
        
        return np.array(denoised_returns, dtype=np.float64)
