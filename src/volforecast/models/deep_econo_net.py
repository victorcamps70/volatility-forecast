import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple
from torch.utils.data import DataLoader, TensorDataset
from src.volforecast.models.base import BaseConfig, BaseVolModel


@dataclass
class DeepEconoNetConfig(BaseConfig):
    """Configuration for DeepEconoNet model."""
    # Sequence and input parameters
    seq_len: int = 20                    # sequence length for LSTM
    
    # Conv1d parameters
    conv_in_channels: int = 1            # input channels (one log-return per day)
    conv_out_channels: int = 16          # output channels (number of filters)
    conv_kernel_size: int = 5            # kernel size for convolution
    conv_padding: int = 2                # padding for convolution
    
    # LSTM parameters
    lstm_input_size: int = 16            # must match conv_out_channels
    lstm_hidden_size: int = 32           # hidden state size
    lstm_num_layers: int = 1             # number of LSTM layers
    
    # Fully connected layers
    fc1_hidden_size: int = 16            # first FC hidden size
    fc_output_size: int = 1              # output size (1 for volatility prediction)
    
    # Training parameters
    learning_rate: float = 1e-3          # optimizer learning rate
    epochs: int = 10                     # number of training epochs
    batch_size: int = 32                 # batch size
    
    # Device parameters
    device: str = "cpu"                  # "cpu" or "cuda"


class DeepEconoNet(BaseVolModel[DeepEconoNetConfig], nn.Module):
    def __init__(self, config: DeepEconoNetConfig):
        super().__init__(config)
        nn.Module.__init__(self)
        
        # Use GPU if available
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ----- Model Layers -----
        self.conv = nn.Conv1d(
            in_channels=self.config.conv_in_channels,
            out_channels=self.config.conv_out_channels,
            kernel_size=self.config.conv_kernel_size,
            padding=self.config.conv_padding
        )

        self.lstm = nn.LSTM(
            input_size=self.config.lstm_input_size,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
        )

        self.fc1 = nn.Linear(self.config.lstm_hidden_size, self.config.fc1_hidden_size)
        self.fc2 = nn.Linear(self.config.fc1_hidden_size, self.config.fc_output_size)

        # ----- Loss & Optimizer -----
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)

        self.seq_len = self.config.seq_len
        self.to(self.device)


    # ---------- Forward Pass ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, 1)
        Returns: (batch, 1)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)        # (batch, 1, seq_len)
        x = torch.relu(self.conv(x)) # (batch, 16, seq_len)

        # LSTM expects (seq_len, batch, features)
        x = x.transpose(1, 2)        # (batch, seq_len, 16)
        x = x.transpose(0, 1)        # (seq_len, batch, 16)

        _, (h, _) = self.lstm(x)
        h = h[-1]  # last layer's hidden state: (batch, 32)

        x = torch.relu(self.fc1(h))  # (batch, 16)
        x = self.fc2(x)              # (batch, 1)

        return x


    # ---------- One Training Step ----------
    def train_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        self.optimizer.zero_grad()
        preds = self.forward(X_batch)
        loss = self.criterion(preds, y_batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()


    # ---------- Private Training Loop (with DataLoaders) ----------
    def _fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
    ) -> None:
        """
        Internal training loop with DataLoaders.
        
        Args:
            train_loader: DataLoader yielding (X, y) tuples of tensors
            val_loader: optional DataLoader for validation
            epochs: number of training epochs
            verbose: whether to print progress
        """
        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0.0
            batches = 0

            for X_batch, y_batch in train_loader:
                loss = self.train_step(X_batch, y_batch)
                total_loss += loss
                batches += 1

            avg_train_loss = total_loss / batches

            # ----- Validation -----
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    vbatches = 0
                    for Xv, yv in val_loader:
                        Xv = Xv.to(self.device)
                        yv = yv.to(self.device)
                        preds = self.forward(Xv)
                        loss_v = self.criterion(preds, yv)
                        val_loss += loss_v.item()
                        vbatches += 1

                avg_val_loss = val_loss / vbatches

                if verbose:
                    print(f"Epoch {epoch}: train={avg_train_loss:.6f}, val={avg_val_loss:.6f}")

            else:
                if verbose:
                    print(f"Epoch {epoch}: train={avg_train_loss:.6f}")

    def fit(self, df: pd.DataFrame) -> "DeepEconoNet":
        """
        Fit the model on a DataFrame following BaseVolModel API.
        
        Args:
            df: DataFrame with columns for return_col and target (RealVol_5d or computed)
            
        Returns:
            self (for chaining)
        """
        # Extract log returns as input features
        log_returns = df[self.config.return_col].values.astype(np.float32)
        
        # Extract target (realized volatility) using build_target()
        target_series = self.build_target(df)
        target = target_series.values.astype(np.float32)
        
        # Handle NaNs
        valid_mask = ~np.isnan(target)
        if not valid_mask.any():
            raise ValueError("All target values are NaN")
        target = np.nan_to_num(target, nan=float(np.nanmean(target[valid_mask])))
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(log_returns.reshape(-1, 1), target, self.seq_len)
        
        # Split train/val (80/20)
        split_idx = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train.reshape(-1, 1))
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val.reshape(-1, 1))
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        # Call internal training loop
        self._fit(train_loader, val_loader, epochs=self.config.epochs, verbose=True)
        
        return self


    # ---------- Helper Methods for API Compatibility ----------
    @staticmethod
    def _create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM from array data."""
        X_seq: list[np.ndarray] = []
        y_seq: list[np.floating] = []
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y[i + seq_len])
        return np.array(X_seq), np.array(y_seq)

    def predict_array(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on numpy array of shape (n_samples, seq_len, 1).
        Returns predictions of shape (n_samples, 1).
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            preds = self.forward(X_tensor)
            return preds.cpu().numpy()

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Override BaseVolModel.predict to work with sequence data.
        Expected: df has columns for building sequences from log returns.
        """
        # Extract log returns and create sequences
        log_returns = df[self.config.return_col].values.astype(np.float32).reshape(-1, 1)
        
        # Only predict on valid sequences
        if len(log_returns) < self.seq_len:
            return pd.Series(np.nan, index=df.index, name="y_pred")
        
        X_seq, _ = self._create_sequences(log_returns, np.zeros(len(log_returns)), self.seq_len)
        preds = self.predict_array(X_seq).flatten()
        
        # Align predictions with original dataframe index
        # Predictions start at index seq_len
        pred_index = df.index[self.seq_len:]
        pred_series = pd.Series(preds, index=pred_index, name="y_pred")
        
        # Reindex to full length with NaNs for missing predictions
        return pred_series.reindex(df.index)

    def build_target(self, df: pd.DataFrame) -> pd.Series:
        """Override BaseVolModel.build_target to use RealVol_5d if available."""
        target_col = "RealVol_5d"
        if target_col in df.columns:
            y = df[target_col].copy()
        else:
            # Fallback to realized volatility (squared returns)
            r = df[self.config.return_col]
            y = (r ** 2).shift(-self.config.target_shift)
        return y.rename("y_true")

    def summary(self) -> Dict[str, Any]:
        """Return model summary information."""
        return {
            "model_type": "DeepEconoNet",
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "device": self.device,
            "config": {
                "seq_len": self.config.seq_len,
                "conv_out_channels": self.config.conv_out_channels,
                "lstm_hidden_size": self.config.lstm_hidden_size,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
            }
        }

