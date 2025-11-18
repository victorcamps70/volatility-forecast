import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Any, Dict, Tuple
from torch.utils.data import DataLoader, TensorDataset
from src.volforecast.models.base import BaseConfig, BaseVolModel
from functools import wraps


def track_and_plot_losses(func):
    """
    Decorator that tracks training and validation losses and optionally plots them.
    Plots only if plot=True is passed (default: False).
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        verbose = kwargs.get('verbose', True)
        plot = kwargs.get('plot', False)  # ← New parameter, default False
        
        # Store original function to capture losses
        train_losses = []
        val_losses = []
        
        # Monkey-patch print to capture epoch outputs
        original_print = print
        
        def tracking_print(*args_p, **kwargs_p):
            msg = ' '.join(str(a) for a in args_p)
            if verbose and 'Epoch' in msg and 'train=' in msg:
                # Parse: "Epoch X: train=Y.XXXXXX, val=Z.XXXXXX"
                try:
                    parts = msg.split(',')
                    train_loss = float(parts[0].split('=')[1])
                    val_loss = float(parts[1].split('=')[1]) if 'val=' in msg else None
                    train_losses.append(train_loss)
                    if val_loss is not None:
                        val_losses.append(val_loss)
                except:
                    pass
            original_print(*args_p, **kwargs_p)
        
        # Temporarily replace print
        import builtins
        builtins.print = tracking_print
        
        try:
            result = func(self, *args, **kwargs)
        finally:
            # Restore original print
            builtins.print = original_print
        
        # Plot losses only if plot=True and we have data
        if plot and (train_losses or val_losses):
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 5))
                epochs_range = range(1, len(train_losses) + 1)
                
                ax.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=5)
                if val_losses:
                    ax.plot(epochs_range, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=5)
                
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Loss (MSE)', fontsize=12)
                ax.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Could not plot losses: {e}")
        
        return result
    
    return wrapper


@dataclass
class DeepEconoNetConfig(BaseConfig):
    """Configuration for DeepEconoNet model."""
    # Target parameters
    #target_shift: int = 1                # days to shift for target calculation
    #use_log_target: bool = True
    scale_features: bool = True

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
    train_val_ratio: float = 0.8          # train/validation split ratio
    
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
    @track_and_plot_losses
    def _fit_ticker(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
        plot: bool = False,
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

    def fit_ticker(self, df: pd.DataFrame, scale_mu:float = 0, scale_sigma:float = 1) -> "DeepEconoNet":
        """
        Fit the model on a DataFrame following BaseVolModel API.
        
        Args:
            df: DataFrame with columns for return_col and target (RealVol_5d or computed)
            
        Returns:
            self (for chaining)
        """
        # Extract log returns as input features
        log_returns = (df[self.config.return_col].values.astype(np.float32) - scale_mu) / scale_sigma
        
        # Extract target (realized volatility) using build_target()
        target_series = (self.build_target(df) - scale_mu) / scale_sigma
        target = target_series.values.astype(np.float32)
        
        # Handle NaNs
        valid_mask = ~np.isnan(target)
        if not valid_mask.any():
            raise ValueError("All target values are NaN")
        target = np.nan_to_num(target, nan=float(np.nanmean(target[valid_mask])))
        
        X_train, X_val, y_train, y_val = self.train_test_split(log_returns, target)
        
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
        self._fit_ticker(train_loader, val_loader, epochs=self.config.epochs, verbose=True)
        
        return self
    
    def train_test_split(self, log_returns: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Split DataFrame into train and test sets.
        
        Args:
            log_returns: numpy array of log returns
            target: numpy array of target volatility values
        """
        # Create sequences
        X_seq, y_seq = self._create_sequences(log_returns.reshape(-1, 1), target, self.seq_len)
        
        # Split train/val based on config ratio
        split_idx = int(self.config.train_val_ratio * len(X_seq))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        return X_train, X_val, y_train, y_val

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

    def fit(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame] = None, 
            feature_col: str = "Log_Return", target_col: str = "RealVol_5d", 
            verbose: bool = True, plot: bool = False) -> "DeepEconoNet":
        """
        Train the model on a large multi-ticker training DataFrame.
        
        The input DataFrame should contain pre-normalized data (already standardized
        per ticker) with rows for multiple tickers combined. This method creates
        sequences and trains the network.
        
        Args:
            df_train: Training DataFrame with feature_col and target_col.
                     Should contain normalized data across all tickers.
            df_val: Optional validation DataFrame. If None, a split is made from df_train.
            feature_col: Column name for input features (default: "Log_Return")
            target_col: Column name for target values (default: "RealVol_5d")
            verbose: Whether to print training progress
        
        Returns:
            self (for method chaining)
        """
        # Extract features and targets
        if feature_col not in df_train.columns:
            raise KeyError(f"Feature column '{feature_col}' not found in training DataFrame")
        if target_col not in df_train.columns:
            raise KeyError(f"Target column '{target_col}' not found in training DataFrame")
        
        X_train = df_train[feature_col].values.astype(np.float32).reshape(-1, 1)
        y_train = df_train[target_col].values.astype(np.float32)
        
        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, self.seq_len)
        
        # Handle validation set
        if df_val is None:
            # Split train data
            split_idx = int(self.config.train_val_ratio * len(X_train_seq))
            X_train_split = X_train_seq[:split_idx]
            y_train_split = y_train_seq[:split_idx]
            X_val = X_train_seq[split_idx:]
            y_val = y_train_seq[split_idx:]
        else:
            X_train_split = X_train_seq
            y_train_split = y_train_seq
            X_val = df_val[feature_col].values.astype(np.float32).reshape(-1, 1)
            y_val = df_val[target_col].values.astype(np.float32)
            X_val, y_val = self._create_sequences(X_val, y_val, self.seq_len)
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_split),
            torch.FloatTensor(y_train_split.reshape(-1, 1))
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
        
        # Train
        self._fit_ticker(train_loader, val_loader, epochs=self.config.epochs, verbose=verbose, plot=plot)
        
        return self

    def train_pipeline(
            self,
            data_dir: str,
            pattern: str = "*_dataset.csv",
            feature_col: str = "Log_Return",
            target_col: str = "RealVol_5d",
            verbose: bool = True,
            plot: bool = False
        ) -> "DeepEconoNet":
        """
        Complete training pipeline: load data, normalize, and train the model.
        
        This method orchestrates the full workflow:
        1. Load all datasets matching the pattern from data_dir
        2. Normalize the data per ticker using the first normalization_fraction
        3. Call fit() to train the model on the normalized data
        
        Args:
            data_dir: Path to directory containing CSV files
            pattern: Glob pattern to match CSV files (default: "*_dataset.csv")
            normalization_fraction: Fraction of data per ticker to use for computing normalization stats
                                   (default: 0.8, meaning 80% for stats, 20% for validation)
            feature_col: Column name for input features (default: "Log_Return")
            target_col: Column name for target values (default: "RealVol_5d")
            verbose: Whether to print training progress
        
        Returns:
            self (the trained model)
        """
        from src.volforecast.data.dataset_loader import DatasetLoader
        
        # Step 1: Load all datasets
        if verbose:
            print(f"Loading datasets from {data_dir} matching pattern '{pattern}'...")
        loader = DatasetLoader()
        df_all = loader.load_all_datasets(data_dir, pattern=pattern)
        if verbose:
            print(f"Loaded {len(df_all)} rows from {df_all['ticker'].nunique()} tickers")
        
        # Step 2: Normalize per ticker
        if verbose:
            print(f"Normalizing data using first {self.config.train_val_ratio*100:.0f}% of each ticker...")
        df_train_norm, df_val_norm = loader.normalize_by_ticker(
            df_all, 
            fraction=self.config.train_val_ratio
        )
        if verbose:
            print(f"Training set: {len(df_train_norm)} rows, Validation set: {len(df_val_norm)} rows")
        
        # Step 3: Train the model
        if verbose:
            print("Training model...")
        self.fit(
            df_train_norm, 
            df_val=df_val_norm,
            feature_col=feature_col,
            target_col=target_col,
            verbose=verbose,
            plot=plot
        )
        
        if verbose:
            print("Training complete!")
        
        return self

