import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Tuple
from torch.utils.data import DataLoader, TensorDataset
from src.volforecast.models.base import BaseConfig, BaseVolModel


@dataclass
class DeepEconoNetConfig(BaseConfig):
    """Configuration for DeepEconoNet model."""
    # Target parameters
    #target_shift: int = 1                # days to shift for target calculation
    #use_log_target: bool = True
    scale_features: bool = True
    scales: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=dict)  # {ticker: ((returns_mu, returns_sigma), (target_mu, target_sigma))}

    # Sequence and input parameters
    seq_len: int = 20                    # sequence length for LSTM
    
    # Conv1d parameters
    conv_in_channels: int = 1            # input channels (one log-return per day)
    conv_out_channels: int = 16          # output channels (number of filters)
    conv_kernel_size: int = 5            # kernel size for convolution
    conv_padding: int = 2                # padding for convolution
    
    # LSTM parameters
    lstm_input_size: int = conv_out_channels
    lstm_hidden_size: int = 32           # hidden state size
    lstm_num_layers: int = 1             # number of LSTM layers
    
    # Fully connected layers
    fc1_hidden_size: int = 16            # first FC hidden size
    fc_output_size: int = 1              # output size (1 for volatility prediction)
    
    # Training parameters
    learning_rate: float = 1e-4          # optimizer learning rate
    epochs: int = 50                     # number of training epochs
    batch_size: int = 64                 # batch size
    train_val_ratio: float = 0.8         # train/validation split ratio
    gradient_accumulation_steps: int = 1 # gradient accumulation steps for larger effective batch size
    
    # Device parameters
    device: str = "cpu"                  # "cpu" or "cuda" (auto-detects GPU if available)
    use_amp: bool = True                 # Use Automatic Mixed Precision for faster training on GPU
    
    # Loss filtering parameters
    skip_high_loss: bool = True          # skip training if loss exceeds threshold
    loss_threshold: float = 1.0          # loss threshold for skipping ticker
    skip_epochs: int = 2                 # number of initial epochs to check threshold
    
    # Training history (for plotting)
    training_history: Dict[str, Dict[str, list]] = field(default_factory=dict)  # {ticker: {"train_losses": [...], "val_losses": [...]}}


class DeepEconoNet(BaseVolModel[DeepEconoNetConfig], nn.Module):
    def __init__(self, config: DeepEconoNetConfig):
        super().__init__(config)
        nn.Module.__init__(self)
        
        # Initialize device with GPU acceleration if available
        # Priority: explicit config.device setting > auto-detect GPU > CPU fallback
        if self.config.device == "cpu":
            # Explicit CPU request - always use CPU
            self.device = "cpu"
        elif self.config.device == "cuda":
            # GPU requested - use GPU if available, fallback to CPU
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            # Fallback: use CPU by default
            self.device = "cpu"
        
        # Log device info for debugging
        if self.device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

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
            batch_first=False,  # Keep False for current implementation, but optimized for GPU
        )

        self.fc1 = nn.Linear(self.config.lstm_hidden_size + 1, self.config.fc1_hidden_size)
        self.fc2 = nn.Linear(self.config.fc1_hidden_size, self.config.fc_output_size)

        # ----- Loss & Optimizer -----
        self.criterion = nn.MSELoss()
        # annotate optimizer with the general Optimizer type so .step has the correct signature
        self.optimizer: optim.Optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        
        # Initialize GradScaler for Automatic Mixed Precision (AMP)
        # Only used if use_amp=True and device is CUDA
        self.scaler = GradScaler() if (self.config.use_amp and self.device == "cuda") else None

        self.seq_len = self.config.seq_len
        self.to(self.device)


    # ---------- Forward Pass ----------
    def forward(self, x: torch.Tensor, vol:torch.Tensor) -> torch.Tensor:
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
        h = h[-1]                    # last layer's hidden state: (batch, 32)

        x = torch.concat([h, vol], dim=1)  # (batch, 33) = 32 hidden + 1 volatility
        x = self.fc1(x)              # (batch, 16)
        x = torch.relu(x)            # (batch, 16)
        x = self.fc2(x)              # (batch, 1)

        return x


    # ---------- Volatility Computation ----------
    @staticmethod
    def _compute_current_vol(X_seq: np.ndarray) -> np.ndarray:
        """
        Compute current realized volatility for each sequence.
        
        Args:
            X_seq: sequences of log returns (batch, seq_len, 1)
            
        Returns:
            current volatility for each sequence (batch, 1)
        """
        # RMS of returns in sequence
        return np.sqrt(np.mean(X_seq[:, :, 0] ** 2, axis=1, keepdims=True))

    # ---------- One Training Step ----------
    def train_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        X_batch = X_batch.to(self.device, dtype=torch.float32)
        y_batch = y_batch.to(self.device, dtype=torch.float32)
        
        # Compute current volatility from sequences
        # Move to CPU only for numpy computation if on GPU (minimal overhead)
        X_np = X_batch.cpu().numpy() if self.device == "cuda" else X_batch.numpy()
        vol_batch = torch.as_tensor(self._compute_current_vol(X_np), dtype=torch.float32, device=self.device)

        self.optimizer.zero_grad()
        
        # Use Automatic Mixed Precision if enabled and on CUDA
        if self.scaler is not None:
            with autocast('cuda', dtype=torch.float16):
                preds = self.forward(X_batch, vol_batch)
                loss = self.criterion(preds, y_batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            preds = self.forward(X_batch, vol_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()
            self.optimizer.step()

        return loss.item()


    # ---------- Private Training Loop (with DataLoaders) ----------
    def _fit_ticker(
        self,
        train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]],
        val_loader: Optional[DataLoader[Tuple[torch.Tensor, torch.Tensor]]] = None,
        epochs: int = 10,
        verbose: bool = True,
        ticker: Optional[str] = None,
    ) -> None:
        """
        Internal training loop with DataLoaders.
        
        Args:
            train_loader: DataLoader yielding (X, y) tuples of tensors
            val_loader: optional DataLoader for validation
            epochs: number of training epochs
            verbose: whether to print progress
            ticker: optional ticker name to track losses
        """
        # Initialize loss tracking for this ticker
        if ticker is not None:
            self.config.training_history[ticker] = {"train_losses": [], "val_losses": []}
        
        early_epoch_val_losses = []
        
        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0.0
            batches = 0

            for X_batch, y_batch in train_loader:
                loss = self.train_step(X_batch, y_batch)
                total_loss += loss
                batches += 1

            avg_train_loss = total_loss / batches
            
            # Track train loss
            if ticker is not None:
                self.config.training_history[ticker]["train_losses"].append(avg_train_loss)

            # ----- Validation -----
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    vbatches = 0
                    for Xv, yv in val_loader:
                        Xv = Xv.to(self.device, dtype=torch.float32)
                        yv = yv.to(self.device, dtype=torch.float32)
                        # Compute current volatility from sequences
                        Xv_np = Xv.cpu().numpy() if self.device == "cuda" else Xv.numpy()
                        vol_v = torch.as_tensor(self._compute_current_vol(Xv_np), dtype=torch.float32, device=self.device)
                        preds = self.forward(Xv, vol_v)
                        loss_v = self.criterion(preds, yv)
                        val_loss += loss_v.item()
                        vbatches += 1

                avg_val_loss = val_loss / vbatches
                
                # Track first N epochs val loss for threshold check
                if epoch <= self.config.skip_epochs:
                    early_epoch_val_losses.append(avg_val_loss)
                
                # Check if all early epochs val loss exceed threshold
                if epoch == self.config.skip_epochs and self.config.skip_high_loss:
                    if all(loss > self.config.loss_threshold for loss in early_epoch_val_losses):
                        if ticker is not None:
                            print(f"⚠️  Skipping {ticker}: validation loss in first {self.config.skip_epochs} epochs exceeds {self.config.loss_threshold}")
                        return
                
                # Track val loss
                if ticker is not None:
                    self.config.training_history[ticker]["val_losses"].append(avg_val_loss)

                if verbose:
                    print(f"Epoch {epoch}: train={avg_train_loss:.6f}, val={avg_val_loss:.6f}")
                

            else:
                if verbose:
                    print(f"Epoch {epoch}: train={avg_train_loss:.6f}")

    def _compute_scaling_params(self, log_returns: np.ndarray, target: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Compute mean and standard deviation separately for log_returns and target.
        
        Statistics are computed from the first train_val_ratio percentage of data,
        avoiding data leakage into validation and test sets.
        
        Args:
            log_returns: Full array of log returns (temporal data)
            target: Full array of target values (temporal data)
            
        Returns:
            Tuple of ((returns_mean, returns_stdev), (target_mean, target_stdev))
        """
        train_idx = int(self.config.train_val_ratio * len(log_returns))
        
        # Compute statistics for log_returns from training portion only
        train_returns = log_returns[:train_idx].flatten()
        valid_returns = train_returns[~np.isnan(train_returns)]
        
        if len(valid_returns) > 0:
            returns_mu = float(np.mean(valid_returns))
            returns_sigma = float(np.std(valid_returns))
        else:
            returns_mu, returns_sigma = 0.0, 1.0
        
        # Avoid division by zero
        if returns_sigma == 0:
            returns_sigma = 1.0
        
        # Compute statistics for target from training portion only
        train_target = target[:train_idx]
        valid_target = train_target[~np.isnan(train_target)]
        
        if len(valid_target) > 0:
            target_mu = float(np.mean(valid_target))
            target_sigma = float(np.std(valid_target))
        else:
            target_mu, target_sigma = 0.0, 1.0
        
        # Avoid division by zero
        if target_sigma == 0:
            target_sigma = 1.0
            
        return (returns_mu, returns_sigma), (target_mu, target_sigma)

    def fit_ticker(self, df: pd.DataFrame, ticker: Optional[str] = None) -> "DeepEconoNet":
        """
        Fit the model on a DataFrame following BaseVolModel API.
        
        Args:
            df: DataFrame with columns for return_col and target (RealVol_5d or computed)
            ticker: Optional ticker name to cache scaling parameters
            
        Returns:
            self (for chaining)
        """
        # Extract log returns as raw features (before scaling)
        # Ensure a 1D numpy array of dtype float32 to avoid ambiguous tuple/object dtypes
        log_returns = df[self.config.return_col].to_numpy(dtype=np.float32).reshape(-1)
        
        # Extract target (realized volatility) as raw values
        target_series = self.build_target(df)
        target = target_series.to_numpy(dtype=np.float32).reshape(-1)
        
        # Apply scaling only if enabled in config
        returns_mu, returns_sigma = 0.0, 1.0
        target_mu, target_sigma = 0.0, 1.0
        
        if self.config.scale_features:
            (returns_mu, returns_sigma), (target_mu, target_sigma) = self._compute_scaling_params(log_returns, target)
            
            # Cache scaling parameters by ticker if provided
            if ticker is not None:
                self.config.scales[ticker] = ((returns_mu, returns_sigma), (target_mu, target_sigma))
            
            # Apply separate scaling for log_returns and target
            log_returns = (log_returns - returns_mu) / returns_sigma
            target = (target - target_mu) / target_sigma
        
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
        
        # Enable pin_memory for GPU acceleration and num_workers for parallel loading
        # Use num_workers=0 for Windows/GPU stability, > 0 for CPU training
        pin_mem = self.device == "cuda"
        num_workers = 0
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=pin_mem,
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=pin_mem,
            num_workers=num_workers
        )
        
        # Call internal training loop
        self._fit_ticker(train_loader, val_loader, epochs=self.config.epochs, verbose=True, ticker=ticker)
        
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
        
        Optimized for GPU inference with efficient tensor handling.
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor and move to device with proper dtype
            X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            # Compute current volatility from sequences
            vol_batch = torch.as_tensor(self._compute_current_vol(X), dtype=torch.float32, device=self.device)
            preds = self.forward(X_tensor, vol_batch)
            return preds.cpu().numpy()

    def predict(self, df: pd.DataFrame, ticker: Optional[str] = None) -> pd.Series:
        """
        Override BaseVolModel.predict to work with sequence data.
        Expected: df has columns for building sequences from log returns.
        
        Args:
            df: DataFrame with log returns column
            ticker: Optional ticker name to retrieve cached scaling parameters
        """
        # Extract log returns and apply scaling if available
        log_returns = df[self.config.return_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        
        # Track scaling parameters for denormalization
        target_mu, target_sigma = 0.0, 1.0
        
        # Apply scaling if enabled and ticker scales are available
        if self.config.scale_features and ticker is not None and ticker in self.config.scales:
            (returns_mu, returns_sigma), (target_mu, target_sigma) = self.config.scales[ticker]
            log_returns = (log_returns - returns_mu) / returns_sigma
        
        # Only predict on valid sequences
        if len(log_returns) < self.seq_len:
            return pd.Series(np.nan, index=df.index, name="y_pred")
        
        X_seq, _ = self._create_sequences(log_returns, np.zeros(len(log_returns)), self.seq_len)
        preds = self.predict_array(X_seq).flatten()
        
        # Denormalize predictions if scaling was applied
        if self.config.scale_features and (target_mu != 0.0 or target_sigma != 1.0):
            preds = preds * target_sigma + target_mu
        
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

    def fit_all_datasets(self, data_dir: str, pattern: str = "*_dataset.csv", verbose: bool = True, shuffle: bool = True, exclude_regex: str = r"^\d") -> "DeepEconoNet":
        """
        Load all CSV files matching the pattern from data_dir and train on each dataset.
        
        This method:
        1. Finds all files matching the pattern in data_dir (non-recursive)
        2. Optionally filters by exclude_regex
        3. Optionally shuffles the order
        4. Loads each CSV file as a DataFrame
        5. Calls fit_ticker() on each DataFrame
        
        Args:
            data_dir: Path to directory containing CSV files
            pattern: Glob pattern to match CSV files (default: "*_dataset.csv")
            verbose: Whether to print progress messages
            shuffle: Whether to shuffle the file order (default: False)
            exclude_regex: Regex pattern to exclude files; default r"^\d" excludes tickers starting with a number
            
        Returns:
            self (for method chaining)
        """
        import os
        import glob
        import re
        
        # Construct search pattern
        search_pattern = os.path.join(data_dir, pattern)
        files: list[str] = sorted(glob.glob(search_pattern))
        
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' found in: {data_dir}")
        
        # Filter by exclude_regex if provided
        if exclude_regex:
            filtered_files = []
            for f in files:
                filename = os.path.basename(f)
                ticker = filename.replace("_dataset.csv", "").replace(".csv", "")
                if not re.match(exclude_regex, ticker):
                    filtered_files.append(f)
            files = filtered_files
        
        # Shuffle if requested
        if shuffle:
            import random
            random.shuffle(files)
        
        if verbose:
            print(f"Found {len(files)} file(s) matching criteria")
        
        for file_path in files:
            filename = os.path.basename(file_path)
            try:
                if verbose:
                    print(f"\nLoading and training on: {filename}")
                
                # Extract ticker name from filename (e.g., "ADSK_dataset.csv" -> "ADSK")
                ticker = filename.replace("_dataset.csv", "").replace(".csv", "")
                
                # Load the CSV file
                df: pd.DataFrame = pd.read_csv(file_path)  # type: ignore
                
                # Train on this dataset with ticker name for caching scales
                self.fit_ticker(df, ticker=ticker)
                
                if verbose:
                    print(f"  ✓ Completed: {filename}")
                    
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error processing {filename}: {e}")
                continue
        
        if verbose:
            print(f"\nTraining complete on {len(files)} file(s)")
        
        return self

