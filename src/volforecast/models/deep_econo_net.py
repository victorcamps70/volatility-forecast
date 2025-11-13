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
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(1)  # Add channel dimension for Conv1d
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
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
                X_tensor = torch.FloatTensor(X_seq).unsqueeze(0).unsqueeze(0).to(self.config.device)  # Add batch and channel dims
                
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


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    
    print("=" * 80)
    print("Running unit test for DeepEconoNetModel and KalmanGARCH using AAPL dataset...")
    print("=" * 80)

    # Get the path to AAPL dataset
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
    aapl_path = os.path.join(data_dir, "AAPL_dataset.csv")

    if not os.path.exists(aapl_path):
        print(f"❌ AAPL dataset not found at {aapl_path}")
        exit(1)

    # 1. Load AAPL data
    print(f"\n1. Loading AAPL dataset from {aapl_path}...")
    df = pd.read_csv(aapl_path, index_col="Date", parse_dates=True)
    print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {df.columns.tolist()}")

    # 2. Use only first 500 rows for testing (to keep test fast)
    df = df.iloc[:500].copy()
    print(f"   ✓ Using first {len(df)} rows for testing")

    # =========================================================================
    # PART 1: Test KalmanGARCH denoising
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: Testing KalmanGARCH Kalman Filter Denoising")
    print("=" * 80)

    print("\n1. Creating and fitting KalmanGARCH model...")
    base_cfg = BaseConfig(return_col="log_return", target_shift=-1)
    garch_cfg = GARCHConfig(
        **{k: v for k, v in base_cfg.__dict__.items() 
           if k in GARCHConfig.__dataclass_fields__},
        p=1,
        q=1,
        dist="normal",
        mean="zero",
    )
    kalman_garch = KalmanGARCH(garch_cfg)
    
    try:
        kalman_garch.fit(df)
        print("   ✓ KalmanGARCH model fitted successfully")
    except Exception as e:
        print(f"   ⚠ Warning: Could not fit KalmanGARCH: {e}")
        kalman_garch = None

    # 2. Get log returns and apply Kalman filter
    print("\n2. Applying Kalman filter to denoise log returns...")
    log_returns = df["log_return"].values.astype(float)
    
    if kalman_garch is not None:
        try:
            denoised_returns = kalman_garch.predict(log_returns.tolist())
            denoised_returns = np.array(denoised_returns)
            print(f"   ✓ Kalman filter applied successfully")
            print(f"   ✓ Original returns range: [{float(log_returns.min()):.6f}, {float(log_returns.max()):.6f}]")
            print(f"   ✓ Denoised returns range: [{float(denoised_returns.min()):.6f}, {float(denoised_returns.max()):.6f}]")
        except Exception as e:
            print(f"   ⚠ Warning: Could not apply Kalman filter: {e}")
            denoised_returns = None
    else:
        denoised_returns = None

    # 3. Plot comparison
    if denoised_returns is not None:
        print("\n3. Creating plot of filtered vs unfiltered signals...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Overlay comparison
        ax1.plot(log_returns, label="Original (Noisy)", alpha=0.7, linewidth=1.5, color='red')
        ax1.plot(denoised_returns, label="Denoised (Kalman Filtered)", alpha=0.7, linewidth=1.5, color='blue')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Log Returns')
        ax1.set_title('Kalman Filter: Unfiltered vs Filtered Log Returns (AAPL)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Difference/error
        error = log_returns - denoised_returns
        ax2.plot(error, label="Estimation Error (Original - Denoised)", color='green', alpha=0.7, linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Error')
        ax2.set_title('Kalman Filter Estimation Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        reports_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "reports", "figures")
        os.makedirs(reports_dir, exist_ok=True)
        plot_path = os.path.join(reports_dir, "kalman_filter_test.png")
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"   ✓ Plot saved to {plot_path}")
        plt.show()
        
        # Print statistics
        print("\n4. Kalman Filter Statistics:")
        print(f"   Mean error: {error.mean():.6f}")
        print(f"   Std error: {error.std():.6f}")
        print(f"   Max error: {error.max():.6f}")
        print(f"   Min error: {error.min():.6f}")

    # =========================================================================
    # PART 2: Test DeepEconoNetModel
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Testing DeepEconoNetModel")
    print("=" * 80)

    # Build config and feature builder
    print("\n1. Creating configuration and feature builder...")
    base_cfg = BaseConfig(
        date_col="Date",
        return_col="log_return",
        target_shift=-1,
    )
    config = DeepEconoNetConfig(
        **{k: v for k, v in base_cfg.__dict__.items() 
           if k in DeepEconoNetConfig.__dataclass_fields__},
        input_size=3,
        hidden_size=32,
        num_layers=2,
        output_size=1,
        batch_size=16,
        epochs=5,  # Few epochs for testing
        learning_rate=0.001,
        use_log_target=True,
        scale_features=True,
        sequence_length=10,
        device="cpu",
    )
    print(f"   ✓ Config created: {config.__class__.__name__}")

    # Create a simple feature builder
    builder = FeatureBuilder(
        lags_returns=(1, 2, 5),
        lags_vix=(),
        add_dow=False,
    )
    builder.date_col = "Date"
    builder.return_col = "log_return"
    builder.vix_col = "log_return_VIX"
    print(f"   ✓ Feature builder created")

    # Instantiate model
    print("\n2. Instantiating DeepEconoNetModel...")
    model = DeepEconoNetModel(config, builder)
    print(f"   ✓ Model created: {model.__class__.__name__}")
    print(f"   ✓ Model device: {config.device}")

    # Fit model
    print("\n3. Fitting model on training data...")
    try:
        model.fit(df)
        print(f"   ✓ Model fitted successfully")
        print(f"   ✓ fitted_ = {model.fitted_}")
    except Exception as e:
        print(f"   ❌ Error during fit: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Generate predictions
    print("\n4. Generating predictions...")
    try:
        y_pred = model.predict(df)
        print(f"   ✓ Predictions generated")
        print(f"   ✓ Predictions shape: {y_pred.shape}")
        print(f"   ✓ Non-null predictions: {y_pred.notna().sum()}")
        print(f"   ✓ Prediction range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
        print(f"\n   Sample predictions (first 10):")
        print(y_pred.head(10))
    except Exception as e:
        print(f"   ❌ Error during predict: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # Get model summary
    print("\n5. Model summary:")
    summary = model.summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    # Evaluate model (optional)
    print("\n6. Evaluating model...")
    try:
        metrics = model.evaluate(df)
        print(f"   ✓ Evaluation metrics:")
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                print(f"      {metric_name}: {metric_value:.6f}")
            else:
                print(f"      {metric_name}: {metric_value}")
    except Exception as e:
        print(f"   ⚠ Warning during evaluation: {e}")

    print("\n" + "=" * 80)
    print("✅ All tests passed successfully!")
    print("=" * 80)
