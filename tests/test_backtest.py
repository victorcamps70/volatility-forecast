"""
Simple backtest test using saved model checkpoint.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
from pathlib import Path

from src.volforecast.models.deep_econo_net import DeepEconoNet, DeepEconoNetConfig
from src.volforecast.evaluation.backtest import rolling_backtest
from src.volforecast.training.training_deep_econo_net import load_training_checkpoint, get_checkpoint_dir


def test_backtest_with_checkpoint():
    """Test backtest functionality using a saved model checkpoint."""
    print("\n" + "=" * 80)
    print("Testing Backtest with Saved Model Checkpoint")
    print("=" * 80)
    
    # 1. Find and load checkpoint
    print("\n1. Loading checkpoint...")
    checkpoint_dir = get_checkpoint_dir()
    checkpoint_files = list(Path(checkpoint_dir).glob("*.pt"))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Use the most recent checkpoint
    checkpoint_path = max(checkpoint_files, key=os.path.getctime)
    print(f"   Found checkpoint: {checkpoint_path.name}")
    
    # Load the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, checkpoint_data = load_training_checkpoint(checkpoint_path, device=device)
    print(f"   ✓ Model loaded on device: {device}")
    if 'epoch' in checkpoint_data:
        print(f"   Epoch: {checkpoint_data['epoch']}")
    
    # 2. Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data/stock_info", "ADSK_dataset.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    print(f"\n2. Loaded data: {len(df)} rows")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Use first 300 rows for quick test
    df = df.iloc[:300].copy()
    print(f"   Using first {len(df)} rows for test")
    # 3. Run rolling backtest
    print("\n3. Running rolling backtest...")
    
    # For pre-trained model, we create a lightweight wrapper that skips retraining
    class PredictionOnlyModel:
        def __init__(self, trained_model):
            self.model = trained_model
        
        def fit(self, train_df):
            # Skip fitting for pre-trained model
            pass
        
        def predict(self, df):
            return self.model.predict(df)
        
        def build_target(self, df):
            return self.model.build_target(df)
    
    prediction_model = PredictionOnlyModel(model)
    
    train_end = df.index[100]
    
    result = rolling_backtest(
        prediction_model,
        df,
        train_start=df.index[0],
        train_end=train_end,
        step="5D",
        min_history=20,
        fixed_window=100
    )
    
    # 4. Check results
    print("\n4. Backtest Results:")
    metrics = result["metrics"]
    print(f"   Number of predictions: {metrics['n']}")
    print(f"   RMSE: {metrics['RMSE']:.6f}")
    print(f"   MAE: {metrics['MAE']:.6f}")
    print(f"   QLIKE: {metrics['QLIKE']:.6f}")
    
    y_true = result["y_true"]
    y_pred = result["y_pred"]
    
    print(f"\n   True values shape: {y_true.shape}")
    print(f"   Pred values shape: {y_pred.shape}")
    print(f"   True values (first 5):\n{y_true.head()}")
    print(f"   Pred values (first 5):\n{y_pred.head()}")
    
    # 5. Basic assertions
    assert metrics['n'] > 0, "No predictions made"
    assert not np.isnan(metrics['RMSE']), "RMSE is NaN"
    assert not np.isnan(metrics['MAE']), "MAE is NaN"
    assert not np.isnan(metrics['QLIKE']), "QLIKE is NaN"
    assert len(y_true) > 0, "No true values returned"
    assert len(y_pred) > 0, "No predictions returned"
    
    print("\n✓ Backtest test passed!")
    return result


if __name__ == "__main__":
    result = test_backtest_with_checkpoint()
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
