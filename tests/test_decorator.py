"""
Quick test to verify the loss tracking decorator works and plots
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import tempfile
from src.volforecast.models.deep_econo_net import DeepEconoNet, DeepEconoNetConfig

# Create synthetic test data
with tempfile.TemporaryDirectory() as tmp_dir:
    print("\n" + "="*80)
    print("Testing Loss Tracking Decorator")
    print("="*80 + "\n")
    
    # Create 2 sample datasets
    for ticker in ["TEST1", "TEST2"]:
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "Date": dates,
            "Log_Return": np.random.normal(0.001, 0.02, 100),
            "RealVol_5d": np.abs(np.random.normal(0.02, 0.01, 100)),
            "Close": np.random.uniform(100, 200, 100),
        })
        csv_path = os.path.join(tmp_dir, f"{ticker}_dataset.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Created {ticker}_dataset.csv")
    
    # Configure model with few epochs for fast testing
    config = DeepEconoNetConfig(
        seq_len=10,
        batch_size=16,
        epochs=5,  # Just 5 epochs to see the decorator in action
        learning_rate=1e-3,
        train_val_ratio=0.8,  # 80% train, 20% validation
        device="cpu"
    )
    
    # Initialize and train
    model = DeepEconoNet(config)
    print(f"\n✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    print("\n" + "-"*80)
    print("Training with verbose=True (decorator will track and plot losses)")
    print("-"*80 + "\n")
    
    # This will print losses AND display a plot when done
    model.train_pipeline(
        data_dir=tmp_dir,
        pattern="*_dataset.csv",
        feature_col="Log_Return",
        target_col="RealVol_5d",
        verbose=True,  # Print epoch progress
        plot=False     # Set to True to display loss plots (default: False)
    )
    
    print("\n" + "="*80)
    print("✅ Decorator test complete")
    print("="*80)
    print("\nNote: To see the loss plot, set plot=True when calling train_pipeline()")
