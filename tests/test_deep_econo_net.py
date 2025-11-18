"""
Test script for DeepEconoNet following BaseVolModel API.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import torch
import tempfile
from pathlib import Path

from src.volforecast.models.deep_econo_net import DeepEconoNet, DeepEconoNetConfig


def test_deep_econo_net_with_features():
    """Test DeepEconoNet with FeatureBuilder."""
    print("\n" + "=" * 80)
    print("Testing DeepEconoNet with FeatureBuilder")
    print("=" * 80)
    
    # 1. Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data/stock_info", "ADSK_dataset.csv")
    df = pd.read_csv(data_path)
    
    print(f"\n1. Loaded data: {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Use first 500 rows for testing
    df = df.iloc[:500].copy()
    print(f"   Using first {len(df)} rows")
    
    # 2. Initialize model with config
    print("\n2. Initializing DeepEconoNet...")
    seq_len = 20
    batch_size = 32
    epochs = 5
    config = DeepEconoNetConfig(
        seq_len=seq_len,
        learning_rate=1e-3,
        batch_size=batch_size,
        epochs=epochs,
        return_col="log_return"  # Use correct column name from ADSK data
    )
    model = DeepEconoNet(config=config)
    print(f"   Device: {model.device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 3. Fit model using BaseVolModel API (takes DataFrame)
    print(f"\n3. Training model for {epochs} epochs using new fit(df) API...")
    model.fit_ticker(df)
    
    print("\n" + "=" * 80)
    print("✅ Test completed successfully!")
    print("=" * 80 + "\n")
    
    return True


def test_train_function_on_reduced_dataset():
    """Test the complete train() pipeline on a reduced multi-ticker dataset."""
    print("\n" + "=" * 80)
    print("Testing DeepEconoNet.train() on reduced multi-ticker dataset")
    print("=" * 80)
    
    # 1. Create temporary directory with reduced test datasets
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("\n1. Creating reduced test datasets...")
        
        # Create 2 sample CSV files with synthetic data
        for ticker in ["AAPL", "MSFT"]:
            dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
            df = pd.DataFrame({
                "Date": dates,
                "log_return": np.random.normal(0.001, 0.02, 100),
                "RealVol_5d": np.abs(np.random.normal(0.02, 0.01, 100)),
                "Close": np.random.uniform(100, 200, 100),
            })
            csv_path = os.path.join(tmp_dir, f"{ticker}_dataset.csv")
            df.to_csv(csv_path, index=False)
            print(f"   Created {ticker}_dataset.csv with {len(df)} rows")
        
        # 2. Initialize model with reduced config for fast testing
        print("\n2. Initializing DeepEconoNet with reduced config...")
        config = DeepEconoNetConfig(
            seq_len=10,
            batch_size=16,
            epochs=2,  # Reduced epochs for fast testing
            learning_rate=1e-3,
            return_col="log_return"
        )
        model = DeepEconoNet(config=config)
        print(f"   Model device: {model.device}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        # 3. Run the train() pipeline
        print(f"\n3. Running train_pipeline() on {tmp_dir}...")
        print(f"   Pattern: *_dataset.csv")
        print(f"   Normalization fraction: 0.8")
        
        trained_model = model.train_pipeline(
            data_dir=tmp_dir,
            pattern="*_dataset.csv",
            feature_col="log_return",
            target_col="RealVol_5d",
            verbose=True
        )
        
        # 4. Verify results
        print("\n4. Verifying results...")
        assert trained_model is model, "train_pipeline() should return the same model instance"
        print("   ✓ train_pipeline() returns self")
        
        assert sum(p.numel() for p in trained_model.parameters()) > 0, "Model has parameters"
        print("   ✓ Model has trainable parameters")
        
        # 5. Test prediction on new data
        print("\n5. Testing prediction on new data...")
        test_data = np.random.normal(0.001, 0.02, (32, 10, 1)).astype(np.float32)
        test_tensor = torch.FloatTensor(test_data).to(trained_model.device)
        
        with torch.no_grad():
            predictions = trained_model(test_tensor)
        
        assert predictions.shape == (32, 1), f"Expected shape (32, 1), got {predictions.shape}"
        print(f"   ✓ Predictions shape: {predictions.shape}")
        print(f"   ✓ Prediction range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    
    print("\n" + "=" * 80)
    print("✅ train_pipeline() test completed successfully!")
    print("=" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    # Run both tests
    test1_success = test_deep_econo_net_with_features()
    test2_success = test_train_function_on_reduced_dataset()
    
    overall_success = test1_success and test2_success
    sys.exit(0 if overall_success else 1)
