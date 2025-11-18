"""
Test script for DeepEconoNet following BaseVolModel API.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

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


if __name__ == "__main__":
    success = test_deep_econo_net_with_features()
    sys.exit(0 if success else 1)
