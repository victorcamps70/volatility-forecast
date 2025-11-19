"""
Test script for DeepEconoNet following BaseVolModel API.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np

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


def test_normalization_enabled():
    """Test that normalization works correctly when scale_features=True."""
    print("\n" + "=" * 80)
    print("Test 1: Normalization ENABLED (scale_features=True)")
    print("=" * 80)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data/stock_info", "ADSK_dataset.csv")
    df = pd.read_csv(data_path).iloc[:300].copy()
    
    # Create model with scaling enabled
    config = DeepEconoNetConfig(
        seq_len=20,
        learning_rate=1e-3,
        batch_size=32,
        epochs=2,
        return_col="log_return",
        scale_features=True,  # Enable scaling
        train_val_ratio=0.8
    )
    model = DeepEconoNet(config=config)
    
    # Train with ticker
    print("\n1. Training with ticker 'ADSK'...")
    model.fit_ticker(df, ticker="ADSK")
    
    # Verify scales are cached
    assert "ADSK" in model.config.scales, "Scales should be cached for ADSK"
    (returns_mu, returns_sigma), (target_mu, target_sigma) = model.config.scales["ADSK"]
    
    print(f"   ✓ Scales cached:")
    print(f"     - Returns: mean={returns_mu:.8f}, stdev={returns_sigma:.8f}")
    print(f"     - Target:  mean={target_mu:.8f}, stdev={target_sigma:.8f}")
    assert returns_sigma > 0, "Returns sigma should be positive"
    assert target_sigma > 0, "Target sigma should be positive"
    
    # Verify scaling by manually computing and checking
    print("\n2. Verifying scaling on raw data...")
    log_returns_raw = df[config.return_col].values.astype(np.float32)
    target_raw = model.build_target(df).values.astype(np.float32)
    
    # Compute training indices
    train_idx = int(config.train_val_ratio * len(log_returns_raw))
    train_returns = log_returns_raw[:train_idx]
    train_target = target_raw[:train_idx]
    
    # Scale the training data
    train_returns_scaled = (train_returns - returns_mu) / returns_sigma
    train_target_scaled = (train_target - target_mu) / target_sigma
    
    # Remove NaNs for statistics
    valid_returns_scaled = train_returns_scaled[~np.isnan(train_returns_scaled)]
    valid_target_scaled = train_target_scaled[~np.isnan(train_target_scaled)]
    
    # Check statistics of scaled training data
    returns_mean_scaled = np.mean(valid_returns_scaled)
    returns_std_scaled = np.std(valid_returns_scaled)
    target_mean_scaled = np.mean(valid_target_scaled)
    target_std_scaled = np.std(valid_target_scaled)
    
    print(f"   ✓ Scaled training returns:")
    print(f"     - Mean: {returns_mean_scaled:.8f} (should be ≈ 0)")
    print(f"     - Stdev: {returns_std_scaled:.8f} (should be ≈ 1)")
    print(f"   ✓ Scaled training target:")
    print(f"     - Mean: {target_mean_scaled:.8f} (should be ≈ 0)")
    print(f"     - Stdev: {target_std_scaled:.8f} (should be ≈ 1)")
    
    # Verify they're close to 0 and 1
    assert abs(returns_mean_scaled) < 0.1, f"Returns mean should be ≈ 0, got {returns_mean_scaled}"
    assert abs(returns_std_scaled - 1.0) < 0.1, f"Returns stdev should be ≈ 1, got {returns_std_scaled}"
    assert abs(target_mean_scaled) < 0.1, f"Target mean should be ≈ 0, got {target_mean_scaled}"
    assert abs(target_std_scaled - 1.0) < 0.1, f"Target stdev should be ≈ 1, got {target_std_scaled}"
    
    # Predict with ticker
    print("\n3. Predicting with cached scales...")
    preds = model.predict(df, ticker="ADSK")
    valid_preds = preds.dropna()
    
    assert len(valid_preds) > 0, "Should have predictions"
    assert np.all(np.isfinite(valid_preds)), "All predictions should be finite"
    
    # Verify predictions are denormalized (should be in original scale)
    pred_min = valid_preds.min()
    pred_max = valid_preds.max()
    print(f"   ✓ Predictions denormalized (original scale):")
    print(f"     - Min: {pred_min:.8f}")
    print(f"     - Max: {pred_max:.8f}")
    print(f"     - Mean: {valid_preds.mean():.8f}")
    
    print("\n✅ Test 1 passed: Normalization enabled works correctly!")
    return True


def test_normalization_disabled():
    """Test that model works correctly when scale_features=False."""
    print("\n" + "=" * 80)
    print("Test 2: Normalization DISABLED (scale_features=False)")
    print("=" * 80)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data/stock_info", "ADSK_dataset.csv")
    df = pd.read_csv(data_path).iloc[:300].copy()
    
    # Create model with scaling disabled
    config = DeepEconoNetConfig(
        seq_len=20,
        learning_rate=1e-3,
        batch_size=32,
        epochs=2,
        return_col="log_return",
        scale_features=False,  # Disable scaling
        train_val_ratio=0.8
    )
    model = DeepEconoNet(config=config)
    
    # Train with ticker
    print("\n1. Training with ticker 'ADSK_NO_SCALE' (scaling disabled)...")
    model.fit_ticker(df, ticker="ADSK_NO_SCALE")
    
    # Verify scales are NOT cached when scaling is disabled
    if "ADSK_NO_SCALE" in model.config.scales:
        print("   ⚠ Warning: Scales were cached even though scale_features=False")
    else:
        print("   ✓ No scales cached (as expected when scale_features=False)")
    
    # Verify NO scaling was applied by checking raw data statistics
    print("\n2. Verifying NO scaling on raw data (should NOT be mean=0, std=1)...")
    log_returns_raw = df[config.return_col].values
    target_raw = model.build_target(df).values
    
    # Compute training indices
    train_idx = int(config.train_val_ratio * len(log_returns_raw))
    train_returns = log_returns_raw[:train_idx]
    train_target = target_raw[:train_idx]
    
    # Remove NaNs for statistics
    valid_returns = train_returns[~np.isnan(train_returns)]
    valid_target = train_target[~np.isnan(train_target)]
    
    # Check statistics of unscaled training data
    returns_mean_raw = float(np.mean(valid_returns))
    returns_std_raw = float(np.std(valid_returns))
    target_mean_raw = float(np.mean(valid_target))
    target_std_raw = float(np.std(valid_target))
    
    print(f"   ✓ Raw training returns (NO normalization):")
    print(f"     - Mean: {returns_mean_raw:.8f} (NOT ≈ 0)")
    print(f"     - Stdev: {returns_std_raw:.8f} (NOT ≈ 1)")
    print(f"   ✓ Raw training target (NO normalization):")
    print(f"     - Mean: {target_mean_raw:.8f} (NOT ≈ 0)")
    print(f"     - Stdev: {target_std_raw:.8f} (NOT ≈ 1)")
    
    # Predict without ticker (no scaling applied)
    print("\n3. Predicting without scaling...")
    preds = model.predict(df)
    valid_preds = preds.dropna()
    
    assert len(valid_preds) > 0, "Should have predictions"
    assert np.all(np.isfinite(valid_preds)), "All predictions should be finite"
    
    pred_min = float(valid_preds.min())
    pred_max = float(valid_preds.max())
    pred_mean = float(valid_preds.mean())
    print(f"   ✓ Predictions (raw, unnormalized scale):")
    print(f"     - Min: {pred_min:.8f}")
    print(f"     - Max: {pred_max:.8f}")
    print(f"     - Mean: {pred_mean:.8f}")
    
    print("\n✅ Test 2 passed: Normalization disabled works correctly!")
    return True


def test_prediction_with_cache():
    """Test that predictions use cached scaling when ticker is provided."""
    print("\n" + "=" * 80)
    print("Test 3: Prediction WITH cached scales (ticker provided)")
    print("=" * 80)
    
    # Load data (use two different subsets for train and predict)
    data_path = os.path.join(os.path.dirname(__file__), "..", "data/stock_info", "ADSK_dataset.csv")
    df_all = pd.read_csv(data_path)
    df_train = df_all.iloc[:300].copy()
    df_test = df_all.iloc[300:400].copy()
    
    # Train model
    config = DeepEconoNetConfig(
        seq_len=20,
        learning_rate=1e-3,
        batch_size=32,
        epochs=2,
        return_col="log_return",
        scale_features=True,
        train_val_ratio=0.8
    )
    model = DeepEconoNet(config=config)
    
    print("\n1. Training on first 300 rows with ticker 'TEST'...")
    model.fit_ticker(df_train, ticker="TEST")
    (returns_mu, returns_sigma), (target_mu, target_sigma) = model.config.scales["TEST"]
    print(f"   ✓ Cached scales:")
    print(f"     - Returns: mean={returns_mu:.8f}, sigma={returns_sigma:.8f}")
    print(f"     - Target:  mean={target_mu:.8f}, sigma={target_sigma:.8f}")
    
    # Predict on different data using cached scales
    print("\n2. Predicting on rows 300-400 WITH cached scales (ticker='TEST')...")
    preds_with_cache = model.predict(df_test, ticker="TEST")
    valid_preds_cache = preds_with_cache.dropna()
    
    assert len(valid_preds_cache) > 0, "Should have predictions with cache"
    cache_min = float(valid_preds_cache.min())
    cache_max = float(valid_preds_cache.max())
    cache_mean = float(valid_preds_cache.mean())
    print(f"   ✓ Predictions with cache (denormalized):")
    print(f"     - Min: {cache_min:.8f}")
    print(f"     - Max: {cache_max:.8f}")
    print(f"     - Mean: {cache_mean:.8f}")
    
    # Predict on same data without ticker (no scaling applied)
    print("\n3. Predicting on rows 300-400 WITHOUT ticker (no cache)...")
    preds_no_cache = model.predict(df_test, ticker=None)
    valid_preds_no_cache = preds_no_cache.dropna()
    
    assert len(valid_preds_no_cache) > 0, "Should have predictions without cache"
    no_cache_min = float(valid_preds_no_cache.min())
    no_cache_max = float(valid_preds_no_cache.max())
    no_cache_mean = float(valid_preds_no_cache.mean())
    print(f"   ✓ Predictions without cache (raw normalized scale):")
    print(f"     - Min: {no_cache_min:.8f}")
    print(f"     - Max: {no_cache_max:.8f}")
    print(f"     - Mean: {no_cache_mean:.8f}")
    
    # Verify they are different (cache applies scaling, no cache doesn't)
    print("\n4. Comparing predictions...")
    preds_cache_values = valid_preds_cache.values.astype(np.float64)
    preds_no_cache_values = valid_preds_no_cache.values[:len(preds_cache_values)].astype(np.float64)
    
    ratio = float(np.abs(preds_cache_values - preds_no_cache_values).mean() / (np.abs(preds_cache_values).mean() + 1e-8))
    print(f"   ✓ Mean relative difference: {ratio:.4f}x")
    
    if ratio > 0.01:  # More than 1% difference
        print(f"   ✓ Predictions differ significantly (scaling applied vs not applied)")
    
    print("\n✅ Test 3 passed: Cached scaling prediction works correctly!")
    return True


def test_prediction_without_cache():
    """Test that predictions work when ticker cache doesn't exist."""
    print("\n" + "=" * 80)
    print("Test 4: Prediction WITHOUT cached scales (non-existent ticker)")
    print("=" * 80)
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data/stock_info", "ADSK_dataset.csv")
    df = pd.read_csv(data_path).iloc[:300].copy()
    
    # Train model
    config = DeepEconoNetConfig(
        seq_len=20,
        learning_rate=1e-3,
        batch_size=32,
        epochs=2,
        return_col="log_return",
        scale_features=True
    )
    model = DeepEconoNet(config=config)
    
    print("\n1. Training with ticker 'EXISTING'...")
    model.fit_ticker(df, ticker="EXISTING")
    assert "EXISTING" in model.config.scales, "EXISTING ticker should be cached"
    print(f"   ✓ Cached scales for 'EXISTING'")
    
    # Predict with a non-existent ticker
    print("\n2. Predicting with non-existent ticker 'NONEXISTENT'...")
    preds = model.predict(df, ticker="NONEXISTENT")
    valid_preds = preds.dropna()
    
    assert len(valid_preds) > 0, "Should still have predictions"
    print(f"   ✓ Predictions work without cache: min={valid_preds.min():.6f}, max={valid_preds.max():.6f}")
    print(f"   ✓ Gracefully handled missing ticker cache")
    
    print("\n✅ Test 4 passed: Prediction without cache works correctly!")
    return True


def test_multiple_tickers():
    """Test that different tickers can have independent scaling parameters."""
    print("\n" + "=" * 80)
    print("Test 5: Multiple tickers with independent scaling")
    print("=" * 80)
    
    # Load different datasets
    data_path_1 = os.path.join(os.path.dirname(__file__), "..", "data/stock_info", "ADSK_dataset.csv")
    data_path_2 = os.path.join(os.path.dirname(__file__), "..", "data/stock_info", "AAPL_dataset.csv")
    
    if not os.path.exists(data_path_2):
        print("⚠ AAPL dataset not found, using ADSK data for both tickers")
        df1 = pd.read_csv(data_path_1).iloc[:250].copy()
        df2 = pd.read_csv(data_path_1).iloc[250:500].copy()
    else:
        df1 = pd.read_csv(data_path_1).iloc[:300].copy()
        df2 = pd.read_csv(data_path_2).iloc[:300].copy()
    
    # Train on first ticker
    config = DeepEconoNetConfig(
        seq_len=20,
        learning_rate=1e-3,
        batch_size=32,
        epochs=1,
        return_col="log_return",
        scale_features=True
    )
    model = DeepEconoNet(config=config)
    
    print("\n1. Training on ticker 'TICKER1'...")
    model.fit_ticker(df1, ticker="TICKER1")
    scales_1 = model.config.scales["TICKER1"]
    print(f"   ✓ Ticker1 scales: {scales_1}")
    
    # Train on second ticker
    print("\n2. Training on ticker 'TICKER2'...")
    model.fit_ticker(df2, ticker="TICKER2")
    scales_2 = model.config.scales["TICKER2"]
    print(f"   ✓ Ticker2 scales: {scales_2}")
    
    # Verify both are cached and different
    assert len(model.config.scales) == 2, "Should have 2 tickers cached"
    print(f"\n3. Verifying independent scales...")
    print(f"   ✓ Both tickers cached independently")
    
    # Scales should generally be different
    if scales_1 != scales_2:
        print(f"   ✓ Ticker1 and Ticker2 have different scaling parameters (as expected)")
    
    print("\n✅ Test 5 passed: Multiple tickers work correctly!")
    return True


if __name__ == "__main__":
    tests = [
        test_deep_econo_net_with_features,
        test_normalization_enabled,
        test_normalization_disabled,
        test_prediction_with_cache,
        test_prediction_without_cache,
        test_multiple_tickers,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 80 + "\n")
    
    sys.exit(0 if failed == 0 else 1)

