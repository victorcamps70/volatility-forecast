"""
Unit tests for DeepEconoNetModel and KalmanGARCH using AAPL dataset.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Handle imports for both package and direct script execution
try:
    from src.volforecast.models.base import BaseConfig
    from src.volforecast.models.garch_model import GARCHConfig
    from src.volforecast.models.deep_econo_net import DeepEconoNetModel, DeepEconoNetConfig, KalmanGARCH
    from src.volforecast.features.builders import FeatureBuilder
except ImportError:
    # If running as a script, add parent directory to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.volforecast.models.base import BaseConfig
    from src.volforecast.models.garch_model import GARCHConfig
    from src.volforecast.models.deep_econo_net import DeepEconoNetModel, DeepEconoNetConfig, KalmanGARCH
    from src.volforecast.features.builders import FeatureBuilder


def test_kalman_garch_denoising():
    """Test KalmanGARCH Kalman Filter denoising and plot results."""
    print("\n" + "=" * 80)
    print("PART 1: Testing KalmanGARCH Kalman Filter Denoising")
    print("=" * 80)

    # Get the path to AAPL dataset
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    aapl_path = os.path.join(data_dir, "AAPL_dataset.csv")

    if not os.path.exists(aapl_path):
        print(f"❌ AAPL dataset not found at {aapl_path}")
        return False

    # 1. Load AAPL data
    print(f"\n1. Loading AAPL dataset from {aapl_path}...")
    df = pd.read_csv(aapl_path, index_col="Date", parse_dates=True)
    print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Use only first 500 rows for testing
    df = df.iloc[:500].copy()
    print(f"   ✓ Using first {len(df)} rows for testing")

    # 2. Create and fit KalmanGARCH model
    print("\n2. Creating and fitting KalmanGARCH model...")
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
        return False

    # 3. Get log returns and apply Kalman filter
    print("\n3. Applying Kalman filter to denoise log returns...")
    log_returns = df["log_return"].values.astype(float)
    
    try:
        denoised_returns = kalman_garch.predict(log_returns.tolist())
        denoised_returns = np.array(denoised_returns)
        print(f"   ✓ Kalman filter applied successfully")
        print(f"   ✓ Original returns range: [{float(log_returns.min()):.6f}, {float(log_returns.max()):.6f}]")
        print(f"   ✓ Denoised returns range: [{float(denoised_returns.min()):.6f}, {float(denoised_returns.max()):.6f}]")
    except Exception as e:
        print(f"   ⚠ Warning: Could not apply Kalman filter: {e}")
        return False

    # 4. Plot comparison
    print("\n4. Creating plot of filtered vs unfiltered signals...")
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot both signals
    ax.plot(log_returns, label="Original (Noisy)", alpha=0.6, linewidth=1.2, color='#FF6B6B', zorder=1)
    ax.plot(denoised_returns, label="Denoised (Kalman Filtered)", alpha=0.8, linewidth=1.5, color='#4ECDC4', zorder=2)
    
    # Styling
    ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Log Returns', fontsize=12, fontweight='bold')
    ax.set_title('Kalman Filter Denoising: Original vs Filtered Log Returns (AAPL)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add some transparency fill between the two curves to highlight the difference
    ax.fill_between(range(len(log_returns)), log_returns, denoised_returns, 
                     alpha=0.1, color='gray', label='Filtering adjustment')
    
    plt.tight_layout()
    
    # Save plot
    reports_dir = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
    os.makedirs(reports_dir, exist_ok=True)
    plot_path = os.path.join(reports_dir, "kalman_filter_test.png")
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"   ✓ Plot saved to {plot_path}")
    plt.show()
    
    # Print statistics
    print("\n5. Kalman Filter Statistics:")
    error = log_returns - denoised_returns
    print(f"   Mean original return: {log_returns.mean():.6f}")
    print(f"   Std original return: {log_returns.std():.6f}")
    print(f"   Mean denoised return: {denoised_returns.mean():.6f}")
    print(f"   Std denoised return: {denoised_returns.std():.6f}")
    print(f"   Mean filtering adjustment: {error.mean():.6f}")
    print(f"   Std filtering adjustment: {error.std():.6f}")
    
    return True


def test_deep_econo_net_model():
    """Test DeepEconoNetModel training and prediction."""
    print("\n" + "=" * 80)
    print("PART 2: Testing DeepEconoNetModel")
    print("=" * 80)

    # Get the path to AAPL dataset
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    aapl_path = os.path.join(data_dir, "AAPL_dataset.csv")

    if not os.path.exists(aapl_path):
        print(f"❌ AAPL dataset not found at {aapl_path}")
        return False

    # 1. Load AAPL data
    print(f"\n1. Loading AAPL dataset...")
    df = pd.read_csv(aapl_path, index_col="Date", parse_dates=True)
    df = df.iloc[:500].copy()
    print(f"   ✓ Loaded {len(df)} rows for testing")

    # 2. Build config and feature builder
    print("\n2. Creating configuration and feature builder...")
    base_cfg = BaseConfig(
        date_col="Date",
        return_col="log_return",
        target_shift=-1,
    )
    # Create feature builder first to determine input size
    builder = FeatureBuilder()
    builder.date_col = "Date"
    builder.return_col = "log_return"
    builder.vix_col = "log_return_VIX"
    
    # Build features on a small sample to get the number of features
    X_sample = builder.build_features(df.head(30))
    num_features = X_sample.shape[1]  # Should be 3 (lag_returns_1, 2, 5) + 2 (lag_vix_1, 2) = 5
    
    config = DeepEconoNetConfig(
        **{k: v for k, v in base_cfg.__dict__.items() 
           if k in DeepEconoNetConfig.__dataclass_fields__},
        input_size=num_features,
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
    print(f"   ✓ Input size (num_features): {num_features}")
    print(f"   ✓ Feature builder created")

    # 3. Instantiate model
    print("\n3. Instantiating DeepEconoNetModel...")
    model = DeepEconoNetModel(config, builder)
    print(f"   ✓ Model created: {model.__class__.__name__}")
    print(f"   ✓ Model device: {config.device}")

    # 4. Fit model
    print("\n4. Fitting model on training data...")
    try:
        model.fit(df)
        print(f"   ✓ Model fitted successfully")
        print(f"   ✓ fitted_ = {model.fitted_}")
    except Exception as e:
        print(f"   ❌ Error during fit: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. Generate predictions
    print("\n5. Generating predictions...")
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
        return False

    # 6. Get model summary
    print("\n6. Model summary:")
    summary = model.summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")

    # 7. Evaluate model
    print("\n7. Evaluating model...")
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

    return True


if __name__ == "__main__":
    print("=" * 80)
    print("Running Kalman Filter test on AAPL dataset...")
    print("=" * 80)

    success = True
    
    # Run Kalman GARCH test
    if not test_kalman_garch_denoising():
        success = False

    print("\n" + "=" * 80)
    if success:
        print("✅ Kalman filter test passed!")
    else:
        print("❌ Kalman filter test failed!")
    print("=" * 80)
    
    sys.exit(0 if success else 1)
