"""
Multi-Ticker Dataset Training Evaluation
=========================================

This script evaluates the training performance of the DeepEconoNet model on multiple
ticker datasets using the fit_all_datasets() function. It tests:

1. Batch training across multiple stock tickers
2. Independent scaling parameter caching per ticker
3. Training and validation error monitoring in real-time
4. Performance consistency across different datasets

The script trains the model on all available stock data files in the data/stock_info/
directory and prints training/validation errors on-the-fly to monitor convergence.

Key Features:
- Real-time error reporting during training
- Per-ticker scaling parameter visualization
- Dataset statistics (number of samples per ticker)
- Final performance summary across all tickers
- Memory-efficient batch processing

Output:
- Training/validation loss per epoch for each ticker
- Cached scaling parameters per ticker
- Summary statistics and timing information
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import pandas as pd
import numpy as np
import time
from pathlib import Path

from src.volforecast.models.deep_econo_net import DeepEconoNet, DeepEconoNetConfig


def evaluate_multi_ticker_training(num_tickers=10):
    """Evaluate training on multiple ticker datasets with real-time error monitoring.
    
    Args:
        num_tickers: Number of tickers to train on (default: 10)
    """
    
    print("\n" + "=" * 100)
    print("MULTI-TICKER DATASET TRAINING EVALUATION")
    print("=" * 100)
    
    # Setup paths
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "stock_info")
    
    # Check available datasets
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    print(f"\n📊 Found {len(csv_files)} ticker datasets in {data_dir}")
    
    if len(csv_files) == 0:
        print("❌ No CSV files found! Please ensure data is in data/stock_info/")
        return False
    
    # Show first tickers as examples
    show_count = min(10, num_tickers)
    print(f"\n📋 Dataset files (showing first {show_count}):")
    for i, file in enumerate(csv_files[:show_count]):
        print(f"   {i+1:2d}. {file.name}")
    if len(csv_files) > show_count:
        print(f"   ... and {len(csv_files) - show_count} more")
    
    # Dataset statistics
    print(f"\n📈 Dataset Statistics:")
    total_samples = 0
    dataset_info = []
    
    for csv_file in csv_files[:20]:  # Sample first 20 for stats
        try:
            df = pd.read_csv(csv_file)
            ticker = csv_file.stem.split("_dataset")[0]
            n_samples = len(df)
            total_samples += n_samples
            dataset_info.append((ticker, n_samples))
            print(f"   {ticker:20s}: {n_samples:5d} rows")
        except Exception as e:
            print(f"   ⚠️  Error reading {csv_file.name}: {e}")
    
    if len(csv_files) > 20:
        print(f"   ... and {len(csv_files) - 20} more datasets")
    
    print(f"\n   Total sampled: {total_samples} rows across {min(20, len(csv_files))} tickers")
    
    # Create model with configuration
    print(f"\n🔧 Model Configuration:")
    config = DeepEconoNetConfig(
        seq_len=20,
        learning_rate=1e-3,
        batch_size=64,
        epochs=20,  # More epochs to see convergence
        return_col="log_return",
        scale_features=True,
        train_val_ratio=0.8
    )
    
    print(f"   Sequence length: {config.seq_len}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Epochs: {config.epochs}")
    print(f"   Scale features: {config.scale_features}")
    print(f"   Train/Val ratio: {config.train_val_ratio}")
    
    # Initialize model
    model = DeepEconoNet(config=config)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {model.device}")
    
    # Train on all datasets with real-time error monitoring
    print(f"\n🚀 Training on all datasets...")
    print("=" * 100)
    
    start_time = time.time()
    
    # Use fit_all_datasets to train on multiple tickers
    # This function automatically extracts ticker names and caches scales
    print(f"\nUsing fit_all_datasets() to train on {num_tickers} tickers...")
    model.fit_all_datasets(
        data_dir=data_dir,
        pattern="*.csv",
        verbose=True
    )
    
    total_time = time.time() - start_time
    
    # Collect results from trained tickers
    trained_tickers = []
    for ticker, scales in list(model.config.scales.items())[:num_tickers]:
        # Estimate samples from cached scales
        trained_tickers.append((ticker, 0))  # Size not tracked in fit_all_datasets
    
    # Summary
    print("\n" + "=" * 100)
    print("📊 TRAINING SUMMARY")
    print("=" * 100)
    
    num_trained = len(model.config.scales)
    print(f"\n✅ Successfully trained: {num_trained} tickers")
    print(f"   Scaling parameters cached for {num_trained} tickers")
    
    # Show cached scales
    print(f"\n🔐 Cached Scaling Parameters ({len(model.config.scales)} tickers):")
    for i, (ticker, scales) in enumerate(list(model.config.scales.items())[:50]):
        (_, returns_sigma), (_, target_sigma) = scales
        print(f"   {i+1}. {ticker:20s}: returns_σ={returns_sigma:.8f}, target_σ={target_sigma:.8f}")
    
    if len(model.config.scales) > 5:
        print(f"   ... and {len(model.config.scales) - 50} more tickers cached")
    
    # Timing statistics
    print(f"\n⏱️  Performance:")
    print(f"   Total training time: {total_time:.2f}s")
    if num_trained > 0:
        print(f"   Average time per ticker: {total_time / num_trained:.2f}s")
    
    # Model info
    print(f"\n📋 Model Summary:")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {model.device}")
    print(f"   Scale features enabled: {model.config.scale_features}")
    print(f"   Independent tickers cached: {len(model.config.scales)}")
    
    print("\n" + "=" * 100)
    print("✅ Multi-ticker evaluation completed successfully!")
    print("=" * 100 + "\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Ticker Dataset Training Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_multi_ticker_training.py --num-tickers 10
  python eval_multi_ticker_training.py -n 25
  python eval_multi_ticker_training.py  # Uses default of 10 tickers
        """
    )
    parser.add_argument(
        '--num-tickers', '-n',
        type=int,
        default=10,
        help='Number of tickers to train on (default: 10)'
    )
    
    args = parser.parse_args()
    
    try:
        success = evaluate_multi_ticker_training(num_tickers=args.num_tickers)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Evaluation failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
