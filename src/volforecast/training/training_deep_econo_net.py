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
import torch
import json
import signal
from pathlib import Path
from datetime import datetime

from src.volforecast.models.deep_econo_net import DeepEconoNet, DeepEconoNetConfig


def get_checkpoint_dir():
    """Get or create checkpoint directory inside training directory."""
    checkpoint_dir = os.path.join(
        os.path.dirname(__file__), 'checkpoints'
    )
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_training_checkpoint(model, epoch, ticker_scales, metadata=None, checkpoint_name=None):
    """Save model checkpoint with training state for resuming later.
    
    Args:
        model: DeepEconoNet model instance
        epoch: Current epoch number
        ticker_scales: Dictionary of per-ticker scaling parameters
        metadata: Optional metadata dict with training info
        checkpoint_name: Optional custom checkpoint name
    
    Returns:
        checkpoint_path: Path to saved checkpoint
    """
    checkpoint_dir = get_checkpoint_dir()
    
    if checkpoint_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"training_checkpoint_{timestamp}_epoch{epoch}.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_config': {
            'seq_len': model.config.seq_len,
            'learning_rate': model.config.learning_rate,
            'batch_size': model.config.batch_size,
            'epochs': model.config.epochs,
            'return_col': model.config.return_col,
            'scale_features': model.config.scale_features,
            'train_val_ratio': model.config.train_val_ratio,
        },
        'ticker_scales': ticker_scales,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }
    
    if metadata:
        checkpoint['metadata'] = metadata
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_training_checkpoint(checkpoint_path, device='cpu'):
    """Load training checkpoint to resume training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict with training state
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Recreate model with saved config
    config_dict = checkpoint['model_config']
    config = DeepEconoNetConfig(**config_dict)
    model = DeepEconoNet(config=config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore ticker scales
    model.config.scales = checkpoint['ticker_scales']
    
    return model, checkpoint


def save_final_model(model, ticker_scales, model_name=None):
    """Save final trained model for inference.
    
    Args:
        model: Trained DeepEconoNet model
        ticker_scales: Dictionary of per-ticker scaling parameters
        model_name: Optional custom model name
    
    Returns:
        model_path: Path to saved model
    """
    checkpoint_dir = get_checkpoint_dir()
    
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        num_tickers = len(ticker_scales)
        model_name = f"deepecono_net_{num_tickers}_tickers_{timestamp}.pt"
    
    model_path = os.path.join(checkpoint_dir, model_name)
    
    model_data = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'seq_len': model.config.seq_len,
            'learning_rate': model.config.learning_rate,
            'batch_size': model.config.batch_size,
            'epochs': model.config.epochs,
            'return_col': model.config.return_col,
            'scale_features': model.config.scale_features,
            'train_val_ratio': model.config.train_val_ratio,
        },
        'ticker_scales': ticker_scales,
        'num_tickers': len(ticker_scales),
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
    }
    
    torch.save(model_data, model_path)
    return model_path


def load_inference_model(model_path, device='cpu'):
    """Load final model for inference.
    
    Args:
        model_path: Path to saved model
        device: Device to load to
    
    Returns:
        model: Loaded model ready for inference
        config: Model configuration
        ticker_scales: Dictionary of ticker scales
    """
    model_data = torch.load(model_path, map_location=device)
    
    # Recreate model
    config_dict = model_data['model_config']
    config = DeepEconoNetConfig(**config_dict)
    model = DeepEconoNet(config=config)
    
    # Load weights
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    return model, config, model_data['ticker_scales']


def list_available_checkpoints():
    """List all available checkpoints and models."""
    checkpoint_dir = get_checkpoint_dir()
    checkpoints = sorted(Path(checkpoint_dir).glob("*.pt"))
    
    print(f"\n📂 Available checkpoints in {checkpoint_dir}:")
    if not checkpoints:
        print("   No checkpoints found.")
        return []
    
    checkpoint_info = []
    for i, cp_path in enumerate(checkpoints):
        size_mb = cp_path.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(cp_path.stat().st_mtime)
        print(f"   {i+1}. {cp_path.name}")
        print(f"      Size: {size_mb:.2f} MB, Modified: {mtime}")
        checkpoint_info.append(cp_path)
    
    return checkpoints


def evaluate_multi_ticker_training(num_tickers=10, resume_checkpoint=None, save_interval=None, config=None):
    """Evaluate training on multiple ticker datasets with real-time error monitoring.
    
    Args:
        num_tickers: Number of tickers to train on (default: 10)
        resume_checkpoint: Path to checkpoint to resume training from
        save_interval: DEPRECATED - now saves automatically on interrupt (Ctrl+C)
        config: Optional DeepEconoNetConfig to override defaults
    """
    
    # Global variables for signal handling
    global model_for_signal, final_model_path_holder, checkpoint_path_holder
    model_for_signal = None
    final_model_path_holder = None
    checkpoint_path_holder = None
    
    def signal_handler(signum, frame):
        """Handle interrupt signals (Ctrl+C) by saving the model."""
        print("\n\n" + "=" * 100)
        print("⚠️  TRAINING INTERRUPTED - SAVING MODEL...")
        print("=" * 100)
        
        if model_for_signal is not None:
            try:
                print(f"\n💾 Saving interrupted model checkpoint...")
                interrupt_checkpoint_path = save_training_checkpoint(
                    model_for_signal,
                    epoch=0,  # We don't track epoch in fit_all_datasets
                    ticker_scales=dict(model_for_signal.config.scales),
                    metadata={
                        'num_tickers': len(model_for_signal.config.scales), 
                        'interrupted': True,
                        'interrupt_time': datetime.now().isoformat()
                    }
                )
                print(f"   ✅ Checkpoint saved: {os.path.basename(interrupt_checkpoint_path)}")
                print(f"   📂 Location: {interrupt_checkpoint_path}")
                checkpoint_path_holder = interrupt_checkpoint_path
            except Exception as e:
                print(f"   ❌ Failed to save checkpoint: {e}")
            
            try:
                print(f"\n💾 Saving interrupted model for inference...")
                interrupt_model_path = save_final_model(
                    model_for_signal,
                    ticker_scales=dict(model_for_signal.config.scales),
                    model_name=f"interrupted_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                )
                print(f"   ✅ Model saved: {os.path.basename(interrupt_model_path)}")
                print(f"   📦 File size: {os.path.getsize(interrupt_model_path) / (1024*1024):.2f} MB")
                final_model_path_holder = interrupt_model_path
            except Exception as e:
                print(f"   ❌ Failed to save model: {e}")
        
        print("\n" + "=" * 100)
        print("✅ Interrupt handling complete. You can resume with --resume-checkpoint.")
        print("=" * 100 + "\n")
        sys.exit(0)
    
    # Register signal handler for SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n" + "=" * 100)
    print("MULTI-TICKER DATASET TRAINING EVALUATION")
    print("=" * 100)
    print("💡 Tip: Press Ctrl+C to safely save model and interrupt training")
    print("=" * 100)
    
    # Handle resume from checkpoint
    start_epoch = 0
    if resume_checkpoint:
        print(f"\n🔄 Resuming training from checkpoint: {resume_checkpoint}")
        if not os.path.exists(resume_checkpoint):
            print(f"❌ Checkpoint file not found: {resume_checkpoint}")
            return False
        
        model, checkpoint = load_training_checkpoint(resume_checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print(f"   Resuming from epoch {start_epoch}")
        print(f"   Checkpoint created: {checkpoint.get('timestamp', 'unknown')}")
    else:
        model = None
    
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
    
    # Create model with configuration (if not resuming)
    if model is None:
        print(f"\n🔧 Model Configuration:")
        if config is None:
            config = DeepEconoNetConfig(
                seq_len=20,
                learning_rate=1e-3,
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
    else:
        print(f"\n🔧 Resumed Model Configuration:")
        config = model.config
        print(f"   Sequence length: {config.seq_len}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Epochs: {config.epochs}")
        print(f"   Scale features: {config.scale_features}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {model.device}")
    
    # Train on all datasets with real-time error monitoring
    print(f"\n🚀 Training on all datasets...")
    print("=" * 100)
    
    start_time = time.time()
    
    # Set model globally for signal handler to access
    model_for_signal = model
    
    try:
        # Use fit_all_datasets to train on multiple tickers
        # This function automatically extracts ticker names and caches scales
        print(f"\nUsing fit_all_datasets() to train on {num_tickers} tickers...")
        model.fit_all_datasets(
            data_dir=data_dir,
            pattern="*.csv",
            verbose=True
        )
    except KeyboardInterrupt:
        # Signal handler will catch Ctrl+C and save
        raise
    
    total_time = time.time() - start_time
    
    # Save final model for inference after successful training
    print(f"\n💾 Saving final model for inference...")
    final_model_path = save_final_model(
        model,
        ticker_scales=dict(model.config.scales)
    )
    print(f"   ✅ Model saved: {os.path.basename(final_model_path)}")
    print(f"   📦 File size: {os.path.getsize(final_model_path) / (1024*1024):.2f} MB")
    
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
    print(f"   Device: {model.device}")
    print(f"   Total training time: {total_time:.2f}s")
    if num_trained > 0:
        print(f"   Average time per ticker: {total_time / num_trained:.2f}s")
    
    # Model info
    print(f"\n📋 Model Summary:")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {model.device}")
    print(f"   Scale features enabled: {model.config.scale_features}")
    print(f"   Independent tickers cached: {len(model.config.scales)}")
    
    # Model saving info
    print(f"\n💾 Model Files:")
    print(f"   Final model: {os.path.basename(final_model_path)}")
    print(f"   Location: {get_checkpoint_dir()}")
    
    print("\n" + "=" * 100)
    print("✅ Multi-ticker evaluation completed successfully!")
    print("=" * 100)
    print(f"\n📌 To load model for inference, use: load_inference_model('{os.path.basename(final_model_path)}')")
    print("=" * 100 + "\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Ticker Dataset Training Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training_deep_econo_net.py --num-tickers 10
  python training_deep_econo_net.py -n 25
  python training_deep_econo_net.py --resume-checkpoint src/volforecast/training/checkpoints/training_checkpoint_*.pt
  python training_deep_econo_net.py --list-checkpoints
  
Interrupt Handling:
  Press Ctrl+C during training to safely save the current model state.
  The model will be saved as both a checkpoint (for resuming) and a final model.
        """
    )
    parser.add_argument(
        '--num-tickers', '-n',
        type=int,
        default=10,
        help='Number of tickers to train on (default: 10)'
    )
    parser.add_argument(
        '--resume-checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--list-checkpoints',
        action='store_true',
        help='List all available checkpoints and exit'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cuda',
        help='Device to train on (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Handle list checkpoints
    if args.list_checkpoints:
        list_available_checkpoints()
        sys.exit(0)
    
    try:
        train_config = DeepEconoNetConfig(device=args.device, return_col="log_return")
        success = evaluate_multi_ticker_training(
            num_tickers=args.num_tickers,
            resume_checkpoint=args.resume_checkpoint,
            config=train_config
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Evaluation failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
