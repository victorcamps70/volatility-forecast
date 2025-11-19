# Training Script Features - training_deep_econo_net.py

## Overview
The `training_deep_econo_net.py` script has been enhanced with model saving and checkpoint/resume capabilities for seamless training workflow management. **Models are automatically saved when training is interrupted (Ctrl+C).**

## New Features

### 1. Model Saving for Inference
After training completes, the model is automatically saved to the `src/volforecast/training/checkpoints/` directory with:
- **Model weights** (state_dict)
- **Model configuration** (all hyperparameters)
- **Per-ticker scaling parameters** (for data normalization)
- **Metadata** (number of tickers, training timestamp)

**File format:** `deepecono_net_<num_tickers>_tickers_<YYYYMMDD_HHMMSS>.pt`

### 2. Interrupt-Based Model Checkpointing
When you press **Ctrl+C** during training:
- ✅ Saves current model state as a checkpoint (for resuming)
- ✅ Saves final model for inference
- ✅ Gracefully exits without data loss
- ✅ Useful for long training runs or experimentation

**Checkpoint file format:** `training_checkpoint_<YYYYMMDD_HHMMSS>_epoch<N>.pt`  
**Interrupted model format:** `interrupted_model_<YYYYMMDD_HHMMSS>.pt`

### 3. Loading Trained Models
Use the `load_inference_model()` function to load saved models for inference:
```python
from src.volforecast.training.training_deep_econo_net import load_inference_model

model, config, ticker_scales = load_inference_model('path/to/checkpoint.pt')
model.eval()  # Set to evaluation mode
```

## Usage Examples

### Train and Save Model (Default)
```bash
python training_deep_econo_net.py --num-tickers 10
```
- Trains on 10 tickers
- Automatically saves final model to `src/volforecast/training/checkpoints/` when done
- Press Ctrl+C to interrupt and save checkpoint

### Interrupt Training (Save on Ctrl+C)
```bash
python training_deep_econo_net.py -n 100
# During training, press Ctrl+C
```
- Model is saved immediately when interrupted
- Two files are created:
  - `training_checkpoint_*_*.pt` - For resuming training
  - `interrupted_model_*.pt` - For inference
- Resume later: `--resume-checkpoint src/volforecast/training/checkpoints/training_checkpoint_*.pt`

### Resume Training from Checkpoint
```bash
python training_deep_econo_net.py --resume-checkpoint src/volforecast/training/checkpoints/training_checkpoint_20241119_143025_epoch10.pt
```
- Resumes from epoch 11
- Continues with original configuration and scales

### List Available Checkpoints
```bash
python training_deep_econo_net.py --list-checkpoints
```
- Shows all saved checkpoints and models
- Displays file sizes and modification times

## New Functions

### `save_training_checkpoint(model, epoch, ticker_scales, metadata=None, checkpoint_name=None)`
Save training state for resuming later.

### `load_training_checkpoint(checkpoint_path, device='cpu')`
Load checkpoint to resume training. Returns model and checkpoint dict with epoch information.

### `save_final_model(model, ticker_scales, model_name=None)`
Save final trained model for inference. Automatically saves to checkpoints directory.

### `load_inference_model(model_path, device='cpu')`
Load trained model for inference. Returns model (in eval mode), config, and ticker scales.

### `get_checkpoint_dir()`
Get checkpoint directory path (creates if not exists).

### `list_available_checkpoints()`
Display all available checkpoints and models.

## Checkpoint Directory Structure
```
volatility-forecast/
└── src/volforecast/training/
    └── checkpoints/
        ├── deepecono_net_010_tickers_20241119_143025.pt      # Final model
        ├── training_checkpoint_20241119_140500_epoch5.pt     # Resume checkpoint
        ├── training_checkpoint_20241119_141000_epoch10.pt    # Resume checkpoint
        └── ...
```

## Key Advantages

✅ **Resume Training** - Stop and resume at any time without losing progress  
✅ **Model Persistence** - Save trained models for inference  
✅ **Experiment Tracking** - Timestamped checkpoints for reproducibility  
✅ **Scale Parameter Preservation** - Per-ticker normalization saved with model  
✅ **Easy Inference** - Single function to load model for predictions  

## Checkpoint Contents

Each checkpoint includes:
- `model_state_dict`: Model weights and parameters
- `model_config`: Dict with all hyperparameters
- `ticker_scales`: Dictionary of per-ticker normalization parameters
- `epoch`: Current epoch (for training checkpoints only)
- `timestamp`: ISO format timestamp
- `pytorch_version`: PyTorch version used
- `metadata`: Additional training info (optional)

## Notes

- Default checkpoint directory: `src/volforecast/training/checkpoints/`
- Models are saved in PyTorch format (.pt files)
- Checkpoint size depends on model complexity (~20-100 MB typically)
- Always use the correct device when loading models (CPU/GPU)
