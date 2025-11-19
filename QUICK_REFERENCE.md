# Training & Inference Quick Reference

## Quick Start

### Train and Save Model
```bash
python src/volforecast/training/training_deep_econo_net.py -n 10
```
✅ Trains on 10 tickers and saves model to `src/volforecast/training/checkpoints/` directory

### Interrupt Training (Save on Ctrl+C)
```bash
python src/volforecast/training/training_deep_econo_net.py -n 50
# Press Ctrl+C during training
```
✅ Model is automatically saved when interrupted  
✅ Two files created: checkpoint (for resuming) + model (for inference)

### Resume Training After Interruption
```bash
python src/volforecast/training/training_deep_econo_net.py --resume-checkpoint src/volforecast/training/checkpoints/training_checkpoint_20241119_143025_epoch10.pt
```
✅ Continues from epoch 11 with saved state

### Check Available Saved Models
```bash
python src/volforecast/training/training_deep_econo_net.py --list-checkpoints
```
✅ Shows all checkpoints and final models

## Python API Usage

### Load Trained Model for Inference
```python
from src.volforecast.training.training_deep_econo_net import load_inference_model
import torch

# Load model
model, config, ticker_scales = load_inference_model('src/volforecast/training/checkpoints/deepecono_net_010_tickers_20241119_143025.pt')

# Make predictions
model.eval()
with torch.no_grad():
    # your prediction code here
    pass
```

### Resume Training Programmatically
```python
from src.volforecast.training.training_deep_econo_net import evaluate_multi_ticker_training

success = evaluate_multi_ticker_training(
    num_tickers=50,
    resume_checkpoint='src/volforecast/training/checkpoints/training_checkpoint_20241119_140500_epoch5.pt'
)
```

## Checkpoint Location
```
src/volforecast/training/checkpoints/
├── deepecono_net_010_tickers_20241119_143025.pt          # Final model (successful training)
├── training_checkpoint_20241119_140500_epoch5.pt         # Checkpoint (for resuming)
├── interrupted_model_20241119_140600.pt                  # Interrupted model
└── ...
```

## Key Features

| Feature | Usage |
|---------|-------|
| **Save After Training** | Automatic - saves final model with scales |
| **Save on Interrupt** | Press Ctrl+C - saves checkpoint + model |
| **Resume Training** | `--resume-checkpoint path/to/checkpoint.pt` |
| **Load for Inference** | `load_inference_model(path)` |
| **List Models** | `--list-checkpoints` |

## Output Interpretation

```
⚠️  TRAINING INTERRUPTED - SAVING MODEL...
======================================
💾 Saving interrupted model checkpoint...
   ✅ Checkpoint saved: training_checkpoint_20241119_143025_epoch0.pt
   📂 Location: /path/to/checkpoints/

💾 Saving interrupted model for inference...
   ✅ Model saved: interrupted_model_20241119_143025.pt
   📦 File size: 45.32 MB

✅ Interrupt handling complete. You can resume with --resume-checkpoint.
```

The saved models contain:
- ✅ Trained weights (for predictions)
- ✅ Configuration (seq_len, lr, batch_size, epochs, etc.)
- ✅ Per-ticker scales (normalization parameters)
- ✅ Metadata (timestamp, PyTorch version, interrupt info)
