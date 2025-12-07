#!/usr/bin/env python3
"""Quick test to verify volatility integration in DeepEconoNet"""

import numpy as np
import sys
sys.path.insert(0, '/home/said-tellez/Desktop/Centrale/Ei3/STASC/volatility-forecast')

from src.volforecast.models.deep_econo_net import DeepEconoNet, DeepEconoNetConfig

# Create minimal config
config = DeepEconoNetConfig()
model = DeepEconoNet(config)

# Test 1: Volatility computation
print("Test 1: Volatility computation")
X_test = np.random.randn(4, 20, 1).astype(np.float32)  # (batch=4, seq_len=20, features=1)
vol = model._compute_current_logRV(X_test)
print(f"  Input shape: {X_test.shape}")
print(f"  Output shape: {vol.shape}")
print(f"  Expected: (4, 1), Got: {vol.shape}")
assert vol.shape == (4, 1), "Volatility shape mismatch!"
print("  ✓ Volatility computation passed")

# Test 2: Forward pass with volatility
print("\nTest 2: Forward pass with volatility")
import torch
X_tensor = torch.FloatTensor(X_test).to(model.device)
vol_tensor = torch.FloatTensor(vol).to(model.device)
try:
    output = model.forward(X_tensor, vol_tensor)
    print(f"  Input shape: {X_tensor.shape}")
    print(f"  Volatility shape: {vol_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (4, 1), Got: {output.shape}")
    assert output.shape == (4, 1), "Forward pass shape mismatch!"
    print("  ✓ Forward pass passed")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    sys.exit(1)

# Test 3: predict_array method
print("\nTest 3: predict_array method")
try:
    preds = model.predict_array(X_test)
    print(f"  Input shape: {X_test.shape}")
    print(f"  Output shape: {preds.shape}")
    print(f"  Expected: (4, 1), Got: {preds.shape}")
    assert preds.shape == (4, 1), "predict_array shape mismatch!"
    print("  ✓ predict_array passed")
except Exception as e:
    print(f"  ✗ predict_array failed: {e}")
    sys.exit(1)

print("\n✅ All volatility integration tests passed!")
