# tests/test_model_forward.py - Unit tests for model forward pass
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import create_model, count_parameters


def test_model_forward():
    print("Testing model forward pass...")
    model = create_model(num_channels=48, num_ds_blocks=3, num_res_blocks=2)
    model.eval()
    
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.shape == input_tensor.shape, f"Output shape {output.shape} != input shape {input_tensor.shape}"
    assert output.dtype == torch.float32, f"Output dtype {output.dtype} != float32"
    
    print(f"[PASS] Forward pass successful")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")


def test_parameter_count():
    print("\nTesting parameter count...")
    model = create_model(num_channels=48, num_ds_blocks=3, num_res_blocks=2)
    
    params = count_parameters(model)
    print(f"[PASS] Parameter count: {params:,}")
    assert params <= 500_000, f"Parameter count {params} exceeds 500k limit!"
    print(f"[PASS] Parameter count within budget (<= 500k)")


def test_different_input_sizes():
    print("\nTesting different input sizes...")
    model = create_model(num_channels=48, num_ds_blocks=3, num_res_blocks=2)
    model.eval()
    
    test_sizes = [(1, 3, 128, 128), (1, 3, 256, 256), (2, 3, 512, 512)]
    
    with torch.no_grad():
        for size in test_sizes:
            x = torch.randn(size)
            y = model(x)
            assert y.shape == x.shape, f"Shape mismatch for input {size}"
            print(f"[PASS] Input {size} -> Output {y.shape}")


def test_model_torchscript():
    print("\nTesting TorchScript compatibility...")
    model = create_model(num_channels=48, num_ds_blocks=3, num_res_blocks=2)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256)
    
    try:
        traced = torch.jit.trace(model, dummy_input)
        print("[PASS] TorchScript tracing successful")
        
        with torch.no_grad():
            ts_output = traced(dummy_input)
        print(f"[PASS] TorchScript inference successful: {ts_output.shape}")
    except Exception as e:
        print(f"[FAIL] TorchScript failed: {e}")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("SRCNN Model Unit Tests")
    print("=" * 60)
    
    test_model_forward()
    test_parameter_count()
    test_different_input_sizes()
    test_model_torchscript()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
