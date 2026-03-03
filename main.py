# main.py - Entry point for all commands
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [args]")
        print("\nAvailable commands:")
        print("  train      - Train the model")
        print("  eval       - Evaluate the model")
        print("  export     - Export model to TorchScript/ONNX")
        print("  quantize-ptq - Post-training quantization")
        print("  quantize-qat - Quantization-aware training")
        print("  benchmark  - Benchmark model performance")
        print("  test       - Run unit tests")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == "train":
        from train import main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_dir', type=str, default='dataset/DIV2K_train_HR/DIV2K_train_HR')
        parser.add_argument('--val_dir', type=str, default='dataset/DIV2K_train_HR/DIV2K_train_HR')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/fp32')
        parser.add_argument('--epochs', type=int, default=150)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--patch_size', type=int, default=96)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--num_channels', type=int, default=48)
        parser.add_argument('--num_ds_blocks', type=int, default=3)
        parser.add_argument('--num_res_blocks', type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--resume', type=str, default='')
        parser.add_argument('--compile', action='store_true')
        args = parser.parse_args()
        main(args)
    
    elif command == "eval":
        from eval import main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', type=str, default='checkpoints/fp32/best.pth')
        parser.add_argument('--train_dir', type=str, default='dataset/DIV2K_train_HR/DIV2K_train_HR')
        parser.add_argument('--val_dir', type=str, default='dataset/DIV2K_train_HR/DIV2K_train_HR')
        parser.add_argument('--patch_size', type=int, default=96)
        parser.add_argument('--num_channels', type=int, default=48)
        parser.add_argument('--num_ds_blocks', type=int, default=3)
        parser.add_argument('--num_res_blocks', type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--seed', type=int, default=42)
        args = parser.parse_args()
        main(args)
    
    elif command == "export":
        from export import main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', type=str, default='checkpoints/fp32/best.pth')
        parser.add_argument('--output_dir', type=str, default='artifacts/models')
        parser.add_argument('--num_channels', type=int, default=48)
        parser.add_argument('--num_ds_blocks', type=int, default=3)
        parser.add_argument('--num_res_blocks', type=int, default=2)
        parser.add_argument('--input_height', type=int, default=256)
        parser.add_argument('--input_width', type=int, default=256)
        parser.add_argument('--opset_version', type=int, default=14)
        args = parser.parse_args()
        main(args)
    
    elif command == "quantize-ptq":
        from quantize_ptq import main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', type=str, default='checkpoints/fp32/best.pth')
        parser.add_argument('--train_dir', type=str, default='dataset/DIV2K_train_HR/DIV2K_train_HR')
        parser.add_argument('--val_dir', type=str, default='dataset/DIV2K_train_HR/DIV2K_train_HR')
        parser.add_argument('--output_dir', type=str, default='artifacts/models')
        parser.add_argument('--patch_size', type=int, default=96)
        parser.add_argument('--num_channels', type=int, default=48)
        parser.add_argument('--num_ds_blocks', type=int, default=3)
        parser.add_argument('--num_res_blocks', type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=4)
        args = parser.parse_args()
        main(args)
    
    elif command == "quantize-qat":
        from quantize_qat import main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint', type=str, default='checkpoints/fp32/best.pth')
        parser.add_argument('--train_dir', type=str, default='dataset/DIV2K_train_HR/DIV2K_train_HR')
        parser.add_argument('--val_dir', type=str, default='dataset/DIV2K_train_HR/DIV2K_train_HR')
        parser.add_argument('--output_dir', type=str, default='artifacts/models')
        parser.add_argument('--epochs', type=int, default=20)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--patch_size', type=int, default=96)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--num_channels', type=int, default=48)
        parser.add_argument('--num_ds_blocks', type=int, default=3)
        parser.add_argument('--num_res_blocks', type=int, default=2)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--seed', type=int, default=42)
        args = parser.parse_args()
        main(args)
    
    elif command == "benchmark":
        from benchmark import main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--torchscript', type=str, default='artifacts/models/model_fp32.pt')
        parser.add_argument('--onnx', type=str, default='artifacts/models/model_fp32.onnx')
        parser.add_argument('--num_runs', type=int, default=100)
        parser.add_argument('--num_threads', type=int, default=1)
        parser.add_argument('--input_height', type=int, default=256)
        parser.add_argument('--input_width', type=int, default=256)
        parser.add_argument('--num_channels', type=int, default=48)
        parser.add_argument('--num_ds_blocks', type=int, default=3)
        parser.add_argument('--num_res_blocks', type=int, default=2)
        args = parser.parse_args()
        main(args)
    
    elif command == "test":
        import subprocess
        subprocess.run([sys.executable, "tests/test_model_forward.py"])
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
