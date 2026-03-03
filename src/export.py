# export.py - Export model to TorchScript and ONNX
import argparse
import torch
import onnx
import os
from src.model import create_model


def export_torchscript(model, output_path, input_size=(1, 3, 256, 256)):
    print(f"Exporting TorchScript to {output_path}")
    model.eval()
    dummy_input = torch.randn(input_size)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    print(f"TorchScript saved: {output_path}")


def export_onnx(model, output_path, input_size=(1, 3, 256, 256), opset_version=14):
    print(f"Exporting ONNX to {output_path}")
    model.eval()
    dummy_input = torch.randn(input_size)
    
    torch.onnx.export(model, dummy_input, output_path, export_params=True, opset_version=opset_version,
                     do_constant_folding=True, input_names=['input'], output_names=['output'],
                     dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},
                                  'output': {0: 'batch', 2: 'height', 3: 'width'}})
    
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX saved and verified: {output_path}")


def verify_exports(torchscript_path, onnx_path, input_size=(1, 3, 256, 256)):
    print("\nVerifying exported models...")
    dummy_input = torch.randn(input_size)
    
    ts_model = torch.jit.load(torchscript_path)
    ts_output = ts_model(dummy_input)
    print(f"TorchScript output shape: {ts_output.shape}")
    
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        print(f"ONNX output shape: {ort_output.shape}")
        
        diff = torch.abs(ts_output - torch.from_numpy(ort_output)).max().item()
        print(f"Max difference TorchScript vs ONNX: {diff:.6f}")
    except ImportError:
        print("onnxruntime not installed, skipping ONNX verification")


def main(args):
    device = torch.device('cpu')
    model = create_model(args.num_channels, args.num_ds_blocks, args.num_res_blocks)
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    input_size = (1, 3, args.input_height, args.input_width)
    
    ts_path = os.path.join(args.output_dir, 'model_fp32.pt')
    export_torchscript(model, ts_path, input_size)
    
    onnx_path = os.path.join(args.output_dir, 'model_fp32.onnx')
    export_onnx(model, onnx_path, input_size, args.opset_version)
    
    verify_exports(ts_path, onnx_path, input_size)
    
    print(f"\nExport complete. Models saved to {args.output_dir}")


if __name__ == "__main__":
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
