# quantize_ptq.py - Post-training quantization
import argparse
import torch
import torch.quantization as tq
import os
from src.model import create_model
from src.data import create_dataloaders
from src.utils import calculate_psnr, calculate_ssim


def calibrate_model(model, calibration_loader, device):
    model.eval()
    with torch.no_grad():
        for lr, _ in calibration_loader:
            lr = lr.to(device)
            model(lr)


def quantize_ptq(model, calibration_loader, device, output_path):
    print("Preparing model for quantization...")
    model.eval()
    
    model.qconfig = tq.get_default_qconfig('fbgemm')
    tq.prepare(model, inplace=True)
    
    print("Calibrating model...")
    calibrate_model(model, calibration_loader, device)
    
    print("Converting to quantized model...")
    tq.convert(model, inplace=True)
    
    print(f"Saving quantized model to {output_path}")
    torch.save(model.state_dict(), output_path)
    
    return model


def evaluate_quantized(model, val_loader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for lr, hr in val_loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            sr = torch.clamp(sr, 0, 1)
            
            psnr = calculate_psnr(sr, hr)
            ssim = calculate_ssim(sr, hr)
            
            total_psnr += psnr
            total_ssim += ssim
    
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    
    return avg_psnr, avg_ssim


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_model(args.num_channels, args.num_ds_blocks, args.num_res_blocks)
    
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    
    _, val_loader = create_dataloaders(
        args.train_dir, args.val_dir, batch_size=1, 
        patch_size=args.patch_size, num_workers=args.num_workers
    )
    
    calibration_loader = val_loader
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    quantized_model = quantize_ptq(model, calibration_loader, device, 
                                   os.path.join(args.output_dir, 'model_int8_ptq.pth'))
    
    print("\nEvaluating quantized model...")
    avg_psnr, avg_ssim = evaluate_quantized(quantized_model, val_loader, device)
    
    print(f"Quantized Model PSNR: {avg_psnr:.2f} dB")
    print(f"Quantized Model SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fp32/best.pth')
    parser.add_argument('--train_dir', type=str, default='data/DIV2K_train_HR')
    parser.add_argument('--val_dir', type=str, default='data/DIV2K_valid_HR')
    parser.add_argument('--output_dir', type=str, default='artifacts/models')
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--num_channels', type=int, default=48)
    parser.add_argument('--num_ds_blocks', type=int, default=3)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    main(args)
