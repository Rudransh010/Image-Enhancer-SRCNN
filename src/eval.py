# eval.py - Evaluation script for testing model on validation set
import argparse
import torch
from tqdm import tqdm
from pathlib import Path

from src.model import create_model
from src.data import create_dataloaders
from src.utils import calculate_psnr, calculate_ssim, set_seed


def evaluate(model, val_loader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for lr, hr in tqdm(val_loader, desc="Evaluating"):
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
    set_seed(args.seed)
    
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
    
    avg_psnr, avg_ssim = evaluate(model, val_loader, device)
    
    print(f"\nEvaluation Results:")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fp32/best.pth')
    parser.add_argument('--train_dir', type=str, default='data/DIV2K_train_HR')
    parser.add_argument('--val_dir', type=str, default='data/DIV2K_valid_HR')
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--num_channels', type=int, default=48)
    parser.add_argument('--num_ds_blocks', type=int, default=3)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    main(args)
