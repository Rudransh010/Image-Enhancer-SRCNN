# quantize_qat.py - Quantization-aware training
import argparse
import torch
import torch.nn as nn
import torch.quantization as tq
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os

from src.model import create_model
from src.data import create_dataloaders
from src.utils import calculate_psnr, calculate_ssim, set_seed


def train_qat_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}")
    for lr, hr in pbar:
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=device.type == 'cuda'):
            sr = model(lr)
            loss = criterion(sr, hr)
        
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def validate_qat(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for lr, hr in tqdm(val_loader, desc="Validating QAT"):
            lr, hr = lr.to(device), hr.to(device)
            
            with autocast(enabled=device.type == 'cuda'):
                sr = model(lr)
                loss = criterion(sr, hr)
            
            total_loss += loss.item()
            
            sr_clamp = torch.clamp(sr, 0, 1)
            psnr = calculate_psnr(sr_clamp, hr)
            ssim = calculate_ssim(sr_clamp, hr)
            
            total_psnr += psnr
            total_ssim += ssim
    
    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    
    return avg_loss, avg_psnr, avg_ssim


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
    
    print("Preparing model for QAT...")
    model.qconfig = tq.get_default_qat_qconfig('fbgemm')
    tq.prepare_qat(model, inplace=True)
    
    train_loader, val_loader = create_dataloaders(
        args.train_dir, args.val_dir, args.batch_size, args.patch_size, args.num_workers
    )
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == 'cuda')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    best_psnr = 0
    
    for epoch in range(args.epochs):
        train_loss = train_qat_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_psnr, val_ssim = validate_qat(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"QAT Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
              f"PSNR={val_psnr:.2f}dB, SSIM={val_ssim:.4f}")
        
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_int8_qat.pth'))
            print(f"Saved best QAT model with PSNR={best_psnr:.2f}dB")
    
    print(f"QAT complete. Best PSNR: {best_psnr:.2f}dB")
    
    print("Converting to quantized model...")
    tq.convert(model, inplace=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_int8_qat_converted.pth'))
    print(f"Quantized model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fp32/best.pth')
    parser.add_argument('--train_dir', type=str, default='data/DIV2K_train_HR')
    parser.add_argument('--val_dir', type=str, default='data/DIV2K_valid_HR')
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
