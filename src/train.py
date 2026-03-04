# train.py - Training script with AMP, checkpointing, and validation
import argparse
import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path

from src.model import create_model, count_parameters
from src.data import create_dataloaders
from src.utils import calculate_psnr, calculate_ssim, set_seed, CSVLogger


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for lr, hr, _ in pbar:
        lr, hr = lr.to(device), hr.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=device.type == 'cuda'):
            sr = model(lr)
            loss = criterion(sr, hr)
        
        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    
    with torch.no_grad():
        for lr, hr, _ in tqdm(val_loader, desc="Validating"):
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
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("artifacts/logs", exist_ok=True)
    
    model = create_model(args.num_channels, args.num_ds_blocks, args.num_res_blocks)
    model = model.to(device)
    
    if hasattr(torch, 'compile') and args.compile:
        print("Using torch.compile()")
        model = torch.compile(model)
    
    train_loader, val_loader = create_dataloaders(
        args.train_dir, args.val_dir, args.batch_size, args.patch_size, args.num_workers
    )
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == 'cuda')
    
    start_epoch = 0
    best_psnr = 0
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0)
    
    with CSVLogger("artifacts/logs/training_log.csv") as logger:
        logger.write_header(['epoch', 'train_loss', 'val_loss', 'val_psnr', 'val_ssim', 'lr'])
        
        for epoch in range(start_epoch, args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
            val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"PSNR={val_psnr:.2f}dB, SSIM={val_ssim:.4f}, LR={current_lr:.6f}")
            
            logger.write_row({
                'epoch': epoch,
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}',
                'val_psnr': f'{val_psnr:.4f}',
                'val_ssim': f'{val_ssim:.6f}',
                'lr': f'{current_lr:.8f}'
            })
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_psnr': val_psnr,
                'best_psnr': best_psnr
            }
            
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'last.pth'))
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best.pth'))
                print(f"Saved best model with PSNR={best_psnr:.2f}dB")
    
    print(f"Training complete. Best PSNR: {best_psnr:.2f}dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='data/DIV2K_train_HR')
    parser.add_argument('--val_dir', type=str, default='data/DIV2K_valid_HR')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/fp32')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=1e-4)
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
