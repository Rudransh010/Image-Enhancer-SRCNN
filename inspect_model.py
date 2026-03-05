import torch
from pathlib import Path

def inspect_checkpoint(checkpoint_path):
    """Inspect saved model checkpoint"""
    
    print("=" * 60)
    print("Model Checkpoint Inspection")
    print("=" * 60)
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Basic info
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"File size: {Path(checkpoint_path).stat().st_size / (1024*1024):.2f} MB")
    
    # Training info
    print("\n" + "-" * 60)
    print("Training Information:")
    print("-" * 60)
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Train Loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
    print(f"Val PSNR: {checkpoint.get('val_psnr', 'N/A'):.2f} dB")
    print(f"Best PSNR: {checkpoint.get('best_psnr', 'N/A'):.2f} dB")
    
    # Model architecture
    print("\n" + "-" * 60)
    print("Model Architecture:")
    print("-" * 60)
    
    state_dict = checkpoint['model_state_dict']
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {total_params:,}")
    
    # Layer details
    print("\nLayers:")
    for name, param in state_dict.items():
        print(f"  {name:40s} {str(list(param.shape)):20s} {param.numel():>10,} params")
    
    # Optimizer info
    if 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        print("\n" + "-" * 60)
        print("Optimizer Information:")
        print("-" * 60)
        print(f"Optimizer: {opt_state.get('param_groups', [{}])[0].get('name', 'Adam')}")
        if opt_state.get('param_groups'):
            pg = opt_state['param_groups'][0]
            print(f"Learning rate: {pg.get('lr', 'N/A')}")
            print(f"Weight decay: {pg.get('weight_decay', 'N/A')}")
            print(f"Betas: {pg.get('betas', 'N/A')}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Check both best and last checkpoints
    checkpoints = [
        r"R:\Projects\srcnn\checkpoints\fp32\best.pth",
        r"R:\Projects\srcnn\checkpoints\fp32\last.pth"
    ]
    
    for ckpt in checkpoints:
        if Path(ckpt).exists():
            inspect_checkpoint(ckpt)
            print("\n")
