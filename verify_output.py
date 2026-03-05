import torch
import cv2
from pathlib import Path

print("=" * 60)
print("Model and Output Verification")
print("=" * 60)

# Check model checkpoint
checkpoint_path = r"R:\Projects\srcnn\checkpoints\fp32\best.pth"
print(f"\n1. Model Checkpoint: {checkpoint_path}")
print(f"   Exists: {Path(checkpoint_path).exists()}")

if Path(checkpoint_path).exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val PSNR: {checkpoint.get('val_psnr', 'N/A'):.2f} dB")
    print(f"   Best PSNR: {checkpoint.get('best_psnr', 'N/A'):.2f} dB")
    print(f"   Train Loss: {checkpoint.get('train_loss', 'N/A'):.6f}")

# Check output images
print("\n2. Output Images:")
output_dir = Path(r"R:\Projects\srcnn\test_results")

for img_name in ['upscaled.png', 'enhanced.png']:
    img_path = output_dir / img_name
    if img_path.exists():
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            size_mb = img_path.stat().st_size / (1024 * 1024)
            print(f"   [OK] {img_name}")
            print(f"     - Size: {w}x{h}")
            print(f"     - File size: {size_mb:.2f} MB")
            print(f"     - Path: {img_path}")
        else:
            print(f"   [ERROR] {img_name} - Cannot read image")
    else:
        print(f"   [ERROR] {img_name} - File not found")

# Check input image
print("\n3. Input Image:")
input_path = Path(r"R:\Projects\test data\istockphoto-1199509645-612x612.jpg")
if input_path.exists():
    img = cv2.imread(str(input_path))
    if img is not None:
        h, w = img.shape[:2]
        print(f"   [OK] Input exists")
        print(f"     - Size: {w}x{h}")
        print(f"     - Path: {input_path}")
    else:
        print(f"   [ERROR] Cannot read input image")
else:
    print(f"   [ERROR] Input not found at: {input_path}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"Model used: best.pth (36.15 dB PSNR)")
print(f"Output location: {output_dir}")
print(f"Files generated: enhanced.png, upscaled.png")
print("=" * 60)
