import sys
import torch
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, 'src')
from model import create_model

def test_image(image_path, checkpoint_path='checkpoints/fp32/best.pth'):
    """Test SRCNN on a single image"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = create_model()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    print(f"Original image size: {w}x{h}")
    
    # Bicubic upscale 2x
    upscaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    print(f"Upscaled image size: {upscaled.shape[1]}x{upscaled.shape[0]}")
    
    # Normalize
    img_tensor = torch.from_numpy(upscaled).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        enhanced = model(img_tensor)
    
    # Denormalize and convert
    enhanced = torch.clamp(enhanced, 0, 1)
    enhanced_np = (enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    enhanced_np = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
    
    # Save results
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)
    
    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_dir / 'upscaled.png'), upscaled_bgr)
    cv2.imwrite(str(output_dir / 'enhanced.png'), enhanced_np)
    
    print(f"Results saved to {output_dir}/")
    print(f"  - upscaled.png: Bicubic upscaled (baseline)")
    print(f"  - enhanced.png: SRCNN enhanced")

if __name__ == "__main__":
    test_image(r'R:\Projects\test data\istockphoto-1199509645-612x612.jpg')
