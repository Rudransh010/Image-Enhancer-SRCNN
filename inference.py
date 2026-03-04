# inference.py - Enhance blurred images using trained SRCNN model
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, 'src')
from model import create_model

def enhance_image(image_path, checkpoint_path='checkpoints/fp32/best.pth', output_path='enhanced.png'):
    """Enhance a blurred image using SRCNN model"""
    
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Bicubic upscale 2x
    upscaled = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    
    # Normalize
    img_tensor = torch.from_numpy(upscaled).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        enhanced = model(img_tensor)
    
    # Denormalize and convert
    enhanced = torch.clamp(enhanced, 0, 1)
    enhanced = (enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    
    # Save
    cv2.imwrite(output_path, enhanced)
    print(f"Enhanced image saved: {output_path}")
    
    return enhanced

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Input blurred image path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/fp32/best.pth')
    parser.add_argument('--output', type=str, default='enhanced.png')
    
    args = parser.parse_args()
    enhance_image(args.image, args.checkpoint, args.output)
