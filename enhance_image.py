import torch
import cv2
import numpy as np
from pathlib import Path
from src.model import create_model

def enhance_image(input_path, output_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model(48, 3, 2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    img = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create LR version (downscale then upscale)
    h, w = img_rgb.shape[:2]
    lr_img = cv2.resize(img_rgb, (w//2, h//2), interpolation=cv2.INTER_CUBIC)
    lr_upscaled = cv2.resize(lr_img, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Convert to tensor
    lr_tensor = torch.from_numpy(lr_upscaled.transpose(2, 0, 1) / 255.0).float().unsqueeze(0).to(device)
    
    # Enhance
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
    
    # Convert back to image
    sr_img = (sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    sr_bgr = cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR)
    
    # Save result
    cv2.imwrite(output_path, sr_bgr)
    print(f"Enhanced image saved to: {output_path}")

if __name__ == "__main__":
    input_path = r"R:\Projects\test data\istockphoto-1199509645-612x612.jpg"
    output_path = r"R:\Projects\srcnn\enhanced_image.jpg"
    model_path = r"R:\Projects\srcnn\checkpoints\fp32\best.pth"
    
    enhance_image(input_path, output_path, model_path)