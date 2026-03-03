# data.py - Dataset and dataloader for DIV2K super-resolution training
import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class SRDataset(Dataset):
    def __init__(self, hr_dir, scale=2, patch_size=96, augment=True):
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        
        self.hr_images = sorted([f for f in self.hr_dir.glob("*.png")])
        if not self.hr_images:
            self.hr_images = sorted([f for f in self.hr_dir.glob("*.jpg")])
        
        print(f"Found {len(self.hr_images)} images in {hr_dir}")
        
    def __len__(self):
        return len(self.hr_images)
    
    def __getitem__(self, idx):
        hr_img = cv2.imread(str(self.hr_images[idx]))
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        
        if self.augment:
            hr_patch = self._extract_patch(hr_img, self.patch_size)
            hr_patch = self._augment(hr_patch)
        else:
            hr_patch = hr_img
        
        h, w = hr_patch.shape[:2]
        lr_h, lr_w = h // self.scale, w // self.scale
        lr_patch = cv2.resize(hr_patch, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        lr_upscaled = cv2.resize(lr_patch, (w, h), interpolation=cv2.INTER_CUBIC)
        
        hr_patch = hr_patch.astype(np.float32) / 255.0
        lr_upscaled = lr_upscaled.astype(np.float32) / 255.0
        
        hr_tensor = torch.from_numpy(hr_patch).permute(2, 0, 1)
        lr_tensor = torch.from_numpy(lr_upscaled).permute(2, 0, 1)
        
        return lr_tensor, hr_tensor
    
    def _extract_patch(self, img, patch_size):
        h, w = img.shape[:2]
        if h < patch_size or w < patch_size:
            return cv2.resize(img, (patch_size, patch_size))
        
        top = random.randint(0, h - patch_size)
        left = random.randint(0, w - patch_size)
        return img[top:top+patch_size, left:left+patch_size]
    
    def _augment(self, img):
        if random.random() < 0.5:
            img = np.fliplr(img)
        if random.random() < 0.5:
            img = np.flipud(img)
        k = random.randint(0, 3)
        img = np.rot90(img, k)
        return img.copy()


def create_dataloaders(train_dir, val_dir, batch_size=16, patch_size=96, num_workers=4, scale=2):
    train_dataset = SRDataset(train_dir, scale=scale, patch_size=patch_size, augment=True)
    val_dataset = SRDataset(val_dir, scale=scale, patch_size=patch_size, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader
