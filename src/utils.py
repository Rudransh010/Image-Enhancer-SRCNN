# utils.py - Utility functions for metrics, logging, and helpers
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import csv
import os


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (tensors or numpy arrays)"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    if img1.ndim == 4:
        img1 = img1[0]
    if img2.ndim == 4:
        img2 = img2[0]
    
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    return peak_signal_noise_ratio(img1, img2, data_range=1.0)


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    
    if img1.ndim == 4:
        img1 = img1[0]
    if img2.ndim == 4:
        img2 = img2[0]
    
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    return structural_similarity(img1, img2, data_range=1.0, channel_axis=2)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.writer = None
        
    def __enter__(self):
        self.file = open(self.filepath, 'w', newline='')
        return self
    
    def __exit__(self, *args):
        if self.file:
            self.file.close()
    
    def write_header(self, headers):
        self.writer = csv.DictWriter(self.file, fieldnames=headers)
        self.writer.writeheader()
        self.file.flush()
    
    def write_row(self, row_dict):
        if self.writer:
            self.writer.writerow(row_dict)
            self.file.flush()


def get_model_size_mb(filepath):
    """Get model file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)


def count_flops(model, input_size=(1, 3, 256, 256)):
    """Count FLOPs using ptflops"""
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_size[1:], as_strings=False,
            print_per_layer_stat=False, verbose=False
        )
        flops = 2 * macs
        return flops / 1e9
    except ImportError:
        print("ptflops not installed, skipping FLOPs calculation")
        return 0.0
